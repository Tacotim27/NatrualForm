"""Integration tests inside Blender (needs ``bpy``)."""

import io
import contextlib
import unittest

import bpy

import naturalform as nf


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


class TestBuild(unittest.TestCase):
    def setUp(self):
        _reset()

    def test_create_every_template(self):
        for what in ("man", "woman", "child", "dog", "cat", "horse", "elephant", "giraffe", "eagle", "penguin",
                     "shark", "goldfish", "cobra", "worm"):
            with self.subTest(what=what):
                r = nf.create(what, detail="preview")
                arm = bpy.data.objects[r["name"]]
                mesh = bpy.data.objects[r["mesh"]]
                self.assertEqual(arm.type, "ARMATURE")
                self.assertEqual(mesh.parent, arm)
                self.assertGreater(len(mesh.data.polygons), 500)
                self.assertGreater(len(mesh.vertex_groups), 3)
                self.assertTrue(mesh.data.materials)
                self.assertIn("nf_rest", mesh.data.attributes)
                self.assertFalse(mesh.data.validate(), "mesh needed repairs")

    def test_weights_cover_every_vertex(self):
        r = nf.create("human", detail="low")
        me = bpy.data.objects[r["mesh"]].data
        weighted = sum(1 for v in me.vertices if v.groups)
        self.assertEqual(weighted, len(me.vertices))

    def test_detail_region_merges_into_human(self):
        low = nf.create("human", detail="low", name="A")
        med = nf.create("human", detail="medium", name="B")
        self.assertGreater(med["vertices"], low["vertices"] * 2)
        me = bpy.data.objects[med["mesh"]].data
        self.assertTrue(any(len(p.vertices) == 3 for p in me.polygons), "zipped seams use triangles")

    def test_names_and_replace(self):
        a = nf.create("dog", detail="preview")
        b = nf.create("dog", detail="preview")
        self.assertNotEqual(a["name"], b["name"])  # auto names are unique
        nf.create("dog", name="Rex", detail="preview")
        nf.create("dog", name="Rex", detail="preview", color="black")  # explicit name replaces
        self.assertEqual(len([o for o in bpy.data.objects if o.name.startswith("Rex")]), 2)

    def test_location_rotation(self):
        r = nf.create("horse", detail="preview", location=(3, 2, 0), rotation=(0, 0, 90))
        root = bpy.data.objects[r["name"]]
        self.assertAlmostEqual(root.location.x, 3.0)
        self.assertAlmostEqual(root.rotation_euler.z, 1.5708, places=3)

    def test_pose_rebuild_describe(self):
        nf.create("woman", name="W", detail="preview")
        out = nf.pose("W", "walk")
        self.assertIn("thigh.L", out["posed_bones"])
        nf.pose("W", {"head": [0, 0, 30]})
        nf.reset_pose("W")
        r = nf.rebuild("W", age=8, pose="sit", detail="preview")
        self.assertLess(r["dimensions_m"][2], 1.3)
        info = nf.describe("W")
        self.assertEqual(info["params"]["age"], 8)
        self.assertIn("head", nf.bones("W"))
        c = nf.load("W")
        self.assertEqual(c.kind, "human")

    def test_custom_creature_from_docs(self):
        c = nf.Creature("Dragon")
        c.chain("body", [(0, -0.6, 1.0), (0, -0.15, 1.05), (0, 0.3, 1.02), (0, 0.75, 0.95)],
                radii=[0.3, 0.38, 0.34, 0.24], bones=["chest", "spine", "hips"], squash=0.85,
                squash_dir=(1, 0, 0))
        c.chain("neck", [(0, -0.7, 1.1), (0, -1.1, 1.6), (0, -1.3, 1.9)], [0.2, 0.14, 0.12],
                parent="chest", blend=0.12)
        c.ellipsoid("head", (0, -1.45, 1.95), (0.16, 0.28, 0.15), bone="neck_1")
        c.chain("leg.L", [(0.3, -0.4, 0.9), (0.38, -0.5, 0.45), (0.36, -0.4, 0.05)],
                [0.14, 0.09, 0.07], parent="chest", mirror=True)
        c.chain("hind_leg.L", [(0.3, 0.5, 0.95), (0.42, 0.7, 0.5), (0.36, 0.55, 0.05)],
                [0.17, 0.1, 0.07], parent="hips", mirror=True)
        c.chain("tail", [(0, 0.8, 0.95), (0, 1.5, 0.7), (0.3, 2.2, 0.45)], [0.2, 0.1, 0.03], parent="hips")
        c.chain("wing.L", [(0.3, -0.3, 1.3), (1.2, -0.1, 1.7), (2.4, 0.2, 1.6)], [0.12, 0.3, 0.05],
                parent="chest", squash=0.12, squash_dir=(0, 0, 1), material="wing", mirror=True)
        c.eye("eye.L", (0.1, -1.62, 2.02), 0.035, look=(0.5, -1, 0.1), color="eye_yellow",
              pupil="slit", sclera=None, mirror=True)
        c.material("body", kind="scales", color="olive_green", belly_color="khaki")
        c.material("wing", kind="membrane", color=(120, 40, 40))
        r = c.build(detail="low", pose={"neck_0": [-20, 0, 0]})
        self.assertEqual(r["name"], "Dragon")
        self.assertIn("wing_1.R", nf.bones("Dragon"))

    def test_pegasus_from_docs(self):
        c = nf.spec("horse", color="white", mane_color="platinum")
        c.chain("wing.L", [(0.2, -0.45, 1.55), (0.9, -0.35, 2.0), (1.8, -0.1, 2.3)], [0.08, 0.22, 0.04],
                parent="chest", squash=0.15, squash_dir=(0, 0, 1), mirror=True)
        r = c.build(name="Pegasus", detail="preview")
        self.assertGreater(r["dimensions_m"][0], 3.0)

    def test_focus_and_help(self):
        r = nf.create("cat", detail="preview")
        out = nf.focus(r["name"], view="front")
        self.assertIn("views_updated", out)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            nf.help()
            nf.help("custom")
        self.assertIn("nf.create", buf.getvalue())

    def test_ui_registration(self):
        nf.register()
        try:
            s = bpy.context.scene.naturalform
            s.kind = "bird"
            s.species = "owl"
            s.detail = "preview"
            self.assertEqual(bpy.ops.naturalform.create(), {"FINISHED"})
            self.assertTrue(any(o.name.startswith("Owl") for o in bpy.data.objects))
        finally:
            nf.unregister()


if __name__ == "__main__":
    unittest.main()
