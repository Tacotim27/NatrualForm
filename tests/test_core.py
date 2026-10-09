"""Engine tests: SDF primitives, meshing and the Creature spec (numpy only)."""

import json
import unittest

import numpy as np

from naturalform.core import mesher
from naturalform.core.creature import Creature, NaturalFormError
from naturalform.core.sdf import (Clipped, Ellipsoid, FieldPart, RoundBox, RoundCone, Torus, evaluate,
                                  primitive_from_dict, rotation_between, smin)


def closed_and_oriented(V, Q, T=None):
    T = np.zeros((0, 3), dtype=np.int64) if T is None else T
    E = mesher._directed_edges(Q, T)
    key = E[:, 0] * len(V) + E[:, 1]
    rkey = E[:, 1] * len(V) + E[:, 0]
    return len(np.unique(key)) == len(key) and np.isin(rkey, key).all()


class TestPrimitives(unittest.TestCase):
    def test_sphere_distance_exact(self):
        e = Ellipsoid((1, 2, 3), (0.5, 0.5, 0.5))
        P = np.random.default_rng(1).normal(size=(500, 3)) + [1, 2, 3]
        d = e.distance(P)
        np.testing.assert_allclose(d, np.linalg.norm(P - [1, 2, 3], axis=1) - 0.5, atol=1e-9)

    def test_capsule_distance_exact(self):
        rc = RoundCone((0, 0, 0), (0, 0, 1), 0.2, 0.2)
        P = np.random.default_rng(2).uniform(-1, 2, size=(500, 3))
        z = np.clip(P[:, 2], 0, 1)
        ref = np.linalg.norm(P - np.stack([np.zeros(500), np.zeros(500), z], 1), axis=1) - 0.2
        np.testing.assert_allclose(rc.distance(P), ref, atol=1e-9)

    def test_round_cone_degenerate(self):
        rc = RoundCone((0, 0, 0), (0, 0, 0.05), 0.5, 0.1)  # big sphere swallows small one
        self.assertAlmostEqual(float(rc.distance(np.array([[0, 0, 1.0]]))[0]), 0.5, places=6)

    def test_box_and_torus(self):
        b = RoundBox((0, 0, 0), (1, 1, 1), rounding=0.0)
        self.assertAlmostEqual(float(b.distance(np.array([[2.0, 0, 0]]))[0]), 1.0)
        t = Torus((0, 0, 0), 1.0, 0.25)
        self.assertAlmostEqual(float(t.distance(np.array([[1.0, 0, 0]]))[0]), -0.25)

    def test_clipped(self):
        c = Clipped(Ellipsoid((0, 0, 0), (1, 1, 1)), [((0, 0, 0), (0, 0, 1))])
        self.assertGreater(float(c.distance(np.array([[0, 0, -0.5]]))[0]), 0)  # cut away
        self.assertLess(float(c.distance(np.array([[0, 0, 0.5]]))[0]), 0)

    def test_roundtrip_dict(self):
        for prim in (RoundCone((0, 0, 0), (1, 0, 0), 0.3, 0.1, 0.5, (0, 0, 1)),
                     Ellipsoid((1, 2, 3), (0.1, 0.2, 0.3), rotation_between((0, 1, 0), (1, 1, 0))),
                     RoundBox((0, 0, 0), (1, 2, 3), rounding=0.2), Torus((0, 0, 1), 1, 0.2),
                     Clipped(Ellipsoid((0, 0, 0), (1, 1, 1)), [((0, 0, 0), (0, 0, 1))], 0.1)):
            P = np.random.default_rng(3).normal(size=(50, 3))
            q = primitive_from_dict(json.loads(json.dumps(prim.to_dict())))
            np.testing.assert_allclose(prim.distance(P), q.distance(P), atol=1e-4)

    def test_mirror(self):
        e = Ellipsoid((0.3, 0, 0), (0.1, 0.2, 0.3), rotation_between((0, 1, 0), (1, 1, 0)))
        m = e.mirrored()
        P = np.random.default_rng(4).normal(size=(50, 3))
        Pm = P * [-1, 1, 1]
        np.testing.assert_allclose(e.distance(P), m.distance(Pm), atol=1e-9)

    def test_smin_bounds(self):
        a, b = np.array([0.0, 1.0]), np.array([0.0, -1.0])
        r = smin(a, b, 0.2)
        self.assertTrue(np.all(r <= np.minimum(a, b)))
        self.assertAlmostEqual(float(r[0]), -0.05)


class TestMesher(unittest.TestCase):
    def test_sphere_mesh(self):
        parts = [FieldPart(Ellipsoid((0, 0, 0), (1, 1, 1)), "union", 0.0)]
        r = mesher.mesh_parts(parts, 0.08)
        self.assertGreater(len(r.faces), 500)
        np.testing.assert_allclose(np.linalg.norm(r.verts, axis=1), 1.0, atol=1e-3)
        self.assertTrue(closed_and_oriented(r.verts, r.faces))
        V, F = r.verts, r.faces
        n = np.cross(V[F[:, 2]] - V[F[:, 0]], V[F[:, 3]] - V[F[:, 1]])
        self.assertTrue(((n * V[F].mean(1)).sum(1) > 0).all(), "faces must point outwards")

    def test_blend_and_carve(self):
        parts = [FieldPart(RoundCone((0, 0, 0), (1, 0, 0), 0.3, 0.2), "union", 0.0),
                 FieldPart(Ellipsoid((1, 0, 0), (0.3, 0.3, 0.3)), "union", 0.1),
                 FieldPart(Ellipsoid((0, 0, 0.3), (0.1, 0.1, 0.1)), "subtract", 0.02)]
        r = mesher.mesh_parts(parts, 0.02)
        f = evaluate(parts, r.verts, 0.1)
        self.assertLess(np.abs(f).max(), 0.02 * 0.05)  # within 5% of a voxel

    def test_detail_region_is_watertight(self):
        parts = [FieldPart(RoundCone((0, 0, 0), (0, 0, 1), 0.2, 0.12), "union", 0.0),
                 FieldPart(Ellipsoid((0, 0, 1.15), (0.2, 0.22, 0.2)), "union", 0.06),
                 FieldPart(Ellipsoid((0.0, -0.2, 1.15), (0.05, 0.05, 0.05)), "union", 0.01)]
        reg = {"center": np.array([0, 0, 1.15]), "radius": 0.35, "voxel": 0.008,
               "plane": (np.array([0, 0, 0.75]), np.array([0, 0, 1.0]))}
        m = mesher.mesh_adaptive(parts, 0.03, [reg])
        self.assertGreater(len(m["tris"]), 0, "region should have been zipped in")
        self.assertTrue(closed_and_oriented(m["V"], m["quads"], m["tris"]))


class TestCreature(unittest.TestCase):
    def make(self):
        c = Creature("Test")
        c.chain("body", [(0, -0.3, 0.5), (0, 0, 0.5), (0, 0.3, 0.5)], [0.15, 0.18, 0.12], bones=["chest", "hips"])
        c.chain("leg.L", [(0.1, -0.2, 0.45), (0.12, -0.2, 0.0)], [0.05, 0.04], parent="chest", mirror=True)
        c.ellipsoid("head", (0, -0.45, 0.6), (0.1, 0.12, 0.1), bone="chest")
        c.eye("eye.L", (0.05, -0.55, 0.63), 0.02, look=(0.3, -1, 0), mirror=True)
        return c

    def test_mirror_and_bones(self):
        c = self.make()
        self.assertIn("leg.R", c.bones)
        self.assertEqual(c.bones["leg.R"].parent, "chest")
        self.assertAlmostEqual(c.bones["leg.R"].head[0], -0.1)
        self.assertEqual({e.name for e in c.eyes}, {"eye.L", "eye.R"})
        self.assertIn("eye_socket.R", [p.name for p in c.parts])

    def test_json_roundtrip(self):
        c = self.make()
        d = Creature.from_dict(c.to_json())
        self.assertEqual(len(d.parts), len(c.parts))
        self.assertEqual(list(d.bones), list(c.bones))
        P = np.random.default_rng(5).normal(size=(100, 3)) * 0.3 + [0, 0, 0.5]
        np.testing.assert_allclose(evaluate(c.field_parts(), P, 1.0), evaluate(d.field_parts(), P, 1.0), atol=1e-4)

    def test_pose_moves_children(self):
        c = self.make()
        foot_before = c.get("leg.L")[0].prim.b.copy()
        c.pose({"leg.L": {"aim": [0, -1, 0]}})
        foot = c.get("leg.L")[0].prim.b
        self.assertLess(foot[1], foot_before[1] - 0.3)  # leg swung forward (-Y)
        np.testing.assert_allclose(c.bones["leg.L"].tail, foot, atol=1e-9)

    def test_transform_scales_everything(self):
        c = self.make()
        lo, hi = c.bounds()
        c.transform(scale=2.0)
        lo2, hi2 = c.bounds()
        np.testing.assert_allclose(hi2 - lo2, (hi - lo) * 2, atol=1e-9)
        self.assertAlmostEqual(c.eyes[0].radius, 0.04)

    def test_errors_are_helpful(self):
        c = self.make()
        with self.assertRaises(NaturalFormError) as cm:
            c.pose({"lge.L": [10, 0, 0]})
        self.assertIn("leg.L", str(cm.exception))
        with self.assertRaises(NaturalFormError):
            c.sphere("head", (0, 0, 0), 1)  # duplicate name
        with self.assertRaises(NaturalFormError):
            c.chain("x", [(0, 0, 0)], [0.1])

    def test_surface_point_and_remove(self):
        c = self.make()
        p = c.surface_point((0, -0.45, 0.6), (0, -1, 0))
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p[1], -0.57, delta=0.02)
        c.remove("leg.L")
        self.assertNotIn("leg.L", c.bones)
        self.assertNotIn("leg.L", c.part_names())

    def test_clothe_and_layers(self):
        c = self.make()
        c.clothe("coat", ["body"], 0.02, "cloth")
        self.assertEqual(c.layers(), ["body", "coat"])
        self.assertTrue(all(p.bone in ("chest", "hips") for p in c.get("coat")))


if __name__ == "__main__":
    unittest.main()
