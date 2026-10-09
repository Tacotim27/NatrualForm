"""Every template / species / pose builds a valid spec (no Blender needed)."""

import json
import unittest

import numpy as np

from naturalform import api
from naturalform.core.creature import Creature, NaturalFormError


class TestTemplates(unittest.TestCase):
    def test_every_species_spec(self):
        for tmpl, info in api.TEMPLATES.items():
            names = info["presets"] if tmpl != "human" else ["male"]
            for sp in names:
                with self.subTest(template=tmpl, species=sp):
                    c = api.human(sex=sp) if tmpl == "human" else info["spec"](sp)
                    self.assertGreater(len(c.parts), 5)
                    lo, hi = c.bounds()
                    self.assertTrue(np.all(hi > lo))
                    # JSON round trip (stored on the Blender object for rebuilds)
                    d = Creature.from_dict(json.loads(c.to_json()))
                    self.assertEqual(len(d.parts), len(c.parts))

    def test_every_pose_preset(self):
        samples = {"human": api.human(), "quadruped": api.quadruped("horse"),
                   "bird": api.bird("sparrow"), "fish": api.fish("shark"), "serpent": api.serpent("snake")}
        for tmpl, c in samples.items():
            for pose in c.poses:
                with self.subTest(template=tmpl, pose=pose):
                    d = c.copy().pose(pose)
                    lo, hi = d.bounds()
                    if d.meta.get("ground"):
                        self.assertAlmostEqual(float(lo[2]), 0.0, delta=0.02)

    def test_human_variants(self):
        for kw in [dict(age=1), dict(age=6, sex="female"), dict(age=15), dict(age=85, beard="full"),
                   dict(style="chibi"), dict(style="stylized", fingers=False), dict(fat=1.0, muscle=0.0),
                   dict(muscle=1.0, fat=0.0)] + [dict(hair_style=h) for h in
                                                 ("bald", "buzz", "short", "bob", "long", "ponytail", "bun",
                                                  "afro", "mohawk")] + [dict(clothing=c) for c in
                                                                        ("none", "underwear", "casual", "shorts",
                                                                         "dress", "suit")]:
            with self.subTest(**kw):
                c = api.human(**kw)
                h = c.bounds()[1][2] - c.bounds()[0][2]
                self.assertGreater(h, 0.3)

    def test_heights(self):
        c = api.human(height=1.9)
        self.assertAlmostEqual(float(c.bounds()[1][2]), 1.9, delta=0.04)
        c = api.quadruped("horse", height=1.7)
        self.assertGreater(float(c.bounds()[1][2]), 1.7)
        f = api.fish("trout", length=0.5)
        lo, hi = f.bounds()
        self.assertAlmostEqual(float(hi[1] - lo[1]), 0.5, delta=0.01)

    def test_routing(self):
        self.assertEqual(api._route("woman")[0], "human")
        self.assertEqual(api._route("Horse")[:2], ("quadruped", "horse"))
        self.assertEqual(api._route("polar-bear")[:2], ("quadruped", "polar_bear"))
        self.assertEqual(api._route("kitten")[:2], ("quadruped", "cat"))
        self.assertEqual(api._route("owls")[:2], ("bird", "owl"))
        with self.assertRaises(NaturalFormError):
            api._route("unicorn-dragon")

    def test_bad_parameters(self):
        with self.assertRaises(NaturalFormError) as cm:
            api.quadruped("dog", colour="red")
        self.assertIn("color", str(cm.exception))
        with self.assertRaises(NaturalFormError):
            api.human(clothing="spacesuit")
        with self.assertRaises(NaturalFormError):
            api.quadruped("dragon")
        from naturalform.core.colors import color
        with self.assertRaises(ValueError):
            color("plaid")
        self.assertEqual(len(color("#ff8000")), 3)
        self.assertAlmostEqual(color((255, 255, 255))[0], 1.0)

    def test_docs(self):
        from naturalform.docs import guide
        for topic in (None, "human", "quadruped", "bird", "fish", "serpent", "custom", "pose", "materials",
                      "colors", "zebra", "woman"):
            self.assertGreater(len(guide(topic)), 200, topic)


if __name__ == "__main__":
    unittest.main()
