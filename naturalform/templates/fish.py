"""Parametric fish and fish-shaped sea animals (sharks, dolphins).

Built in units of body length (snout at y = -0.5, tail tip at y ~ +0.55),
then scaled to ``length``. Faces -Y, belly down, centred at z = 0 (place it
in water with ``location``).
"""

import math

import numpy as np

from ..core.creature import Creature, NaturalFormError
from .common import choice, resolve, unit

TAILS = ("fork", "round", "lunate", "shark", "fluke", "double")
DORSALS = ("normal", "long", "sail", "shark", "dolphin", "none")

PARAMS = {
    "length": (None, "body length in metres (default: species size * size)"),
    "size": (1.0, "multiplier on the species' typical size"),
    "body_height": (0.28, "/ length"),
    "body_width": (0.45, "width / height"),
    "head_size": (1.0, "multiplier"),
    "snout": (0.0, "0..1 pointed snout / beak (dolphin)"),
    "eye_size": (1.0, "multiplier"),
    "tail": ("fork", "fork | round | lunate | shark | fluke | double"),
    "tail_size": (1.0, "multiplier"),
    "dorsal": ("normal", "normal | long | sail | shark | dolphin | none"),
    "fin_size": (1.0, "multiplier for all fins"),
    "pectoral_size": (1.0, "multiplier"),
    "gill_slits": (0, "number of shark gill slits (0 = bony fish gill cover)"),
    "material": ("scales", "scales | skin"),
    "color": ((150, 160, 170), "main colour"),
    "belly_color": ("white", "underside colour"),
    "fin_color": (None, "fin colour (default: main colour)"),
    "pattern": ("none", "none | stripes | spots | patches | mottled"),
    "pattern_color": ("black", "pattern colour"),
    "pattern_scale": (None, "feature size in metres"),
    "eye_color": ("gold", "iris colour"),
}

PRESETS = {
    "goldfish": dict(_size=0.15, body_height=0.42, body_width=0.55, tail="double", tail_size=1.5, fin_size=1.3,
                     color=(240, 130, 30), belly_color=(250, 190, 90), eye_color="black"),
    "clownfish": dict(_size=0.1, body_height=0.4, body_width=0.5, tail="round", color=(240, 110, 20),
                      belly_color=None, pattern="stripes", pattern_color="white", pattern_scale=0.03,
                      fin_color=(240, 110, 20), eye_color="black"),
    "trout": dict(_size=0.45, body_height=0.24, color=(120, 125, 90), belly_color=(230, 220, 210), pattern="spots",
                  pattern_color=(40, 40, 35), pattern_scale=0.02, tail="fork"),
    "salmon": dict(_size=0.75, body_height=0.24, color=(150, 160, 175), belly_color="white", tail="fork",
                   pattern="spots", pattern_color=(60, 60, 70), pattern_scale=0.02),
    "tuna": dict(_size=1.5, body_height=0.27, body_width=0.65, tail="lunate", tail_size=1.2, dorsal="normal",
                 color=(30, 50, 90), belly_color=(210, 215, 220), fin_color=(200, 170, 40)),
    "shark": dict(_size=2.5, body_height=0.2, body_width=0.8, snout=0.6, tail="shark", tail_size=1.2,
                  dorsal="shark", pectoral_size=1.6, gill_slits=5, material="skin", color=(110, 120, 130),
                  belly_color="white", eye_color="black", eye_size=0.6),
    "angelfish": dict(_size=0.12, body_height=0.75, body_width=0.3, tail="round", dorsal="sail", fin_size=1.6,
                      color=(220, 215, 190), pattern="stripes", pattern_color="charcoal", pattern_scale=0.03,
                      eye_color="red"),
    "koi": dict(_size=0.6, body_height=0.26, body_width=0.6, tail="fork", color="white", belly_color=None,
                pattern="patches", pattern_color=(235, 90, 20), pattern_scale=0.12, eye_color="black"),
    "piranha": dict(_size=0.25, body_height=0.48, body_width=0.4, tail="fork", color=(130, 135, 140),
                    belly_color=(200, 60, 40), eye_color="red"),
    "dolphin": dict(_size=2.2, body_height=0.2, body_width=0.9, snout=1.0, tail="fluke", dorsal="dolphin",
                    pectoral_size=1.2, material="skin", color=(110, 120, 130), belly_color=(220, 220, 225),
                    eye_color="black", eye_size=0.6),
    "orca": dict(_size=6.0, body_height=0.23, body_width=0.9, snout=0.2, tail="fluke", dorsal="shark",
                 fin_size=1.4, pectoral_size=1.4, material="skin", color=(20, 20, 24), belly_color="white",
                 eye_color="black", eye_size=0.5),
}


def _pose_presets():
    P = {"straight": {}}
    P["swim"] = {"spine_1": [0, 0, 10], "spine_2": [0, 0, -16], "tail": [0, 0, -14]}
    P["turn_left"] = {"spine_0": [0, 0, 12], "spine_1": [0, 0, 14], "spine_2": [0, 0, 14], "tail": [0, 0, 10]}
    P["turn_right"] = {"spine_0": [0, 0, -12], "spine_1": [0, 0, -14], "spine_2": [0, 0, -14], "tail": [0, 0, -10]}
    P["dive"] = {"spine_0": [25, 0, 0]}
    P["leap"] = {"spine_0": [-30, 0, 0], "spine_2": [15, 0, 0], "tail": [20, 0, 0]}
    return P


POSES = tuple(_pose_presets())


def fish(species="trout", **kwargs):
    """Return a :class:`Creature` for a fish / shark / dolphin (see PRESETS / PARAMS)."""
    if species not in PRESETS:
        import difflib
        close = difflib.get_close_matches(str(species), PRESETS, n=3)
        raise NaturalFormError("unknown fish species %r.%s Presets: %s" % (
            species, (" Did you mean %s?" % ", ".join(close)) if close else "", ", ".join(PRESETS)))
    preset = dict(PRESETS[species])
    base_size = preset.pop("_size", 0.4)
    p = resolve(PARAMS, kwargs, "fish(%s)" % species, preset)
    choice(p["tail"], TAILS, "tail")
    choice(p["dorsal"], DORSALS, "dorsal")
    c = Creature(species.capitalize(), "fish")
    c.default_material = "body"

    h = float(p["body_height"])
    wr = float(p["body_width"])
    fs = float(p["fin_size"])
    sn = float(p["snout"])
    # body profile along the length: (y, z centre, half height)
    prof = [(-0.5 - 0.05 * sn, -0.02, 0.07 + 0.17 * (1 - sn)), (-0.36, 0.0, 0.4), (-0.12, 0.01, 0.5),
            (0.12, 0.01, 0.42), (0.3, 0.0, 0.2), (0.4, 0.0, 0.1)]
    pts = [np.array([0.0, y, z * h]) for y, z, _ in prof]
    radii = [r * h for _, _, r in prof]
    c.chain("body", pts, radii, bones=["head", "spine_0", "spine_1", "spine_2", "tail"],
            blend=0.0, joint_blend=0.05 * h, squash=wr, squash_dir=(1, 0, 0))
    if sn > 0:
        c.capsule("snout", pts[0] + [0, 0.02, 0], pts[0] + [0, -0.08 * sn, -0.015 * h], radii[0] * 0.9,
                  end_radius=0.018, squash=wr * 1.2, squash_dir=(1, 0, 0), bone="head", blend=0.03)
    # belly / head shaping
    c.ellipsoid("belly", np.array([0, -0.08, -0.12 * h]), (0.35 * h * wr, 0.3, 0.32 * h), bone="spine_0", blend=0.06)
    # mouth & gills
    mouth_y = pts[0][1] - (0.08 * sn if sn > 0 else 0.0)
    c.ellipsoid("mouth", np.array([0, mouth_y + 0.03, -0.03 * h]), (radii[0] * wr * 1.3, 0.06, 0.006),
                bone="head", blend=0.004, op="subtract")
    if int(p["gill_slits"]) > 0:
        for k in range(int(p["gill_slits"])):
            y = -0.3 + k * 0.025
            c.ellipsoid("gill%d" % k, np.array([0.22 * h * wr, y, -0.02 * h]), (0.02 * h, 0.004, 0.14 * h),
                        bone="head", blend=0.003, op="subtract", mirror=True)
    else:
        c.ellipsoid("gill", np.array([0.2 * h * wr, -0.26, -0.02 * h]), (0.05 * h, 0.006, 0.3 * h),
                    rotate=[0, 0, -20], bone="head", blend=0.004, op="subtract", mirror=True)
    # eyes
    er = 0.07 * h * float(p["eye_size"]) * float(p["head_size"]) + 0.006
    ey = -0.38 + 0.02 * sn
    look = unit([1.0, -0.35, 0.05])
    hit = c.surface_point(np.array([0, ey, 0.12 * h]), look)
    ec = (np.array(hit) if hit is not None else np.array([0.2 * h, ey, 0.12 * h])) - look * er * 0.45
    c.eye("eye.L", ec, er, look=look, bone="head", color=p["eye_color"], iris_size=0.7, pupil_size=0.55,
          sclera=None, socket_blend=er * 0.3, mirror=True)

    # fins (thin ellipsoids; material "fin")
    ft = 0.012 * max(h, 0.2) / 0.28
    d = p["dorsal"]
    top = 0.45 * h
    if d == "normal":
        c.ellipsoid("dorsal", np.array([0, -0.02, top + 0.06 * fs]), (ft, 0.13 * fs, 0.12 * fs), rotate=[-25, 0, 0],
                    bone="spine_0", blend=0.02, material="fin", clip=[((0, 0, top * 0.7), (0, 0, 1))])
    elif d == "long":
        c.ellipsoid("dorsal", np.array([0, 0.05, top + 0.03 * fs]), (ft, 0.3 * fs, 0.07 * fs), bone="spine_0",
                    blend=0.02, material="fin", clip=[((0, 0, top * 0.7), (0, 0, 1))])
    elif d == "sail":
        c.ellipsoid("dorsal", np.array([0, 0.08, top + 0.12 * fs]), (ft, 0.16 * fs, 0.3 * fs), rotate=[-35, 0, 0],
                    bone="spine_1", blend=0.02, material="fin", clip=[((0, 0, top * 0.6), (0, 0, 1))])
        c.ellipsoid("anal", np.array([0, 0.08, -top - 0.12 * fs]), (ft, 0.16 * fs, 0.3 * fs), rotate=[35, 0, 0],
                    bone="spine_1", blend=0.02, material="fin", clip=[((0, 0, -top * 0.6), (0, 0, -1))])
    elif d in ("shark", "dolphin"):
        base = np.array([0, -0.05, top * 0.85])
        tip = base + (np.array([0, 0.12, 0.22]) if d == "shark" else np.array([0, 0.14, 0.13])) * fs
        c.capsule("dorsal", base, tip, 0.06 * fs, end_radius=0.01, squash=0.25, squash_dir=(1, 0, 0),
                  bone="spine_0", blend=0.03, material="fin" if d == "shark" else None)
    # pectoral fins
    ps = float(p["pectoral_size"]) * fs
    pb = np.array([0.3 * h * wr, -0.24, -0.2 * h])
    c.capsule("pectoral", pb, pb + np.array([0.12, 0.1, -0.08]) * ps, 0.035 * ps, end_radius=0.015 * ps,
              squash=0.25, squash_dir=(0.3, 0, 1), bone="head", blend=0.015,
              material="fin" if p["material"] == "scales" else None, mirror=True)
    if p["material"] == "scales" and d != "sail":
        c.ellipsoid("pelvic", np.array([0.12 * h * wr, 0.0, -0.45 * h]), (ft, 0.06 * fs, 0.05 * fs),
                    rotate=[35, 0, 20], bone="spine_0", blend=0.01, material="fin", mirror=True)
        c.ellipsoid("anal", np.array([0, 0.22, -0.28 * h - 0.04 * fs]), (ft, 0.07 * fs, 0.06 * fs), rotate=[30, 0, 0],
                    bone="spine_2", blend=0.015, material="fin", clip=[((0, 0, -0.15 * h), (0, 0, -1))])
    _tail(c, p, pts[-1], h, ft)
    _materials(c, p, base_size)
    c.poses.update(_pose_presets())
    length = float(p["length"]) if p["length"] else base_size * float(p["size"])
    lo, hi = c.bounds()
    s = length / float(hi[1] - lo[1])
    c.transform(scale=s)
    c.meta["min_feature"] = ft * s * 1.2
    c.meta["params"] = dict(p, species=species)
    c.meta["user_params"] = dict(kwargs)
    c.meta["template"] = "fish"
    return c


def _tail(c, p, root, h, ft):
    t = p["tail"]
    ts = float(p["tail_size"])
    mat = "fin" if t not in ("fluke",) else None
    if t == "fluke":
        for sgn in (1, -1):
            c.capsule("fluke%s" % ("L" if sgn > 0 else "R"), root, root + np.array([0.14 * sgn, 0.1, 0.0]) * ts,
                      0.05 * ts, end_radius=0.02 * ts, squash=0.18, squash_dir=(0, 0, 1), bone="tail", blend=0.03)
        return
    if t == "shark":
        c.capsule("tail_upper", root, root + np.array([0, 0.2, 0.2]) * ts, 0.05 * ts, end_radius=0.012,
                  squash=0.22, squash_dir=(1, 0, 0), bone="tail", blend=0.03, material=mat)
        c.capsule("tail_lower", root, root + np.array([0, 0.1, -0.11]) * ts, 0.045 * ts, end_radius=0.012,
                  squash=0.22, squash_dir=(1, 0, 0), bone="tail", blend=0.03, material=mat)
        return
    if t in ("round", "double"):
        n = 2 if t == "double" else 1
        for k in range(n):
            off = np.array([0.03 * (k - 0.5) * 2, 0, 0]) if n == 2 else np.zeros(3)
            c.ellipsoid("caudal%d" % k, root + off + np.array([0, 0.11 * ts, 0]), (ft, 0.12 * ts, 0.12 * ts * (h / 0.28) ** 0.3),
                        rotate=[0, 0, (k - 0.5) * 30 if n == 2 else 0], bone="tail", blend=0.02, material=mat)
        return
    spread = 34 if t == "fork" else 48
    ln = 0.17 if t == "fork" else 0.2
    for sgn, name in ((1, "upper"), (-1, "lower")):
        a = math.radians(spread) * sgn
        d = np.array([0.0, math.cos(a), math.sin(a)])
        c.capsule("tail_%s" % name, root - d * 0.01, root + d * ln * ts, 0.04 * ts, end_radius=0.012 * ts,
                  squash=0.2 if t == "fork" else 0.16, squash_dir=(1, 0, 0), bone="tail", blend=0.02, material=mat)


def _materials(c, p, base_size):
    size = float(p["length"]) if p["length"] else base_size * float(p["size"])
    kind = p["material"]
    c.material("body", kind=kind, color=p["color"], belly_color=p["belly_color"], pattern=p["pattern"],
               pattern_color=p["pattern_color"], pattern_scale=p["pattern_scale"] or 0.08 * size,
               scale_size=0.015 * size, roughness=0.35 if kind == "scales" else 0.4,
               subsurface=0.0 if kind == "scales" else 0.05, subsurface_scale=0.002 * size)
    c.material("fin", kind="membrane", color=p["fin_color"] or p["color"], roughness=0.4, subsurface=0.4,
               subsurface_scale=0.002 * size)
