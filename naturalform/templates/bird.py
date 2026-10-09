"""Parametric birds with species presets.

Built in units of the body length, then scaled to the standing height.
Faces -Y, left wing = +X, feet on the ground. ``wings="spread"`` builds
open wings (use it for flying birds); folded wings lie along the body.
"""

import math

import numpy as np

from ..core.creature import Creature, NaturalFormError
from .common import choice, clamp, resolve, sym, unit

BEAKS = ("cone", "stout", "hook", "flat", "long", "bent")
WINGS = ("folded", "spread", "raised")
TAILS = ("fan", "long", "short", "upright")
FEET = ("perching", "webbed", "none")

PARAMS = {
    "height": (None, "standing height in metres (default: species size * size)"),
    "size": (1.0, "multiplier on the species' typical size"),
    "body_angle": (30.0, "degrees the body axis rises toward the head (10 duck .. 80 penguin)"),
    "body_width": (0.55, "/ body length"),
    "body_depth": (0.6, "/ body length"),
    "fat": (0.3, "0..1 plumpness"),
    "neck_length": (0.12, "/ body length"),
    "neck_angle": (60.0, "degrees above horizontal"),
    "neck_shape": ("straight", "straight | s (flamingo / swan)"),
    "head_size": (1.0, "multiplier"),
    "beak": ("cone", "cone | stout | hook | flat | long | bent"),
    "beak_length": (0.25, "/ body length"),
    "beak_width": (1.0, "multiplier"),
    "eye_size": (1.0, "multiplier"),
    "eye_angle": (70.0, "0 = facing forward (owl) .. 80 = on the sides"),
    "facial_disc": (False, "owl face"),
    "face_color": (None, "facial disc colour"),
    "ear_tufts": (False, "owl ear tufts"),
    "crest": (False, "crest feathers on the head"),
    "comb": (False, "chicken comb and wattle"),
    "wings": ("folded", "folded | spread | raised"),
    "wing_length": (1.0, "multiplier"),
    "wing_span": (2.2, "spread wing span / body length"),
    "flippers": (False, "penguin flippers instead of wings"),
    "tail": ("fan", "fan | long | short | upright"),
    "tail_length": (0.45, "/ body length"),
    "leg_length": (0.35, "visible leg length / body length"),
    "leg_thickness": (1.0, "multiplier"),
    "feet": ("perching", "perching | webbed | none"),
    "color": ("tawny", "main plumage colour"),
    "belly_color": (None, "underside colour"),
    "breast_color": (None, "breast colour"),
    "head_color": (None, "head colour"),
    "wing_color": (None, "wing / tail colour"),
    "pattern": ("none", "none | mottled | spots | stripes"),
    "pattern_color": ("charcoal", "pattern colour"),
    "beak_color": ("charcoal", "beak colour"),
    "leg_color": ((190, 150, 120), "legs and feet colour"),
    "eye_color": ("black", "iris colour"),
}

PRESETS = {
    "sparrow": dict(_size=0.15, body_angle=28, color=(140, 105, 70), wing_color=(110, 78, 50), belly_color="cream",
                    head_color=(125, 118, 110), pattern="mottled", pattern_color=(70, 50, 35), beak="cone",
                    beak_length=0.2, beak_color=(60, 55, 50)),
    "robin": dict(_size=0.14, body_angle=32, fat=0.5, color=(110, 95, 72), breast_color=(215, 100, 45),
                  belly_color="cream", beak="cone", beak_length=0.18, beak_color=(50, 45, 40)),
    "crow": dict(_size=0.45, body_angle=25, color=(28, 28, 34), beak="stout", beak_length=0.32,
                 beak_color=(25, 25, 28), leg_color=(30, 30, 30), tail_length=0.55),
    "pigeon": dict(_size=0.3, body_angle=25, fat=0.6, color=(135, 140, 150), head_color=(95, 105, 110),
                   wing_color=(150, 152, 160), beak="cone", beak_length=0.15, beak_color=(70, 60, 60),
                   leg_color=(200, 90, 90), eye_color="orange"),
    "eagle": dict(_size=0.85, body_angle=35, color="chocolate", head_color="white", wing_color=(70, 45, 30),
                  beak="hook", beak_length=0.3, beak_width=1.2, beak_color="beak_yellow", leg_color="beak_yellow",
                  eye_color="eye_amber", eye_angle=45, tail_length=0.5, wing_span=2.6),
    "owl": dict(_size=0.45, body_angle=72, body_width=0.6, fat=0.6, head_size=1.45, neck_length=0.02,
                facial_disc=True, face_color=(215, 190, 150), ear_tufts=True, eye_angle=12, eye_size=1.5, eye_color="eye_yellow",
                beak="hook", beak_length=0.12, beak_color=(60, 55, 50), color="tawny", belly_color="cream",
                pattern="mottled", pattern_color=(80, 55, 35), tail="short", tail_length=0.25),
    "parrot": dict(_size=0.4, body_angle=55, color="red", wing_color="blue", head_color="red",
                   beak="hook", beak_length=0.28, beak_width=1.4, beak_color="ivory", tail="long", tail_length=1.0,
                   eye_color="white", leg_color="charcoal", eye_angle=60),
    "duck": dict(_size=0.35, body_angle=8, body_width=0.55, fat=0.5, neck_length=0.25, neck_angle=70,
                 beak="flat", beak_length=0.3, beak_color=(220, 190, 60), head_color=(30, 90, 50),
                 breast_color="chestnut", color=(140, 135, 125), wing_color=(110, 100, 90), feet="webbed",
                 leg_color="orange", leg_length=0.18, tail="short", tail_length=0.25),
    "chicken": dict(_size=0.5, body_angle=25, fat=0.8, body_width=0.6, comb=True, beak="cone", beak_length=0.16,
                    beak_color="beak_yellow", leg_color="beak_yellow", color="white", tail="upright",
                    tail_length=0.55, neck_length=0.22, neck_angle=75),
    "penguin": dict(_size=0.65, body_angle=84, body_width=0.55, body_depth=0.58, fat=0.7, neck_length=0.04,
                    head_size=0.95, flippers=True, beak="stout", beak_length=0.3, beak_width=0.8,
                    beak_color=(40, 35, 35),
                    color=(25, 28, 34), belly_color="white", feet="webbed", leg_length=0.05,
                    leg_color="orange", tail="short", tail_length=0.15, eye_angle=60),
    "flamingo": dict(_size=1.3, body_angle=15, neck_length=0.95, neck_shape="s", neck_angle=70, head_size=0.75,
                     beak="bent", beak_length=0.38, beak_color=(30, 30, 30), color=(240, 140, 150),
                     wing_color=(230, 110, 120), leg_length=1.6, leg_thickness=0.8, leg_color=(230, 120, 130),
                     feet="webbed", tail="short", tail_length=0.25, eye_color="eye_yellow"),
    "seagull": dict(_size=0.5, body_angle=20, color="white", wing_color=(160, 165, 172), beak="stout",
                    beak_length=0.28, beak_color="beak_yellow", leg_color=(230, 200, 120), feet="webbed",
                    eye_color="eye_yellow", wing_span=2.8),
    "hummingbird": dict(_size=0.08, body_angle=35, color=(40, 140, 80), belly_color=(200, 200, 190),
                        breast_color=(200, 40, 70), beak="long", beak_length=0.55, beak_width=0.5,
                        beak_color="black", leg_length=0.12, wing_length=1.2),
    "swan": dict(_size=0.9, body_angle=8, body_width=0.6, fat=0.6, neck_length=0.9, neck_shape="s", neck_angle=75,
                 head_size=0.8, beak="flat", beak_length=0.25, beak_color="orange", color="white",
                 feet="webbed", leg_color="charcoal", leg_length=0.12, tail="short", tail_length=0.2),
}


def _pose_presets():
    P = {"stand": {}}
    P["perch"] = sym({"leg.L": [-25, 0, 0], "foot.L": [25, 0, 0]})
    P["peck"] = {"neck": [55, 0, 0], "head": [20, 0, 0]}
    P["look_left"] = {"head": [0, 0, 50]}
    P["look_right"] = {"head": [0, 0, -50]}
    P["fly"] = sym({"leg.L": [70, 0, 0], "foot.L": [30, 0, 0]})
    return P


POSES = tuple(_pose_presets())


def bird(species="sparrow", **kwargs):
    """Return a :class:`Creature` for a bird (see PRESETS / PARAMS)."""
    if species not in PRESETS:
        import difflib
        close = difflib.get_close_matches(str(species), PRESETS, n=3)
        raise NaturalFormError("unknown bird species %r.%s Presets: %s" % (
            species, (" Did you mean %s?" % ", ".join(close)) if close else "", ", ".join(PRESETS)))
    preset = dict(PRESETS[species])
    base_size = preset.pop("_size", 0.2)
    p = resolve(PARAMS, kwargs, "bird(%s)" % species, preset)
    choice(p["beak"], BEAKS, "beak")
    choice(p["wings"], WINGS, "wings")
    choice(p["tail"], TAILS, "tail")
    choice(p["feet"], FEET, "feet")
    fat = clamp(p["fat"])

    c = Creature(species.capitalize(), "bird")
    c.default_material = "body"
    c.meta["ground"] = True

    th = math.radians(float(p["body_angle"]))
    ax = np.array([0.0, -math.cos(th), math.sin(th)])       # tail -> head
    upb = np.array([0.0, math.sin(th), math.cos(th)])       # body "up"
    side = np.array([1.0, 0.0, 0.0])
    BF = np.column_stack([side, ax, upb])
    bw = float(p["body_width"]) * (1 + 0.25 * fat)
    bd = float(p["body_depth"]) * (1 + 0.2 * fat)
    O = np.zeros(3)

    # ---------------------------------------------------------------- body
    c.chain("spine", [O - ax * 0.45, O, O + ax * 0.4], 0.05, bones=["hips", "chest"], visible=False)
    c.ellipsoid("body", O, (bw / 2, 0.5, bd / 2), rotate=BF, bone="hips", blend=0.0)
    breast_mat = "breast" if p["breast_color"] else None
    c.ellipsoid("breast", O + ax * 0.2 - upb * 0.06, (bw * 0.44, 0.27, bd * 0.42), rotate=BF, bone="chest",
                blend=0.08, material=breast_mat)
    # ---------------------------------------------------------------- neck & head
    nl = float(p["neck_length"])
    na = math.radians(float(p["neck_angle"]))
    hr = 0.21 * float(p["head_size"])
    nb = O + ax * 0.36 + upb * bd * 0.12
    if p["neck_shape"] == "s":
        pts = [nb, nb + np.array([0, -0.12, 0.25]) * nl, nb + np.array([0, 0.06, 0.6]) * nl,
               nb + np.array([0, -0.05, 0.95]) * nl, nb + np.array([0, -0.2, 1.05]) * nl]
        neck_r = [0.13, 0.07, 0.06, 0.06, 0.07]
    else:
        d = np.array([0.0, -math.cos(na), math.sin(na)])
        pts = [nb, nb + d * (nl + hr * 0.6)]
        neck_r = [0.15 * (1 + fat * 0.3), hr * 0.75]
    head_c = pts[-1] + np.array([0, -hr * 0.15, hr * 0.15])
    head_mat = "head" if p["head_color"] else None
    c.chain("neck", pts, neck_r, parent="chest", bones=["neck"] if len(pts) == 2 else
            ["neck_%d" % i for i in range(len(pts) - 1)], blend=0.06, material=head_mat)
    last_neck = "neck" if len(pts) == 2 else "neck_%d" % (len(pts) - 2)
    fwd = np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    c.bone("head", head_c, head_c + fwd * hr * 1.5, parent=last_neck)
    c.sphere("head", head_c, hr, bone="head", blend=0.04, material=head_mat)
    c.ellipsoid("cheeks", head_c - up * hr * 0.15 + fwd * hr * 0.15, (hr * 0.85, hr * 0.8, hr * 0.7),
                bone="head", blend=0.03, material=head_mat)
    _beak(c, p, head_c, hr, fwd, up, side)
    # eyes
    ea = math.radians(float(p["eye_angle"]))
    er = hr * 0.24 * float(p["eye_size"])
    look = unit(fwd * math.cos(ea) + side * math.sin(ea) + up * 0.1)
    hit = c.surface_point(head_c + up * hr * 0.12, look)
    ec = (np.array(hit) if hit is not None else head_c + look * hr) - look * er * 0.5
    c.eye("eye.L", ec, er, look=look, bone="head", color=p["eye_color"], pupil="round",
          iris_size=0.75, pupil_size=0.55 if p["eye_color"] not in ("black",) else 0.9, sclera=None,
          socket_blend=er * 0.4, mirror=True)
    if p["facial_disc"]:
        c.ellipsoid("facial_disc", ec + look * er * 0.05 - up * er * 0.2, (er * 2.0, er * 0.45, er * 2.1),
                    rotate=_frame(look), bone="head", blend=er * 0.8, material="face", mirror=True,
                    clip=[(ec + look * er * 0.1, -look)], clip_blend=er * 0.4)
        c.material("face", kind="feathers", color=p["face_color"] or p["belly_color"] or p["color"])
    if p["ear_tufts"]:
        base = head_c + up * hr * 0.8 + side * hr * 0.5
        c.capsule("ear_tuft", base, base + (up * 1.0 + side * 0.35 - fwd * 0.2) * hr * 0.6, hr * 0.15,
                  end_radius=hr * 0.03, squash=0.4, squash_dir=fwd, bone="head", blend=0.02, mirror=True,
                  material=head_mat)
    if p["crest"]:
        base = head_c + up * hr * 0.8
        c.capsule("crest", base, base + (up * 0.7 - fwd * 0.6) * hr * 1.2, hr * 0.25,
                  end_radius=hr * 0.05, squash=0.3, squash_dir=side, bone="head", blend=0.03, material=head_mat)
    if p["comb"]:
        for k, t in enumerate((-0.3, 0.0, 0.3, 0.55)):
            cc = head_c + up * hr * (0.95 - abs(t) * 0.25) + fwd * hr * (0.2 - t * 0.9)
            c.sphere("comb%d" % k, cc, hr * (0.32 - abs(t) * 0.12), bone="head", blend=hr * 0.15, material="comb")
        c.ellipsoid("wattle", head_c - up * hr * 1.0 + fwd * hr * 0.6, (hr * 0.22, hr * 0.15, hr * 0.35),
                    bone="head", blend=hr * 0.1, material="comb", mirror=False)
        c.material("comb", kind="skin", color=(200, 30, 30), roughness=0.45)

    # ---------------------------------------------------------------- wings
    wl = float(p["wing_length"])
    shoulder = O + ax * 0.22 + upb * bd * 0.22 + side * bw * 0.36
    if p["flippers"]:
        c.chain("wing.L", [shoulder - upb * 0.05, shoulder + (-ax * 0.6 + side * 0.16 - upb * 0.04) * wl],
                [0.1, 0.05], parent="chest", bones=["wing.L"], blend=0.04, squash=0.22, squash_dir=side,
                material="wing", mirror=True)
    elif p["wings"] == "folded":
        c.bone("wing.L", shoulder, shoulder - ax * 0.7 * wl, parent="chest", mirror=True)
        c.ellipsoid("wing.L", O + side * bw * 0.42 - ax * 0.08 + upb * bd * 0.08,
                    (0.07, 0.48 * wl, bd * 0.36), rotate=np.column_stack([side, ax, upb]) @ _rx(-8),
                    bone="wing.L", blend=0.03, material="wing", mirror=True)
        tip0 = O + side * bw * 0.3 - ax * 0.25 + upb * bd * 0.12
        c.capsule("primaries.L", tip0, tip0 - ax * 0.55 * wl - upb * 0.05 - side * bw * 0.12, 0.1,
                  end_radius=0.03, squash=0.28, squash_dir=side, bone="wing.L", blend=0.02, material="wing",
                  mirror=True)
    else:
        span = float(p["wing_span"]) * wl
        raise_ = 0.35 if p["wings"] == "raised" else 0.04
        S = shoulder
        E = S + np.array([0.28, 0.02, raise_ * 0.5]) * span * 0.5
        W = E + np.array([0.3, 0.06, raise_ * 0.6]) * span * 0.5
        T = W + np.array([0.42, 0.22, raise_ * 0.4]) * span * 0.5
        c.chain("wing.L", [S, E, W, T], np.array([0.17, 0.2, 0.22, 0.05]) * wl, parent="chest",
                bones=["wing_0.L", "wing_1.L", "wing_2.L"], blend=0.04, joint_blend=0.04, squash=0.14,
                squash_dir=(0, 0, 1), material="wing", mirror=True)
        # trailing-edge feathers widen the inner wing
        c.capsule("secondaries.L", E + [0, 0.12, 0], W + [0, 0.18, 0], 0.16, end_radius=0.2, squash=0.12,
                  squash_dir=(0, 0, 1), bone="wing_1.L", blend=0.05, material="wing", mirror=True)

    # ---------------------------------------------------------------- tail
    tl = float(p["tail_length"])
    t0 = O - ax * 0.42 + upb * bd * 0.05
    if p["tail"] == "upright":
        tdir = unit(-ax + upb * 1.4)
    elif p["tail"] == "long":
        tdir = unit(-ax * 1.0 - upb * 0.15)
    else:
        tdir = unit(-ax - upb * 0.25)
    tw = {"fan": 0.18, "long": 0.09, "short": 0.12, "upright": 0.16}[p["tail"]]
    c.chain("tail", [t0, t0 + tdir * tl], [0.08, tw], parent="hips", bones=["tail"], blend=0.05,
            squash=0.18 if p["tail"] != "upright" else 0.3,
            squash_dir=upb if p["tail"] != "upright" else side, material="wing")

    # ---------------------------------------------------------------- legs
    ll = float(p["leg_length"])
    lt = float(p["leg_thickness"])
    hip = O + side * bw * 0.22 - upb * bd * 0.28 - ax * 0.02
    knee = np.array([hip[0], hip[1] - 0.02, hip[2] - 0.12])
    ankle = np.array([hip[0], hip[1] + 0.0, knee[2] - ll])
    foot = np.array([hip[0], hip[1] - 0.05, ankle[2] - 0.05])
    c.chain("thigh.L", [hip, knee], [0.11, 0.07], parent="hips", bones=["thigh.L"], blend=0.05, mirror=True)
    c.chain("leg.L", [knee, ankle, foot], [0.035 * lt, 0.03 * lt, 0.03 * lt], parent="thigh.L",
            bones=["leg.L", "foot.L"], blend=0.02, material="legs", mirror=True)
    _feet(c, p, foot, lt)

    # ---------------------------------------------------------------- materials
    _materials(c, p, base_size)
    c.meta["belly_axis"] = (-upb).tolist()  # countershading follows the body, not world-down
    c.poses.update(_pose_presets())
    c.meta["detail_regions"] = [{"bone": "head", "at": 0.0, "radius": hr * 1.9 + float(p["beak_length"]) * 0.6,
                                 "factor": 2.0, "cut_bone": last_neck, "cut_at": 0.6, "layers": ["body"]}]
    lo, hi = c.bounds()
    h = float(hi[2] - lo[2])
    s = (float(p["height"]) if p["height"] else base_size * float(p["size"])) / h
    c.transform(scale=s)
    c.snap_to_ground()
    c.meta["min_feature"] = 0.03 * lt * s
    c.meta["params"] = dict(p, species=species)
    c.meta["user_params"] = dict(kwargs)
    c.meta["template"] = "bird"
    return c


def _rx(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _frame(z):
    z = unit(z)
    x = unit(np.cross([0.0, 0.0, 1.0], z)) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    y = np.cross(z, x)
    return np.column_stack([x, z, y])


def _beak(c, p, head_c, hr, fwd, up, side):
    b = p["beak"]
    bl = float(p["beak_length"])
    bw = float(p["beak_width"])
    root = head_c + fwd * hr * 0.78 - up * hr * 0.05
    if b == "flat":
        c.ellipsoid("beak", root + fwd * bl * 0.5 - up * hr * 0.1, (hr * 0.38 * bw, bl * 0.55, hr * 0.14),
                    bone="head", blend=hr * 0.15, material="beak")
    elif b in ("hook", "bent"):
        down = 0.55 if b == "hook" else 0.9
        pts = [root, root + fwd * bl * 0.55 + up * bl * 0.05, root + fwd * bl * 0.9 - up * bl * 0.25 * down,
               root + fwd * bl * 0.85 - up * bl * 0.6 * down]
        c.chain("beak", pts, np.array([0.36, 0.26, 0.15, 0.03]) * hr * bw, parent="head", make_bones=False,
                blend=hr * 0.12, material="beak", squash=0.75, squash_dir=side)
        c.capsule("lower_beak", root - up * hr * 0.2, root + fwd * bl * 0.6 - up * hr * 0.25, hr * 0.22 * bw,
                  end_radius=hr * 0.05, bone="head", blend=hr * 0.08, material="beak")
    else:
        r0 = {"cone": 0.33, "stout": 0.36, "long": 0.17}[b] * hr * bw
        tip = root + fwd * bl - up * bl * (0.08 if b != "long" else 0.12)
        c.capsule("beak", root, tip, r0, end_radius=r0 * 0.12, bone="head", blend=hr * 0.12, material="beak",
                  squash=0.85, squash_dir=up)
    c.ellipsoid("beak_line", root + fwd * bl * 0.45 - up * hr * 0.06, (hr * 0.5, bl * 0.6, hr * 0.012),
                bone="head", blend=hr * 0.02, op="subtract")


def _feet(c, p, foot, lt):
    if p["feet"] == "none":
        return
    tl = 0.13
    if p["feet"] == "webbed":
        c.ellipsoid("web.L", foot + np.array([0, -tl * 0.55, 0.0]), (0.09, tl * 0.7, 0.012), bone="foot.L",
                    blend=0.02, material="legs", mirror=True)
        return
    for k, ang in enumerate((-28, 0, 28)):
        a = math.radians(ang)
        d = np.array([math.sin(a), -math.cos(a), -0.15])
        c.capsule("toe%d.L" % k, foot, foot + d * tl, 0.022 * lt, end_radius=0.012 * lt, bone="foot.L",
                  blend=0.01, material="legs", mirror=True)
    c.capsule("toe_back.L", foot, foot + np.array([0, tl * 0.6, -0.02]), 0.02 * lt, end_radius=0.011 * lt,
              bone="foot.L", blend=0.01, material="legs", mirror=True)


def _materials(c, p, base_size):
    size = float(p["height"]) if p["height"] else base_size * float(p["size"])
    c.material("body", kind="feathers", color=p["color"], belly_color=p["belly_color"], pattern=p["pattern"],
               pattern_color=p["pattern_color"], pattern_scale=0.08 * size)
    c.material("wing", kind="feathers", color=p["wing_color"] or p["color"], pattern=p["pattern"],
               pattern_color=p["pattern_color"], pattern_scale=0.08 * size)
    if p["head_color"]:
        c.material("head", kind="feathers", color=p["head_color"])
    if p["breast_color"]:
        c.material("breast", kind="feathers", color=p["breast_color"])
    c.material("beak", kind="keratin", color=p["beak_color"])
    c.material("legs", kind="skin", color=p["leg_color"], roughness=0.5, subsurface=0.05, bump=0.3)
