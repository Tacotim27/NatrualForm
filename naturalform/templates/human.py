"""Parametric human: anatomy-based blocking with smooth blending.

Built in a reference space (adult male, 1.75 m) from landmarks and
multipliers, then scaled to the requested height. Faces -Y, left = +X,
feet on the ground (z = 0), arms hanging slightly away from the body.
"""

import math

import numpy as np

from ..core.creature import Creature, NaturalFormError
from .common import choice, clamp, interp, lerp, resolve, smoothstep, sym, unit

HAIR_STYLES = ("bald", "buzz", "short", "bob", "long", "ponytail", "bun", "afro", "mohawk")
BEARDS = ("none", "mustache", "goatee", "full")
CLOTHING = ("none", "underwear", "casual", "shorts", "dress", "suit")
STYLES = ("realistic", "stylized", "chibi")

PARAMS = {
    "height": (None, "metres (default from sex and age)"),
    "sex": ("male", "male | female | neutral"),
    "masculinity": (None, "0 (female) .. 1 (male); overrides sex"),
    "age": (30, "years; children get big heads and short limbs, elders stoop"),
    "muscle": (0.5, "0..1"),
    "fat": (0.3, "0..1"),
    "style": ("realistic", "realistic | stylized | chibi"),
    "head_scale": (1.0, "multiplier"),
    "shoulder_width": (1.0, "multiplier"),
    "hip_width": (1.0, "multiplier"),
    "waist": (1.0, "waist width multiplier"),
    "torso_length": (1.0, "multiplier"),
    "leg_length": (1.0, "multiplier"),
    "arm_length": (1.0, "multiplier"),
    "neck_length": (1.0, "multiplier"),
    "neck_thickness": (1.0, "multiplier"),
    "breast_size": (None, "0..1 (default 0.5 for adult women, 0 otherwise)"),
    "eye_size": (1.0, "multiplier"),
    "nose_size": (1.0, "multiplier"),
    "nose_length": (1.0, "multiplier (how far it projects)"),
    "ear_size": (1.0, "multiplier"),
    "jaw_width": (1.0, "multiplier"),
    "chin_size": (1.0, "multiplier"),
    "lip_fullness": (1.0, "multiplier"),
    "brow_ridge": (1.0, "multiplier"),
    "cheekbones": (1.0, "multiplier"),
    "skin_color": ("light", "porcelain pale fair light medium olive tan brown dark deep ebony, hex or rgb"),
    "eye_color": ("brown", "brown blue green hazel gray amber, hex or rgb"),
    "hair_style": (None, "bald buzz short bob long ponytail bun afro mohawk (default by sex)"),
    "hair_color": (None, "black dark_brown brown light_brown blonde platinum red auburn ginger gray white, hex/rgb"),
    "eyebrows": (True, "add eyebrows"),
    "beard": ("none", "none mustache goatee full"),
    "clothing": ("casual", "none underwear casual shorts dress suit"),
    "top_color": (None, "colour of the shirt / dress / jacket"),
    "bottom_color": (None, "colour of trousers / shorts / skirt"),
    "shoe_color": (None, "colour of shoes"),
    "fingers": (True, "separate fingers (False = mitten hands, faster)"),
}


def _pose_presets():
    P = {}
    P["stand"] = {}
    P["A"] = sym({"upper_arm.L": {"aim": [0.72, 0.0, -0.7]}, "forearm.L": {"aim": [0.72, -0.05, -0.7]},
                  "hand.L": {"aim": [0.72, -0.05, -0.7]}})
    P["T"] = sym({"upper_arm.L": {"aim": [1, 0, 0]}, "forearm.L": {"aim": [1, 0, 0]},
                  "hand.L": {"aim": [1, 0, 0]}})
    P["walk"] = {
        "thigh.L": {"aim": [0, -0.42, -1]}, "shin.L": {"aim": [0, -0.08, -1]}, "foot.L": {"aim": [0, -1, -0.15]},
        "thigh.R": {"aim": [0, 0.3, -1]}, "shin.R": {"aim": [0, 0.75, -1]}, "foot.R": {"aim": [0, -0.45, -1]},
        "upper_arm.L": {"aim": [0.12, 0.38, -1]}, "forearm.L": {"aim": [0.15, 0.05, -1]},
        "upper_arm.R": {"aim": [-0.12, -0.35, -1]}, "forearm.R": {"aim": [-0.12, -0.8, -1]},
        "spine": [0, 0, 4], "chest": [0, 0, -8], "head": [0, 0, 4],
    }
    P["run"] = {
        "spine": [-10, 0, 0],
        "thigh.L": {"aim": [0, -1, -0.45]}, "shin.L": {"aim": [0, 0.15, -1]}, "foot.L": {"aim": [0, -1, -0.3]},
        "thigh.R": {"aim": [0, 0.45, -1]}, "shin.R": {"aim": [0, 1, -0.2]}, "foot.R": {"aim": [0, 0.3, -1]},
        "upper_arm.L": {"aim": [0.15, 0.6, -0.8]}, "forearm.L": {"aim": [0.1, -0.2, -0.95]},
        "upper_arm.R": {"aim": [-0.15, -0.6, -0.75]}, "forearm.R": {"aim": [-0.1, -1, 0.45]},
        "head": [10, 0, 0],
    }
    P["sit"] = sym({"thigh.L": {"aim": [0.08, -1, 0.0]}, "shin.L": {"aim": [0.0, 0.05, -1]},
                    "foot.L": {"aim": [0, -1, -0.15]},
                    "upper_arm.L": {"aim": [0.1, -0.15, -1]}, "forearm.L": {"aim": [0.0, -1, -0.35]},
                    "hand.L": {"aim": [0.0, -1, -0.5]}})
    P["sit_ground"] = sym({"thigh.L": {"aim": [0.25, -1, 0.15]}, "shin.L": {"aim": [0.1, -1, -0.15]},
                           "foot.L": {"aim": [0, -0.4, 1]},
                           "upper_arm.L": {"aim": [0.3, 0.6, -1]}, "forearm.L": {"aim": [0.05, 0.25, -1]},
                           "hand.L": {"aim": [0.05, 0.6, -1]}})
    P["wave"] = {"upper_arm.R": {"aim": [-1, -0.1, 0.35]}, "forearm.R": {"aim": [-0.2, -0.15, 1]},
                 "hand.R": {"aim": [-0.12, -0.1, 1]}, "head": [0, 0, -8]}
    P["hands_up"] = sym({"shoulder.L": [0, -12, 0], "upper_arm.L": {"aim": [0.35, -0.05, 1]},
                         "forearm.L": {"aim": [0.15, -0.05, 1]}, "hand.L": {"aim": [0.1, 0, 1]}})
    P["point"] = {"upper_arm.R": {"aim": [-0.2, -1, 0.15]}, "forearm.R": {"aim": [-0.15, -1, 0.15]},
                  "hand.R": {"aim": [-0.15, -1, 0.15]}, "head": [0, 0, -10]}
    P["hands_on_hips"] = sym({"upper_arm.L": {"aim": [0.55, 0.25, -0.75]},
                              "forearm.L": {"aim": [-0.75, -0.25, -0.35]},
                              "hand.L": {"aim": [-0.5, 0.0, -0.9]}})
    P["kneel"] = {"thigh.L": {"aim": [0, -1, -0.05]}, "shin.L": {"aim": [0, 0.05, -1]}, "foot.L": {"aim": [0, -1, -0.1]},
                  "thigh.R": {"aim": [0, -0.1, -1]}, "shin.R": {"aim": [0, 1, -0.05]}, "foot.R": {"aim": [0, 0.6, -1]},
                  "upper_arm.L": {"aim": [0.1, -0.3, -1]}, "forearm.L": {"aim": [0, -1, -0.2]}}
    P["think"] = {"upper_arm.R": {"aim": [-0.1, -0.35, -1]}, "forearm.R": {"aim": [0.25, -0.45, 1]},
                  "hand.R": {"aim": [0.4, -0.3, 1]}, "head": [12, 0, -6],
                  "upper_arm.L": {"aim": [0.05, -0.3, -1]}, "forearm.L": {"aim": [-0.85, -0.5, 0.0]}}
    P["arms_crossed"] = sym({"upper_arm.L": {"aim": [0.25, -0.45, -1]}, "forearm.L": {"aim": [-1, -0.25, 0.05]},
                             "hand.L": {"aim": [-1, 0.15, 0.2]}})
    return P


POSES = tuple(_pose_presets())


def _defaults_by_age(sex_m, age):
    male = interp(age, [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 60, 85],
                  [0.52, 0.76, 0.87, 1.03, 1.16, 1.28, 1.39, 1.50, 1.64, 1.72, 1.76, 1.75, 1.71])
    female = interp(age, [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 60, 85],
                    [0.51, 0.74, 0.86, 1.02, 1.15, 1.27, 1.38, 1.51, 1.59, 1.62, 1.63, 1.62, 1.57])
    return lerp(female, male, sex_m)


def human(**kwargs):
    """Return a :class:`Creature` describing a human (see PARAMS)."""
    p = resolve(PARAMS, kwargs, "human")
    sex = str(p["sex"]).lower()
    if p["masculinity"] is not None:
        m = clamp(p["masculinity"])
    else:
        m = {"male": 1.0, "man": 1.0, "boy": 1.0, "m": 1.0, "female": 0.0, "woman": 0.0,
             "girl": 0.0, "f": 0.0, "neutral": 0.5, "androgynous": 0.5}.get(sex)
        if m is None:
            raise NaturalFormError("human: sex must be male, female or neutral (or use masculinity=0..1)")
    style = choice(p["style"], STYLES, "human style")
    age = clamp(p["age"], 0.3, 110.0)
    pub = smoothstep(9.0, 16.0, age)
    me = 0.5 + (m - 0.5) * pub  # children look androgynous

    def L(f, mm):
        return lerp(f, mm, me)

    old = smoothstep(55.0, 90.0, age)
    mu = clamp(p["muscle"]) * (1 - 0.35 * old) * lerp(0.5, 1.0, pub)
    fa = clamp(p["fat"])
    if age < 4:
        fa = max(fa, lerp(0.7, 0.4, age / 4))
    height = p["height"] if p["height"] else _defaults_by_age(m, age)
    if not 0.2 <= float(height) <= 3.5:
        raise NaturalFormError("human: height %r m is out of range 0.2 .. 3.5" % height)

    heads = interp(age, [0, 1, 3, 6, 10, 14, 18, 25], [3.9, 4.2, 4.9, 5.6, 6.2, 6.8, 7.2, 7.5])
    eye_mult = float(p["eye_size"]) * interp(age, [0, 6, 18], [1.18, 1.08, 1.0])
    limb_style = 1.0
    if style == "stylized":
        heads *= 0.86
        eye_mult *= 1.3
    elif style == "chibi":
        heads = min(heads, 3.0)
        eye_mult *= 1.75
        limb_style = 0.8

    lm = float(p["leg_length"]) * interp(age, [0, 2, 6, 10, 14, 18], [0.60, 0.70, 0.84, 0.92, 0.98, 1.0]) * limb_style
    am = float(p["arm_length"]) * interp(age, [0, 2, 6, 10, 18], [0.78, 0.84, 0.92, 0.96, 1.0]) * limb_style
    tm = float(p["torso_length"]) * L(0.97, 1.0)
    nm = float(p["neck_length"]) * interp(age, [0, 6, 18], [0.55, 0.85, 1.0])
    wk = interp(age, [0, 6, 14, 18], [1.12, 1.0, 0.97, 1.0])
    sw = float(p["shoulder_width"]) * wk
    hw = float(p["hip_width"]) * wk
    thick = L(0.9, 1.0) * (0.9 + 0.22 * mu) * (0.93 + 0.32 * fa) * interp(age, [0, 3, 8, 18], [1.3, 1.12, 0.95, 1.0])
    breast = p["breast_size"]
    if breast is None:
        breast = 0.5 * (1 - m) * pub
    breast = clamp(breast) * pub

    c = Creature("Human", "human")
    c.default_material = "skin"
    c.meta["ground"] = True

    # ------------------------------------------------------------- landmarks
    ankle_z = 0.08 * lm ** 0.6
    knee_z = ankle_z + 0.42 * lm
    hip_z = knee_z + 0.41 * lm
    T = 0.52 * tm
    waist_z = hip_z + 0.33 * T
    rib_z = hip_z + 0.62 * T
    chest_z = hip_z + 0.68 * T
    sh_z = hip_z + 0.925 * T
    nb_z = hip_z + T
    chin_z = nb_z + 0.09 * nm
    head_h = chin_z / (heads - 1.0) * float(p["head_scale"])
    hs = head_h / 0.23
    O = np.array([0.0, 0.0, chin_z + 0.115 * hs])  # head centre

    def H(x, y, z):
        return O + hs * np.array([x, y, z], dtype=np.float64)

    pelvis_w = L(0.172, 0.155) * hw * (1 + 0.22 * fa)
    pelvis_d = L(0.102, 0.098) * (1 + 0.25 * fa)
    waist_w = L(0.104, 0.12) * float(p["waist"]) * (1 + 0.42 * fa) * lerp(1.0, 0.96, mu)
    waist_d = L(0.085, 0.092) * (1 + 0.6 * fa)
    rib_w = L(0.126, 0.152) * sw ** 0.5 * (1 + 0.12 * fa + 0.07 * mu)
    rib_d = L(0.092, 0.104) * (1 + 0.22 * fa)
    sh_x = L(0.172, 0.2) * sw * lerp(0.97, 1.04, mu)
    sj_x = sh_x - 0.022
    hip_x = L(0.086, 0.082) * hw * (1 + 0.1 * fa)

    # ------------------------------------------------------------- skeleton
    spine_pts = [(0, 0.02, hip_z), (0, 0.022, waist_z), (0, 0.03, rib_z), (0, 0.02, nb_z)]
    c.chain("spine", spine_pts, 0.05, bones=["hips", "spine", "chest"], visible=False)

    # ------------------------------------------------------------- torso
    c.ellipsoid("pelvis", (0, 0.008, hip_z + 0.035), (pelvis_w, pelvis_d, 0.115), bone="hips", blend=0.0)
    c.ellipsoid("glute", (0.062, 0.05 + 0.01 * fa, hip_z - 0.015),
                np.array([0.07, 0.058, 0.085]) * L(1.12, 1.0) * (0.95 + 0.25 * fa + 0.1 * mu),
                rotate=[0, -12, 0], bone="hips", blend=0.04, mirror=True)
    c.ellipsoid("abdomen", (0, -0.004, waist_z), (waist_w, waist_d, 0.125), bone="spine", blend=0.05)
    c.ellipsoid("ribcage", (0, 0.012, rib_z), (rib_w, rib_d, 0.16), rotate=[-6, 0, 0], bone="chest", blend=0.05)
    c.capsule("upper_chest", (-sh_x + 0.075, 0.012, sh_z - 0.03), (sh_x - 0.075, 0.012, sh_z - 0.03),
              L(0.06, 0.068) * (0.95 + 0.15 * mu), bone="chest", blend=0.05)
    c.ellipsoid("scapula", (0.068, 0.072, sh_z - 0.1), (0.06, 0.028, 0.075), rotate=[0, 0, 12],
                bone="chest", blend=0.04, mirror=True)
    pec = (0.85 + 0.45 * mu) * L(0.6, 1.0)
    c.ellipsoid("pec", (0.06, -0.064, chest_z + 0.035), (0.068, 0.03 * pec, 0.05 * pec ** 0.5),
                rotate=[0, 0, -10], bone="chest", blend=0.04, mirror=True)
    if breast > 0.03:
        b = 0.6 + 0.8 * breast
        c.ellipsoid("breast", (0.066, -0.078 - 0.012 * b, chest_z - 0.012 - 0.006 * b),
                    (0.05 * b, 0.042 * b, 0.046 * b), rotate=[12, 0, -14], bone="chest",
                    blend=0.035, mirror=True)
    if fa > 0.2:
        f = (fa - 0.2) / 0.8
        c.ellipsoid("belly", (0, -0.04 - 0.045 * f, waist_z - 0.03),
                    (waist_w * 0.85, 0.06 + 0.06 * f, 0.1 + 0.03 * f), bone="spine", blend=0.06)
        c.ellipsoid("love_handle", (waist_w * 0.75, 0.01, waist_z - 0.04), (0.04 + 0.03 * f, 0.06, 0.06),
                    bone="spine", blend=0.05, mirror=True)

    # ------------------------------------------------------------- neck & head
    neck_r = L(0.045, 0.053) * float(p["neck_thickness"]) * (1 + 0.25 * fa + 0.1 * mu) * interp(age, [0, 6, 18], [1.25, 1.05, 1.0])
    neck_top = H(0, 0.018, -0.074)
    c.chain("neck", [(0, 0.012, nb_z - 0.015), neck_top], [neck_r * 1.12, neck_r * 0.86], parent="chest",
            bones=["neck"], blend=0.04)
    c.bone("head", neck_top, H(0, 0.01, 0.12), parent="neck", connect=True)
    _head(c, H, hs, p, me, age, eye_mult)

    # ------------------------------------------------------------- arms
    c.bone("shoulder", (0.025, -0.01, nb_z - 0.02), (sj_x - 0.012, 0.0, sh_z + 0.012), parent="chest",
           mirror=True)
    c.capsule("trapezius", (0.03, 0.035, nb_z + 0.035), (sh_x - 0.035, 0.022, sh_z + 0.012),
              L(0.024, 0.03) * (0.9 + 0.35 * mu), end_radius=0.026 * (0.9 + 0.25 * mu),
              bone="shoulder.L", blend=0.03, mirror=True)
    abd = math.radians(7.0)
    sj = np.array([sj_x, 0.0, sh_z])
    d_ua = unit([math.sin(abd), 0.02, -math.cos(abd)])
    elbow = sj + d_ua * 0.30 * am
    d_fa = unit([math.sin(abd + 0.09), -0.17, -math.cos(abd + 0.09)])
    wrist = elbow + d_fa * 0.255 * am
    r_sh, r_el, r_wr = 0.047 * thick, 0.034 * thick, 0.026 * thick ** 0.7
    c.chain("arm.L", [sj, elbow, wrist], [r_sh, r_el, r_wr], parent="shoulder.L",
            bones=["upper_arm.L", "forearm.L"], blend=0.025, mirror=True)
    c.ellipsoid("deltoid", sj + np.array([0.016, 0.0, -0.035]), np.array([0.044, 0.05, 0.068]) * (0.9 + 0.3 * mu) * L(0.9, 1.0),
                rotate=[0, 8, 0], bone="upper_arm.L", blend=0.03, mirror=True)
    mf = (0.85 + 0.4 * mu) * L(0.82, 1.0) * (1 + 0.2 * fa)
    c.ellipsoid("biceps", sj + d_ua * 0.55 * 0.30 * am + np.array([0, -0.01, 0]),
                np.array([0.032, 0.085 * am, 0.03]) * mf, align=d_ua, bone="upper_arm.L", blend=0.02, mirror=True)
    c.ellipsoid("triceps", sj + d_ua * 0.42 * 0.30 * am + np.array([0, 0.014, 0]),
                np.array([0.034, 0.09 * am, 0.032]) * mf, align=d_ua, bone="upper_arm.L", blend=0.02, mirror=True)
    c.ellipsoid("forearm_muscle", elbow + d_fa * 0.3 * 0.255 * am + np.array([0.002, -0.004, 0]),
                np.array([0.031, 0.075 * am, 0.037]) * (0.9 + 0.25 * mu) * L(0.88, 1.0),
                align=d_fa, bone="forearm.L", blend=0.02, mirror=True)

    # hand frame: u = along fingers, v = toward thumb, w = out of the palm
    u = d_fa
    w = unit(np.array([-1.0, 0.0, 0.0]) - np.dot([-1.0, 0.0, 0.0], u) * u)
    v = np.cross(w, u)
    hsc = am * L(0.9, 1.0) * interp(age, [0, 6, 18], [0.85, 0.95, 1.0])
    _hand(c, wrist, u, v, w, hsc, thick, bool(p["fingers"]))

    # ------------------------------------------------------------- legs
    hip = np.array([hip_x, 0.0, hip_z])
    knee = np.array([hip_x + 0.006, -0.008, knee_z])
    ankle = np.array([hip_x + 0.012, 0.012, ankle_z])
    tl = thick * L(1.08, 1.0)
    c.chain("leg.L", [hip, knee, ankle], [0.086 * tl, 0.052 * tl, 0.034 * tl ** 0.6], parent="hips",
            bones=["thigh.L", "shin.L"], blend=0.05, mirror=True)
    d_th = unit(knee - hip)
    d_sh = unit(ankle - knee)
    c.ellipsoid("quads", hip + (knee - hip) * 0.42 + np.array([0.008, -0.014, 0]),
                np.array([0.074, 0.16 * lm, 0.076]) * tl * (0.92 + 0.2 * mu), align=d_th,
                bone="thigh.L", blend=0.03, mirror=True)
    c.ellipsoid("hamstring", hip + (knee - hip) * 0.4 + np.array([-0.006, 0.016, 0]),
                np.array([0.07, 0.15 * lm, 0.07]) * tl, align=d_th, bone="thigh.L", blend=0.03, mirror=True)
    c.ellipsoid("calf", knee + (ankle - knee) * 0.3 + np.array([-0.002, 0.02, 0]),
                np.array([0.052, 0.11 * lm, 0.055]) * tl * (0.9 + 0.3 * mu), align=d_sh,
                bone="shin.L", blend=0.025, mirror=True)
    c.sphere("kneecap", knee + np.array([0, -0.038, 0.004]) * tl, 0.022 * tl, bone="shin.L", blend=0.015, mirror=True)
    _foot(c, ankle, lm, L)

    # ------------------------------------------------------------- hair & clothes
    body_lo, body_hi = c.bounds()
    _hair(c, H, hs, p, m, age)
    _clothing(c, p, hip_z, waist_z, sh_z, knee_z, ankle_z, m, sj, elbow, wrist, H)

    _materials(c, p, m, age)
    c.poses.update(_pose_presets())
    if old > 0.05:
        c.poses["stand"] = {"chest": [10 * old, 0, 0], "neck": [8 * old, 0, 0], "head": [-10 * old, 0, 0]}

    # finer meshing for the face and hands (zipped into the body mesh)
    c.meta["detail_regions"] = [
        {"bone": "head", "at": 0.4, "radius": 0.16 * hs, "factor": 3.6, "cut_bone": "neck", "cut_at": 0.55,
         "layers": ["body", "hair", "brows", "beard"]},
        {"bone": "hand.L", "at": 0.9, "radius": 0.13 * hsc, "factor": 3.0, "cut_bone": "forearm.L", "cut_at": 0.85,
         "layers": ["body"]},
        {"bone": "hand.R", "at": 0.9, "radius": 0.13 * hsc, "factor": 3.0, "cut_bone": "forearm.R", "cut_at": 0.85,
         "layers": ["body"]},
    ]

    # ------------------------------------------------------------- final scale
    s = float(height) / float(body_hi[2] - body_lo[2])
    c.transform(scale=s)
    c.snap_to_ground()
    c.meta["params"] = dict(p, masculinity=m, height=float(height))
    c.meta["user_params"] = dict(kwargs)
    c.meta["template"] = "human"
    return c


def _head(c, H, hs, p, me, age, eye_mult):
    L = lambda f, m: lerp(f, m, me)
    ff = interp(age, [0, 6, 18], [0.8, 0.9, 1.0])  # children have small faces
    ez = interp(age, [0, 6, 18], [-0.014, -0.007, 0.0])
    jaw = float(p["jaw_width"]) * L(0.92, 1.05)
    c.ellipsoid("cranium", H(0, 0.012, 0.025), np.array([0.074, 0.097, 0.09]) * hs, rotate=[-8, 0, 0],
                bone="head", blend=0.02)
    c.ellipsoid("face", H(0, -0.03, -0.028 + ez * 0.5), np.array([0.054, 0.062, 0.062]) * hs, bone="head", blend=0.018)
    c.ellipsoid("jaw", H(0, -0.026, -0.074), np.array([0.044 * jaw, 0.062, 0.04 * ff]) * hs, rotate=[8, 0, 0],
                bone="head", blend=0.018)
    c.ellipsoid("muzzle", H(0, -0.064, -0.064 + ez * 0.6), np.array([0.03, 0.028, 0.03]) * hs * ff,
                bone="head", blend=0.009)
    c.sphere("jaw_angle", H(0.044 * jaw, 0.016, -0.064), 0.013 * hs * L(0.8, 1.0), bone="head", blend=0.014,
             mirror=True)
    ch = float(p["chin_size"])
    c.ellipsoid("chin", H(0, -0.077 * ff, -0.097 * lerp(1, 0.95, 1 - ff)),
                np.array([0.021 * L(0.85, 1.1), 0.016, 0.018]) * hs * ch * ff, bone="head", blend=0.01)
    cb = float(p["cheekbones"])
    c.ellipsoid("cheekbone", H(0.047, -0.052, -0.018 + ez), np.array([0.017, 0.015, 0.013]) * hs * cb,
                rotate=[0, 0, -25], bone="head", blend=0.016, mirror=True)
    c.ellipsoid("cheek", H(0.031, -0.062, -0.036 + ez), np.array([0.025, 0.02, 0.024]) * hs * ff,
                rotate=[0, 0, -20], bone="head", blend=0.012, mirror=True)
    br = float(p["brow_ridge"]) * L(0.7, 1.0)
    c.capsule("brow", H(0.0, -0.08, 0.02 + ez * 0.5), H(0.044, -0.072, 0.022 + ez * 0.5), 0.0085 * hs * br,
              bone="head", blend=0.012, mirror=True)
    # nose
    ns, nl = float(p["nose_size"]) * ff, float(p["nose_length"]) * ff
    c.capsule("nose_bridge", H(0, -0.084, -0.002 + ez), H(0, -0.086 - 0.022 * nl, -0.037 + ez), 0.0052 * hs * ns,
              end_radius=0.0086 * hs * ns, bone="head", blend=0.007)
    c.sphere("nose_wing", H(0.0105 * ns, -0.093 - 0.006 * (nl - 1), -0.041 + ez), 0.0066 * hs * ns, bone="head",
             blend=0.004, mirror=True)
    c.sphere("nostril", H(0.0062 * ns, -0.097 - 0.006 * (nl - 1), -0.0475 + ez), 0.0025 * hs * ns, bone="head",
             blend=0.003, op="subtract", mirror=True)
    # mouth
    lf = float(p["lip_fullness"]) * L(1.12, 0.95) * ff
    mz = -0.071 + ez * 0.6
    # each lip = two halves that curve back around the dental arch
    c.ellipsoid("upper_lip", H(0.0062, -0.0905, mz + 0.0045), np.array([0.0152, 0.0076, 0.0044 * lf]) * hs,
                rotate=[-14, 0, -16], bone="head", blend=0.0032, material="lips", mirror=True)
    c.ellipsoid("lower_lip", H(0.0056, -0.0885, mz - 0.0064), np.array([0.014, 0.008, 0.0052 * lf]) * hs,
                rotate=[12, 0, -15], bone="head", blend=0.0022, material="lips", mirror=True)
    # ears
    es = float(p["ear_size"]) * interp(age, [0, 18], [0.85, 1.0])
    c.ellipsoid("ear", H(0.073, 0.012, -0.006), np.array([0.011, 0.023, 0.031]) * hs * es, rotate=[-12, 0, -18],
                bone="head", blend=0.006, mirror=True)
    c.ellipsoid("ear_hollow", H(0.083, 0.008, -0.006), np.array([0.006, 0.014, 0.019]) * hs * es,
                rotate=[-12, 0, -18], bone="head", blend=0.004, op="subtract", mirror=True)
    # eyes
    er = 0.0118 * hs * eye_mult
    c.eye("eye.L", H(0.032, -0.0685, ez), er, look=(0.0, -1.0, 0.0), bone="head", color=p["eye_color"],
          iris_size=0.5, pupil_size=0.42, sclera="white", socket_blend=er * 0.42, mirror=True)
    # eyelids: soft lids over the top and bottom of each eye
    ec = H(0.032, -0.0685, ez)
    c.sphere("eyelid", ec, er * 1.1, bone="head", blend=0.002, mirror=True,
             clip=[(ec + [0, 0, er * 0.3], (0, 0.25, 1))], clip_blend=0.001)
    c.sphere("lower_lid", ec, er * 1.07, bone="head", blend=0.002, mirror=True,
             clip=[(ec - [0, 0, er * 0.62], (0, 0.3, -1))], clip_blend=0.001)


def _hand(c, wrist, u, v, w, s, thick, fingers):
    R = np.column_stack([v, u, w])  # local x = across, y = along, z = palm normal

    def P(du, dv, dw):
        return wrist + s * (du * u + dv * v + dw * w)

    t = thick ** 0.5
    c.bone("hand.L", wrist, P(0.095, 0.004, 0.0), parent="forearm.L", connect=True, mirror=True)
    c.box("palm", P(0.05, -0.002, -0.001), np.array([0.041, 0.047, 0.0125]) * s * np.array([1, 1, t]),
          rounding=0.0115 * s, rotate=R, bone="hand.L", blend=0.012, mirror=True)
    c.ellipsoid("thenar", P(0.035, 0.026, 0.008), np.array([0.016, 0.026, 0.013]) * s * t, rotate=R,
                bone="hand.L", blend=0.01, mirror=True)
    c.ellipsoid("hypothenar", P(0.048, -0.03, 0.005), np.array([0.012, 0.032, 0.011]) * s * t, rotate=R,
                bone="hand.L", blend=0.008, mirror=True)
    rf = 0.0085 * s * t
    if fingers:
        defs = [  # name, knuckle (u, v, w), length, spread deg
            ("index", (0.093, 0.027, -0.001), 0.082, 7.0),
            ("middle", (0.097, 0.008, 0.0), 0.09, 1.5),
            ("ring", (0.093, -0.011, -0.001), 0.085, -4.5),
            ("pinky", (0.084, -0.028, -0.004), 0.066, -11.0),
        ]
        curls = (14.0, 32.0, 50.0)
        for name, (ku, kv, kw), ln, spread in defs:
            a = math.radians(spread)
            us = math.cos(a) * u + math.sin(a) * v
            pts = [P(ku, kv, kw)]
            for seg, frac in enumerate((0.45, 0.3, 0.25)):
                th = math.radians(curls[seg])
                d = math.cos(th) * us + math.sin(th) * w
                pts.append(pts[-1] + d * ln * frac * s)
            k = 1.0 if name != "pinky" else 0.86
            radii = [rf * k * 1.08, rf * k * 0.98, rf * k * 0.9, rf * k * 0.8]
            c.chain("%s.L" % name, pts, radii, parent="hand.L",
                    bones=["%s_%d.L" % (name, i) for i in range(3)], blend=0.006 * s, joint_blend=0.002 * s,
                    squash=0.86, squash_dir=w, mirror=True)
    else:
        c.box("fingers", P(0.135, -0.0, 0.006), np.array([0.039, 0.042, 0.0095]) * s,
              rounding=0.009 * s, rotate=R, bone="hand.L", blend=0.012, mirror=True)
    thumb = [P(0.016, 0.024, 0.004), P(0.05, 0.047, 0.016), P(0.079, 0.06, 0.024), P(0.103, 0.066, 0.03)]
    c.chain("thumb.L", thumb, [rf * 1.45, rf * 1.25, rf * 1.1, rf * 0.95], parent="hand.L",
            bones=["thumb_0.L", "thumb_1.L", "thumb_2.L"], blend=0.008 * s, joint_blend=0.003 * s,
            squash=0.88, squash_dir=w, mirror=True)


def _foot(c, ankle, lm, L):
    fs = lm ** 0.7 * L(0.92, 1.0)
    ax, ay = ankle[0], ankle[1]

    def F(x, y, z):
        return np.array([ax + x * fs, ay + y * fs, z * fs])

    c.chain("foot.L", [ankle, F(0.006, -0.12, 0.024), F(0.008, -0.2, 0.015)], [0.001, 0.001, 0.001],
            parent="shin.L", bones=["foot.L", "toe.L"], visible=False, mirror=True)
    c.ellipsoid("heel", F(-0.002, 0.032, 0.036), np.array([0.029, 0.04, 0.036]) * fs, bone="foot.L",
                blend=0.02, mirror=True)
    c.ellipsoid("midfoot", F(0.004, -0.05, 0.032), np.array([0.038, 0.09, 0.032]) * fs, rotate=[6, 0, -3],
                bone="foot.L", blend=0.022, mirror=True)
    c.ellipsoid("instep", F(0.0, -0.02, 0.058), np.array([0.03, 0.06, 0.026]) * fs, rotate=[-22, 0, 0],
                bone="foot.L", blend=0.02, mirror=True)
    c.ellipsoid("forefoot", F(0.008, -0.148, 0.019), np.array([0.044, 0.048, 0.019]) * fs, rotate=[0, 0, -6],
                bone="toe.L", blend=0.016, mirror=True)
    c.ellipsoid("big_toe", F(-0.016, -0.186, 0.015), np.array([0.013, 0.02, 0.014]) * fs,
                bone="toe.L", blend=0.008, mirror=True)
    c.sphere("ankle_bone", F(0.0, 0.004, ankle[2] / fs) + np.array([0.0, 0.0, 0.0]), 0.034 * fs, bone="shin.L",
             blend=0.015, mirror=True)


def _hair(c, H, hs, p, m, age):
    style = p["hair_style"]
    if style is None:
        style = "short" if m >= 0.5 else "long"
        if age < 2:
            style = "buzz"
    style = choice(style, HAIR_STYLES, "hair_style")
    color = p["hair_color"]
    if color is None:
        color = "gray_hair" if age > 62 else "brown"
    c.material("hair", kind="hair", color=color)
    LAYER = "hair"
    c.material("brows", kind="hair", color=color, bump=0.2)

    if p["eyebrows"] and age >= 0.5:
        bm = lerp(0.8, 1.15, m)
        c.capsule("eyebrow", H(0.012, -0.0818, 0.0245), H(0.048, -0.073, 0.0275), 0.0075 * hs,
                  end_radius=0.0058 * hs, squash=0.33 * bm, squash_dir=(0, 0, 1), bone="head",
                  blend=0.002, material="brows", mirror=True, layer="brows")
    beard = choice(p["beard"], BEARDS, "beard")
    if beard != "none":
        LAYER = "beard"
        c.ellipsoid("mustache", H(0, -0.096, -0.0585), np.array([0.025, 0.012, 0.0055]) * hs, rotate=[-10, 0, 0],
                    bone="head", blend=0.003, material="hair", layer=LAYER)
        if beard in ("goatee", "full"):
            c.ellipsoid("goatee", H(0, -0.084, -0.1), np.array([0.022, 0.018, 0.022]) * hs, bone="head",
                        blend=0.004, material="hair", layer=LAYER)
        if beard == "full":
            c.ellipsoid("beard", H(0, -0.03, -0.083), np.array([0.064, 0.068, 0.046]) * hs, bone="head",
                        blend=0.005, material="hair", layer=LAYER, clip=[(H(0, 0, -0.088), (0, 0, -1))], clip_blend=0.004)
            c.ellipsoid("beard_side", H(0.043, -0.028, -0.06), np.array([0.025, 0.058, 0.045]) * hs,
                        bone="head", blend=0.004, material="hair", layer=LAYER, mirror=True,
                        clip=[(H(0.03, 0, 0), (1, 0, 0)), (H(0, 0, -0.022), (0, 0, -1))], clip_blend=0.004)
    LAYER = "hair"
    if style == "bald":
        return
    t = {"buzz": 0.0025, "afro": 0.0}.get(style, 0.0075)
    shell = (np.array([0.074, 0.097, 0.09]) + t + 0.001) * hs
    cen = H(0, 0.013, 0.027)
    front = (H(0, -0.084, 0.052), unit([0, 0.52, 0.86]))
    top_cut = (H(0, 0, 0.026), (0, 0, 1))
    if style == "afro":
        c.ellipsoid("hair", H(0, 0.012, 0.045), np.array([0.122, 0.13, 0.118]) * hs, bone="head", blend=0.004,
                    material="hair", layer=LAYER, clip=[front, (H(0, 0, -0.035), (0, 0, 1))], clip_blend=0.006)
        return
    if style == "mohawk":
        c.ellipsoid("hair", H(0, 0.01, 0.1), np.array([0.014, 0.105, 0.05]) * hs, bone="head", blend=0.004,
                    material="hair", layer=LAYER, clip=[front])
        return
    c.ellipsoid("hair", cen, shell, rotate=[-8, 0, 0], bone="head", blend=0.003, material="hair", layer=LAYER,
                clip=[front, top_cut], clip_blend=0.004)
    c.ellipsoid("hair_back", cen, shell, rotate=[-8, 0, 0], bone="head", blend=0.003, material="hair", layer=LAYER,
                clip=[(H(0, 0.012, 0), (0, 1, 0)), (H(0, 0, -0.06), (0, 0, 1))], clip_blend=0.004)
    if m > 0.5 and style in ("short", "buzz"):
        c.capsule("sideburn", H(0.071, -0.004, 0.03), H(0.073, -0.0, -0.002), 0.0055 * hs, squash=0.5,
                  squash_dir=(1, 0, 0), bone="head", blend=0.003, material="hair", layer=LAYER, mirror=True)
    if style == "bob":
        c.ellipsoid("hair_bob", H(0, 0.022, -0.012), np.array([0.091, 0.107, 0.1]) * hs, bone="head", blend=0.004,
                    material="hair", layer=LAYER, clip=[(H(0, -0.048, 0), (0, 1, 0)), (H(0, 0, -0.078), (0, 0, 1))],
                    clip_blend=0.006)
    elif style == "long":
        c.ellipsoid("hair_bob", H(0, 0.022, -0.0), np.array([0.089, 0.106, 0.1]) * hs, bone="head", blend=0.004,
                    material="hair", layer=LAYER, clip=[(H(0, -0.035, 0), (0, 1, 0))], clip_blend=0.006)
        c.ellipsoid("hair_long", H(0, 0.062, -0.13), np.array([0.092, 0.055, 0.2]) * hs, rotate=[-6, 0, 0],
                    bone="head", blend=0.03 * hs, material="hair", layer=LAYER,
                    clip=[(H(0, 0.0, 0), (0, 1, 0)), (H(0, 0, -0.3), (0, 0, 1))], clip_blend=0.01)
    elif style == "ponytail":
        c.chain("ponytail", [H(0, 0.1, 0.03), H(0, 0.135, -0.03), H(0, 0.13, -0.13), H(0, 0.118, -0.2)],
                np.array([0.022, 0.024, 0.018, 0.008]) * hs, parent="head", bones=["ponytail_0", "ponytail_1", "ponytail_2"],
                blend=0.01, material="hair", layer=LAYER)
    elif style == "bun":
        c.sphere("bun", H(0, 0.078, 0.085), 0.036 * hs, bone="head", blend=0.008, material="hair", layer=LAYER)


def _clothing(c, p, hip_z, waist_z, sh_z, knee_z, ankle_z, m, sj, elbow, wrist, H):
    style = choice(p["clothing"], CLOTHING, "clothing")
    if style == "none":
        return
    top = p["top_color"] or {"casual": (70, 100, 150), "shorts": (215, 205, 180), "dress": (150, 40, 60),
                             "suit": (34, 36, 44), "underwear": (232, 230, 226)}[style]
    bottom = p["bottom_color"] or {"casual": (52, 62, 92), "shorts": (90, 96, 70), "dress": top,
                                   "suit": (34, 36, 44), "underwear": (232, 230, 226)}[style]
    shoes = p["shoe_color"] or {"casual": (235, 235, 235), "shorts": (110, 80, 60), "dress": (30, 26, 26),
                                "suit": (24, 20, 18), "underwear": None}[style]
    c.material("top", kind="plain", color=top, roughness=0.85, sheen=0.4, bump=0.0)
    c.material("bottom", kind="plain", color=bottom, roughness=0.8, sheen=0.3, bump=0.0)
    torso = ["pelvis", "glute.L", "glute.R"]
    upper = ["abdomen", "ribcage", "upper_chest", "scapula.L", "scapula.R", "pec.L", "pec.R"]
    upper += [n for n in ("breast.L", "breast.R", "belly", "love_handle.L", "love_handle.R") if _has(c, n)]
    if style == "underwear":
        c.clothe("briefs", torso, 0.003, "bottom", layer="underwear",
                 clip=[((0, 0, hip_z + 0.045), (0, 0, -1))], clip_blend=0.002)
        c.clothe("briefs_legs", ["leg.L", "leg.R"], 0.003, "bottom", layer="underwear",
                 clip=[((0, 0, hip_z - 0.06), (0, 0, 1)), ((0, 0, hip_z + 0.045), (0, 0, -1))])
        if m < 0.5:
            src = [n for n in ("breast.L", "breast.R") if _has(c, n)] or ["pec.L", "pec.R"]
            c.clothe("bra", src, 0.003, "top", layer="underwear")
        return
    sleeve_end = sj + (elbow - sj) * 0.55
    belly = [n for n in ("belly", "love_handle.L", "love_handle.R") if _has(c, n)]
    waist_cut = [((0, 0, waist_z + 0.02), (0, 0, -1))]
    if style in ("casual", "shorts"):
        c.clothe("shirt", upper + ["trapezius.L", "trapezius.R"] + torso, 0.008, "top", layer="top",
                 clip=[((0, 0, hip_z - 0.01), (0, 0, 1))], clip_blend=0.003)
        c.clothe("sleeve", ["deltoid.L", "deltoid.R"], 0.01, "top", layer="top")
        for side, sgn in (("L", 1), ("R", -1)):
            a = sj * [sgn, 1, 1]
            b = sleeve_end * [sgn, 1, 1]
            c.capsule("sleeve_arm.%s" % side, a, b, 0.056 * (1 + 0.3 * p["fat"]), end_radius=0.047,
                      bone="upper_arm.%s" % side, blend=0.01, material="top", layer="top")
    elif style == "dress":
        c.clothe("bodice", upper + torso, 0.006, "top", layer="top", clip=[((0, 0, sh_z - 0.06), (0, 0, -1))],
                 clip_blend=0.003)
        # skirt pieces follow the thighs, so it drapes over the lap when sitting
        c.clothe("skirt_hips", torso + ["abdomen"] + belly, 0.02, "top", layer="top", clip=waist_cut,
                 clip_blend=0.004)
        for side, sgn in (("L", 1), ("R", -1)):
            th = c.bones["thigh.%s" % side]
            d = unit(th.tail - th.head)
            hem = th.head + (th.tail - th.head) * 0.92
            c.capsule("skirt_leg.%s" % side, th.head + d * 0.02 + [0.012 * sgn, 0, 0], hem + [0.03 * sgn, 0, 0],
                      0.12 + 0.03 * p["fat"], end_radius=0.17, squash=0.85, squash_dir=(0, 1, 0),
                      bone="thigh.%s" % side, blend=0.03, material="top", layer="top",
                      clip=[(hem, d * -1)], clip_blend=0.006)
    elif style == "suit":
        c.clothe("jacket", upper + ["trapezius.L", "trapezius.R"] + torso, 0.016, "top", layer="top",
                 clip=[((0, 0, hip_z - 0.07), (0, 0, 1))], clip_blend=0.004)
        arm_parts = ["deltoid", "arm", "biceps", "triceps", "forearm_muscle"]
        for side, sgn in (("L", 1), ("R", -1)):
            mir = np.array([sgn, 1, 1])
            cuff = (wrist + (elbow - wrist) * 0.12) * mir
            c.clothe("jacket_arm.%s" % side, ["%s.%s" % (n, side) for n in arm_parts], 0.015, "top", layer="top",
                     clip=[(cuff, unit((elbow - wrist) * mir))])
        c.clothe("collar", ["neck"], 0.008, "shirt_white", layer="top", clip=[((0, 0, sh_z + 0.07), (0, 0, -1))])
        c.material("shirt_white", kind="plain", color=(235, 235, 232), roughness=0.7, bump=0.0)
    # bottoms (include the belly so the waistband covers whatever the body bulges)
    legs = ["leg.L", "leg.R", "quads.L", "quads.R", "hamstring.L", "hamstring.R"]
    if style in ("casual", "suit"):
        c.clothe("trousers", torso + ["abdomen"] + belly, 0.009, "bottom", layer="bottom", clip=waist_cut,
                 clip_blend=0.003)
        c.clothe("trouser_legs", legs + ["calf.L", "calf.R", "kneecap.L", "kneecap.R"], 0.013, "bottom",
                 layer="bottom", clip=[((0, 0, ankle_z + 0.02), (0, 0, 1))], clip_blend=0.003)
    elif style == "shorts":
        c.clothe("shorts", torso + ["abdomen"] + belly, 0.009, "bottom", layer="bottom", clip=waist_cut,
                 clip_blend=0.003)
        c.clothe("shorts_legs", legs, 0.016, "bottom", layer="bottom",
                 clip=[((0, 0, knee_z + 0.1), (0, 0, 1))], clip_blend=0.003)
    if shoes:
        c.material("shoes", kind="plain", color=shoes, roughness=0.55, bump=0.0)
        c.clothe("shoe", ["heel.L", "heel.R", "midfoot.L", "midfoot.R", "instep.L", "instep.R", "forefoot.L",
                          "forefoot.R", "big_toe.L", "big_toe.R"], 0.0065, "shoes", blend=0.02)


def _has(c, name):
    return any(pp.group == name for pp in c.parts)


def _materials(c, p, m, age):
    from ..core.colors import color as parse
    skin = p["skin_color"]
    c.material("skin", kind="skin", color=skin)
    try:
        base = parse(skin)
        lip = tuple(min(1.0, x) for x in (base[0] * 0.8, base[1] * 0.55, base[2] * 0.55))
        # back to an sRGB-ish triple in 0..255 for the material spec
        lip = tuple(int(255 * (x ** (1 / 2.2))) for x in lip)
    except Exception:
        lip = (178, 96, 90)
    c.material("lips", kind="lips", color=lip)
