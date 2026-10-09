"""Snakes and other limbless, elongated animals (worms, eels).

The body follows a path: straight, wave (S-curves on the ground), coil
(resting spiral with the head up) or raised (front third lifted, e.g. a
cobra). Faces -Y, lies on the ground.
"""

import math

import numpy as np

from ..core.creature import Creature, NaturalFormError
from .common import choice, resolve, unit

SHAPES = ("straight", "wave", "coil", "raised")
HEADS = ("round", "viper", "none")

PARAMS = {
    "length": (None, "total length in metres (default: species size * size)"),
    "size": (1.0, "multiplier on the species' typical size"),
    "thickness": (0.035, "max body radius / length"),
    "shape": ("wave", "straight | wave | coil | raised"),
    "waves": (1.5, "number of S-bends for shape=wave"),
    "head": ("round", "round | viper (triangular) | none (worm)"),
    "head_size": (1.0, "multiplier"),
    "hood": (False, "cobra hood"),
    "rattle": (False, "rattlesnake rattle"),
    "tongue": (False, "forked tongue sticking out"),
    "segments": (False, "segmented look (worms)"),
    "eye_size": (1.0, "multiplier"),
    "pupil": ("round", "round | slit"),
    "material": ("scales", "scales | skin"),
    "color": ((80, 110, 50), "main colour"),
    "belly_color": ("cream", "underside colour"),
    "pattern": ("none", "none | stripes | spots | patches | rosettes | mottled"),
    "pattern_color": ("black", "pattern colour"),
    "pattern_scale": (None, "feature size in metres"),
    "eye_color": ("eye_amber", "iris colour"),
}

PRESETS = {
    "snake": dict(_size=1.2, color=(80, 110, 50), belly_color=(200, 200, 150)),
    "python": dict(_size=3.5, thickness=0.032, color=(150, 120, 70), belly_color="cream", pattern="patches",
                   pattern_color=(60, 45, 30), pattern_scale=0.12, shape="coil"),
    "cobra": dict(_size=1.8, thickness=0.028, shape="raised", hood=True, color=(60, 50, 40),
                  belly_color=(190, 170, 120), tongue=True),
    "rattlesnake": dict(_size=1.2, thickness=0.04, head="viper", rattle=True, shape="coil", pupil="slit",
                        color=(160, 140, 100), pattern="rosettes", pattern_color=(80, 60, 40), pattern_scale=0.05),
    "viper": dict(_size=0.7, thickness=0.045, head="viper", pupil="slit", color=(120, 110, 80),
                  pattern="rosettes", pattern_color=(50, 40, 30), pattern_scale=0.03),
    "green_mamba": dict(_size=2.2, thickness=0.018, color=(90, 170, 60), belly_color=(170, 210, 120), waves=2.5),
    "worm": dict(_size=0.15, thickness=0.035, head="none", segments=True, material="skin", color=(200, 120, 120),
                 belly_color=None, waves=1.0, eye_size=0.0),
    "eel": dict(_size=1.0, thickness=0.03, color=(70, 70, 50), belly_color=(170, 160, 120), shape="wave",
                waves=1.0, material="skin"),
}


def _path(shape, n, waves):
    """Points along the body (head first), total length ~1, on the ground."""
    t = np.linspace(0.0, 1.0, n)
    if shape == "straight":
        return np.stack([np.zeros(n), t - 0.5, np.zeros(n)], axis=1)
    if shape == "wave":
        amp = 0.09
        y = t * 0.85 - 0.45
        x = amp * np.sin(t * waves * 2 * math.pi) * np.minimum(1.0, t * 6)
        return _resample(np.stack([x, y, np.zeros(n)], axis=1), n)
    if shape == "coil":
        pts = [[0.0, -0.09, 0.2], [0.0, -0.055, 0.2]]  # head held level, looking forward
        for k in range(n * 3):
            u = k / (n * 3 - 1)
            if u < 0.2:  # neck rises out of the middle of the pile
                s_ = u / 0.2
                pts.append([0.0, -0.03 + 0.04 * s_, 0.19 * (1 - s_) ** 1.2 + 0.07 * s_])
            else:  # spiral outwards; inner turns rest on top of outer ones
                s_ = (u - 0.2) / 0.8
                ang = s_ * 2.2 * 2 * math.pi
                r = 0.035 + 0.1 * s_
                pts.append([r * math.sin(ang), 0.045 - r * math.cos(ang), 0.07 * (1 - s_) ** 1.5])
        return _resample(np.array(pts), n)
    if shape == "raised":
        pts = [[0.0, -0.24, 0.27], [0.0, -0.2, 0.27]]
        for k in range(n * 3):
            u = k / (n * 3 - 1)
            if u < 0.32:
                s_ = u / 0.32
                pts.append([0.0, -0.17 + 0.11 * s_, 0.26 * (1 - s_) ** 0.9])
            else:
                s_ = (u - 0.32) / 0.68
                ang = s_ * 1.4 * 2 * math.pi
                r = 0.08 + 0.06 * s_
                pts.append([r * math.sin(ang), -0.06 + r * (1 - math.cos(ang)), 0.0])
        return _resample(np.array(pts), n)
    raise NaturalFormError("unknown shape %r" % shape)


def _resample(P, n):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def serpent(species="snake", **kwargs):
    """Return a :class:`Creature` for a snake / worm / eel (see PRESETS / PARAMS)."""
    if species not in PRESETS:
        import difflib
        close = difflib.get_close_matches(str(species), PRESETS, n=3)
        raise NaturalFormError("unknown serpent species %r.%s Presets: %s" % (
            species, (" Did you mean %s?" % ", ".join(close)) if close else "", ", ".join(PRESETS)))
    preset = dict(PRESETS[species])
    base_size = preset.pop("_size", 1.0)
    p = resolve(PARAMS, kwargs, "serpent(%s)" % species, preset)
    shape = choice(p["shape"], SHAPES, "shape")
    choice(p["head"], HEADS, "head")
    c = Creature(species.replace("_", " ").title().replace(" ", ""), "serpent")
    c.default_material = "body"
    c.meta["ground"] = True
    n = 17
    P = _path(shape, n, float(p["waves"]))
    # normalise to unit arc length
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    P = P / L
    th = float(p["thickness"])
    u = np.linspace(0, 1, n)
    prof = np.interp(u, [0, 0.04, 0.1, 0.4, 0.75, 1.0], [0.55, 0.6, 0.75, 1.0, 0.7, 0.08])
    if p["head"] == "none":
        prof = np.interp(u, [0, 0.05, 0.3, 0.8, 1.0], [0.6, 0.85, 1.0, 0.9, 0.5])
    radii = prof * th
    # lift every point by its (squashed) radius so the body rests on the ground
    P = P + np.stack([np.zeros(n), np.zeros(n), radii * 0.82], axis=1)
    bones = ["spine_%d" % i for i in range(n - 1)]
    c.chain("body", list(P), list(radii), bones=bones, blend=0.0, joint_blend=th * 0.3,
            squash=0.82, squash_dir=(0, 0, 1))
    if p["segments"]:
        for k in range(1, n - 1, 1):
            c.torus("ring%d" % k, P[k], radii[k] * 0.98, radii[k] * 0.12, align=None,
                    rotate=_frame_z(P[k + 1] - P[k - 1]), bone=bones[min(k, n - 2)], blend=th * 0.1)
    # head
    hs = float(p["head_size"])
    fwd = unit(P[0] - P[1])
    side = unit(np.cross(fwd, [0, 0, 1])) if abs(fwd[2]) < 0.95 else np.array([1.0, 0, 0])
    if side[0] < 0:
        side = -side
    up = unit(np.cross(side, fwd))
    HF = np.column_stack([side, fwd, up])
    hc = P[0] + fwd * th * 0.8 * hs
    if p["head"] != "none":
        hw = th * (1.15 if p["head"] == "round" else 1.55) * hs
        c.bone("head", P[0], P[0] + fwd * th * 2.5 * hs, parent=bones[0])
        c.ellipsoid("head", hc, (hw, th * 1.6 * hs, th * 0.62 * hs), rotate=HF, bone="head", blend=th * 0.5)
        c.ellipsoid("snout", hc + fwd * th * 1.3 * hs, (hw * 0.7, th * 0.8 * hs, th * 0.5 * hs), rotate=HF,
                    bone="head", blend=th * 0.4)
        c.ellipsoid("mouth", hc + fwd * th * 1.1 * hs - up * th * 0.25 * hs, (hw * 1.05, th * 1.4 * hs, th * 0.04),
                    rotate=HF, bone="head", blend=th * 0.05, op="subtract")
        if float(p["eye_size"]) > 0:
            er = th * 0.26 * float(p["eye_size"]) * hs
            look = unit(side * 0.8 + fwd * 0.5 + up * 0.2)
            hit = c.surface_point(hc + fwd * th * 0.6 + up * th * 0.2, look)
            ec = (np.array(hit) if hit is not None else hc + side * hw) - look * er * 0.4
            c.eye("eye.L", ec, er, look=look, bone="head", color=p["eye_color"], pupil=p["pupil"],
                  iris_size=0.8, pupil_size=0.5, sclera=None, socket_blend=er * 0.3, mirror=False)
            c.eye("eye.R", ec - 2 * side * np.dot(ec - hc, side), er, look=look - 2 * side * np.dot(look, side),
                  bone="head", color=p["eye_color"], pupil=p["pupil"], iris_size=0.8, pupil_size=0.5,
                  sclera=None, socket_blend=er * 0.3, mirror=False)
        if p["tongue"]:
            t0 = hc + fwd * th * 2.0 * hs - up * th * 0.25
            t1 = t0 + fwd * th * 2.0
            c.capsule("tongue", t0, t1, th * 0.08, bone="head", blend=th * 0.02, material="tongue")
            for sgn in (1, -1):
                c.capsule("tongue_fork%d" % (sgn > 0), t1, t1 + (fwd + side * 0.5 * sgn) * th * 0.7, th * 0.07,
                          end_radius=th * 0.03, bone="head", blend=th * 0.02, material="tongue")
            c.material("tongue", kind="skin", color=(170, 30, 50), roughness=0.3)
    if p["hood"]:
        k = 2
        c.ellipsoid("hood", P[k] + fwd * th * 0.3, (th * 2.4, th * 0.4, th * 2.8),
                    rotate=_frame_z(P[k - 1] - P[k + 1]), bone=bones[k - 1], blend=th * 0.6)
    if p["rattle"]:
        tail = P[-1]
        d = unit(P[-1] - P[-2])
        for k in range(4):
            c.ellipsoid("rattle%d" % k, tail + d * th * (0.3 + k * 0.45), (th * 0.38, th * 0.3, th * 0.3),
                        rotate=_frame_z(d), bone=bones[-1], blend=th * 0.08, material="rattle")
        c.material("rattle", kind="keratin", color=(170, 150, 120))
    _materials(c, p, base_size)
    c.poses["rest"] = {}
    length = float(p["length"]) if p["length"] else base_size * float(p["size"])
    c.transform(scale=length)
    c.snap_to_ground()
    c.meta["min_feature"] = th * length * 0.5
    c.meta["detail_regions"] = [] if p["head"] == "none" else [
        {"bone": "head", "at": 0.3, "radius": th * length * 4.0 * hs, "factor": 2.0, "layers": ["body"]}]
    c.meta["params"] = dict(p, species=species)
    c.meta["user_params"] = dict(kwargs)
    c.meta["template"] = "serpent"
    return c


def _frame_z(d):
    """Frame whose local Z axis follows d (for tori / rings around the body)."""
    z = unit(d)
    x = unit(np.cross([0.0, 0.0, 1.0], z)) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _materials(c, p, base_size):
    size = float(p["length"]) if p["length"] else base_size * float(p["size"])
    th = float(p["thickness"]) * size
    c.material("body", kind=p["material"], color=p["color"], belly_color=p["belly_color"], pattern=p["pattern"],
               pattern_color=p["pattern_color"], pattern_scale=p["pattern_scale"] or th * 3,
               scale_size=th * 0.35, roughness=0.4, subsurface=0.05 if p["material"] == "skin" else 0.0,
               subsurface_scale=th * 0.2)
