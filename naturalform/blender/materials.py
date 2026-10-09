"""Procedural materials for organic bodies (no image textures, no UVs).

Patterns are driven by the ``nf_rest`` attribute (rest-pose position in
metres) so they stay glued to the body when the rig deforms it, and
countershading by ``nf_belly`` (0 on the back, 1 underneath).
"""

import bpy

from ..core.colors import color as parse_color
from ..core.colors import mix, scale

KINDS = ("skin", "fur", "wool", "scales", "feathers", "hair", "keratin", "lips", "mouth", "teeth",
         "membrane", "plain", "eye")
PATTERNS = ("none", "stripes", "spots", "rosettes", "patches", "giraffe", "mottled")

# kind -> defaults
_DEFAULTS = {
    "skin": dict(color="light", roughness=0.5, subsurface=0.25, bump=0.15, sheen=0.0),
    "fur": dict(color="fawn", roughness=0.85, subsurface=0.0, bump=0.35, sheen=0.6),
    "wool": dict(color="cream", roughness=0.95, subsurface=0.0, bump=1.0, sheen=0.8, scale_size=0.015),
    "scales": dict(color="olive_green", roughness=0.45, subsurface=0.0, bump=0.6, sheen=0.0,
                   scale_size=0.01),
    "feathers": dict(color="tawny", roughness=0.7, subsurface=0.0, bump=0.25, sheen=0.4),
    "hair": dict(color="brown_hair", roughness=0.45, subsurface=0.0, bump=0.4, sheen=0.4),
    "keratin": dict(color="horn", roughness=0.35, subsurface=0.05, bump=0.1, sheen=0.0),
    "lips": dict(color=(178, 96, 90), roughness=0.35, subsurface=0.3, bump=0.05, sheen=0.0),
    "mouth": dict(color=(120, 40, 44), roughness=0.3, subsurface=0.3, bump=0.0, sheen=0.0),
    "teeth": dict(color="ivory", roughness=0.25, subsurface=0.1, bump=0.0, sheen=0.0),
    "membrane": dict(color=(140, 100, 90), roughness=0.5, subsurface=0.5, bump=0.1, sheen=0.0),
    "plain": dict(color="gray", roughness=0.5, subsurface=0.0, bump=0.0, sheen=0.0),
}


def _sock(sockets, *names):
    for n in names:
        for s in sockets:
            if s.identifier == n:
                return s
        s = sockets.get(n)
        if s is not None:
            return s
    return None


def _set(node, value, *names):
    s = _sock(node.inputs, *names)
    if s is None:
        return False
    try:
        s.default_value = value
    except (TypeError, ValueError):
        try:
            s.default_value = tuple(value) + (1.0,)
        except Exception:
            return False
    return True


class _G:
    """Tiny node-graph helper."""

    def __init__(self, mat):
        if mat.node_tree is None:
            mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = -1400

    def node(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, len(self.nt.nodes) * -40)
        self.x += 120
        return n

    def link(self, out, inp):
        self.nt.links.new(out, inp)

    def attr(self, name):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        try:
            n.attribute_type = "GEOMETRY"
        except Exception:
            pass
        return n

    def value(self, v):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = v
        return n.outputs[0]

    def math(self, op, a, b=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if hasattr(v, "is_output"):
                self.link(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    def smooth(self, x, lo, hi):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.link(x, _sock(n.inputs, "Value"))
        _sock(n.inputs, "From Min").default_value = lo
        _sock(n.inputs, "From Max").default_value = hi
        return _sock(n.outputs, "Result")

    def mix(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="RGBA")
        f = _sock(n.inputs, "Factor_Float")
        if hasattr(fac, "is_output"):
            self.link(fac, f)
        else:
            f.default_value = fac
        for ident, v in (("A_Color", a), ("B_Color", b)):
            s = _sock(n.inputs, ident)
            if hasattr(v, "is_output"):
                self.link(v, s)
            else:
                s.default_value = tuple(v)[:3] + (1.0,)
        return _sock(n.outputs, "Result_Color")


def _coords(g, size):
    """Rest-position coordinates scaled so one unit ~ one pattern feature."""
    rest = g.attr("nf_rest")
    m = g.node("ShaderNodeMapping")
    s = 1.0 / max(size, 1e-4)
    m.inputs["Scale"].default_value = (s, s, s)
    g.link(rest.outputs["Vector"], m.inputs["Vector"])
    return m.outputs["Vector"]


def _pattern_mask(g, pattern, size, seed=0.0):
    co = _coords(g, size)
    if pattern == "stripes":
        # bands around the body (perpendicular to the spine, which runs along Y)
        w = g.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y")
        _set(w, 1.0, "Scale")
        _set(w, 6.0, "Distortion")
        _set(w, 1.5, "Detail")
        _set(w, 1.5, "Detail Scale")
        _set(w, seed, "Phase Offset")
        g.link(co, w.inputs["Vector"])
        return g.smooth(w.outputs["Fac"], 0.62, 0.7)
    if pattern in ("spots", "rosettes"):
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, 2.0, "Scale")
        g.link(co, nz.inputs["Vector"])
        warp = g.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        g.link(nz.outputs["Color"], warp.inputs[0])
        warp.inputs[1].default_value = (0.35, 0.35, 0.35)
        g.link(co, warp.inputs[2])
        v = g.node("ShaderNodeTexVoronoi", feature="F1")
        _set(v, 1.0, "Scale")
        _set(v, 0.9, "Randomness")
        g.link(warp.outputs["Vector"], v.inputs["Vector"])
        d = v.outputs["Distance"]
        if pattern == "spots":
            return g.math("SUBTRACT", 1.0, g.smooth(d, 0.22, 0.3))
        ring = g.math("MULTIPLY", g.smooth(d, 0.16, 0.22), g.math("SUBTRACT", 1.0, g.smooth(d, 0.34, 0.4)))
        return ring
    if pattern == "giraffe":
        v = g.node("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        _set(v, 1.0, "Scale")
        _set(v, 0.85, "Randomness")
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, 3.0, "Scale")
        g.link(co, nz.inputs["Vector"])
        warp = g.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        g.link(nz.outputs["Color"], warp.inputs[0])
        warp.inputs[1].default_value = (0.12, 0.12, 0.12)
        g.link(co, warp.inputs[2])
        g.link(warp.outputs["Vector"], v.inputs["Vector"])
        return g.smooth(v.outputs["Distance"], 0.05, 0.08)
    if pattern == "patches":
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, 0.6, "Scale")
        _set(nz, 3.0, "Detail")
        g.link(co, nz.inputs["Vector"])
        return g.smooth(nz.outputs["Fac"], 0.54, 0.58)
    if pattern == "mottled":
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, 1.5, "Scale")
        _set(nz, 6.0, "Detail")
        g.link(co, nz.inputs["Vector"])
        return g.smooth(nz.outputs["Fac"], 0.45, 0.65)
    return None


def _principled(g):
    out = g.node("ShaderNodeOutputMaterial")
    b = g.node("ShaderNodeBsdfPrincipled")
    g.link(b.outputs[0], out.inputs["Surface"])
    out.location = (400, 0)
    b.location = (100, 0)
    return b


def _bump(g, bsdf, kind, strength, size):
    if strength <= 0:
        return
    rest = g.attr("nf_rest")
    if kind == "scales":
        v = g.node("ShaderNodeTexVoronoi", feature="F1")
        _set(v, 1.0 / max(size, 1e-4), "Scale")
        g.link(rest.outputs["Vector"], v.inputs["Vector"])
        h = v.outputs["Distance"]
        dist = size * 0.4
    elif kind == "wool":
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, 1.0 / max(size, 1e-4), "Scale")
        _set(nz, 6.0, "Detail")
        _set(nz, 0.7, "Roughness")
        g.link(rest.outputs["Vector"], nz.inputs["Vector"])
        h = nz.outputs["Fac"]
        dist = size * 0.5
    else:
        nz = g.node("ShaderNodeTexNoise")
        _set(nz, {"skin": 900.0, "fur": 260.0, "hair": 300.0, "feathers": 120.0}.get(kind, 200.0), "Scale")
        _set(nz, 4.0, "Detail")
        g.link(rest.outputs["Vector"], nz.inputs["Vector"])
        if kind == "hair":
            # strands: stretch the noise along Z
            m = g.node("ShaderNodeMapping")
            m.inputs["Scale"].default_value = (1.0, 1.0, 0.05)
            g.link(rest.outputs["Vector"], m.inputs["Vector"])
            g.link(m.outputs["Vector"], nz.inputs["Vector"])
        h = nz.outputs["Fac"]
        dist = 0.0015
    bp = g.node("ShaderNodeBump")
    _set(bp, float(strength), "Strength")
    _set(bp, dist, "Distance")
    g.link(h, _sock(bp.inputs, "Height"))
    g.link(bp.outputs["Normal"], _sock(bsdf.inputs, "Normal"))


def organic_material(name, spec):
    """Create (or rebuild) a material from a spec dict."""
    kind = spec.get("kind", "skin")
    if kind not in _DEFAULTS:
        raise ValueError("unknown material kind %r (choose from %s)" % (kind, ", ".join(_DEFAULTS)))
    opts = dict(_DEFAULTS[kind])
    opts.update({k: v for k, v in spec.items() if v is not None})
    ctx = "hair" if kind == "hair" else None
    base = parse_color(opts["color"], ctx)

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    g = _G(mat)
    bsdf = _principled(g)
    mat.diffuse_color = tuple(base) + (1.0,)

    # subtle value variation so large areas are not flat
    nz = g.node("ShaderNodeTexNoise")
    _set(nz, 6.0, "Scale")
    _set(nz, 3.0, "Detail")
    g.link(g.attr("nf_rest").outputs["Vector"], nz.inputs["Vector"])
    var = g.smooth(nz.outputs["Fac"], 0.3, 0.7)
    col_sock = g.mix(var, scale(base, 0.88), scale(base, 1.1))

    belly = opts.get("belly_color")
    if belly is not None:
        bc = parse_color(belly)
        b_attr = g.attr("nf_belly")
        fac = g.smooth(b_attr.outputs["Fac"], 0.25, 0.75)
        col_sock = g.mix(fac, col_sock, bc)

    pattern = opts.get("pattern", "none") or "none"
    if pattern not in PATTERNS:
        raise ValueError("unknown pattern %r (choose from %s)" % (pattern, ", ".join(PATTERNS)))
    if pattern != "none":
        pc = parse_color(opts.get("pattern_color", "black"))
        size = float(opts.get("pattern_scale", 0.1))
        mask = _pattern_mask(g, pattern, size, float(opts.get("seed", 0.0)))
        if mask is not None:
            if belly is not None and opts.get("pattern_on_belly", False) is False and pattern in ("stripes", "spots", "rosettes"):
                b_attr = g.attr("nf_belly")
                keep = g.math("SUBTRACT", 1.0, g.smooth(b_attr.outputs["Fac"], 0.55, 0.9))
                mask = g.math("MULTIPLY", mask, keep)
            col_sock = g.mix(mask, col_sock, pc)

    g.link(col_sock, _sock(bsdf.inputs, "Base Color"))
    _set(bsdf, float(opts["roughness"]), "Roughness")
    ss = float(opts.get("subsurface", 0.0))
    if ss > 0:
        _set(bsdf, ss, "Subsurface Weight", "Subsurface")
        _set(bsdf, (1.0, 0.35, 0.2), "Subsurface Radius")
        _set(bsdf, float(opts.get("subsurface_scale", 0.01)), "Subsurface Scale")
    sh = float(opts.get("sheen", 0.0))
    if sh > 0:
        _set(bsdf, sh, "Sheen Weight", "Sheen")
        _set(bsdf, 0.5, "Sheen Roughness")
    if kind in ("keratin", "teeth"):
        _set(bsdf, 0.3, "Coat Weight", "Clearcoat")
    if kind == "mouth":
        _set(bsdf, 0.5, "Coat Weight", "Clearcoat")
    _bump(g, bsdf, kind, float(opts.get("bump", 0.0)), float(opts.get("scale_size", 0.01)))
    mat["naturalform"] = kind
    return mat


def eye_material(name, iris="eye_brown", pupil="round", iris_size=0.55, pupil_size=0.45,
                 sclera="white"):
    """Glossy eyeball using the ``nf_eye`` attribute (unit sphere coords,
    +Z = gaze). ``iris_size`` = iris radius on the unit sphere, ``pupil_size``
    = pupil radius as a fraction of the iris."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    g = _G(mat)
    bsdf = _principled(g)
    ic = parse_color(iris, "eye")
    sc = parse_color(sclera) if sclera else scale(ic, 0.7)
    mat.diffuse_color = tuple(sc) + (1.0,)
    e = g.attr("nf_eye")
    sep = g.node("ShaderNodeSeparateXYZ")
    g.link(e.outputs["Vector"], sep.inputs["Vector"])
    x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]
    front = g.smooth(z, 0.0, 0.2)

    def radial(kx, ky):
        a = g.math("MULTIPLY", x, kx)
        b = g.math("MULTIPLY", y, ky)
        return g.math("SQRT", g.math("ADD", g.math("MULTIPLY", a, a), g.math("MULTIPLY", b, b)))

    r = radial(1.0, 1.0)
    iris_r = float(iris_size)
    iris_mask = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.smooth(r, iris_r * 0.94, iris_r)), front)
    if pupil == "slit":
        rp = radial(4.5, 1.0)
    elif pupil == "horizontal":
        rp = radial(1.0, 3.0)
    else:
        rp = r
    pr = iris_r * float(pupil_size)
    pupil_mask = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.smooth(rp, pr * 0.85, pr)), front)
    if pupil == "none":
        pupil_mask = g.value(0.0)
    # iris: darker rim, lighter around the pupil
    ring = g.smooth(r, iris_r * 0.4, iris_r)
    iris_col = g.mix(ring, mix(ic, (1, 1, 1), 0.12), scale(ic, 0.45))
    col = g.mix(iris_mask, sc, iris_col)
    col = g.mix(pupil_mask, col, (0.004, 0.004, 0.004))
    g.link(col, _sock(bsdf.inputs, "Base Color"))
    _set(bsdf, 0.3, "Roughness")
    _set(bsdf, 1.0, "Coat Weight", "Clearcoat")
    _set(bsdf, 0.02, "Coat Roughness", "Clearcoat Roughness")
    _set(bsdf, 0.1, "Subsurface Weight", "Subsurface")
    mat["naturalform"] = "eye"
    return mat
