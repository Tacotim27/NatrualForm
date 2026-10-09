"""Creature: a declarative, JSON-serialisable description of an organic body.

A creature is an ordered list of *parts* (SDF primitives combined with
smooth unions / subtractions), a hierarchy of *bones* (which become the
Blender armature) and a list of *eyes*. Templates (human, quadruped, bird,
...) produce a ``Creature`` and an LLM can also build one from scratch:

    c = Creature("Snail")
    c.chain("body", [(0, -0.1, 0.02), (0, 0.05, 0.02), (0, 0.1, 0.03)],
            radii=[0.02, 0.025, 0.012], squash=0.6, squash_dir=(0, 0, 1))
    c.ellipsoid("shell", (0, 0.03, 0.06), (0.035, 0.045, 0.045), blend=0.004,
                material="shell")
    c.material("shell", kind="keratin", color="chestnut")
    c.build()

World conventions: metres, +Z up, creatures face -Y (Blender "front"
view looks at their face), their left side is +X. Names ending in ``.L`` /
``.R`` are mirrored pairs.
"""

import copy
import difflib
import json
import math

import numpy as np

from .sdf import (MIRROR, Clipped, Ellipsoid, FieldPart, RoundBox, RoundCone, Torus,
                  axis_angle_matrix, euler_matrix, frame_from_axis, normalize,
                  primitive_from_dict, raymarch, rotation_between,
                  transform_point, vec)
from .sdf import _r

_SIDE_FLIP = {".L": ".R", ".R": ".L", "_L": "_R", "_R": "_L", ".l": ".r", ".r": ".l"}


def split_side(name):
    for suf in _SIDE_FLIP:
        if name.endswith(suf):
            return name[:-2], name[-2:]
    return name, ""


def flip_name(name):
    base, side = split_side(name)
    return base + _SIDE_FLIP[side] if side else name


def socket_name(eye_name):
    base, side = split_side(eye_name)
    return base + "_socket" + side


def _mirror_names(name):
    base, side = split_side(name)
    if side:
        return name, flip_name(name)
    return name + ".L", name + ".R"


def _rotation(rotate=None, align=None):
    """``rotate``: XYZ world euler degrees or a 3x3 matrix (columns = local
    axes). ``align``: direction for the local +Y axis (applied first)."""
    R = np.eye(3)
    if align is not None:
        R = frame_from_axis(vec(align))
    if rotate is not None:
        r = np.asarray(rotate, dtype=np.float64)
        R = (r if r.shape == (3, 3) else euler_matrix(r)) @ R
    return R


def _suggest(name, options):
    m = difflib.get_close_matches(str(name), list(options), n=3)
    return (" Did you mean %s?" % ", ".join(repr(x) for x in m)) if m else ""


class NaturalFormError(ValueError):
    """Raised for invalid input; the message always says how to fix it."""


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

class Bone:
    __slots__ = ("name", "head", "tail", "parent", "roll", "connect", "deform")

    def __init__(self, name, head, tail, parent=None, roll=None, connect=False, deform=True):
        self.name = name
        self.head = vec(head)
        self.tail = vec(tail)
        if np.linalg.norm(self.tail - self.head) < 1e-6:
            self.tail = self.head + np.array([0.0, 0.0, 1e-3])
        self.parent = parent
        self.roll = None if roll is None else normalize(vec(roll))
        self.connect = bool(connect)
        self.deform = bool(deform)

    @property
    def length(self):
        return float(np.linalg.norm(self.tail - self.head))

    def transformed(self, M):
        b = copy.copy(self)
        b.head = transform_point(M, self.head)
        b.tail = transform_point(M, self.tail)
        if self.roll is not None:
            b.roll = M[:3, :3] @ self.roll
        return b

    def mirrored(self, name, parent):
        return Bone(name, MIRROR @ self.head, MIRROR @ self.tail, parent,
                    None if self.roll is None else MIRROR @ self.roll, self.connect, self.deform)

    def distance_to(self, p):
        a, b = self.head, self.tail
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + ab * t)))

    def to_dict(self):
        d = {"name": self.name, "head": _r(self.head), "tail": _r(self.tail), "parent": self.parent}
        if self.roll is not None:
            d["roll"] = _r(self.roll)
        if self.connect:
            d["connect"] = True
        if not self.deform:
            d["deform"] = False
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["head"], d["tail"], d.get("parent"), d.get("roll"),
                   d.get("connect", False), d.get("deform", True))


class Part:
    __slots__ = ("name", "prim", "op", "blend", "bone", "material", "group", "auto_bone", "layer")

    def __init__(self, name, prim, op="union", blend=0.0, bone=None, material=None, group=None,
                 layer=None):
        if op not in ("union", "subtract", "intersect"):
            raise NaturalFormError("op must be 'union', 'subtract' or 'intersect', not %r" % op)
        self.name = name
        self.prim = prim
        self.op = op
        self.blend = float(blend)
        self.bone = bone
        self.material = material
        self.group = group or name
        self.auto_bone = None
        self.layer = layer or "body"

    @property
    def rig_bone(self):
        return self.bone or self.auto_bone

    def copy_with(self, **kw):
        p = copy.copy(self)
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def to_dict(self):
        d = {"name": self.name, "op": self.op, "blend": _r(self.blend)}
        d.update(self.prim.to_dict())
        if self.bone:
            d["bone"] = self.bone
        if self.material:
            d["material"] = self.material
        if self.group != self.name:
            d["group"] = self.group
        if self.layer != "body":
            d["layer"] = self.layer
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], primitive_from_dict(d), d.get("op", "union"), d.get("blend", 0.0),
                   d.get("bone"), d.get("material"), d.get("group"), d.get("layer"))


class Eye:
    __slots__ = ("name", "center", "radius", "look", "bone", "color", "pupil", "iris_size",
                 "pupil_size", "sclera")

    def __init__(self, name, center, radius, look=(0, -1, 0), bone=None, color="eye_brown",
                 pupil="round", iris_size=0.55, pupil_size=0.45, sclera="white"):
        if pupil not in ("round", "slit", "horizontal", "none"):
            raise NaturalFormError("pupil must be 'round', 'slit', 'horizontal' or 'none'")
        self.name = name
        self.center = vec(center)
        self.radius = float(radius)
        self.look = normalize(vec(look))
        self.bone = bone
        self.color = color
        self.pupil = pupil
        self.iris_size = float(iris_size)
        self.pupil_size = float(pupil_size)
        self.sclera = sclera

    def transformed(self, M):
        e = copy.copy(self)
        e.center = transform_point(M, self.center)
        e.look = M[:3, :3] @ self.look
        return e

    def mirrored(self, name, bone):
        e = copy.copy(self)
        e.name = name
        e.center = MIRROR @ self.center
        e.look = MIRROR @ self.look
        e.bone = bone
        return e

    def to_dict(self):
        return {"name": self.name, "center": _r(self.center), "radius": _r(self.radius),
                "look": _r(self.look), "bone": self.bone, "color": self.color, "pupil": self.pupil,
                "iris_size": self.iris_size, "pupil_size": self.pupil_size, "sclera": self.sclera}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["center"], d["radius"], d.get("look", (0, -1, 0)), d.get("bone"),
                   d.get("color", "eye_brown"), d.get("pupil", "round"), d.get("iris_size", 0.55),
                   d.get("pupil_size", 0.45), d.get("sclera", "white"))


# ---------------------------------------------------------------------------
# the creature
# ---------------------------------------------------------------------------

class Creature:
    """Editable description of an organic body. See module docstring."""

    def __init__(self, name="Creature", kind="custom"):
        self.name = name
        self.kind = kind
        self.parts = []
        self.bones = {}
        self.eyes = []
        self.materials = {}
        self.default_material = "body"
        self.poses = {}
        self.meta = {}

    # ----------------------------------------------------------- utilities
    def __repr__(self):
        lo, hi = self.bounds() if self.parts else (np.zeros(3), np.zeros(3))
        return ("<Creature %r kind=%s parts=%d bones=%d eyes=%d size=%s>"
                % (self.name, self.kind, len(self.parts), len(self.bones), len(self.eyes),
                   _r(hi - lo, 3)))

    def copy(self):
        return copy.deepcopy(self)

    def _new_name(self, name):
        if any(p.name == name for p in self.parts) or any(e.name == name for e in self.eyes):
            raise NaturalFormError("a part named %r already exists - pick another name or "
                                   "remove it first with .remove(%r)" % (name, name))
        return name

    def resolve_bone(self, ref):
        """Accept a bone name or a part/chain name and return a bone name."""
        if ref is None:
            return None
        if ref in self.bones:
            return ref
        for p in self.parts:
            if p.name == ref or p.group == ref:
                b = p.rig_bone or self._nearest_bone(p.prim.center())
                if b:
                    return b
        raise NaturalFormError("unknown bone or part %r.%s Bones: %s" % (
            ref, _suggest(ref, list(self.bones) + [p.group for p in self.parts]),
            ", ".join(list(self.bones)[:40])))

    def _nearest_bone(self, p, exclude=()):
        best, bd = None, 1e18
        for b in self.bones.values():
            if b.name in exclude or not b.deform:
                continue
            d = b.distance_to(p)
            if d < bd:
                best, bd = b.name, d
        return best

    def _add_part(self, part, mirror):
        if not mirror:
            self.parts.append(part)
            return
        self.parts.append(part)
        mp = part.copy_with(name=flip_name(part.name), prim=part.prim.mirrored(),
                            bone=flip_name(part.bone) if part.bone else None,
                            group=flip_name(part.group))
        mp.auto_bone = None
        self.parts.append(mp)

    def _check_parent_mirror(self, parent):
        if parent and split_side(parent)[1] and flip_name(parent) not in self.bones:
            raise NaturalFormError("mirrored parent %r has no %r counterpart" % (parent, flip_name(parent)))

    # ----------------------------------------------------------- bones
    def bone(self, name, head, tail, parent=None, roll=None, connect=False, deform=True, mirror=False):
        """Add a bone without geometry (e.g. a root or a control bone)."""
        parent = self.resolve_bone(parent)
        names = _mirror_names(name) if mirror else (name,)
        if names[0] in self.bones:
            raise NaturalFormError("bone %r already exists" % names[0])
        b = Bone(names[0], head, tail, parent, roll, connect, deform)
        self.bones[b.name] = b
        if mirror:
            self._check_parent_mirror(parent)
            mb = b.mirrored(names[1], flip_name(parent) if parent else None)
            self.bones[mb.name] = mb
        return self

    # ----------------------------------------------------------- shapes
    def chain(self, name, points, radii, parent=None, bones=None, blend=None, joint_blend=None,
              material=None, squash=1.0, squash_dir=(0, 1, 0), roll=None, op="union",
              visible=True, make_bones=True, mirror=False, layer=None):
        """A limb / spine / tail / neck: tapered capsules through ``points``.

        points      list of 2+ (x, y, z) joint positions
        radii       one radius per point (or a single number)
        parent      bone (or part) the first bone hangs from
        bones       names for the segment bones (default "<name>_<i>")
        blend       smooth-union radius where the chain meets the body
        joint_blend smoothing inside the chain (default 20% of the radius)
        squash      cross-section scale along ``squash_dir`` (0.5 = flat oval)
        visible     False -> bones only (e.g. a spine inside a torso)
        make_bones  False -> geometry only
        mirror      also create the opposite side (.L <-> .R)
        layer       mesh this as a separate surface layered over the body
                    (hair, clothing, shells...). Parts only blend / carve
                    within their own layer.
        """
        pts = [vec(p) for p in points]
        if len(pts) < 2:
            raise NaturalFormError("chain %r needs at least 2 points" % name)
        if np.isscalar(radii):
            radii = [float(radii)] * len(pts)
        radii = [float(r) for r in radii]
        if len(radii) != len(pts):
            raise NaturalFormError("chain %r: %d points but %d radii" % (name, len(pts), len(radii)))
        nseg = len(pts) - 1
        names = _mirror_names(name) if mirror else (name,)
        base, side = split_side(names[0])
        if bones is None:
            bone_names = [names[0]] if nseg == 1 else ["%s_%d%s" % (base, i, side) for i in range(nseg)]
        else:
            bone_names = list(bones)
            if len(bone_names) != nseg:
                raise NaturalFormError("chain %r has %d segments but %d bone names" % (name, nseg, len(bone_names)))
            if mirror:
                bone_names = [b if split_side(b)[1] else b + ".L" for b in bone_names]
        parent = self.resolve_bone(parent) if make_bones else parent
        if make_bones:
            for b in bone_names:
                if b in self.bones:
                    raise NaturalFormError("bone %r already exists (chain %r)" % (b, name))
        if blend is None:
            blend = 0.3 * radii[0]
        self._new_name(names[0])
        new_bones = []
        if make_bones:
            prev = parent
            for i in range(nseg):
                b = Bone(bone_names[i], pts[i], pts[i + 1], prev, roll, connect=i > 0)
                self.bones[b.name] = b
                new_bones.append(b)
                prev = b.name
        if visible:
            for i in range(nseg):
                ra, rb = radii[i], radii[i + 1]
                k = blend if i == 0 else (joint_blend if joint_blend is not None else 0.2 * min(ra, rb))
                prim = RoundCone(pts[i], pts[i + 1], ra, rb, squash, squash_dir)
                bone = bone_names[i] if make_bones else (self.resolve_bone(parent) if parent else None)
                self.parts.append(Part("%s[%d]" % (names[0], i), prim, op, k, bone, material, names[0], layer))
        if mirror:
            self._check_parent_mirror(parent)
            if make_bones:
                for b in new_bones:
                    mb = b.mirrored(flip_name(b.name), flip_name(b.parent) if b.parent else None)
                    self.bones[mb.name] = mb
            if visible:
                for p in [p for p in self.parts if p.group == names[0]]:
                    self.parts.append(p.copy_with(name=flip_name(names[0]) + p.name[len(names[0]):],
                                                  prim=p.prim.mirrored(),
                                                  bone=flip_name(p.bone) if p.bone else None,
                                                  group=names[1]))
        return self

    def _shape(self, name, prim, op, blend, bone, material, mirror, clip=None, clip_blend=0.0, layer=None):
        if clip:
            prim = Clipped(prim, clip, clip_blend)
        names = _mirror_names(name) if mirror else (name,)
        self._new_name(names[0])
        bone = self.resolve_bone(bone) if bone else None
        if blend is None:
            blend = (0.1 if op == "subtract" else 0.3) * prim.scale()
        if mirror and bone and split_side(bone)[1]:
            self._check_parent_mirror(bone)
        self._add_part(Part(names[0], prim, op, blend, bone, material, layer=layer), mirror)
        return self

    def ellipsoid(self, name, center, radii, rotate=None, align=None, bone=None, blend=None,
                  material=None, op="union", mirror=False, clip=None, clip_blend=0.0,
                 layer=None):
        """Ellipsoid with semi-axes ``radii`` (x, y, z). ``align`` points the
        local Y axis (radii[1]) along a direction; ``rotate`` = world XYZ
        euler degrees applied afterwards. Every shape also accepts
        ``clip=[(point, normal), ...]`` to cut it with planes (the side the
        normal points to is kept) and ``clip_blend`` to soften the cut."""
        prim = Ellipsoid(center, radii, _rotation(rotate, align))
        return self._shape(name, prim, op, blend, bone, material, mirror, clip, clip_blend, layer)

    def sphere(self, name, center, radius, bone=None, blend=None, material=None, op="union",
               mirror=False, clip=None, clip_blend=0.0,
                 layer=None):
        return self._shape(name, Ellipsoid(center, (radius,) * 3), op, blend, bone, material, mirror, clip, clip_blend, layer)

    def capsule(self, name, start, end, radius, end_radius=None, bone=None, blend=None,
                material=None, squash=1.0, squash_dir=(0, 1, 0), op="union", mirror=False, clip=None, clip_blend=0.0,
                 layer=None):
        """Tapered capsule between two points (no bones are created)."""
        prim = RoundCone(start, end, radius, radius if end_radius is None else end_radius,
                         squash, squash_dir)
        return self._shape(name, prim, op, blend, bone, material, mirror, clip, clip_blend, layer)

    def box(self, name, center, size, rounding=None, rotate=None, align=None, bone=None,
            blend=None, material=None, op="union", mirror=False, clip=None, clip_blend=0.0,
                 layer=None):
        """Rounded box; ``size`` are half extents (x, y, z)."""
        size = vec(size)
        rounding = 0.25 * float(size.min()) if rounding is None else rounding
        prim = RoundBox(center, size, _rotation(rotate, align), rounding)
        return self._shape(name, prim, op, blend, bone, material, mirror, clip, clip_blend, layer)

    def torus(self, name, center, major_radius, minor_radius, rotate=None, align=None, bone=None,
              blend=None, material=None, op="union", mirror=False, clip=None, clip_blend=0.0,
                 layer=None):
        """Ring around the local Z axis (``align`` points local Y; use rotate to tilt)."""
        prim = Torus(center, major_radius, minor_radius, _rotation(rotate, align))
        return self._shape(name, prim, op, blend, bone, material, mirror, clip, clip_blend, layer)

    def eye(self, name, center, radius, look=(0, -1, 0), bone=None, color="brown", pupil="round",
            iris_size=0.55, pupil_size=0.45, sclera="white", socket=True, socket_blend=None,
            mirror=False):
        """Add an eyeball (a separate shiny sphere merged into the mesh).

        ``look`` is the gaze direction. ``socket`` carves a soft hole in the
        body so the eye sits in an eyelid rim - add eyes *after* the head
        parts. ``pupil``: round, slit (cats), horizontal (goats), none.
        ``sclera``: colour of the eye white, or None for all-iris animal eyes.
        """
        names = _mirror_names(name) if mirror else (name,)
        self._new_name(names[0])
        bone = self.resolve_bone(bone) if bone else (self._nearest_bone(vec(center)) if self.bones else None)
        e = Eye(names[0], center, radius, look, bone, color, pupil, iris_size, pupil_size, sclera)
        self.eyes.append(e)
        if mirror:
            self.eyes.append(e.mirrored(names[1], flip_name(bone) if bone else None))
        if socket:
            sb = radius * 0.45 if socket_blend is None else socket_blend
            self.sphere(socket_name(names[0]), center, radius * 1.06, bone=bone, blend=sb,
                        op="subtract", mirror=mirror)
        return self

    # ----------------------------------------------------------- materials
    def material(self, key, kind=None, color=None, **options):
        """Define / update the look of a material key used by parts.

        kind: skin, fur, scales, feathers, hair, keratin, lips, mouth, teeth,
        membrane, plain. Options: belly_color, pattern (stripes, spots,
        rosettes, patches, giraffe, none), pattern_color, pattern_scale,
        roughness, subsurface, bump.
        """
        spec = dict(self.materials.get(key, {}))
        if kind is not None:
            spec["kind"] = kind
        if color is not None:
            spec["color"] = color
        spec.update({k: v for k, v in options.items() if v is not None})
        self.materials[key] = spec
        return self

    def set_material(self, names, key):
        """Assign material ``key`` to parts / chains named in ``names``."""
        if isinstance(names, str):
            names = [names]
        for n in names:
            hits = [p for p in self.parts if p.name == n or p.group == n]
            if not hits:
                raise NaturalFormError("no part %r.%s" % (n, _suggest(n, self.part_names())))
            for p in hits:
                p.material = key
        return self

    # ----------------------------------------------------------- queries
    def part_names(self):
        seen = []
        for p in self.parts:
            if p.group not in seen:
                seen.append(p.group)
        return seen

    def bone_names(self):
        return list(self.bones)

    def get(self, name):
        hits = [p for p in self.parts if p.name == name or p.group == name]
        if not hits:
            raise NaturalFormError("no part %r.%s" % (name, _suggest(name, self.part_names())))
        return hits

    def bounds(self):
        fps = self.field_parts()
        los = [fp.lo for fp in fps if fp.op == "union"]
        his = [fp.hi for fp in fps if fp.op == "union"]
        for e in self.eyes:
            los.append(e.center - e.radius)
            his.append(e.center + e.radius)
        return np.min(los, axis=0), np.max(his, axis=0)

    def dimensions(self):
        lo, hi = self.bounds()
        return _r(hi - lo, 4)

    def summary(self, bones=True, parts=True):
        """Compact human/LLM-readable description (joint positions in metres)."""
        lines = ["%r" % self]
        lo, hi = self.bounds()
        lines.append("bounds min=%s max=%s" % (_r(lo, 3), _r(hi, 3)))
        if bones and self.bones:
            lines.append("bones (name: head -> tail, parent):")
            for b in self.bones.values():
                lines.append("  %s: %s -> %s  parent=%s" % (b.name, _r(b.head, 3), _r(b.tail, 3), b.parent))
        if parts:
            lines.append("parts (group: shape, op, bone, material):")
            for g in self.part_names():
                ps = [p for p in self.parts if p.group == g]
                p = ps[0]
                extra = " x%d segments" % len(ps) if len(ps) > 1 else ""
                lay = "" if p.layer == "body" else " layer=%s" % p.layer
                lines.append("  %s: %s%s %s bone=%s mat=%s%s" % (g, p.prim.kind, extra, p.op, p.rig_bone,
                                                                p.material or self.default_material, lay))
        if self.eyes:
            lines.append("eyes: " + ", ".join("%s@%s r=%.3f" % (e.name, _r(e.center, 3), e.radius) for e in self.eyes))
        if self.poses:
            lines.append("pose presets: " + ", ".join(self.poses))
        return "\n".join(lines)

    # ----------------------------------------------------------- editing
    def remove(self, names, bones=True):
        """Remove parts / chains / eyes (and, for chains, their bones) by name.
        Children of removed bones are re-parented to the removed bone's parent."""
        if isinstance(names, str):
            names = [names]
        for n in names:
            hit = False
            dropped_bones = set()
            keep = []
            for p in self.parts:
                if p.name == n or p.group == n or p.name == socket_name(n):
                    hit = True
                    if p.group == n and p.name.startswith(n + "["):
                        dropped_bones.add(p.bone)
                else:
                    keep.append(p)
            self.parts = keep
            ne = len(self.eyes)
            self.eyes = [e for e in self.eyes if e.name != n]
            hit = hit or len(self.eyes) != ne
            if n in self.bones:
                dropped_bones.add(n)
                hit = True
            if not hit:
                raise NaturalFormError("nothing named %r.%s" % (n, _suggest(n, self.part_names() + list(self.bones))))
            if bones:
                for b in [b for b in dropped_bones if b in self.bones]:
                    if any(p.bone == b for p in self.parts):
                        continue
                    par = self.bones[b].parent
                    for ob in self.bones.values():
                        if ob.parent == b:
                            ob.parent = par
                            ob.connect = False
                    for p in self.parts:
                        if p.auto_bone == b:
                            p.auto_bone = None
                    del self.bones[b]
        return self

    def remove_matching(self, prefix):
        """Remove every part group / eye / bone whose name starts with ``prefix``."""
        names = [g for g in self.part_names() if g.startswith(prefix)]
        names += [e.name for e in self.eyes if e.name.startswith(prefix)]
        if names:
            self.remove(names)
        for b in [b for b in self.bones if b.startswith(prefix)]:
            if b in self.bones:
                self.remove(b)
        return self

    def transform(self, translate=(0, 0, 0), rotate=(0, 0, 0), scale=1.0, names=None, pivot=(0, 0, 0)):
        """Move / rotate (world XYZ degrees) / uniformly scale the creature,
        or only the parts, eyes and bones listed in ``names`` (around ``pivot``)."""
        s = float(scale)
        piv = vec(pivot)
        R = euler_matrix(rotate)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = vec(translate) + piv - R @ piv

        if isinstance(names, str):
            names = [names]

        def sel(n, g=None):
            return names is None or n in names or (g is not None and g in names)

        moved_bones = set(self.bones) if names is None else {b for b in self.bones if b in names}
        if names is not None:
            self.assign_bones()
            for b in self.bones:
                owned = [p for p in self.parts if p.rig_bone == b]
                if owned and all(sel(p.name, p.group) for p in owned):
                    moved_bones.add(b)

        for p in self.parts:
            if sel(p.name, p.group):
                p.prim = (p.prim.scaled(s, piv) if s != 1.0 else p.prim).transformed(M)
                p.blend *= s
        if names is None and "belly_axis" in self.meta:
            self.meta["belly_axis"] = (R @ vec(self.meta["belly_axis"])).tolist()
        if names is None:
            for r in self.meta.get("detail_regions", []):
                r["radius"] = float(r["radius"]) * s
                if "center" in r:
                    r["center"] = transform_point(M, piv + (vec(r["center"]) - piv) * s).tolist()
                if "offset" in r:
                    r["offset"] = (R @ (vec(r["offset"]) * s)).tolist()
        for e in self.eyes:
            if sel(e.name):
                e.center = piv + (e.center - piv) * s
                e.radius *= s
                e.center = transform_point(M, e.center)
                e.look = R @ e.look
        for b in self.bones.values():
            if b.name in moved_bones:
                b.head = transform_point(M, piv + (b.head - piv) * s)
                b.tail = transform_point(M, piv + (b.tail - piv) * s)
                if b.roll is not None:
                    b.roll = R @ b.roll
        return self

    def scale_part(self, name, factor):
        """Thicken (factor > 1) or thin a part or chain around its own centre."""
        for p in self.get(name):
            p.prim = p.prim.scaled(float(factor), p.prim.center())
        return self

    def inflate_part(self, name, amount):
        """Grow (or shrink, negative) a part or chain by ``amount`` metres."""
        for p in self.get(name):
            p.prim = p.prim.inflated(float(amount))
        return self

    def clothe(self, name, source, thickness=0.006, material=None, clip=None, clip_blend=0.0,
               blend=None, layer=None):
        """Add a clothing / covering layer: copies of the ``source`` parts
        (part or chain names) grown by ``thickness``, optionally cut by
        ``clip`` planes [(point, normal), ...] (the normal side is kept).
        The layer follows the same bones, so it moves with the body. It is
        meshed as its own surface (``layer``, default = ``name``)."""
        if isinstance(source, str):
            source = [source]
        self._new_name(name)
        added = 0
        for sname in source:
            for p in self.get(sname):
                if p.op != "union":
                    continue
                prim = p.prim.inflated(float(thickness))
                if clip:
                    prim = Clipped(prim, clip, clip_blend)
                k = p.blend if blend is None else blend
                np_ = Part("%s{%s}" % (name, p.name), prim, "union", k, p.bone, material, name,
                           layer or name)
                np_.auto_bone = p.auto_bone
                self.parts.append(np_)
                added += 1
        if not added:
            raise NaturalFormError("clothe(%r): no source parts found" % name)
        return self

    def merge(self, other, prefix="", parent=None, keep_materials=True):
        """Add every part, bone, eye and material of ``other`` (optionally
        prefixing names and hanging its root bones from ``parent``)."""
        other = other.copy()
        ren = (lambda n: prefix + n) if prefix else (lambda n: n)
        parent = self.resolve_bone(parent) if parent else None
        for b in other.bones.values():
            nb = copy.copy(b)
            nb.name = ren(b.name)
            nb.parent = ren(b.parent) if b.parent else parent
            if nb.name in self.bones:
                raise NaturalFormError("bone %r exists - use a prefix" % nb.name)
            self.bones[nb.name] = nb
        mat_map = {}
        for k, spec in other.materials.items():
            nk = prefix + k if (k in self.materials and prefix) else k
            mat_map[k] = nk
            if nk not in self.materials or not keep_materials:
                self.materials[nk] = spec
        for p in other.parts:
            m = p.material or other.default_material
            np_ = p.copy_with(name=ren(p.name), group=ren(p.group), bone=ren(p.bone) if p.bone else None,
                              material=mat_map.get(m, m))
            np_.auto_bone = None
            self.parts.append(np_)
        for e in other.eyes:
            ne = copy.copy(e)
            ne.name = ren(e.name)
            ne.bone = ren(e.bone) if e.bone else None
            self.eyes.append(ne)
        return self

    # ----------------------------------------------------------- posing
    def _bone_order(self):
        order, seen = [], set()

        def visit(n, depth=0):
            if n in seen:
                return
            if depth > 512:
                raise NaturalFormError("bone hierarchy has a cycle at %r" % n)
            b = self.bones[n]
            if b.parent and b.parent in self.bones:
                visit(b.parent, depth + 1)
            seen.add(n)
            order.append(n)

        for n in self.bones:
            visit(n)
        return order

    def assign_bones(self):
        """Give every part without an explicit bone the nearest bone."""
        for p in self.parts:
            if not p.bone:
                p.auto_bone = self._nearest_bone(p.prim.center()) if self.bones else None
        for e in self.eyes:
            if not e.bone and self.bones:
                e.bone = self._nearest_bone(e.center)
        return self

    def pose(self, pose):
        """Re-pose the body *before* meshing (joints blend perfectly).

        ``pose`` is a preset name (see ``.poses``) or a dict
        ``{bone: spec}``. spec can be:
          [rx, ry, rz]                 world-axis degrees (X then Y then Z)
          {"aim": [x, y, z]}           point the bone along a world direction
          {"aim": [...], "twist": deg} then twist around its own axis
          {"axis": [x, y, z], "angle": deg}
        Children follow their parents; rotations happen around the bone head.
        Bones are processed parent-first, so child specs see the parent pose.
        """
        if isinstance(pose, str):
            if pose not in self.poses:
                raise NaturalFormError("unknown pose %r. Presets: %s" % (pose, ", ".join(self.poses) or "none"))
            pose = self.poses[pose]
        if not pose:
            return self
        unknown = [b for b in pose if b not in self.bones]
        if unknown:
            raise NaturalFormError("pose: unknown bone(s) %s.%s Bones: %s" % (
                unknown, _suggest(unknown[0], self.bones), ", ".join(self.bones)))
        self.assign_bones()
        W = {}
        I4 = np.eye(4)
        for n in self._bone_order():
            b = self.bones[n]
            Wb = W[b.parent].copy() if b.parent in W else I4.copy()
            spec = pose.get(n)
            if spec is not None:
                R = self._pose_rotation(spec, Wb, b)
                h = transform_point(Wb, b.head)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = h - R @ h
                Wb = T @ Wb
            W[n] = Wb
        for p in self.parts:
            bn = p.rig_bone
            if bn in W:
                p.prim = p.prim.transformed(W[bn])
        for e in self.eyes:
            if e.bone in W:
                e2 = e.transformed(W[e.bone])
                e.center, e.look = e2.center, e2.look
        for n, b in list(self.bones.items()):
            self.bones[n] = b.transformed(W[n])
        if self.meta.get("ground"):
            self.snap_to_ground()
        return self

    def snap_to_ground(self, z=0.0):
        """Translate the creature so its lowest point rests at height ``z``."""
        fps = [fp for fp in self.field_parts() if fp.op == "union"]
        if not fps:
            return self
        low = min(fp.prim.distance_bound_low() if hasattr(fp.prim, "distance_bound_low") else fp.lo[2]
                  for fp in fps)
        return self.transform(translate=(0, 0, z - low))

    @staticmethod
    def _pose_rotation(spec, Wb, bone):
        if isinstance(spec, dict):
            if "aim" in spec:
                cur = Wb[:3, :3] @ (bone.tail - bone.head)
                R = rotation_between(cur, vec(spec["aim"]))
                if spec.get("twist"):
                    R = axis_angle_matrix(R @ cur, math.radians(float(spec["twist"]))) @ R
                return R
            if "axis" in spec:
                return axis_angle_matrix(vec(spec["axis"]), math.radians(float(spec.get("angle", 0))))
            if "rotate" in spec:
                return euler_matrix(spec["rotate"])
            raise NaturalFormError("pose spec needs 'aim', 'axis'+'angle' or 'rotate': %r" % spec)
        return euler_matrix(spec)

    # ----------------------------------------------------------- geometry
    def layers(self):
        """Ordered list of layer names (``body`` first if present)."""
        out = []
        for p in self.parts:
            if p.layer not in out:
                out.append(p.layer)
        if "body" in out:
            out.remove("body")
            out.insert(0, "body")
        return out

    def field_parts(self, include_subtract=True, layer=None):
        """FieldParts for evaluation. ``layer=None`` -> union of all layers'
        additive parts plus the body's carves (used for bounds / ray casts)."""
        out = []
        for i, p in enumerate(self.parts):
            if p.op != "union" and not include_subtract:
                continue
            if layer is not None and p.layer != layer:
                continue
            if layer is None and p.op != "union" and p.layer != "body":
                continue
            out.append(FieldPart(p.prim, p.op, p.blend, i))
        return out

    def surface_point(self, origin, direction, max_dist=None):
        """Ray-march from ``origin`` along ``direction`` and return the first
        point on the body surface (None if nothing is hit). Handy for
        placing eyes, horns, ears or accessories on the skin."""
        fps = self.field_parts()
        if max_dist is None:
            lo, hi = self.bounds()
            max_dist = float(np.linalg.norm(hi - lo)) * 2
        p = raymarch(fps, origin, direction, max_dist)
        return None if p is None else [round(float(x), 5) for x in p]

    # ----------------------------------------------------------- io
    def to_dict(self):
        return {
            "name": self.name, "kind": self.kind,
            "bones": [b.to_dict() for b in self.bones.values()],
            "parts": [p.to_dict() for p in self.parts],
            "eyes": [e.to_dict() for e in self.eyes],
            "materials": copy.deepcopy(self.materials),
            "default_material": self.default_material,
            "poses": copy.deepcopy(self.poses),
            "meta": copy.deepcopy(self.meta),
        }

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, str):
            d = json.loads(d)
        c = cls(d.get("name", "Creature"), d.get("kind", "custom"))
        for bd in d.get("bones", []):
            b = Bone.from_dict(bd)
            c.bones[b.name] = b
        c.parts = [Part.from_dict(pd) for pd in d.get("parts", [])]
        c.eyes = [Eye.from_dict(ed) for ed in d.get("eyes", [])]
        c.materials = dict(d.get("materials", {}))
        c.default_material = d.get("default_material", "body")
        c.poses = dict(d.get("poses", {}))
        c.meta = dict(d.get("meta", {}))
        return c

    # ----------------------------------------------------------- build
    def build(self, **kwargs):
        """Create the Blender object(s). See ``naturalform.build`` for options."""
        from ..blender.builder import build_creature
        return build_creature(self, **kwargs)
