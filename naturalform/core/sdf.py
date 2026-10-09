"""Signed distance field primitives and field evaluation.

Pure numpy - no ``bpy`` import - so the geometry engine can be tested and
used outside of Blender.

Conventions
-----------
* Distances are in scene units (metres). Negative = inside.
* Every primitive exposes ``bounds()``, ``distance(P)``, ``transformed(M)``,
  ``mirrored()``, ``scale()`` (a characteristic radius) and ``to_dict()``.
* Parts are combined in order with a polynomial smooth-min (union) or
  smooth-max (subtract), so the order of parts matters for carving.
"""

import math

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------

def vec(v, n=3):
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 1 and n > 1:
        a = np.repeat(a, n)
    if a.size != n:
        raise ValueError("expected %d numbers, got %r" % (n, v))
    return a


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v * 0.0


def rotation_between(a, b):
    """3x3 rotation matrix turning direction ``a`` onto direction ``b``."""
    a = normalize(a)
    b = normalize(b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 degrees: rotate around any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle_matrix(axis, math.pi)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def axis_angle_matrix(axis, angle):
    """Rotation matrix for ``angle`` radians around ``axis`` (right hand rule)."""
    x, y, z = normalize(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def euler_matrix(deg_xyz):
    """World-axis rotation: rotate around X, then Y, then Z (degrees)."""
    rx, ry, rz = (math.radians(float(a)) for a in vec(deg_xyz))
    return (axis_angle_matrix((0, 0, 1), rz)
            @ axis_angle_matrix((0, 1, 0), ry)
            @ axis_angle_matrix((1, 0, 0), rx))


def frame_from_axis(axis, up=(0.0, 0.0, 1.0)):
    """Rotation whose local +Y axis points along ``axis`` (local Z ~ ``up``)."""
    y = normalize(axis)
    up = normalize(up)
    if abs(np.dot(y, up)) > 0.999:
        up = np.array([0.0, -1.0, 0.0]) if abs(y[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(y, up))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


MIRROR = np.diag([-1.0, 1.0, 1.0])


def transform_point(M, p):
    return M[:3, :3] @ p + M[:3, 3]


# ---------------------------------------------------------------------------
# smooth booleans
# ---------------------------------------------------------------------------

def smin(a, b, k):
    """Polynomial smooth minimum (union) with blend radius ``k``."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * (k * 0.25)


def smax(a, b, k):
    """Polynomial smooth maximum (intersection / subtraction)."""
    if k <= 0.0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * (k * 0.25)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

class Primitive:
    kind = "primitive"

    def bounds(self):
        raise NotImplementedError

    def distance(self, P):
        raise NotImplementedError

    def transformed(self, M):
        raise NotImplementedError

    def mirrored(self):
        raise NotImplementedError

    def scale(self):
        raise NotImplementedError

    def center(self):
        lo, hi = self.bounds()
        return (lo + hi) * 0.5

    def to_dict(self):
        raise NotImplementedError


class RoundCone(Primitive):
    """Segment a->b with radius ra at a and rb at b (a capsule when ra == rb).

    ``squash`` (scale of the cross section) along ``squash_dir`` turns the
    round cross-section into an ellipse - e.g. a torso that is wider than it
    is deep, or a flat tail/fin.
    """

    kind = "round_cone"

    def __init__(self, a, b, ra, rb, squash=1.0, squash_dir=(0.0, 1.0, 0.0)):
        self.a = vec(a)
        self.b = vec(b)
        self.ra = max(float(ra), 1e-5)
        self.rb = max(float(rb), 1e-5)
        self.squash = max(float(squash), 0.02)
        self.squash_dir = normalize(vec(squash_dir))

    def _squash_normal(self):
        axis = self.b - self.a
        n = self.squash_dir
        L = np.linalg.norm(axis)
        if L > EPS:
            ax = axis / L
            n = n - np.dot(n, ax) * ax
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            return None
        return n / nn

    def bounds(self):
        r = max(self.ra, self.rb) * max(1.0, self.squash)
        lo = np.minimum(self.a - self.ra * max(1.0, self.squash), self.b - r)
        hi = np.maximum(self.a + self.ra * max(1.0, self.squash), self.b + r)
        return lo, hi

    def scale(self):
        return min(self.ra, self.rb) * min(1.0, self.squash) * 0.5 + max(self.ra, self.rb) * 0.5

    def distance(self, P):
        P = np.asarray(P, dtype=np.float64)
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        pa = P - a
        f = 1.0
        if abs(self.squash - 1.0) > 1e-6:
            n = self._squash_normal()
            if n is not None:
                s = pa @ n
                pa = pa + np.outer(s * (1.0 / self.squash - 1.0), n)
                f = min(1.0, self.squash)
        ba = b - a
        l2 = float(np.dot(ba, ba))
        rr = r1 - r2
        # degenerate: one sphere swallows the other
        if l2 < EPS or math.sqrt(l2) <= abs(rr) + 1e-9:
            if r1 >= r2:
                return (np.linalg.norm(pa, axis=1) - r1) * f
            return (np.linalg.norm(pa - ba, axis=1) - r2) * f
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        y = pa @ ba
        z = y - l2
        # |pa*l2 - y*ba|^2 expanded (avoids an N x 3 temporary)
        x2 = np.maximum(l2 * (l2 * np.einsum("ij,ij->i", pa, pa) - y * y), 0.0)
        y2 = y * y * l2
        z2 = z * z * l2
        k = math.copysign(1.0, rr) * rr * rr * x2
        d_b = np.sqrt(x2 + z2) * il2 - r2
        d_a = np.sqrt(x2 + y2) * il2 - r1
        d_m = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
        out = np.where(np.sign(z) * a2 * z2 > k, d_b,
                       np.where(np.sign(y) * a2 * y2 < k, d_a, d_m))
        return out * f

    def transformed(self, M):
        R = M[:3, :3]
        return RoundCone(transform_point(M, self.a), transform_point(M, self.b),
                         self.ra, self.rb, self.squash, R @ self.squash_dir)

    def mirrored(self):
        return RoundCone(MIRROR @ self.a, MIRROR @ self.b, self.ra, self.rb,
                         self.squash, MIRROR @ self.squash_dir)

    def scaled(self, s, pivot):
        return RoundCone(pivot + (self.a - pivot) * s, pivot + (self.b - pivot) * s,
                         self.ra * s, self.rb * s, self.squash, self.squash_dir)

    def inflated(self, t):
        return RoundCone(self.a, self.b, self.ra + t, self.rb + t, self.squash, self.squash_dir)

    def to_dict(self):
        d = {"shape": "round_cone", "a": _r(self.a), "b": _r(self.b),
             "ra": _r(self.ra), "rb": _r(self.rb)}
        if abs(self.squash - 1.0) > 1e-6:
            d["squash"] = _r(self.squash)
            d["squash_dir"] = _r(self.squash_dir)
        return d


class Ellipsoid(Primitive):
    """Ellipsoid with semi-axes ``radii`` in its local frame ``rot`` (3x3)."""

    kind = "ellipsoid"

    def __init__(self, center, radii, rot=None):
        self.c = vec(center)
        self.r = np.maximum(vec(radii), 1e-5)
        self.R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64).reshape(3, 3)

    def bounds(self):
        ext = np.sqrt(((self.R * self.r[None, :]) ** 2).sum(axis=1))
        return self.c - ext, self.c + ext

    def scale(self):
        return float(np.sort(self.r)[:2].mean())

    def distance(self, P):
        p = (np.asarray(P, dtype=np.float64) - self.c) @ self.R  # to local
        k0 = np.linalg.norm(p / self.r, axis=1)
        k1 = np.linalg.norm(p / (self.r * self.r), axis=1)
        safe = k1 > 1e-12
        d = np.where(safe, k0 * (k0 - 1.0) / np.where(safe, k1, 1.0), -float(self.r.min()))
        return d

    def transformed(self, M):
        return Ellipsoid(transform_point(M, self.c), self.r, M[:3, :3] @ self.R)

    def mirrored(self):
        return Ellipsoid(MIRROR @ self.c, self.r, MIRROR @ self.R @ MIRROR)

    def scaled(self, s, pivot):
        return Ellipsoid(pivot + (self.c - pivot) * s, self.r * s, self.R)

    def inflated(self, t):
        return Ellipsoid(self.c, self.r + t, self.R)

    def to_dict(self):
        d = {"shape": "ellipsoid", "center": _r(self.c), "radii": _r(self.r)}
        if not np.allclose(self.R, np.eye(3)):
            d["rot"] = _r(self.R)
        return d


class RoundBox(Primitive):
    """Box with half extents ``size`` and corner rounding ``rounding``."""

    kind = "round_box"

    def __init__(self, center, size, rot=None, rounding=0.0):
        self.c = vec(center)
        self.h = np.maximum(vec(size), 1e-5)
        self.R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64).reshape(3, 3)
        self.rnd = float(min(max(rounding, 0.0), self.h.min()))

    def bounds(self):
        ext = np.abs(self.R) @ self.h
        return self.c - ext, self.c + ext

    def scale(self):
        return float(np.sort(self.h)[:2].mean())

    def distance(self, P):
        p = (np.asarray(P, dtype=np.float64) - self.c) @ self.R
        q = np.abs(p) - self.h + self.rnd
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside - self.rnd

    def transformed(self, M):
        return RoundBox(transform_point(M, self.c), self.h, M[:3, :3] @ self.R, self.rnd)

    def mirrored(self):
        return RoundBox(MIRROR @ self.c, self.h, MIRROR @ self.R @ MIRROR, self.rnd)

    def scaled(self, s, pivot):
        return RoundBox(pivot + (self.c - pivot) * s, self.h * s, self.R, self.rnd * s)

    def inflated(self, t):
        return RoundBox(self.c, self.h + t, self.R, self.rnd + t)

    def to_dict(self):
        d = {"shape": "round_box", "center": _r(self.c), "size": _r(self.h),
             "rounding": _r(self.rnd)}
        if not np.allclose(self.R, np.eye(3)):
            d["rot"] = _r(self.R)
        return d


class Torus(Primitive):
    """Torus around local Z axis (major radius ``R``, tube radius ``r``)."""

    kind = "torus"

    def __init__(self, center, major, minor, rot=None):
        self.c = vec(center)
        self.Rm = float(major)
        self.rm = float(minor)
        self.R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64).reshape(3, 3)

    def bounds(self):
        e = self.Rm + self.rm
        ext = np.abs(self.R) @ np.array([e, e, self.rm])
        return self.c - ext, self.c + ext

    def scale(self):
        return self.rm

    def distance(self, P):
        p = (np.asarray(P, dtype=np.float64) - self.c) @ self.R
        q = np.stack([np.hypot(p[:, 0], p[:, 1]) - self.Rm, p[:, 2]], axis=1)
        return np.linalg.norm(q, axis=1) - self.rm

    def transformed(self, M):
        return Torus(transform_point(M, self.c), self.Rm, self.rm, M[:3, :3] @ self.R)

    def mirrored(self):
        return Torus(MIRROR @ self.c, self.Rm, self.rm, MIRROR @ self.R @ MIRROR)

    def scaled(self, s, pivot):
        return Torus(pivot + (self.c - pivot) * s, self.Rm * s, self.rm * s, self.R)

    def inflated(self, t):
        return Torus(self.c, self.Rm, self.rm + t, self.R)

    def to_dict(self):
        d = {"shape": "torus", "center": _r(self.c), "major": _r(self.Rm), "minor": _r(self.rm)}
        if not np.allclose(self.R, np.eye(3)):
            d["rot"] = _r(self.R)
        return d


class Clipped(Primitive):
    """A primitive cut by half-spaces. Each plane is (point, normal); the
    side the normal points to is kept. ``blend`` softens the cut edge."""

    kind = "clipped"

    def __init__(self, base, planes, blend=0.0):
        self.base = base
        self.planes = [(vec(p), normalize(vec(n))) for p, n in planes]
        self.blend = float(blend)

    @property
    def inner_kind(self):
        return self.base.kind

    def bounds(self):
        lo, hi = self.base.bounds()
        lo, hi = lo.copy(), hi.copy()
        for p, n in self.planes:
            ax = int(np.argmax(np.abs(n)))
            if abs(n[ax]) > 0.999:  # axis aligned cut -> tighten the box
                if n[ax] > 0:
                    lo[ax] = max(lo[ax], min(p[ax], hi[ax]))
                else:
                    hi[ax] = min(hi[ax], max(p[ax], lo[ax]))
        return lo, hi

    def scale(self):
        return self.base.scale()

    def center(self):
        lo, hi = self.bounds()
        return (lo + hi) * 0.5

    def distance(self, P):
        P = np.asarray(P, dtype=np.float64)
        d = self.base.distance(P)
        for p0, n in self.planes:
            d = smax(d, -((P - p0) @ n), self.blend)
        return d

    def transformed(self, M):
        R = M[:3, :3]
        return Clipped(self.base.transformed(M),
                       [(transform_point(M, p), R @ n) for p, n in self.planes], self.blend)

    def mirrored(self):
        return Clipped(self.base.mirrored(), [(MIRROR @ p, MIRROR @ n) for p, n in self.planes],
                       self.blend)

    def scaled(self, s, pivot):
        return Clipped(self.base.scaled(s, pivot),
                       [(pivot + (p - pivot) * s, n) for p, n in self.planes], self.blend * s)

    def inflated(self, t):
        return Clipped(self.base.inflated(t), self.planes, self.blend)

    def to_dict(self):
        return {"shape": "clipped", "base": self.base.to_dict(),
                "planes": [[_r(p), _r(n)] for p, n in self.planes], "clip_blend": _r(self.blend)}


def primitive_from_dict(d):
    s = d["shape"]
    if s == "clipped":
        return Clipped(primitive_from_dict(d["base"]), d["planes"], d.get("clip_blend", 0.0))
    if s == "round_cone":
        return RoundCone(d["a"], d["b"], d["ra"], d["rb"], d.get("squash", 1.0),
                         d.get("squash_dir", (0, 1, 0)))
    if s == "ellipsoid":
        return Ellipsoid(d["center"], d["radii"], d.get("rot"))
    if s == "round_box":
        return RoundBox(d["center"], d["size"], d.get("rot"), d.get("rounding", 0.0))
    if s == "torus":
        return Torus(d["center"], d["major"], d["minor"], d.get("rot"))
    raise ValueError("unknown shape %r" % s)


def _r(x, nd=5):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return round(float(a), nd)
    return np.round(a, nd).tolist()


# ---------------------------------------------------------------------------
# field evaluation
# ---------------------------------------------------------------------------

class FieldPart:
    """Light-weight evaluation record: primitive + boolean op + blend."""

    __slots__ = ("prim", "op", "blend", "lo", "hi", "index")

    def __init__(self, prim, op="union", blend=0.0, index=0):
        self.prim = prim
        self.op = op
        self.blend = float(max(blend, 0.0))
        self.lo, self.hi = prim.bounds()
        self.index = index


def evaluate(parts, P, margin=0.0, on_part=None):
    """Evaluate the combined field of ``parts`` at points ``P`` (N x 3).

    Each part is only evaluated for points inside its (blend + margin)
    expanded bounding box. ``margin`` must be at least as large as the band
    of values the caller cares about; outside of that band only the sign of
    the result is meaningful.

    ``on_part(i, idx, d)`` - optional callback receiving, for part ``i``, the
    indices of the evaluated points and their raw (unblended) distance.
    """
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    order = None
    if n > 2048:
        # sort along x once so each part only scans a slice of the points
        order = np.argsort(P[:, 0], kind="stable")
        P = P[order]
        xs = P[:, 0]
    field = np.full(n, 1e9, dtype=np.float64)
    for i, fp in enumerate(parts):
        pad = fp.blend + margin
        lo = fp.lo - pad
        hi = fp.hi + pad
        if order is not None:
            s0 = int(np.searchsorted(xs, lo[0], "left"))
            s1 = int(np.searchsorted(xs, hi[0], "right"))
            if s1 <= s0:
                continue
            sub = P[s0:s1]
            m = ((sub[:, 1] >= lo[1]) & (sub[:, 1] <= hi[1])
                 & (sub[:, 2] >= lo[2]) & (sub[:, 2] <= hi[2]))
            idx = np.nonzero(m)[0] + s0
        else:
            idx = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        if not len(idx):
            continue
        d = fp.prim.distance(P[idx])
        if on_part is not None:
            on_part(i, idx if order is None else order[idx], d)
        if fp.op == "subtract":
            field[idx] = smax(field[idx], -d, fp.blend)
        elif fp.op == "intersect":
            field[idx] = smax(field[idx], d, fp.blend)
        else:
            field[idx] = smin(field[idx], d, fp.blend)
    if order is not None:
        out = np.empty_like(field)
        out[order] = field
        return out
    return field


def union_bounds(parts):
    los = [p.lo - p.blend for p in parts if p.op == "union"]
    his = [p.hi + p.blend for p in parts if p.op == "union"]
    if not los:
        raise ValueError("nothing to mesh: no additive parts")
    return np.min(los, axis=0), np.max(his, axis=0)


_TET = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=np.float64)


def value_and_gradient(parts, P, h, margin=0.0):
    """Field value and tetrahedral-difference gradient in one batched pass."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    allp = np.concatenate([P] + [P + k * h for k in _TET])
    f = evaluate(parts, allp, margin)
    g = np.zeros_like(P)
    for i, k in enumerate(_TET):
        g += k[None, :] * f[(i + 1) * n:(i + 2) * n, None]
    return f[:n], g / (4.0 * h)


def gradient(parts, P, h, margin=0.0):
    """Tetrahedral finite-difference gradient of the field."""
    return value_and_gradient(parts, P, h, margin)[1]


def raymarch(parts, origin, direction, max_dist=10.0, tol=1e-5):
    """Sphere-trace from ``origin`` along ``direction``; return hit point or None."""
    o = vec(origin)
    d = normalize(vec(direction))
    t = 0.0
    start = evaluate(parts, o[None, :], margin=max_dist)[0]
    inside = start < 0
    for _ in range(256):
        p = o + d * t
        f = evaluate(parts, p[None, :], margin=max_dist)[0]
        if inside:
            f = -f
        if f < tol:
            return p
        t += max(f * 0.9, tol)
        if t > max_dist:
            return None
    return None
