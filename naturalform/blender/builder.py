"""Build Blender objects (mesh + armature + materials) from a Creature."""

import json
import math
import time

import bpy
import numpy as np

from ..core.creature import NaturalFormError
from ..core.mesher import choose_voxel, mesh_adaptive
from ..core.sdf import evaluate
from . import materials as M
from .rig import create_armature, ensure_object_mode

DETAIL = {"preview": 70, "low": 110, "medium": 170, "high": 260, "ultra": 400}
# how strongly "min_feature" (e.g. finger radius) may refine the voxel size
_FEATURE = {"preview": None, "low": None, "medium": 1.35, "high": 1.0, "ultra": 0.75}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _collection(name):
    scene = bpy.context.scene
    if not name:
        coll = bpy.context.collection
        return coll if coll is not None else scene.collection
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        scene.collection.children.link(coll)
    elif coll.name not in scene.collection.children and coll != scene.collection:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass
    return coll


def _delete_tree(name):
    """Delete a previously built NaturalForm object tree (never user objects)."""
    obj = bpy.data.objects.get(name)
    if obj is None or "nf_kind" not in obj:
        return
    victims = [obj] + list(obj.children_recursive)
    for o in victims:
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)


def _cube_sphere(n=8):
    faces_def = [
        ((1, -1, -1), (0, 2, 0), (0, 0, 2)), ((-1, -1, -1), (0, 0, 2), (0, 2, 0)),
        ((-1, 1, -1), (0, 0, 2), (2, 0, 0)), ((-1, -1, -1), (2, 0, 0), (0, 0, 2)),
        ((-1, -1, 1), (2, 0, 0), (0, 2, 0)), ((-1, -1, -1), (0, 2, 0), (2, 0, 0)),
    ]
    V, F = [], []
    t = np.linspace(0.0, 1.0, n + 1)
    for o, u, v in faces_def:
        o, u, v = (np.array(a, dtype=np.float64) for a in (o, u, v))
        ii, jj = np.meshgrid(t, t, indexing="ij")
        pts = o + ii[..., None] * u + jj[..., None] * v
        base = sum(len(x) for x in V)
        V.append(pts.reshape(-1, 3))
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) + base
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
        F.append(q)
    V = np.concatenate(V)
    F = np.concatenate(F)
    key = np.round(V, 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    F = inv[F]
    U = uniq / np.linalg.norm(uniq, axis=1)[:, None]
    # equal-area-ish warp so quads are more even
    return U, F


def _eye_frame(look):
    z = look / max(np.linalg.norm(look), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.95:
        up = np.array([0.0, -1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# weights / materials from the field
# ---------------------------------------------------------------------------

def _part_keys(c, keys):
    return np.array([keys.index(p.material or c.default_material) for p in c.parts])


def _face_materials(c, fps, V, Q, T, keys):
    """Material index per face (quads first, then tris): the material of the
    closest additive part."""
    part_key = _part_keys(c, keys)
    nfaces = len(Q) + len(T)
    if len(set(part_key[[fp.index for fp in fps]])) == 1:
        return np.full(nfaces, part_key[fps[0].index], dtype=np.int64)
    centers = np.concatenate([V[Q].mean(axis=1) if len(Q) else np.zeros((0, 3)),
                              V[T].mean(axis=1) if len(T) else np.zeros((0, 3))])
    best = np.full(nfaces, np.inf)
    owner = np.full(nfaces, fps[0].index, dtype=np.int64)

    def cb(i, idx, d):
        if fps[i].op != "union":
            return
        m = d < best[idx]
        ii = idx[m]
        best[ii] = d[m]
        owner[ii] = fps[i].index

    evaluate(fps, centers, margin=0.0, on_part=cb)
    return part_key[owner]


def _material_distances(c, fps, P, mats_needed, part_key):
    """Per-material minimum raw distance at points P -> dict key -> array."""
    D = {m: np.full(len(P), np.inf) for m in mats_needed}

    def cb(i, idx, d):
        fp = fps[i]
        if fp.op != "union":
            return
        k = part_key[fp.index]
        if k in D:
            D[k][idx] = np.minimum(D[k][idx], d)

    evaluate(fps, P, margin=0.05, on_part=cb)
    return D


def _snap_material_borders(c, fps, V, Q, T, face_mat, keys, voxel, iters=3):
    """Move vertices on material borders onto the curve where the two
    materials' parts are equally close -> smooth borders instead of voxel
    staircases."""
    nv = len(V)
    nq = len(Q)
    fm = np.concatenate([np.repeat(face_mat[:nq], 4), np.repeat(face_mat[nq:], 3)])
    flat = np.concatenate([Q.ravel(), T.ravel()])
    vmin = np.full(nv, 1 << 30, dtype=np.int64)
    vmax = np.full(nv, -1, dtype=np.int64)
    np.minimum.at(vmin, flat, fm)
    np.maximum.at(vmax, flat, fm)
    bnd = np.nonzero((vmax >= 0) & (vmin != vmax))[0]
    if not len(bnd):
        return V
    A, B = vmin[bnd], vmax[bnd]
    part_key = _part_keys(c, keys)
    mats = set(A.tolist()) | set(B.tolist())
    P = V[bnd].copy()
    h = voxel * 0.25
    tet = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=np.float64)

    def g_at(pts):
        D = _material_distances(c, fps, pts, mats, part_key)
        da = np.choose(np.searchsorted(sorted(mats), A), [D[m] for m in sorted(mats)])
        db = np.choose(np.searchsorted(sorted(mats), B), [D[m] for m in sorted(mats)])
        out = da - db
        return np.where(np.isfinite(out), out, 0.0)

    from ..core.mesher import _project
    from ..core.sdf import gradient
    for _ in range(iters):
        g = g_at(P)
        gr = np.zeros_like(P)
        for k in tet:
            gr += k[None, :] * g_at(P + k * h)[:, None]
        gr /= 4.0 * h
        n = gradient(fps, P, h, 4 * voxel)
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        gt = gr - (gr * n).sum(axis=1)[:, None] * n
        gg = np.maximum((gt * gt).sum(axis=1), 1e-12)
        step = (g / gg)[:, None] * gt
        ln = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, 0.7 * voxel / np.maximum(ln, 1e-12))[:, None]
        P = _project(fps, P - step, voxel, 1)
    V = V.copy()
    V[bnd] = P
    return V


def _bone_weights(c, fps, V):
    deform = [b for b in c.bones if c.bones[b].deform]
    if not deform or not len(V):
        return deform, None
    bidx = {b: i for i, b in enumerate(deform)}
    nb, nv = len(deform), len(V)
    D = np.full((nb, nv), np.inf, dtype=np.float32)
    best = np.full(nv, np.inf)
    best_scale = np.ones(nv)
    part_bone = [bidx.get(c.parts[fp.index].rig_bone, -1) if fp.op == "union" else -1 for fp in fps]
    part_scale = [fp.prim.scale() for fp in fps]

    def cb(i, idx, d):
        b = part_bone[i]
        if b < 0:
            return
        D[b, idx] = np.minimum(D[b, idx], d.astype(np.float32))
        m = d < best[idx]
        ii = idx[m]
        best[ii] = d[m]
        best_scale[ii] = part_scale[i]

    margin = 0.8 * max(part_scale) if part_scale else 0.0
    evaluate(fps, V, margin=margin, on_part=cb)
    # vertices no part reached (should not happen) -> nearest bone segment
    lost = ~np.isfinite(D.min(axis=0))
    if lost.any():
        for j in np.nonzero(lost)[0]:
            D[bidx[c._nearest_bone(V[j])], j] = 0.0
    related = np.eye(nb, dtype=bool)
    for b in deform:
        p = c.bones[b].parent
        while p is not None and p not in bidx:
            p = c.bones[p].parent if p in c.bones else None
        if p is not None:
            related[bidx[b], bidx[p]] = related[bidx[p], bidx[b]] = True
    nearest = np.argmin(D, axis=0)
    dmin = D[nearest, np.arange(nv)]
    fall = np.maximum(0.6 * best_scale, 1e-4).astype(np.float32)
    W = np.clip(1.0 - (D - dmin[None, :]) / fall[None, :], 0.0, 1.0) ** 2
    W *= related[:, nearest]
    # keep the 4 strongest influences
    if nb > 4:
        kth = np.partition(W, nb - 4, axis=0)[nb - 4]
        W[W < kth[None, :]] = 0.0
    W /= np.maximum(W.sum(axis=0), 1e-9)[None, :]
    return deform, W


def _assign_groups(obj, names, W, extra=None):
    groups = {}
    for i, n in enumerate(names):
        w = W[i] if W is not None else None
        g = obj.vertex_groups.new(name=n)
        groups[n] = g
        if w is None:
            continue
        q = np.round(w * 100.0).astype(np.int32)
        nz = np.nonzero(q)[0]
        if not len(nz):
            continue
        qs = q[nz]
        order = np.argsort(qs, kind="stable")
        nz, qs = nz[order], qs[order]
        cuts = np.nonzero(np.diff(qs))[0] + 1
        for seg_idx, seg_val in zip(np.split(nz, cuts), np.split(qs, cuts)):
            g.add(seg_idx.tolist(), float(seg_val[0]) / 100.0, "REPLACE")
    for n, idx in (extra or {}).items():
        g = groups.get(n) or obj.vertex_groups.new(name=n)
        g.add(idx.tolist(), 1.0, "REPLACE")
    return groups


def _detail_regions(c, voxel, detail):
    """Resolve creature.meta['detail_regions'] -> list of region dicts.

    Each entry: {"bone": name, "at": 0..1 along the bone, "offset": [x,y,z],
    "radius": r, "factor": f} or {"center": [x,y,z], ...}. Optional
    "cut_bone"/"cut_at": a plane across that bone (normal pointing to its
    tail) that bounds the region - gives clean seams across necks / wrists.
    Optional "layers": only refine these layers (default: all).
    """
    if detail in ("preview", "low"):
        return []
    out = []
    for r in c.meta.get("detail_regions", []):
        if "bone" in r:
            b = c.bones.get(r["bone"])
            if b is None:
                continue
            cen = b.head + (b.tail - b.head) * float(r.get("at", 0.5)) + np.asarray(r.get("offset", (0, 0, 0)))
        else:
            cen = np.asarray(r["center"], dtype=np.float64)
        f = float(r.get("factor", 2.0))
        # cap the cell count of a region (~75 cells across its radius)
        rv = max(voxel / f, float(r["radius"]) / 75.0)
        reg = {"center": cen, "radius": float(r["radius"]), "voxel": rv, "layers": r.get("layers")}
        cb = c.bones.get(r.get("cut_bone", ""))
        if cb is not None:
            t = float(r.get("cut_at", 0.5))
            reg["plane"] = (cb.head + (cb.tail - cb.head) * t, cb.tail - cb.head)
        out.append(reg)
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def build_creature(creature, name=None, location=(0, 0, 0), rotation=(0, 0, 0), scale=1.0,
                   detail="medium", rig=True, pose=None, collection=None, smooth=1,
                   replace=True, select=True, voxel=None):
    """Mesh a Creature and create Blender objects.

    detail      preview | low | medium | high | ultra, or an int (cells
                along the longest side). Higher = slower but finer.
    rig         create an armature (the creature root) with weights
    pose        preset name or pose dict applied *before* meshing
    location / rotation (degrees) / scale  placement of the root object
    replace     delete an existing object tree with the same name first
    Returns a dict describing what was created.
    """
    t0 = time.perf_counter()
    if not creature.parts:
        raise NaturalFormError("creature %r has no parts to build" % creature.name)
    name = name or creature.name
    c = creature.copy()
    c.assign_bones()
    if pose:
        c.pose(pose)
    ensure_object_mode()
    if replace:
        _delete_tree(name)
    coll = _collection(collection)

    # ---- mesh every layer
    fps_all = c.field_parts()
    if voxel is None:
        if isinstance(detail, str):
            if detail not in DETAIL:
                raise NaturalFormError("detail must be one of %s or an integer" % ", ".join(DETAIL))
            res = DETAIL[detail]
            f = _FEATURE[detail]
        else:
            res = int(detail)
            f = 1.0
        minf = c.meta.get("min_feature")
        voxel = choose_voxel(fps_all, res, minf * f if (minf and f) else None)
    keys = []
    for p in c.parts:
        k = p.material or c.default_material
        if k not in keys:
            keys.append(k)
    rigged = bool(rig and c.bones)
    regions = _detail_regions(c, voxel, detail)
    Vs, Qs, Ts, Ns, FMq, FMt, Ws, layer_info = [], [], [], [], [], [], [], []
    weights_bones = [b for b in c.bones if c.bones[b].deform] if rigged else []
    nbase = 0
    for layer in c.layers():
        fps = c.field_parts(layer=layer)
        if not any(fp.op == "union" for fp in fps):
            continue
        tl = time.perf_counter()
        regs = [r for r in regions if not r["layers"] or layer in r["layers"]]
        mesh = mesh_adaptive(fps, voxel, regs, relax_iters=smooth)
        V, Q, T, N = mesh["V"], mesh["quads"], mesh["tris"], mesh["normals"]
        if not len(Q) + len(T):
            continue
        fmat = _face_materials(c, fps, V, Q, T, keys)
        if len(np.unique(fmat)) > 1:
            V = _snap_material_borders(c, fps, V, Q, T, fmat, keys, voxel)
        if rigged:
            _, W = _bone_weights(c, fps, V)
            Ws.append(W)
        Vs.append(V)
        Qs.append(Q + nbase)
        Ts.append(T + nbase)
        Ns.append(N)
        FMq.append(fmat[:len(Q)])
        FMt.append(fmat[len(Q):])
        layer_info.append({"layer": layer, "vertices": len(V), "seconds": round(time.perf_counter() - tl, 2)})
        nbase += len(V)
    if not Vs:
        raise NaturalFormError("meshing produced no surface - parts may be too small for the detail level")
    V = np.concatenate(Vs)
    Q = np.concatenate(Qs)
    T = np.concatenate(Ts)
    N = np.concatenate(Ns)
    face_mat_q = np.concatenate(FMq)
    face_mat_t = np.concatenate(FMt)
    W = np.concatenate(Ws, axis=1) if rigged else None

    # ---- eyes (merged into the same mesh)
    eye_unit, eye_faces = _cube_sphere(8)
    eye_V, eye_F, eye_local, eye_mat_idx, eye_groups = [], [], [], [], {}
    eye_keys = []
    eye_bone_defs = {}
    nbase = len(V)
    for e in c.eyes:
        R = _eye_frame(e.look)
        pts = e.center + (eye_unit @ R.T) * e.radius
        eye_V.append(pts)
        eye_F.append(eye_faces + nbase)
        eye_local.append(eye_unit)
        sig = (str(e.color), e.pupil, round(e.iris_size, 3), round(e.pupil_size, 3), str(e.sclera))
        if sig not in eye_keys:
            eye_keys.append(sig)
        eye_mat_idx.append(np.full(len(eye_faces), len(keys) + eye_keys.index(sig)))
        if rigged:
            bname = e.name if e.name not in c.bones else e.name + "_eye"
            eye_bone_defs[bname] = e
            eye_groups[bname] = np.arange(nbase, nbase + len(pts))
        nbase += len(pts)
    if len(N):
        # countershading mask: normals facing the belly axis (default: down)
        bax = np.asarray(c.meta.get("belly_axis", (0.0, 0.0, -1.0)), dtype=np.float64)
        bax /= max(np.linalg.norm(bax), 1e-9)
        belly = _smoothstep(0.05, 0.75, N @ bax)
    else:
        belly = np.zeros(0)
    nv_body = len(V)
    if eye_V:
        V = np.concatenate([V] + eye_V)
        Q = np.concatenate([Q] + eye_F)
        face_mat_q = np.concatenate([face_mat_q] + eye_mat_idx)
        belly = np.concatenate([belly, np.zeros(len(V) - nv_body)])
        if W is not None:
            W = np.concatenate([W, np.zeros((W.shape[0], len(V) - nv_body), dtype=W.dtype)], axis=1)

    # ---- Blender mesh
    mesh_name = name + "_mesh" if rigged else name
    if replace:
        _delete_tree(mesh_name)
    me = bpy.data.meshes.new(mesh_name)
    nv, nq, nt = len(V), len(Q), len(T)
    nf = nq + nt
    me.vertices.add(nv)
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(nq * 4 + nt * 3)
    me.loops.foreach_set("vertex_index", np.concatenate([Q.ravel(), T.ravel()]).astype(np.int32))
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(0, nq * 4, 4),
                                                          nq * 4 + np.arange(0, nt * 3, 3)]).astype(np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.concatenate([np.full(nq, 4), np.full(nt, 3)]).astype(np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    me.polygons.foreach_set("material_index", np.concatenate([face_mat_q, face_mat_t]).astype(np.int32))

    rest = me.attributes.new("nf_rest", "FLOAT_VECTOR", "POINT")
    rest.data.foreach_set("vector", V.astype(np.float32).ravel())
    ba = me.attributes.new("nf_belly", "FLOAT", "POINT")
    ba.data.foreach_set("value", belly.astype(np.float32))
    if eye_V:
        loc = np.zeros((nv, 3), dtype=np.float32)
        loc[nv_body:] = np.concatenate(eye_local)
        ea = me.attributes.new("nf_eye", "FLOAT_VECTOR", "POINT")
        ea.data.foreach_set("vector", loc.ravel())

    mats = []
    for k in keys:
        spec = dict(c.materials.get(k, {}))
        spec.setdefault("kind", "skin")
        mats.append(M.organic_material("%s_%s" % (name, k), spec))
    for i, sig in enumerate(eye_keys):
        col, pupil, iris_size, pupil_size, sclera = sig
        e = next(e for e in c.eyes if (str(e.color), e.pupil, round(e.iris_size, 3), round(e.pupil_size, 3), str(e.sclera)) == sig)
        mats.append(M.eye_material("%s_eye%s" % (name, "" if i == 0 else str(i)), e.color, e.pupil,
                                   e.iris_size, e.pupil_size, e.sclera))
    for m in mats:
        me.materials.append(m)
    me.validate(clean_customdata=False)

    obj = bpy.data.objects.new(mesh_name, me)
    coll.objects.link(obj)
    obj["nf_kind"] = c.kind

    # ---- rig
    root = obj
    arm = None
    if rigged:
        bones = dict(c.bones)
        order = c._bone_order()
        from ..core.creature import Bone
        for bname, e in eye_bone_defs.items():
            bones[bname] = Bone(bname, e.center, e.center + e.look * e.radius * 2.0, e.bone)
            order.append(bname)
        arm = create_armature(name, bones, order, coll)
        _assign_groups(obj, weights_bones, W, eye_groups)
        obj.parent = arm
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        root = arm
        arm["nf_poses"] = json.dumps(c.poses)

    root.location = tuple(float(x) for x in location)
    root.rotation_euler = tuple(math.radians(float(a)) for a in rotation)
    s = float(scale) if np.isscalar(scale) else None
    root.scale = (s, s, s) if s is not None else tuple(float(x) for x in scale)

    root["nf_kind"] = c.kind
    root["nf_spec"] = creature.to_json()
    root["nf_pose"] = json.dumps(pose) if pose is not None else ""
    if c.meta.get("params"):
        root["nf_params"] = json.dumps(c.meta["params"], default=str)

    if select:
        for o in list(bpy.context.selected_objects):
            o.select_set(False)
        obj.select_set(True)
        if arm is not None:
            arm.select_set(True)
        bpy.context.view_layer.objects.active = root
    bpy.context.view_layer.update()

    lo, hi = V[:nv_body].min(axis=0), V[:nv_body].max(axis=0)
    sc = np.array(root.scale)
    return {
        "name": root.name,
        "mesh": obj.name,
        "armature": arm.name if arm else None,
        "kind": c.kind,
        "location": [round(float(x), 4) for x in root.location],
        "dimensions_m": [round(float(x), 3) for x in (hi - lo) * sc],
        "vertices": int(nv),
        "faces": int(nf),
        "bones": len(c.bones) + len(eye_bone_defs) if arm else 0,
        "materials": [m.name for m in mats],
        "pose_presets": list(c.poses),
        "voxel_mm": round(voxel * 1000.0, 2),
        "layers": [l["layer"] for l in layer_info],
        "time_s": round(time.perf_counter() - t0, 2),
        "note": ("Move/rotate/scale the creature via its root object %r." % root.name),
    }
