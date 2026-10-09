"""Turn a list of SDF parts into a quad mesh (numpy only).

Pipeline
--------
1. Narrow band: evaluate the field at the centre of coarse blocks and only
   keep blocks that can contain the surface.
2. Evaluate the fine grid inside those blocks, culling each part to the
   blocks its bounding box touches.
3. Naive surface nets: one vertex per sign-changing cell, one quad per
   sign-changing grid edge.
4. Newton projection of the vertices onto the exact iso-surface, followed by
   a few relax + re-project iterations for nicer quads.
"""

import math
import time

import numpy as np

from .sdf import FieldPart, evaluate, gradient, smax, smin, union_bounds, value_and_gradient


class MeshResult:
    def __init__(self, verts, faces, normals, voxel, stats):
        self.verts = verts
        self.faces = faces
        self.normals = normals
        self.voxel = voxel
        self.stats = stats


def choose_voxel(parts, resolution, min_feature=None, max_cells=40_000_000):
    """Pick a voxel size: ``resolution`` cells along the largest dimension,
    refined so ``min_feature`` (a radius) spans at least ~2 voxels."""
    lo, hi = union_bounds(parts)
    size = hi - lo
    voxel = float(size.max()) / float(resolution)
    if min_feature:
        voxel = min(voxel, float(min_feature) / 1.6)
    # never exceed the dense grid budget
    cells = np.prod(np.ceil(size / voxel) + 5)
    if cells > max_cells:
        voxel *= (cells / max_cells) ** (1.0 / 3.0)
    return voxel


def _fine_field(parts, lo, nb, B, voxel, safety=1.6):
    """Narrow-band evaluation; returns dense grid (n0, n1, n2) float32."""
    nb = np.asarray(nb)
    bidx = np.indices(nb).reshape(3, -1).T
    centers = lo + (bidx * B + (B - 1) * 0.5) * voxel
    hd = math.sqrt(3.0) * (B - 1) * 0.5 * voxel
    band = hd * safety + voxel
    dc = evaluate(parts, centers, margin=band)
    active = np.abs(dc) < band
    act = bidx[active]
    M = len(act)
    off = np.indices((B, B, B)).reshape(3, -1).T.astype(np.float64) * voxel
    fb = np.full((M, B ** 3), 1e9, dtype=np.float64)
    blo = lo + act * (B * voxel)
    bhi = blo + (B - 1) * voxel
    fine_margin = 3.0 * voxel
    for fp in parts:
        pad = fp.blend + fine_margin
        sel = np.nonzero(np.all((bhi >= fp.lo - pad) & (blo <= fp.hi + pad), axis=1))[0]
        if not len(sel):
            continue
        pts = (blo[sel][:, None, :] + off[None, :, :]).reshape(-1, 3)
        d = fp.prim.distance(pts).reshape(len(sel), -1)
        cur = fb[sel]
        if fp.op == "subtract":
            fb[sel] = smax(cur, -d, fp.blend)
        elif fp.op == "intersect":
            fb[sel] = smax(cur, d, fp.blend)
        else:
            fb[sel] = smin(cur, d, fp.blend)
    big = np.float32(band * 2.0)
    grid = np.empty((nb[0], nb[1], nb[2], B, B, B), dtype=np.float32)
    grid[...] = np.where(dc.reshape(tuple(nb)) < 0, -big, big)[..., None, None, None]
    if M:
        grid[act[:, 0], act[:, 1], act[:, 2]] = np.clip(fb, -big, big).reshape(M, B, B, B)
    grid = grid.transpose(0, 3, 1, 4, 2, 5).reshape(nb[0] * B, nb[1] * B, nb[2] * B)
    # closed surfaces: the border of the grid is always outside
    grid[0, :, :] = big
    grid[-1, :, :] = big
    grid[:, 0, :] = big
    grid[:, -1, :] = big
    grid[:, :, 0] = big
    grid[:, :, -1] = big
    return grid, M, len(bidx)


_EDGES = []
for _a in range(3):
    for _u in (0, 1):
        for _v in (0, 1):
            o0 = [0, 0, 0]
            o0[(_a + 1) % 3] = _u
            o0[(_a + 2) % 3] = _v
            o1 = list(o0)
            o1[_a] = 1
            _EDGES.append((np.array(o0), np.array(o1)))


def surface_nets(grid):
    """Naive surface nets. Returns (verts in grid units, quads)."""
    inside = grid < 0
    n0, n1, n2 = grid.shape
    cnt = np.zeros((n0 - 1, n1 - 1, n2 - 1), dtype=np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cnt += inside[dx:n0 - 1 + dx, dy:n1 - 1 + dy, dz:n2 - 1 + dz]
    mixed = (cnt > 0) & (cnt < 8)
    del cnt
    cells = np.argwhere(mixed)
    del mixed
    if not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    cdims = (n0 - 1, n1 - 1, n2 - 1)
    lin = np.ravel_multi_index(cells.T, cdims)

    acc = np.zeros((len(cells), 3), dtype=np.float64)
    num = np.zeros(len(cells), dtype=np.float64)
    for o0, o1 in _EDGES:
        p0 = cells + o0
        p1 = cells + o1
        v0 = grid[p0[:, 0], p0[:, 1], p0[:, 2]].astype(np.float64)
        v1 = grid[p1[:, 0], p1[:, 1], p1[:, 2]].astype(np.float64)
        s = (v0 < 0) != (v1 < 0)
        if not s.any():
            continue
        t = v0[s] / (v0[s] - v1[s])
        acc[s] += p0[s] + t[:, None] * (o1 - o0)[None, :]
        num[s] += 1.0
    verts = acc / np.maximum(num, 1.0)[:, None]

    quads = []
    shape = np.array(grid.shape)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, shape[a] - 1)
        sl1[a] = slice(1, shape[a])
        e = inside[tuple(sl0)] != inside[tuple(sl1)]
        idx = np.argwhere(e)
        if not len(idx):
            continue
        keep = ((idx[:, b] >= 1) & (idx[:, b] <= shape[b] - 2)
                & (idx[:, c] >= 1) & (idx[:, c] <= shape[c] - 2))
        idx = idx[keep]
        q = np.empty((len(idx), 4), dtype=np.int64)
        for j, (ob, oc) in enumerate(((-1, -1), (0, -1), (0, 0), (-1, 0))):
            cc = idx.copy()
            cc[:, b] += ob
            cc[:, c] += oc
            q[:, j] = np.searchsorted(lin, np.ravel_multi_index(cc.T, cdims))
        start_inside = inside[idx[:, 0], idx[:, 1], idx[:, 2]]
        q[~start_inside] = q[~start_inside][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    return verts, quads


def _edges_from_quads(quads):
    """Undirected edges of a closed quad mesh (each interior edge appears as
    a->b in one face and b->a in the other, so keeping a < b dedupes it)."""
    e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    return e[e[:, 0] < e[:, 1]]


def _edges_from_tris(tris):
    if not len(tris):
        return np.zeros((0, 2), dtype=np.int64)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return e[e[:, 0] < e[:, 1]]


def _project(parts, V, voxel, iters=2):
    margin = 4.0 * voxel
    h = voxel * 0.1
    for _ in range(iters):
        f, g = value_and_gradient(parts, V, h, margin)
        gg = np.maximum((g * g).sum(axis=1), 1e-12)
        step = (f / gg)[:, None] * g
        ln = np.linalg.norm(step, axis=1)
        lim = np.minimum(1.0, voxel / np.maximum(ln, 1e-12))
        V = V - step * lim[:, None]
    return V


def vertex_normals(V, quads, tris=None):
    """Area-weighted vertex normals of a quad (+ tri) mesh."""
    N = np.zeros_like(V)
    if len(quads):
        n = np.cross(V[quads[:, 2]] - V[quads[:, 0]], V[quads[:, 3]] - V[quads[:, 1]])
        for k in range(4):
            np.add.at(N, quads[:, k], n)
    if tris is not None and len(tris):
        n = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
        for k in range(3):
            np.add.at(N, tris[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]


def relax(V, edges, lam=0.5):
    n = len(V)
    s = np.zeros_like(V)
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    np.add.at(s, edges[:, 0], V[edges[:, 1]])
    np.add.at(s, edges[:, 1], V[edges[:, 0]])
    avg = s / np.maximum(deg, 1.0)[:, None]
    return V + lam * (avg - V) * (deg > 0)[:, None]


def mesh_parts(parts, voxel, block=8, project_iters=1, relax_iters=1, log=None, bounds=None):
    """Mesh a list of :class:`FieldPart`. Returns :class:`MeshResult`.
    ``bounds`` (lo, hi) restricts meshing to a box (surfaces are closed at
    the box walls)."""
    t0 = time.perf_counter()
    if not isinstance(parts[0], FieldPart):
        raise TypeError("mesh_parts expects FieldPart records")
    lo, hi = union_bounds(parts)
    if bounds is not None:
        lo = np.maximum(lo, bounds[0])
        hi = np.minimum(hi, bounds[1])
        if np.any(hi <= lo):
            return MeshResult(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3)),
                              voxel, {"voxel": voxel})
    pad = 3.0 * voxel
    lo = lo - pad
    hi = hi + pad
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    B = int(block)
    nb = np.maximum(np.ceil(dims / B).astype(int), 1)
    grid, active, total = _fine_field(parts, lo, nb, B, voxel)
    t1 = time.perf_counter()
    gv, quads = surface_nets(grid)
    del grid
    t2 = time.perf_counter()
    V = lo + gv * voxel
    if len(V) and project_iters:
        V = _project(parts, V, voxel, project_iters)
        if relax_iters:
            edges = _edges_from_quads(quads)
            for _ in range(relax_iters):
                V = relax(V, edges, 0.5)
                V = _project(parts, V, voxel, 1)
    normals = vertex_normals(V, quads)
    t3 = time.perf_counter()
    stats = {
        "voxel": voxel,
        "grid": [int(x) for x in nb * B],
        "active_blocks": int(active),
        "total_blocks": int(total),
        "field_s": round(t1 - t0, 3),
        "nets_s": round(t2 - t1, 3),
        "refine_s": round(t3 - t2, 3),
    }
    if log:
        log(stats)
    return MeshResult(V, quads, normals, voxel, stats)


# ---------------------------------------------------------------------------
# adaptive meshing: finer "detail regions" zipped into a coarse mesh
# ---------------------------------------------------------------------------

def _directed_edges(quads, tris):
    e = []
    for F in (quads, tris):
        if len(F):
            k = F.shape[1]
            for i in range(k):
                e.append(np.stack([F[:, i], F[:, (i + 1) % k]], axis=1))
    return np.concatenate(e) if e else np.zeros((0, 2), dtype=np.int64)


def boundary_loops(quads, tris, nverts):
    """Closed loops of boundary edges (edges whose twin is missing)."""
    E = _directed_edges(quads, tris)
    if not len(E):
        return []
    key = E[:, 0] * nverts + E[:, 1]
    rkey = E[:, 1] * nverts + E[:, 0]
    is_b = ~np.isin(key, rkey)
    B = E[is_b]
    nxt = {}
    for a, b in B.tolist():
        nxt.setdefault(a, []).append(b)
    loops, used = [], set()
    for a, b in B.tolist():
        if (a, b) in used:
            continue
        loop = [a]
        cur_a, cur_b = a, b
        ok = True
        while True:
            used.add((cur_a, cur_b))
            if cur_b == loop[0]:
                break
            loop.append(cur_b)
            cands = [x for x in nxt.get(cur_b, []) if (cur_b, x) not in used]
            if not cands or len(loop) > len(B) + 1:
                ok = False
                break
            cur_a, cur_b = cur_b, cands[0]
        if ok and len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
        elif not ok:
            return None  # broken boundary -> caller falls back
    return loops


def _arc(P):
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return t / max(seg.sum(), 1e-12)


def _zip(V, fine, coarse):
    """Triangulate the band between two closed boundary loops.

    ``fine`` runs along its faces' boundary edges, ``coarse`` must run the
    opposite way (reverse of its faces' boundary direction) so the strip is
    consistently oriented with both meshes. Advances along whichever loop is
    behind in normalised arc length."""
    Pf = V[fine]
    j0 = int(np.argmin(np.linalg.norm(V[coarse] - Pf[0], axis=1)))
    coarse = np.roll(coarse, -j0)
    tf = _arc(Pf)
    tc = _arc(V[coarse])
    nf, nc = len(fine), len(coarse)
    tf_next = np.append(tf[1:], 1.0)
    tc_next = np.append(tc[1:], 1.0)
    i = j = 0
    tris = []
    while i < nf or j < nc:
        a, an = fine[i % nf], fine[(i + 1) % nf]
        b, bn = coarse[j % nc], coarse[(j + 1) % nc]
        adv_f = j >= nc or (i < nf and tf_next[i] <= tc_next[j])
        if adv_f:
            tris.append((an, a, b))
            i += 1
        else:
            tris.append((bn, a, b))
            j += 1
    return np.array(tris, dtype=np.int64)


def _loop_gap(V, la, lb):
    A, B = V[la], V[lb]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)) if len(A) * len(B) < 4e6 else None
    if d is None:
        return np.inf
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _components(F, nv):
    """Connected-component label per vertex (faces F: (n, k))."""
    lab = np.arange(nv)
    if not len(F):
        return lab
    k = F.shape[1]
    for _ in range(2000):
        fl = lab[F].min(axis=1)
        new = lab.copy()
        np.minimum.at(new, F.ravel(), np.repeat(fl, k))
        new = new[new]
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _manifold(Q, T, nv):
    E = _directed_edges(Q, T)
    key = E[:, 0] * nv + E[:, 1]
    if len(np.unique(key)) != len(key):
        return False
    return bool(np.isin(E[:, 1] * nv + E[:, 0], key).all())


def _seam_ok(Q, T, seam, nv):
    """Every directed edge of the seam triangles is unique and has a twin."""
    E = _directed_edges(Q, T)
    key = np.sort(E[:, 0] * nv + E[:, 1])
    se = _directed_edges(np.zeros((0, 4), dtype=np.int64), seam)
    sk = se[:, 0] * nv + se[:, 1]
    rk = se[:, 1] * nv + se[:, 0]
    cnt = np.searchsorted(key, sk, "right") - np.searchsorted(key, sk, "left")
    rcnt = np.searchsorted(key, rk, "right") - np.searchsorted(key, rk, "left")
    return bool((cnt == 1).all() and (rcnt == 1).all())


def merge_region(parts, coarse, region, log=None):
    """Replace the part of ``coarse`` (dict V, quads, tris, normals) around
    ``region`` with a finer mesh and zip the seam.

    region = dict(center, radius, voxel, plane=None | (point, normal)).
    The fine patch is the connected piece of surface containing the point
    closest to ``center``, inside ``radius`` and on the normal side of
    ``plane``. Returns the unchanged mesh if the seam cannot be zipped
    cleanly (so the result is always watertight)."""
    c = np.asarray(region["center"], dtype=np.float64)
    R = float(region["radius"])
    vf = float(region["voxel"])
    plane = region.get("plane")
    vc = coarse["voxel"]

    def skip(reason, **kw):
        if log:
            log(dict({"region": "skipped", "reason": reason}, **kw))
        return coarse

    pad = R + 4 * vf
    fine = mesh_parts(parts, vf, bounds=(c - pad, c + pad), relax_iters=1)
    if not len(fine.faces):
        return coarse
    Vf = fine.verts
    ok = np.linalg.norm(Vf - c, axis=1) < R - 0.5 * vc
    if plane is not None:
        p0, n = np.asarray(plane[0], dtype=np.float64), np.asarray(plane[1], dtype=np.float64)
        ok &= (Vf - p0) @ (n / np.linalg.norm(n)) > 0.0
    keep_fq = ok[fine.faces].all(axis=1)
    if not keep_fq.any():
        return coarse
    lab = _components(fine.faces[keep_fq], len(Vf))
    seed = np.nonzero(ok)[0][np.argmin(np.linalg.norm(Vf[ok] - c, axis=1))]
    # keep the seed's piece plus pieces that are closed (entirely inside)
    fl_all = boundary_loops(fine.faces[keep_fq], np.zeros((0, 3), dtype=np.int64), len(Vf))
    open_labs = set() if fl_all is None else {lab[l[0]] for l in fl_all}
    keep_lab = (lab[fine.faces[:, 0]] == lab[seed]) | ~np.isin(lab[fine.faces[:, 0]], list(open_labs))
    keep_fq &= keep_lab
    V, Q, T, N = coarse["V"], coarse["quads"], coarse["tris"], coarse["normals"]
    # coarse vertices lying on the kept fine patch (spatial hash) ...
    used_f = np.unique(fine.faces[keep_fq].ravel())
    cell = 0.5 * vc
    fk = np.floor(Vf[used_f] / cell).astype(np.int64)
    base = 1 << 20
    keys = np.unique((fk[:, 0] * base + fk[:, 1]) * base + fk[:, 2])
    near_c = np.zeros(len(V), dtype=bool)
    cand = np.nonzero(np.linalg.norm(V - c, axis=1) < R + 2 * vc)[0]
    if len(cand):
        cki = np.floor(V[cand] / cell).astype(np.int64)
        hit = np.zeros(len(cki), dtype=bool)
        last = len(keys) - 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    kk = ((cki[:, 0] + dx) * base + cki[:, 1] + dy) * base + cki[:, 2] + dz
                    hit |= keys[np.minimum(np.searchsorted(keys, kk), last)] == kk
        near_c[cand[hit]] = True
    if not near_c.any():
        return skip("no overlap")
    # ... grown by one ring so a seam band separates the two meshes
    grow = near_c.copy()
    for F in (Q, T):
        if len(F):
            touch = near_c[F].any(axis=1)
            grow[F[touch].ravel()] = True
    near_c = grow
    keep_cq = ~near_c[Q].any(axis=1) if len(Q) else np.zeros(0, dtype=bool)
    keep_ct = ~near_c[T].any(axis=1) if len(T) else np.zeros(0, dtype=bool)
    # drop coarse islands left inside the hole (keep components that also
    # have vertices far away from the region)
    nV = len(V)
    V2 = np.concatenate([V, Vf])
    N2 = np.concatenate([N, fine.normals])
    fq = fine.faces[keep_fq] + nV
    Qk, Tk = Q[keep_cq], T[keep_ct]
    floops = boundary_loops(fq, np.zeros((0, 3), dtype=np.int64), len(V2))
    cloops = boundary_loops(Qk, Tk, len(V2))
    if floops is None or cloops is None:
        return skip("broken boundary")
    if len(cloops) > len(floops):
        # small coarse fragments inside the hole -> remove them
        big = []
        for cl in cloops:
            if np.linalg.norm(V2[cl] - c, axis=1).max() < R + 2 * vc and len(cl) <= 8:
                continue
            big.append(cl)
        if len(big) != len(cloops):
            lab_c = _components(np.concatenate([Qk[:, :3], Tk]) if len(Tk) else Qk, len(V2))
            frag = set(lab_c[cl[0]] for cl in cloops if not any(cl is b for b in big))
            # only drop fragments that are tiny
            bad = np.isin(lab_c, list(frag))
            if bad.sum() < 64:
                Qk = Qk[~bad[Qk].any(axis=1)]
                Tk = Tk[~bad[Tk].any(axis=1)] if len(Tk) else Tk
                cloops = boundary_loops(Qk, Tk, len(V2))
    if cloops is None or len(floops) != len(cloops):
        return skip("loop count", fine_loops=len(floops), coarse_loops=0 if cloops is None else len(cloops))
    if not floops:  # closed pieces swapped wholesale (e.g. hair inside the head region)
        Q2 = np.concatenate([Qk, fq])
        used_v = np.zeros(len(V2), dtype=bool)
        used_v[Q2.ravel()] = True
        if len(Tk):
            used_v[Tk.ravel()] = True
        remap = np.cumsum(used_v) - 1
        return {"V": V2[used_v], "normals": N2[used_v], "quads": remap[Q2],
                "tris": remap[Tk] if len(Tk) else Tk, "voxel": vc}
    used = set()
    zips = []
    for fl in floops:
        cen = V2[fl].mean(axis=0)
        order = sorted(range(len(cloops)), key=lambda k: np.linalg.norm(V2[cloops[k]].mean(axis=0) - cen))
        match, gap = None, np.inf
        for k in order:
            if k in used:
                continue
            gap = _loop_gap(V2, fl, cloops[k])
            if gap < 6.0 * vc:
                match = k
                break
        if match is None:
            return skip("unmatched loop", gap=gap / vc)
        used.add(match)
        zips.append(_zip(V2, fl, cloops[match][::-1]))
    zt = np.concatenate(zips)
    # sanity: the strip must face outwards like the surface
    cen = V2[zt].mean(axis=1)
    n = np.cross(V2[zt[:, 1]] - V2[zt[:, 0]], V2[zt[:, 2]] - V2[zt[:, 0]])
    g = gradient(parts, cen, vf * 0.1, 4 * vc)
    if ((n * g).sum(axis=1) > 0).mean() < 0.9:
        return skip("seam orientation")
    Q2 = np.concatenate([Qk, fq])
    T2 = np.concatenate([Tk, zt]) if len(Tk) else zt
    if not _seam_ok(Q2, T2, zt, len(V2)):
        return skip("seam not manifold")
    # relax the seam band (zip vertices + one ring) and put it back on the surface
    band = np.zeros(len(V2), dtype=bool)
    band[zt.ravel()] = True
    for F in (Q2, T2):
        touch = band[F].any(axis=1)
        band[F[touch].ravel()] = True
    edges = np.concatenate([_edges_from_quads(Q2), _edges_from_tris(T2)])
    bi = np.nonzero(band)[0]
    for _ in range(3):
        R_ = relax(V2, edges, 0.5)
        V2[bi] = R_[bi]
        V2[bi] = _project(parts, V2[bi], vf, 1)
    used_v = np.zeros(len(V2), dtype=bool)
    used_v[Q2.ravel()] = True
    used_v[T2.ravel()] = True
    remap = np.cumsum(used_v) - 1
    return {"V": V2[used_v], "normals": N2[used_v], "quads": remap[Q2], "tris": remap[T2], "voxel": vc}


def _inside_region(parts, reg):
    """True when every additive part lies inside the region (sphere + plane)."""
    lo, hi = union_bounds(parts)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = np.asarray(reg["center"], dtype=np.float64)
    if np.linalg.norm(corners - c, axis=1).max() > reg["radius"] - reg["voxel"]:
        return False
    if reg.get("plane") is not None:
        p0, n = (np.asarray(x, dtype=np.float64) for x in reg["plane"])
        if ((corners - p0) @ n).min() <= 0:
            return False
    return True


def mesh_adaptive(parts, voxel, regions=(), relax_iters=1, log=None):
    """Coarse mesh + finer detail regions (see :func:`merge_region`)."""
    for reg in regions:
        if reg["voxel"] < voxel and _inside_region(parts, reg):
            # the whole layer sits in one region: mesh it finely in one go
            fine = mesh_parts(parts, reg["voxel"], relax_iters=1, log=log)
            return {"V": fine.verts, "quads": fine.faces, "tris": np.zeros((0, 3), dtype=np.int64),
                    "normals": fine.normals, "voxel": reg["voxel"]}
    base = mesh_parts(parts, voxel, relax_iters=relax_iters, log=log)
    mesh = {"V": base.verts, "quads": base.faces, "tris": np.zeros((0, 3), dtype=np.int64),
            "normals": base.normals, "voxel": voxel}
    for reg in regions:
        if len(mesh["V"]) == 0 or reg["voxel"] >= voxel * 0.95:
            continue
        lo, hi = union_bounds(parts)
        c = np.asarray(reg["center"])
        if np.any(c + reg["radius"] < lo) or np.any(c - reg["radius"] > hi):
            continue
        mesh = merge_region(parts, mesh, reg, log)
    return mesh
