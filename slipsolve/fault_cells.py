"""Cells that follow a fault: the fault's own triangles extruded horizontally.

A regular voxel grid cannot follow a curved or dipping fault, and a voxel the fault passes through cannot carry a
departure: its one strain value would straddle the discontinuity. These cells can. Each triangle of the fault mesh is
extruded along the horizontal fault normal, in layers of given thickness, on one or both sides, so the innermost cells
share the fault's triangles as faces and every cell ends on the fault or on a neighbour. Extruding horizontally keeps
each cell within its triangle's depth range, so within its layer when the mesh has element edges on the interfaces;
neighbouring cells share their faces because each fault vertex moves along one normal (its elements' horizontal normals
averaged by area), and a shared side quad is split along the same diagonal by both (chosen from its corners alone).
Thinner layers refine across the fault only; `split` divides the triangles for the cells alone, so the cells can be
finer along the fault than the mesh that carries the slip. For dip-slip the cells' size along the fault is what sets
the error, for strike-slip their thickness.

`PrismSource` is one cell, a triangular prism ("wedge"): `slipsolve.heterogeneity.VoxelSource`'s
construction with its own faces (two triangles and three quads, triangulated, outward; faces in the free surface
dropped), inside test (the faces' planes) and volume rule (collapsed Gauss on the triangle times Gauss-Legendre across,
mapped with the Jacobian; the map is polynomial, so order 3 gives the volume and centroid exactly). Its far paths and
remainder are the voxel's. `FaultCells` is the model: interchangeable with
`slipsolve.heterogeneity.VoxelModel` in `slipsolve.heterogeneity.HeterogeneousEngine` for the dense
solve, `Gamma` pair by pair (a curved fault has no translation invariance).
"""
import numpy as np

from .heterogeneity import (FAR_VOXEL, MAX_ASPECT, STRAIN_FAR_VOXEL, VOXEL_MAX_ORDER, VOXEL_TOLERANCE, VoxelSource,
                            _make_source, stiffness_block)
from .layered_tde import _gauss01, _rule_gauss

#: Most times a prism is halved (midpoint subdivision, or split across) on the way to near-cubic pieces.
MAX_SPLITS = 4


def _mapped_rule(inner, outer, order):
    """Points `(q, 3)` and volume weights `(q,)` of the prism `inner` -> `outer` (vertex i joined to vertex i)."""
    a, b = inner, outer
    bary, w_tri = _rule_gauss(int(order))
    t, w_t = _gauss01(int(order))
    base_a, base_b = bary @ a, bary @ b                                         # (q, 3)
    points = (1.0 - t)[None, :, None] * base_a[:, None, :] + t[None, :, None] * base_b[:, None, :]
    d1 = (1.0 - t)[:, None] * (a[1] - a[0]) + t[:, None] * (b[1] - b[0])        # (k, 3)
    d2 = (1.0 - t)[:, None] * (a[2] - a[0]) + t[:, None] * (b[2] - b[0])
    across = bary @ (b - a)                                                     # (q, 3)
    det = np.einsum("kc,qc->qk", np.cross(d1, d2), across)
    weights = 0.5 * w_tri[:, None] * w_t[None, :] * np.abs(det)
    return points.reshape(-1, 3), weights.ravel()


def _side_triangles(p, q, r, s):
    """The quad `p q r s` (in order round it) as two triangles, split on the diagonal from its least corner."""
    corners = [p, q, r, s]
    first = min(range(4), key=lambda i: tuple(np.round(corners[i], 6)))
    c = corners[first:] + corners[:first]
    return [np.array([c[0], c[1], c[2]]), np.array([c[0], c[2], c[3]])]


class PrismSource(VoxelSource):
    """A triangular prism of uniform moment density: `inner` and `outer` triangles `(3, 3)`, vertex i to vertex i.

    East/north/up metres in the `slipsolve.frame.LocalFrame`, at or below the surface, in one layer of the
    tables. Its fields are `slipsolve.heterogeneity.VoxelSource`'s: per unit moment density in each of the six
    `slipsolve.heterogeneity.unit_moments`. A prism longer than `slipsolve.heterogeneity.MAX_ASPECT`
    times its thickness (or thicker than that times its longest edge) is summed as pieces: midpoint subdivision of its
    triangles, or layers across.
    """

    def __init__(self, engine, inner, outer, far=FAR_VOXEL, tolerance=VOXEL_TOLERANCE, max_order=VOXEL_MAX_ORDER,
                 strain_far=STRAIN_FAR_VOXEL, _splits=0):
        self.engine = engine
        tables = engine.tables
        a = np.asarray(inner, dtype=float).reshape(3, 3)
        b = np.asarray(outer, dtype=float).reshape(3, 3)
        corners = np.vstack([a, b])
        if corners[:, 2].max() > 1e-6:
            raise ValueError("a prism must lie at or below the surface (z <= 0)")
        top, bottom = max(0.0, -corners[:, 2].max()), -corners[:, 2].min()
        crossed = [t for t in tables.tops[1:] if top + 1e-6 < t < bottom - 1e-6]
        if crossed:
            raise ValueError(f"a prism must lie in one layer; its depths {top:.0f}-{bottom:.0f} m cross {crossed} m: put "
                             "the fault mesh's element edges on the interfaces")
        self.inner, self.outer = a, b
        self.layer = int(tables.layer(0.5 * (top + bottom)))
        self.lam = float(tables.lam[self.layer])
        self.mu = float(tables.mu[self.layer])
        self.nu = self.lam / (2.0 * (self.lam + self.mu))
        self.lo, self.hi = corners.min(axis=0), corners.max(axis=0)
        points, weights = _mapped_rule(a, b, 3)
        self.volume = float(weights.sum())
        if not self.volume > 0.0:
            raise ValueError("a degenerate prism (zero volume)")
        self.centre = (points * weights[:, None]).sum(axis=0) / self.volume
        self.size = self.volume ** (1.0 / 3.0)
        self.free_surface = bool(np.isclose(top, 0.0))
        self.faces, self._planes = self._make_faces()
        self.far = None if far is None or not tables.has_surface else float(far)
        self.strain_far = None if strain_far is None else float(strain_far)
        self.tolerance = float(tolerance)
        self.max_order = int(max_order)
        from .heterogeneity import compliance, moment_sources, tensor_packing, unit_moments

        self.eigenstrain = compliance(unit_moments(), self.lam, self.mu)
        self.packing = tensor_packing(unit_moments(), self.lam, self.mu)
        self.sources = moment_sources(self.lam, self.mu)
        self.pieces = self._split(engine, far, tolerance, max_order, strain_far, _splits)

    # -- geometry -------------------------------------------------------------------
    def _make_faces(self):
        """Outward face triangles (those in the free surface dropped) and every face's plane, for `inside`."""
        a, b = self.inner, self.outer
        candidates = [np.array(a), np.array(b)]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            candidates += _side_triangles(a[i], a[j], b[j], b[i])
        faces, planes = [], []
        for tri in candidates:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.linalg.norm(n) < 1e-12 * max(1.0, np.abs(tri).max()) ** 2:
                continue
            if n @ (tri.mean(axis=0) - self.centre) < 0.0:
                tri, n = tri[[0, 2, 1]], -n
            surface = bool(np.all(np.abs(tri[:, 2]) < 1e-6))
            planes.append((n / np.linalg.norm(n), tri[0].copy(), surface))
            if not surface:
                faces.append(tri)
        return np.array(faces), planes

    def inside(self, x, y, z):
        """Points strictly inside the prism; on the free surface over a surface prism, inside."""
        from .greens import _as_points

        x, y, z = _as_points(x, y, z)
        p = np.column_stack([x, y, z])
        scale = 1e-9 * max(1.0, self.size)
        out = np.ones(x.size, bool)
        for n, origin, surface in self._planes:
            d = (p - origin) @ n
            out &= (d <= scale) if surface else (d < -scale)
        return out

    def _rule(self, order):
        return _mapped_rule(self.inner, self.outer, order)

    def _split(self, engine, far, tolerance, max_order, strain_far, splits):
        """Near-cubic pieces (None when this prism is one): halve the triangles or the thickness, then recurse."""
        if splits >= MAX_SPLITS:
            return None
        a, b = self.inner, self.outer
        longest = max(np.linalg.norm(a[(i + 1) % 3] - a[i]) for i in range(3))
        thickness = float(np.linalg.norm(b - a, axis=1).mean())
        kw = dict(far=far, tolerance=tolerance, max_order=max_order, strain_far=strain_far, _splits=splits + 1)
        if longest > MAX_ASPECT * thickness:                     # flat: four sub-prisms by midpoints
            ma = [(a[i] + a[(i + 1) % 3]) / 2 for i in range(3)]
            mb = [(b[i] + b[(i + 1) % 3]) / 2 for i in range(3)]
            subs = [((a[0], ma[0], ma[2]), (b[0], mb[0], mb[2])), ((ma[0], a[1], ma[1]), (mb[0], b[1], mb[1])),
                    ((ma[2], ma[1], a[2]), (mb[2], mb[1], b[2])), ((ma[0], ma[1], ma[2]), (mb[0], mb[1], mb[2]))]
            return [PrismSource(engine, np.array(i), np.array(o), **kw) for i, o in subs]
        if thickness > MAX_ASPECT * longest:                     # tall: layers across
            m = int(np.ceil(thickness / (MAX_ASPECT * longest)))
            t = np.linspace(0.0, 1.0, m + 1)
            return [PrismSource(engine, (1 - t[k]) * a + t[k] * b, (1 - t[k + 1]) * a + t[k + 1] * b, **kw)
                    for k in range(m)]
        return None

    def __repr__(self):
        return (f"<PrismSource centre ({self.centre[0]:.0f}, {self.centre[1]:.0f}, {self.centre[2]:.0f}) m, volume "
                f"{self.volume / 1e9:.3g} km^3, layer {self.layer}, mu {self.mu / 1e9:.2f} GPa"
                f"{'' if self.pieces is None else f', {len(self.pieces)} pieces'}>")


def _leaf_prisms(a, b, splits=0):
    """The pieces `PrismSource` computes a prism `a -> b` as, by its own rule, as `[(inner, outer), ...]`:
    geometry only, for the face guard (a flat cell's midpoint pieces have faces of their own inside it)."""
    if splits >= MAX_SPLITS:
        return [(a, b)]
    longest = max(np.linalg.norm(a[(i + 1) % 3] - a[i]) for i in range(3))
    thickness = float(np.linalg.norm(b - a, axis=1).mean())
    if longest > MAX_ASPECT * thickness:
        ma = [(a[i] + a[(i + 1) % 3]) / 2 for i in range(3)]
        mb = [(b[i] + b[(i + 1) % 3]) / 2 for i in range(3)]
        subs = [((a[0], ma[0], ma[2]), (b[0], mb[0], mb[2])), ((ma[0], a[1], ma[1]), (mb[0], b[1], mb[1])),
                ((ma[2], ma[1], a[2]), (mb[2], mb[1], b[2])), ((ma[0], ma[1], ma[2]), (mb[0], mb[1], mb[2]))]
        return [leaf for i, o in subs for leaf in _leaf_prisms(np.array(i), np.array(o), splits + 1)]
    if thickness > MAX_ASPECT * longest:
        m = int(np.ceil(thickness / (MAX_ASPECT * longest)))
        t = np.linspace(0.0, 1.0, m + 1)
        return [leaf for k in range(m) for leaf in _leaf_prisms((1 - t[k]) * a + t[k] * b,
                                                                (1 - t[k + 1]) * a + t[k + 1] * b, splits + 1)]
    return [(a, b)]


def _prism_centroids(inner, outer):
    """Volumes `(n,)` and centroids `(n, 3)` of many prisms, exactly (the order-3 mapped rule)."""
    vols, cents = np.empty(inner.shape[0]), np.empty((inner.shape[0], 3))
    for i, (a, b) in enumerate(zip(inner, outer)):
        p, w = _mapped_rule(a, b, 3)
        vols[i] = w.sum()
        cents[i] = (p * w[:, None]).sum(axis=0) / vols[i]
    return vols, cents


def _midpoint_weights(level):
    """A triangle's midpoint subdivision, `level` times: barycentric weights `(4**level, 3, 3)` (child, corner,
    parent vertex), each child wound as its parent, in `PrismSource`'s order."""
    w = np.eye(3)[None]
    for _ in range(int(level)):
        m = 0.5 * (w + np.roll(w, -1, axis=1))                                 # the midpoints of edges 01, 12, 20
        w = np.concatenate([np.stack([w[:, 0], m[:, 0], m[:, 2]], axis=1),
                            np.stack([m[:, 0], w[:, 1], m[:, 1]], axis=1),
                            np.stack([m[:, 2], m[:, 1], w[:, 2]], axis=1), m])
    return w


def _split_prisms(inner, outer, levels):
    """Each prism `inner[i]` -> `outer[i]` divided `levels[i]` times: the children's `inner` and `outer` and
    the index of each one's parent, a parent's children together and the parents in order.

    Inner and outer triangles get the same weights, so a child is its parent's map (`_mapped_rule`) restricted to
    a part of the triangle: volumes and centroids add up exactly, also where a side quad twists."""
    counts = 4 ** levels
    parent = np.repeat(np.arange(levels.size), counts)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    w = np.empty((parent.size, 3, 3))
    for level in np.unique(levels):
        weights = _midpoint_weights(level)
        w[first[levels == level][:, None] + np.arange(weights.shape[0])] = weights
    return w @ inner[parent], w @ outer[parent], parent


def _split_levels(split, n):
    """`split` as one level for each of `n` elements; a mask means level 1 where it is set."""
    levels = np.asarray(split)
    if (not (np.issubdtype(levels.dtype, np.integer) or levels.dtype == bool) or np.any(levels < 0)
            or levels.shape not in ((), (n,))):
        raise ValueError(f"split: a non-negative integer, or one for each of the {n} fault elements")
    return np.broadcast_to(levels, (n,)).astype(int)


class PrismCells:
    """Cells of uniform departure, each a triangular prism `inner` -> `outer` (`(n, 3, 3)` each, vertex i to i).

    `lam` and `mu` (Pa, `(n,)` or scalars) are the cells' moduli; a cell whose moduli match the reference's to
    `tol` times its `mu` is inactive. A model for `slipsolve.heterogeneity.HeterogeneousEngine`'s dense solve
    (`Gamma` pair by pair) and for `CombinedCells`. `FaultCells` builds one from a fault mesh.
    """

    def __init__(self, tables, inner, outer, lam, mu, tol=1e-9, _centroids=None):
        self.tables = tables
        self.inner = np.asarray(inner, dtype=float).reshape(-1, 3, 3)
        self.outer = np.asarray(outer, dtype=float).reshape(-1, 3, 3)
        if self.inner.shape != self.outer.shape:
            raise ValueError("inner and outer must hold the same number of triangles")
        self.volumes, centroids = _centroids if _centroids is not None else _prism_centroids(self.inner, self.outer)
        depth = -centroids[:, 2]
        ref = tables.layer(depth)
        self.lam0, self.mu0 = tables.lam[ref], tables.mu[ref]
        self.lam = np.broadcast_to(np.asarray(lam, dtype=float), depth.shape).copy()
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), depth.shape).copy()
        self.dlam, self.dmu = self.lam - self.lam0, self.mu - self.mu0
        self.active = (np.abs(self.dlam) + np.abs(self.dmu)) > float(tol) * self.mu0
        self._all_centroids = centroids
        self.ids = np.flatnonzero(self.active)
        #: East/north/up centroids of the active cells, in the operators' order.
        self.centres = centroids[self.active]
        top = np.maximum(0.0, -np.maximum(self.inner[:, :, 2].max(1), self.outer[:, :, 2].max(1)))
        bottom = -np.minimum(self.inner[:, :, 2].min(1), self.outer[:, :, 2].min(1))
        bad = [i for i in self.ids if any(top[i] + 1e-6 < t < bottom[i] - 1e-6 for t in tables.tops[1:])]
        if bad:
            raise ValueError(f"{len(bad)} active cells cross a layer interface (cells {bad[:10]}...): put the cells' "
                             f"edges on the interfaces {tables.tops[1:].tolist()} m")

    @property
    def n_active(self):
        return int(self.ids.size)

    def stiffness(self):
        """`D`, block-diagonal `(6 n, 6 n)` sparse: the moment density is `M = -D eps`."""
        import scipy.sparse as sp

        return sp.block_diag(list(stiffness_block(self.dlam[self.active], self.dmu[self.active])), format="csr")

    def spec(self, i):
        """Active cell `i`'s picklable description, for `slipsolve.heterogeneity._make_source`."""
        k = self.ids[i]
        return ("prism", self.inner[k], self.outer[k])

    def source(self, engine, i):
        return _make_source(engine, self.spec(i))

    def contains(self, points):
        """Which `points` `(m, 3)` lie strictly inside an active cell (a convex-cell test on each cell's faces)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.zeros(points.shape[0], bool)
        for k in self.ids:
            a, b = self.inner[k], self.outer[k]
            lo, hi = np.minimum(a.min(0), b.min(0)), np.maximum(a.max(0), b.max(0))
            near = np.flatnonzero(np.all((points > lo) & (points < hi), axis=1) & ~out)
            if not near.size:
                continue
            centre = self._all_centroids[k]
            inside = np.ones(near.size, bool)
            candidates = [a, b]
            for i, j in ((0, 1), (1, 2), (2, 0)):
                candidates += _side_triangles(a[i], a[j], b[j], b[i])
            for tri in candidates:
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if np.linalg.norm(n) == 0.0:
                    continue
                if n @ (tri.mean(axis=0) - centre) < 0.0:
                    n = -n
                inside &= (points[near] - tri[0]) @ n < -1e-9 * np.linalg.norm(n) * max(1.0, np.abs(tri).max())
            out[near[inside]] = True
        return out

    def check_clearance(self, mesh, buffer=None, samples=24, contact=False):
        """Refuse a fault triangle that passes through an active cell (sampled `samples` times an edge)."""
        n = int(samples)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        bary = np.column_stack([i[keep], j[keep], n - i[keep] - j[keep]]) / n
        points = np.concatenate([bary @ tri for tri in np.asarray(mesh.vertices)])
        hit = self.contains(points)
        if hit.any():
            raise ValueError(f"the fault passes through an active cell (at {points[np.argmax(hit)].round(0).tolist()})")
        return True

    def on_faces(self, x, y, z=None, tol=None):
        """Surface points on a face of an active cell or of one of the pieces it is computed as (`_leaf_prisms`):
        within `tol` (default 1e-6 of the cell's size) of an edge or corner in the surface plane, where the field
        is singular. Points at depth are not tested."""
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        z = np.zeros(x.size) if z is None else np.asarray(z, dtype=float).ravel()
        out = np.zeros(x.size, bool)
        surface = np.flatnonzero(z == 0.0)
        if not surface.size or not self.n_active:
            return out
        p = np.column_stack([x[surface], y[surface]])
        for k in self.ids:
            if not (np.abs(np.r_[self.inner[k][:, 2], self.outer[k][:, 2]]) < 1e-6).any():
                continue
            size = float(self.volumes[k]) ** (1.0 / 3.0)
            eps = (1e-6 * size) if tol is None else float(tol)
            hit = np.zeros(p.shape[0], bool)
            for a3, b3 in _leaf_prisms(self.inner[k], self.outer[k]):
                corners = np.vstack([a3, b3])
                at = np.abs(corners[:, 2]) < 1e-6
                pts = corners[at, :2]
                hit |= (np.linalg.norm(p[:, None, :] - pts[None], axis=2) < eps).any(axis=1)
                for i in range(len(pts)):                   # every pair of surface corners spans a candidate edge
                    for j in range(i + 1, len(pts)):
                        u, v = pts[i], pts[j]
                        d = v - u
                        t = np.clip(((p - u) @ d) / max(d @ d, 1e-30), 0.0, 1.0)
                        hit |= np.linalg.norm(p - (u + t[:, None] * d), axis=1) < eps
            out[surface[hit]] = True
        return out

    def strain_operator(self, engine, log=None, workers=None):
        """`Gamma` pair by pair, `(6 n, 6 n)`: each active cell's field at every active centroid."""
        return _pairwise_gamma([self], engine, log=log, workers=workers)

    def describe(self):
        from .heterogeneity import _digest

        return {"kind": "prism_cells", "n_cells": int(self.active.size), "n_active": self.n_active,
                "digest": _digest(self.lam, self.mu, self.inner, self.outer)}

    def __repr__(self):
        return (f"<{type(self).__name__} {self.active.size} cells ({self.n_active} active); mu "
                f"{self.mu.min() / 1e9:.2f}-{self.mu.max() / 1e9:.2f} GPa>")


class FaultCells(PrismCells):
    """Cells that follow a fault mesh: its triangles extruded horizontally, in layers, on one or both sides.

    `thickness` (metres) lists the layers' thicknesses outward from the fault; `sides` takes `+1` (the side the
    mesh's normals point into) and/or `-1`. `moduli` gives `(lam, mu)` (Pa) at the cells: a callable
    `f(x, y, depth, distance, side)` of the cells' centroids, `distance` the centroid's distance from the fault along
    its layer (the layer's mid-distance), returning arrays. Cells whose moduli differ from the reference's by less than
    `tol` times its `mu` are inactive. The cells end on the fault by construction, so the clearance guard holds for
    the mesh they were built from (`check_clearance`).

    `split` makes the cells finer than that mesh, which otherwise sets their extent along the fault: each triangle is
    divided at its edges' midpoints that many times for the cells alone, `4**split` cells a triangle and layer. A
    non-negative integer, or one per element (a mask means 1 where set; 0 keeps a triangle whole). The children partition
    their parent on the same fault and each has its own strain value, unlike the pieces of a flat `PrismSource`,
    which sum one cell's field for one value. `moduli` is asked at the children. `Gamma` goes pair by pair, so a
    uniform split costs ~16x a level: split where the field needs it. Elements at different levels meet at hanging
    corners, which a cell's field does not see (it comes from its own faces alone; where a side quad twists, on a curved
    fault, the two sides' diagonals differ by the twist). A child can still cross a layer interface, which is refused.
    """

    def __init__(self, tables, mesh, thickness, moduli, sides=(-1, 1), tol=1e-9, split=0):
        self.mesh = mesh
        self.thickness = np.asarray(thickness, dtype=float).ravel()
        if self.thickness.size == 0 or np.any(self.thickness <= 0):
            raise ValueError("thickness: one or more positive layer thicknesses")
        self.sides = tuple(int(s) for s in sides)
        if not self.sides or any(s not in (-1, 1) for s in self.sides):
            raise ValueError("sides: +1 and/or -1")
        nodes = np.asarray(mesh.nodes, dtype=float)
        tris = np.asarray(mesh.triangles)
        self.split = _split_levels(split, tris.shape[0])
        normals = np.asarray(mesh.normals, dtype=float)
        horizontal = normals.copy()
        horizontal[:, 2] = 0.0
        length = np.linalg.norm(horizontal, axis=1)
        if np.any(length < 0.2):
            raise ValueError("fault cells need a steep fault: an element dips less than ~12 degrees")
        weight = np.asarray(mesh.areas, dtype=float)[:, None] * horizontal / length[:, None]
        vertex = np.zeros_like(nodes)
        np.add.at(vertex, tris.ravel(), np.repeat(weight, 3, axis=0))
        vertex /= np.linalg.norm(vertex, axis=1, keepdims=True)
        self.vertex_normals = vertex
        reach = np.concatenate([[0.0], np.cumsum(self.thickness)])
        inner, outer, side, layer, element = [], [], [], [], []
        for s in self.sides:
            for k in range(self.thickness.size):
                inner.append(nodes[tris] + s * reach[k] * vertex[tris])
                outer.append(nodes[tris] + s * reach[k + 1] * vertex[tris])
                side.append(np.full(tris.shape[0], s))
                layer.append(np.full(tris.shape[0], k))
                element.append(np.arange(tris.shape[0]))
        inner, outer = np.concatenate(inner), np.concatenate(outer)
        self.side = np.concatenate(side)
        self.layer_index = np.concatenate(layer)
        self.element = np.concatenate(element)
        if self.split.any():
            inner, outer, parent = _split_prisms(inner, outer, self.split[self.element])
            self.side, self.layer_index, self.element = self.side[parent], self.layer_index[parent], self.element[parent]
        volumes, centroids = _prism_centroids(inner, outer)
        self.distance = 0.5 * (reach[self.layer_index] + reach[self.layer_index + 1])
        lam, mu = moduli(centroids[:, 0], centroids[:, 1], -centroids[:, 2], self.distance, self.side)
        try:
            super().__init__(tables, inner, outer, lam, mu, tol=tol, _centroids=(volumes, centroids))
        except ValueError as err:
            if "cross a layer interface" not in str(err):
                raise
            raise ValueError(str(err).replace("cells' edges", "fault mesh's element edges")) from None

    def check_clearance(self, mesh, buffer=None, samples=24, contact=False):
        """The cells end on the fault they were built from; refuse any other mesh."""
        if mesh.digest() != self.mesh.digest():
            raise ValueError("these fault cells were built from another mesh")
        return True

    def describe(self):
        out = super().describe()
        out.update(kind="fault_cells", thickness=self.thickness.tolist(), sides=list(self.sides),
                   mesh=self.mesh.digest())
        if self.split.any():
            out["split"] = np.bincount(self.split).tolist()                     # elements at each level
        return out

    def __repr__(self):
        return (f"<FaultCells {self.active.size} cells ({self.n_active} active) on {self.mesh.n_elements} elements, "
                f"layers {np.round(self.thickness / 1e3, 3).tolist()} km, sides {list(self.sides)}"
                f"{f', split to level {self.split.max()}' if self.split.any() else ''}; mu "
                f"{self.mu.min() / 1e9:.2f}-{self.mu.max() / 1e9:.2f} GPa>")


def _pairwise_gamma(models, engine, log=None, workers=None, blocks=None):
    """`Gamma` of the cells of `models` (in order), pair by pair; `blocks` maps (a, b) to a precomputed block."""
    import time

    from .heterogeneity import _spec_task
    from .layered_tde import engine_pool

    sizes = [m.n_active for m in models]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    n = int(starts[-1])
    out = np.zeros((6 * n, 6 * n))
    centres = np.concatenate([m.centres for m in models]) if n else np.zeros((0, 3))
    blocks = blocks or {}
    for (a, b), block in blocks.items():
        out[6 * starts[b]:6 * starts[b + 1], 6 * starts[a]:6 * starts[a + 1]] = block
    jobs = []                                                   # (model a, cell j, receiver rows)
    for a, model in enumerate(models):
        rows = np.concatenate([np.arange(starts[b], starts[b + 1]) for b in range(len(models)) if (a, b) not in blocks]
                              + [np.zeros(0, int)]).astype(int)
        if rows.size:
            jobs += [(a, j, rows) for j in range(model.n_active)]
    t0 = time.time()

    def place(job, field):                                      # field (receivers, 6 moment b, 6 strain a)
        a, j, rows = job
        col = 6 * (starts[a] + j)
        r6 = (6 * rows[:, None] + np.arange(6)[None, :]).ravel()
        out[r6, col:col + 6] = np.swapaxes(field, 1, 2).reshape(-1, 6)

    pool = engine_pool(engine, workers)
    if pool is None:
        for job in jobs:
            a, j, rows = job
            x, y, z = centres[rows].T
            place(job, models[a].source(engine, j).strain(x, y, z))
    else:
        from .layered_tde import pool_map

        def tasks():
            for k, (a, j, rows) in enumerate(jobs):
                x, y, z = centres[rows].T
                yield k, (models[a].spec(j), x, y, z, True)

        with pool:
            for k, field in pool_map(pool, _spec_task, tasks()):
                place(jobs[k], field)
    if log:
        log(f"Gamma pair by pair: {n} cells in {len(models)} model(s), {len(jobs)} sources [{time.time() - t0:.0f}s]")
    return out


class CombinedCells:
    """Several cell models in one solve: cells beside the fault and voxels beyond, say.

    `models` are `slipsolve.heterogeneity.VoxelModel`, `PrismCells` / `FaultCells` or other
    models with the same interface, on one set of tables. Each model's own block of `Gamma` is its own operator (by
    translation for a voxel grid); the blocks between models go pair by pair. The models must not overlap: a voxel centre
    inside an active prism cell, or a prism centroid inside an active voxel, is refused (a voxel only partly inside a cell
    is not detected: put the voxels' faces on the cells' outer edge, or clear them, `VoxelModel.keep_clear`).
    Dense: for the FFT solver use one `slipsolve.heterogeneity.VoxelModel`.
    """

    def __init__(self, *models):
        if not models:
            raise ValueError("CombinedCells needs at least one model")
        self.models = list(models)
        self.tables = models[0].tables
        if any(m.tables is not self.tables for m in self.models):
            raise ValueError("every model must share one set of tables")
        from .heterogeneity import VoxelModel

        prisms = [m for m in self.models if isinstance(m, PrismCells)]
        boxes = [m for m in self.models if isinstance(m, VoxelModel)]
        for p in prisms:
            for v in boxes:
                if v.n_active and p.contains(v.centres).any():
                    raise ValueError("a voxel centre lies inside a prism cell: the models overlap")
                if p.n_active and _in_active_voxels(v, p.centres).any():
                    raise ValueError("a prism cell's centroid lies inside an active voxel: the models overlap")
        self._starts = np.concatenate([[0], np.cumsum([m.n_active for m in self.models])]).astype(int)

    @property
    def n_active(self):
        return int(self._starts[-1])

    @property
    def centres(self):
        return np.concatenate([m.centres for m in self.models]) if self.n_active else np.zeros((0, 3))

    def stiffness(self):
        import scipy.sparse as sp

        return sp.block_diag([m.stiffness() for m in self.models if m.n_active], format="csr")

    def _locate(self, i):
        a = int(np.searchsorted(self._starts, i, side="right") - 1)
        return self.models[a], i - self._starts[a]

    def spec(self, i):
        model, j = self._locate(i)
        return model.spec(j)

    def source(self, engine, i):
        return _make_source(engine, self.spec(i))

    def on_faces(self, x, y, z=None, tol=None):
        out = np.zeros(np.size(x), bool)
        for m in self.models:
            if m.n_active:
                out |= m.on_faces(x, y, z) if tol is None else m.on_faces(x, y, z, tol=tol)
        return out

    def check_clearance(self, mesh, buffer=None, samples=24, contact=False):
        for m in self.models:
            if m.n_active:
                m.check_clearance(mesh, buffer=buffer, samples=samples, contact=contact)
        return True

    def strain_operator(self, engine, log=None, workers=None):
        """`Gamma`: each model's own block by its own operator, the blocks between models pair by pair."""
        from .heterogeneity import strain_operator

        live = [m for m in self.models if m.n_active]
        own = {}
        for k, m in enumerate(live):
            own[(k, k)] = (m.strain_operator(engine, log=log, workers=workers) if hasattr(m, "strain_operator")
                           else strain_operator(m, engine, log=log, workers=workers))
        return _pairwise_gamma(live, engine, log=log, workers=workers, blocks=own)

    def describe(self):
        parts = []
        for m in self.models:
            if hasattr(m, "describe"):
                parts.append(m.describe())
            else:
                parts.append({"kind": "voxels", "shape": list(m.shape), "n_active": m.n_active})
        return {"kind": "combined", "n_active": self.n_active, "parts": parts}

    def __repr__(self):
        return f"<CombinedCells {self.n_active} active cells: {', '.join(repr(m) for m in self.models)}>"


def _in_active_voxels(voxels, points):
    """Which points lie inside an active voxel of a `slipsolve.heterogeneity.VoxelModel`."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    ix = np.searchsorted(voxels.x_edges, points[:, 0]) - 1
    iy = np.searchsorted(voxels.y_edges, points[:, 1]) - 1
    iz = np.searchsorted(voxels.z_edges, -points[:, 2]) - 1
    nx, ny, nz = voxels.shape
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    out = np.zeros(points.shape[0], bool)
    out[inside] = voxels.active[ix[inside], iy[inside], iz[inside]]
    return out


__all__ = ["CombinedCells", "FaultCells", "PrismCells", "PrismSource"]
