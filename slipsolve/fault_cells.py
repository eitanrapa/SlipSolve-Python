"""Cells that follow a fault (plan Part IV 1(b), 2026-11-05): the fault's own triangles extruded horizontally.

A regular voxel grid cannot follow a curved or dipping fault, and a voxel the fault passes through cannot carry a
departure: its one strain value would straddle the discontinuity. These cells can. Each triangle of the fault mesh is
extruded along the horizontal fault normal, in layers of given thickness, on one or both sides, so the innermost cells
share the fault's triangles as faces and every cell ends on the fault or on a neighbour. Extruding horizontally keeps
each cell within its triangle's depth range, so within its layer when the mesh has element edges on the interfaces;
neighbouring cells share their faces because each fault vertex moves along ONE normal (its elements' horizontal normals
averaged by area), and a shared side quad is split along the same diagonal by both (chosen from its corners alone).

:class:`PrismSource` is one cell, a triangular prism ("wedge"): :class:`~slipsolve.heterogeneity.VoxelSource`'s
construction with its own faces (two triangles and three quads, triangulated, outward; faces in the free surface
dropped), inside test (the faces' planes) and volume rule (collapsed Gauss on the triangle times Gauss-Legendre across,
mapped with the Jacobian; the map is polynomial, so order 3 gives the volume and centroid exactly). Its far paths and
remainder are the voxel's. :class:`FaultCells` is the model: interchangeable with
:class:`~slipsolve.heterogeneity.VoxelModel` in :class:`~slipsolve.heterogeneity.HeterogeneousEngine` for the dense
solve, ``Gamma`` pair by pair (a curved fault has no translation invariance).
"""
import numpy as np

from .heterogeneity import (FAR_VOXEL, MAX_ASPECT, STRAIN_FAR_VOXEL, VOXEL_MAX_ORDER, VOXEL_TOLERANCE, VoxelSource,
                            _make_source, stiffness_block)
from .layered_tde import _gauss01, _rule_gauss

#: Most times a prism is halved (midpoint subdivision, or split across) on the way to near-cubic pieces.
MAX_SPLITS = 4


def _mapped_rule(inner, outer, order):
    """Points ``(q, 3)`` and volume weights ``(q,)`` of the prism ``inner`` -> ``outer`` (vertex i joined to vertex i)."""
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
    """The quad ``p q r s`` (in order round it) as two triangles, split on the diagonal from its least corner."""
    corners = [p, q, r, s]
    first = min(range(4), key=lambda i: tuple(np.round(corners[i], 6)))
    c = corners[first:] + corners[:first]
    return [np.array([c[0], c[1], c[2]]), np.array([c[0], c[2], c[3]])]


class PrismSource(VoxelSource):
    """A triangular prism of uniform moment density: ``inner`` and ``outer`` triangles ``(3, 3)``, vertex i to vertex i.

    East/north/up metres in the :class:`~slipsolve.frame.LocalFrame`, at or below the surface, in one layer of the
    tables. Its fields are :class:`~slipsolve.heterogeneity.VoxelSource`'s: per unit moment density in each of the six
    :func:`~slipsolve.heterogeneity.unit_moments`. A prism longer than :data:`~slipsolve.heterogeneity.MAX_ASPECT`
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
        """Outward face triangles (those in the free surface dropped) and every face's plane, for :meth:`inside`."""
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


def _prism_centroids(inner, outer):
    """Volumes ``(n,)`` and centroids ``(n, 3)`` of many prisms, exactly (the order-3 mapped rule)."""
    vols, cents = np.empty(inner.shape[0]), np.empty((inner.shape[0], 3))
    for i, (a, b) in enumerate(zip(inner, outer)):
        p, w = _mapped_rule(a, b, 3)
        vols[i] = w.sum()
        cents[i] = (p * w[:, None]).sum(axis=0) / vols[i]
    return vols, cents


class FaultCells:
    """Cells that follow a fault mesh: its triangles extruded horizontally, in layers, on one or both sides.

    ``thickness`` (metres) lists the layers' thicknesses outward from the fault; ``sides`` takes ``+1`` (the side the
    mesh's normals point into) and/or ``-1``. ``moduli`` gives ``(lam, mu)`` (Pa) at the cells: a callable
    ``f(x, y, depth, distance, side)`` of the cells' centroids, ``distance`` the centroid's distance from the fault along
    its layer (the layer's mid-distance), returning arrays. Cells whose moduli differ from the reference's by less than
    ``tol`` times its ``mu`` are inactive. The cells end on the fault by construction, so the clearance guard holds for
    the mesh they were built from (:meth:`check_clearance`).
    """

    def __init__(self, tables, mesh, thickness, moduli, sides=(-1, 1), tol=1e-9):
        self.tables = tables
        self.mesh = mesh
        self.thickness = np.asarray(thickness, dtype=float).ravel()
        if self.thickness.size == 0 or np.any(self.thickness <= 0):
            raise ValueError("thickness: one or more positive layer thicknesses")
        self.sides = tuple(int(s) for s in sides)
        if not self.sides or any(s not in (-1, 1) for s in self.sides):
            raise ValueError("sides: +1 and/or -1")
        nodes = np.asarray(mesh.nodes, dtype=float)
        tris = np.asarray(mesh.triangles)
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
        self.inner = np.concatenate(inner)
        self.outer = np.concatenate(outer)
        self.side = np.concatenate(side)
        self.layer_index = np.concatenate(layer)
        self.element = np.concatenate(element)
        self.volumes, centroids = _prism_centroids(self.inner, self.outer)
        self.distance = 0.5 * (reach[self.layer_index] + reach[self.layer_index + 1])
        depth = -centroids[:, 2]
        ref = tables.layer(depth)
        self.lam0, self.mu0 = tables.lam[ref], tables.mu[ref]
        lam, mu = moduli(centroids[:, 0], centroids[:, 1], depth, self.distance, self.side)
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
            raise ValueError(f"{len(bad)} active cells cross a layer interface (elements "
                             f"{sorted(set(self.element[bad].tolist()))[:10]}...): put the mesh's element edges on the "
                             f"interfaces {tables.tops[1:].tolist()} m")

    @property
    def n_active(self):
        return int(self.ids.size)

    def stiffness(self):
        """``D``, block-diagonal ``(6 n, 6 n)`` sparse: the moment density is ``M = -D eps``."""
        import scipy.sparse as sp

        return sp.block_diag(list(stiffness_block(self.dlam[self.active], self.dmu[self.active])), format="csr")

    def spec(self, i):
        """Active cell ``i``'s picklable description, for :func:`~slipsolve.heterogeneity._make_source`."""
        k = self.ids[i]
        return ("prism", self.inner[k], self.outer[k])

    def source(self, engine, i):
        return _make_source(engine, self.spec(i))

    def check_clearance(self, mesh, buffer=None, samples=24, contact=False):
        """The cells end on the fault they were built from; refuse any other mesh."""
        if mesh.digest() != self.mesh.digest():
            raise ValueError("these fault cells were built from another mesh")
        return True

    def on_faces(self, x, y, z=None, tol=None):
        """Surface points on a cell's vertical side faces are not detected here: the fields' finite check backs it up."""
        return np.zeros(np.size(x), bool)

    def strain_operator(self, engine, log=None, workers=None):
        """``Gamma`` pair by pair, ``(6 n, 6 n)``: each active cell's field at every active centroid."""
        import time

        from .heterogeneity import _spec_task
        from .layered_tde import engine_pool

        n = self.n_active
        out = np.zeros((6 * n, 6 * n))
        x, y, z = self.centres.T
        t0 = time.time()

        def place(j, field):                                          # field (n receivers, 6 moment b, 6 strain a)
            out[:, 6 * j:6 * j + 6] = np.swapaxes(field, 1, 2).reshape(6 * n, 6)

        pool = engine_pool(engine, workers)
        if pool is None:
            for j in range(n):
                place(j, self.source(engine, j).strain(x, y, z))
        else:
            from concurrent.futures import as_completed

            with pool:
                futures = {pool.submit(_spec_task, self.spec(j), x, y, z, True): j for j in range(n)}
                for future in as_completed(futures):
                    place(futures[future], future.result())
        if log:
            log(f"Gamma pair by pair: {n} fault cells [{time.time() - t0:.0f}s]")
        return out

    def describe(self):
        from .heterogeneity import _digest

        return {"kind": "fault_cells", "n_cells": int(self.active.size), "n_active": self.n_active,
                "thickness": self.thickness.tolist(), "sides": list(self.sides), "mesh": self.mesh.digest(),
                "digest": _digest(self.lam, self.mu, self.inner, self.outer)}

    def __repr__(self):
        return (f"<FaultCells {self.active.size} cells ({self.n_active} active) on {self.mesh.n_elements} elements, "
                f"layers {np.round(self.thickness / 1e3, 3).tolist()} km, sides {list(self.sides)}; mu "
                f"{self.mu.min() / 1e9:.2f}-{self.mu.max() / 1e9:.2f} GPa>")


__all__ = ["FaultCells", "PrismSource"]
