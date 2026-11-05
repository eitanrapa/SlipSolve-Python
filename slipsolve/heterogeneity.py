"""Fully 3-D elastic heterogeneity: any ``lambda(x)``, ``mu(x)`` on the layered reference.

The crust is the layered reference of a set of remainder tables
(:class:`~slipsolve.wavenumber.RemainderTables`, the moduli ``C0(z)`` that
:class:`~slipsolve.layered_tde.LayeredTDE` already solves exactly) plus a departure
``dC(x)`` held on voxels. The true field is the reference field of the fault plus the
reference field of an equivalent moment density in the voxels::

    M = -dC : eps,        eps = eps0 + Gamma M,        u = u0 + H M,

with ``eps`` the true strain, ``eps0`` the reference strain of the fault's slip, ``Gamma``
the reference medium's moment-to-strain operator and ``H`` its moment-to-surface
operator. This is Barbot, Fialko & Sandwell (2009, GJI 179, 500-520)'s decomposition
with ``C = C0 - C'`` and moment density ``C' : eps``; their equivalent surface traction
(their eq. 21) is not a separate term here, because a moment density's field in a
medium whose own Green's function is traction-free at the surface already carries it
(integrate the body force and the surface traction by parts: the surface terms cancel).
Solving ``(I + Gamma dC) eps = eps0`` gives the Green's matrix of the heterogeneous crust,
``G = G0 - H dC (I + Gamma dC)^-1 E``, with ``E`` the strain of each slip basis function
at the voxels (:meth:`~slipsolve.layered_tde.LayeredTDE.strain`).

**One voxel's field** (:class:`VoxelSource`) is split as ``LayeredTDE`` splits an
element's. The half-space part, with the voxel's own layer's constants, is Eshelby's
cut and weld: a uniform eigenstrain ``eps* = C0^-1 : M`` in the cuboid has, outside it,
the field of a dislocation on its faces with Burgers vector ``eps* . (xi - x0)`` (the
outward side minus the inside, the triangular dislocation's convention), and inside it
also ``eps* . (x - x0)``; a face in the free surface radiates nothing and is dropped. The
Burgers vector is linear on each face triangle, so it is three vertex tents, each
integrated exactly by :func:`~slipsolve.layered_tde.tent_levelset`. The remainder is a
volume rule of the tables' point moment sources -- not face dislocations, which on a
voxel whose face lies on a layer interface would sit where the remainder is singular.
Away from the voxel both parts are smooth over it, and its strain is a Gauss-Legendre volume
rule of whole point moments: Okada (1992) in the voxel's half-space plus the remainder at
each point (:data:`STRAIN_FAR_VOXEL`, 2026-11-05; 7-28x the faces' speed at 2-40 sizes and
as accurate, the faces' cancellation costing them ~1e-3 beyond 10 sizes). Surface receivers
far away take point sources of the whole tabulated surface field (:data:`FAR_VOXEL`).

**Limits, as built (2026-11-04).**

- A voxel's strain is collocated at its centre and taken as uniform, so voxels keep at
  least a voxel clear of the fault (:meth:`VoxelModel.check_clearance`); a contrast across
  the fault itself is :mod:`slipsolve.bimaterial`'s.
- The grid is regular: ``Gamma`` is assembled by translation, one source per depth row.
- A receiver exactly on a voxel face is singular (NaN on the face's edges, one side's limit
  inside it): :func:`los_operator` refuses one, naming it (:meth:`VoxelModel.on_faces`,
  2026-11-04), and refuses any non-finite field. Receivers within a few percent of a size of
  a face are resolved by the split level integral (:func:`~slipsolve.layered_tde.tent_levelset`).
- A voxel longer than :data:`MAX_ASPECT` times its shortest side is summed as near-cubic pieces.
- The layered remainder is table-limited, at second order in the tables' grids (strain is
  more demanding than displacement); the strain-bearing tables want ~200 m in depth.
- ``Gamma`` and its LU are dense: ~10k voxels on 128 GB, ~5-6k on 64 GB.

The design, the tests and the measurements are in ``HETEROGENEOUS_3D_PLAN.local.md``.
"""

import numpy as np

from ._okada92 import point_source_enu
from ._tde import STRAIN_COMPONENTS, _element_basis, tde_disp_hs_unit, tde_strain_hs_unit
from .greens import _as_points
from .layered import _QUADRATURE_CONSTANT, _QUADRATURE_SAFETY
from .layered_tde import (STRAIN_TENT_GAIN, STRAIN_TENT_MIN_POINTS, _combine,
                          _gauss01, combine_strain, tensor_packing, tent_levelset)

#: Surface receivers farther than this many voxel sizes (box distance) take point sources of
#: the whole tabulated surface field, as :data:`~slipsolve.layered_tde.FAR` does for elements.
#: Was 4 until 2026-11-05: on the Venezuela CRUST1.0 grid (32,577 voxels, 4 km, 17,034
#: samples) the 1% of pairs inside 4 sizes took 96% of ``H`` (the faces, 2.5 ms a pair against
#: 0.04 ms for 64 point sources). Measured the same day against a 10-point rule and the faces,
#: surface and buried cubes and a 2 x 2 x 1.5 km surface voxel: 4 points an axis are 3-4e-4 off
#: the 10-point rule from 1.5 sizes out and 0.6-1e-3 off the faces at 1.5-4 sizes, as the
#: 3-point rule already was at 4-6 sizes (the tables' own interpolation, ~1e-4-5e-4, is the
#: floor of every point rule).
FAR_VOXEL = 2.0

#: Receivers farther than this many voxel sizes (box distance) take the voxel's STRAIN from a
#: Gauss-Legendre volume rule of point moments: Okada (1992) point sources in the voxel's own
#: half-space (:func:`~slipsolve._okada92.point_source_enu`) plus the tabulated remainder, both at
#: every point (:meth:`VoxelSource._point_moments`). As ``strain_far`` does for elements.
STRAIN_FAR_VOXEL = 2.0

#: Relative accuracy that volume rule aims for (:data:`~slipsolve.layered_tde.STRAIN_FAR_TOLERANCE`).
VOXEL_FAR_TOLERANCE = 1e-3

#: Its error model: ``n`` Gauss-Legendre points a side leave about ``C (a / 2R)**(2 n)`` of the
#: strain at ``R`` from the centre of a voxel of longest side ``a`` (the 1-D rule's error for a
#: kernel singular ``R`` away); ``C`` measured (:meth:`VoxelSource._far_orders`).
VOXEL_FAR_CONSTANT = 8.0

#: Most (receiver, point) pairs the volume rule handles at once.
VOXEL_FAR_BATCH = 400_000

#: Voxels in a block of ``H`` when :class:`HeterogeneousEngine` streams it (``solver="fft"``):
#: 17k samples x 6 x 1000 voxels is 0.8 GB a block.
H_CHUNK = 1000

#: Relative accuracy the remainder's volume rule aims for, as
#: :data:`~slipsolve.layered_tde.REMAINDER_TOLERANCE`.
VOXEL_TOLERANCE = 3e-3

#: Most Gauss-Legendre points per axis of the remainder's volume rule.
VOXEL_MAX_ORDER = 12

#: Most a voxel's longest side may exceed its shortest before its field is computed as the sum
#: of near-cubic pieces (an odd number along each long side, so its centre stays a piece's
#: centre, never on an internal face). Uniform eigenstrain in the pieces sums to the voxel's
#: exactly; what changes is the quadrature. Its faces are two triangles each and its volume
#: rule takes the same number of points along every side, so a long thin voxel is
#: under-resolved near itself: measured (2026-11-04, uniform crust) at the centres of its
#: neighbours, a 7.84 x 1 x 1 km voxel was 3.8% of the peak off 16 cubes' sum and a
#: 7.84 x 0.5 x 0.5 km one 9.8%, where 16 and 32 pieces agreed to 7e-4 and 3e-4. A cube, and
#: anything up to this aspect, is computed as before, bit for bit.
MAX_ASPECT = 2.0


def _piece_counts(sides):
    """Near-cubic pieces along each side of a voxel with these ``sides`` (:data:`MAX_ASPECT`): odd, 1 for a cube."""
    sides = np.asarray(sides, dtype=float)
    counts = np.ceil(sides / (MAX_ASPECT * sides.min()) - 1e-9).astype(int)
    return np.where(counts % 2 == 0, counts + 1, counts)


def _triangle_enters_boxes(tri, lo, hi):
    """Whether the triangle ``tri`` (3, 3) enters each box's interior, ``(nbox,)``: the separating-axis test.

    Boxes are ``lo``/``hi`` (nbox, 3), already shrunk by the caller where touching must not count. No separating
    axis among the box's three, the triangle's normal and the nine cross products of the two sets of edges means
    the two overlap (Akenine-Moeller 2001).
    """
    centre, e = 0.5 * (lo + hi), 0.5 * (hi - lo)
    v = tri[None, :, :] - centre[:, None, :]                       # (nbox, 3 vertices, 3)
    apart = (v.min(axis=1) > e).any(axis=1) | (v.max(axis=1) < -e).any(axis=1)
    edges = np.array([tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2]])
    normal = np.cross(edges[0], edges[1])
    axes = [normal] + [np.cross(np.eye(3)[i], f) for i in range(3) for f in edges]
    for a in axes:
        if np.linalg.norm(a) < 1e-12 * max(np.linalg.norm(normal), 1.0):
            continue
        proj = v @ a                                                # (nbox, 3)
        reach = e @ np.abs(a)
        apart |= (proj.min(axis=1) > reach) | (proj.max(axis=1) < -reach)
    return ~apart


def _cells(v, start, step, n, tol):
    """The (at most two) grid cells holding each value, edges included within ``tol``: ``(lo, hi, inside)``."""
    lo = np.floor((v - tol - start) / step).astype(int)
    hi = np.floor((v + tol - start) / step).astype(int)
    inside = (v >= start - tol) & (v <= start + n * step + tol)
    return np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1), inside

#: The far path's points per axis by distance from the voxel's centre, in voxel sizes (the
#: longest side for a flat or long one): four inside the first, three inside the second, two
#: inside the third, one beyond. A tensor Gauss-Legendre rule on a cube converges far faster
#: than the triangle rule the near path borrows. Measured (2026-11-04, whole surface table,
#: surface and 3-4 km voxels, against 8 points an axis): one point 9e-3-2.1e-2 at 4 sizes,
#: 3.2-3.6e-3 at 10, 9e-4 at 20; two points 1.2-4.7e-4 at 4-10 sizes; three 8e-5-2e-4. The
#: triangle rule asked for six an axis at 4 sizes. The four-point band came with
#: :data:`FAR_VOXEL` 2 (2026-11-05, against 10 points: at a gap of 1.5-2 sizes two points
#: 5-7e-3, three and four 4e-4; at 2-3 sizes four 3-4e-4).
FAR_ORDER_SIZES = (4.0, 6.0, 15.0)

#: The faces' tent rule, displacement and strain alike: the strain tents' (gain 4, at least
#: 3 points). With the displacement tents' (gain 2, at least 2), a voxel's displacement
#: 1-2.5 sizes away was 1.3e-3 off exact point moment sources where its strain was 1e-4
#: (2026-11-04): the two-point floor's ``~0.05 (L/d)^2``. Receivers the rule would give more
#: than the cap take the split level integral (``tent_levelset(split=True)``): 1% of a size
#: from a face, displacement 1.4e-2 -> 8e-5 and strain 0.26 -> 4.5e-3 against a 400-point
#: rule; 3%, 3.7e-3 -> 4e-6 and 5.6e-2 -> 1.4e-4.
VOXEL_TENT_GAIN = STRAIN_TENT_GAIN
VOXEL_TENT_MIN_POINTS = STRAIN_TENT_MIN_POINTS

_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def unit_moments():
    """The six unit moment-density tensors, in :data:`~slipsolve._tde.STRAIN_COMPONENTS` order: ``(6, 3, 3)``.

    ``e_i e_i`` for the three diagonal components and ``e_i e_j + e_j e_i`` for the three
    shears, so that a symmetric ``M`` is ``sum_k m_k E_k`` with ``m`` its components
    ``(M_xx, M_yy, M_zz, M_xy, M_xz, M_yz)`` in east/north/up.
    """
    out = np.zeros((6, 3, 3))
    for k, (i, j) in enumerate(_PAIRS):
        out[k, i, j] = out[k, j, i] = 1.0
    return out


def to_voigt(tensor):
    """``(..., 3, 3)`` symmetric tensors to their six components ``(xx, yy, zz, xy, xz, yz)``."""
    t = np.asarray(tensor, dtype=float)
    return np.stack([t[..., i, j] for i, j in _PAIRS], axis=-1)


def from_voigt(components):
    """The inverse of :func:`to_voigt`: ``(..., 6)`` to ``(..., 3, 3)``."""
    c = np.asarray(components, dtype=float)
    out = np.empty(c.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(_PAIRS):
        out[..., i, j] = out[..., j, i] = c[..., k]
    return out


def compliance(stress, lam, mu):
    """The strain of a stress (or eigenstrain of a moment density) in an isotropic medium: ``C^-1 : stress``."""
    s = np.asarray(stress, dtype=float)
    trace = np.trace(s, axis1=-2, axis2=-1)[..., None, None]
    return (s - float(lam) / (3.0 * float(lam) + 2.0 * float(mu)) * trace * np.eye(3)) / (2.0 * float(mu))


def moment_sources(lam, mu):
    """Okada point sources making the six unit moments: ``(sources, coefficients)``.

    ``sources`` is ``[(normal, slips), ...]`` for :func:`~slipsolve._okada92.point_source_enu`:
    openings along east, north and up, and the double couples east-north, east-up and
    north-up, six in all. A source of unit potency with slip ``s`` on normal ``n`` has the
    moment ``mu (n s + s n) + lam (n . s) I``; ``coefficients`` ``(6, 6)`` combine the six
    unit-potency fields into the fields of :func:`unit_moments`, ``field_k = sum_s c[k, s] field_s``.
    """
    e = np.eye(3)
    sources = [(e[0], e[[0, 1, 2]]), (e[1], e[[1, 2]]), (e[2], e[[2]])]
    moments = [to_voigt(mu * (np.outer(n, d) + np.outer(d, n)) + lam * float(n @ d) * np.eye(3))
               for n, slips in sources for d in slips]
    return sources, np.linalg.inv(np.array(moments))


def _faces(lo, hi, free_surface):
    """The cuboid's face triangles, normals outward: ``(n, 3, 3)``; the top dropped in the free surface."""
    x = (lo[0], hi[0])
    y = (lo[1], hi[1])
    z = (lo[2], hi[2])

    def corner(i, j, k):
        return np.array([x[i], y[j], z[k]])

    quads = [  # four corners in order, and the outward normal
        ((corner(0, 0, 0), corner(0, 1, 0), corner(0, 1, 1), corner(0, 0, 1)), (-1.0, 0.0, 0.0)),
        ((corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, 0, 1)), (1.0, 0.0, 0.0)),
        ((corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)), (0.0, -1.0, 0.0)),
        ((corner(0, 1, 0), corner(1, 1, 0), corner(1, 1, 1), corner(0, 1, 1)), (0.0, 1.0, 0.0)),
        ((corner(0, 0, 0), corner(1, 0, 0), corner(1, 1, 0), corner(0, 1, 0)), (0.0, 0.0, -1.0)),
    ]
    if not free_surface:
        quads.append(((corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)),
                      (0.0, 0.0, 1.0)))
    out = []
    for (a, b, c, d), want in quads:
        for tri in (np.array([a, b, c]), np.array([a, c, d])):
            if np.cross(tri[1] - tri[0], tri[2] - tri[0]) @ np.asarray(want) < 0:
                tri = tri[[0, 2, 1]]
            out.append(tri)
    return np.array(out)


class VoxelSource:
    """A cuboid of uniform moment density in the layered reference of ``engine``.

    ``engine`` is a :class:`~slipsolve.layered_tde.LayeredTDE`; its tables need ``strain``
    for :meth:`strain`. ``x`` and ``y`` are the cuboid's ``(low, high)`` east and north
    in the :class:`~slipsolve.frame.LocalFrame`, ``depth`` its ``(top, bottom)`` in
    metres, positive down. It must lie in one layer of the tables (faces may lie on an
    interface); its constants are that layer's.

    :meth:`displacement` and :meth:`strain` give the field of a unit moment density in
    each of the six components of :func:`unit_moments`: ``(npts, 6, 3)`` east/north/up
    and ``(npts, 6, 6)`` in :data:`~slipsolve._tde.STRAIN_COMPONENTS`. A total moment is
    the density times the volume.

    Receivers more than ``strain_far`` voxel sizes from the cuboid take its strain from
    :meth:`_point_moments`, and surface receivers more than ``far`` sizes away take its
    displacement from the whole surface table's point sources; None turns either off.
    """

    def __init__(self, engine, x, y, depth, far=FAR_VOXEL, tolerance=VOXEL_TOLERANCE,
                 max_order=VOXEL_MAX_ORDER, strain_far=STRAIN_FAR_VOXEL):
        self.engine = engine
        tables = engine.tables
        top, bottom = (float(v) for v in depth)
        if not 0.0 <= top < bottom:
            raise ValueError(f"depth must be (top, bottom) with 0 <= top < bottom, got {depth}")
        x0, x1 = (float(v) for v in x)
        y0, y1 = (float(v) for v in y)
        if not (x0 < x1 and y0 < y1):
            raise ValueError("x and y must be (low, high)")
        inside = [z for z in tables.tops[1:] if top + 1e-6 < z < bottom - 1e-6]
        if inside:
            raise ValueError(f"a voxel must lie in one layer; cut it at {inside} m")
        self.layer = int(tables.layer(0.5 * (top + bottom)))
        self.lam = float(tables.lam[self.layer])
        self.mu = float(tables.mu[self.layer])
        self.nu = self.lam / (2.0 * (self.lam + self.mu))
        self.lo = np.array([x0, y0, -bottom])
        self.hi = np.array([x1, y1, -top])
        self.centre = 0.5 * (self.lo + self.hi)
        self.volume = float(np.prod(self.hi - self.lo))
        self.size = self.volume ** (1.0 / 3.0)
        self.free_surface = top == 0.0
        self.faces = _faces(self.lo, self.hi, self.free_surface)
        counts = _piece_counts(self.hi - self.lo)
        #: Near-cubic pieces whose fields sum to this voxel's (:data:`MAX_ASPECT`); None for a
        #: voxel computed whole.
        self.pieces = None
        if counts.max() > 1:
            ex, ey, ez = (np.linspace(self.lo[a], self.hi[a], counts[a] + 1) for a in range(3))
            self.pieces = [VoxelSource(engine, (ex[i], ex[i + 1]), (ey[j], ey[j + 1]), (-ez[k + 1], -ez[k]),
                                       far=far, tolerance=tolerance, max_order=max_order, strain_far=strain_far)
                           for i in range(counts[0]) for j in range(counts[1]) for k in range(counts[2])]
        self.far = None if far is None or not tables.has_surface else float(far)
        #: Box distance, in voxel sizes, past which strain takes :meth:`_point_moments`; None: never.
        self.strain_far = None if strain_far is None else float(strain_far)
        self.tolerance = float(tolerance)
        self.max_order = int(max_order)
        #: ``eps* = C0^-1 : E_k`` for each unit moment component, ``(6, 3, 3)``.
        self.eigenstrain = compliance(unit_moments(), self.lam, self.mu)
        #: The tables' packing of each unit moment component, ``(6, 6)``.
        self.packing = tensor_packing(unit_moments(), self.lam, self.mu)
        #: Okada point sources making the unit moments, and how to combine them (:func:`moment_sources`).
        self.sources = moment_sources(self.lam, self.mu)

    # -- the fields -------------------------------------------------------------------
    def displacement(self, x, y, z=None):
        """East/north/up displacement per unit moment density, ``(npts, 6, 3)``."""
        return self._field(x, y, z, strain=False)

    def strain(self, x, y, z=None):
        """Strain per unit moment density, ``(npts, 6 moment components, 6 strain components)``."""
        if not self.engine.tables.has_strain:
            raise ValueError("strain needs remainder tables computed with strain=True")
        return self._field(x, y, z, strain=True)

    def inside(self, x, y, z):
        """Points strictly inside the cuboid; at the free surface over a surface voxel, inside."""
        x, y, z = _as_points(x, y, z)
        top_ok = z <= self.hi[2] if self.free_surface else z < self.hi[2]
        return ((x > self.lo[0]) & (x < self.hi[0]) & (y > self.lo[1]) & (y < self.hi[1])
                & (z > self.lo[2]) & top_ok)

    def _field(self, x, y, z, strain):
        x, y, z = _as_points(x, y, z)
        if self.pieces is not None:
            # Farther than the longest side from a cell in pieces, the whole cell's rule stands in (its order
            # set by that side): strain by its point moments, surface displacement by the whole tabulated
            # surface field. The pieces are for its near field; a cell cut into many of them (a long fault
            # cell, a flat surface voxel) would otherwise sum them at every receiver.
            whole = self._far_whole(x, y, z) if strain else self._far_whole_surface(x, y, z)
            if not whole.any():
                return sum(piece._field(x, y, z, strain) for piece in self.pieces)
            out = np.zeros((x.size, 6, 6 if strain else 3))
            near = ~whole
            if near.any():
                out[near] = sum(piece._field(x[near], y[near], z[near], strain) for piece in self.pieces)
            out[whole] = (self._point_moments(x[whole], y[whole], z[whole]) if strain
                          else self._remainder(x[whole], y[whole], z[whole], strain=False, whole=True))
            return out
        width = 6 if strain else 3
        out = np.zeros((x.size, 6, width))
        far = self._far_strain(x, y, z) if strain else self._far(x, y, z)
        near = np.flatnonzero(~far)
        if near.size:
            out[near] = self._halfspace(x[near], y[near], z[near], strain)
            out[near] += self._remainder(x[near], y[near], z[near], strain, whole=False)
        if far.any():
            out[far] = (self._point_moments(x[far], y[far], z[far]) if strain
                        else self._remainder(x[far], y[far], z[far], strain=False, whole=True))
        return out

    def _gap(self, x, y, z):
        """Distance from each point to the cuboid (zero inside)."""
        points = np.column_stack([x, y, z])
        return np.linalg.norm(np.maximum(np.maximum(self.lo - points, points - self.hi), 0.0), axis=1)

    def _far(self, x, y, z):
        if self.far is None:
            return np.zeros(x.size, bool)
        return (z == 0.0) & (self._gap(x, y, z) > self.far * self.size)

    def _far_strain(self, x, y, z):
        if self.strain_far is None:
            return np.zeros(x.size, bool)
        return self._gap(x, y, z) > self.strain_far * self.size

    def _far_whole(self, x, y, z):
        """For a cell in pieces: receivers beyond both the strain far path's reach and its longest side."""
        if self.strain_far is None:
            return np.zeros(x.size, bool)
        reach = max(self.strain_far * self.size, float(np.max(self.hi - self.lo)))
        return self._gap(x, y, z) > reach

    def _far_whole_surface(self, x, y, z):
        """For a cell in pieces: surface receivers beyond the surface far path's reach in its LONGEST side."""
        if self.far is None:
            return np.zeros(x.size, bool)
        return (z == 0.0) & (self._gap(x, y, z) > self.far * float(np.max(self.hi - self.lo)))

    def _far_orders(self, x, y, z, tolerance=VOXEL_FAR_TOLERANCE):
        """Points a side for :meth:`_point_moments`: the least ``n`` with ``C (a / 2R)**(2n)`` within ``tolerance``."""
        side = float(np.max(self.hi - self.lo))
        reach = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2 + (z - self.centre[2]) ** 2)
        ratio = np.maximum(2.0 * reach / side, 1.5)
        n = np.ceil(np.log(VOXEL_FAR_CONSTANT / tolerance) / (2.0 * np.log(ratio)))
        return np.clip(n, 1, self.max_order).astype(int)

    def _point_moments(self, x, y, z, orders=None):
        """Strain per unit moment density from a volume rule of point moments, ``(npts, 6, 6)``.

        Each Gauss-Legendre point carries the whole layered field of a point moment: the
        half-space part from Okada (1992) with the voxel's constants, the remainder from the
        tables. For receivers outside the voxel, where both are smooth over it; ``orders``
        (points a side, per receiver) defaults to :meth:`_far_orders`.
        """
        tables = self.engine.tables
        n = x.size
        out = np.zeros((n, 6, 6))
        orders = self._far_orders(x, y, z) if orders is None else np.broadcast_to(np.asarray(orders, int), (n,))
        receiver = tables.depth_stencil(z)
        sources, coefficients = self.sources
        for order in np.unique(orders):
            group = np.flatnonzero(orders == order)
            points, weights = self._rule(order)
            q = weights.size
            for at in np.array_split(group, max(1, group.size * q // VOXEL_FAR_BATCH)):
                rec = np.repeat(at, q)
                px, py, pz = (np.tile(points[:, i], at.size) for i in range(3))
                de, dn = x[rec] - px, y[rec] - py
                stencils = (tables.distance_stencil(np.hypot(de, dn)), tables.depth_stencil(pz),
                            tuple(a[rec] for a in receiver))
                pair = combine_strain(tables.gather(*stencils), tables.gather_strain(*stencils),
                                      np.arctan2(de, dn), self.packing)
                hs = np.concatenate([point_source_enu(self.nu, normal, slips, px, py, pz, x[rec], y[rec], z[rec])[1]
                                     for normal, slips in sources], axis=1)          # (pairs, 6 sources, 6)
                pair += np.einsum("ks,nsc->nkc", coefficients, hs)
                out[at] += np.einsum("nqkc,q->nkc", pair.reshape(at.size, q, 6, 6), weights)
        return out

    def _halfspace(self, x, y, z, strain):
        """The faces' dislocations and the interior term, with the voxel's own constants."""
        kernel = tde_strain_hs_unit if strain else tde_disp_hs_unit
        gain, minimum = VOXEL_TENT_GAIN, VOXEL_TENT_MIN_POINTS
        width = 6 if strain else 3
        out = np.zeros((x.size, 6, width))
        everyone = np.arange(x.size)

        def poisson(depths):
            return np.full(np.shape(depths), self.nu)

        for tri in self.faces:
            vnorm, vstrike, vdip = _element_basis(*tri)
            frame = np.stack([vstrike, vdip, vnorm])                      # TDE components
            tents = np.zeros((x.size, 3, 3, width))
            tent_levelset(tri, x, y, z, everyone, tents, [0, 1, 2], kernel, gain, minimum,
                          (), poisson, split=True, batched=True)
            # Burgers vector eps* . (v_j - x0) at each vertex, in the triangle's components
            burgers = np.einsum("kab,jb->kja", self.eigenstrain, tri - self.centre)
            coefficients = np.einsum("kja,ca->kjc", burgers, frame)
            out += np.einsum("njcw,kjc->nkw", tents, coefficients)
        inside = self.inside(x, y, z)
        if inside.any():
            if strain:
                out[inside] += to_voigt(self.eigenstrain)[None]
            else:
                rel = np.column_stack([x[inside], y[inside], z[inside]]) - self.centre
                out[inside] += np.einsum("kab,nb->nka", self.eigenstrain, rel)
        return out

    # -- the remainder ----------------------------------------------------------------
    def _orders(self, x, y, z, whole):
        """Gauss-Legendre points per axis, from the length the integrand varies over.

        :meth:`LayeredTDE._orders`' rule with the cube root of the volume as the size: the
        larger of the distance to the centre and, in the voxel's own layer, the distance
        to the nearest image across its interfaces. ``whole`` (surface receivers far
        away) takes :data:`FAR_ORDER_SIZES`.
        """
        tables = self.engine.tables
        distance = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2
                           + (z - self.centre[2]) ** 2)
        if whole:
            # in the longest side, so a flat or long cell is not under-integrated (a cube's side is its size)
            bands = np.array(FAR_ORDER_SIZES) * max(self.size, float(np.max(self.hi - self.lo)))
            return (1 + np.searchsorted(-bands[::-1], -distance, side="left")).astype(int)
        scale = distance
        if not whole:
            z_c, z_r = -self.centre[2], -z
            same = tables.layer(z_r) == self.layer
            top = tables.tops[self.layer]
            bottom = tables.tops[self.layer + 1] if self.layer + 1 < tables.tops.size else np.inf
            image = np.full(x.size, np.inf)
            if self.layer > 0:
                image = np.minimum(image, z_c + z_r - 2.0 * top)
            if np.isfinite(bottom):
                image = np.minimum(image, 2.0 * bottom - z_c - z_r)
            scale = np.where(same, np.maximum(distance, image), distance)
        scale = np.maximum(scale, 1e-3 * self.size)
        gain = np.sqrt(_QUADRATURE_SAFETY * _QUADRATURE_CONSTANT / self.tolerance)
        return np.clip(np.ceil(gain * self.size / scale), 1, self.max_order).astype(int)

    def _rule(self, order):
        t, w = _gauss01(int(order))
        grids = [self.lo[i] + (self.hi[i] - self.lo[i]) * t for i in range(3)]
        px, py, pz = (a.ravel() for a in np.meshgrid(*grids, indexing="ij"))
        weight = np.einsum("i,j,k->ijk", w, w, w).ravel() * self.volume
        return np.column_stack([px, py, pz]), weight

    def _remainder(self, x, y, z, strain, whole):
        """The remainder's volume integral (or with ``whole``, the whole surface field's), ``(npts, 6, width)``."""
        tables = self.engine.tables
        n = x.size
        width = 6 if strain else 3
        out = np.zeros((n, 6, width))
        at_surface = (z == 0.0) & tables.has_surface & (not strain)
        receiver = tables.depth_stencil(np.where(at_surface, 0.0, z))
        orders = self._orders(x, y, z, whole)
        rec, points, weights = [], [], []
        for order in np.unique(orders):
            at = np.flatnonzero(orders == order)
            p, w = self._rule(order)
            rec.append(np.repeat(at, w.size))
            points.append(np.tile(p, (at.size, 1)))
            weights.append(np.tile(w, at.size))
        rec = np.concatenate(rec)
        points = np.concatenate(points)
        weights = np.concatenate(weights)
        dn = y[rec] - points[:, 1]
        de = x[rec] - points[:, 0]
        distance = np.hypot(dn, de)
        azimuth = np.arctan2(de, dn)
        if whole:
            pair = _combine(tables.surface_lookup(distance, points[:, 2], "whole"), azimuth,
                            self.packing)
        elif strain:
            stencils = (tables.distance_stencil(distance), tables.depth_stencil(points[:, 2]),
                        tuple(a[rec] for a in receiver))
            pair = combine_strain(tables.gather(*stencils), tables.gather_strain(*stencils),
                                  azimuth, self.packing)
        else:
            values = np.empty((rec.size, tables.values.shape[-1]))
            top = at_surface[rec]
            if top.any():
                values[top] = tables.surface_lookup(distance[top], points[top, 2], "remainder")
            if not top.all():
                deep = ~top
                values[deep] = tables.gather(tables.distance_stencil(distance[deep]),
                                             tables.depth_stencil(points[deep, 2]),
                                             tuple(a[rec[deep]] for a in receiver))
            pair = _combine(values, azimuth, self.packing)
        pair *= weights[:, None, None]
        for k in range(6):
            for c in range(width):
                out[:, k, c] = np.bincount(rec, pair[:, k, c], minlength=n)
        return out

    def __repr__(self):
        return (f"<VoxelSource x {self.lo[0]:.0f}..{self.hi[0]:.0f} y {self.lo[1]:.0f}..{self.hi[1]:.0f} "
                f"depth {-self.hi[2]:.0f}..{-self.lo[2]:.0f} m, layer {self.layer}, "
                f"mu {self.mu / 1e9:.2f} GPa>")


class Box:
    """An axis-aligned box: ``x`` and ``y`` ``(low, high)`` in the local frame, ``depth`` ``(top, bottom)``."""

    def __init__(self, x, y, depth):
        self.x, self.y, self.depth = (tuple(float(v) for v in r) for r in (x, y, depth))

    def contains(self, x, y, depth):
        return ((x > self.x[0]) & (x < self.x[1]) & (y > self.y[0]) & (y < self.y[1])
                & (depth > self.depth[0]) & (depth < self.depth[1]))


class Prism:
    """A vertical prism: a polygon in map view (``(n, 2)`` east, north) between two depths."""

    def __init__(self, polygon, depth):
        from matplotlib.path import Path as _Path

        self.polygon = np.asarray(polygon, dtype=float)
        self.path = _Path(self.polygon)
        self.depth = tuple(float(v) for v in depth)

    def contains(self, x, y, depth):
        x, y, depth = (np.asarray(a, dtype=float) for a in (x, y, depth))
        inside = self.path.contains_points(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
        return inside & (depth > self.depth[0]) & (depth < self.depth[1])


def stiffness_block(dlam, dmu):
    """The ``6 x 6`` map from strain components to moment-density components of ``dC : eps``.

    ``dC : eps = dlam tr(eps) I + 2 dmu eps``; in :func:`unit_moments`' basis a shear's
    moment component is the tensor component itself (``M_xy = m_xy``), so its row is
    ``2 dmu`` on the tensor strain ``eps_xy``. ``(n, 6, 6)`` for arrays of ``dlam``, ``dmu``.
    """
    dlam = np.atleast_1d(np.asarray(dlam, dtype=float))
    dmu = np.atleast_1d(np.asarray(dmu, dtype=float))
    out = np.zeros(dlam.shape + (6, 6))
    out[..., :3, :3] = dlam[..., None, None]
    for k in range(6):
        out[..., k, k] += 2.0 * dmu
    return out


class VoxelModel:
    """A regular grid of voxels with their own ``lambda`` and ``mu``, against a layered reference.

    ``tables`` (:class:`~slipsolve.wavenumber.RemainderTables`) define the reference
    ``C0(z)``. ``x_edges`` and ``y_edges`` (east and north in the
    :class:`~slipsolve.frame.LocalFrame`, metres, uniform spacing each) and ``z_edges``
    (depths, metres, increasing from ``>= 0``) bound the voxels; every layer interface
    strictly inside the depth range is inserted into ``z_edges``, so no voxel crosses one.
    ``lam`` and ``mu`` (Pa) are arrays ``(nx, ny, nz)`` on the grid as given, or a callable
    ``f(x, y, depth) -> (lam, mu)`` at the voxel centres (evaluated after the insertion).
    Voxels whose moduli differ from the reference's by less than ``tol`` times the
    reference ``mu`` are inactive: they add nothing and cost nothing.
    """

    def __init__(self, tables, x_edges, y_edges, z_edges, lam, mu=None, tol=1e-9, frame=None):
        self.tables = tables
        self.frame = frame
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        z_edges = np.asarray(z_edges, dtype=float)
        for edges, name in ((self.x_edges, "x"), (self.y_edges, "y")):
            step = np.diff(edges)
            if edges.size < 2 or np.any(step <= 0) or not np.allclose(step, step[0], rtol=1e-9):
                raise ValueError(f"{name}_edges must be uniform and increasing")
        if z_edges[0] < 0 or np.any(np.diff(z_edges) <= 0):
            raise ValueError("z_edges are depths: >= 0 and increasing")
        self.z_edges, added = _with_interfaces(tables, z_edges)
        self.inserted = list(added)
        self.shape = (self.x_edges.size - 1, self.y_edges.size - 1, self.z_edges.size - 1)
        cx = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
        cy = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        cz = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        if callable(lam):
            lam, mu = lam(gx, gy, gz)
        elif added:
            raise ValueError(f"z_edges were cut at the interfaces {added}; give lam and mu as a "
                             "callable, or put the interfaces in z_edges")
        self.lam = np.broadcast_to(np.asarray(lam, dtype=float), self.shape).copy()
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), self.shape).copy()
        layer = tables.layer(gz)
        self.lam0 = tables.lam[layer]
        self.mu0 = tables.mu[layer]
        self.tol = float(tol)
        self._grid = (gx, gy, gz)
        self._refresh()

    def _refresh(self):
        """Departures, the active set, its indices and centres, from :attr:`lam` and :attr:`mu`."""
        gx, gy, gz = self._grid
        self.dlam = self.lam - self.lam0
        self.dmu = self.mu - self.mu0
        self.active = (np.abs(self.dlam) + np.abs(self.dmu)) > self.tol * self.mu0
        #: ``(n_active, 3)`` grid indices and east/north/up centres of the active voxels, in
        #: the order of the operators' blocks.
        self.index = np.argwhere(self.active)
        self.centres = np.column_stack([gx[self.active], gy[self.active], -gz[self.active]])

    @classmethod
    def from_bodies(cls, tables, x_edges, y_edges, z_edges, bodies, tol=1e-9, frame=None):
        """A model whose voxels take the moduli of the bodies holding their centres.

        ``bodies`` is a list of ``(shape, moduli)``; later bodies override earlier ones, and a
        voxel in none keeps the reference. A shape is a :class:`Box` or a :class:`Prism`;
        ``moduli`` is ``(lam, mu)`` in Pa, or a callable ``f(depth) -> (lam, mu)``, or the
        string ``"scale"`` with a ``(lam factor, mu factor)`` pair as a third element. A body
        is a staircase of whole voxels: for a sharp boundary, put voxel edges on it.
        """
        bodies = list(bodies)

        def moduli(x, y, depth):
            layer = tables.layer(depth)
            lam = tables.lam[layer].astype(float).copy()
            mu = tables.mu[layer].astype(float).copy()
            for body in bodies:
                shape, value = body[0], body[1]
                inside = shape.contains(x, y, depth)
                if not inside.any():
                    continue
                if isinstance(value, str) and value == "scale":
                    f_lam, f_mu = body[2]
                    lam[inside] = tables.lam[layer[inside]] * f_lam
                    mu[inside] = tables.mu[layer[inside]] * f_mu
                elif callable(value):
                    a, b = value(depth[inside])
                    lam[inside], mu[inside] = a, b
                else:
                    lam[inside], mu[inside] = value
            return lam, mu

        return cls(tables, x_edges, y_edges, z_edges, moduli, tol=tol, frame=frame)

    @classmethod
    def from_crust1(cls, tables, x_edges, y_edges, z_edges, crust, frame, tol=1e-9):
        """Voxels with CRUST1.0's moduli (:class:`~slipsolve.crust1.Crust1`) over the tables' reference.

        Each voxel takes the thickness-weighted mean of ``lambda`` and ``mu`` over its depth range
        (below the rock surface: water and ice dropped) in the 1-degree cell holding its centre;
        ``frame`` (the :class:`~slipsolve.frame.LocalFrame` of the edges) places the centres.
        The tables' interfaces are inserted into ``z_edges`` first, as the constructor does. Every
        voxel departs from a 1-D reference somewhere, so the model fills the grid: it is meant for
        ``HeterogeneousEngine(solver="fft")``. Voxels through the fault must then be returned to the
        reference (:meth:`keep_clear`).
        """
        z_edges, _ = _with_interfaces(tables, np.asarray(z_edges, dtype=float))
        cx = 0.5 * (np.asarray(x_edges, dtype=float)[1:] + np.asarray(x_edges, dtype=float)[:-1])
        cy = 0.5 * (np.asarray(y_edges, dtype=float)[1:] + np.asarray(y_edges, dtype=float)[:-1])
        gx, gy = np.meshgrid(cx, cy, indexing="ij")
        lon, lat = (np.asarray(v) for v in frame.to_lonlat(gx, gy))
        lam, mu = crust.average(lon[:, :, None], lat[:, :, None], z_edges[None, None, :-1], z_edges[None, None, 1:])
        return cls(tables, x_edges, y_edges, z_edges, lam, mu, tol=tol, frame=frame)

    @classmethod
    def from_grid(cls, tables, x_edges, y_edges, z_edges, lon, lat, depth, vp, vs, rho, frame, samples=8, tol=1e-9):
        """Voxels from a model on a regular grid (a tomography): ``vp``, ``vs``, ``rho`` (SI) on ``(lon, lat, depth)``.

        ``lon`` and ``lat`` in degrees and ``depth`` in metres below the surface, each increasing;
        the three arrays ``(nlon, nlat, ndepth)``. At each voxel's centre (``frame`` places it) the
        model is trilinear at ``samples`` depths spread evenly over the voxel's range, and
        ``lambda = rho (vp^2 - 2 vs^2)``, ``mu = rho vs^2`` are averaged over them. Refuses a voxel
        outside the grid.
        """
        from scipy.interpolate import RegularGridInterpolator

        z_edges, _ = _with_interfaces(tables, np.asarray(z_edges, dtype=float))
        cx = 0.5 * (np.asarray(x_edges, dtype=float)[1:] + np.asarray(x_edges, dtype=float)[:-1])
        cy = 0.5 * (np.asarray(y_edges, dtype=float)[1:] + np.asarray(y_edges, dtype=float)[:-1])
        gx, gy = np.meshgrid(cx, cy, indexing="ij")
        plon, plat = (np.asarray(v) for v in frame.to_lonlat(gx, gy))
        axes = tuple(np.asarray(a, dtype=float) for a in (lon, lat, depth))
        t = (np.arange(int(samples)) + 0.5) / int(samples)
        sample_depth = z_edges[:-1, None] + np.diff(z_edges)[:, None] * t[None, :]           # (nz, samples)
        outside = ((plon < axes[0][0]) | (plon > axes[0][-1]) | (plat < axes[1][0]) | (plat > axes[1][-1])).any()
        if outside or sample_depth.min() < axes[2][0] or sample_depth.max() > axes[2][-1]:
            raise ValueError("the voxels reach outside the model's grid: extend the grid or shrink the voxels")
        shape = gx.shape + sample_depth.shape
        query = np.stack(np.broadcast_arrays(plon[:, :, None, None], plat[:, :, None, None],
                                             sample_depth[None, None]), axis=-1).reshape(-1, 3)
        fields = [RegularGridInterpolator(axes, np.asarray(a, dtype=float))(query).reshape(shape) for a in (vp, vs, rho)]
        p, s, r = fields
        mu = (r * s ** 2).mean(axis=-1)
        lam = (r * p ** 2).mean(axis=-1) - 2.0 * mu
        return cls(tables, x_edges, y_edges, z_edges, lam, mu, tol=tol, frame=frame)

    @classmethod
    def from_dataset(cls, tables, x_edges, y_edges, z_edges, ds, frame, vp="vp", vs="vs", rho="rho", lon="longitude",
                     lat="latitude", depth="depth", depth_unit=1e3, speed_unit=1e3, density_unit=1e3, samples=8,
                     tol=1e-9):
        """:meth:`from_grid` from an ``xarray.Dataset`` (a tomography as distributed, e.g. IRIS EMC's netCDF files).

        ``ds`` is opened by the caller, with whatever engine reads the file. The coordinates ``lon`` / ``lat`` are degrees
        (0-360 longitudes are turned to -180-180 when the voxels lie west of Greenwich) and ``depth`` is in
        ``depth_unit`` metres (km by default); ``vp``, ``vs`` in ``speed_unit`` m/s (km/s) and ``rho`` in ``density_unit``
        kg/m^3 (g/cm^3), in any dimension order. ``vs=None`` or ``rho=None`` derive them from Vp by
        :func:`brocher_vs` / :func:`brocher_density` (Brocher 2005's crustal fits), for models that carry Vp alone.
        """
        def coord(name):
            return np.asarray(ds[name].values, dtype=float)

        lons, lats, deps = coord(lon), coord(lat), coord(depth) * float(depth_unit)
        cx = 0.5 * (np.asarray(x_edges, dtype=float)[1:] + np.asarray(x_edges, dtype=float)[:-1])
        cy = 0.5 * (np.asarray(y_edges, dtype=float)[1:] + np.asarray(y_edges, dtype=float)[:-1])
        want_lon, _ = (np.asarray(v) for v in frame.to_lonlat(*np.meshgrid(cx, cy, indexing="ij")))
        if lons.max() > 180.0 and want_lon.min() < 0.0:
            lons = np.where(lons > 180.0, lons - 360.0, lons)

        def field(name):
            return np.asarray(ds[name].transpose(lon, lat, depth).values, dtype=float)

        p_kms = field(vp) * float(speed_unit) / 1e3
        s_kms = brocher_vs(p_kms) if vs is None else field(vs) * float(speed_unit) / 1e3
        r_gcc = brocher_density(p_kms) if rho is None else field(rho) * float(density_unit) / 1e3
        order = [np.argsort(lons), np.argsort(lats), np.argsort(deps)]
        grids = [a[np.ix_(*order)] for a in (p_kms, s_kms, r_gcc)]
        return cls.from_grid(tables, x_edges, y_edges, z_edges, lons[order[0]], lats[order[1]], deps[order[2]],
                             grids[0] * 1e3, grids[1] * 1e3, grids[2] * 1e3, frame, samples=samples, tol=tol)

    def keep_clear(self, mesh, buffer=None, samples=24):
        """Return active voxels within ``buffer`` of the fault to the reference; how many, ``int``.

        For a model that fills the region (:meth:`from_crust1`, :meth:`from_grid`): the voxels the
        fault passes through or near cannot carry a departure (:meth:`check_clearance`), so their
        departure is dropped. ``buffer`` defaults to a voxel size, as there.
        """
        hit, _ = self._near_fault(mesh, buffer, samples)
        if hit.size:
            ix, iy, iz = self.index[hit].T
            self.lam[ix, iy, iz] = self.lam0[ix, iy, iz]
            self.mu[ix, iy, iz] = self.mu0[ix, iy, iz]
            self._refresh()
        return int(hit.size)

    @property
    def n_active(self):
        return int(self.index.shape[0])

    def spec(self, i):
        """Active voxel ``i``'s picklable description, for :func:`_make_source`."""
        return ("box", self.bounds(i))

    def source(self, engine, i):
        return VoxelSource(engine, *self.bounds(i))

    def bounds(self, i):
        """``(x, y, depth)`` ranges of active voxel ``i``."""
        ix, iy, iz = self.index[i]
        return ((self.x_edges[ix], self.x_edges[ix + 1]), (self.y_edges[iy], self.y_edges[iy + 1]),
                (self.z_edges[iz], self.z_edges[iz + 1]))

    def stiffness(self):
        """``D``, block-diagonal ``(6 n, 6 n)`` sparse: the moment density is ``M = -D eps``."""
        import scipy.sparse as sp

        blocks = stiffness_block(self.dlam[self.active], self.dmu[self.active])
        return sp.block_diag(list(blocks), format="csr")

    def rigidity_at(self, points):
        """``mu`` at east/north/up points: a voxel's where one holds them, the reference's elsewhere."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        depth = -pts[:, 2]
        out = self.tables.mu[self.tables.layer(depth)].astype(float)
        ix = np.searchsorted(self.x_edges, pts[:, 0], side="right") - 1
        iy = np.searchsorted(self.y_edges, pts[:, 1], side="right") - 1
        iz = np.searchsorted(self.z_edges, depth, side="right") - 1
        ok = ((ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])
              & (iz >= 0) & (iz < self.shape[2]))
        out[ok] = self.mu[ix[ok], iy[ok], iz[ok]]
        return out

    def on_faces(self, x, y, z=None, tol=None):
        """Points on a face of an active voxel, where its field is singular: a boolean mask.

        Every face counts -- a voxel's own, one shared by two active voxels, and those between
        the near-cubic pieces a long voxel is summed from (:data:`MAX_ASPECT`) -- except a top
        face in the free surface, which is not modelled: a surface point over a surface voxel is
        fine. ``tol`` (m) defaults to 1e-6 of the smallest voxel side.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        depth = np.zeros(x.size) if z is None else -np.asarray(z, dtype=float).ravel()
        out = np.zeros(x.size, bool)
        if self.n_active == 0:
            return out
        x0, y0 = float(self.x_edges[0]), float(self.y_edges[0])
        dx, dy = (float(e[1] - e[0]) for e in (self.x_edges, self.y_edges))
        nx, ny, _ = self.shape
        thick = np.diff(self.z_edges)
        tol = 1e-6 * min(dx, dy, float(thick.min())) if tol is None else float(tol)
        ix_lo, ix_hi, in_x = _cells(x, x0, dx, nx, tol)
        jy_lo, jy_hi, in_y = _cells(y, y0, dy, ny, tol)
        for k in range(self.shape[2]):
            act = self.active[:, :, k]
            top, bottom = float(self.z_edges[k]), float(self.z_edges[k + 1])
            rows = (depth >= top - tol) & (depth <= bottom + tol)
            if not (act.any() and rows.any()):
                continue
            cx, cy, cz = _piece_counts([dx, dy, bottom - top])
            # planes across x: x0 + m dx / cx; a cell boundary (m % cx == 0) has the cells on both sides
            for coord, start, step, count, n_cells, other_lo, other_hi, other_in, swap_axes in (
                    (x, x0, dx, cx, nx, jy_lo, jy_hi, in_y, False),
                    (y, y0, dy, cy, ny, ix_lo, ix_hi, in_x, True)):
                sub = step / count
                m = np.rint((coord - start) / sub).astype(int)
                near = rows & other_in & (np.abs(coord - (start + m * sub)) <= tol) & (m >= 0) & (m <= n_cells * count)
                left = np.clip(np.where(m % count == 0, m // count - 1, m // count), 0, n_cells - 1)
                right = np.clip(m // count, 0, n_cells - 1)
                hit = np.zeros(x.size, bool)
                for a in (left, right):
                    for b in (other_lo, other_hi):
                        hit |= act[b, a] if swap_axes else act[a, b]
                out |= near & hit
            # planes across depth: top + m thickness / cz; the free surface is not a face
            sub = (bottom - top) / cz
            m = np.rint((depth - top) / sub).astype(int)
            near = (rows & in_x & in_y & (np.abs(depth - (top + m * sub)) <= tol) & (m >= 0) & (m <= cz)
                    & ~((m == 0) & (top == 0.0)))
            hit = act[ix_lo, jy_lo] | act[ix_lo, jy_hi] | act[ix_hi, jy_lo] | act[ix_hi, jy_hi]
            out |= near & hit
        return out

    def check_clearance(self, mesh, buffer=None, samples=24, contact=False):
        """Refuse an active voxel within ``buffer`` (default: one voxel size) of the fault mesh.

        With ``contact`` (2026-11-04, plan Part IV 1a) a voxel may END on the fault instead: it may
        touch it, a face on the fault's plane, but the fault must not enter its interior (a
        separating-axis test of each triangle against each nearby voxel, shrunk by 1e-6 of its
        size). The strain is collocated at each voxel's centre, on one side of the fault; the
        equation stays exact with soft material up to the fault (measured: a zone against a planar
        fault as accurate as one 2 km away for strike-slip, twice the error for dip-slip at 1 km
        cells, both converging), so cells beside it should be fine. ``buffer`` is then unused.

        A voxel's strain is collocated at its centre and taken as uniform; next to the fault
        the strain varies on the scale of the distance to it and is singular on it. A
        contrast across the fault itself is :mod:`slipsolve.bimaterial`'s. The distance is
        from each triangle, sampled ``samples`` times along each edge, to each voxel's box,
        so it can read high by up to the sample spacing; voxels whose centre is farther
        than ``buffer`` plus their half-diagonal are cleared without it.
        """
        if self.n_active == 0:
            return True
        if contact:
            return self._check_contact(mesh)
        hit, distance = self._near_fault(mesh, buffer, samples)
        if hit.size:
            k = hit[int(np.argmin(distance))]
            raise ValueError(f"active voxel {tuple(self.index[k])} lies {distance.min():.0f} m from the "
                             f"fault, within {self._buffer(buffer):.0f} m; keep the heterogeneity a voxel clear of it")
        return True

    def _buffer(self, buffer):
        dz = np.diff(self.z_edges)[self.index[:, 2]]
        size = float(np.cbrt(float(np.diff(self.x_edges[:2])[0]) * float(np.diff(self.y_edges[:2])[0]) * dz.min()))
        return size if buffer is None else float(buffer)

    def _near_fault(self, mesh, buffer, samples):
        """Active voxels within ``buffer`` of the fault mesh and their distances, ``(indices, metres)``."""
        from ._tde import distance_to_triangle

        if self.n_active == 0:
            return np.zeros(0, int), np.zeros(0)
        dz = np.diff(self.z_edges)[self.index[:, 2]]
        dx = float(np.diff(self.x_edges[:2])[0])
        dy = float(np.diff(self.y_edges[:2])[0])
        buffer = self._buffer(buffer)
        half = 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz)
        lo = np.column_stack([self.x_edges[self.index[:, 0]], self.y_edges[self.index[:, 1]],
                              -self.z_edges[self.index[:, 2] + 1]])
        hi = np.column_stack([self.x_edges[self.index[:, 0] + 1], self.y_edges[self.index[:, 1] + 1],
                              -self.z_edges[self.index[:, 2]]])
        n = int(samples)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        bary = np.column_stack([i[keep], j[keep], n - i[keep] - j[keep]]) / n
        closest = np.full(self.n_active, np.inf)
        for tri in np.asarray(mesh.vertices):
            near = np.flatnonzero(distance_to_triangle(tri, self.centres) - half < buffer)
            if not near.size:
                continue
            points = bary @ tri                                             # (m, 3)
            gap = np.maximum(np.maximum(lo[near, None, :] - points[None], points[None] - hi[near, None, :]), 0.0)
            closest[near] = np.minimum(closest[near], np.linalg.norm(gap, axis=2).min(axis=1))
        hit = np.flatnonzero(closest < buffer)
        return hit, closest[hit]

    def _check_contact(self, mesh):
        """``check_clearance(contact=True)``: no triangle of the fault may enter an active voxel's interior."""
        from ._tde import distance_to_triangle

        dz = np.diff(self.z_edges)[self.index[:, 2]]
        dx = float(np.diff(self.x_edges[:2])[0])
        dy = float(np.diff(self.y_edges[:2])[0])
        half = 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz)
        lo = np.column_stack([self.x_edges[self.index[:, 0]], self.y_edges[self.index[:, 1]],
                              -self.z_edges[self.index[:, 2] + 1]])
        hi = np.column_stack([self.x_edges[self.index[:, 0] + 1], self.y_edges[self.index[:, 1] + 1],
                              -self.z_edges[self.index[:, 2]]])
        shrink = 1e-6 * np.minimum(min(dx, dy), dz)[:, None]
        for tri in np.asarray(mesh.vertices):
            near = np.flatnonzero(distance_to_triangle(tri, self.centres) <= half * (1.0 + 1e-9))
            if not near.size:
                continue
            enters = _triangle_enters_boxes(tri, lo[near] + shrink[near], hi[near] - shrink[near])
            if enters.any():
                k = near[int(np.flatnonzero(enters)[0])]
                raise ValueError(f"the fault passes through active voxel {tuple(self.index[k])}: with "
                                 "contact=True a voxel may end on the fault, not straddle it")
        return True

    def __repr__(self):
        return (f"<VoxelModel {self.shape[0]}x{self.shape[1]}x{self.shape[2]}, {self.n_active} active; "
                f"mu {self.mu[self.active].min() / 1e9 if self.n_active else 0:.2f}-"
                f"{self.mu[self.active].max() / 1e9 if self.n_active else 0:.2f} GPa>")


def brocher_density(vp_kms):
    """Density (g/cm^3) from Vp (km/s), Brocher (2005, BSSA 95, 2081-2092) eq. 1 (Nafe-Drake), 1.5 < Vp < 8.5."""
    v = np.asarray(vp_kms, dtype=float)
    return 1.6612 * v - 0.4721 * v ** 2 + 0.0671 * v ** 3 - 0.0043 * v ** 4 + 0.000106 * v ** 5


def brocher_vs(vp_kms):
    """Vs (km/s) from Vp (km/s), Brocher (2005) eq. 6, his regression for crustal rocks, 1.5 < Vp < 8 km/s."""
    v = np.asarray(vp_kms, dtype=float)
    return 0.7858 - 1.2344 * v + 0.7949 * v ** 2 - 0.1238 * v ** 3 + 0.0064 * v ** 4


def _with_interfaces(tables, z_edges):
    """``z_edges`` with every interface of the tables strictly inside them inserted: ``(edges, inserted)``."""
    added = [t for t in tables.tops[1:] if z_edges[0] + 1e-6 < t < z_edges[-1] - 1e-6
             and not np.any(np.isclose(z_edges, t, rtol=0, atol=1e-6))]
    return (np.unique(np.concatenate([z_edges, added])) if added else z_edges), added


def _steps(voxels):
    return (float(voxels.x_edges[1] - voxels.x_edges[0]), float(voxels.y_edges[1] - voxels.y_edges[0]))


def strain_operator(voxels, engine, log=None, workers=None):
    """``Gamma``, ``(6 n, 6 n)``: the strain at each active voxel's centre per unit moment density in each.

    Rows are voxel ``r``'s six strain components (:data:`~slipsolve._tde.STRAIN_COMPONENTS`),
    columns voxel ``s``'s six moment components (:func:`unit_moments`), in
    :attr:`VoxelModel.index` order. The reference is laterally uniform, so the strain at
    voxel ``r`` from voxel ``s`` depends only on their depth rows and horizontal offset:
    one :class:`VoxelSource` per source row, evaluated at every offset that occurs, fills
    every pair. A voxel's own block is its Eshelby self-strain at its centre. ``workers`` > 1
    evaluates the pairs of rows in that many processes (:func:`~slipsolve.layered_tde.engine_pool`).
    """
    from .layered_tde import engine_pool

    n = voxels.n_active
    out = np.zeros((6 * n, 6 * n))
    if n == 0:
        return out
    dx, dy = _steps(voxels)
    index = voxels.index
    rows = {int(iz): np.flatnonzero(index[:, 2] == iz) for iz in np.unique(index[:, 2])}
    x_edges, y_edges, z_edges = voxels.x_edges, voxels.y_edges, voxels.z_edges
    tasks = []
    for iz_s, sources in rows.items():
        bounds = (x_edges[:2], y_edges[:2], z_edges[iz_s:iz_s + 2])
        centre = 0.5 * np.array([x_edges[0] + x_edges[1], y_edges[0] + y_edges[1]])
        for iz_r, receivers in rows.items():
            di = index[receivers, 0][:, None] - index[sources, 0][None, :]
            dj = index[receivers, 1][:, None] - index[sources, 1][None, :]
            offsets, inverse = np.unique(np.column_stack([di.ravel(), dj.ravel()]), axis=0,
                                         return_inverse=True)
            x = centre[0] + offsets[:, 0] * dx
            y = centre[1] + offsets[:, 1] * dy
            z = np.full(offsets.shape[0], -0.5 * (z_edges[iz_r] + z_edges[iz_r + 1]))
            tasks.append((iz_s, iz_r, sources, receivers, inverse, bounds, x, y, z))

    def scatter(task, field):                                       # field (n_offsets, 6 moment, 6 strain)
        iz_s, iz_r, sources, receivers, inverse = task[:5]
        pair = field[inverse.ravel()].reshape(receivers.size, sources.size, 6, 6)
        r6 = 6 * receivers[:, None, None, None] + np.arange(6)[None, None, None, :]
        s6 = 6 * sources[None, :, None, None] + np.arange(6)[None, None, :, None]
        out[r6, s6] = pair                                          # [r, s, moment b, strain a]
        if log:
            log(f"Gamma: source row {iz_s}, receiver row {iz_r}: {field.shape[0]} offsets")

    pool = engine_pool(engine, workers)
    if pool is None:
        sources_by_row = {}
        for task in tasks:
            iz_s, bounds = task[0], task[5]
            if iz_s not in sources_by_row:
                sources_by_row[iz_s] = VoxelSource(engine, *bounds)
            scatter(task, sources_by_row[iz_s].strain(*task[6:]))
        return out
    from .layered_tde import pool_map

    with pool:
        for k, field in pool_map(pool, _voxel_task, ((k, (task[5], *task[6:], True)) for k, task in enumerate(tasks))):
            scatter(tasks[k], field)
    return out


def _make_source(engine, spec):
    """A cell's source from its picklable ``spec``: ``("box", bounds)`` or ``("prism", inner, outer)``."""
    kind = spec[0]
    if kind == "box":
        return VoxelSource(engine, *spec[1])
    if kind == "prism":
        from .fault_cells import PrismSource

        return PrismSource(engine, spec[1], spec[2])
    raise ValueError(f"unknown cell kind {kind!r}")


def _spec_task(spec, x, y, z, strain):
    """A pool worker's field of the cell ``spec`` at the points, ``(npts, 6, 3 or 6)``."""
    from .layered_tde import worker_engine

    return _make_source(worker_engine(), spec)._field(x, y, z, strain)


def _voxel_task(bounds, x, y, z, strain):
    """A pool worker's :class:`VoxelSource` field: the voxel ``bounds`` at the points, ``(npts, 6, 3 or 6)``."""
    from .layered_tde import worker_engine

    return VoxelSource(worker_engine(), *bounds)._field(x, y, z, strain)


#: Most receivers in one call of a reference voxel's field when :func:`_los_columns` evaluates
#: translated voxels together (npts x the voxels batched).
TRANSLATED_RECEIVERS = 200_000


def _box_key(bounds):
    """Boxes equal up to a lateral translation share this key (their widths to 1e-6 m and their depths)."""
    (x0, x1), (y0, y1), (d0, d1) = bounds
    return round(x1 - x0, 6), round(y1 - y0, 6), float(d0), float(d1)


def _los_columns(engine, specs, x, y, z, look):
    """:func:`los_operator`'s columns for the cells ``specs``, ``(npts, 6 n)`` (or ``(npts, 3, 6 n)`` without ``look``).

    Boxes that are translates of one another (a regular grid's depth row) are evaluated as ONE
    reference voxel at receivers moved by each box's offset, many boxes in one call: the field
    of a voxel in a laterally uniform reference depends only on the receiver's position relative
    to it. The near path's cost is set per call by its closest receivers (the face tents' point
    groups), not by how many it serves, so one voxel at a time paid it once a voxel: on the
    Venezuela CRUST1.0 grid a shallow voxel cost 0.33 s with 13 near samples and 0.38 s with 46
    (2026-11-05). Other cells (prisms) are evaluated one by one.
    """
    out = np.zeros((x.size, 6 * len(specs)) if look is not None else (x.size, 3, 6 * len(specs)))

    def put(i, u):                                                  # u (npts, 6, 3)
        if look is not None:
            out[:, 6 * i:6 * i + 6] = np.einsum("pkc,pc->pk", u, look)
        else:
            out[:, :, 6 * i:6 * i + 6] = np.moveaxis(u, 2, 1)

    groups = {}
    for i, spec in enumerate(specs):
        if spec[0] == "box":
            groups.setdefault(_box_key(spec[1]), []).append(i)
        else:
            put(i, _make_source(engine, spec).displacement(x, y, z))
    per_call = max(1, TRANSLATED_RECEIVERS // max(1, x.size))
    for members in groups.values():
        reference = specs[members[0]][1]
        source = VoxelSource(engine, *reference)
        for start in range(0, len(members), per_call):
            batch = members[start:start + per_call]
            shift_x = np.array([specs[i][1][0][0] - reference[0][0] for i in batch])
            shift_y = np.array([specs[i][1][1][0] - reference[1][0] for i in batch])
            u = source.displacement((x[None, :] - shift_x[:, None]).ravel(), (y[None, :] - shift_y[:, None]).ravel(),
                                    np.tile(z, len(batch)))
            u = u.reshape(len(batch), x.size, 6, 3)
            for j, i in enumerate(batch):
                put(i, u[j])
    return out


def _los_share(specs, x, y, z, look):
    """A pool worker's columns of :func:`los_operator` for the cells ``specs`` (:func:`_make_source`)."""
    from .layered_tde import worker_engine

    return _los_columns(worker_engine(), specs, x, y, z, look)


def los_operator(voxels, engine, x, y, look=None, workers=None, subset=None):
    """``H``, ``(npts, 6 n)``: line of sight at surface points per unit moment density in each voxel.

    ``look`` is ``(look_e, look_n, look_u)``; without it, east/north/up, ``(npts, 3, 6 n)``.
    ``workers`` > 1 shares the voxels among that many processes. ``subset`` (indices of active
    voxels, increasing) gives only their columns, ``6 len(subset)`` of them, in that order.
    """
    from .layered_tde import engine_pool

    x, y, z = _as_points(x, y, None)
    _refuse_on_faces(voxels, x, y, z)
    voxel_ids = np.arange(voxels.n_active) if subset is None else np.asarray(subset, dtype=int)
    n = voxel_ids.size
    if look is not None:
        look = np.stack([np.asarray(v, dtype=float).ravel() for v in look], axis=1)
        out = np.zeros((x.size, 6 * n))
    else:
        out = np.zeros((x.size, 3, 6 * n))
    pool = engine_pool(engine, workers)
    if pool is not None:
        count = pool.pool._max_workers if hasattr(pool, "pool") else int(workers)
        shares = [s for s in np.array_split(np.arange(n), 4 * count) if s.size]
        from .layered_tde import pool_map

        with pool:
            jobs = ((k, ([voxels.spec(i) for i in voxel_ids[share]], x, y, z, look)) for k, share in enumerate(shares))
            for k, cols_field in pool_map(pool, _los_share, jobs):
                share = shares[k]
                out[..., 6 * share[0]:6 * (share[-1] + 1)] = cols_field
    else:
        out[...] = _los_columns(engine, [voxels.spec(i) for i in voxel_ids], x, y, z, look)
    bad = ~np.isfinite(out.reshape(x.size, -1)).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} point(s) got a non-finite voxel field, indices "
                         f"{np.flatnonzero(bad)[:10].tolist()}; a point on or within rounding of a voxel face?")
    return out


def _refuse_on_faces(voxels, x, y, z):
    """Raise, naming them, if any points lie on a voxel face (:meth:`VoxelModel.on_faces`)."""
    bad = voxels.on_faces(x, y, z)
    if bad.any():
        i = np.flatnonzero(bad)
        raise ValueError(f"{i.size} point(s) lie on voxel faces, where the field is singular: indices "
                         f"{i[:10].tolist()}{' ...' if i.size > 10 else ''}, the first at x {x[i[0]]:.3f}, "
                         f"y {y[i[0]]:.3f}, z {z[i[0]]:.3f} m; shift the voxel grid or the points")


def _digest(*arrays):
    import hashlib

    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class HeterogeneousEngine:
    """Triangular dislocations in a crust with any ``lambda(x)``, ``mu(x)``: the layered reference plus voxels.

    ``base`` is a :class:`~slipsolve.layered_tde.LayeredTDE` whose tables carry strain;
    ``voxels`` a :class:`VoxelModel` on the same tables. Satisfies the engine protocol, so
    :class:`~slipsolve.inversion.SlipInversion` takes it with ``engine=`` and every basis
    the base takes. Its Green's matrix is::

        G = G0 - H D (I + Gamma D)^-1 E          (order="born1": G0 - H D E)

    ``G0`` the base's, ``E`` the strain of each basis function at the voxel centres
    (:meth:`~slipsolve.layered_tde.LayeredTDE.strain`), ``Gamma`` (:func:`strain_operator`)
    and its LU depend only on the voxels and are built once; ``H`` (:func:`los_operator`)
    is kept per sample set, ``E`` per mesh and basis. Voxels must keep clear of the fault
    (:meth:`VoxelModel.check_clearance`), so the moment's rigidity is the reference's.

    ``solver="fft"`` never forms ``Gamma``: it applies it as one lateral convolution per pair
    of depth rows and solves by GMRES (:mod:`slipsolve.voxel_solve`), in memory ``O(n)`` in the
    voxels rather than ``O(n^2)``; the dense LU (``"dense"``) is exact and faster while it fits.
    Nor does it keep ``H`` (samples x 6 voxels, 50 GB at 17k samples and 61k voxels): it applies
    it to the solved moment density ``h_chunk`` voxels at a time, recomputed per sample set.
    """

    supports_basis = True

    def __init__(self, base, voxels, order="full", log=None, workers=None, solver="dense", h_chunk=H_CHUNK):
        if not base.tables.has_strain:
            raise ValueError("the base engine's tables need strain=True")
        if voxels.tables is not base.tables:
            raise ValueError("the voxels and the base engine must share one set of tables")
        if order not in ("full", "born1"):
            raise ValueError(f"order must be 'full' or 'born1', got {order!r}")
        if solver not in ("dense", "fft"):
            raise ValueError(f"solver must be 'dense' or 'fft', got {solver!r}")
        if solver == "fft" and not isinstance(voxels, VoxelModel):
            raise ValueError("solver='fft' needs a regular VoxelModel; fault cells are solved densely")
        self.solver = solver
        #: Voxels a block of ``H`` holds when ``solver="fft"`` streams it.
        self.h_chunk = int(h_chunk)
        self._system = None
        self.base = base
        self.voxels = voxels
        self.order = order
        self.log = log
        #: Processes for ``H``, ``E`` and ``Gamma`` (:func:`~slipsolve.layered_tde.engine_pool`):
        #: each loads the base's tables from their file once, and the one pool serves all three
        #: until :meth:`close`. None or 1 runs here.
        self.workers = workers
        self._pool = None
        self.name = f"heterogeneous:{base.name}"
        self.nu = base.nu
        self.interfaces = base.interfaces
        self._gamma = None
        self._lu = None
        self._h = {}
        self._e = {}

    # -- the voxel operators --------------------------------------------------------
    def _workers(self):
        """The engine's pool, made on first use, or None to run here."""
        if self._pool is None and self.workers and int(self.workers) > 1:
            from .layered_tde import engine_pool

            self._pool = engine_pool(self.base, self.workers)
        return self._pool

    def close(self):
        """Shut the worker pool down (a new one is made if the engine is used again)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _timed(self, label, build):
        import time

        t0 = time.time()
        out = build()
        if self.log:
            self.log(f"{label} [{time.time() - t0:.0f}s]")
        return out

    @property
    def gamma(self):
        if self._gamma is None:
            n = self.voxels.n_active
            build = (lambda: self.voxels.strain_operator(self.base, workers=self._workers())) \
                if hasattr(self.voxels, "strain_operator") else \
                (lambda: strain_operator(self.voxels, self.base, workers=self._workers()))
            self._gamma = self._timed(f"Gamma: {6 * n} x {6 * n}", build)
        return self._gamma

    def system(self):
        """``I + Gamma D``, the strain equation's matrix (dense, Fortran-ordered).

        One dense array beside ``Gamma``: the product's transpose is a Fortran-ordered view,
        the identity goes onto its diagonal in place, and :func:`scipy.linalg.lu_factor`
        overwrites it without a copy. Building ``np.eye(n) + Gamma D`` instead held four
        such arrays at once (77 GB at 49,000 unknowns, 2026-11-04); the values are the same.
        """
        d = self.voxels.stiffness()
        a = (d.T @ self.gamma.T).T
        a[np.diag_indices_from(a)] += 1.0
        return a

    def _factor(self):
        if self._lu is None:
            import scipy.linalg as sla

            self._lu = self._timed(f"LU of I + Gamma D ({6 * self.voxels.n_active} unknowns)",
                                   lambda: sla.lu_factor(self.system(), overwrite_a=True))
        return self._lu

    def _strain_at_voxels(self, mesh, basis):
        key = (mesh.digest(), getattr(basis, "name", "element"))
        if key not in self._e:
            c = self.voxels.centres
            e = self._timed(f"E: strain of {mesh.n_elements} elements at {c.shape[0]} voxel centres",
                            lambda: self.base.strain(mesh, c[:, 0], c[:, 1], c[:, 2], basis=basis,
                                                     workers=self._workers()))
            self._e[key] = e.reshape(-1, e.shape[-1])
        return self._e[key]

    def strain_system(self):
        """The FFT solver's :class:`~slipsolve.voxel_solve.StrainSystem` (built on first use)."""
        if self._system is None:
            from .voxel_solve import ConvolutionGamma, StrainSystem

            conv = ConvolutionGamma(self.voxels, self.base, log=self.log, workers=self._workers())
            self._system = StrainSystem(conv, log=self.log)
        return self._system

    def voxel_strain(self, mesh, basis=None):
        """``eps`` at the voxels per unit slip of each basis function, ``(6 n, cols)``: the solved strain."""
        e = self._strain_at_voxels(mesh, basis)
        if self.order == "born1":
            return e
        if self.solver == "fft":
            return self.strain_system().solve(e)
        import scipy.linalg as sla

        return sla.lu_solve(self._factor(), e)

    def _surface(self, x, y, look):
        key = _digest(x, y, *(look if look is not None else ()))
        if key not in self._h:
            self._h[key] = self._timed(f"H: {np.size(x)} surface points x {self.voxels.n_active} voxels",
                                       lambda: los_operator(self.voxels, self.base, x, y, look,
                                                            workers=self._workers()))
        return self._h[key]

    def _apply_h(self, x, y, look, m):
        """``H @ m``: from the kept ``H`` (dense), or streamed ``h_chunk`` voxels at a time (fft)."""
        if self.solver != "fft":
            h = self._surface(x, y, look)
            return h @ m if look is not None else np.einsum("pcv,vk->pck", h, m)
        import time

        from .layered_tde import engine_pool

        t0 = time.time()
        n = self.voxels.n_active
        npts = np.size(x)
        out = np.zeros((npts, m.shape[1]) if look is not None else (npts, 3, m.shape[1]))
        chunks = [c for c in np.array_split(np.arange(n), max(1, -(-n // self.h_chunk))) if c.size]

        def add(chunk, h):
            rows = (6 * chunk[:, None] + np.arange(6)[None, :]).ravel()
            if look is not None:
                out[...] += h @ m[rows]
            else:
                out[...] += np.einsum("pcv,vk->pck", h, m[rows])

        pool = engine_pool(self.base, self._workers())
        px, py, pz = _as_points(x, y, None)
        _refuse_on_faces(self.voxels, px, py, pz)
        lk = None if look is None else np.stack([np.asarray(v, dtype=float).ravel() for v in look], axis=1)
        if pool is None:
            for chunk in chunks:
                add(chunk, los_operator(self.voxels, self.base, x, y, look, subset=chunk))
        else:
            from .layered_tde import pool_map

            with pool:
                jobs = ((k, ([self.voxels.spec(i) for i in chunk], px, py, pz, lk)) for k, chunk in enumerate(chunks))
                for k, h in pool_map(pool, _los_share, jobs):
                    if not np.isfinite(h).all():
                        raise ValueError("a surface point got a non-finite voxel field; on or within rounding of a voxel face?")
                    add(chunks[k], h)
        if self.log:
            self.log(f"H streamed: {npts} surface points x {n} voxels in {len(chunks)} blocks [{time.time() - t0:.0f}s]")
        return out

    # -- the engine protocol ----------------------------------------------------------
    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, basis=None):
        """Line-of-sight design matrix, ``(npts, 2 * n_parameters)``; surface points only."""
        if self.voxels.n_active:
            self._surface_only(z)
            _refuse_on_faces(self.voxels, *_as_points(x, y, None))   # before E, Gamma and the solve, not after
        g0 = self.base.los_matrix(mesh, x, y, look_e, look_n, look_u, z=z, basis=basis)
        if self.voxels.n_active == 0:
            return g0
        m = self.voxels.stiffness() @ self.voxel_strain(mesh, basis)
        return g0 - self._apply_h(x, y, (look_e, look_n, look_u), m)

    def displacement(self, mesh, x, y, z=None, basis=None):
        """East/north/up displacement per unit slip, ``(npts, 3, 2 * n_parameters)``; surface points only."""
        if self.voxels.n_active:
            self._surface_only(z)
            _refuse_on_faces(self.voxels, *_as_points(x, y, None))   # before E, Gamma and the solve, not after
        g0 = self.base.displacement(mesh, x, y, z=z, basis=basis)
        if self.voxels.n_active == 0:
            return g0
        m = self.voxels.stiffness() @ self.voxel_strain(mesh, basis)    # (6 n, cols)
        return g0 - self._apply_h(x, y, None, m)

    def forward(self, mesh, slip, x, y, look=None, z=None, basis=None):
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        if look is None:
            return self.displacement(mesh, x, y, z, basis=basis) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z, basis=basis) @ slip

    @staticmethod
    def _surface_only(z):
        if z is not None and np.any(np.asarray(z, dtype=float) != 0.0):
            raise ValueError("the heterogeneous engine's H holds surface points only")

    def describe(self):
        """A JSON-able record for a saved model's options (the tables and voxels are not saved)."""
        v = self.voxels
        if hasattr(v, "describe"):
            return {"kind": "heterogeneous", "engine": self.base.name, "order": self.order, "solver": self.solver,
                    "cells": v.describe()}
        return {"kind": "heterogeneous", "engine": self.base.name, "order": self.order, "solver": self.solver,
                "voxels": {"shape": list(v.shape), "n_active": v.n_active,
                           "x_edges": v.x_edges.tolist(), "y_edges": v.y_edges.tolist(),
                           "z_edges": v.z_edges.tolist(),
                           "digest": _digest(v.lam, v.mu, v.x_edges, v.y_edges, v.z_edges)}}

    def __repr__(self):
        return f"<HeterogeneousEngine {self.voxels!r} on {self.base!r}, order {self.order}>"


__all__ = ["Box", "HeterogeneousEngine", "Prism", "VoxelSource", "VoxelModel", "strain_operator", "los_operator",
           "moment_sources",
           "stiffness_block", "unit_moments", "to_voigt", "from_voigt", "compliance", "FAR_VOXEL",
           "STRAIN_COMPONENTS"]
