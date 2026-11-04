"""Triangular dislocations in a layered half-space, exact near the element.

:class:`~slipsolve.layered.LayeredPointSource` cuts an element into point sources
and looks each up in EDGRN's tables. That is accurate a few element sizes away and
fails close in, where the point sources' ``1/r**2`` field has to be resolved by the
tables and the quadrature both: the fine tables read 11% low 300 m from the trace
and 75% at 60 m. It also has no field at depth and no opening source, which the
exact bimaterial solve (:mod:`slipsolve.bimaterial`) needs for every element.

This engine splits the layered field instead::

    u = TDE(each piece of the triangle, with its own layer's Poisson ratio)
        + the integral of the remainder R over the triangle,

where ``R`` (:class:`~slipsolve.wavenumber.RemainderTables`) is the layered point
source minus the half-space one with the source layer's constants. The first term
is the closed form of :mod:`slipsolve._tde`, singularity and all, at any point,
at the surface or at depth. The second is smooth wherever the source and receiver
are in the same layer, so a low-order rule integrates it, and it carries the whole
effect of the layering. A triangle cut by an interface is split into pieces, each in
one layer, because the half-space each piece subtracts is its own layer's.

⚠️ **Next to an interface the remainder is not smooth.** A source near an interface
has an image across it, and a receiver on the far side sees the transmitted field,
which is singular where they meet. The quadrature order follows the distance to
those images (:meth:`LayeredTDE._orders`), and the depth grid of the tables is
graded toward every interface, but an element edge lying along an interface is the
least accurate place in the model.
"""

import numpy as np

from ._tde import (STRAIN_STEP, _element_basis, tde_disp_hs, tde_disp_hs_unit,
                   tde_disp_hs_unit_many, tde_strain_hs_unit, tde_strain_hs_unit_many)
from ._tde import barycentric as _barycentric
from ._tde import distance_to_triangle as _distance_to_triangle
from .greens import _as_points, _require_finite
from .layered import _QUADRATURE_CONSTANT, _QUADRATURE_SAFETY, _triangle_quadrature
from .wavenumber import TABLE_KEYS

#: Relative accuracy the remainder's quadrature aims for, per piece and receiver.
#: The remainder is a correction to the half-space part, so this bounds a fraction
#: of a fraction of the field.
REMAINDER_TOLERANCE = 3e-3

#: Surface receivers farther than this many triangle sizes from a triangle take point
#: sources of the whole tabulated surface field instead of a dislocation plus the
#: remainder: the point-source rule is accurate there and costs table lookups, where a
#: tent costs at least six dislocations at every receiver.
FAR = 4.0

#: Most points per side of a piece in the remainder's quadrature (``order**2``
#: points).
MAX_ORDER = 30

#: Most Gauss-Legendre points in a tent's level-set integral (see :meth:`LayeredTDE.tent_field`).
TENT_MAX_POINTS = 24

#: Points in that integral: ``ceil(TENT_GAIN * L / d) + 1`` for a triangle of size ``L`` and a receiver
#: ``d`` from its nearest point (not its centroid), and never fewer than two, which make the tent's first moment exact far away.
#: Measured on a vertical surface element against 96x96 oriented constant-slip sub-triangles
#: (2026-10-12), worst over its three tents and two slips: 2 points 4e-4 at d/L 10.7 and 3e-3 at
#: 3.6; 3 points 3e-4 at 1.8; 6 points 1e-4 at 0.36; 12 points 6e-5 at 0.11; 24 points 1.4e-4 at
#: 0.04. Projecting the element's mean instead puts a single tent 9-60% off at those distances.
TENT_GAIN = 2.0

#: :data:`TENT_GAIN` and the fewest Gauss-Legendre points for a tent's STRAIN
#: (:meth:`LayeredTDE.tent_strain`). Strain at depth has no far path, so the tent rule
#: must hold at any distance, and two points do not: far away the level integrand is
#: ``(1-t)^2`` times a polynomial in ``(1-t)``, whose ``(1-t)^4`` term two points miss,
#: leaving ``~0.05 (L/d)^2`` (5e-3 at 3 sizes; the displacement's far path hides it
#: beyond 4). Measured (2026-11-04, the three tents' sum against the element, two
#: triangles, 0.6-30 sizes, worst over components): gain 2 / min 2 up to 5e-3;
#: gain 4 / min 3 at most 4e-5 within 2 sizes and 9e-5 beyond.
STRAIN_TENT_GAIN = 4.0
STRAIN_TENT_MIN_POINTS = 3

#: A piece smaller than this fraction of its triangle is dropped. Clipping at an
#: interface that passes within rounding of a vertex leaves slivers whose
#: frame is not defined.
SLIVER = 1e-9


def split_by_layers(vertices, interfaces):
    """Cut a triangle at the interface depths: ``[(piece (3, 3), mid-depth), ...]``.

    ``vertices`` is ``(3, 3)`` east/north/up. The pieces are fan triangles of the
    polygon between consecutive interfaces, in the parent's winding, so each has the
    parent's normal and frame. Vertical edges stay exactly vertical: a cut point on
    an edge whose ends share ``x, y`` inherits them exactly.
    """
    v = np.asarray(vertices, dtype=float)
    depth = -v[:, 2]
    lo, hi = depth.min(), depth.max()
    cuts = [z for z in np.asarray(interfaces, dtype=float)
            if lo + 1e-6 < z < hi - 1e-6]
    if not cuts:
        return [(v, 0.5 * (lo + hi))]
    bounds = [lo] + cuts + [hi]
    area = 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
    pieces = []
    for top, bottom in zip(bounds[:-1], bounds[1:]):
        poly = _clip(_clip(list(v), top, above=False), bottom, above=True)
        for i in range(1, len(poly) - 1):
            tri = np.array([poly[0], poly[i], poly[i + 1]])
            piece_area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            if piece_area > SLIVER * area:
                pieces.append((tri, 0.5 * (top + bottom)))
    return pieces


def _clip(poly, depth, above):
    """Keep the part of a convex polygon above (``above=True``: shallower than) or below ``depth``."""
    def inside(p):
        return -p[2] <= depth if above else -p[2] >= depth

    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = inside(a), inside(b)
        if ina:
            out.append(a)
        if ina != inb:
            t = (depth + a[2]) / (a[2] - b[2])          # where -z crosses `depth`
            p = a + t * (b - a)
            p[2] = -depth
            # an edge whose ends share x and y keeps them exactly
            if a[0] == b[0]:
                p[0] = a[0]
            if a[1] == b[1]:
                p[1] = a[1]
            out.append(p)
    return out


def moment_packing(normal, direction):
    """``(m1, m2, m3, m4, m5, e)`` of a unit dislocation with this normal and slip direction.

    The deviatoric part of ``n (x) d + d (x) n``, in north/east/down, packed as
    EDCMP's five (``m1 = M_ne``, ``m2 = M_nd``, ``m3 = M_dd``, ``m4 = (M_nn - M_ee)/2``,
    ``m5 = M_ed``), and ``e = n . d``, the isotropic part's weight on the ``ep`` table.
    The tables carry ``mu`` and ``K``, so the moment is ``mu`` times the five plus
    ``K e`` times the identity -- the ``lam (n . d) I + mu (n d + d n)`` of an
    opening, split into its deviatoric and isotropic parts.
    """
    n = np.asarray(normal, dtype=float)
    d = np.asarray(direction, dtype=float)
    tensor = np.outer(n, d) + np.outer(d, n)
    tensor -= np.trace(tensor) / 3.0 * np.eye(3)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    m = swap @ tensor @ swap.T
    return np.array([m[0, 1], m[0, 2], m[2, 2], 0.5 * (m[0, 0] - m[1, 1]), m[1, 2],
                     float(n @ d)])


class LayeredTDE:
    """Triangular dislocations in the layered half-space of a set of remainder tables.

    Satisfies the engine protocol of :class:`~slipsolve.greens.HalfSpaceTDE`, so
    :class:`~slipsolve.inversion.SlipInversion` takes it with ``engine=``. It
    returns element columns, which :class:`SlipInversion` projects onto a nodal
    basis. ``element_field`` gives one triangle's field of unit strike-, dip- and
    tensile-slip at any points, for :class:`~slipsolve.bimaterial.InterfaceSolve`.
    """

    name = "layered_tde"
    #: Nodal columns are integrated over each tent itself (:meth:`tent_field`), as
    #: :class:`~slipsolve.layered.LayeredPointSource` integrates them, rather than
    #: projected from element means.
    supports_basis = True

    def __init__(self, tables, tolerance=REMAINDER_TOLERANCE, max_order=MAX_ORDER, far=FAR):
        self.tables = tables
        self.tolerance = float(tolerance)
        self.max_order = int(max_order)
        #: Surface receivers farther than this many triangle sizes take point sources of
        #: the whole tabulated surface field (``tables.surface``) instead of a dislocation;
        #: ``None``, or tables without a surface field, keeps dislocations everywhere.
        self.far = None if far is None or not tables.has_surface else float(far)
        #: The surface layer's, for the engine protocol; each piece uses its own.
        self.nu = float(tables.poisson(0.0))
        #: Depths (m) where the crust steps. An element crossing one is cut there,
        #: so meshes laid out for this engine should put element edges on them.
        self.interfaces = np.asarray(tables.tops[1:], dtype=float)
        self._interfaces = self.interfaces

    # -- the engine protocol ----------------------------------------------------
    def displacement(self, mesh, x, y, z=None, basis=None):
        """ENU displacement per unit slip, ``(npts, 3, 2 * n_parameters)``.

        Element columns without a basis (or with the element basis); with a nodal
        basis, each node's column is its tent integrated over every incident triangle.
        """
        x, y, z = _as_points(x, y, z)
        out = None
        for k, cols, enu in self._columns(mesh, x, y, z, basis):
            if out is None:
                out = np.zeros((x.size, 3, 2 * cols))
            out[:, :, k] += enu[:, 0]
            out[:, :, cols + k] += enu[:, 1]
        _require_finite(out, mesh, x, y, z)
        return out

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, basis=None):
        """Line-of-sight design matrix, ``(npts, 2 * n_parameters)``; columns as :meth:`displacement`."""
        x, y, z = _as_points(x, y, z)
        look = np.stack([np.asarray(v, dtype=float).ravel()
                         for v in (look_e, look_n, look_u)], axis=1)
        if look.shape[0] != x.size:
            raise ValueError("Look-vector components must match the observation count")
        g = None
        for k, cols, enu in self._columns(mesh, x, y, z, basis):
            if g is None:
                g = np.zeros((x.size, 2 * cols))
            los = np.einsum("pci,pi->pc", enu, look)
            g[:, k] += los[:, 0]
            g[:, cols + k] += los[:, 1]
        _require_finite(g, mesh, x, y, z)
        return g

    def _columns(self, mesh, x, y, z, basis):
        """Yield ``(column index, n_columns, (npts, 2, 3) ENU of strike and dip)`` contributions."""
        nodal = basis is not None and getattr(basis, "name", "element") != "element"
        if not nodal:
            n = mesh.n_elements
            for k in range(n):
                yield k, n, self.element_field(mesh.vertices[k], x, y, z, components=(0, 1))
            return
        if basis.name != "node":
            raise ValueError(f"LayeredTDE cannot assemble the {basis.name!r} basis")
        n = basis.n_basis
        triangles = np.asarray(mesh.triangles)
        for e in range(mesh.n_elements):
            tents = self.tent_field(mesh.vertices[e], x, y, z, components=(0, 1))
            for j, node in enumerate(triangles[e]):
                yield int(node), n, tents[:, j]

    def forward(self, mesh, slip, x, y, look=None, z=None, basis=None):
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        expected = 2 * (mesh.n_elements if basis is None or getattr(basis, "name", "element") == "element"
                        else basis.n_basis)
        if slip.size != expected:
            raise ValueError(f"slip has {slip.size} values; expected {expected}")
        if look is None:
            return self.displacement(mesh, x, y, z, basis=basis) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z, basis=basis) @ slip

    # -- one element ---------------------------------------------------------------
    def element_field(self, vertices, x, y, z=None, offset=None, components=(0, 1, 2)):
        """ENU field of unit slip on one triangle: ``(npts, len(components), 3)``.

        ``components`` index the triangle's own strike, dip and normal (tensile)
        directions, as :func:`slipsolve._tde.tde_disp_hs` defines them. With
        ``offset`` (``(npts, 3)``), the half-space part is averaged between
        ``point + offset`` and ``point - offset``: the mean of the two sides at a
        point on the element itself. The remainder is continuous there and is
        taken at the point. Surface receivers beyond :attr:`far` take point sources.
        """
        x, y, z = _as_points(x, y, z)
        vertices = np.asarray(vertices, dtype=float)
        components = list(components)
        packing = self._packing(vertices)[components]
        pieces = split_by_layers(vertices, self._interfaces)
        out = np.zeros((x.size, len(components), 3))
        far = self._far(vertices, x, y, z) if offset is None else np.zeros(x.size, bool)
        near = np.flatnonzero(~far)
        if near.size:
            xn, yn, zn = x[near], y[near], z[near]
            m = near.size
            if offset is None:
                px, py, pz = xn, yn, zn
            else:   # both sides in one call
                px, py, pz = (np.concatenate([a + offset[near, i], a - offset[near, i]])
                              for i, a in enumerate((xn, yn, zn)))
            nu = self.tables.poisson(np.array([mid for _, mid in pieces]))
            for (tri, _), nu_piece in zip(pieces, nu):
                u = np.moveaxis(tde_disp_hs_unit(px, py, pz, *tri, nu_piece)[components], -1, 0)
                out[near] += u if offset is None else 0.5 * (u[:m] + u[m:])
            out[near] += self._remainder(pieces, packing, xn, yn, zn)
        if far.any():
            out[far] = self._remainder(pieces, packing, x[far], y[far], z[far], whole=True)
        return out

    def slip_field(self, vertices, slip, x, y, z=None):
        """ENU field of one slip vector ``(ss, ds, ts)`` on one triangle, ``(npts, 3)``."""
        x, y, z = _as_points(x, y, z)
        vertices = np.asarray(vertices, dtype=float)
        slip = np.asarray(slip, dtype=float)
        packing = (slip[:, None] * self._packing(vertices)).sum(axis=0, keepdims=True)
        pieces = split_by_layers(vertices, self._interfaces)
        out = np.zeros((x.size, 3))
        far = self._far(vertices, x, y, z)
        near = np.flatnonzero(~far)
        if near.size:
            nu = self.tables.poisson(np.array([mid for _, mid in pieces]))
            for (tri, _), nu_piece in zip(pieces, nu):
                out[near] += np.column_stack(tde_disp_hs(x[near], y[near], z[near], *tri, *slip,
                                                         nu_piece))
            out[near] += self._remainder(pieces, packing, x[near], y[near], z[near])[:, 0]
        if far.any():
            out[far] = self._remainder(pieces, packing, x[far], y[far], z[far], whole=True)[:, 0]
        return out

    def tent_field(self, vertices, x, y, z=None, components=(0, 1)):
        """ENU field of unit slip shaped by each of the triangle's three tents: ``(npts, 3, ncomp, 3)``.

        Axis 1 is the tent of vertex 0, 1, 2 (1 there, 0 at the other two). The
        half-space part uses the tent's level sets: a linear tent is
        ``int_0^1 [tent >= t] dt``, and its set ``tent >= t`` is the triangle shrunk
        toward that vertex by ``1 - t``. So the tent's field is a 1-D integral of
        constant-slip dislocations, each exact, taken by Gauss-Legendre in ``t``
        with more points for closer receivers (:data:`TENT_GAIN`); the shrunk
        triangles are cut at the interfaces like any other. The remainder is
        integrated with each quadrature point weighted by the tent. Surface
        receivers beyond :attr:`far` take tent-weighted point sources.
        """
        x, y, z = _as_points(x, y, z)
        vertices = np.asarray(vertices, dtype=float)
        components = list(components)
        packing = self._packing(vertices)[components]
        pieces = split_by_layers(vertices, self._interfaces)
        out = np.zeros((x.size, 3, len(components), 3))
        far = self._far(vertices, x, y, z)
        near = np.flatnonzero(~far)
        if near.size:
            self._tent_halfspace(vertices, x, y, z, near, out, components, tde_disp_hs_unit)
            out[near] += self._remainder(pieces, packing, x[near], y[near], z[near], parent=vertices)
        if far.any():
            out[far] = self._remainder(pieces, packing, x[far], y[far], z[far], parent=vertices,
                                       whole=True)
        return out

    def _tent_halfspace(self, vertices, x, y, z, near, out, components, kernel,
                        gain=None, min_points=2):
        """Add each tent's half-space field at the points ``near`` into ``out[near]``.

        :func:`tent_levelset` with this engine's interfaces and each piece's own layer's
        Poisson ratio. ``kernel`` is :func:`~slipsolve._tde.tde_disp_hs_unit` (displacement)
        or :func:`~slipsolve._tde.tde_strain_hs_unit` (strain); ``out`` is
        ``(npts, 3, ncomp, 3 or 6)``.
        """
        tent_levelset(vertices, x, y, z, near, out, components, kernel,
                      TENT_GAIN if gain is None else gain, min_points, self._interfaces,
                      self.tables.poisson)

    # -- strain at depth: the heterogeneous engine's E ------------------------------
    def strain(self, mesh, x, y, z, basis=None):
        """Strain per unit slip at points, ``(npts, 6, 2 * n_parameters)``.

        Components as :data:`~slipsolve._tde.STRAIN_COMPONENTS` (east, north, up; tensor
        shears); columns as :meth:`displacement`: element columns without a basis, each
        node's tent with a nodal one. For points at depth (the voxel centres of
        :mod:`slipsolve.heterogeneity`); there is no far path, since no table holds the
        whole field's strain. Needs tables computed with ``strain=True``.
        """
        x, y, z = _as_points(x, y, z)
        nodal = basis is not None and getattr(basis, "name", "element") != "element"
        if not nodal:
            n = mesh.n_elements
            out = np.zeros((x.size, 6, 2 * n))
            for k in range(n):
                eps = self.element_strain(mesh.vertices[k], x, y, z, components=(0, 1))
                out[:, :, k] += eps[:, 0]
                out[:, :, n + k] += eps[:, 1]
        else:
            if basis.name != "node":
                raise ValueError(f"LayeredTDE cannot assemble the {basis.name!r} basis")
            n = basis.n_basis
            out = np.zeros((x.size, 6, 2 * n))
            triangles = np.asarray(mesh.triangles)
            for e in range(mesh.n_elements):
                tents = self.tent_strain(mesh.vertices[e], x, y, z, components=(0, 1))
                for j, node in enumerate(triangles[e]):
                    out[:, :, node] += tents[:, j, 0]
                    out[:, :, n + node] += tents[:, j, 1]
        _require_finite(out, mesh, x, y, z)
        return out

    def element_strain(self, vertices, x, y, z=None, components=(0, 1, 2)):
        """Strain of unit slip on one triangle: ``(npts, len(components), 6)``.

        The half-space strain of each piece (:func:`~slipsolve._tde.tde_strain_hs_unit`,
        with its layer's Poisson ratio) plus the remainder's, integrated over the pieces
        (:meth:`~slipsolve.wavenumber.RemainderTables.gather_strain`,
        :func:`combine_strain`). Receivers anywhere at or below the surface; there is no
        far path.
        """
        x, y, z = _as_points(x, y, z)
        vertices = np.asarray(vertices, dtype=float)
        components = list(components)
        packing = self._packing(vertices)[components]
        pieces = split_by_layers(vertices, self._interfaces)
        out = np.zeros((x.size, len(components), 6))
        nu = self.tables.poisson(np.array([mid for _, mid in pieces]))
        for (tri, _), nu_piece in zip(pieces, nu):
            out += np.moveaxis(tde_strain_hs_unit(x, y, z, *tri, nu_piece)[components], -1, 0)
        out += self._remainder(pieces, packing, x, y, z, strain=True)
        return out

    def tent_strain(self, vertices, x, y, z=None, components=(0, 1)):
        """Strain of unit slip shaped by each of the triangle's tents: ``(npts, 3, ncomp, 6)``.

        :meth:`tent_field`'s construction with the half-space strain in the level-set
        integral and the remainder's strain weighted by the tents; no far path.
        """
        x, y, z = _as_points(x, y, z)
        vertices = np.asarray(vertices, dtype=float)
        components = list(components)
        packing = self._packing(vertices)[components]
        pieces = split_by_layers(vertices, self._interfaces)
        out = np.zeros((x.size, 3, len(components), 6))
        tent_levelset(vertices, x, y, z, np.arange(x.size), out, components, tde_strain_hs_unit,
                      STRAIN_TENT_GAIN, STRAIN_TENT_MIN_POINTS, self._interfaces,
                      self.tables.poisson, batched=True)
        out += self._remainder(pieces, packing, x, y, z, parent=vertices, strain=True)
        return out

    def _far(self, vertices, x, y, z):
        """Surface receivers more than :attr:`far` triangle sizes from the triangle."""
        if self.far is None:
            return np.zeros(x.size, bool)
        surface = z == 0.0
        if not surface.any():
            return surface
        v = np.asarray(vertices, dtype=float)
        size = np.sqrt(np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])))
        out = np.zeros(x.size, bool)
        at = np.flatnonzero(surface)
        out[at] = _distance_to_triangle(v, np.column_stack([x[at], y[at], z[at]])) > self.far * size
        return out

    @staticmethod
    def _packing(vertices):
        """The moment packing of unit strike-, dip- and tensile-slip, ``(3, 6)``."""
        vnorm, vstrike, vdip = _element_basis(*vertices)
        return np.array([moment_packing(vnorm, d) for d in (vstrike, vdip, vnorm)])

    def _remainder(self, pieces, packing, x, y, z, parent=None, whole=False, strain=False):
        """The remainder's integral over the pieces, ``(npts, n_components, 3)``.

        Every (receiver, quadrature point) pair of a piece goes through one table
        lookup, whatever order each receiver was given, and the receivers' depth
        stencils are worked out once for all the pieces. With ``parent`` (the whole
        triangle) each point is also weighted by the three tents, and the result is
        ``(npts, 3, n_components, 3)``. With ``whole`` the receivers are at the
        surface and the table is the whole surface field, not the remainder: point
        sources, ordered by the point-source rule. With ``strain`` the last axis is the
        six strain components (:func:`combine_strain`), every receiver reads the 3-D
        tables (the surface ones carry no strain), and ``whole`` is refused.
        """
        n = x.size
        tents = parent is not None
        width = 6 if strain else 3
        out = np.zeros((n, 3, packing.shape[0], width) if tents else (n, packing.shape[0], width))
        tables = self.tables
        if strain and (whole or not tables.has_strain):
            raise ValueError("strain needs remainder tables computed with strain=True, and has "
                             "no far path" if whole else
                             "strain needs remainder tables computed with strain=True")
        # Receivers at the surface read the fine 2-D surface tables when there are any;
        # the 3-D table is too coarse in depth for a point source seen from the surface.
        at_surface = (z == 0.0) & tables.has_surface & (not strain)
        receiver = tables.depth_stencil(np.where(at_surface, 0.0, z))
        for tri, _ in pieces:
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            size = np.sqrt(2.0 * area)
            orders = (self._point_orders(tri, size, x, y, z) if whole
                      else self._orders(tri, size, x, y, z))
            rec, points, weights = [], [], []
            for order in np.unique(orders):
                at = np.flatnonzero(orders == order)
                bary, share = _rule(int(order))
                rec.append(np.repeat(at, share.size))
                points.append(np.tile(bary @ tri, (at.size, 1)))
                weights.append(np.tile(share * area, at.size))
            rec = np.concatenate(rec)
            points = np.concatenate(points)
            dn = y[rec] - points[:, 1]
            de = x[rec] - points[:, 0]
            distance = np.hypot(dn, de)
            if whole:
                values = tables.surface_lookup(distance, points[:, 2], "whole")
            elif strain:
                stencils = (tables.distance_stencil(distance), tables.depth_stencil(points[:, 2]),
                            tuple(a[rec] for a in receiver))
                values = tables.gather(*stencils)
                derivatives = tables.gather_strain(*stencils)
            else:
                values = np.empty((rec.size, len(TABLE_KEYS)))
                top = at_surface[rec]
                if top.any():
                    values[top] = tables.surface_lookup(distance[top], points[top, 2], "remainder")
                if not top.all():
                    deep = ~top
                    values[deep] = tables.gather(tables.distance_stencil(distance[deep]),
                                                 tables.depth_stencil(points[deep, 2]),
                                                 tuple(a[rec[deep]] for a in receiver))
            if strain:
                pair = combine_strain(values, derivatives, np.arctan2(de, dn), packing)
            else:
                pair = _combine(values, np.arctan2(de, dn), packing)
            pair *= np.concatenate(weights)[:, None, None]
            if not tents:
                for i in range(packing.shape[0]):
                    for c in range(width):
                        out[:, i, c] += np.bincount(rec, pair[:, i, c], minlength=n)
                continue
            phi = _barycentric(parent, points)
            for j in range(3):
                for i in range(packing.shape[0]):
                    for c in range(width):
                        out[:, j, i, c] += np.bincount(rec, phi[:, j] * pair[:, i, c], minlength=n)
        return out

    def _point_orders(self, tri, size, x, y, z):
        """The point-source rule of :class:`~slipsolve.layered.LayeredPointSource` for the whole field."""
        centroid = tri.mean(axis=0)
        distance = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2 + (z - centroid[2]) ** 2)
        gain = np.sqrt(_QUADRATURE_SAFETY * _QUADRATURE_CONSTANT / self.tolerance)
        return np.clip(np.ceil(gain * size / np.maximum(distance, size / 100.0)), 1,
                       self.max_order).astype(int)

    def _orders(self, tri, size, x, y, z):
        """Quadrature order per receiver, from the length the remainder varies over.

        The point-source rule's ``C (k L / l)**-2`` error model
        (:data:`~slipsolve.layered._QUADRATURE_CONSTANT`), with ``l`` the larger of
        the receiver's distance and, for a receiver in the piece's own layer, the
        distance to the nearest image across that layer's interfaces -- the
        remainder is smooth on that scale even at the source. A receiver in another
        layer gets the distance alone: the transmitted field is singular where
        source and receiver meet at the interface.
        """
        centroid = tri.mean(axis=0)
        z_c, z_r = -centroid[2], -z
        distance = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
                           + (z - centroid[2]) ** 2)
        tables = self.tables
        layer = int(tables.layer(z_c))
        same = tables.layer(z_r) == layer
        top = tables.tops[layer]
        bottom = tables.tops[layer + 1] if layer + 1 < tables.tops.size else np.inf
        image = np.full(x.size, np.inf)
        if layer > 0:
            image = np.minimum(image, z_c + z_r - 2.0 * top)
        if np.isfinite(bottom):
            image = np.minimum(image, 2.0 * bottom - z_c - z_r)
        scale = np.where(same, np.maximum(distance, image), distance)
        scale = np.maximum(scale, 1e-3 * size)
        gain = np.sqrt(_QUADRATURE_SAFETY * _QUADRATURE_CONSTANT / self.tolerance)
        return np.clip(np.ceil(gain * size / scale), 1, self.max_order).astype(int)

    def __repr__(self):
        return f"<LayeredTDE {self.tables!r} tolerance={self.tolerance:g}>"


def _combine(values, azimuth, packing):
    """East, north and up of each tabulated pair for each moment packing: ``(npairs, ncomp, 3)``.

    ``values`` are the ten table columns at each pair and ``azimuth`` the direction
    from source to receiver, clockwise from north. The azimuthal factors are the
    ones :class:`~slipsolve.layered.LayeredPointSource` puts back, plus the
    explosion's, which is axisymmetric like the CLVD.
    """
    cos1, sin1 = np.cos(azimuth), np.sin(azimuth)
    cos2, sin2 = np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)
    (ss_uz, ss_ur, ss_ut, ds_uz, ds_ur, ds_ut,
     cl_uz, cl_ur, ep_uz, ep_ur) = values.T
    out = np.empty((azimuth.size, packing.shape[0], 3))
    for i, (m1, m2, m3, m4, m5, e) in enumerate(packing):
        ps_ss = m1 * sin2 + m4 * cos2
        sh_ss = m1 * cos2 - m4 * sin2
        ps_ds = m2 * cos1 + m5 * sin1
        sh_ds = m2 * sin1 - m5 * cos1
        uz = ps_ss * ss_uz + ps_ds * ds_uz + m3 * cl_uz + e * ep_uz
        ur = ps_ss * ss_ur + ps_ds * ds_ur + m3 * cl_ur + e * ep_ur
        ut = sh_ss * ss_ut + sh_ds * ds_ut
        out[:, i, 0] = ur * sin1 + ut * cos1
        out[:, i, 1] = ur * cos1 - ut * sin1
        out[:, i, 2] = -uz
    return out


def tent_levelset(vertices, x, y, z, near, out, components, kernel, gain, min_points,
                  interfaces, poisson, split=False, batched=False):
    """Add each of a triangle's three tents' half-space fields at the points ``near`` into ``out[near]``.

    A linear tent is ``int_0^1 [tent >= t] dt`` and the set ``tent >= t`` is the triangle
    shrunk toward its vertex by ``1 - t``, so a tent's field is a 1-D integral of
    constant-slip dislocations, each exact, taken by Gauss-Legendre in ``t`` with
    ``ceil(gain L / d) + 1`` points for a receiver ``d`` from the triangle (at least
    ``min_points``, at most :data:`TENT_MAX_POINTS`), split where a shrinking triangle
    stops crossing an interface. Each shrunk triangle is cut at ``interfaces`` and each
    piece takes ``poisson(mid-depth)``. ``kernel`` is
    :func:`~slipsolve._tde.tde_disp_hs_unit` or :func:`~slipsolve._tde.tde_strain_hs_unit`;
    ``out`` is ``(npts, 3, ncomp, 3 or 6)``, axis 1 the tent of vertex 0, 1, 2.

    ``split`` treats receivers the rule would give more than :data:`TENT_MAX_POINTS` (closer
    than about ``gain / TENT_MAX_POINTS`` sizes) one at a time, splitting the level integral
    where the shrinking triangle passes them: its moving edge at ``t* = tent_j`` of the
    receiver's projection, each moving vertex where it comes closest. The integrand has a
    near-jump there, of width ``height / size``, which a capped rule cannot resolve and each
    piece's end-clustered nodes can (:data:`SPLIT_POINTS` a piece).

    ``batched`` evaluates every shrunk piece of a receiver group in one call of the
    triangle-vectorised kernels (:func:`~slipsolve._tde.tde_disp_hs_unit_many`,
    :func:`~slipsolve._tde.tde_strain_hs_unit_many`), instead of three calls a level: the
    scalar kernels cost ~10 ms a call in overhead whatever the number of receivers. Equal
    to the scalar path to rounding for displacement; for strain the difference step is the
    receiver's distance to the parent triangle (not to each piece), so the two agree to the
    differences' own accuracy (~1e-7).
    """
    if batched:
        _tent_levelset_batched(vertices, x, y, z, near, out, components, kernel, gain, min_points,
                               interfaces, poisson, split)
        return
    area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    size = np.sqrt(2.0 * area)
    distance = _distance_to_triangle(vertices, np.column_stack([x[near], y[near], z[near]]))
    requested = np.ceil(float(gain) * size / np.maximum(distance, 1e-3 * size)) + 1
    points = np.clip(requested, int(min_points), TENT_MAX_POINTS).astype(int)
    alone = (requested > TENT_MAX_POINTS) if split else np.zeros(near.size, bool)
    for i in near[alone]:
        receiver = np.array([x[i], y[i], z[i]])
        for j in range(3):
            levels, weights = _tent_levels_split(vertices, j, interfaces, receiver, SPLIT_POINTS)
            for level, weight in zip(levels, weights):
                shrunk = vertices[j] + (1.0 - level) * (vertices - vertices[j])
                cut = split_by_layers(shrunk, interfaces)
                nu = poisson(np.array([mid for _, mid in cut]))
                for (tri, _), nu_piece in zip(cut, nu):
                    u = kernel(x[i:i + 1], y[i:i + 1], z[i:i + 1], *tri, nu_piece)[components]
                    out[i, j] += weight * u[..., 0]
    near, points = near[~alone], points[~alone]
    for count in np.unique(points):
        at = near[points == count]
        for j in range(3):
            levels, weights = _tent_levels(vertices, j, interfaces, int(count))
            for level, weight in zip(levels, weights):
                shrunk = vertices[j] + (1.0 - level) * (vertices - vertices[j])
                cut = split_by_layers(shrunk, interfaces)
                nu = poisson(np.array([mid for _, mid in cut]))
                for (tri, _), nu_piece in zip(cut, nu):
                    u = kernel(x[at], y[at], z[at], *tri, nu_piece)[components]
                    out[at, j] += weight * np.moveaxis(u, -1, 0)


def tensor_packing(moment, lam, mu):
    """``(m1, m2, m3, m4, m5, e)`` of a moment tensor, for tables carrying ``mu`` and ``K``.

    ``moment`` is ``(3, 3)`` east/north/up (or ``(n, 3, 3)``), at a source whose layer
    has ``lam`` and ``mu``. The tables give the field per unit potency with the moment
    ``mu`` times a deviatoric unit tensor (``ss``, ``ds``, ``cl``) or ``K = lam + 2 mu / 3``
    times the identity (``ep``), so the packing is ``dev(M) / mu`` in EDCMP's five and
    ``e = tr(M) / (3 K)``. A dislocation's moment, ``mu (n d + d n) + lam (n . d) I``,
    packs exactly as :func:`moment_packing` packs ``n`` and ``d``.
    """
    m = np.asarray(moment, dtype=float)
    single = m.ndim == 2
    m = m.reshape(-1, 3, 3)
    m = 0.5 * (m + np.swapaxes(m, 1, 2))
    trace = np.trace(m, axis1=1, axis2=2)
    dev = (m - trace[:, None, None] / 3.0 * np.eye(3)) / float(mu)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    ned = swap @ dev @ swap.T
    out = np.column_stack([ned[:, 0, 1], ned[:, 0, 2], ned[:, 2, 2],
                           0.5 * (ned[:, 0, 0] - ned[:, 1, 1]), ned[:, 1, 2],
                           trace / (3.0 * (float(lam) + 2.0 * float(mu) / 3.0))])
    return out[0] if single else out


def combine_strain(values, derivatives, azimuth, packing):
    """Strain of each tabulated pair for each moment packing: ``(npairs, ncomp, 6)``.

    ``values`` are the ten table columns at each pair (:func:`_combine`'s input) and
    ``derivatives`` the 27 strain columns (:data:`~slipsolve.wavenumber.STRAIN_KEYS`:
    derivatives in distance and in receiver depth, then the ``1/r`` combinations), as
    :meth:`~slipsolve.wavenumber.RemainderTables.gather_strain` returns them; ``azimuth``
    is the direction from source to receiver clockwise from north. Components as
    :data:`~slipsolve._tde.STRAIN_COMPONENTS`: east, north, up, tensor shears.

    In EDGRN's north/east/down frame ``(r, phi, z)`` is a right-handed cylindrical
    system with ``z`` down, so the usual formulas hold::

        e_rr = d_r u_r          e_pp = (d_phi u_t + u_r) / r       e_zz = d_z u_z
        e_rp = (d_r u_t + (d_phi u_r - u_t) / r) / 2
        e_rz = (d_z u_r + d_r u_z) / 2          e_pz = (d_z u_t + d_phi u_z / r) / 2

    The azimuthal derivatives need no table: the patterns :func:`_combine` puts back
    differentiate into each other (``sin 2phi`` and ``cos 2phi`` swap with a factor 2,
    ``sin phi`` and ``cos phi`` with 1; ``cl`` and ``ep`` have none). What is divided by
    ``r`` is read from the tables' ``1/r`` columns, never divided here, so the axis
    (``r = 0``, where a volume rule puts points) needs no special case. Then
    ``eps_ENU = Q' eps_cyl Q`` with rows ``e_r = (sin, cos, 0)``, ``e_phi = (cos, -sin, 0)``,
    ``e_down = (0, 0, -1)``.
    """
    values = np.asarray(values, dtype=float).T
    derivatives = np.asarray(derivatives, dtype=float).T
    radial, depth, over = derivatives[:10], derivatives[10:20], derivatives[20:27]
    azimuth = np.asarray(azimuth, dtype=float)
    cos1, sin1 = np.cos(azimuth), np.sin(azimuth)
    cos2, sin2 = np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)
    out = np.empty((azimuth.size, np.shape(packing)[0], 6))
    for i, (m1, m2, m3, m4, m5, e) in enumerate(np.asarray(packing, dtype=float)):
        ps_ss = m1 * sin2 + m4 * cos2
        sh_ss = m1 * cos2 - m4 * sin2
        ps_ds = m2 * cos1 + m5 * sin1
        sh_ds = m2 * sin1 - m5 * cos1
        fields = []
        for t in (radial, depth):
            fields.append((ps_ss * t[0] + ps_ds * t[3] + m3 * t[6] + e * t[8],
                           ps_ss * t[1] + ps_ds * t[4] + m3 * t[7] + e * t[9],
                           sh_ss * t[2] + sh_ds * t[5]))
        (uz_r, ur_r, ut_r), (uz_z, ur_z, ut_z) = fields
        # (d_phi u_t + u_r) / r, (d_phi u_r - u_t) / r and d_phi u_z / r, from the 1/r columns
        hoop = ps_ss * over[0] + ps_ds * over[3] + m3 * over[5] + e * over[6]
        twist = sh_ss * over[1] - sh_ds * over[3]
        tilt = 2.0 * sh_ss * over[2] - sh_ds * over[4]
        e_rr = ur_r
        e_pp = hoop
        e_zz = uz_z
        e_rp = 0.5 * (ut_r + twist)
        e_rz = 0.5 * (ur_z + uz_r)
        e_pz = 0.5 * (ut_z + tilt)
        out[:, i, 0] = sin1 * sin1 * e_rr + 2.0 * sin1 * cos1 * e_rp + cos1 * cos1 * e_pp
        out[:, i, 1] = cos1 * cos1 * e_rr - 2.0 * sin1 * cos1 * e_rp + sin1 * sin1 * e_pp
        out[:, i, 2] = e_zz
        out[:, i, 3] = sin1 * cos1 * (e_rr - e_pp) + (cos1 * cos1 - sin1 * sin1) * e_rp
        out[:, i, 4] = -(sin1 * e_rz + cos1 * e_pz)
        out[:, i, 5] = -(cos1 * e_rz - sin1 * e_pz)
    return out


def _tent_levels(vertices, j, interfaces, count):
    """Gauss-Legendre levels and weights for vertex ``j``'s tent, split where the integrand kinks.

    The level-``t`` triangle has vertex ``j`` fixed and the others at
    ``v_j + (1 - t)(v_k - v_j)``, so it stops crossing an interface ``z_I`` at
    ``1 - t = (z_I - z_j) / (z_k - z_j)``. One rule across that is slow (a triangle
    crossing 2 km: 2 points 3e-3, 4 points 7e-4 at 30 sizes, against 1e-5 inside one
    layer); a rule on each side is not.
    """
    depth = -np.asarray(vertices, dtype=float)[:, 2]
    cuts = []
    for k in range(3):
        if k == j or depth[k] == depth[j]:
            continue
        for z_i in np.asarray(interfaces, dtype=float):
            u = (z_i - depth[j]) / (depth[k] - depth[j])
            if 1e-9 < u < 1.0 - 1e-9:
                cuts.append(1.0 - u)
    edges = np.unique(np.concatenate([[0.0, 1.0], cuts]))
    t, w = _gauss01(count)
    levels = np.concatenate([a + (b - a) * t for a, b in zip(edges[:-1], edges[1:])])
    weights = np.concatenate([(b - a) * w for a, b in zip(edges[:-1], edges[1:])])
    return levels, weights


#: Most triangle-receiver pairs (times the strain's difference stencil) in one batched call.
BATCH_PAIRS = 2_000_000


def _tent_levelset_batched(vertices, x, y, z, near, out, components, kernel, gain, min_points,
                           interfaces, poisson, split):
    """:func:`tent_levelset` with every shrunk piece of a receiver group in one batched call."""
    strain = kernel is tde_strain_hs_unit
    if not strain and kernel is not tde_disp_hs_unit:
        raise ValueError("batched tents take tde_disp_hs_unit or tde_strain_hs_unit")
    area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    size = np.sqrt(2.0 * area)
    distance = _distance_to_triangle(vertices, np.column_stack([x[near], y[near], z[near]]))
    requested = np.ceil(float(gain) * size / np.maximum(distance, 1e-3 * size)) + 1
    points = np.clip(requested, int(min_points), TENT_MAX_POINTS).astype(int)
    alone = (requested > TENT_MAX_POINTS) if split else np.zeros(near.size, bool)

    def pieces(level_sets):
        tris, nus, tents, weights = [], [], [], []
        for j, (levels, w) in enumerate(level_sets):
            for level, weight in zip(levels, w):
                shrunk = vertices[j] + (1.0 - level) * (vertices - vertices[j])
                cut = split_by_layers(shrunk, interfaces)
                nu = poisson(np.array([mid for _, mid in cut]))
                for (tri, _), nu_piece in zip(cut, nu):
                    tris.append(tri)
                    nus.append(nu_piece)
                    tents.append(j)
                    weights.append(weight)
        return np.array(tris), np.array(nus), np.array(tents), np.array(weights)

    def accumulate(at, tris, nus, tents, weights, d_parent):
        stencil = 8 if strain else 1
        chunk = max(1, int(BATCH_PAIRS // max(1, at.size * stencil)))
        for start in range(0, tris.shape[0], chunk):
            sl = slice(start, start + chunk)
            if strain:
                field = tde_strain_hs_unit_many(x[at], y[at], z[at], tris[sl], nus[sl],
                                                h=STRAIN_STEP * d_parent)
            else:
                field = tde_disp_hs_unit_many(x[at], y[at], z[at], tris[sl], nus[sl])
            field = field[:, components]                                # (t, ncomp, width, n_at)
            for j in range(3):
                pick = tents[sl] == j
                if pick.any():
                    out[at, j] += np.einsum("t,tcwn->ncw", weights[sl][pick], field[pick])

    for k in np.flatnonzero(alone):
        i = near[k]
        receiver = np.array([x[i], y[i], z[i]])
        sets = [_tent_levels_split(vertices, j, interfaces, receiver, SPLIT_POINTS) for j in range(3)]
        accumulate(np.array([i]), *pieces(sets), distance[k:k + 1])
    keep = ~alone
    for count in np.unique(points[keep]):
        group = keep & (points == count)
        sets = [_tent_levels(vertices, j, interfaces, int(count)) for j in range(3)]
        accumulate(near[group], *pieces(sets), distance[group])


def _tent_levels_split(vertices, j, interfaces, receiver, count):
    """:func:`_tent_levels` for one receiver, also split where the shrinking triangle passes it.

    The level-``t`` triangle's moving edge is the level line ``tent_j = t``, so it crosses the
    receiver's projection at ``t* = tent_j(projection)``; its moving vertex ``k`` comes closest
    at ``1 - (p - v_j) . (v_k - v_j) / |v_k - v_j|^2``. ``count`` points on each piece.
    """
    from ._tde import barycentric

    v = np.asarray(vertices, dtype=float)
    normal = np.cross(v[1] - v[0], v[2] - v[0])
    normal /= np.linalg.norm(normal)
    projection = receiver - ((receiver - v[0]) @ normal) * normal
    cuts = [float(barycentric(v, projection[None])[0, j])]
    for k in range(3):
        if k != j:
            edge = v[k] - v[j]
            cuts.append(1.0 - float((receiver - v[j]) @ edge / (edge @ edge)))
    base, _ = _tent_levels(vertices, j, interfaces, 1)       # one point a piece: recover the kinks
    edges = [0.0, 1.0] + [c for c in cuts if 1e-9 < c < 1.0 - 1e-9]
    depth = -v[:, 2]
    for k in range(3):
        if k == j or depth[k] == depth[j]:
            continue
        for z_i in np.asarray(interfaces, dtype=float):
            u = (z_i - depth[j]) / (depth[k] - depth[j])
            if 1e-9 < u < 1.0 - 1e-9:
                edges.append(1.0 - u)
    edges = np.unique(edges)
    t, w = _gauss01(count)
    levels = np.concatenate([a + (b - a) * t for a, b in zip(edges[:-1], edges[1:])])
    weights = np.concatenate([(b - a) * w for a, b in zip(edges[:-1], edges[1:])])
    return levels, weights


#: Gauss-Legendre points on each piece of a split level integral (:func:`tent_levelset`).
SPLIT_POINTS = 16

_GAUSS = {}


def _gauss01(count):
    """Gauss-Legendre points and weights on ``[0, 1]``, cached."""
    if count not in _GAUSS:
        t, w = np.polynomial.legendre.leggauss(count)
        _GAUSS[count] = (0.5 * (t + 1.0), 0.5 * w)
    return _GAUSS[count]


_RULES = {}


def _rule(order):
    """:func:`~slipsolve.layered._triangle_quadrature`'s rule as barycentric rows, cached.

    The same points (sub-triangle centroids) and equal weights; building them once
    per order instead of once per piece and receiver group took 14% of an
    interface solve.
    """
    if order not in _RULES:
        unit = np.eye(3)
        points, share = _triangle_quadrature(unit[0], unit[1], unit[2], order)
        _RULES[order] = (points, share)
    return _RULES[order]


def _refuse_nodal(basis):
    if basis is not None and getattr(basis, "name", "element") != "element":
        raise ValueError("LayeredTDE returns element columns; project them onto the basis "
                         "afterwards, as SlipInversion does for the half-space engine")
