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

from ._tde import _element_basis, tde_disp_hs, tde_disp_hs_unit
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
            area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
            size = np.sqrt(2.0 * area)
            distance = _distance_to_triangle(vertices, np.column_stack([x[near], y[near], z[near]]))
            points = np.clip(np.ceil(TENT_GAIN * size / np.maximum(distance, 1e-3 * size)) + 1,
                             2, TENT_MAX_POINTS).astype(int)
            for count in np.unique(points):
                at = near[points == count]
                for j in range(3):
                    levels, weights = _tent_levels(vertices, j, self._interfaces, int(count))
                    for level, weight in zip(levels, weights):
                        shrunk = vertices[j] + (1.0 - level) * (vertices - vertices[j])
                        cut = split_by_layers(shrunk, self._interfaces)
                        nu = self.tables.poisson(np.array([mid for _, mid in cut]))
                        for (tri, _), nu_piece in zip(cut, nu):
                            u = tde_disp_hs_unit(x[at], y[at], z[at], *tri, nu_piece)[components]
                            out[at, j] += weight * np.moveaxis(u, -1, 0)
            out[near] += self._remainder(pieces, packing, x[near], y[near], z[near], parent=vertices)
        if far.any():
            out[far] = self._remainder(pieces, packing, x[far], y[far], z[far], parent=vertices,
                                       whole=True)
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

    def _remainder(self, pieces, packing, x, y, z, parent=None, whole=False):
        """The remainder's integral over the pieces, ``(npts, n_components, 3)``.

        Every (receiver, quadrature point) pair of a piece goes through one table
        lookup, whatever order each receiver was given, and the receivers' depth
        stencils are worked out once for all the pieces. With ``parent`` (the whole
        triangle) each point is also weighted by the three tents, and the result is
        ``(npts, 3, n_components, 3)``. With ``whole`` the receivers are at the
        surface and the table is the whole surface field, not the remainder: point
        sources, ordered by the point-source rule.
        """
        n = x.size
        tents = parent is not None
        out = np.zeros((n, 3, packing.shape[0], 3) if tents else (n, packing.shape[0], 3))
        tables = self.tables
        # Receivers at the surface read the fine 2-D surface tables when there are any;
        # the 3-D table is too coarse in depth for a point source seen from the surface.
        at_surface = (z == 0.0) & tables.has_surface
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
            pair = _combine(values, np.arctan2(de, dn), packing)
            pair *= np.concatenate(weights)[:, None, None]
            if not tents:
                for i in range(packing.shape[0]):
                    for c in range(3):
                        out[:, i, c] += np.bincount(rec, pair[:, i, c], minlength=n)
                continue
            phi = _barycentric(parent, points)
            for j in range(3):
                for i in range(packing.shape[0]):
                    for c in range(3):
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


def _distance_to_triangle(triangle, points):
    """Each point's distance to the nearest point of the triangle, ``(npoints,)``."""
    v0, v1, v2 = triangle
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    height = (points - v0) @ normal
    inside = np.all(_barycentric(triangle, points - np.outer(height, normal)) >= 0.0, axis=1)
    edge = np.full(points.shape[0], np.inf)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        t = np.clip((points - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        edge = np.minimum(edge, np.linalg.norm(points - (a + np.outer(t, b - a)), axis=1))
    return np.where(inside, np.abs(height), edge)


def _barycentric(triangle, points):
    """Barycentric coordinates of points in the triangle's plane, ``(npoints, 3)``."""
    v0 = triangle[0]
    e1, e2 = triangle[1] - v0, triangle[2] - v0
    gram = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
    rhs = np.stack([(points - v0) @ e1, (points - v0) @ e2])
    a, b = np.linalg.solve(gram, rhs)
    return np.column_stack([1.0 - a - b, a, b])


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
