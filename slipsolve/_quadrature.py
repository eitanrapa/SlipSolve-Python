"""Point sources on triangles: the quadrature rule, tents and moment tensors.

Shared by the layered engine's far path (`slipsolve.layered_tde`), the voxel engine
(`slipsolve.heterogeneity`), the viscoelastic tables (`slipsolve.relaxation`), the
nodal basis and the moment integral.
"""

import numpy as np

#: Constant in the error model `err = C * (k * r / L) ** -2` of an order-`k` rule on
#: an element of size `L` seen from distance `r`, fitted against a fixed order-40
#: rule over `r/L` 1.5-35 and `k` 1-9 (error taken as the 2-norm of the ENU
#: displacement vector, so per-component zero crossings cannot inflate it).
#:
#: The exponent is 2, not 1: a triangle's first moment vanishes about its own
#: centroid, so the centroid rule's leading error term is second order (the fitted
#: exponent is -2.14). A law linear in `k * r / L` would make the delivered error
#: scale as `tolerance ** 2`, so the tolerance would not mean what it says.
_QUADRATURE_CONSTANT = 0.312

#: Fit scatter is 1.89x at one sigma, so a model centred on the median leaves
#: half of all pairs above the tolerance they asked for. This multiplies `C`
#: before the solve, which raises the order by `sqrt` of itself: 4x here takes
#: the fraction of pairs exceeding their tolerance from about half to about 6%.
_QUADRATURE_SAFETY = 4.0


def _tent(vertices, triangle, node, points):
    """The nodal tent function of `node`, sampled inside its own triangle."""
    from .basis import _barycentric_weight

    return _barycentric_weight(vertices, triangle, node, points)


def _slip_directions(normal):
    """The element's strike and dip unit vectors, in the order the columns use.

    The same `Vstrike = cross(eZ, Vnorm)`, `Vdip = cross(Vnorm, Vstrike)` the
    triangular-dislocation solution uses, so column `k` of a point-source design
    matrix means precisely what column `k` of a half-space one means.
    """
    # The same expression `_tde._element_basis` uses, reproduced from the normal
    # alone because there are no vertices to hand here.
    e_z = np.array([0.0, 0.0, 1.0])
    vstrike = np.cross(e_z, normal)
    if np.linalg.norm(vstrike) == 0:
        vstrike = np.array([0.0, 1.0, 0.0]) * normal[2]
    vstrike = vstrike / np.linalg.norm(vstrike)
    return vstrike, np.cross(normal, vstrike)


def _moment_components(normal, slip):
    """Pack a unit dislocation's moment tensor into five components.

    A dislocation of area `A` and slip `s` on a plane with unit normal `n`
    in the unit direction `d` has moment tensor `mu*A*s*(n(x)d + d(x)n)`. The
    tables already carry the `mu`, so what is needed here is the dimensionless
    tensor, converted to north/east/down and projected onto the basis the three
    tabulated source types (strike-slip, dip-slip, CLVD) span, EDCMP's:

        m1 = M_ne      m2 = M_nd      m3 = M_dd
        m4 = (M_nn - M_ee) / 2        m5 = M_ed

    Five components, not six, because a shear dislocation's tensor is traceless
    -- `n` and `d` are perpendicular -- so `M_nn` and `M_ee` are recovered
    from `m3` and `m4` together.
    """
    n = np.asarray(normal, dtype=float)
    d = np.asarray(slip, dtype=float)
    tensor = np.outer(n, d) + np.outer(d, n)

    # east/north/up -> north/east/down.
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    m = swap @ tensor @ swap.T
    return (m[0, 1], m[0, 2], m[2, 2], 0.5 * (m[0, 0] - m[1, 1]), m[1, 2])


def _triangle_quadrature(p1, p2, p3, order):
    """Equal-weight point sources on a triangle: `order**2` sub-triangle centroids.

    Subdividing each edge into `order` gives `order**2` congruent
    sub-triangles -- `order*(order+1)/2` pointing the same way as the parent and
    `order*(order-1)/2` inverted -- and their centroids with equal weights is a
    second-order rule, which is exactly the exponent `_QUADRATURE_CONSTANT`'s error
    model assumes. Keeping both orientations is what buys that order.

    The reference's rule (SlipSolve-Curve) is different, and first order. It is
    `interpolate_triangle2(v1,v2,v3,15)`: the lattice points strictly inside the
    triangle, `(n-1)(n-2)/2` of them (which is where 91 comes from, as a
    consequence of `n = 15` rather than as a target), weighted equally by its
    caller `comb_green_nodes_xyz.m` (`pmoment = area/npatch`), with a tent applied
    through `taperTriangle`. It converges as `N**-0.48` in point count against this
    rule's `N**-1.00`. The cause is the equal weights, not the excluded boundary:
    a lattice's nodes own unequal areas -- an interior node is shared by six
    sub-triangles, an edge node by three, a corner by one -- and only weights in
    that `6:3:1` ratio (the P1 vertex rule) restore second order. Centroids avoid
    the question: a sub-triangle centroid is interior to exactly one cell and owns
    exactly `area/k**2`, so there is no sharing to weight for.
    """
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    order = max(1, int(order))
    if order == 1:
        return ((p1 + p2 + p3) / 3.0)[None, :], np.array([1.0])

    points = []
    for a in range(order):
        for b in range(order - a):
            c = order - 1 - a - b
            points.append(((a + 1 / 3) * p1 + (b + 1 / 3) * p2 + (c + 1 / 3) * p3) / order)
            if c >= 1:
                points.append(((a + 2 / 3) * p1 + (b + 2 / 3) * p2
                               + (c - 1 / 3) * p3) / order)
    points = np.array(points)
    return points, np.full(points.shape[0], 1.0 / points.shape[0])


def _sub_triangles(order):
    """The `order**2` sub-triangles of `_triangle_quadrature`, as vertices.

    `(order**2, 3, 3)` barycentric coordinates: sub-triangle, its vertex, parent
    vertex. Same lattice and same order as `_triangle_quadrature`, whose points
    are exactly these triangles' centroids -- so a caller that needs a piece's
    extent as well as its centroid (`SlipModel.moment`, which averages the
    rigidity over each piece's depth range) stays on the rule the rest of the
    package uses.
    """
    order = max(1, int(order))
    pieces = []
    for a in range(order):
        for b in range(order - a):
            c = order - 1 - a - b
            pieces.append([[a + 1, b, c], [a, b + 1, c], [a, b, c + 1]])
            if c >= 1:
                pieces.append([[a + 1, b + 1, c - 1], [a + 1, b, c], [a, b + 1, c]])
    return np.array(pieces, dtype=float) / order
