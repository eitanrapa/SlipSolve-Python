"""Triangular dislocation elements: surface displacement in an elastic half-space.

A vectorized numpy port of Nikkhoo & Walter (2015), "Triangular dislocation: an
analytical, artefact-free solution", *Geophysical Journal International* **201**,
1117-1139, doi:10.1093/gji/ggv035 -- the reference implementation being the
author's ``TDdispHS.m``.

A triangular element is what lets a fault curve. A rectangular (Okada) patch can
only tile a plane, so a curved or segmented fault built from rectangles has gaps
and overlaps along every bend, and those produce spurious near-field
displacement. Triangles tile any surface exactly.

The half-space solution is assembled from three pieces, exactly as the reference
does: the full-space dislocation, its mirror image across the free surface, and a
harmonic correction that cancels the normal tractions the first two leave on that
surface.

Conventions, which are the reference's and are load-bearing everywhere
downstream:

* Coordinates are east, north, up in a flat Cartesian frame, with the free
  surface at ``z = 0``. **All z must be <= 0.**
* The slip components are named for the element's own strike and dip:
  ``Vnorm = normalize(cross(P2 - P1, P3 - P1))``,
  ``Vstrike = cross(eZ, Vnorm)``, ``Vdip = cross(Vnorm, Vstrike)``. Winding
  therefore fixes the sign of both strike- and dip-slip -- see
  :class:`~slipsolve.mesh.FaultMesh`, which pins it against the fault
  trace rather than against a fixed axis.

The original file carries an MIT permission grant; this port is a derivative work
and the copyright notice is reproduced here.

Copyright (c) 2014 Mehdi Nikkhoo. Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions: the above
copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
WARRANTY OF ANY KIND.
"""

import numpy as np


def tde_disp_hs(x, y, z, p1, p2, p3, ss, ds, ts, nu=0.25):
    """Displacement from one triangular dislocation in an elastic half-space.

    ``x``, ``y``, ``z`` are the observation coordinates (any common shape, ``z <=
    0``); ``p1``, ``p2``, ``p3`` are the vertices as length-3 sequences; ``ss``,
    ``ds``, ``ts`` are the strike-, dip- and tensile-slip components. Returns
    ``(ue, un, uv)`` flattened to 1-D, in the same units as the slip.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    p3 = np.asarray(p3, dtype=float).ravel()

    if np.any(z > 0) or p1[2] > 0 or p2[2] > 0 or p3[2] > 0:
        raise ValueError("Half-space solution: z coordinates must be <= 0")
    if p1[2] == 0 and p2[2] == 0 and p3[2] == 0:
        # A TD lying entirely in the free surface radiates nothing; the image
        # and the main dislocation cancel identically.
        zero = np.zeros(x.size)
        return zero, zero.copy(), zero.copy()

    # An observation sitting exactly on the dislocation surface is a genuine
    # singularity, and the solution is *meant* to come back non-finite there --
    # slipsolve.greens catches it and tells the caller to exclude
    # observations near the trace. Divide-by-zero warnings from that case are
    # noise, not news. (numpy's error state is per-thread, so this is safe under
    # the thread pool that assembles the design matrix.)
    with np.errstate(divide="ignore", invalid="ignore"):
        ue_ms, un_ms, uv_ms = _disp_fs(x, y, z, p1, p2, p3, ss, ds, ts, nu)
        ue_fsc, un_fsc, uv_fsc = _disp_har_func(x, y, z, p1, p2, p3, ss, ds, ts, nu)

        # The image dislocation is the element mirrored through z = 0. Copy
        # first: the reference mutates its inputs, which is harmless in MATLAB's
        # copy-on-write world and a bug here.
        q1, q2, q3 = p1.copy(), p2.copy(), p3.copy()
        q1[2], q2[2], q3[2] = -q1[2], -q2[2], -q3[2]
        ue_is, un_is, uv_is = _disp_fs(x, y, z, q1, q2, q3, ss, ds, ts, nu)

    return (ue_ms + ue_is + ue_fsc,
            un_ms + un_is + un_fsc,
            uv_ms + uv_is + uv_fsc)


def tde_disp_hs_unit(x, y, z, p1, p2, p3, nu=0.25):
    """:func:`tde_disp_hs` for unit strike-, dip- and tensile-slip at once: ``(3, 3, npts)``.

    Axis 0 is the slip component (strike, dip, tensile), axis 1 east/north/up. Every
    term of the solution is a Burgers-vector component times a function of the
    geometry alone, so the three slips share all the geometry and the cost is
    close to one call's, not three. Equal to three :func:`tde_disp_hs` calls to
    rounding (``tests/test_slip_tde.py``).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    p3 = np.asarray(p3, dtype=float).ravel()
    if np.any(z > 0) or p1[2] > 0 or p2[2] > 0 or p3[2] > 0:
        raise ValueError("Half-space solution: z coordinates must be <= 0")
    if p1[2] == 0 and p2[2] == 0 and p3[2] == 0:
        return np.zeros((3, 3, x.size))
    unit = np.eye(3)[:, :, None]                     # (slip, component, 1)
    ss, ds, ts = unit[:, 0], unit[:, 1], unit[:, 2]  # each (3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ms = _disp_fs_m(x, y, z, p1, p2, p3, ss, ds, ts, nu)
        fsc = _disp_har_func_m(x, y, z, p1, p2, p3, ss, ds, ts, nu)
        q1, q2, q3 = p1.copy(), p2.copy(), p3.copy()
        q1[2], q2[2], q3[2] = -q1[2], -q2[2], -q3[2]
        image = _disp_fs_m(x, y, z, q1, q2, q3, ss, ds, ts, nu)
    return ms + image + fsc


#: Central-difference step of :func:`tde_strain_hs_unit`, as a fraction of each point's
#: distance to the triangle. Measured against ``cutde``'s analytic strain (2026-11-04; four
#: triangles, vertical, dipping, shallow, one reaching the surface): worst over components
#: and slips, relative to the point's largest strain, 1e-4 gives <= 1.4e-7 within 10
#: triangle sizes and <= 1e-5 beyond 100. A larger step loses to truncation near the
#: triangle (1e-3: 1e-6 to 1e-5 everywhere), a smaller one to round-off far away (1e-5:
#: 1.6e-4 beyond 100 sizes, where the displacement is a small difference of large terms).
#: Richardson extrapolation of 1e-3 and 2e-3 agrees with ``cutde`` to 1e-9 within 10 sizes,
#: so the reference is good there; beyond 100 sizes both carry ~1e-6 of round-off.
STRAIN_STEP = 1e-4

#: The smallest step :func:`tde_strain_hs_unit` takes, in units in the last place of the
#: coordinates; a point whose step would be smaller is on the triangle and gets NaN.
STRAIN_MIN_STEP_ULPS = 1e4

#: Order of the six strain components :func:`tde_strain_hs_unit` returns, ``cutde``'s:
#: tensor components (``xy`` is half the engineering shear), east/north/up.
STRAIN_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def tde_strain_hs_unit(x, y, z, p1, p2, p3, nu=0.25, step=STRAIN_STEP):
    """Strain of unit strike-, dip- and tensile-slip on one triangle in a half-space: ``(3, 6, npts)``.

    Axis 0 is the slip component (strike, dip, tensile) as :func:`tde_disp_hs_unit` has
    it, axis 1 the components of :data:`STRAIN_COMPONENTS`. Units: strain per unit slip.

    By central differences of :func:`tde_disp_hs_unit`, with a step of ``step`` times each
    point's distance to the triangle (:data:`STRAIN_STEP`), all stencil points in one call.
    A point closer to the free surface than its step takes the one-sided second-order
    difference downward for ``d/dz``, so points at ``z = 0`` are fine; there the field is
    traction-free. A point on the triangle itself has no strain (it is singular on the
    edges and two-valued on the face) and gets NaN, as does one so close that its step
    would be under :data:`STRAIN_MIN_STEP_ULPS` units in the last place of its coordinates.

    Nikkhoo & Walter's ``TDstressHS`` gives the same analytically. It is not ported: these
    differences cost six displacement calls, agree with ``cutde`` (an independent port of
    it) to 1e-5 or better, and the kernels they feed carry 1e-4 to 1e-3 of table error.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    tri = np.array([np.asarray(p, dtype=float).ravel() for p in (p1, p2, p3)])
    if np.any(z > 0) or np.any(tri[:, 2] > 0):
        raise ValueError("Half-space solution: z coordinates must be <= 0")
    n = x.size
    h = float(step) * distance_to_triangle(tri, np.column_stack([x, y, z]))
    # A difference needs a step well above the coordinates' rounding: a point within
    # rounding of the triangle is on it (at 100 km from the origin, within ~1.5 mm).
    scale = np.maximum(np.abs(np.column_stack([x, y, z])).max(axis=1), np.abs(tri).max())
    h = np.where(h > STRAIN_MIN_STEP_ULPS * np.spacing(scale), h, np.nan)
    central = np.flatnonzero(z + np.nan_to_num(h) <= 0.0)
    below = np.setdiff1d(np.arange(n), central, assume_unique=True)
    xs = [x + h, x - h, x, x, x, x[central], x[below], x[below]]
    ys = [y, y, y + h, y - h, y, y[central], y[below], y[below]]
    zs = [z, z, z, z, z - h, z[central] + h[central], z[below], z[below] - 2.0 * h[below]]
    sizes = [n, n, n, n, n, central.size, below.size, below.size]
    stencil = [np.concatenate(v) for v in (xs, ys, zs)]
    finite = np.isfinite(stencil[0]) & np.isfinite(stencil[1]) & np.isfinite(stencil[2])
    u = np.full((3, 3, finite.size), np.nan)
    if finite.any():
        u[:, :, finite] = tde_disp_hs_unit(*(v[finite] for v in stencil), *tri, nu)
    xp, xm, yp, ym, zm, zp, z0, zmm = np.split(u, np.cumsum(sizes)[:-1], axis=-1)
    grad = np.empty((3, 3, 3, n))              # slip, displacement component, derivative
    with np.errstate(divide="ignore", invalid="ignore"):
        grad[:, :, 0] = (xp - xm) / (2.0 * h)
        grad[:, :, 1] = (yp - ym) / (2.0 * h)
        grad[:, :, 2, central] = (zp - zm[:, :, central]) / (2.0 * h[central])
        grad[:, :, 2, below] = (3.0 * z0 - 4.0 * zm[:, :, below] + zmm) / (2.0 * h[below])
    sym = 0.5 * (grad + np.swapaxes(grad, 1, 2))
    return np.stack([sym[:, 0, 0], sym[:, 1, 1], sym[:, 2, 2],
                     sym[:, 0, 1], sym[:, 0, 2], sym[:, 1, 2]], axis=1)


def distance_to_triangle(triangle, points):
    """Each point's distance to the nearest point of the triangle, ``(npoints,)``.

    ``triangle`` is ``(3, 3)`` and ``points`` ``(npoints, 3)``, in any one Cartesian frame.
    """
    triangle = np.asarray(triangle, dtype=float)
    points = np.asarray(points, dtype=float)
    v0, v1, v2 = triangle
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    height = (points - v0) @ normal
    inside = np.all(barycentric(triangle, points - np.outer(height, normal)) >= 0.0, axis=1)
    edge = np.full(points.shape[0], np.inf)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        t = np.clip((points - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        edge = np.minimum(edge, np.linalg.norm(points - (a + np.outer(t, b - a)), axis=1))
    return np.where(inside, np.abs(height), edge)


def barycentric(triangle, points):
    """Barycentric coordinates of points in the triangle's plane, ``(npoints, 3)``."""
    triangle = np.asarray(triangle, dtype=float)
    points = np.asarray(points, dtype=float)
    v0 = triangle[0]
    e1, e2 = triangle[1] - v0, triangle[2] - v0
    gram = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
    rhs = np.stack([(points - v0) @ e1, (points - v0) @ e2])
    a, b = np.linalg.solve(gram, rhs)
    return np.column_stack([1.0 - a - b, a, b])


def _rotate_m(a, u, v, w):
    """``a @ [u; v; w]`` for components with a leading slip axis; ``(3, m, npts)``."""
    return np.stack([a[i, 0] * u + a[i, 1] * v + a[i, 2] * w for i in range(3)])


def _disp_fs_m(x, y, z, p1, p2, p3, ss, ds, ts, nu):
    """:func:`_disp_fs` with ``ss``, ``ds``, ``ts`` of shape ``(m, 1)``; returns ``(m, 3, npts)``."""
    bx, by, bz = ts, ss, ds
    m = bx.shape[0]
    vnorm, vstrike, vdip = _element_basis(p1, p2, p3, image=True)
    at = np.column_stack([vnorm, vstrike, vdip]).T
    xt, yt, zt = _coord_trans(x - p2[0], y - p2[1], z - p2[2], at)
    q1 = np.asarray(_coord_trans(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2], at)).ravel()
    q2 = np.zeros(3)
    q3 = np.asarray(_coord_trans(p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2], at)).ravel()
    e12 = (q2 - q1) / np.linalg.norm(q2 - q1)
    e13 = (q3 - q1) / np.linalg.norm(q3 - q1)
    e23 = (q3 - q2) / np.linalg.norm(q3 - q2)
    a_ang = np.arccos(np.clip(e12 @ e13, -1.0, 1.0))
    b_ang = np.arccos(np.clip(-e12 @ e23, -1.0, 1.0))
    c_ang = np.arccos(np.clip(e23 @ e13, -1.0, 1.0))
    trimode = _trimodefinder(yt, zt, xt, q1[1:], q2[1:], q3[1:])
    u = np.zeros((m, xt.size))
    v = np.zeros((m, xt.size))
    w = np.zeros((m, xt.size))
    for mask, sign in ((trimode == 1, 1.0), (trimode == -1, -1.0)):
        if not mask.any():
            continue
        xs, ys, zs = xt[mask], yt[mask], zt[mask]
        u1, v1, w1 = _setup_d_m(xs, ys, zs, a_ang, bx, by, bz, nu, q1, -sign * e13)
        u2, v2, w2 = _setup_d_m(xs, ys, zs, b_ang, bx, by, bz, nu, q2, sign * e12)
        u3, v3, w3 = _setup_d_m(xs, ys, zs, c_ang, bx, by, bz, nu, q3, sign * e23)
        u[:, mask] = u1 + u2 + u3
        v[:, mask] = v1 + v2 + v3
        w[:, mask] = w1 + w2 + w3
    on_side = trimode == 0
    if on_side.any():
        u[:, on_side] = v[:, on_side] = w[:, on_side] = np.nan
    a = np.column_stack([-xt, q1[1] - yt, q1[2] - zt])
    b = np.column_stack([-xt, -yt, -zt])
    c = np.column_stack([-xt, q3[1] - yt, q3[2] - zt])
    na = np.sqrt((a * a).sum(axis=1))
    nb = np.sqrt((b * b).sum(axis=1))
    nc = np.sqrt((c * c).sum(axis=1))
    fi_n = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
            - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
            + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
    fi_d = (na * nb * nc + (a * b).sum(axis=1) * nc
            + (a * c).sum(axis=1) * nb + (b * c).sum(axis=1) * na)
    fi_n = np.where(fi_n == 0, -0.0, fi_n)
    fi = -2.0 * np.arctan2(fi_n, fi_d) / (4.0 * np.pi)
    u = bx * fi + u
    v = by * fi + v
    w = bz * fi + w
    return np.moveaxis(_rotate_m(np.column_stack([vnorm, vstrike, vdip]), u, v, w), 0, 1)


def _setup_d_m(x, y, z, alpha, bx, by, bz, nu, tri_vertex, side_vec):
    """:func:`_setup_d` with a leading slip axis on the Burgers vector and the result."""
    a = np.array([[side_vec[2], -side_vec[1]],
                  [side_vec[1], side_vec[2]]])
    r1 = a @ np.vstack([y - tri_vertex[1], z - tri_vertex[2]])
    y1, z1 = r1[0], r1[1]
    by1 = a[0, 0] * by + a[0, 1] * bz
    bz1 = a[1, 0] * by + a[1, 1] * bz
    u, v0, w0 = _ang_dis_disp(x, y1, z1, -np.pi + alpha, bx, by1, bz1, nu)
    return u, a[0, 0] * v0 + a[1, 0] * w0, a[0, 1] * v0 + a[1, 1] * w0


def _disp_har_func_m(x, y, z, p1, p2, p3, ss, ds, ts, nu):
    """:func:`_disp_har_func` with a leading slip axis; returns ``(m, 3, npts)``."""
    bx, by, bz = ts, ss, ds
    vnorm, vstrike, vdip = _element_basis(p1, p2, p3, image=False)
    a = np.column_stack([vnorm, vstrike, vdip])
    b_x, b_y, b_z = _rotate_m(a, bx, by, bz)
    total = 0.0
    for pa, pb in ((p1, p2), (p2, p3), (p3, p1)):
        total = total + _ang_setup_fsc_m(x, y, z, b_x, b_y, b_z, pa, pb, nu)
    return total


def _ang_setup_fsc_m(x, y, z, b_x, b_y, b_z, pa, pb, nu):
    """:func:`_ang_setup_fsc` with a leading slip axis; returns ``(m, 3, npts)``."""
    m = b_x.shape[0]
    side_vec = pb - pa
    e_z = np.array([0.0, 0.0, 1.0])
    beta = np.arccos(np.clip(-(side_vec @ e_z) / np.linalg.norm(side_vec), -1.0, 1.0))
    eps = np.finfo(float).eps
    if abs(beta) < eps or abs(np.pi - beta) < eps:
        return np.zeros((m, 3, x.size))
    ey1 = np.array([side_vec[0], side_vec[1], 0.0])
    ey1 = ey1 / np.linalg.norm(ey1)
    ey3 = -e_z
    ey2 = np.cross(ey3, ey1)
    a = np.column_stack([ey1, ey2, ey3])
    y1a, y2a, y3a = _coord_trans(x - pa[0], y - pa[1], z - pa[2], a)
    y1ab, y2ab, y3ab = _coord_trans(side_vec[0], side_vec[1], side_vec[2], a)
    y1b, y2b, y3b = y1a - y1ab[0], y2a - y2ab[0], y3a - y3ab[0]
    b1, b2, b3 = _rotate_m(a, b_x, b_y, b_z)
    va = np.zeros((3, m, x.size))
    vb = np.zeros((3, m, x.size))
    sel = (beta * y1a) >= 0
    for mask, ang in ((sel, -np.pi + beta), (~sel, beta)):
        if not mask.any():
            continue
        va[:, :, mask] = _ang_dis_disp_fsc(y1a[mask], y2a[mask], y3a[mask], ang,
                                           b1, b2, b3, nu, -pa[2])
        vb[:, :, mask] = _ang_dis_disp_fsc(y1b[mask], y2b[mask], y3b[mask], ang,
                                           b1, b2, b3, nu, -pb[2])
    d = vb - va
    return np.moveaxis(_rotate_m(a.T, d[0], d[1], d[2]), 0, 1)


def _element_basis(p1, p2, p3, image=False):
    """Unit normal, strike and dip vectors of a TD, the reference's way.

    A horizontal element is the degenerate case: its normal is parallel to
    ``eZ`` so ``cross(eZ, Vnorm)`` vanishes, and the reference falls back to
    pointing strike north or south according to the normal's sign. A *vertical*
    element -- the whole of a strike-slip fault mesh -- is not degenerate at all:
    its normal is horizontal, so the cross product is well conditioned.
    """
    vnorm = np.cross(p2 - p1, p3 - p1)
    vnorm = vnorm / np.linalg.norm(vnorm)

    e_y = np.array([0.0, 1.0, 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    vstrike = np.cross(e_z, vnorm)
    if np.linalg.norm(vstrike) == 0:
        vstrike = e_y * vnorm[2]
        # Horizontal elements only: correct the image dislocation's strike.
        if image and p1[2] > 0:
            vstrike = -vstrike
    vstrike = vstrike / np.linalg.norm(vstrike)
    vdip = np.cross(vnorm, vstrike)
    return vnorm, vstrike, vdip


def _disp_fs(x, y, z, p1, p2, p3, ss, ds, ts, nu):
    """Full-space displacement of a triangular dislocation."""
    bx, by, bz = ts, ss, ds  # tensile, strike-slip, dip-slip

    vnorm, vstrike, vdip = _element_basis(p1, p2, p3, image=True)

    # Into the triangular-dislocation coordinate system, origin at P2.
    at = np.column_stack([vnorm, vstrike, vdip]).T
    xt, yt, zt = _coord_trans(x - p2[0], y - p2[1], z - p2[2], at)
    q1 = np.asarray(_coord_trans(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2], at)).ravel()
    q2 = np.zeros(3)
    q3 = np.asarray(_coord_trans(p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2], at)).ravel()

    e12 = (q2 - q1) / np.linalg.norm(q2 - q1)
    e13 = (q3 - q1) / np.linalg.norm(q3 - q1)
    e23 = (q3 - q2) / np.linalg.norm(q3 - q2)

    a_ang = np.arccos(np.clip(e12 @ e13, -1.0, 1.0))
    b_ang = np.arccos(np.clip(-e12 @ e23, -1.0, 1.0))
    c_ang = np.arccos(np.clip(e23 @ e13, -1.0, 1.0))

    # Which of the two artefact-free angular-dislocation configurations to use.
    trimode = _trimodefinder(yt, zt, xt, q1[1:], q2[1:], q3[1:])
    pos = trimode == 1
    neg = trimode == -1
    on_side = trimode == 0

    u = np.zeros(xt.size)
    v = np.zeros(xt.size)
    w = np.zeros(xt.size)

    for mask, sign in ((pos, 1.0), (neg, -1.0)):
        if not mask.any():
            continue
        xs, ys, zs = xt[mask], yt[mask], zt[mask]
        u1, v1, w1 = _setup_d(xs, ys, zs, a_ang, bx, by, bz, nu, q1, -sign * e13)
        u2, v2, w2 = _setup_d(xs, ys, zs, b_ang, bx, by, bz, nu, q2, sign * e12)
        u3, v3, w3 = _setup_d(xs, ys, zs, c_ang, bx, by, bz, nu, q3, sign * e23)
        u[mask] = u1 + u2 + u3
        v[mask] = v1 + v2 + v3
        w[mask] = w1 + w2 + w3

    if on_side.any():
        u[on_side] = np.nan
        v[on_side] = np.nan
        w[on_side] = np.nan

    # The Burgers function: the solid angle the triangle subtends.
    a = np.column_stack([-xt, q1[1] - yt, q1[2] - zt])
    b = np.column_stack([-xt, -yt, -zt])
    c = np.column_stack([-xt, q3[1] - yt, q3[2] - zt])
    na = np.sqrt((a * a).sum(axis=1))
    nb = np.sqrt((b * b).sum(axis=1))
    nc = np.sqrt((c * c).sum(axis=1))

    fi_n = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
            - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
            + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
    fi_d = (na * nb * nc + (a * b).sum(axis=1) * nc
            + (a * c).sum(axis=1) * nb + (b * c).sum(axis=1) * na)
    # atan2 distinguishes +0.0 from -0.0, and points exactly on the TD plane
    # land on that zero; force the negative branch as the reference does.
    fi_n = np.where(fi_n == 0, -0.0, fi_n)
    fi = -2.0 * np.arctan2(fi_n, fi_d) / (4.0 * np.pi)

    u = bx * fi + u
    v = by * fi + v
    w = bz * fi + w

    return _coord_trans(u, v, w, np.column_stack([vnorm, vstrike, vdip]))


def _disp_har_func(x, y, z, p1, p2, p3, ss, ds, ts, nu):
    """Harmonic correction cancelling the free-surface normal tractions."""
    bx, by, bz = ts, ss, ds
    vnorm, vstrike, vdip = _element_basis(p1, p2, p3, image=False)

    a = np.column_stack([vnorm, vstrike, vdip])
    b_x, b_y, b_z = _coord_trans(bx, by, bz, a)
    b_x, b_y, b_z = float(b_x[0]), float(b_y[0]), float(b_z[0])

    u1, v1, w1 = _ang_setup_fsc(x, y, z, b_x, b_y, b_z, p1, p2, nu)
    u2, v2, w2 = _ang_setup_fsc(x, y, z, b_x, b_y, b_z, p2, p3, nu)
    u3, v3, w3 = _ang_setup_fsc(x, y, z, b_x, b_y, b_z, p3, p1, nu)
    return u1 + u2 + u3, v1 + v2 + v3, w1 + w2 + w3


def _coord_trans(x1, x2, x3, a):
    """Rotate vector components by ``a``; returns three 1-D arrays."""
    x1 = np.atleast_1d(np.asarray(x1, dtype=float)).ravel()
    x2 = np.atleast_1d(np.asarray(x2, dtype=float)).ravel()
    x3 = np.atleast_1d(np.asarray(x3, dtype=float)).ravel()
    r = a @ np.vstack([x1, x2, x3])
    return r[0], r[1], r[2]


def _trimodefinder(x, y, z, p1, p2, p3):
    """Pick the artefact-free angular-dislocation configuration per point.

    ``x``/``y`` are the in-plane TDCS coordinates and ``z`` the off-plane one
    (the caller passes them in that order, not as named). Returns ``1``, ``-1``,
    or ``0`` for points lying on a triangle side.
    """
    det = ((p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]))
    a = ((p2[1] - p3[1]) * (x - p3[0]) + (p3[0] - p2[0]) * (y - p3[1])) / det
    b = ((p3[1] - p1[1]) * (x - p3[0]) + (p1[0] - p3[0]) * (y - p3[1])) / det
    c = 1.0 - a - b

    trimode = np.ones(np.size(x), dtype=int)
    trimode[(a <= 0) & (b > c) & (c > a)] = -1
    trimode[(b <= 0) & (c > a) & (a > b)] = -1
    trimode[(c <= 0) & (a > b) & (b > c)] = -1
    trimode[(a == 0) & (b >= 0) & (c >= 0)] = 0
    trimode[(a >= 0) & (b == 0) & (c >= 0)] = 0
    trimode[(a >= 0) & (b >= 0) & (c == 0)] = 0
    # Only points genuinely in the TD plane are ambiguous; off-plane points that
    # merely project onto a side are fine in configuration I.
    trimode[(trimode == 0) & (z != 0)] = 1
    return trimode


def _setup_d(x, y, z, alpha, bx, by, bz, nu, tri_vertex, side_vec):
    """One angular dislocation's contribution, ADCS <-> TDCS transforms included."""
    a = np.array([[side_vec[2], -side_vec[1]],
                  [side_vec[1], side_vec[2]]])

    r1 = a @ np.vstack([y - tri_vertex[1], z - tri_vertex[2]])
    y1, z1 = r1[0], r1[1]

    r2 = a @ np.array([by, bz])
    by1, bz1 = float(r2[0]), float(r2[1])

    u, v0, w0 = _ang_dis_disp(x, y1, z1, -np.pi + alpha, bx, by1, bz1, nu)

    r3 = a.T @ np.vstack([v0, w0])
    return u, r3[0], r3[1]


def _ang_setup_fsc(x, y, z, b_x, b_y, b_z, pa, pb, nu):
    """Free-surface correction from the angular-dislocation pair on one TD side."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    side_vec = pb - pa
    e_z = np.array([0.0, 0.0, 1.0])
    beta = np.arccos(np.clip(-(side_vec @ e_z) / np.linalg.norm(side_vec), -1.0, 1.0))

    eps = np.finfo(float).eps
    if abs(beta) < eps or abs(np.pi - beta) < eps:
        # A side parallel to the free-surface normal contributes nothing.
        zero = np.zeros(x.size)
        return zero, zero.copy(), zero.copy()

    ey1 = np.array([side_vec[0], side_vec[1], 0.0])
    ey1 = ey1 / np.linalg.norm(ey1)
    ey3 = -e_z
    ey2 = np.cross(ey3, ey1)
    a = np.column_stack([ey1, ey2, ey3])

    y1a, y2a, y3a = _coord_trans(x - pa[0], y - pa[1], z - pa[2], a)
    y1ab, y2ab, y3ab = _coord_trans(side_vec[0], side_vec[1], side_vec[2], a)
    y1b = y1a - y1ab[0]
    y2b = y2a - y2ab[0]
    y3b = y3a - y3ab[0]

    b1, b2, b3 = _coord_trans(b_x, b_y, b_z, a)
    b1, b2, b3 = float(b1[0]), float(b2[0]), float(b3[0])

    v1a = np.zeros(x.size)
    v2a = np.zeros(x.size)
    v3a = np.zeros(x.size)
    v1b = np.zeros(x.size)
    v2b = np.zeros(x.size)
    v3b = np.zeros(x.size)

    # Two configurations keep the calculation artefact-free near the surface.
    sel = (beta * y1a) >= 0
    for mask, ang in ((sel, -np.pi + beta), (~sel, beta)):
        if not mask.any():
            continue
        v1a[mask], v2a[mask], v3a[mask] = _ang_dis_disp_fsc(
            y1a[mask], y2a[mask], y3a[mask], ang, b1, b2, b3, nu, -pa[2])
        v1b[mask], v2b[mask], v3b[mask] = _ang_dis_disp_fsc(
            y1b[mask], y2b[mask], y3b[mask], ang, b1, b2, b3, nu, -pb[2])

    return _coord_trans(v1b - v1a, v2b - v2a, v3b - v3a, a.T)


def _ang_dis_disp(x, y, z, alpha, bx, by, bz, nu):
    """"Incomplete" full-space displacement of an angular dislocation."""
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    eta = y * cos_a - z * sin_a
    zeta = y * sin_a + z * cos_a
    r = np.sqrt(x * x + y * y + z * z)

    # Keep the logarithms real: round-off can push these marginally past r.
    zeta = np.minimum(zeta, r)
    z = np.minimum(z, r)

    k = 1.0 / (8.0 * np.pi * (1.0 - nu))
    rz = r - z
    rzeta = r - zeta

    ux = bx * k * (x * y / r / rz - x * eta / r / rzeta)
    vx = bx * k * (eta * sin_a / rzeta - y * eta / r / rzeta
                   + y * y / r / rz + (1 - 2 * nu) * (cos_a * np.log(rzeta) - np.log(rz)))
    wx = bx * k * (eta * cos_a / rzeta - y / r - eta * z / r / rzeta
                   - (1 - 2 * nu) * sin_a * np.log(rzeta))

    uy = by * k * (x * x * cos_a / r / rzeta - x * x / r / rz
                   - (1 - 2 * nu) * (cos_a * np.log(rzeta) - np.log(rz)))
    vy = by * x * k * (y * cos_a / r / rzeta - sin_a * cos_a / rzeta - y / r / rz)
    wy = by * x * k * (z * cos_a / r / rzeta - cos_a * cos_a / rzeta + 1.0 / r)

    uz = bz * sin_a * k * ((1 - 2 * nu) * np.log(rzeta) - x * x / r / rzeta)
    vz = bz * x * sin_a * k * (sin_a / rzeta - y / r / rzeta)
    wz = bz * x * sin_a * k * (cos_a / rzeta - z / r / rzeta)

    return ux + uy + uz, vx + vy + vz, wx + wy + wz


def _ang_dis_disp_fsc(y1, y2, y3, beta, b1, b2, b3, nu, a):
    """Harmonic contribution of one angular dislocation in a half-space.

    A direct transcription of the reference's ``AngDisDispFSC``. Note that
    ``cotB`` and ``cosB`` are the *same* floating-point value at ``beta = pi/2``
    (since ``sin(pi/2) == 1`` exactly), so the ``cotB * (.../cosB)`` products --
    which are what a horizontal TD side produces, and a vertical fault mesh is
    full of them -- cancel to full precision rather than blowing up.
    """
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)
    # cos/sin, never 1/tan: at beta = pi/2 these two spellings differ, and it is
    # the exact equality of `cot_b` and `cos_b` there that makes the
    # `cot_b * (.../cos_b)` products cancel instead of overflowing.
    cot_b = cos_b / sin_b
    k = 1.0 / (4.0 * np.pi * (1.0 - nu))

    y3b = y3 + 2.0 * a
    z1b = y1 * cos_b + y3b * sin_b
    z3b = -y1 * sin_b + y3b * cos_b
    rb = np.sqrt(y1 * y1 + y2 * y2 + y3b * y3b)

    fib = 2.0 * np.arctan(-y2 / (-(rb + y3b) / np.tan(beta / 2.0) + y1))

    ry = rb + y3b
    rz = rb + z3b

    v1cb1 = b1 * k * (
        -2 * (1 - nu) * (1 - 2 * nu) * fib * cot_b ** 2
        + (1 - 2 * nu) * y2 / ry * ((1 - 2 * nu - a / rb) * cot_b - y1 / ry * (nu + a / rb))
        + (1 - 2 * nu) * y2 * cos_b * cot_b / rz * (cos_b + a / rb)
        + a * y2 * (y3b - a) * cot_b / rb ** 3
        + y2 * (y3b - a) / (rb * ry) * (-(1 - 2 * nu) * cot_b + y1 / ry * (2 * nu + a / rb) + a * y1 / rb ** 2)
        + y2 * (y3b - a) / (rb * rz) * (cos_b / rz * ((rb * cos_b + y3b) * ((1 - 2 * nu) * cos_b - a / rb) * cot_b
                                                      + 2 * (1 - nu) * (rb * sin_b - y1) * cos_b)
                                        - a * y3b * cos_b * cot_b / rb ** 2))

    v2cb1 = b1 * k * (
        (1 - 2 * nu) * ((2 * (1 - nu) * cot_b ** 2 - nu) * np.log(ry)
                        - (2 * (1 - nu) * cot_b ** 2 + 1 - 2 * nu) * cos_b * np.log(rz))
        - (1 - 2 * nu) / ry * (y1 * cot_b * (1 - 2 * nu - a / rb) + nu * y3b - a + y2 ** 2 / ry * (nu + a / rb))
        - (1 - 2 * nu) * z1b * cot_b / rz * (cos_b + a / rb)
        - a * y1 * (y3b - a) * cot_b / rb ** 3
        + (y3b - a) / ry * (-2 * nu + 1.0 / rb * ((1 - 2 * nu) * y1 * cot_b - a)
                            + y2 ** 2 / (rb * ry) * (2 * nu + a / rb) + a * y2 ** 2 / rb ** 3)
        + (y3b - a) / rz * (cos_b ** 2 - 1.0 / rb * ((1 - 2 * nu) * z1b * cot_b + a * cos_b)
                            + a * y3b * z1b * cot_b / rb ** 3
                            - 1.0 / (rb * rz) * (y2 ** 2 * cos_b ** 2 - a * z1b * cot_b / rb * (rb * cos_b + y3b))))

    v3cb1 = b1 * k * (
        2 * (1 - nu) * (((1 - 2 * nu) * fib * cot_b) + (y2 / ry * (2 * nu + a / rb))
                        - (y2 * cos_b / rz * (cos_b + a / rb)))
        + y2 * (y3b - a) / rb * (2 * nu / ry + a / rb ** 2)
        + y2 * (y3b - a) * cos_b / (rb * rz) * (1 - 2 * nu - (rb * cos_b + y3b) / rz * (cos_b + a / rb)
                                                - a * y3b / rb ** 2))

    v1cb2 = b2 * k * (
        (1 - 2 * nu) * ((2 * (1 - nu) * cot_b ** 2 + nu) * np.log(ry)
                        - (2 * (1 - nu) * cot_b ** 2 + 1) * cos_b * np.log(rz))
        + (1 - 2 * nu) / ry * (-(1 - 2 * nu) * y1 * cot_b + nu * y3b - a + a * y1 * cot_b / rb
                               + y1 ** 2 / ry * (nu + a / rb))
        - (1 - 2 * nu) * cot_b / rz * (z1b * cos_b - a * (rb * sin_b - y1) / (rb * cos_b))
        - a * y1 * (y3b - a) * cot_b / rb ** 3
        + (y3b - a) / ry * (2 * nu + 1.0 / rb * ((1 - 2 * nu) * y1 * cot_b + a)
                            - y1 ** 2 / (rb * ry) * (2 * nu + a / rb) - a * y1 ** 2 / rb ** 3)
        + (y3b - a) * cot_b / rz * (-cos_b * sin_b + a * y1 * y3b / (rb ** 3 * cos_b)
                                    + (rb * sin_b - y1) / rb * (2 * (1 - nu) * cos_b
                                                                - (rb * cos_b + y3b) / rz * (1 + a / (rb * cos_b)))))

    v2cb2 = b2 * k * (
        2 * (1 - nu) * (1 - 2 * nu) * fib * cot_b ** 2
        + (1 - 2 * nu) * y2 / ry * (-(1 - 2 * nu - a / rb) * cot_b + y1 / ry * (nu + a / rb))
        - (1 - 2 * nu) * y2 * cot_b / rz * (1 + a / (rb * cos_b))
        - a * y2 * (y3b - a) * cot_b / rb ** 3
        + y2 * (y3b - a) / (rb * ry) * ((1 - 2 * nu) * cot_b - 2 * nu * y1 / ry
                                        - a * y1 / rb * (1.0 / rb + 1.0 / ry))
        + y2 * (y3b - a) * cot_b / (rb * rz) * (-2 * (1 - nu) * cos_b
                                                + (rb * cos_b + y3b) / rz * (1 + a / (rb * cos_b))
                                                + a * y3b / (rb ** 2 * cos_b)))

    v3cb2 = b2 * k * (
        -2 * (1 - nu) * (1 - 2 * nu) * cot_b * (np.log(ry) - cos_b * np.log(rz))
        - 2 * (1 - nu) * y1 / ry * (2 * nu + a / rb)
        + 2 * (1 - nu) * z1b / rz * (cos_b + a / rb)
        + (y3b - a) / rb * ((1 - 2 * nu) * cot_b - 2 * nu * y1 / ry - a * y1 / rb ** 2)
        - (y3b - a) / rz * (cos_b * sin_b
                            + (rb * cos_b + y3b) * cot_b / rb * (2 * (1 - nu) * cos_b - (rb * cos_b + y3b) / rz)
                            + a / rb * (sin_b - y3b * z1b / rb ** 2 - z1b * (rb * cos_b + y3b) / (rb * rz))))

    v1cb3 = b3 * k * (
        (1 - 2 * nu) * (y2 / ry * (1 + a / rb) - y2 * cos_b / rz * (cos_b + a / rb))
        - y2 * (y3b - a) / rb * (a / rb ** 2 + 1.0 / ry)
        + y2 * (y3b - a) * cos_b / (rb * rz) * ((rb * cos_b + y3b) / rz * (cos_b + a / rb) + a * y3b / rb ** 2))

    v2cb3 = b3 * k * (
        (1 - 2 * nu) * (-sin_b * np.log(rz) - y1 / ry * (1 + a / rb) + z1b / rz * (cos_b + a / rb))
        + y1 * (y3b - a) / rb * (a / rb ** 2 + 1.0 / ry)
        - (y3b - a) / rz * (sin_b * (cos_b - a / rb) + z1b / rb * (1 + a * y3b / rb ** 2)
                            - 1.0 / (rb * rz) * (y2 ** 2 * cos_b * sin_b - a * z1b / rb * (rb * cos_b + y3b))))

    v3cb3 = b3 * k * (
        2 * (1 - nu) * fib + 2 * (1 - nu) * (y2 * sin_b / rz * (cos_b + a / rb))
        + y2 * (y3b - a) * sin_b / (rb * rz) * (1 + (rb * cos_b + y3b) / rz * (cos_b + a / rb)
                                                + a * y3b / rb ** 2))

    return (v1cb1 + v1cb2 + v1cb3,
            v2cb1 + v2cb2 + v2cb3,
            v3cb1 + v3cb2 + v3cb3)
