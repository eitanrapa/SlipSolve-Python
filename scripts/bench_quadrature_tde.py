"""Point-source quadrature against the analytic triangular dislocation.

After Jin, Fialko, Yang & Li (2023), *JGR Solid Earth* 128, e2023JB026643,
Supporting Information Figs S1-S3: sum point sources over one triangle, compare the
surface displacement with the analytic solution, and sweep the source spacing (S2)
and the depth (S3). Run here on this package's two pieces, **separately**:

1. the sub-triangle centroid rule (`layered._triangle_quadrature`) at a fixed order;
2. the adaptive order `LayeredPointSource` picks per triangle and observation.

Two equal-weight rules run beside them for context: SlipSolve-Curve's
`interpolate_triangle2` interior lattice, and the strike/dip grid clipped to the
triangle that the SI caption describes. ⚠️ **The clipped grid is measured and written
to the CSV but no longer drawn** (`PLOTTED_RULES`, 2026-09-15): it is the evidence for
the paragraph below about which rule the SI used, and its layout depends on where the
rows fall against the long edge, so its curve is not comparable rung by rung with the
two barycentric rules.

**Only quadrature is measured.** The engine is the real `LayeredPointSource` --
moment packing, azimuthal recombination, NED/ENU swaps -- with its EDGRN tables
swapped for the exact surface point source of a homogeneous half-space (Okada 1985,
eqs 25-27, recast into EDGRN's cylindrical basis), so there is no table
interpolation error. The truth is `tde_disp_hs` on the same triangle. The kernel
is checked against `EdgrnTables.homogeneous()` at its grid nodes before anything
runs (5.8e-6, which is that table's own 1 m source size; see `check_kernel`).

**Geometry, recovered from the SI.** Fitting `tde_disp_hs` to the six digitised
"Analytic" panels of S1(b,c) reproduces them to one 8-bit colour level (normalised
RMS 0.0024) at the vertices in :data:`S1_TRIANGLE`, scaled by 1.012 -- within the
pixel-to-km reading, so the round numbers are used; their maxima match S1's
colourbars (0.074/0.079/0.067 m strike-slip, 0.244 m dip-slip up). ⚠️ It is
**not** the triangle S1(a) draws (6.7-11.3 km deep), nor the paper's "centroid
depth of ~9 km": top vertex 1 km, base 8.49 km long and horizontal at 6 km striking
N45E, dip 67 degrees, centroid 4.33 km. ``L = sqrt(2A) = 6.79 km`` puts every
receiver at ``r/L`` 0.64-4.4 -- a near-field test; the Venezuela run's median is 38.

**Metric.** S2/S3 plot ``||dU|| / ||U||`` over the grid, per component. The SI does
not say so; inferred because a first-order rule's RMS/max error ratio on this grid
is 0.31 (east) and 0.70 (up), which applied to S1's diff colourbars (~6.4e-4)
predicts 0.27% and 0.66% against S2's plotted 0.28% and 0.63%.

⚠️ **The SI's rule was not `interpolate_triangle2`.** At equal nominal spacing the
interior lattice is 2.5x S2 (0.84% vs 0.28% east at 100 m) and its error map
correlates -0.16 to +0.46 with S1's published "Diff" panels. The caption's own
layout, the clipped strike/dip grid, correlates 0.88-0.97 and alone reproduces
S3's east/north plateau at depth (0.27/0.26% at 10 km against 0.22/0.23%; the
lattice falls to 0.11%). Its size depends on where the rows fall against the long
base edge -- 0.36x to 2.4x S2 at 100 m over anchor offsets -- and it is not even
monotone in spacing (0.43% at 175 m, 0.89% at 125 m), so the SI's straight lines
came from a layout detail that cannot be read off the figures.

**1. Sub-triangle centroids: second order, 71-94x the SI's accuracy at 100 m.**
Strike-slip, east / up, percent:

    spacing (m)        100            300            500
    centroids      0.0040/0.0074  0.037/0.069    0.100/0.187     5x -> 25x
    lattice        0.84/1.20      2.52/3.62      4.10/5.94       5x -> 4.9x
    Jin et al. S2  0.28/0.63      0.80/1.83      1.34/3.16       5x -> 4.7x

``N**-1.00`` in point count against the lattice's ``N**-0.49``. At matched
accuracy **121 centroids** (order 11) beat the SI's 100 m grid (~2300 points) in all
six components, and 64 beat the interior lattice at 100 m (3486). Against depth at
100 m the worst component falls from 0.019% with the top vertex at the surface to
0.0006% at 10 km.

**2. Adaptive order: error tracks the request, worst component 4x inside it.**
Top vertex at 1 km, worst-component grid RMS in percent, with the cap raised to
300**2 so only the error model is tested:

    tolerance      3e-2   1e-2   3e-3   1e-3   3e-4    1e-4
    points/pair    15.6   41.9   130    379    1236    3668
    RMS worst      0.67   0.23   0.073  0.024  0.0074  0.0025

Per receiver the 95th percentile sits at 0.40-0.56x the tolerance. At the default
cap (900 points, order 30) the result is within 3% of that down to 3e-3; below it
the RMS floors at 0.060% however tight the request, with 8% of pairs capped at 1e-3,
36% at 3e-4 and 96% at 1e-4. With the vertex at the surface the default-cap RMS is
0.16% at 3e-3, still 1.9x inside.

⚠️ **Two places a receiver misses its tolerance.**

* **Near a shallow vertex, by ~2x at every tolerance.** `_orders` measures ``r``
  from the centroid. The receiver over the 1 km-deep top vertex is 0.65 L from
  the centroid but 0.18 L from the vertex, and gets 1.9-2.1x its tolerance from
  3e-2 to 1e-4 alike -- a constant factor, so the model's shape, not a failure to
  converge. Misses reach out to 0.45 L (1.5x with the vertex 2 km deep).
  Measuring ``r`` to the nearest point of the triangle instead
  (`NearestDistanceOrders`) brings every receiver inside for 2.1x the points with
  the vertex at 1 km and 2.6-2.9x at the surface -- the ``distance`` experiment.
  The default cap hides the miss at 3e-3 by refusing those receivers' orders.
* **Over a strike-slip source's nodal region, up to 2.2x at 10 km depth** on 0.9%
  of receivers, where ``|u|`` is 4-12% of its maximum. A small denominator: the
  absolute error there is under 0.05% of max ``|u|``. None for dip-slip.

**3. Adaptive beats the fixed rule by ALLOCATION, not by converging faster.** Both
are the same sub-triangle centroid rule and both go as ``N**-1`` in points per
triangle -- fitted slopes **-1.004** fixed against **-1.023** adaptive over the
whole sweep -- so the adaptive curve in `convergence_points.png` is the same line
shifted, by **5.3-5.9x** in points at matched grid error from tolerance 3e-2 to
3e-4. `geometry.png`(e) is the reason: ``_orders`` scales with ``L/r``, so at
tolerance 3e-3 the per-receiver order runs k = 5-32 (25 to 1024 points, max/mean
**7.9x**), while a fixed rule must give the whole grid what its nearest receiver
needs. At order 20 (400 points everywhere) the median receiver's error is 0.009%
against a worst of 1.6%, and the 35% of receivers inside ``2 L`` carry
essentially all of the squared error -- the rest is paid for and wasted.

    python scripts/bench_quadrature_tde.py [--out DIR] [--quick] [--figures-only]
"""

import argparse
import csv
import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import slipsolve.layered as layered
from slipsolve._tde import _element_basis, tde_disp_hs
from slipsolve.edgrn import EdgrnTables
from slipsolve.layered import DEFAULT_TOLERANCE, LayeredPointSource
from slipsolve.mesh import FaultMesh

NU = 0.25

#: The triangle behind S1's maps, metres (east, north, up). Recovered by fitting
#: `tde_disp_hs` to the digitised "Analytic" panels of S1(b,c); see the module
#: docstring for how well it fits and why it is not the triangle S1(a) draws.
S1_TRIANGLE = np.array([[2.5, -0.5, -1.0], [3.5, 3.5, -6.0], [-2.5, -2.5, -6.0]]) * 1e3

#: S1's observation grid: 41 x 41 at 1 km over +/-20 km, on the free surface.
GRID = np.arange(-20.0, 21.0) * 1e3

SPACINGS = (100.0, 200.0, 300.0, 400.0, 500.0)
TOP_DEPTHS = (0.0, 1e3, 2e3, 3e3, 5e3, 7e3, 10e3)
TOLERANCES = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)

#: A cap high enough never to bind on this triangle, to separate the error model
#: from the cap in the tolerance sweep.
RAISED_MAX_POINTS = 300 ** 2

#: Source-receiver pairs per engine call; bounds the (nobs, nq) temporaries.
PAIRS_PER_CALL = 400_000

#: Read from the vector paths of the SI's EMF figures (not eyeballed): relative
#: error in percent, per slip type and component, at SPACINGS (S2) and at
#: TOP_DEPTHS with 100 m spacing (S3).
JIN_S2 = {
    "ss": {"E": (0.2836, 0.5531, 0.8019, 1.0412, 1.3447),
           "N": (0.2844, 0.5511, 0.7936, 1.0271, 1.3211),
           "U": (0.6274, 1.2535, 1.8319, 2.4261, 3.1571)},
    "ds": {"E": (0.3845, 0.7330, 1.0487, 1.3558, 1.7522),
           "N": (0.3899, 0.7385, 1.0567, 1.3740, 1.7823),
           "U": (0.4347, 0.8583, 1.2497, 1.6189, 2.0598)},
}
JIN_S3 = {
    "ss": {"E": (0.3668, 0.3050, 0.2821, 0.2638, 0.2413, 0.2291, 0.2246),
           "N": (0.3649, 0.3055, 0.2817, 0.2655, 0.2436, 0.2316, 0.2255),
           "U": (0.8567, 0.6790, 0.5615, 0.4612, 0.3107, 0.2252, 0.1565)},
    "ds": {"E": (0.4809, 0.4112, 0.3749, 0.3420, 0.2930, 0.2563, 0.2208),
           "N": (0.4855, 0.4178, 0.3783, 0.3475, 0.2955, 0.2597, 0.2247),
           "U": (0.6190, 0.4670, 0.3886, 0.3271, 0.2455, 0.2032, 0.1616)},
}

SLIPS = ("ss", "ds")
COMPONENTS = ("E", "N", "U")
FIXED_RULES = ("grid", "lattice", "centroid")

#: Which fixed rules the figures draw. ⚠️ The clipped strike/dip grid is still **run** and
#: written to the CSV -- it is the evidence that the SI's own rule was not
#: `interpolate_triangle2` (see the docstring) -- but it is not plotted: its layout depends on
#: where the rows fall against the long edge, so its curve is not even monotone in spacing and
#: reading it beside the two barycentric rules invites comparing what cannot be compared.
PLOTTED_RULES = ("lattice", "centroid")
LABELS = {
    "grid": "Strike/dip grid clipped to the triangle (SI caption)",
    "lattice": "Interior lattice, interpolate_triangle2 (SlipSolve-Curve)",
    "centroid": "Sub-triangle centroids (this package)",
    "adaptive": "Adaptive order (this package)",
}


# -- the exact kernel ------------------------------------------------------------
def point_source_basis(r, d, nu=NU):
    """EDGRN's surface basis functions for a homogeneous half-space, in closed form.

    ``r`` is the epicentral distance and ``d`` the source depth, both metres; the
    values are displacement per unit potency (area x slip), keyed like
    ``EdgrnTables.tables``: ``(kind, component)`` for ``ss``/``ds``/``cl`` and
    ``uz`` (down), ``ur``, ``ut``.

    From Okada's point source on a vertical plane (strike-slip for ``ss``, dip-slip
    for ``ds``) and three tensile sources for the CLVD, exactly as
    `EdgrnTables.homogeneous` combines them. The ``ss`` sign is Okada's reversed:
    his strike-slip is the opposite moment to the table's "normal north, slip east".
    """
    c = 1.0 - 2.0 * nu                      # mu / (lambda + mu)
    big_r2 = r * r + d * d
    big_r = np.sqrt(big_r2)
    r3 = big_r2 * big_r
    r5 = r3 * big_r2
    rd = big_r + d
    k = 1.0 / (2.0 * np.pi)
    zero = np.zeros_like(big_r)

    tz_up, tz_r = 3.0 * k * d ** 3 / r5, 3.0 * k * r * d * d / r5
    txy_up = k * (3.0 * d * r * r / r5
                  - c * (2.0 / (big_r * rd) - r * r * (2.0 * big_r + d) / (r3 * rd * rd)))
    txy_r = k * (3.0 * r ** 3 / r5 - c * r / r3)
    return {
        ("ss", "uz"): -0.5 * k * r * r * (3.0 * d / r5 - c * (2.0 * big_r + d) / (r3 * rd * rd)),
        ("ss", "ur"): k * (1.5 * r ** 3 / r5
                           + c * r * (1.0 / (big_r * rd * rd)
                                      - 0.5 * r * r * (3.0 * big_r + d) / (r3 * rd ** 3))),
        ("ss", "ut"): k * c * r / (big_r * rd * rd),
        ("ds", "uz"): 3.0 * k * d * d * r / r5,
        ("ds", "ur"): -3.0 * k * d * r * r / r5,
        ("ds", "ut"): zero,
        ("cl", "uz"): -0.5 * (tz_up - 0.5 * txy_up),
        ("cl", "ur"): 0.5 * (tz_r - 0.5 * txy_r),
        ("cl", "ut"): zero,
    }


class HalfSpacePointKernel:
    """Stands in for `EdgrnTables`: same ``interpolate``, exact instead of tabulated.

    `LayeredPointSource._sum_sources` asks for eight basis functions at the same
    ``(distance, depth)`` arrays, so all nine are computed on the first request and
    served from that until the arrays change.
    """

    def __init__(self, nu=NU):
        self.nu = float(nu)
        self.attrs = {"source": "okada1985_point_source", "nu": self.nu}
        self._cached = None

    def interpolate(self, kind, component, distance, depth):
        cached = self._cached
        if cached is None or cached[0] is not distance or cached[1] is not depth:
            cached = self._cached = (distance, depth, point_source_basis(
                np.asarray(distance, dtype=float), np.asarray(depth, dtype=float), self.nu))
        return cached[2][kind, component]


def check_kernel(nu=NU):
    """Largest disagreement with `EdgrnTables.homogeneous` at its nodes; raises if large.

    The tables' nodes carry no interpolation error, only their 1 m source's own
    finite size, which is ``(size / depth)**2``-small at these depths.
    """
    r = np.array([100.0, 1e3, 3e3, 10e3, 30e3])
    z = np.array([200.0, 900.0, 2e3, 9e3])
    tables = EdgrnTables.homogeneous(nu=nu, r=r, z=z, n_azimuth=16, source_size=1.0)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    exact = point_source_basis(rr, zz, nu)
    worst = 0.0
    for kind in ("ss", "ds", "cl"):
        scale = max(np.abs(tables.tables[kind][c]).max() for c in ("uz", "ur", "ut"))
        for component in ("uz", "ur", "ut"):
            difference = np.abs(exact[kind, component] - tables.tables[kind][component]).max()
            worst = max(worst, difference / scale)
    if worst > 1e-4:
        raise RuntimeError(f"point-source kernel disagrees with EdgrnTables.homogeneous "
                           f"by {worst:.1e}; nothing below would mean anything")
    return worst


# -- the rules ---------------------------------------------------------------------
def interior_lattice(p1, p2, p3, n):
    """SlipSolve-Curve's `interpolate_triangle2`, equally weighted as its caller does.

    Lattice nodes ``(i, j, k) / n`` with ``i, j, k >= 1``: ``(n-1)(n-2)/2`` points,
    so ``n = 15`` is the reference's 91.
    """
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    n = max(3, int(n))
    points = np.asarray([(i * p1 + j * p2 + (n - i - j) * p3) / n
                         for i in range(1, n - 1) for j in range(1, n - i)])
    return points, np.full(len(points), 1.0 / len(points))


def clipped_grid(spacing, offset=(0.5, 0.5)):
    """The SI caption's layout: a strike/dip grid kept inside the triangle, equal weights.

    "Both horizontal and vertical spacings between the neighboring point sources are
    set to 100 m." Grid lines run along the triangle's own strike and dip vectors,
    anchored at its first vertex and shifted by ``offset`` cells; the order argument
    the engine passes is ignored, since the spacing fixes the layout.
    """
    def rule(p1, p2, p3, order=None):
        vertices = np.array([p1, p2, p3], dtype=float)
        _, strike, dip = _element_basis(*vertices)
        origin = vertices[0]
        uv = (vertices - origin) @ np.stack([strike, dip]).T
        low = np.floor(uv.min(axis=0) / spacing) - 1
        high = np.ceil(uv.max(axis=0) / spacing) + 1
        u = (np.arange(low[0], high[0] + 1) + offset[0]) * spacing
        v = (np.arange(low[1], high[1] + 1) + offset[1]) * spacing
        uu, vv = np.meshgrid(u, v)
        q = np.column_stack([uu.ravel(), vv.ravel()])
        edges = np.column_stack([uv[1] - uv[0], uv[2] - uv[0]])
        lam = np.linalg.solve(edges, (q - uv[0]).T).T
        q = q[(lam[:, 0] >= 0) & (lam[:, 1] >= 0) & (lam.sum(axis=1) <= 1)]
        if q.size == 0:
            return vertices.mean(axis=0)[None, :], np.array([1.0])
        points = origin + q[:, :1] * strike + q[:, 1:] * dip
        return points, np.full(len(points), 1.0 / len(points))
    return rule


def nearest_distance(vertices, x, y):
    """Distance from surface points to the nearest point of a triangle.

    Ericson's closest-point regions (Real-Time Collision Detection, 5.1.5), taken in
    the same order as his if/else chain so ``np.select`` picks the same region.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in vertices)
    p = np.column_stack([x, y, np.zeros(np.size(x))])
    ab, ac = b - a, c - a
    d1, d2 = (p - a) @ ab, (p - a) @ ac
    d3, d4 = (p - b) @ ab, (p - b) @ ac
    d5, d6 = (p - c) @ ab, (p - c) @ ac
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        interior = (a + ab * (vb / (va + vb + vc))[:, None]
                    + ac * (vc / (va + vb + vc))[:, None])
        choices = [np.broadcast_to(a, p.shape), np.broadcast_to(b, p.shape),
                   np.broadcast_to(c, p.shape),
                   a + ab * (d1 / (d1 - d3))[:, None],
                   a + ac * (d2 / (d2 - d6))[:, None],
                   b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]]
    conditions = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6),
                  (vc <= 0) & (d1 >= 0) & (d3 <= 0), (vb <= 0) & (d2 >= 0) & (d6 <= 0),
                  (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
    closest = np.column_stack([np.select(conditions, [choice[:, k] for choice in choices],
                                         default=interior[:, k]) for k in range(3)])
    return np.linalg.norm(p - closest, axis=1)


class NearestDistanceOrders(LayeredPointSource):
    """The package's error model with ``r`` taken to the triangle, not its centroid.

    A what-if for this one-triangle benchmark, not a proposal: it needs the
    vertices, which `_orders` is not given, and assumes surface receivers.
    """

    def __init__(self, *args, vertices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertices = np.asarray(vertices, dtype=float)

    def _orders(self, centroid, size, x, y, z, clip=True):
        if self.tolerance is None:
            return super()._orders(centroid, size, x, y, z, clip)
        r = np.maximum(nearest_distance(self.vertices, x, y), size / 100.0)
        gain = np.sqrt(layered._QUADRATURE_SAFETY * layered._QUADRATURE_CONSTANT / self.tolerance)
        wanted = np.ceil(gain * size / r)
        if not clip:
            return wanted
        return np.clip(wanted, 1, np.sqrt(self.max_points)).astype(int)


@contextmanager
def quadrature_rule(rule):
    """Integrate with ``rule`` instead of the package's rule; ``None`` keeps it."""
    saved = layered._triangle_quadrature
    if rule is not None:
        layered._triangle_quadrature = rule
    try:
        yield
    finally:
        layered._triangle_quadrature = saved


# -- fields ------------------------------------------------------------------------
def observation_grid(grid=GRID):
    gx, gy = np.meshgrid(grid, grid)
    return gx.ravel(), gy.ravel()


def triangle_mesh(vertices):
    return FaultMesh(vertices, [[0, 1, 2]], np.zeros((3, 2)))


def analytic_field(vertices, x, y):
    """ENU displacement per unit slip, ``(nobs, 3, 2)``: strike-slip, then dip-slip."""
    out = np.empty((x.size, 3, 2))
    z = np.zeros(x.size)
    for column, (ss, ds) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        out[:, :, column] = np.column_stack(tde_disp_hs(x, y, z, *vertices, ss, ds, 0.0, NU))
    return out


def point_source_field(vertices, x, y, rule=None, order=None, tolerance=None,
                       max_points=None, points_hint=None, distance="centroid"):
    """The same field from `LayeredPointSource`, and what its quadrature spent.

    ``tolerance=None`` is a fixed rule at ``order`` everywhere; otherwise the
    adaptive order at that tolerance, capped at ``max_points`` (the package default
    when None), with ``distance="nearest"`` swapping in `NearestDistanceOrders`.
    """
    mesh = triangle_mesh(vertices)
    kernel = HalfSpacePointKernel(NU)
    if tolerance is None:
        engine = LayeredPointSource(kernel, nu=NU, tolerance=None, max_points=int(order) ** 2)
    else:
        options = {} if max_points is None else {"max_points": max_points}
        if distance == "nearest":
            engine = NearestDistanceOrders(kernel, nu=NU, tolerance=tolerance,
                                           vertices=vertices, **options)
        else:
            engine = LayeredPointSource(kernel, nu=NU, tolerance=tolerance, **options)
    per_receiver = points_hint or engine.max_points
    chunk = int(max(1, min(x.size, PAIRS_PER_CALL // max(1, per_receiver))))

    out = np.empty((x.size, 3, 2))
    pairs = capped = 0
    points = 0.0
    with quadrature_rule(rule), warnings.catch_warnings():
        # Capped pairs are counted and reported below rather than warned about.
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, x.size, chunk):
            stop = start + chunk
            out[start:stop] = engine.displacement(mesh, x[start:stop], y[start:stop])
            report = engine.quadrature_report
            pairs += report["pairs"]
            capped += report["capped"]
            points += report["points_per_pair"] * report["pairs"]
    return out, {"pairs": pairs, "capped_fraction": capped / pairs,
                 "points_per_pair": points / pairs, "engine": engine, "mesh": mesh}


def capped_receivers(result, x, y):
    """Which observations asked for more than the cap -- the tolerance is unmet there."""
    engine, mesh = result["engine"], result["mesh"]
    if engine.tolerance is None:
        return np.zeros(x.size, dtype=bool)
    wanted = engine._orders(mesh.centroids[0], np.sqrt(2.0 * mesh.areas[0]),
                            x, y, np.zeros(x.size), clip=False)
    return wanted > int(np.sqrt(engine.max_points))


def rms_percent(numerical, analytic):
    """The SI's error: ``100 * ||dU|| / ||U||`` over the grid, ``(slip, component)``."""
    return 100.0 * (np.linalg.norm(numerical - analytic, axis=0)
                    / np.linalg.norm(analytic, axis=0)).T


def vector_error(numerical, analytic):
    """Relative error of the ENU vector per observation, ``(nobs, slip)``.

    What the adaptive tolerance is calibrated on, so the per-receiver check.
    """
    return np.linalg.norm(numerical - analytic, axis=1) / np.linalg.norm(analytic, axis=1)


def longest_edge(vertices):
    return max(np.linalg.norm(vertices[i] - vertices[j]) for i, j in ((0, 1), (1, 2), (2, 0)))


def with_top_depth(vertices, depth):
    shifted = np.array(vertices, dtype=float)
    shifted[:, 2] += -depth - shifted[:, 2].max()
    return shifted


def fixed_rule(name, vertices, spacing=None, order=None):
    """``(rule, order, points)``. The barycentric rules cut the longest edge ~``spacing``."""
    if name == "grid":
        rule, order = clipped_grid(spacing), 1
    else:
        order = order if order is not None else max(1, round(longest_edge(vertices) / spacing))
        rule = interior_lattice if name == "lattice" else None
    points = len((rule or layered._triangle_quadrature)(*vertices, order)[0])
    return rule, order, points


def error_columns(numerical, analytic, mask=None):
    rms = rms_percent(numerical, analytic)
    vec = vector_error(numerical, analytic)
    keep = np.ones(len(vec), dtype=bool) if mask is None else mask
    row = {f"rms_{s}_{c}": float(rms[i, j])
           for i, s in enumerate(SLIPS) for j, c in enumerate(COMPONENTS)}
    worst = vec.max(axis=1)
    row["rms_worst"] = float(rms.max())
    row["vec_p50"] = float(np.median(worst[keep])) if keep.any() else np.nan
    row["vec_p95"] = float(np.quantile(worst[keep], 0.95)) if keep.any() else np.nan
    row["vec_max"] = float(worst[keep].max()) if keep.any() else np.nan
    return row, worst


# -- experiments -------------------------------------------------------------------
def run_fixed(name, vertices, x, y, analytic, spacing=None, order=None, **tags):
    rule, order, points = fixed_rule(name, vertices, spacing, order)
    start = time.perf_counter()
    numerical, _ = point_source_field(vertices, x, y, rule=rule, order=order,
                                      points_hint=points)
    row, _ = error_columns(numerical, analytic)
    row.update(rule=name, spacing_m=spacing, order=order, points=points,
               seconds=time.perf_counter() - start, **tags)
    return row


def run_adaptive(vertices, x, y, analytic, tolerance, max_points=None, distance="centroid",
                 **tags):
    start = time.perf_counter()
    numerical, result = point_source_field(vertices, x, y, tolerance=tolerance,
                                           max_points=max_points, distance=distance)
    capped = capped_receivers(result, x, y)
    row, worst = error_columns(numerical, analytic, mask=~capped)
    uncapped = ~capped
    row.update(rule="adaptive", tolerance=tolerance, distance=distance,
               max_points=result["engine"].max_points,
               points=result["points_per_pair"], capped_fraction=result["capped_fraction"],
               over_tolerance=(float(np.mean(worst[uncapped] > tolerance))
                               if uncapped.any() else np.nan),
               seconds=time.perf_counter() - start, **tags)
    if uncapped.any():
        row.update(_worst_receiver(result, x, y, np.where(uncapped, worst, -np.inf)))
    return row


def _worst_receiver(result, x, y, error):
    """Where the worst uncapped receiver sits, against the size the order model uses.

    `_orders` measures ``r`` from the centroid; the nearest part of a large shallow
    triangle can be much closer than that, so both distances are recorded.
    """
    engine, mesh = result["engine"], result["mesh"]
    i = int(np.argmax(error))
    size = float(np.sqrt(2.0 * mesh.areas[0]))
    receiver = np.array([x[i], y[i], 0.0])
    dense, _ = layered._triangle_quadrature(*mesh.vertices[0], 64)
    dense = np.vstack([dense, mesh.vertices[0]])
    order = engine._orders(mesh.centroids[0], size, x[i:i + 1], y[i:i + 1], np.zeros(1),
                           clip=False)[0]
    return {"worst_error": float(error[i]), "worst_x_m": float(x[i]), "worst_y_m": float(y[i]),
            "worst_r_centroid_over_L": float(np.linalg.norm(receiver - mesh.centroids[0]) / size),
            "worst_r_nearest_over_L": float(np.linalg.norm(dense - receiver, axis=1).min() / size),
            "worst_order": int(order)}


def experiments(quick=False):
    spacings = (100.0, 300.0, 500.0) if quick else SPACINGS
    depths = (0.0, 1e3, 5e3, 10e3) if quick else TOP_DEPTHS
    tolerances = (1e-2, 3e-3, 1e-3) if quick else TOLERANCES
    x, y = observation_grid(GRID[::2] if quick else GRID)
    base = S1_TRIANGLE
    truth = analytic_field(base, x, y)
    rows = []

    def show(row, *keys):
        print("  " + "  ".join(f"{k}={row[k]:.4g}" if isinstance(row[k], float) else f"{k}={row[k]}"
                               for k in keys)
              + "  | rms% ss " + " ".join(f"{row[f'rms_ss_{c}']:.4f}" for c in COMPONENTS)
              + "  ds " + " ".join(f"{row[f'rms_ds_{c}']:.4f}" for c in COMPONENTS))

    print("\nS2 analogue: error against spacing, top vertex at 1 km")
    for name in FIXED_RULES:
        for spacing in spacings:
            rows.append(run_fixed(name, base, x, y, truth, spacing=spacing,
                                  experiment="spacing", top_depth_m=1e3))
            show(rows[-1], "rule", "spacing_m", "points")

    print("\nS3 analogue: error against depth of the top vertex, 100 m spacing")
    for depth in depths:
        vertices = with_top_depth(base, depth)
        depth_truth = analytic_field(vertices, x, y)
        for name in FIXED_RULES:
            rows.append(run_fixed(name, vertices, x, y, depth_truth, spacing=100.0,
                                  experiment="depth", top_depth_m=depth))
            show(rows[-1], "rule", "top_depth_m", "points")
        for tolerance in (DEFAULT_TOLERANCE, 1e-3):
            rows.append(run_adaptive(vertices, x, y, depth_truth, tolerance,
                                     experiment="depth", top_depth_m=depth))
            show(rows[-1], "rule", "top_depth_m", "tolerance", "points", "capped_fraction",
                 "over_tolerance", "vec_max")

    print("\nConvergence against point count, top vertex at 1 km")
    orders = {"centroid": (2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128),
              "lattice": (4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181)}
    grids = (1500.0, 1000.0, 700.0, 500.0, 350.0, 250.0, 175.0, 125.0, 90.0, 64.0)
    if quick:
        orders = {k: v[::3] for k, v in orders.items()}
        grids = grids[::3]
    for name, values in orders.items():
        for order in values:
            rows.append(run_fixed(name, base, x, y, truth, order=order,
                                  experiment="points", top_depth_m=1e3))
            show(rows[-1], "rule", "order", "points")
    for spacing in grids:
        rows.append(run_fixed("grid", base, x, y, truth, spacing=spacing,
                              experiment="points", top_depth_m=1e3))
        show(rows[-1], "rule", "spacing_m", "points")

    print("\nAdaptive order against tolerance, top vertex at 1 km")
    for max_points in (None, RAISED_MAX_POINTS):
        for tolerance in tolerances:
            rows.append(run_adaptive(base, x, y, truth, tolerance, max_points=max_points,
                                     experiment="tolerance", top_depth_m=1e3))
            show(rows[-1], "tolerance", "max_points", "points", "capped_fraction",
                 "over_tolerance", "vec_p95", "vec_max", "worst_r_centroid_over_L",
                 "worst_r_nearest_over_L", "worst_order")

    print("\nOrder model's distance: from the centroid, or to the nearest point (no cap)")
    for depth in (0.0, 1e3):
        vertices = with_top_depth(base, depth)
        depth_truth = analytic_field(vertices, x, y)
        for tolerance in (1e-2, 3e-3, 1e-3):
            for distance in ("centroid", "nearest"):
                rows.append(run_adaptive(vertices, x, y, depth_truth, tolerance,
                                         max_points=RAISED_MAX_POINTS, distance=distance,
                                         experiment="distance", top_depth_m=depth))
                show(rows[-1], "top_depth_m", "tolerance", "distance", "points",
                     "over_tolerance", "vec_max")
    return rows


# -- figures -----------------------------------------------------------------------
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
COMPONENT_STYLE = {"E": (SERIES[0], "o"), "N": (SERIES[1], "*"), "U": (SERIES[2], "^")}
RULE_COLOR = dict(zip(("grid", "lattice", "centroid", "adaptive"), SERIES))
SLIP_TITLE = {"ss": "Unit strike-slip", "ds": "Unit dip-slip"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _log_axis(ax, axis="y", subs=(1.0, 2.0, 5.0)):
    """Plain-number ticks on a log axis, so a narrow range still gets labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10.0, subs=subs))
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    target.set_minor_formatter(NullFormatter())


def _figure(nrows, ncols, height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, height), squeeze=False,
                             facecolor=SURFACE, constrained_layout=True)
    for ax in axes.ravel():
        _style(ax)
    return plt, fig, axes


def _select(rows, **match):
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


def figure_sweep(rows, experiment, xkey, xlabel, jin, path):
    """S2/S3 layout: one row per plotted rule, strike-slip left, dip-slip right."""
    plt, fig, axes = _figure(len(PLOTTED_RULES), 2, 3.0 * len(PLOTTED_RULES))
    for i, name in enumerate(PLOTTED_RULES):
        selected = sorted(_select(rows, experiment=experiment, rule=name), key=lambda r: r[xkey])
        xs = np.array([r[xkey] for r in selected]) / (1e3 if xkey == "top_depth_m" else 1.0)
        for j, slip in enumerate(SLIPS):
            ax = axes[i, j]
            if name != "centroid":
                jx = (np.array(SPACINGS) if experiment == "spacing"
                      else np.array(TOP_DEPTHS) / 1e3)
                for c in COMPONENTS:
                    _, marker = COMPONENT_STYLE[c]
                    ax.plot(jx, jin[slip][c], color=MUTED, lw=0.9, marker=marker, ms=5,
                            mfc="none", label=f"{c} Jin et al. (2023)")
            for c in COMPONENTS:
                color, marker = COMPONENT_STYLE[c]
                ax.plot(xs, [r[f"rms_{slip}_{c}"] for r in selected], color=color, lw=1.4,
                        marker=marker, ms=6, label=c)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{SLIP_TITLE[slip]} -- {LABELS[name]}", fontsize=9, loc="left")
            ax.set_ylabel("Relative error (%)", fontsize=8)
            if i == len(PLOTTED_RULES) - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            ax.legend(fontsize=7, frameon=False, ncol=2 if name != "centroid" else 1)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


#: Short names for the geometry figure's panels; `LABELS` is written for legends.
RULE_SHORT = {"centroid": "Sub-triangle centroids", "lattice": "Interior lattice",
              "grid": "Clipped strike/dip grid"}


def figure_geometry(path, spacing=500.0, tolerance=DEFAULT_TOLERANCE, top_depth=1e3):
    """S1(a)'s picture, for the rules compared here -- and why the adaptive order wins.

    (a) the triangle, its sub-triangle centroids at ~``spacing``, and the receiver grid;
    (b, c) the plotted fixed layouts in the triangle's own plane at the same nominal spacing;
    (d) the points the adaptive rule asks for at each receiver. It integrates with rule (b),
    so the only difference is that count: fixed rules spend the near field's order on the
    whole grid, and the far field is already exact at one point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = with_top_depth(S1_TRIANGLE, top_depth)
    x, y = observation_grid()
    mesh = triangle_mesh(vertices)
    size = np.sqrt(2.0 * mesh.areas[0])

    fig = plt.figure(figsize=(10, 8.8), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=(1.35, 1.0))

    ax = fig.add_subplot(gs[0, 0:4], projection="3d")
    ax.set_facecolor(SURFACE)
    order = max(1, round(longest_edge(vertices) / spacing))
    points = layered._triangle_quadrature(*vertices, order)[0] / 1e3
    ax.add_collection3d(Poly3DCollection([vertices / 1e3], facecolors=SERIES[2], alpha=0.2,
                                         edgecolors=INK, linewidths=1.0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, color=SERIES[2], depthshade=False)
    near = GRID[np.abs(GRID) <= 12e3][::4]
    gx, gy = np.meshgrid(near, near)
    ax.scatter(gx.ravel() / 1e3, gy.ravel() / 1e3, np.zeros(gx.size), s=3, color=MUTED, alpha=0.7)
    top = int(np.argmax(vertices[:, 2]))
    for index, name in ((top, "top"), (int(np.argmin(vertices[:, 1])), "base")):
        v = vertices[index] / 1e3
        ax.text(v[0], v[1], v[2] + 0.4, f"  {name} {-v[2]:.1f} km", fontsize=7, color=INK_2)
    ax.set(xlabel="East (km)", ylabel="North (km)", zlabel="Up (km)",
           xlim=(-13, 13), ylim=(-13, 13), zlim=(-8, 0))
    ax.set_box_aspect((1.0, 1.0, 0.42), zoom=0.86)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(colors=INK_2, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(INK_2)
        axis.label.set_fontsize(8)
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title(f"(a) The triangle of S1, top vertex {top_depth / 1e3:.0f} km deep, with "
                 f"{len(points)} sub-triangle centroids\n(order {order}, ~{spacing:.0f} m "
                 "spacing) and the receiver grid (every 4 km shown; it runs to +/-20 km)",
                 fontsize=9, loc="left", color=INK)

    ax = fig.add_subplot(gs[0, 4:6])
    _style(ax)
    engine = LayeredPointSource(HalfSpacePointKernel(NU), nu=NU, tolerance=tolerance,
                                max_points=RAISED_MAX_POINTS)
    k = engine._orders(mesh.centroids[0], size, x, y, np.zeros(x.size), clip=False)
    count = np.clip(k, 1, None).astype(float) ** 2
    image = ax.pcolormesh(GRID / 1e3, GRID / 1e3, count.reshape(GRID.size, GRID.size),
                          norm=LogNorm(), cmap="magma_r", shading="nearest")
    loop = np.append(np.arange(3), 0)
    ax.plot(vertices[loop, 0] / 1e3, vertices[loop, 1] / 1e3, color=INK, lw=1.0)
    ax.set(xlabel="East (km)", ylabel="North (km)", aspect="equal")
    ax.set_title(f"(d) Adaptive order, tolerance {tolerance:g}\nmean {count.mean():.0f} / median "
                 f"{np.median(count):.0f} / max {count.max():.0f}\npoints per receiver",
                 fontsize=9, loc="left")
    bar = fig.colorbar(image, ax=ax, pad=0.03, shrink=0.62)
    bar.set_label("Point sources", fontsize=8, color=INK_2)
    bar.ax.tick_params(colors=INK_2, labelsize=7)

    _, strike, dip = _element_basis(*vertices)
    frame = np.stack([strike, dip]).T
    outline = (vertices - vertices[0]) @ frame / 1e3
    width = 6 // len(PLOTTED_RULES)
    for column, name in enumerate(PLOTTED_RULES[::-1]):          # centroids first, as in (a)
        ax = fig.add_subplot(gs[1, column * width:(column + 1) * width])
        _style(ax)
        rule, rule_order, total = fixed_rule(name, vertices, spacing=spacing)
        located = (rule or layered._triangle_quadrature)(*vertices, rule_order)[0]
        uv = (located - vertices[0]) @ frame / 1e3
        ax.fill(outline[:, 0], outline[:, 1], facecolor="none", edgecolor=INK, lw=1.0)
        ax.plot(uv[:, 0], uv[:, 1], ".", ms=3, color=RULE_COLOR[name])
        ax.set(xlabel="Along strike (km)", aspect="equal")
        if column == 0:
            ax.set_ylabel("Down dip (km)", fontsize=8)
        ax.set_title(f"({'bcd'[column]}) {RULE_SHORT[name]}: {total} points", fontsize=9, loc="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_convergence(rows, path):
    plt, fig, axes = _figure(1, 2, 4.2)
    points_rows = _select(rows, experiment="points")
    adaptive = sorted(_select(rows, experiment="tolerance", max_points=RAISED_MAX_POINTS),
                      key=lambda r: r["points"])
    for j, slip in enumerate(SLIPS):
        ax = axes[0, j]
        for name in PLOTTED_RULES:
            selected = sorted(_select(points_rows, rule=name), key=lambda r: r["points"])
            n = np.array([r["points"] for r in selected], dtype=float)
            err = np.array([max(r[f"rms_{slip}_{c}"] for c in COMPONENTS) for r in selected])
            ax.loglog(n, err, color=RULE_COLOR[name], lw=1.4, marker="o", ms=4,
                      label=LABELS[name])
        n = np.array([r["points"] for r in adaptive])
        err = np.array([max(r[f"rms_{slip}_{c}"] for c in COMPONENTS) for r in adaptive])
        ax.loglog(n, err, color=RULE_COLOR["adaptive"], lw=1.4, marker="D", ms=5,
                  label=LABELS["adaptive"] + " (mean points per pair)")
        for r, xv, yv in zip(adaptive, n, err):
            ax.annotate(f"{r['tolerance']:g}", (xv, yv), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color=INK_2)
        guide = np.array([10.0, 2e4])
        ax.loglog(guide, 30.0 * (guide / 10.0) ** -0.5, color=MUTED, lw=0.8, ls=":")
        ax.loglog(guide, 3.0 * (guide / 10.0) ** -1.0, color=MUTED, lw=0.8, ls=":")
        ax.text(guide[1], 30.0 * (guide[1] / 10.0) ** -0.5, "  N^-1/2\n  (first order)",
                fontsize=6, color=MUTED, va="center")
        ax.text(guide[1], 3.0 * (guide[1] / 10.0) ** -1.0, "  N^-1\n  (second order)",
                fontsize=6, color=MUTED, va="center")
        _log_axis(ax, "y", subs=(1.0,))
        _log_axis(ax, "x", subs=(1.0,))
        ax.set_title(f"{SLIP_TITLE[slip]}: worst of E/N/U", fontsize=9, loc="left")
        ax.set_xlabel("Point sources per triangle", fontsize=8)
        ax.set_ylabel("Relative error (%)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_adaptive(rows, experiment, xkey, xlabel, path, logx):
    """Adaptive order: error by component, per-receiver error, and what it cost."""
    plt, fig, axes = _figure(3, 2, 10.0)
    if experiment == "tolerance":
        default = sorted(_select(rows, experiment=experiment, max_points=layered.MAX_QUADRATURE_POINTS),
                         key=lambda r: r[xkey])
        raised = sorted(_select(rows, experiment=experiment, max_points=RAISED_MAX_POINTS),
                        key=lambda r: r[xkey])
    else:
        default = sorted(_select(rows, experiment=experiment, rule="adaptive",
                                 tolerance=DEFAULT_TOLERANCE), key=lambda r: r[xkey])
        raised = sorted(_select(rows, experiment=experiment, rule="adaptive", tolerance=1e-3),
                        key=lambda r: r[xkey])
    scale = 1e3 if xkey == "top_depth_m" else 1.0
    xs = np.array([r[xkey] for r in default]) / scale
    xr = np.array([r[xkey] for r in raised]) / scale
    requested = 100.0 * (xs if experiment == "tolerance" else np.full(xs.size, DEFAULT_TOLERANCE))
    for j, slip in enumerate(SLIPS):
        ax = axes[0, j]
        for c in COMPONENTS:
            color, marker = COMPONENT_STYLE[c]
            ax.plot(xs, [r[f"rms_{slip}_{c}"] for r in default], color=color, lw=1.4,
                    marker=marker, ms=6, label=c)
        ax.plot(xs, requested, color=MUTED, lw=0.9, ls="--", label="tolerance requested")
        ax.set_yscale("log")
        _log_axis(ax)
        ax.set_title(f"{SLIP_TITLE[slip]}: grid RMS error (SI metric)", fontsize=9, loc="left")
        ax.set_ylabel("Relative error (%)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 0]
    label_default = ("cap 900 points" if experiment == "tolerance"
                     else f"tolerance {DEFAULT_TOLERANCE:g}")
    label_raised = ("cap 90 000 points" if experiment == "tolerance" else "tolerance 0.001")
    for rows_, xv, color, lab in ((default, xs, SERIES[0], label_default),
                                  (raised, xr, SERIES[1], label_raised)):
        ax.plot(xv, [100 * r["vec_p95"] for r in rows_], color=color, lw=1.4, marker="o", ms=5,
                label=f"95th percentile, {lab}")
        ax.plot(xv, [100 * r["vec_max"] for r in rows_], color=color, lw=0.9, marker="^", ms=5,
                mfc="none", label=f"maximum, {lab}")
    ax.plot(xs, requested, color=MUTED, lw=0.9, ls="--", label="tolerance requested")
    if experiment != "tolerance":
        ax.plot(xr, np.full(xr.size, 0.1), color=MUTED, lw=0.9, ls=":",
                label="tolerance 0.001 requested")
    ax.set_yscale("log")
    _log_axis(ax)
    ax.set_title("Per-receiver vector error, uncapped receivers", fontsize=9, loc="left")
    ax.set_ylabel("Relative error (%)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1, 1]
    for rows_, xv, color, lab in ((default, xs, SERIES[0], label_default),
                                  (raised, xr, SERIES[1], label_raised)):
        ax.plot(xv, [100 * r["over_tolerance"] for r in rows_], color=color, lw=1.4,
                marker="o", ms=5, label=lab)
    ax.set_title("Uncapped receivers over their tolerance", fontsize=9, loc="left")
    ax.set_ylabel("Receivers (%)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2, 0]
    for rows_, xv, color, lab in ((default, xs, SERIES[0], label_default),
                                  (raised, xr, SERIES[1], label_raised)):
        ax.plot(xv, [r["points"] for r in rows_], color=color, lw=1.4, marker="o", ms=5, label=lab)
    ax.set_yscale("log")
    _log_axis(ax)
    ax.set_title("Cost: mean point sources per source-receiver pair", fontsize=9, loc="left")
    ax.set_ylabel("Points per pair", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2, 1]
    for rows_, xv, color, lab in ((default, xs, SERIES[0], label_default),
                                  (raised, xr, SERIES[1], label_raised)):
        ax.plot(xv, [100 * r["capped_fraction"] for r in rows_], color=color, lw=1.4,
                marker="o", ms=5, label=lab)
    ax.set_title("Pairs refused their order by the cap", fontsize=9, loc="left")
    ax.set_ylabel("Pairs (%)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        if logx:
            ax.set_xscale("log")
            _log_axis(ax, "x", subs=(1.0, 3.0))
            ax.invert_xaxis()
        ax.set_xlabel(xlabel, fontsize=8)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_rows(rows, path):
    keys = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    """The CSV back as the rows the figures take -- numbers as numbers, blanks as None."""
    def value(text):
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return text

    with open(path, newline="") as handle:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("workdir/quadrature_tde"),
                        help="directory for the CSV, JSON and figures (default: %(default)s)")
    parser.add_argument("--quick", action="store_true",
                        help="coarse grid and fewer sweep points, for a smoke run")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from an existing quadrature_tde.csv")
    args = parser.parse_args()

    start = time.perf_counter()
    if args.figures_only:
        rows = read_rows(args.out / "quadrature_tde.csv")
        print(f"redrawing figures from {len(rows)} rows in {args.out}/quadrature_tde.csv")
    else:
        worst = check_kernel()
        print(f"kernel vs EdgrnTables.homogeneous at its nodes: worst {worst:.1e}")
        rows = experiments(quick=args.quick)
        args.out.mkdir(parents=True, exist_ok=True)
        write_rows(rows, args.out / "quadrature_tde.csv")
        meta = {"triangle_m": S1_TRIANGLE.tolist(), "grid_m": [float(GRID[0]), float(GRID[-1]),
                float(GRID[1] - GRID[0])], "nu": NU, "kernel_check": worst, "quick": args.quick,
                "default_tolerance": DEFAULT_TOLERANCE,
                "max_quadrature_points": layered.MAX_QUADRATURE_POINTS,
                "seconds": time.perf_counter() - start}
        (args.out / "quadrature_tde.json").write_text(json.dumps(meta, indent=2))
    if not args.no_figures:
        figure_geometry(args.out / "geometry.png")
        figure_sweep(rows, "spacing", "spacing_m", "Nominal spacing between point sources (m)",
                     JIN_S2, args.out / "fixed_rules_spacing.png")
        figure_sweep(rows, "depth", "top_depth_m", "Depth of the triangle's top vertex (km)",
                     JIN_S3, args.out / "fixed_rules_depth.png")
        figure_convergence(rows, args.out / "convergence_points.png")
        figure_adaptive(rows, "tolerance", "tolerance", "Requested tolerance",
                        args.out / "adaptive_tolerance.png", logx=True)
        figure_adaptive(rows, "depth", "top_depth_m", "Depth of the triangle's top vertex (km)",
                        args.out / "adaptive_depth.png", logx=False)
    print(f"\nwrote {args.out}/ in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
