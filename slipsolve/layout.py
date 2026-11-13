"""Where the mesh nodes go in `(s, z)`, and how they become triangles.

A `slipsolve.mesh.FaultMesh` is two separable things: a set of points in the
`(arc length, z)` parameter plane together with a triangulation, and an
embedding that carries those points into 3-D through a fitted `cross(s, z)`
surface. This module owns the first half. Splitting it out is what lets the
node set be chosen freely (by hand, by a target element-size field, or by an
optimizer reading model resolution) while the embedding stays exactly what
`slipsolve.surface` fits.

**Winding.** `_lattice_triangles`, the quad split a uniform layout takes,
produces triangles whose signed area in `(s, z)` is uniformly negative, and
that is exactly the condition under which the 3-D normal lands on the trace's
left-hand side (`FaultMesh._check_winding` verifies it). Since
`(s, z) -> (x, y, z)` is orientation preserving wherever the fitted surface
does not fold (which `FaultMesh.curved` refuses), fixing the sign in parameter
space fixes it in 3-D. That is why Delaunay is safe here, where
SlipSolve-Curve's `orientTriFromLeft.m` (which tests against a hard-coded
`[1 0 0]`) is not.

**Boundaries stay exact.** `boundary_node_mask` compares against
`s.min()`/`s.max()`/`z.min()`/`z.max()` with `==`. Every builder here places
boundary nodes on literally shared floats so that keeps working;
`snap_boundary` is how an arbitrary point cloud earns the same guarantee, and
`require_boundaries` refuses a layout that has not, because a boundary that
silently collapses to one node removes the zero-slip constraint without
raising anything.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

#: Sign of a triangle's signed area in `(s, z)` for the normal to come out on
#: the trace's left-hand side. Negative; see this module's docstring.
PARAM_ORIENTATION = -1.0

#: Triangles smaller than this fraction of the median are dropped as degenerate.
#: Delaunay over a rectangle can emit a zero-area sliver from three nearly
#: collinear boundary points, and `basis._barycentric_weight` divides by a
#: quantity that vanishes on one.
MIN_AREA_FRACTION = 1e-9

#: Largest node-count ratio allowed between adjacent depth levels. A coarse row
#: against a fine one is what makes Delaunay reach for long thin triangles, so
#: the grading is limited rather than the triangulation policed.
MAX_COUNT_RATIO = 2.0

EDGE_SIDES = ("top", "bottom", "left", "right")


# -- triangulation ---------------------------------------------------------
def _lattice_triangles(ns, nz):
    """Split an `ns x nz` node lattice into triangles with one fixed winding.

    Node `(i, j)` (`i` along strike, `j` down dip) is index `j * ns + i`.
    Each quad becomes `(a, b, c)` and `(a, c, d)` with `a = (i, j)`,
    `b = (i + 1, j)`, `c = (i + 1, j + 1)`, `d = (i, j + 1)`. For a fault
    extruded downward along an increasing-`s` trace, both halves' normals come
    out on the trace's left-hand side.

    Kept as the path a uniform layout takes even though `triangulate`
    could do it, because on a regular lattice the four corners of a quad are
    cocircular: Delaunay's choice of diagonal is then arbitrary, and qhull picks
    a different one, which would change `FaultMesh.digest()` and invalidate
    every persisted store for no geometric gain.
    """
    i = np.arange(ns - 1)
    j = np.arange(nz - 1)
    a = (j[:, None] * ns + i[None, :]).ravel()
    b = a + 1
    c = a + ns + 1
    d = a + ns
    tris = np.empty((2 * a.size, 3), dtype=int)
    tris[0::2] = np.column_stack([a, b, c])
    tris[1::2] = np.column_stack([a, c, d])
    return tris


def signed_areas(params, triangles):
    """Twice the signed area of each triangle in the `(s, z)` plane."""
    params = np.asarray(params, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    a, b, c = params[triangles[:, 0]], params[triangles[:, 1]], params[triangles[:, 2]]
    return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def orient_triangles(params, triangles):
    """Reverse any triangle whose `(s, z)` winding is the wrong way round.

    One cross product per triangle. This is what makes
    `FaultMesh._check_winding` an assertion rather than a guard: a flipped
    triangle negates its dip-slip Green's-function column while leaving
    strike-slip alone, so an unpoliced Delaunay would give a plausible model
    with a randomly-signed dip-slip field.
    """
    triangles = np.array(triangles, dtype=int, copy=True)
    wrong = np.sign(signed_areas(params, triangles)) != PARAM_ORIENTATION
    triangles[wrong] = triangles[wrong][:, [0, 2, 1]]
    return triangles


def triangulate(params, structured=None):
    """Triangles over an `(s, z)` point set, wound consistently.

    `structured=(ns, nz)` takes the exact quad split of a uniform lattice;
    anything else goes through Delaunay. The parameter domain is a convex
    rectangle, so an unconstrained triangulation covers it exactly and no
    constrained mesher is needed.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must be (n, 2) of (s, z)")

    if structured is not None:
        ns, nz = (int(v) for v in structured)
        if ns * nz != params.shape[0]:
            raise ValueError(
                f"structured={(ns, nz)} describes {ns * nz} nodes but params has "
                f"{params.shape[0]}"
            )
        return _lattice_triangles(ns, nz)

    from scipy.spatial import Delaunay

    if params.shape[0] < 3:
        raise ValueError("a triangulation needs at least 3 nodes")
    triangles = orient_triangles(params, Delaunay(params).simplices)

    area = np.abs(signed_areas(params, triangles)) / 2.0
    keep = area > MIN_AREA_FRACTION * np.median(area)
    if not keep.all():
        triangles = triangles[keep]
    orphan = np.setdiff1d(np.arange(params.shape[0]), triangles.ravel())
    if orphan.size:
        raise ValueError(
            f"{orphan.size} node(s) ended up in no triangle "
            f"(indices {orphan[:10].tolist()}); they carry a slip parameter that "
            "nothing constrains and no Green's function reads. Duplicate or "
            "collinear parameter points are the usual cause."
        )
    return triangles


# -- the layout ------------------------------------------------------------
@dataclass
class MeshLayout:
    """An `(s, z)` node set with its triangulation, before any embedding.

    `levels` and `counts` are present only for a row-structured layout (one
    depth per level, that many nodes spread along strike); a freeform cloud
    leaves them `None`. `uniform_counts` is what tells
    `slipsolve.mesh.FaultMesh` it may take the bit-identical lattice path.
    """

    params: np.ndarray
    triangles: np.ndarray
    levels: np.ndarray = None
    counts: np.ndarray = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.params = np.asarray(self.params, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=int)
        if self.levels is not None:
            self.levels = np.asarray(self.levels, dtype=float)
        if self.counts is not None:
            self.counts = np.asarray(self.counts, dtype=int)
        self.attrs = dict(self.attrs or {})

    @property
    def n_nodes(self):
        """Number of nodes."""
        return int(self.params.shape[0])

    @property
    def n_elements(self):
        """Number of triangles."""
        return int(self.triangles.shape[0])

    @property
    def length(self):
        """Along-strike extent, i.e. the trace length the layout spans."""
        return float(self.params[:, 0].max())

    @property
    def uniform_counts(self):
        """`(ns, nz)` when every level holds the same node count, else `None`."""
        if self.counts is None or self.counts.size == 0:
            return None
        if not np.all(self.counts == self.counts[0]):
            return None
        return int(self.counts[0]), int(self.counts.size)

    def level_slices(self):
        """One `slice` per depth level, in the layout's own node order."""
        if self.counts is None:
            raise ValueError("level_slices() needs a row-structured layout")
        stop = np.cumsum(self.counts)
        return [slice(int(a), int(b)) for a, b in zip(stop - self.counts, stop)]

    def quality(self):
        """`layout_quality` of this layout."""
        return layout_quality(self.params, self.triangles)

    def __repr__(self):
        shape = "uniform" if self.uniform_counts else (
            "graded" if self.counts is not None else "freeform")
        return (f"<MeshLayout {shape} nodes={self.n_nodes} "
                f"elements={self.n_elements}>")


def layout_quality(params, triangles):
    """Shape statistics of a layout, in parameter space.

    Parameter space is what the layout controls; the 3-D elements differ from
    these by the embedding's own stretch, which is small while the surface does
    not fold. Reported so a graded layout can be compared against the uniform
    one it replaces: a uniform lattice of 5 km along strike and 1 km levels
    already has an 11.3 degree minimum angle, so a number near that is normal.
    """
    params = np.asarray(params, dtype=float)
    v = params[np.asarray(triangles, dtype=int)]
    edges = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]], axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    angles = []
    for k in range(3):
        opposite = lengths[:, k]
        b, c = lengths[:, (k + 1) % 3], lengths[:, (k + 2) % 3]
        cos = (b ** 2 + c ** 2 - opposite ** 2) / np.maximum(2.0 * b * c, 1e-300)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    angles = np.stack(angles, axis=1)
    area = np.abs(signed_areas(params, triangles)) / 2.0
    return {
        "n_nodes": int(params.shape[0]),
        "n_elements": int(len(triangles)),
        "min_angle_deg": float(angles.min()),
        "median_min_angle_deg": float(np.median(angles.min(axis=1))),
        "aspect_max": float((lengths.max(axis=1) / lengths.min(axis=1)).max()),
        "area_min": float(area.min()),
        "area_median": float(np.median(area)),
        "area_max": float(area.max()),
    }


# -- boundaries ------------------------------------------------------------
def node_components(triangles, n_nodes):
    """An integer label per node: which connected piece of the mesh it is in.

    Two faults carried in one mesh share a frame and a parameterization but no
    triangles, so "which fault is this node on" is a property of the connectivity
    and needs nothing stored. A single-fault mesh labels every node `0`, which
    is what keeps `boundary_node_mask` bit-identical for one.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    triangles = np.asarray(triangles, dtype=int)
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(int(n_nodes), int(n_nodes)))
    return connected_components(graph, directed=False)[1]


def boundary_node_mask(params, side, labels=None):
    """Nodes on one edge of the parameter rectangle, by exact equality.

    Exact, deliberately: every builder in this module puts a level's nodes on a
    literally shared `z` and a level's ends on a literally shared `s`, so
    there is nothing to round. A tolerance would instead have to be guessed from
    the local spacing, and guessing it too small is silent.

    `labels` is what makes this work for more than one fault. Without it the
    test is against a global extremum, so on a mesh holding two faults only
    the deeper one has a bottom edge and only the outer ends of the pair are
    `left` and `right`: the second fault's rim silently loses its zero-slip
    rows and slip runs off it with nothing raised. Pass one integer per node
    (`node_components`) and each piece is measured against its own extent.
    A single label reproduces the global test exactly.
    """
    params = np.asarray(params, dtype=float)
    s, z = params[:, 0], params[:, 1]
    if side not in EDGE_SIDES:
        raise ValueError(f"Unknown boundary {side!r}; expected one of {EDGE_SIDES}")
    value, extreme = ((z, np.min) if side == "bottom" else
                      (z, np.max) if side == "top" else
                      (s, np.min) if side == "left" else (s, np.max))
    if labels is None:
        return value == extreme(value)

    labels = np.asarray(labels)
    if labels.shape != value.shape:
        raise ValueError(
            f"labels has {labels.shape} entries against {value.shape} nodes -- "
            "one component label per node is needed")
    mask = np.zeros(value.shape, dtype=bool)
    for label in np.unique(labels):
        piece = labels == label
        mask[piece] = value[piece] == extreme(value[piece])
    return mask


def require_boundaries(params, sides=EDGE_SIDES, minimum=2, labels=None):
    """Refuse a layout whose edges have collapsed to a node or two.

    The failure this exists to catch is silent: `zero_slip_boundary` selects
    whatever `boundary_node_mask` flags, so an edge that resolves to one node
    stops constraining the mesh's rim and slip runs off it with nothing raised.

    With `labels` the count is per component and the thinnest piece is what
    is reported, since one unconstrained fault is enough to invalidate the run.
    """
    if labels is None:
        thin = {side: int(boundary_node_mask(params, side).sum()) for side in sides}
    else:
        labels = np.asarray(labels)
        thin = {}
        for side in sides:
            mask = boundary_node_mask(params, side, labels)
            thin[side] = min(int(mask[labels == label].sum())
                             for label in np.unique(labels))
    bad = {k: v for k, v in thin.items() if v < minimum}
    if bad:
        pieces = ("" if labels is None
                  else " (counted on the thinnest of "
                       f"{len(np.unique(labels))} connected pieces)")
        raise ValueError(
            f"Boundary side(s) {sorted(bad)} hold only {[bad[k] for k in sorted(bad)]} "
            f"node(s){pieces} (at least {minimum} needed). Boundary membership is "
            "decided by exact equality in (s, z), so nodes meant to sit on an edge "
            "must share the identical float -- see layout.snap_boundary()."
        )
    return thin


def snap_boundary(params, length=None, z_top=None, z_bottom=None, tol=1.0):
    """Pull near-edge nodes onto exactly the four boundary values.

    How an arbitrary `(s, z)` cloud earns the exact-equality guarantee
    `boundary_node_mask` relies on. `tol` is in metres and should be far
    below the node spacing: it is meant to absorb the round-off of however the
    points were generated, not to move a node that was placed inside on purpose.
    """
    params = np.array(params, dtype=float, copy=True)
    s, z = params[:, 0], params[:, 1]
    length = float(s.max()) if length is None else float(length)
    z_top = float(z.max()) if z_top is None else float(z_top)
    z_bottom = float(z.min()) if z_bottom is None else float(z_bottom)

    s[np.abs(s - 0.0) <= tol] = 0.0
    s[np.abs(s - length) <= tol] = length
    z[np.abs(z - z_top) <= tol] = z_top
    z[np.abs(z - z_bottom) <= tol] = z_bottom
    return params


# -- builders --------------------------------------------------------------
def _resolve_counts(depths, length, counts, along_spacing):
    """One node count per depth level, from whichever knob was given."""
    depths = np.asarray(depths, dtype=float)
    if (counts is None) == (along_spacing is None):
        raise ValueError("give exactly one of counts= or along_spacing=")

    if along_spacing is not None:
        spacing = (np.asarray([float(along_spacing(d)) for d in depths])
                   if callable(along_spacing)
                   else np.broadcast_to(np.asarray(along_spacing, dtype=float),
                                        depths.shape))
        if np.any(spacing <= 0):
            raise ValueError("along_spacing must be positive")
        resolved = np.maximum(1, np.round(length / spacing).astype(int)) + 1
    else:
        resolved = (np.asarray([int(counts(d)) for d in depths], dtype=int)
                    if callable(counts)
                    else np.broadcast_to(np.asarray(counts, dtype=int),
                                         depths.shape).astype(int))

    resolved = np.array(resolved, dtype=int, copy=True)
    if resolved.shape != depths.shape:
        raise ValueError(
            f"counts/along_spacing gave {resolved.shape} values for "
            f"{depths.shape} depth levels"
        )
    if np.any(resolved < 2):
        raise ValueError("every depth level needs at least 2 nodes along strike")
    return resolved


def _balance_counts(counts, balance=MAX_COUNT_RATIO, warn=True):
    """Limit how fast the node count may change between adjacent levels.

    A fine row against a coarse one is what makes Delaunay reach across for a
    long thin triangle. Counts are only ever raised, so a request is honoured
    where it can be and refined where it cannot.

    `warn` is off when the counts came from a continuous size field rather than
    from the caller: refining one is then part of the algorithm, not a request
    being overridden, and the budget search rebuilds a dozen times.
    """
    balance = float(balance)
    if balance < 1.0:
        raise ValueError("balance must be at least 1")
    out = np.array(counts, dtype=int, copy=True)
    for _ in range(out.size):
        before = out.copy()
        out[1:] = np.maximum(out[1:], np.ceil(out[:-1] / balance).astype(int))
        out[:-1] = np.maximum(out[:-1], np.ceil(out[1:] / balance).astype(int))
        if np.array_equal(out, before):
            break
    changed = int(np.sum(out != np.asarray(counts, dtype=int)))
    if changed and warn:
        warnings.warn(
            f"{changed} of {out.size} depth levels were refined to keep adjacent "
            f"node counts within {balance}:1; asking for a sharper jump than that "
            "buys long thin triangles rather than resolution.",
            RuntimeWarning, stacklevel=3,
        )
    return out


def level_stations(n, length, stations=None):
    """`n` along-strike positions spanning `[0, length]`.

    With `stations` given (the arc lengths of the resampled trace's own
    vertices) the level is that sequence resampled uniformly in index, which
    keeps the trace's parameterisation rather than imposing a fresh uniform one.
    At `n == stations.size`, `np.interp` returns its input array bit for bit,
    which is what makes a uniform layout reproduce the classic lattice's node
    coordinates exactly and leave `FaultMesh.digest()` unchanged.
    """
    n = int(n)
    if stations is None:
        return np.linspace(0.0, float(length), n)
    stations = np.asarray(stations, dtype=float)
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, stations.size), stations)


def graded_layout(length, depths, counts=None, along_spacing=None,
                  stations=None, balance=MAX_COUNT_RATIO):
    """A layout of freely-placed depth levels, each with its own node count.

    `depths` is a sequence of depths in metres, positive down, shallowest
    first, with arbitrary spacing (not just the geometric `bias_w` progression).
    `counts` is a scalar, one value per level, or a callable of depth;
    `along_spacing` says the same thing in metres instead. `stations` is the
    optional reference along-strike sampling (see `level_stations`); every
    level starts at exactly `0.0` and ends at exactly `length` either way,
    which is what keeps `boundary_node_mask` working without a tolerance.

    A uniform `counts` reproduces the classic lattice exactly, triangles
    included.
    """
    depths = np.asarray(depths, dtype=float).ravel()
    if depths.size < 2:
        raise ValueError("a mesh needs at least 2 depth levels")
    if np.any(np.diff(depths) <= 0):
        raise ValueError("depths must increase strictly, shallowest first")
    if np.any(depths < 0):
        raise ValueError("depths are positive downward, in metres")
    length = float(length)
    if stations is not None:
        stations = np.asarray(stations, dtype=float)
        if stations.ndim != 1 or stations.size < 2:
            raise ValueError("stations must be a 1-D array of at least 2 arc lengths")
        length = float(stations[-1])

    resolved = _balance_counts(
        _resolve_counts(depths, length, counts, along_spacing), balance)

    rows = [np.column_stack([level_stations(n, length, stations),
                             np.full(int(n), -float(d))])
            for d, n in zip(depths, resolved)]
    params = np.vstack(rows)

    shape = (int(resolved[0]), int(depths.size)) if np.all(resolved == resolved[0]) else None
    layout = MeshLayout(params, triangulate(params, structured=shape),
                        levels=depths, counts=resolved,
                        attrs={"layout": "uniform" if shape else "graded"})
    require_boundaries(layout.params)
    return layout


def from_points(params, length=None, snap=True, tol=1.0):
    """A freeform layout from an arbitrary `(s, z)` cloud.

    Boundary nodes are snapped onto the exact edge values by default; pass
    `snap=False` only if they already are.
    """
    params = np.asarray(params, dtype=float)
    if snap:
        params = snap_boundary(params, length=length, tol=tol)
    layout = MeshLayout(params, triangulate(params), attrs={"layout": "freeform"})
    require_boundaries(layout.params)
    return layout


def _size_field(size):
    """Normalise a target element size into a callable of `(s, depth)`."""
    if callable(size):
        return lambda s, d: np.asarray(size(s, d), dtype=float)
    value = float(size)
    if value <= 0:
        raise ValueError("the target element size must be positive")
    return lambda s, d: np.full(np.broadcast(s, d).shape, value)


def _edge_stations(extent, spacing_at):
    """Positions along one edge honouring a varying target spacing.

    Marching by the local spacing would leave the far end a ragged fraction of a
    step short; integrating `1/h` and then splitting into a whole number of
    equal parameter steps lands on both ends exactly, which is what
    `boundary_node_mask` needs.
    """
    dense = np.linspace(0.0, extent, 512)
    density = 1.0 / np.maximum(spacing_at(dense), 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(dense))])
    n = max(1, int(round(cumulative[-1])))
    stations = np.interp(np.linspace(0.0, cumulative[-1], n + 1), cumulative, dense)
    stations[0], stations[-1] = 0.0, extent
    return stations


def lloyd_layout(length, max_depth, size, top_depth=0.0, iterations=25, seed=0):
    """An unstructured layout relaxed toward a target element size.

    `size` is a scalar or a callable `h(s, depth) -> metres`. Boundary nodes
    are placed first, at the local target spacing along each of the four edges,
    and are then held fixed; interior nodes are rejection-sampled against
    `1/h**2` and relaxed toward the density-weighted centroid of their
    incident triangles, the cheap approximation to a centroidal Voronoi
    tessellation with density `1/h**2`, stable because the domain is a
    rectangle. The weighting is what makes the result honour `size`.

    There are no depth levels here. Element size follows `h` in two
    dimensions, so a fine patch can sit at one along-strike position and a coarse
    one beside it at the same depth, which a row-structured layout cannot
    express, since a level is one depth for the whole fault. That is what
    `slipsolve.resolution.adapt_layout`'s `structure="unstructured"` drives
    from model resolution.

    The outline stays a rectangle in `(s, z)`: the four edges are straight,
    because `boundary_node_mask` identifies them by exact equality and the
    zero-slip constraint depends on it. Freeform means the discretisation, not
    the fault's extent.

    Slower and harder to reason about than `graded_layout`, and it gives
    up the row structure that `adapt_layout`'s `"rows"` mode uses; reach for
    it when the target size really does vary along strike as much as it does
    with depth.
    """
    h = _size_field(size)
    length, top_depth, max_depth = float(length), float(top_depth), float(max_depth)
    span = max_depth - top_depth
    if span <= 0:
        raise ValueError("max_depth must be deeper than top_depth")

    top = _edge_stations(length, lambda s: h(s, np.full_like(s, top_depth)))
    bottom = _edge_stations(length, lambda s: h(s, np.full_like(s, max_depth)))
    left = _edge_stations(span, lambda d: h(np.zeros_like(d), top_depth + d))
    right = _edge_stations(span, lambda d: h(np.full_like(d, length), top_depth + d))

    edge = np.vstack([
        np.column_stack([top, np.full(top.size, -top_depth)]),
        np.column_stack([bottom, np.full(bottom.size, -max_depth)]),
        np.column_stack([np.zeros(left.size - 2), -(top_depth + left[1:-1])]),
        np.column_stack([np.full(right.size - 2, length), -(top_depth + right[1:-1])]),
    ])
    edge = np.unique(edge, axis=0)

    rng = np.random.default_rng(seed)
    probe_s, probe_d = np.meshgrid(np.linspace(0.0, length, 64),
                                   np.linspace(top_depth, max_depth, 64))
    finest = float(np.min(h(probe_s, probe_d)))
    target = int(np.sum(1.0 / np.maximum(h(probe_s, probe_d), 1e-9) ** 2)
                 * (length * span) / probe_s.size)
    interior = np.empty((0, 2))
    while interior.shape[0] < target:
        draw = rng.random((4 * max(target, 16), 2))
        s = draw[:, 0] * length
        d = top_depth + draw[:, 1] * span
        accept = rng.random(s.size) < (finest / h(s, d)) ** 2
        interior = np.vstack([interior, np.column_stack([s[accept], -d[accept]])])
    interior = interior[:target]

    params = np.vstack([edge, interior])
    fixed = np.zeros(params.shape[0], dtype=bool)
    fixed[:edge.shape[0]] = True

    for _ in range(int(iterations)):
        triangles = triangulate(params)
        centroid = params[triangles].mean(axis=1)
        # Area and density: a plain area-weighted centroid is the centroidal
        # Voronoi iteration for a uniform density, so it relaxes toward equal
        # element sizes and undoes the grading the rejection sampling just put
        # in. Weighting by 1/h**2, the node density the size field asks for,
        # gives the density-weighted CVT, whose equilibrium spacing is
        # proportional to h. It also improves element quality.
        weight = (np.abs(signed_areas(params, triangles)) / 2.0
                  / np.maximum(h(centroid[:, 0], -centroid[:, 1]), 1e-9) ** 2)
        total = np.zeros(params.shape[0])
        moved = np.zeros_like(params)
        np.add.at(total, triangles.ravel(), np.repeat(weight, 3))
        np.add.at(moved, triangles.ravel(), np.repeat(centroid * weight[:, None], 3, axis=0))
        live = (~fixed) & (total > 0)
        params[live] = moved[live] / total[live, None]
        params[:, 0] = np.clip(params[:, 0], 0.0, length)
        params[:, 1] = np.clip(params[:, 1], -max_depth, -top_depth)

    params = snap_boundary(params, length=length, z_top=-top_depth, z_bottom=-max_depth,
                           tol=1e-6 * max(length, span))
    layout = MeshLayout(params, triangulate(params), attrs={"layout": "lloyd"})
    require_boundaries(layout.params)
    return layout


#: `clip_layout`: a node within this fraction of its shortest edge of the curve snaps onto it.
CLIP_SNAP = 0.3


def _douglas_peucker(x, y, tol):
    """Indices of the vertices of the polyline `(x, y)` kept by Douglas-Peucker at tolerance `tol`."""
    keep = np.zeros(x.size, bool)
    keep[[0, -1]] = True
    stack = [(0, x.size - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        length = np.hypot(dx, dy)
        seg = slice(a + 1, b)
        d = (np.abs(dy * (x[seg] - x[a]) - dx * (y[seg] - y[a])) / length if length > 0
             else np.hypot(x[seg] - x[a], y[seg] - y[a]))
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.flatnonzero(keep)


def _curve_distance(q, cs, cz):
    """Distance in `(s, z)` from each point of `q` `(n, 2)` to the polyline `(cs, cz)`."""
    ax, ay, vx, vy = cs[:-1], cz[:-1], np.diff(cs), np.diff(cz)
    l2 = np.maximum(vx * vx + vy * vy, 1e-12)
    out = np.empty(q.shape[0])
    for i0 in range(0, q.shape[0], 4000):
        qq = q[i0:i0 + 4000]
        t = np.clip(((qq[:, :1] - ax) * vx + (qq[:, 1:] - ay) * vy) / l2, 0.0, 1.0)
        out[i0:i0 + 4000] = np.hypot(qq[:, :1] - ax - t * vx, qq[:, 1:] - ay - t * vy).min(axis=1)
    return out


def clip_layout(layout, s_curve, depth_curve, snap=None, tol=25.0, dp_tol=100.0, tol_mid=200.0):
    """The part of `layout` above a depth that varies along strike, with that curve as its lower edge.

    `depth_curve` (m, positive down) is given at the arc lengths `s_curve`; it is where
    a sheet meets another one (`slipsolve.production.Sheet.meeting_depth`). Nodes within
    `snap` (default `CLIP_SNAP`) times their shortest edge of the curve move onto it; an
    end node moves only up or down, and a surface node never. The nodes left above the
    curve, the points where the layout's edges cross it and the curve's own vertices
    (simplified by Douglas-Peucker at `dp_tol`, m, so every bend of the curve that reaches
    into the sheet is a node) are triangulated again. A triangle is kept only if its
    centroid lies above the curve to within `tol` and its edge midpoints to within
    `tol_mid` (m). A layout the curve does not reach comes back unchanged.

    Cutting each triangle that straddles the curve along its own chord does not work: a
    triangle whose three nodes lie above the curve can still span a bend of it, and then
    reaches through the other sheet.
    """
    from scipy.spatial import Delaunay

    snap = CLIP_SNAP if snap is None else snap
    p = np.asarray(layout.params, float).copy()
    tri = np.asarray(layout.triangles, int)
    cs, cd = np.asarray(s_curve, float), np.asarray(depth_curve, float)
    ze = lambda s: np.interp(s, cs, cd)                                      # noqa: E731
    depth = -p[:, 1]
    if np.all(depth <= ze(p[:, 0]) + 1e-6):
        return layout
    s_lo, s_hi = p[:, 0].min(), p[:, 0].max()
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    el = np.hypot(*(p[e[:, 0]] - p[e[:, 1]]).T)
    hmin = np.full(p.shape[0], np.inf)
    np.minimum.at(hmin, e[:, 0], el)
    np.minimum.at(hmin, e[:, 1], el)
    # 1. snap onto the nearest point of the curve (an end node only up or down; never a surface node)
    cz = -cd
    ax, ay, vx, vy = cs[:-1], cz[:-1], np.diff(cs), np.diff(cz)
    l2 = np.maximum(vx * vx + vy * vy, 1e-12)
    t = np.clip(((p[:, :1] - ax) * vx + (p[:, 1:] - ay) * vy) / l2, 0.0, 1.0)
    nx_, ny_ = ax + t * vx, ay + t * vy
    dist = np.hypot(p[:, :1] - nx_, p[:, 1:] - ny_)
    k = np.argmin(dist, axis=1)
    rows = np.arange(p.shape[0])
    near_s, near_z, near_d = nx_[rows, k], ny_[rows, k], dist[rows, k]
    end = np.isclose(p[:, 0], s_lo, atol=1e-6) | np.isclose(p[:, 0], s_hi, atol=1e-6)
    phi = depth - ze(p[:, 0])
    cand = (depth > 0.0) & np.where(end, (np.abs(phi) < snap * hmin) & (ze(p[:, 0]) > 0.0),
                                    (near_d < snap * hmin) & (-near_z < ze(near_s) + 1.0) & (near_z < 0.0))
    p[cand & end, 1] = -ze(p[cand & end, 0])
    inner = cand & ~end
    p[inner, 0], p[inner, 1] = near_s[inner], -ze(near_s[inner])
    phi = -p[:, 1] - ze(p[:, 0])
    inside = phi <= tol
    # 2. where the layout's edges cross the curve
    ins, outs = phi < -tol, phi > tol
    cross_e = e[(ins[e[:, 0]] & outs[e[:, 1]]) | (outs[e[:, 0]] & ins[e[:, 1]])]
    pts = []
    for a, b in cross_e:
        lo, hi = (a, b) if ins[a] else (b, a)
        t0, t1 = 0.0, 1.0
        for _ in range(40):
            tm = 0.5 * (t0 + t1)
            q = p[lo] + tm * (p[hi] - p[lo])
            if -q[1] - ze(q[0]) > 0:
                t1 = tm
            else:
                t0 = tm
        q = p[lo] + t0 * (p[hi] - p[lo])
        pts.append([q[0], -ze(q[0])])
    # 3. the curve's own vertices where it cuts the sheet (simplified), inside the sheet's s range, and points along
    #    its arc at the typical element size, so a steep stretch of the curve has nodes down its length
    shape_pts = []
    cut = (cd < cd.max() - tol) & (cs >= s_lo) & (cs <= s_hi)
    h_typ = float(np.median(el))
    if cut.any():
        i0, i1 = max(np.flatnonzero(cut)[0] - 1, 0), min(np.flatnonzero(cut)[-1] + 1, cs.size - 1)
        sub = np.arange(i0, i1 + 1)
        for j in sub[_douglas_peucker(cs[sub], cd[sub], dp_tol)]:
            if s_lo <= cs[j] <= s_hi:
                shape_pts.append([cs[j], -cd[j]])
        arc_c = np.r_[0.0, np.cumsum(np.hypot(np.diff(cs[sub]), np.diff(cd[sub])))]
        for a in np.arange(0.0, arc_c[-1], 0.75 * h_typ)[1:]:
            ps, pd_ = np.interp(a, arc_c, cs[sub]), np.interp(a, arc_c, cd[sub])
            if s_lo <= ps <= s_hi and pd_ < cd.max() - tol:
                pts.append([ps, -pd_])
    # The curve's points are the new edge: all kept but near-duplicates (< 0.3 x the typical edge), and the layout's
    # own nodes within half their shortest edge of the curve dropped instead, since a node that close leaves slivers.
    # The Douglas-Peucker vertices carry the curve's shape and are never thinned.
    on_curve = inside & (np.abs(phi) <= tol)
    shape = np.array(shape_pts) if shape_pts else np.zeros((0, 2))
    rest = np.vstack([np.array(pts).reshape(-1, 2), p[on_curve]])
    rest = rest[np.lexsort((rest[:, 1], rest[:, 0]))] if rest.size else rest
    kept = [q for q in shape]
    for q in rest:
        if not kept or np.hypot(*(np.array(kept) - q).T).min() > 0.3 * h_typ:
            kept.append(q)
    extra = np.array(kept).reshape(-1, 2)
    interior = inside & ~on_curve
    if extra.size:
        d_int = _curve_distance(p[interior], cs, -cd)
        far = d_int > 0.5 * hmin[interior]
        idx = np.flatnonzero(interior)
        top_or_end = (p[idx, 1] >= -1e-6) | np.isclose(p[idx, 0], s_lo, atol=1e-6) | np.isclose(p[idx, 0], s_hi, atol=1e-6)
        interior[idx[~far & ~top_or_end]] = False
    base = p[interior]
    allp = np.vstack([base, extra]) if extra.size else base
    tris = Delaunay(allp).simplices
    # 4. keep a triangle only if it lies inside: centroid and edge midpoints
    corners = allp[tris]
    probes = np.stack([corners.mean(axis=1), 0.5 * (corners[:, 0] + corners[:, 1]),
                       0.5 * (corners[:, 1] + corners[:, 2]), 0.5 * (corners[:, 2] + corners[:, 0])], 1)
    below = (-probes[..., 1] - ze(probes[..., 0])) > 0
    dist = _curve_distance(probes.reshape(-1, 2), cs, -cd).reshape(probes.shape[:2])
    ph = np.where(below, dist, -dist)
    ok = (ph[:, 0] <= tol) & np.all(ph[:, 1:] <= tol_mid, axis=1)
    area = np.abs(signed_areas(allp, tris)) / 2.0
    ok &= area > 1e-4 * np.median(area[ok])
    tris = tris[ok]
    used = np.unique(tris)
    remap = -np.ones(allp.shape[0], int)
    remap[used] = np.arange(used.size)
    params = allp[used]
    tris = orient_triangles(params, remap[tris])
    return MeshLayout(params, tris, attrs={**dict(layout.attrs), "clipped": True})
