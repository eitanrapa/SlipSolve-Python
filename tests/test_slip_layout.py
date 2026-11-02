"""Pins the freeform mesh layout: node placement, Delaunay, and the guards.

The load-bearing invariant is **winding**, as in ``test_slip_mesh.py``, but for a
harder case: an irregular node set has no quads, so the triangles come from
Delaunay, whose vertex order is arbitrary. The claim these tests exist to hold is
that orienting every simplex by its signed area *in the ``(s, z)`` parameter
plane* is equivalent to the lattice's own winding -- and therefore that a mesh
with a different number of nodes at each depth level carries the same dip-slip
sign as one without.

The second theme is **backward compatibility**. A uniform layout must reproduce
the classic lattice bit for bit, triangles and node coordinates alike, or every
persisted ``.slip.zip`` and workspace hash silently goes stale.
"""

import numpy as np
import pytest

from slipsolve import FaultTrace
from slipsolve import layout as L
from slipsolve.mesh import FaultMesh

# The same near-east-west trace the mesh tests use: the orientation that defeats
# a fixed-axis winding test, and so the one worth triangulating irregularly.
_LON = np.array([-68.681, -68.400, -68.000, -67.600, -67.200, -66.800, -66.523])
_LAT = np.array([10.410, 10.487, 10.541, 10.563, 10.593, 10.621, 10.630])


@pytest.fixture
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture
def frame(trace):
    return trace.local_frame()


def _stations(trace, frame, spacing=3e3):
    x, y = trace.resample(spacing, frame).to_local(frame)
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


# -- winding ---------------------------------------------------------------
def test_the_lattice_split_is_clockwise_in_parameter_space():
    """The measured fact the whole Delaunay path rests on.

    ``PARAM_ORIENTATION`` is not a convention picked for tidiness; it is what
    ``_lattice_triangles`` happens to produce, and the 3-D normals that come out
    on the trace's left-hand side follow from it.
    """
    ns, nz = 6, 4
    s, z = np.meshgrid(np.arange(ns, dtype=float),
                       -np.arange(nz, dtype=float), indexing="xy")
    params = np.column_stack([s.ravel(), z.ravel()])
    areas = L.signed_areas(params, L._lattice_triangles(ns, nz))

    assert np.all(np.sign(areas) == L.PARAM_ORIENTATION)


def test_delaunay_reproduces_the_lattice_meshs_geometry(trace, frame):
    """Retriangulating a lattice's own nodes changes the diagonals, nothing else.

    The acceptance test for the winding rule. Delaunay picks different diagonals
    -- the four corners of a quad are cocircular, so its choice is arbitrary --
    but every normal must still land on the trace's left, the dips must match and
    the total area is a property of the node set, not the triangulation.
    """
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=5e3)
    triangles = L.triangulate(mesh.params)
    rebuilt = FaultMesh(mesh.nodes, triangles, mesh.params, frame=frame)

    assert rebuilt.n_elements == mesh.n_elements
    assert set(map(tuple, np.sort(triangles, axis=1))) != set(
        map(tuple, np.sort(mesh.triangles, axis=1))
    ), "a regular lattice is cocircular, so Delaunay should differ from the quad split"

    rebuilt._check_winding(trace)
    np.testing.assert_allclose(rebuilt.areas.sum(), mesh.areas.sum(), rtol=1e-12)
    np.testing.assert_allclose(rebuilt.dip, mesh.dip, atol=1e-9)
    np.testing.assert_array_less(0.0, np.abs(rebuilt.normals[:, 1]))


def test_orient_triangles_fixes_a_deliberately_reversed_triangulation():
    # Node (i, j) is index j * ns + i, so the second row is the deeper one.
    params = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    triangles = L._lattice_triangles(2, 2)
    flipped = triangles[:, [0, 2, 1]]

    np.testing.assert_array_equal(L.orient_triangles(params, flipped), triangles)
    # Idempotent: orienting an already-correct set changes nothing.
    np.testing.assert_array_equal(L.orient_triangles(params, triangles), triangles)


# -- backward compatibility ------------------------------------------------
def test_a_uniform_layout_is_the_classic_lattice_bit_for_bit(trace, frame):
    stations = _stations(trace, frame)
    depths = np.linspace(0.0, 20e3, 8)
    layout = L.graded_layout(stations[-1], depths, counts=stations.size,
                             stations=stations)

    assert layout.uniform_counts == (stations.size, depths.size)
    np.testing.assert_array_equal(layout.triangles,
                                  L._lattice_triangles(stations.size, depths.size))
    for j, sl in enumerate(layout.level_slices()):
        # Bit for bit, not to a tolerance: `np.interp` returns its own knots
        # exactly, which is what leaves `FaultMesh.digest()` unchanged.
        np.testing.assert_array_equal(layout.params[sl, 0], stations)
        np.testing.assert_array_equal(layout.params[sl, 1], -depths[j])


def test_curved_with_the_new_knobs_defaults_to_the_old_mesh(trace, frame):
    """The gate: adding the layout arguments must not move an existing mesh."""
    plain = FaultMesh.curved(trace, frame, uniform_dip=75.0,
                             max_depth=20e3, edge_length=4e3)
    explicit = FaultMesh.curved(
        trace, frame, uniform_dip=75.0, max_depth=20e3, edge_length=4e3,
        nodes_per_level=plain.attrs["n_along"],
    )
    assert explicit.digest() == plain.digest()
    assert FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=4e3).digest() \
        == FaultMesh.curved(trace, frame, uniform_dip=90.0,
                            max_depth=20e3, edge_length=4e3).digest()


# -- irregular layouts -----------------------------------------------------
def test_a_layout_with_a_different_count_at_every_level_builds_and_winds(trace, frame):
    counts = [40, 34, 28, 22, 16, 12]
    depths = np.array([0.0, 1.5e3, 4e3, 8e3, 14e3, 22e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=80.0, depths=depths,
                            nodes_per_level=counts, edge_length=3e3)

    assert mesh.n_nodes == sum(counts)
    assert mesh.attrs["nodes_per_level"] == counts
    assert "n_along" not in mesh.attrs, "n_along means nothing once the levels differ"
    mesh._check_winding(trace)
    # Every node is in a triangle and every triangle has real area.
    assert set(np.unique(mesh.triangles)) == set(range(mesh.n_nodes))
    np.testing.assert_array_less(0.0, mesh.areas)


def test_the_boundary_sides_still_resolve_to_whole_node_rows(trace, frame):
    counts = [40, 30, 20, 12]
    depths = np.array([0.0, 3e3, 9e3, 20e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=depths,
                            nodes_per_level=counts, edge_length=3e3)

    assert int(mesh.boundary_nodes("top").sum()) == counts[0]
    assert int(mesh.boundary_nodes("bottom").sum()) == counts[-1]
    # One node per level at each end, whatever the level holds.
    assert int(mesh.boundary_nodes("left").sum()) == len(counts)
    assert int(mesh.boundary_nodes("right").sum()) == len(counts)
    for side in L.EDGE_SIDES:
        assert mesh.boundary_elements(side).any()


def test_along_spacing_may_be_a_callable_of_depth(trace, frame):
    depths = np.array([0.0, 5e3, 12e3, 25e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=depths,
                            along_spacing=lambda z: 2e3 + 0.4 * z, edge_length=3e3)
    counts = mesh.attrs["nodes_per_level"]

    assert counts == sorted(counts, reverse=True), "deeper levels must be coarser"
    assert counts[0] > 2 * counts[-1]


# -- guards ----------------------------------------------------------------
def test_a_collapsed_boundary_raises_instead_of_silently_unconstraining_it():
    """The silent failure this guard exists for.

    ``zero_slip_boundary`` selects whatever the boundary mask flags, so an edge
    that resolves to a single node stops constraining the mesh's rim -- and slip
    runs off it with nothing raised anywhere.
    """
    params = np.array([[0.0, 0.0], [1e3, 0.0], [2e3, 0.0],
                       [1.0, -1e3], [1e3, -1e3], [2e3 - 1.0, -1e3]])
    with pytest.raises(ValueError, match="exact equality"):
        L.require_boundaries(params)

    snapped = L.snap_boundary(params, length=2e3, tol=10.0)
    assert L.require_boundaries(snapped)["left"] == 2


def test_an_unbalanced_request_is_refined_rather_than_triangulated_badly():
    counts = [100, 4, 100]
    with pytest.warns(RuntimeWarning, match="within 2.0:1"):
        layout = L.graded_layout(50e3, np.array([0.0, 5e3, 20e3]), counts=counts)

    ratios = layout.counts[1:] / layout.counts[:-1]
    np.testing.assert_array_less(np.maximum(ratios, 1.0 / ratios), 2.0 + 1e-9)
    assert layout.counts[0] == 100 and layout.counts[-1] == 100


def test_a_duplicated_parameter_point_is_refused_by_name():
    params = np.array([[0.0, 0.0], [1e3, 0.0], [1e3, 0.0], [0.0, -1e3], [1e3, -1e3]])
    with pytest.raises(ValueError, match="no triangle"):
        L.from_points(params, length=1e3, snap=False)


def test_an_irregular_vertical_mesh_is_accurate_despite_looking_near_vertical():
    """Why there is no near-vertical guard, pinned so nobody adds one.

    An irregular triangulation leaves most of a vertical mesh's elements a hair
    off 90 degrees -- ``arccos(|n_z|)`` cannot return exactly 90 when the normal's
    vertical component is at machine epsilon -- which looks exactly like the
    catastrophic band recorded beside vertical. It is not: the band is a
    rectangle-as-two-TDEs disagreement with the *Okada* oracle, and against
    ``cutde``, an independent implementation of the same triangular dislocation,
    the whole design matrix is fine.

    Dev-only: ``cutde`` is in the ``dev`` extra and is never imported at runtime.
    """
    hs = pytest.importorskip("cutde.halfspace")
    from slipsolve.greens import HalfSpaceTDE

    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0,
                            depths=np.array([0.0, 3e3, 9e3, 20e3]),
                            nodes_per_level=[40, 30, 20, 12], edge_length=3e3)

    offset = np.abs(mesh.dip - 90.0)
    assert ((offset > 0) & (offset < 1e-2)).sum() > 0.5 * mesh.n_elements

    rng = np.random.default_rng(3)
    x, y = rng.uniform(-60e3, 60e3, 300), rng.uniform(-60e3, 60e3, 300)
    reference = hs.disp_matrix(np.column_stack([x, y, np.zeros_like(x)]),
                               mesh.vertices, 0.25)
    up = np.ones_like(x)
    green = HalfSpaceTDE(0.25).los_matrix(mesh, x, y, 0 * up, 0 * up, up)
    for component in (0, 1):
        block = green[:, component * mesh.n_elements:(component + 1) * mesh.n_elements]
        slip = np.zeros(3)
        slip[component] = 1.0
        expected = (reference @ slip)[:, 2, :]
        assert np.abs(block - expected).max() / np.abs(expected).max() < 1e-12


def test_bias_l_still_raises_but_now_points_at_the_knob_that_works(trace, frame):
    with pytest.raises(ValueError, match="nodes_per_level"):
        FaultMesh.curved(trace, frame, uniform_dip=90.0, bias_l=1.2)


def test_depths_and_bias_w_are_refused_together(trace, frame):
    with pytest.raises(ValueError, match="bias_w"):
        FaultMesh.curved(trace, frame, uniform_dip=90.0,
                         depths=np.array([0.0, 5e3, 10e3]), bias_w=1.15)


def test_a_layout_and_the_knobs_that_would_contradict_it_are_refused(trace, frame):
    layout = L.graded_layout(100e3, np.array([0.0, 5e3, 10e3]), counts=20)
    with pytest.raises(ValueError, match="contradicting"):
        FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                              nodes_per_level=10)


# -- freeform --------------------------------------------------------------
def test_lloyd_places_small_elements_where_the_size_field_is_small(trace, frame):
    layout = L.lloyd_layout(120e3, 20e3, lambda s, d: 2e3 + 0.5 * d, iterations=10)
    mesh = FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                                 edge_length=3e3)

    assert layout.counts is None, "a relaxed cloud has no rows"
    mesh._check_winding(trace)
    depth = -mesh.centroids[:, 2]
    size = np.sqrt(2.0 * mesh.areas)
    shallow = size[depth < 5e3].mean()
    deep = size[depth > 12e3].mean()
    assert deep > 1.5 * shallow, f"shallow {shallow:.0f} m vs deep {deep:.0f} m"
    for side in L.EDGE_SIDES:
        assert int(L.boundary_node_mask(layout.params, side).sum()) >= 2


def test_the_relaxation_honours_the_size_field_it_was_given():
    """A plain area-weighted centroid is the CVT iteration for a *uniform*
    density, so it relaxes toward equal element sizes and flattens the grading
    the rejection sampling just put in. Weighting by ``1/h**2`` is what makes the
    achieved size track ``h``; without it the shallow-to-deep size ratio came
    back at 2.16 against a wanted 2.85.
    """
    h = lambda s, d: 1e3 + 0.15 * d          # noqa: E731 -- 1 km to 4 km down dip
    layout = L.lloyd_layout(60e3, 20e3, h, iterations=25)
    params, triangles = layout.params, layout.triangles
    centre = params[triangles].mean(axis=1)
    size = np.sqrt(np.abs(L.signed_areas(params, triangles)))
    depth = -centre[:, 1]

    shallow = np.median(size[depth < 4e3])
    deep = np.median(size[depth > 16e3])
    wanted = h(0.0, 18e3) / h(0.0, 2e3)
    assert abs((deep / shallow) / wanted - 1) < 0.15, (
        f"size ratio {deep / shallow:.2f} against a target ratio of {wanted:.2f}"
    )


def test_the_relaxation_is_what_makes_the_cloud_usable():
    """The rejection-sampled cloud on its own is full of slivers -- 0.4 degrees
    measured -- so the iteration count is not a tuning nicety."""
    h = lambda s, d: 2e3 + 0.2 * d           # noqa: E731
    raw = L.lloyd_layout(60e3, 20e3, h, iterations=0)
    relaxed = L.lloyd_layout(60e3, 20e3, h, iterations=25)
    assert raw.quality()["min_angle_deg"] < 5.0
    assert relaxed.quality()["min_angle_deg"] > 15.0


def test_the_layout_is_reproducible():
    """``FaultMesh.digest()`` hashes the node coordinates, so a layout that moved
    between runs would stale every persisted store for no reason."""
    h = lambda s, d: 2e3 + 0.2 * d           # noqa: E731
    a = L.lloyd_layout(60e3, 20e3, h, iterations=10)
    b = L.lloyd_layout(60e3, 20e3, h, iterations=10)
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.triangles, b.triangles)


def test_a_layout_longer_than_the_resampled_trace_is_refused(trace, frame):
    """``np.interp`` clamps past the last station, so nodes beyond the end of the
    trace as ``curved`` resamples it would all land on the end point. An end
    column a few metres over is let through (clamped, as before); a longer layout,
    or an interior node past the end, is refused. Shorter layouts mesh part of
    the trace, as before."""
    from slipsolve.mesh import LAYOUT_OVERRUN_FRACTION

    edge = 3e3
    x, y = trace.resample(edge, frame).to_local(frame)
    end = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    tol = LAYOUT_OVERRUN_FRACTION * edge

    def rect(length, extra=None):
        s, z = np.meshgrid(np.linspace(0.0, length, 6), -np.linspace(0.0, 15e3, 3))
        params = np.column_stack([s.ravel(), z.ravel()])
        if extra is not None:
            params = np.vstack([params, extra])
        return L.MeshLayout(params=params, triangles=L.triangulate(params))

    def build(layout):
        return FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0, edge_length=edge)

    build(rect(end))
    build(rect(end + 0.5 * tol))
    build(rect(0.5 * end))
    with pytest.raises(ValueError, match="clamped onto the end point"):
        build(rect(end + 1e3))
    with pytest.raises(ValueError, match="clamped onto the end point"):
        build(rect(end + 0.5 * tol, extra=[[end + 0.25 * tol, -7.5e3]]))


def test_an_unstructured_layout_keeps_a_rectangular_outline():
    """Freeform means the discretisation, not the fault's extent: the four edges
    stay straight because ``boundary_node_mask`` finds them by exact equality and
    the zero-slip constraint depends on that."""
    layout = L.lloyd_layout(60e3, 20e3, lambda s, d: 2e3 + 0.2 * d, iterations=15)
    s, z = layout.params[:, 0], layout.params[:, 1]
    for side, expected in (("left", 0.0), ("right", 60e3),
                           ("top", 0.0), ("bottom", -20e3)):
        mask = L.boundary_node_mask(layout.params, side)
        assert mask.sum() >= 2
        got = (s if side in ("left", "right") else z)[mask]
        np.testing.assert_array_equal(got, np.full(mask.sum(), expected))


# -- more than one fault in one mesh ------------------------------------------

@pytest.fixture
def two_faults(trace, frame):
    """One mesh carrying two disconnected faults, of different depths.

    Different depths on purpose: it is what makes the global-extremum test fail
    in the way that matters, since only the deeper piece can own the mesh's
    minimum ``z``.
    """
    deep = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    shallow = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=10e3)
    gap = deep.params[:, 0].max() + 40e3
    return FaultMesh(
        np.vstack([deep.nodes, shallow.nodes + np.array([0.0, 300e3, 0.0])]),
        triangles=np.vstack([deep.triangles, shallow.triangles + deep.n_nodes]),
        params=np.vstack([deep.params, shallow.params + np.array([gap, 0.0])]),
        frame=frame,
    )


def test_a_single_fault_is_one_component_and_the_mask_does_not_move(trace, frame):
    """The backward-compatibility gate: every mesh any builder here produces has
    one component, and there the per-component test *is* the global one."""
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    assert mesh.node_components.max() == 0
    for side in L.EDGE_SIDES:
        np.testing.assert_array_equal(
            mesh.boundary_nodes(side),
            L.boundary_node_mask(mesh.params, side))


def test_the_global_extremum_test_loses_a_second_faults_edges(two_faults):
    """⚠️ The bug, stated directly. Do not "simplify" the mask back to a global
    extremum: on two faults only the deeper one has a bottom, and only the outer
    ends of the pair are left and right, so three of the second fault's four
    edges vanish and nothing raises."""
    labels = two_faults.node_components
    assert sorted(np.unique(labels)) == [0, 1]

    per_component = {
        side: [int(two_faults.boundary_nodes(side)[labels == k].sum())
               for k in (0, 1)]
        for side in L.EDGE_SIDES
    }
    global_only = {
        side: [int(L.boundary_node_mask(two_faults.params, side)[labels == k].sum())
               for k in (0, 1)]
        for side in L.EDGE_SIDES
    }

    # The shallower fault has no bottom edge at all under the global test.
    assert global_only["bottom"][1] == 0
    assert per_component["bottom"][1] > 0
    # And each fault owns exactly one of left/right, instead of both.
    assert 0 in (global_only["left"][1], global_only["right"][0])
    for side in L.EDGE_SIDES:
        assert all(n >= 2 for n in per_component[side]), side


def test_require_boundaries_passes_a_pair_it_should_refuse(two_faults):
    """The failure is silent, which is the only reason it is worth a test: the
    global check is satisfied by one fault's edges and never looks at the other."""
    labels = two_faults.node_components
    L.require_boundaries(two_faults.params)          # passes, wrongly

    counts = L.require_boundaries(two_faults.params, labels=labels)
    assert all(n >= 2 for n in counts.values())
    # It reports the THINNEST piece, since one unconstrained fault invalidates
    # the run whatever the other one has.
    assert counts["bottom"] == min(
        int(L.boundary_node_mask(two_faults.params, "bottom", labels)[labels == k].sum())
        for k in (0, 1))


def test_a_component_with_a_collapsed_edge_is_still_refused(trace, frame):
    """Per-component counting must not become a way of passing: a piece whose
    own edge has one node is as unconstrained as it was before."""
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    params = np.vstack([mesh.params, [[1e6, 0.0], [1e6 + 1.0, -1.0]]])
    labels = np.concatenate([np.zeros(mesh.n_nodes, int), [1, 1]])
    with pytest.raises(ValueError, match="connected pieces"):
        L.require_boundaries(params, labels=labels)


def test_the_boundary_mask_checks_its_label_count(trace, frame):
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    with pytest.raises(ValueError, match="one component label per node"):
        L.boundary_node_mask(mesh.params, "top", np.zeros(3, int))
