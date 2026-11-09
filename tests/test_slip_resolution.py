"""Pins model resolution and the mesh and sampling it drives.

Variance reduction measures fit, and a slip model fits happily on a
discretisation the data never constrained -- the spare freedom is spent on noise.
Resolution measures the other thing, and these tests hold the two claims that
make it usable: that it behaves like a resolution (falls with the damping, and
falls with depth), and that the mesh built from it converges instead of running
away.

The runaway is the reason `budget=` exists, and it is pinned here rather than
described: without a budget the loop can run end to end, report convergence, and
return a mesh of a few dozen elements that fits almost nothing.
"""

import warnings

import numpy as np
import pytest

from slipsolve import FaultTrace, FaultMesh, Observations, SlipInversion
from slipsolve import resolution as R

import slip_synthetic as ss

_LON = np.array([-69.0, -68.5, -68.0, -67.5, -67.0, -66.5, -66.0])
_LAT = np.array([10.55, 10.56, 10.57, 10.58, 10.585, 10.59, 10.60])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    slip = ss.tapered_slip(mesh, peak=-2.0)
    parts = [
        Observations.from_los(
            ss.forward_los_stack(mesh, slip, trace, frame, geometry=geometry,
                                 spacing=8000.0, noise=0.004, seed=1),
            name=geometry, frame=frame, trace=trace, rms_min=0.006,
            width_min=10000.0, width_max=60000.0, exclude_within=8000.0)
        for geometry in ("asc", "desc")
    ]
    obs = Observations.concat(parts)
    return trace, frame, mesh, SlipInversion(mesh, obs, ramp="offset")


# -- does it behave like a resolution? -------------------------------------
def test_resolution_falls_with_the_damping(setup):
    """The defining property: at zero damping the data decides everything."""
    _, _, _, inversion = setup
    ranks = [R.resolution_diagonal(inversion, w).attrs["effective_rank"]
             for w in (0.001, 0.1, 10.0, 1000.0)]

    assert ranks == sorted(ranks, reverse=True), ranks
    assert ranks[0] > 5.0 * ranks[-1]
    assert ranks[-1] < 1.0, "at heavy damping almost nothing is resolved"

    loose = R.resolution_diagonal(inversion, 1e-6)["resolution"].values
    tight = R.resolution_diagonal(inversion, 1e6)["resolution"].values
    assert loose.max() > 0.9
    assert tight.max() < 0.01
    for values in (loose, tight):
        np.testing.assert_array_less(-1e-9, values)
        np.testing.assert_array_less(values, 1.0 + 1e-9)


def test_shallow_elements_resolve_better_than_deep_ones(setup):
    """Surface data sees a shallow patch sharply and a deep one hardly at all."""
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    depth = -mesh.centroids[:, 2]

    shallow = per_element[depth < 0.3 * depth.max()].mean()
    deep = per_element[depth > 0.7 * depth.max()].mean()
    assert shallow > 10.0 * deep, f"shallow {shallow:.4f} vs deep {deep:.4f}"


def test_the_ramp_columns_are_resolved_because_nothing_damps_them(setup):
    _, _, _, inversion = setup
    ramp = R.resolution_diagonal(inversion, 100.0).attrs["ramp_resolution"]

    assert len(ramp) == inversion.n_ramp
    np.testing.assert_array_less(0.9, np.asarray(ramp))


# -- the size field --------------------------------------------------------
def test_a_flat_resolution_field_asks_for_the_mesh_it_was_given(setup):
    """The identity the budget arithmetic has to satisfy, or every mesh drifts."""
    _, _, mesh, _ = setup
    size = R.target_element_size(mesh, np.full(mesh.n_elements, 0.5))

    np.testing.assert_allclose(size, R.element_size(mesh), rtol=1e-12)


def test_the_budget_sets_the_element_count_and_resolution_sets_the_places(setup):
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))

    size = R.target_element_size(mesh, per_element)
    for factor in (0.5, 1.0, 2.0):
        wanted = int(factor * mesh.n_elements)
        got = R.adapt_layout(mesh, size, budget=wanted).n_elements
        # The 2:1 balance rule and integer node counts move it a little; the
        # point is that it tracks the request instead of drifting on its own.
        assert 0.8 * wanted <= got <= 1.25 * wanted, (factor, wanted, got)


def test_an_unbudgeted_step_drifts_and_a_budgeted_one_does_not(setup):
    """The failure `budget=` exists for, as a regression test.

    Targeting the median looks self-normalising and is not: most elements sit
    near zero resolution and hit the growth cap while few can shrink. Left to
    itself the loop can lose most of its elements in a few rounds while the
    variance reduction collapses -- converging, and returning nothing.
    """
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    size = R.target_element_size(mesh, per_element)

    assert size.mean() > 1.2 * R.element_size(mesh).mean(), "the size field does drift"
    free = R.adapt_layout(mesh, size, budget=None).n_elements
    held = R.adapt_layout(mesh, size, budget="conserve").n_elements

    assert abs(free - mesh.n_elements) > 0.25 * mesh.n_elements, free
    assert abs(held - mesh.n_elements) <= 0.25 * mesh.n_elements, held


def test_max_ratio_bounds_a_single_step_even_where_resolution_underflows(setup):
    """Deep resolution is not merely small, it is exactly zero.

    ``target / r`` is then whatever the floor makes it and ``gain`` cannot temper
    an infinity, so the bound has to be on the ratio itself.
    """
    _, _, mesh, _ = setup
    resolution = np.zeros(mesh.n_elements)
    resolution[0] = 1.0

    ratio = R.target_element_size(mesh, resolution, target=1.0,
                                  max_ratio=2.0) / R.element_size(mesh)
    np.testing.assert_array_less(ratio, 2.0 + 1e-9)
    np.testing.assert_array_less(0.5 - 1e-9, ratio)


def test_smoothing_the_size_field_keeps_it_lipschitz(setup):
    _, _, mesh, _ = setup
    spike = np.full(mesh.n_elements, 5e3)
    spike[mesh.n_elements // 2] = 50e3

    smoothed = R.smooth_over_mesh(mesh, spike, rounds=3)
    assert smoothed.max() < 0.5 * spike.max()
    np.testing.assert_allclose(smoothed.min(), 5e3, rtol=0.5)


# -- the adapted layout ----------------------------------------------------
def test_the_adapted_layout_is_finer_where_the_data_resolves_better(setup):
    trace, frame, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    layout = R.adapt_layout(mesh, R.target_element_size(mesh, per_element))

    assert layout.counts is not None
    assert layout.counts[0] > layout.counts[-1], "the surface must be denser than the base"
    # Levels thicken downward, the same thing bias_w does by hand.
    thickness = np.diff(layout.levels)
    assert thickness[-1] > thickness[0]

    built = FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                                  edge_length=10e3)
    built._check_winding(trace)
    for side in ("top", "bottom", "left", "right"):
        assert built.boundary_elements(side).any()


def test_the_adapted_mesh_spans_exactly_the_same_fault(setup):
    """A re-mesh must not quietly change the fault's extent."""
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    layout = R.adapt_layout(mesh, R.target_element_size(mesh, per_element))

    np.testing.assert_allclose(layout.params[:, 0].max(), mesh.params[:, 0].max())
    np.testing.assert_allclose(layout.params[:, 1].min(), mesh.params[:, 1].min())
    assert layout.params[:, 0].min() == 0.0


def test_the_levels_never_jump_by_more_than_the_balance_factor(setup):
    _, _, mesh, _ = setup
    # A size field that saturates at depth would put a 17 km level under a
    # 2.5 km one without `_balance_levels`.
    depth = -mesh.centroids[:, 2]
    size = 1e3 + 40.0 * depth
    layout = R.adapt_layout(mesh, size, balance=2.0)

    thickness = np.diff(layout.levels)
    ratio = thickness[1:] / thickness[:-1]
    np.testing.assert_array_less(np.maximum(ratio, 1.0 / ratio), 2.0 + 1e-6)


# -- the cheap initialiser and the sampling field --------------------------
def test_the_column_norm_is_not_a_resolution_proxy(setup):
    """Why there is no cheap stand-in for `R`, pinned so nobody adds one.

    ``||G[:, k]||`` is the obvious way to avoid a factorisation and it is blind to
    the thing that matters: a deeper element radiates a broader but weaker pattern
    whose norm is roughly conserved, while resolution collapses because deep
    elements trade off with each other. A column norm cannot see a trade-off.
    """
    _, _, mesh, inversion = setup
    n = inversion.basis.n_basis
    norms = np.linalg.norm(inversion.g[:, :2 * n] * inversion.w[:, None], axis=0)
    column = np.maximum(norms[:n], norms[n:2 * n])
    proper = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    depth = -mesh.centroids[:, 2]

    assert np.corrcoef(proper, depth)[0, 1] < -0.5, "resolution must fall with depth"
    assert abs(np.corrcoef(column, depth)[0, 1]) < 0.2, "the column norm does not"

    deep = depth > 0.7 * depth.max()
    shallow = depth < 0.3 * depth.max()
    assert proper[shallow].mean() > 100.0 * proper[deep].mean()
    assert 0.5 < column[shallow].mean() / column[deep].mean() < 2.0


def test_resolution_gain_points_at_the_unresolved_part_of_the_fault(setup):
    """Lohman & Simons' criterion, as a field the existing quadtree can split on."""
    trace, frame, mesh, inversion = setup
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    stack = ss.forward_los_stack(mesh, ss.tapered_slip(mesh, peak=-2.0), trace, frame,
                                 geometry="asc", spacing=8000.0, seed=2)

    resolution = R.resolution_diagonal(inversion, 0.3)
    # Declare the western third unresolved and everything else perfect, so the
    # gain has to follow the fault rather than the scene's own structure.
    values = np.ones(resolution.sizes["basis"])
    values[mesh.element_params[:, 0] < 0.35 * mesh.params[:, 0].max()] = 0.0
    resolution["resolution"].values[:] = values

    gain = R.resolution_gain(model, stack, resolution, frame=frame, spacing=8000.0)

    x = np.asarray(stack.ds["x"].values, dtype=float)
    assert gain.shape == (stack.ds["y"].size, x.size)
    columns = np.broadcast_to(x, gain.shape)
    # The frame's x runs west to east, and so does arc length along this trace.
    west = np.nanmean(np.where(columns < np.median(x), gain, np.nan))
    east = np.nanmean(np.where(columns > np.median(x), gain, np.nan))
    assert west > east, f"gain west {west:.3f} vs east {east:.3f}"


def test_a_wrongly_shaped_resolution_vector_is_refused_by_name(setup):
    _, _, mesh, _ = setup
    with pytest.raises(ValueError, match="element_resolution"):
        R.target_element_size(mesh, np.ones(3))


# -- unstructured adaptation, where the depth levels stop being straight ------
#
# A row-structured layout reads its depth levels off one vertical, at the fault's
# midpoint, because a level is a single depth for the whole fault. When the size
# field varies along strike -- which is what uneven coverage along the fault
# produces -- that constraint is what limits how well the mesh can follow it.

def _along_strike_size_field(mesh, fine=3e3, coarse=9e3):
    """A target size that is fine in the west and coarse in the east.

    Deliberately *not* from a solved inversion: the property under test is what
    the builders do with a size field that varies along strike, and constructing
    it directly keeps the test fast and its intent legible.
    """
    s = mesh.element_params[:, 0]
    fraction = (s - s.min()) / max(np.ptp(s), 1.0)
    return fine + (coarse - fine) * fraction


def _follows(layout, want_at):
    """Median |relative error| of the achieved element size, global scale removed.

    The scale is divided out because the element budget sets it -- what is being
    measured is whether the *shape* of the size field survived.
    """
    from slipsolve.layout import signed_areas

    centre = layout.params[layout.triangles].mean(axis=1)
    got = np.sqrt(np.abs(signed_areas(layout.params, layout.triangles)))
    want = want_at(centre[:, 0], -centre[:, 1])
    return float(np.median(np.abs(got / (want * np.median(got / want)) - 1)))


def test_unstructured_follows_an_along_strike_size_field_better_than_rows(setup):
    """The measurement the ``structure="unstructured"`` mode exists for."""
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    want_at = R.size_field(mesh, sizes)

    rows = R.adapt_layout(mesh, sizes, budget="conserve", structure="rows")
    free = R.adapt_layout(mesh, sizes, budget="conserve", structure="unstructured")

    assert rows.counts is not None
    assert free.counts is None, "an unstructured layout has no depth levels"
    assert _follows(free, want_at) < _follows(rows, want_at)


def test_unstructured_respects_the_budget_and_the_boundaries(setup):
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    for budget in ("conserve", 120):
        layout = R.adapt_layout(mesh, sizes, budget=budget,
                                structure="unstructured")
        wanted = mesh.n_elements if budget == "conserve" else budget
        assert abs(layout.n_elements - wanted) <= 0.2 * wanted
        for side in ("top", "bottom", "left", "right"):
            from slipsolve.layout import boundary_node_mask
            assert int(boundary_node_mask(layout.params, side).sum()) >= 2


def test_unstructured_records_its_provenance_and_quality(setup):
    _, _, mesh, _ = setup
    layout = R.adapt_layout(mesh, _along_strike_size_field(mesh),
                            structure="unstructured")
    assert layout.attrs["layout"] == "adapted_unstructured"
    assert layout.attrs["quality"]["n_elements"] == layout.n_elements
    assert layout.attrs["quality"]["min_angle_deg"] > R.MIN_ANGLE_WARN_DEG


def test_an_unrelaxed_cloud_is_flagged_as_slivered(setup):
    """The adaptive loop runs unattended, so a degenerate triangulation has to
    say so; ``iterations=0`` is the rejection-sampled cloud, which is 0.4 deg."""
    _, _, mesh, _ = setup
    with pytest.warns(RuntimeWarning, match="minimum triangle angle"):
        R.adapt_layout(mesh, _along_strike_size_field(mesh), budget=None,
                       structure="unstructured", iterations=0)


def test_the_rows_structure_is_unchanged(setup):
    """The default must be exactly the rows builder, or every existing run's
    mesh moves."""
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    default = R.adapt_layout(mesh, sizes, budget="conserve")
    explicit = R.adapt_layout(mesh, sizes, budget="conserve", structure="rows")
    np.testing.assert_array_equal(default.params, explicit.params)
    assert default.attrs["layout"] == "adapted"


def test_an_unknown_structure_is_refused(setup):
    _, _, mesh, _ = setup
    with pytest.raises(ValueError, match="structure must be"):
        R.adapt_layout(mesh, _along_strike_size_field(mesh),
                       budget=None, structure="quadtree")


def test_the_size_field_keeps_the_shape_it_was_asked_on(setup):
    """The row builder asks along a line and the unstructured one on a 2-D probe
    grid; a lookup that only handled 1-D would break the second with a broadcast error."""
    _, _, mesh, _ = setup
    at = R.size_field(mesh, _along_strike_size_field(mesh))
    line = at(np.linspace(0.0, 40e3, 7), np.full(7, 5e3))
    assert line.shape == (7,)
    grid_s, grid_d = np.meshgrid(np.linspace(0.0, 40e3, 5),
                                 np.linspace(0.0, 15e3, 4))
    assert at(grid_s, grid_d).shape == (4, 5)


# -- the full model/data resolution pair --------------------------------------
#
# The diagonal answers "how much of this estimate is data". It cannot answer
# "over what distance", which is the question that decides whether a mesh is
# finer than the data supports -- for that you need the off-diagonal structure.

def test_the_two_traces_are_the_same_number(setup):
    """``trace(A M^-1 A') == trace(M^-1 A'A)`` identically, so model and data
    resolution must agree on the effective parameter count. It is the one exact
    check available on either of them, and it catches the two halves being built
    from different factorizations."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    assert report.attrs["trace_identity_error"] < 1e-8
    assert report.attrs["trace_model_resolution"] == pytest.approx(
        float(R.data_importance(inv, 1.0).sum()), rel=1e-9)


def test_the_matrix_diagonal_is_the_diagonal_function(setup):
    _, _, _, inv = setup
    matrix = R.resolution_matrix(inv, 1.0)
    diag = R.resolution_diagonal(inv, 1.0)
    n = inv.basis.n_basis
    np.testing.assert_allclose(np.diag(matrix)[:n], diag["strike_slip"].values,
                               rtol=1e-10, atol=1e-12)


def test_kernel_width_survives_a_smoothing_change_where_the_diagonal_does_not(setup):
    """The measurement behind ``resolution_report``'s advice, and the reason the
    adaptive loop misbehaves: ``target_element_size`` sizes the mesh from the
    diagonal, which is mostly a function of lambda, rather than from the kernel
    width, which is set by the data geometry."""
    _, _, _, inv = setup
    widths, diagonals = [], []
    for lam in (0.1, 10.0):
        report = R.resolution_report(inv, lam)
        widths.append(report.attrs["median_kernel_width_m"])
        diagonals.append(float(np.median(report["resolution"].values)))
    width_ratio = max(widths) / min(widths)
    diagonal_ratio = max(diagonals) / min(diagonals)
    assert diagonal_ratio > 3.0, diagonal_ratio       # lambda moves the diagonal
    assert width_ratio < 1.5, width_ratio             # ...and barely the width
    assert width_ratio < diagonal_ratio


def test_the_kernel_is_wider_than_the_elements_it_averages(setup):
    """A mesh cannot claim more detail than its own averaging kernels carry."""
    _, _, mesh, inv = setup
    report = R.resolution_report(inv, 1.0)
    assert report.attrs["over_discretisation"] > 1.0
    assert report.attrs["median_kernel_width_m"] > np.median(R.element_size(mesh))


def test_data_importance_is_a_diagonal_of_a_projection(setup):
    """``N`` is a projection-like operator, so its diagonal lies in [0, 1] and
    sums to the effective rank -- a datum near 1 is one nothing else replaces."""
    _, _, _, inv = setup
    importance = R.data_importance(inv, 1.0)
    assert importance.shape == (inv.obs.n,)
    assert importance.min() >= -1e-12
    assert importance.max() <= 1.0 + 1e-9
    assert importance.sum() == pytest.approx(
        R.resolution_report(inv, 1.0).attrs["effective_rank"]
        + sum(R.resolution_diagonal(inv, 1.0).attrs["ramp_resolution"]), rel=1e-8)


def test_more_smoothing_resolves_fewer_parameters_and_less_data(setup):
    _, _, _, inv = setup
    ranks = [float(R.data_importance(inv, lam).sum()) for lam in (0.1, 1.0, 10.0)]
    assert ranks[0] > ranks[1] > ranks[2]


def test_kernels_refuse_a_matrix_from_a_different_basis(setup):
    _, _, mesh, inv = setup
    matrix = R.resolution_matrix(inv, 1.0)
    wrong = SlipInversion(mesh, inv.obs, basis="element"
                          if inv.basis.name == "node" else "node")
    with pytest.raises(ValueError, match="parameters"):
        R.resolution_kernels(wrong, matrix)


# -- sizing a mesh from the kernel width instead of the diagonal --------------

def test_the_diagonal_size_field_saturates_where_the_kernel_one_does_not(setup):
    """The reason the kernel criterion exists, and it is *not* lambda drift.

    `target_element_size` forms `(target / r) ** gain` on a diagonal that has
    underflowed over most of the fault, so it is a ratio of two nearly-zero
    numbers and runs to its clamp: here a large fraction of elements are pinned
    at `max_ratio` and carry no information at all. The kernel width is a real
    length everywhere.
    """
    _, _, mesh, inv = setup
    per_el = R.element_resolution(inv, R.resolution_diagonal(inv, 1.0))
    ratio = (np.median(per_el) / np.maximum(per_el, 1e-12)) ** 0.5
    pinned = (ratio >= 3.0 - 1e-9) | (ratio <= 1 / 3 + 1e-9)
    assert pinned.mean() > 0.3, f"only {100 * pinned.mean():.0f}% pinned"

    width = R.kernel_element_size(inv, 1.0, size_min=1.0, size_max=1e9)
    assert np.all(np.isfinite(width)) and width.min() > 0.0
    # ...and it is not a constant either: it still varies across the mesh.
    assert np.ptp(width) > 0.05 * np.median(width)


def test_kernel_element_size_never_degenerates(setup):
    """Where resolution underflows, ``target / r`` saturates its clamp; the
    kernel width stays a real length everywhere, which is the whole point."""
    _, _, _, inv = setup
    for lam in (0.1, 1.0, 100.0):
        size = R.kernel_element_size(inv, lam)
        assert np.all(np.isfinite(size))
        assert size.min() > 0.0


def test_kernel_reduction_ordering(setup):
    """``min`` keeps the better-resolved direction and so asks for the finest
    mesh; ``max`` discards it. Their order cannot silently swap."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    sizes = {r: R.kernel_element_size(inv, 1.0, reduce=r, report=report,
                                      size_min=1.0, size_max=1e9)
             for r in R.KERNEL_REDUCTIONS}
    assert np.median(sizes["min"]) <= np.median(sizes["geometric"])
    assert np.median(sizes["geometric"]) <= np.median(sizes["max"])


def test_kernel_element_size_rejects_an_unknown_reduction(setup):
    _, _, _, inv = setup
    with pytest.raises(ValueError, match="reduce must be one of"):
        R.kernel_element_size(inv, 1.0, reduce="rms")


def test_kernel_element_size_scales_with_fraction(setup):
    """``fraction`` is elements per kernel width, so it multiplies straight
    through -- there is no hidden normalisation."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    kw = dict(report=report, size_min=1.0, size_max=1e9)
    half = R.kernel_element_size(inv, 1.0, fraction=0.5, **kw)
    quarter = R.kernel_element_size(inv, 1.0, fraction=0.25, **kw)
    np.testing.assert_allclose(quarter, 0.5 * half, rtol=1e-12)


# -- the Gaussian kernel width (Barnhart & Lohman's own metric) ---------------

def _distances(n=240, extent=120e3, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, extent, n))[None, :]


@pytest.mark.parametrize("sigma", [3e3, 12e3, 45e3])
def test_a_planted_gaussian_is_recovered_exactly(sigma):
    """The estimator has a closed-form answer on its own model, so anything
    short of machine precision here is an arithmetic error, not a fit."""
    distance = _distances()
    weights = np.exp(-distance ** 2 / (2 * sigma ** 2))
    got = R.gaussian_kernel_width(weights, distance)
    assert got[0] == pytest.approx(sigma, rel=1e-9)


def test_the_gaussian_width_is_not_biased_by_a_heavy_tail():
    """The whole reason this is fitted rather than linearised.

    A log-space fit is dominated by the tail entries -- the same behaviour that
    makes the *second moment* disagree with the published metric. Against a
    15 km Gaussian carrying a 2% flat floor, the weighted log fits return 28 km
    and 35 km; least squares returns 15.4 km.
    """
    distance = _distances()
    sigma = 15e3
    clean = np.exp(-distance ** 2 / (2 * sigma ** 2))
    floored = clean + 0.02
    tailed = clean + 0.02 * np.exp(-distance / 60e3)

    assert R.gaussian_kernel_width(floored, distance)[0] == pytest.approx(
        15.43e3, rel=0.02)
    assert R.gaussian_kernel_width(tailed, distance)[0] == pytest.approx(
        15.22e3, rel=0.02)

    # The seed on its own is the biased answer, which is why it is only a seed.
    keep = np.ones_like(floored, dtype=bool)
    _, decay = R._log_space_seed(floored, distance ** 2, keep)
    assert 1.0 / np.sqrt(2.0 * decay[0]) > 25e3


def test_a_row_that_does_not_decay_has_no_width():
    """Least squares *succeeds* on a flat row -- the best Gaussian is one so wide
    it is locally constant, of order 1e13 m. Nothing past the largest distance in
    the row is evidence, so a fit out there is reported as no answer rather than
    as a number that swamps a size field."""
    distance = _distances()
    assert np.isnan(R.gaussian_kernel_width(np.ones_like(distance), distance)[0])
    # Too few entries above the floor to fit a curvature at all.
    spike = np.zeros_like(distance)
    spike[0, 0] = 1.0
    assert np.isnan(R.gaussian_kernel_width(spike, distance)[0])


def test_the_gaussian_width_is_smaller_than_the_second_moment(setup):
    """Both widths describe the same kernel and they are not interchangeable.

    Second moments weight by distance squared and so read the tails; the fit
    follows the core. On a real kernel the fit is therefore the smaller of the
    two, and treating them as one number is what the docstrings warn about.
    """
    _, _, _, inversion = setup
    report = R.resolution_report(inversion, 1.0)
    fitted = report["width_gaussian"].values
    along = report["width_along"].values

    assert np.isfinite(fitted).mean() > 0.9, "most rows should fit a width"
    assert np.nanmedian(fitted) < np.median(along)
    assert np.nanmedian(fitted) > 0.0


def test_the_gaussian_width_reaches_the_size_field(setup):
    """``reduce="gaussian"`` is how the published criterion gets into a mesh."""
    _, _, mesh, inversion = setup
    report = R.resolution_report(inversion, 1.0)
    kw = dict(report=report, size_min=1.0, size_max=1e9)
    fitted = R.kernel_element_size(inversion, 1.0, reduce="gaussian", **kw)
    moment = R.kernel_element_size(inversion, 1.0, reduce="min", **kw)

    assert fitted.shape == (mesh.n_elements,)
    assert np.isfinite(fitted).all(), "a NaN row must fall back, not propagate"
    assert not np.allclose(fitted, moment), "it is a different criterion"


def test_gaussian_kernel_width_checks_its_shapes():
    with pytest.raises(ValueError, match="same 2-D shape"):
        R.gaussian_kernel_width(np.ones((4, 4)), np.ones((4, 5)))
    with pytest.raises(ValueError, match="same 2-D shape"):
        R.gaussian_kernel_width(np.ones(4), np.ones(4))


def test_parameter_positions_are_three_dimensional(setup):
    """The fit is against 3-D distance, which is what B&L use -- and on a dipping
    fault that is not the parameter-plane distance."""
    _, _, mesh, inversion = setup
    xyz = R.parameter_positions(inversion)
    assert xyz.shape == (inversion.basis.n_basis, 3)
    assert xyz.shape[0] == R.parameter_params(inversion).shape[0]


# -- generalized cross-validation ---------------------------------------------

def test_gcv_whitens_the_data_exactly_once(setup):
    """`whiten` already applies the diagonal weights when there is no
    covariance model, so whitening a pre-weighted vector squares them and leaves
    the data in a different space from `weighted_design`. The residual must be
    the one `solve` would form."""
    from scipy.linalg import cho_solve

    _, _, _, inversion = setup
    data, chol = R._factorized(inversion, 1.0, 1.0, 3.0, 1.0,
                               ("bottom", "left", "right"), "uniform")
    design = R.weighted_design(inversion)
    observed = inversion.whiten(inversion.d)
    residual = observed - design @ cho_solve(chol, design.T @ observed)

    score = R.generalized_cross_validation(inversion, 1.0)
    assert score["rss"] == pytest.approx(float(residual @ residual), rel=1e-12)
    # Squaring the weights would change the answer, so this is a real check.
    doubled = inversion.whiten(inversion.d * inversion.w)
    assert not np.allclose(observed, doubled)


def test_gcv_charges_for_the_ramp_but_reports_the_slip_rank(setup):
    """Two traces, and confusing them shifts every rank comparison here.

    An unregularised nuisance column costs very nearly a whole degree of freedom,
    so GCV's denominator must count it. But `effective_rank` everywhere else in
    this package -- `resolution_diagonal`, the L-curve, the calibration ladder
    -- is the slip-only trace, and a GCV pick has to be comparable with those.
    """
    _, _, _, inversion = setup
    assert inversion.n_ramp > 0
    score = R.generalized_cross_validation(inversion, 1.0)

    assert score["n_ramp"] == inversion.n_ramp
    assert score["effective_parameters"] > score["effective_rank"]
    # The ramp is unregularised, so it resolves nearly perfectly.
    gap = score["effective_parameters"] - score["effective_rank"]
    assert gap == pytest.approx(inversion.n_ramp, rel=0.1)
    # And the slip-only figure is the one resolution_diagonal reports.
    assert score["effective_rank"] == pytest.approx(
        R.resolution_diagonal(inversion, 1.0).attrs["effective_rank"], rel=1e-9)


def test_gcvs_denominator_is_the_data_resolution_trace(setup):
    """``trace(N)`` computed two ways from the same factorization must agree, or
    GCV is scoring a different problem than ``data_importance`` describes."""
    _, _, _, inversion = setup
    score = R.generalized_cross_validation(inversion, 1.0)
    importance = float(R.data_importance(inversion, 1.0).sum())
    assert score["effective_parameters"] == pytest.approx(importance, rel=1e-10)


def test_the_gcv_curve_turns_over_between_its_ends(setup):
    """The point of GCV: residual falls with rank, the denominator penalty rises,
    so the score has an interior minimum rather than running to an end."""
    _, _, _, inversion = setup
    curve = R.gcv_curve(inversion, np.logspace(-1, 3, 12))

    assert curve.attrs["interior"], "no interior minimum -- widen the sweep"
    assert curve["gcv"].values[0] > curve.attrs["best_gcv"]
    assert curve["gcv"].values[-1] > curve.attrs["best_gcv"]
    # Rank falls monotonically with the weight, which is what makes the curve
    # readable in rank rather than in lambda.
    ranks = curve["effective_rank"].values
    assert (np.diff(ranks) < 0).all()
    # `best_rank` must be the rank at `best_smoothing`, not a nearby one.
    best = int(np.argmin(curve["gcv"].values))
    assert curve.attrs["best_smoothing"] == curve["smoothing"].values[best]
    assert curve.attrs["best_rank"] == ranks[best]


def test_gcv_refuses_a_problem_with_no_degrees_of_freedom_left(setup):
    """The denominator is ``n_obs - trace(N)``; at vanishing damping a mesh with
    more parameters than data drives it to zero and the score is meaningless."""
    _, _, mesh, inversion = setup
    if 2 * inversion.basis.n_basis + inversion.n_ramp <= inversion.d.size:
        pytest.skip("this fixture has more observations than parameters")
    with pytest.raises(ValueError, match="denominator is not positive"):
        R.generalized_cross_validation(inversion, 1e-12)


def test_a_gcv_curve_needs_something_to_compare(setup):
    _, _, _, inversion = setup
    with pytest.raises(ValueError, match="at least two weights"):
        R.gcv_curve(inversion, [1.0])


# -- EPIC: a prior that equalises posterior information -----------------------

@pytest.fixture(scope="module")
def epic_setup(setup):
    """The same problem with a real ``Cd``, which EPIC requires.

    ``sigma_target`` is a posterior standard deviation in metres, and that is
    only a metre when the whitening is by sigma rather than by ``1/sqrt(n)``.
    """
    from slipsolve import SlipInversion
    from slipsolve.covariance import CovarianceModel

    _, _, mesh, inversion = setup
    obs = Observations(inversion.obs.ds.assign_attrs(normalize="none"))
    # Range chosen against the fixture's own 56 km cells: much shorter and
    # `cell_ratio_warning` fires, because a cell-as-a-point covariance overstates
    # the correlation between adjacent samples.
    model = CovarianceModel(sill=0.004 ** 2, range_m=120e3, nugget=0.001 ** 2)
    return SlipInversion(mesh, obs, basis="node", ramp="offset",
                         covariance=model)


def _epic_window(inversion, **kwargs):
    """The (floor, ceiling) of reachable ``sigma_target``, by bisection on the
    two errors ``epic_prior`` raises."""
    lo, hi = 1e-4, 1e6
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        try:
            # A feasibility probe, not a solve -- one evaluation is enough to
            # learn which of the two bounds (if either) it lands outside.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                R.epic_prior(inversion, mid, max_iter=1, **kwargs)
            return mid
        except ValueError as exc:
            lo, hi = (mid, hi) if "below what" in str(exc) else (lo, mid)
    return None


def test_epic_delivers_the_uniform_posterior_it_promises(epic_setup):
    """The defining property, and the only thing that makes it EPIC.

    Tikhonov leaves the posterior variance to the observation geometry; EPIC
    solves for the prior that flattens it. If this does not hold the weights are
    just an expensive way of writing a spatially variable lambda.
    """
    inversion = epic_setup
    target = _epic_window(inversion)
    assert target is not None, "no reachable sigma_target on this fixture"

    weights = R.epic_prior(inversion, target)
    assert weights.shape == (2 * inversion.basis.n_basis,)
    assert (weights > 0).all()

    achieved = R.posterior_covariance(inversion, weights,
                                  kind="posterior")[:2 * inversion.basis.n_basis]
    spread = np.abs(achieved - target).max() / target
    assert spread <= R.EPIC_UNIFORMITY_TOLERANCE, spread
    # And it is genuinely non-uniform prior weight doing it -- a constant would
    # be Tikhonov, which is the thing being improved on.
    assert weights.max() / weights.min() > 1.5


def test_a_scalar_prior_does_not_equalise_the_posterior(epic_setup):
    """The control. Tikhonov at the weight that gets the *median* right still
    leaves the spread the data geometry put there."""
    inversion = epic_setup
    target = _epic_window(inversion)
    weights = R.epic_prior(inversion, target)
    flat = np.full_like(weights, float(np.exp(np.mean(np.log(weights)))))

    n = 2 * inversion.basis.n_basis
    epic = R.posterior_covariance(inversion, weights, kind="posterior")[:n]
    tikhonov = R.posterior_covariance(inversion, flat, kind="posterior")[:n]
    assert epic.max() / epic.min() < tikhonov.max() / tikhonov.min()


def test_the_epic_jacobian_identity_is_exact():
    """Load-bearing for the cost, so pinned where it can be pinned sharply.

    `d[M^-1]_ii/da_j = -([M^-1 H']_ij)^2` gives the whole Jacobian from the
    inverse the residual already formed. With finite differences instead it is
    one factorization per unknown, and EPIC would not be affordable at a
    realistic size. On a well-conditioned problem the identity holds to
    floating point; the fixture test below can only confirm the wiring, because
    a real normal matrix has a condition number around 3e8 and differencing
    variances there has no digits left.
    """
    rng = np.random.default_rng(0)
    n = 30
    design = rng.normal(size=(90, n))
    fixed = design.T @ design + np.eye(n)
    operator = (np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1)
                + np.diag(np.ones(n - 1), -1))
    scale = np.exp(rng.normal(size=n) * 0.3) + 0.5

    def variances(a):
        return np.diag(np.linalg.inv(fixed + operator.T @ (a[:, None] * operator)))

    inverse = np.linalg.inv(fixed + operator.T @ (scale[:, None] * operator))
    analytic = -((inverse @ operator.T) ** 2)

    step = 1e-6
    numeric = np.empty_like(analytic)
    for j in range(n):
        bumped = scale.copy()
        bumped[j] += step
        numeric[:, j] = (variances(bumped) - variances(scale)) / step
    error = np.abs(analytic - numeric).max() / np.abs(numeric).max()
    assert error < 1e-5, error


def test_the_epic_jacobian_is_wired_to_the_right_matrices(epic_setup):
    """The identity above, against the blocks ``epic_prior`` actually builds.

    A coarse step, and do not "improve" it by shrinking. A real normal
    matrix here has a condition number around 3e8, so `inv` loses about eight
    digits and differencing two nearly-equal variances loses the rest: the
    disagreement *grows* as the step shrinks -- 6e-4 at a 1e-2 relative
    step, 7e-3 at 1e-3, 3e-1 at 1e-5, 6e0 at 1e-6. That is rounding, not the
    Jacobian, which the test above pins to 1e-5.
    """
    inversion = epic_setup
    smooth, fixed = R._epic_blocks(inversion, 1.0, 3.0, 1.0,
                                   ("bottom", "left", "right"))
    operator = smooth.toarray()
    n_slip = 2 * inversion.basis.n_basis
    rng = np.random.default_rng(0)
    scale = np.exp(rng.normal(size=operator.shape[0]) * 0.2) * 1e-3

    def variances(a):
        return np.diag(np.linalg.inv(
            fixed + operator.T @ (a[:, None] * operator)))[:n_slip]

    inverse = np.linalg.inv(fixed + operator.T @ (scale[:, None] * operator))
    analytic = -((inverse @ operator.T)[:n_slip] ** 2)

    for j in rng.choice(operator.shape[0], size=6, replace=False):
        step = 1e-2 * scale[j]
        up, down = scale.copy(), scale.copy()
        up[j] += step
        down[j] -= step
        numeric = (variances(up) - variances(down)) / (2 * step)
        big = np.abs(analytic[:, j]) > 1e-3 * np.abs(analytic[:, j]).max()
        assert numeric[big] == pytest.approx(analytic[big, j], rel=0.02)


def test_sigma_target_is_bounded_from_both_sides(epic_setup):
    """Both bounds are structural, not solver failures.

    Above: ``H' diag(a) H`` is positive semidefinite, so a prior can only tighten
    a posterior variance. Below: a curvature operator has a null space -- a slip
    field varying linearly across the fault has zero second derivative -- so that
    component is left to the data however strong the prior.
    """
    inversion = epic_setup
    with pytest.raises(ValueError, match="above what a positive prior can reach"):
        R.epic_prior(inversion, 1e6)
    with pytest.raises(ValueError, match="below what this smoothing operator"):
        R.epic_prior(inversion, 1e-6)
    with pytest.raises(ValueError, match="must be positive"):
        R.epic_prior(inversion, 0.0)


def test_epic_refuses_the_element_basis_by_name(setup):
    """``neighbor_smoothing`` has one row per undirected edge, so diag(Ch) would
    be a field on the edges rather than on the fault -- a different quantity from
    the published one, and not one anybody could plot."""
    _, _, mesh, inversion = setup
    from slipsolve import SlipInversion

    element = SlipInversion(mesh, inversion.obs, basis="element", ramp="offset")
    with pytest.raises(ValueError, match="basis='node'"):
        R.epic_prior(element, 1.0)


def test_epic_weights_drop_straight_into_solve(epic_setup):
    """The whole point of returning weights rather than a covariance: the prior
    reaches the solver through the argument that was already there."""
    inversion = epic_setup
    target = _epic_window(inversion)
    weights = R.epic_prior(inversion, target)
    model = inversion.solve(smoothing=weights, polarity=(-1, 0, 0), max_iter=200)

    assert model.x.shape == (2 * inversion.basis.n_basis + inversion.n_ramp,)
    assert np.isfinite(model.x).all()
    assert np.allclose(model.smoothing, weights)


def test_jri_is_the_residual_plus_twice_the_data_resolution_trace(setup):
    """Barnhart & Lohman's jRi, reduced.

    Their Appendix A form is ``(1/k)||r||^2 - (1/k)tr(C2) + (1/k)tr(Cr)`` with
    ``Cr = M diag(Cd, Cd) M'`` and ``C2 = M [[Cd, Cd], [Cd, Cd]] M'`` for
    ``M = [I, -N]``. Whitened, ``Cd = I`` makes those ``I + N^2`` and
    ``(I - N)^2``, the ``tr(N^2)`` cancels, and what is left is Mallows' Cp.
    This pins the reduction against the pieces it is built from.
    """
    _, _, _, inversion = setup
    score = R.generalized_cross_validation(inversion, 1.0)
    expected = (score["rss"] + 2.0 * score["effective_parameters"]) / score["n_obs"]
    assert score["jri"] == pytest.approx(expected, rel=1e-12)
    # It charges for the ramp, as GCV's denominator does -- the perturbation
    # term is a trace over every fitted parameter, not the slip-only rank.
    assert score["effective_parameters"] >= score["effective_rank"]


def test_the_gcv_curve_reports_jri_and_its_own_minimum(setup):
    """jRi is a different criterion on the same sweep, so it gets its own argmin."""
    _, _, _, inversion = setup
    curve = R.gcv_curve(inversion, np.logspace(-2, 3, 12))
    assert "jri" in curve
    best = int(np.argmin(curve["jri"].values))
    assert curve.attrs["jri_smoothing"] == pytest.approx(
        float(curve["smoothing"].values[best]))
    assert curve.attrs["jri_rank"] == pytest.approx(
        float(curve["effective_rank"].values[best]))


# -- blocked cross-validation ----------------------------------------------
def test_blocked_folds_partition_the_data_into_equal_width_blocks(setup):
    """Every observation in exactly one fold, and the cuts are equal *width*.

    Equal width rather than equal count is the whole design: the validity
    condition is that a block outruns the correlation length, which is a
    distance, so the counts have to come out however the sampling density says.
    """
    _, _, _, inversion = setup
    folds = R.blocked_folds(inversion.obs, n_folds=4)

    assert folds.size == inversion.obs.n
    assert sorted(np.unique(folds.values)) == [0, 1, 2, 3]
    assert sum(folds.attrs["counts"]) == inversion.obs.n

    axis = np.array(folds.attrs["direction"])
    t = (inversion.obs.ds["x"].values * axis[0]
         + inversion.obs.ds["y"].values * axis[1])
    # Contiguous: every fold's projected extent is disjoint from the next one's.
    spans = [(t[folds.values == k].min(), t[folds.values == k].max())
             for k in range(4)]
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert hi <= lo + 1e-9, spans
    width = folds.attrs["block_width"]
    assert width == pytest.approx((t.max() - t.min()) / 4)


def test_blocked_folds_pick_the_long_axis_of_the_sample_cloud(setup):
    """`direction="auto"` must find along-strike, not across.

    An across-strike block holds out one side of the fault, which makes it a
    coverage experiment rather than a noise one -- and on a coastal fault seen
    mostly from one side, one with a foregone conclusion.
    """
    _, _, mesh, inversion = setup
    folds = R.blocked_folds(inversion.obs, n_folds=4)
    axis = np.array(folds.attrs["direction"])

    points = mesh.nodes[:, :2] - mesh.nodes[:, :2].mean(axis=0)
    strike = np.linalg.svd(points, full_matrices=False)[2][0]
    assert abs(float(axis @ strike)) > 0.95, (axis, strike)


def test_held_in_slices_the_ramp_columns_instead_of_rebuilding_them(setup):
    """The trap that would make the whole criterion meaningless.

    `ramp_columns` builds monomials in *centred, span-scaled* coordinates, so
    rebuilding it on a training subset recentres against that subset's extent --
    giving the fit a different set of nuisance columns from the ones the held-out
    rows are predicted with. The column space has to be identical on both sides.
    """
    from slipsolve.regularize import ramp_columns
    from slipsolve.sampling import Observations

    _, _, _, inversion = setup
    linear = inversion.with_ramp("linear")          # gradients, so centring bites
    keep = np.flatnonzero(np.arange(linear.obs.n) % 4 != 0)
    held = R._held_in(linear, keep)

    assert np.array_equal(held.ramp, linear.ramp[keep])
    rebuilt, _ = ramp_columns(Observations(linear.obs.ds.isel(obs=keep)), "linear")
    # Real difference, so the assertion above is a real check and not a tautology.
    assert not np.allclose(held.ramp, rebuilt)


def test_held_in_refactorises_the_covariance_rather_than_slicing_it(setup):
    """``chol(Cd)[keep, keep]`` is not ``chol(Cd[keep, keep])``.

    The parent's Cholesky factors cannot be sliced -- a triangular factor of a
    submatrix is not the submatrix of the factor -- so the training problem has
    to refactorise on the rows it kept, or it is whitened by the wrong matrix.
    """
    from slipsolve.covariance import CovarianceModel

    _, _, _, inversion = setup
    model = CovarianceModel(sill=4e-4, range_m=22e3, nugget=1e-6)
    obs = inversion.obs.ds.copy()
    obs.attrs["normalize"] = "none"
    whitened = SlipInversion(inversion.mesh, Observations(obs), ramp="offset",
                             covariance=model)

    keep = np.flatnonzero(np.arange(whitened.obs.n) % 4 != 0)
    held = R._held_in(whitened, keep)
    name = whitened.obs.tracks[0]
    parent_rows = whitened._track_rows[name]
    sliced = whitened._chol[name][np.ix_(
        np.isin(parent_rows, keep), np.isin(parent_rows, keep))]

    assert held._chol[name].shape == sliced.shape
    assert not np.allclose(held._chol[name], sliced)
    # And it is a genuine factor of the retained block.
    from slipsolve.covariance import observation_covariance
    sub = observation_covariance(held.obs, model, track=name)
    got = held._chol[name] @ held._chol[name].T
    assert np.allclose(got, sub, rtol=1e-6, atol=1e-9)


def test_every_observation_is_predicted_by_a_model_that_never_saw_it(setup):
    """The property the criterion rests on, asserted directly.

    Each fold's training set is the complement of its test set, and the test
    sets tile the data -- so the pooled residual is one out-of-sample prediction
    per observation, never a fitted one.
    """
    _, _, _, inversion = setup
    folds = R.blocked_folds(inversion.obs, n_folds=4)
    problems = R._fold_problems(inversion, folds)

    seen = np.zeros(inversion.obs.n, dtype=int)
    for held_in, test in problems:
        seen[test] += 1
        assert held_in.obs.n == inversion.obs.n - test.size
        # The training rows really are the complement, by value not just count.
        assert np.allclose(np.sort(held_in.d), np.sort(np.delete(inversion.d, test)))
    assert np.array_equal(seen, np.ones(inversion.obs.n, dtype=int))


def test_prediction_error_exceeds_the_fitted_residual(setup):
    """Out-of-sample must be worse than in-sample, or nothing was held out.

    This is the one-line check that the folds are wired the right way round: a
    score that matched ``rms_misfit`` would mean the model had seen its own test
    data, which is precisely the leave-one-out failure being designed against.
    """
    _, _, _, inversion = setup
    score = R.blocked_cross_validation(inversion, 1.0, n_folds=4)
    fitted = inversion.solve(smoothing=1.0).rms_misfit

    assert score["cv_rms"] > fitted
    assert score["n_folds"] == 4 and score["n_obs"] == inversion.obs.n
    assert score["cv_rms_worst"] >= score["cv_rms"] - 1e-12


def test_the_curve_turns_up_at_both_ends(setup):
    """A criterion that only falls has not chosen anything.

    Under-smoothing fits the noise in the training block and predicts the
    held-out one badly; over-smoothing cannot predict either. The minimum
    between them is the choice, and ``interior`` is what says it exists.
    """
    _, _, _, inversion = setup
    curve = R.blocked_cv_curve(inversion, np.logspace(-2, 3, 6), n_folds=4)

    assert curve.attrs["interior"], curve["cv_rms"].values
    best = curve.attrs["best_smoothing"]
    assert curve["cv_rms"].sel(smoothing=best) < curve["cv_rms"].values[0]
    assert curve["cv_rms"].sel(smoothing=best) < curve["cv_rms"].values[-1]
    # Rank is the unit every other criterion here is quoted in.
    assert 0 < curve.attrs["best_rank"] < 2 * inversion.basis.n_basis


def test_a_fold_that_starves_a_tracks_ramp_is_refused(setup):
    """Unconstrained nuisance columns make a held-out block unpredictable.

    Hold out a whole track and its own ramp columns have nothing to fit against,
    so the training solve puts an arbitrary constant there and the score is
    noise. That is the reason the blocks are cut along strike and not by track.
    """
    _, _, _, inversion = setup
    track = np.asarray(inversion.obs.ds["track"].values, dtype=object)
    by_track = (track == inversion.obs.tracks[0]).astype(int)

    with pytest.raises(ValueError, match="unconstrained"):
        R._fold_problems(inversion, by_track)


def test_narrow_blocks_warn_against_the_fitted_range(setup):
    """The block width is the validity condition, so it is checked."""
    from slipsolve.covariance import CovarianceModel
    from slipsolve.sampling import Observations

    _, _, _, inversion = setup
    obs = inversion.obs.ds.copy()
    obs.attrs["normalize"] = "none"
    wide = CovarianceModel(sill=4e-4, range_m=400e3, nugget=1e-6)
    whitened = SlipInversion(inversion.mesh, Observations(obs), ramp="offset",
                             covariance=wide)

    folds = R.blocked_folds(whitened.obs, n_folds=4)
    assert folds.attrs["block_width"] < wide.range_m
    with pytest.warns(RuntimeWarning, match="below the longest fitted"):
        R._warn_narrow_blocks(whitened, folds)


def test_blocked_folds_refuse_a_cut_they_cannot_make(setup):
    _, _, _, inversion = setup
    with pytest.raises(ValueError, match="at least two folds"):
        R.blocked_folds(inversion.obs, n_folds=1)
    with pytest.raises(ValueError, match="non-zero"):
        R.blocked_folds(inversion.obs, n_folds=4, direction=(0.0, 0.0))
    with pytest.raises(ValueError, match="at least two weights"):
        R.blocked_cv_curve(inversion, [1.0])


def test_profiled_score_is_the_projection_onto_the_ramps_complement(setup):
    """The identity that makes `cv_rms_profiled` a criterion and not a fudge.

    Re-fitting the held-out block's ramp columns by least squares and keeping the
    residual is the projection onto the orthogonal complement of the ramp
    subspace. Because that projector does not depend on `smoothing`, it biases
    every rung of a sweep identically and leaves the ranking intact.
    """
    _, _, _, inversion = setup
    assert inversion.n_ramp > 0
    folds = R.blocked_folds(inversion.obs, n_folds=4)
    problems = R._fold_problems(inversion, folds)
    score = R._score_folds(inversion, problems, 1.0)

    # Rebuild it as an explicit orthogonal projection, from a QR of each block's
    # ramp columns -- a different route to the same number.
    total = 0.0
    for held_in, test in problems:
        model = held_in.solve(smoothing=1.0)
        fault = inversion.g[test] @ model.x[:inversion.n_slip]
        r = inversion.d[test] - fault
        q, _ = np.linalg.qr(inversion.ramp[test])
        total += float(np.sum((r - q @ (q.T @ r)) ** 2))
    assert score["cv_rms_profiled"] == pytest.approx(
        np.sqrt(total / inversion.obs.n), rel=1e-8)


def test_profiled_score_is_below_the_raw_one_and_absent_without_a_ramp(setup):
    """It removes a subspace, so it can only shrink the residual -- and it is
    meaningless when there is no ramp to remove, so it is not reported."""
    _, _, _, inversion = setup
    folds = R.blocked_folds(inversion.obs, n_folds=4)

    with_ramp = R._score_folds(inversion, R._fold_problems(inversion, folds), 1.0)
    assert with_ramp["cv_rms_profiled"] < with_ramp["cv_rms"]

    bare = inversion.with_ramp("none")
    assert bare.n_ramp == 0
    plain = R._score_folds(bare, R._fold_problems(bare, folds), 1.0)
    assert "cv_rms_profiled" not in plain


def test_the_curve_scores_the_profiled_minimum_separately(setup):
    """A different criterion on the same sweep gets its own minimum.

    The two can pick ranks far apart, so reading the profiled score off
    `best_smoothing` would quote the wrong weight.
    """
    _, _, _, inversion = setup
    curve = R.blocked_cv_curve(inversion, np.logspace(-1, 2, 5), n_folds=4)

    assert "cv_rms_profiled" in curve
    assert "profiled_smoothing" in curve.attrs
    assert "profiled_rank" in curve.attrs
    best = float(curve["cv_rms_profiled"].sel(
        smoothing=curve.attrs["profiled_smoothing"]))
    assert best == pytest.approx(curve.attrs["profiled_cv_rms"])
    assert best == curve["cv_rms_profiled"].values.min()
