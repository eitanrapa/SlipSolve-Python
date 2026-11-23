"""Pins the segment-by-segment dip sweep.

The sweep rests on one identity -- the design matrix of any combination of dips is
the sum of per-segment blocks -- and on a solver that must agree with
`SlipInversion.solve`. Both are tested against the direct path. Recovery is
the acceptance test, as for the inversion itself: a stepped fault with a
different dip on each segment, forward-modelled independently (element-constant
slip on the joined segment meshes, through a raster and a quadtree), must come
back out of the sweep as the dips it was built with.
"""

import numpy as np
import pytest

from slipsolve import (CovarianceModel, DipSweep, FaultMesh, FaultTrace, HalfSpaceTDE, Observations,
                       SlipInversion, dip_profile_offsets, geometry_key)
from slipsolve.covariance import observation_covariance
from slipsolve.evidence import pinned_nodes
from slipsolve.mesh import _arc_length, _depth_levels
from slipsolve.sweep import segment_breaks, segmented_layout
from slipsolve.trace import dip_offset

import slip_synthetic as ss

_LON = np.array([-68.9, -68.6, -68.3, -68.0, -67.7])
_LAT = np.array([10.50, 10.52, 10.53, 10.55, 10.56])

TRUE_DIPS = (70.0, 110.0)
GRID = (70.0, 90.0, 110.0)
RANK = 30.0
KW = dict(max_depth=15e3, edge_length=10e3, down_dip_levels=5, ramp="offset",
          polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0))
SOLVE = dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=1.0,
             sides=("bottom", "left", "right"), polarity=(-1, 0, 0),
             strike=(-8.0, 8.0), dip=(-3.0, 3.0))


def _stepped_truth(trace, frame, dips):
    """The two-segment fault the sweep should find, built without the sweep.

    Same layout recipe (`segmented_layout` on the same breaks and levels), each
    segment cut out of a uniform-dip mesh by its elements' arc length, and the
    pieces joined -- so the truth's geometry does not come from the code whose
    block assembly is under test.
    """
    edge = KW["edge_length"]
    length = _arc_length(*trace.resample(edge, frame).to_local(frame))[-1]
    breaks = segment_breaks(length, count=len(dips))
    layout = segmented_layout(breaks, _depth_levels(0.0, KW["max_depth"],
                                                    KW["down_dip_levels"], 1.0), edge)
    pieces = []
    for j, dip in enumerate(dips):
        mesh = FaultMesh.from_layout(layout, trace, frame, uniform_dip=dip, edge_length=edge)
        s = mesh.element_params[:, 0]
        keep = np.flatnonzero((s > breaks[j]) & (s < breaks[j + 1]))
        nodes = np.unique(mesh.triangles[keep])
        remap = np.full(mesh.n_nodes, -1)
        remap[nodes] = np.arange(nodes.size)
        pieces.append(FaultMesh(mesh.nodes[nodes], remap[mesh.triangles[keep]],
                                mesh.params[nodes], frame=frame, attrs=dict(mesh.attrs)))
    return FaultMesh.join(pieces), length


def _truth_slip(mesh, length):
    """Right-lateral with a reverse component, smooth along strike, tapering with depth."""
    s, z = mesh.element_params[:, 0], mesh.element_params[:, 1]
    amplitude = (np.exp(-((s - 0.5 * length) / (0.3 * length)) ** 2)
                 * np.clip(1.0 + z / 12e3, 0.0, 1.0))
    return np.concatenate([-2.0 * amplitude, 0.8 * amplitude])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="sweep")
    frame = trace.local_frame()
    truth, length = _stepped_truth(trace, frame, TRUE_DIPS)
    slip = _truth_slip(truth, length)
    obs = Observations.concat([
        Observations.from_los(
            ss.forward_los_stack(truth, slip, trace, frame, geometry=geometry,
                                 spacing=6000.0, half_x=90e3, half_y=50e3,
                                 noise=0.001, seed=seed),
            name=geometry, frame=frame, trace=trace, rms_min=0.002,
            width_min=6000.0, width_max=40000.0, exclude_within=3000.0)
        for seed, geometry in enumerate(("asc", "desc"))])
    sweep = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, **KW)
    return trace, frame, obs, sweep


# -- the pieces --------------------------------------------------------------
def test_segment_breaks():
    np.testing.assert_allclose(segment_breaks(100e3, count=4), [0, 25e3, 50e3, 75e3, 100e3])
    np.testing.assert_allclose(segment_breaks(100e3, size=30e3), np.linspace(0, 100e3, 4))
    np.testing.assert_allclose(segment_breaks(100e3, breaks=[60e3, 20e3]), [0, 20e3, 60e3, 100e3])
    np.testing.assert_allclose(segment_breaks(100e3, breaks=[0.0, 40e3, 100e3]), [0, 40e3, 100e3])
    with pytest.raises(ValueError, match="exactly one"):
        segment_breaks(100e3, count=2, size=10e3)
    with pytest.raises(ValueError, match="on the trace"):
        segment_breaks(100e3, breaks=[120e3])


def test_every_boundary_is_a_node_column_and_no_triangle_straddles_one(setup):
    _, _, _, sweep = setup
    params = sweep.layout.params
    for b in sweep.breaks:
        on = params[params[:, 0] == b]
        assert np.unique(on[:, 1]).size == sweep.depths.size, f"break {b} is not a full column"
    s = params[:, 0][sweep.layout.triangles]
    seg_lo = np.searchsorted(sweep.breaks, s.min(axis=1), side="right") - 1
    seg_hi = np.searchsorted(sweep.breaks, s.max(axis=1), side="left") - 1
    np.testing.assert_array_equal(np.maximum(seg_lo, 0), np.maximum(seg_hi, 0))
    assert sum(e.size for e in sweep.segment_elements) == sweep.reference.n_elements


def test_one_dip_everywhere_is_exactly_the_direct_assembly(setup):
    """The identity the whole sweep rests on, including the shared boundary columns."""
    trace, frame, obs, sweep = setup
    for dip in (70.0, 110.0):
        mesh = FaultMesh.from_layout(sweep.layout, trace, frame, uniform_dip=dip,
                                     edge_length=KW["edge_length"])
        direct = SlipInversion(mesh, obs, basis="node").g
        np.testing.assert_allclose(sweep.design((dip, dip)), direct,
                                   rtol=1e-10, atol=1e-12 * np.abs(direct).max())


def test_the_solver_matches_slipinversion(setup):
    """Same objective, bounds and ramp as `SlipInversion.solve`, same answer."""
    _, _, obs, sweep = setup
    lam = sweep.evaluate((90.0, 90.0), rank=RANK)["lam"]
    result = sweep.evaluate((90.0, 90.0), smoothing=lam)
    model = SlipInversion(sweep.reference, obs, ramp="offset", basis="node").solve(
        smoothing=lam, **SOLVE)
    np.testing.assert_allclose(result["x"], model.x, atol=1e-5 * np.abs(model.x).max())
    assert result["rms"] == pytest.approx(model.rms_misfit, rel=1e-5)


def test_a_rank_target_is_hit(setup):
    _, _, _, sweep = setup
    for dips in ((90.0, 90.0), TRUE_DIPS):
        assert sweep.evaluate(dips, rank=RANK)["rank"] == pytest.approx(RANK, abs=1e-3)


def test_exactly_one_of_rank_or_smoothing(setup):
    _, _, _, sweep = setup
    with pytest.raises(ValueError, match="exactly one"):
        sweep.evaluate((90.0, 90.0))
    with pytest.raises(ValueError, match="one dip per segment"):
        sweep.evaluate((90.0, 90.0, 90.0), rank=RANK)


# -- the search --------------------------------------------------------------
def test_recovers_the_planted_dips(setup):
    """Acceptance: a stepped 70 / 110 fault comes back as 70 / 110."""
    _, _, _, sweep = setup
    result = sweep.run(rank=RANK, method="exhaustive")

    assert len(result.table) == len(GRID) ** 2
    assert result.best == TRUE_DIPS
    profile = result.profiles()
    for j, dip in enumerate(TRUE_DIPS):
        rows = profile[profile.segment == j + 1]
        assert rows.loc[rows.score.idxmin(), "dip"] == dip
    uniform = result.table[(result.table.dip_1 == result.table.dip_2)].score.min()
    assert result.table.score.iloc[0] < 0.9 * uniform, "a stepped truth must beat every uniform dip"


def test_the_coordinate_search_finds_the_exhaustive_optimum(setup):
    _, _, _, sweep = setup
    exhaustive = sweep.run(rank=RANK, method="exhaustive")
    coordinate = sweep.run(rank=RANK, method="coordinate")
    assert coordinate.best == exhaustive.best
    assert len(coordinate.table) <= len(exhaustive.table)


def test_a_fixed_segment_stays_fixed(setup):
    _, _, _, sweep = setup
    result = sweep.run(rank=RANK, fix={0: 90.0})
    assert (result.table.dip_1 == 90.0).all()
    assert len(result.table) == len(GRID)
    assert result.best[0] == 90.0


def test_planted_data_on_the_sweeps_own_geometry_is_found(setup):
    """`predict` + `data=`: the recovery test a real coverage needs, run on synthetic data."""
    _, _, _, sweep = setup
    planted = (110.0, 70.0)
    x = sweep.evaluate(planted, rank=RANK)["x"]
    data = sweep.predict(planted, x)
    result = sweep.run(rank=RANK, data=data, method="exhaustive")
    assert result.best == planted


# -- the evidence ----------------------------------------------------------------
def test_the_evidence_recovers_the_planted_dips(setup):
    """The acceptance test again, scored by the marginal likelihood: no rank to choose."""
    _, _, _, sweep = setup
    result = sweep.run(score="evidence", method="exhaustive")
    assert result.best == TRUE_DIPS
    profile = result.profiles()
    for j, dip in enumerate(TRUE_DIPS):
        rows = profile[profile.segment == j + 1]
        assert rows.loc[rows.score.idxmin(), "dip"] == dip
    assert result.table["delta_log_evidence"].iloc[0] == 0.0
    assert (result.table["delta_log_evidence"] <= 0.0).all()
    assert not result.table["alpha_edge"].any()


def test_the_evidence_takes_no_rank(setup):
    _, _, _, sweep = setup
    with pytest.raises(ValueError, match="neither rank"):
        sweep.run(score="evidence", rank=RANK)
    with pytest.raises(ValueError, match="only to score='evidence'"):
        sweep.run(rank=RANK, covariance="weights")


# -- dips that change with depth -------------------------------------------------
def test_a_profile_of_equal_dips_is_the_planar_dip():
    assert geometry_key((70, 70)) == 70.0
    assert geometry_key([70, 50]) == (70.0, 50.0)
    assert geometry_key(np.float32(80.0)) == 80.0
    for bad in (0.0, 180.0, (70.0,), (70.0, 190.0)):
        with pytest.raises(ValueError):
            geometry_key(bad)


def test_the_profile_offset_integrates_cot_dip():
    depths = np.linspace(0.0, 15e3, 7)
    np.testing.assert_allclose(dip_profile_offsets(depths, (70.0, 70.0)), dip_offset(depths, 70.0),
                               rtol=1e-12, atol=1e-9)
    fine = np.linspace(0.0, 15e3, 300001)
    for dips in ((70.0, 50.0), (60.0, 85.0, 110.0)):
        knots = np.linspace(0.0, 15e3, len(dips))
        theta = np.radians(np.interp(fine, knots, dips))
        cot = np.cos(theta) / np.sin(theta)
        integral = np.concatenate([[0.0], np.cumsum(0.5 * (cot[1:] + cot[:-1]) * np.diff(fine))])
        np.testing.assert_allclose(dip_profile_offsets(depths, dips), np.interp(depths, fine, integral),
                                   rtol=1e-6, atol=1e-3)


def test_a_dip_that_changes_with_depth_is_recovered(setup):
    """A listric segment is built, assembled and found like a planar one."""
    trace, frame, obs, _ = setup
    profile = (70.0, 50.0)
    sweep = DipSweep(trace, obs, frame=frame, segments=2,
                     dips=(70.0, profile, 110.0, (110.0, 130.0)), **KW)
    # Every node sits off the vertical mesh by the profile's offset at its depth.
    bent, upright = sweep._mesh(profile), sweep._mesh(90.0)
    shift = np.hypot(*(bent.nodes[:, :2] - upright.nodes[:, :2]).T)
    np.testing.assert_allclose(shift, np.abs(dip_profile_offsets(-bent.params[:, 1], profile,
                                                                 0.0, sweep.depths.max())),
                               rtol=1e-6, atol=1e-3)
    planted = (profile, 110.0)
    x = sweep.evaluate(planted, rank=RANK)["x"]
    data = sweep.predict(planted, x)
    assert sweep.run(score="evidence", data=data, method="exhaustive").best == planted
    assert sweep.run(rank=RANK, data=data, method="exhaustive").best == planted


# -- the bootstrap -----------------------------------------------------------------
@pytest.fixture(scope="module")
def single(setup):
    """One segment, a 2 degree grid and a smooth planted slip at 70 degrees."""
    trace, frame, obs, _ = setup
    sweep = DipSweep(trace, obs, frame=frame, segments=1, dips=np.arange(62.0, 78.1, 2.0), **KW)
    mesh = sweep._mesh(70.0)
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    amp = np.sin(np.pi * np.clip(s / sweep.length, 0.0, 1.0)) * np.clip(1.0 + z / 15e3, 0.0, 1.0)
    amp[pinned_nodes(sweep.reference, sweep.sides)] = 0.0
    x = np.concatenate([-2.0 * amp, 0.5 * amp, np.zeros(sweep.ramp.shape[1])])
    model = CovarianceModel(sill=(5e-3) ** 2, range_m=15e3, nugget=(1e-3) ** 2)
    factors = {t: np.linalg.cholesky(observation_covariance(obs, model, track=t))
               for t in ("asc", "desc")}
    return sweep, sweep.predict((70.0,), x), model, factors


def _noisy(sweep, signal, factors, seed):
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(signal)
    for t, factor in factors.items():
        rows = sweep.track == t
        noise[rows] = factor @ rng.standard_normal(int(rows.sum()))
    return signal + noise


def test_the_evidence_estimate_is_exact_without_noise(single):
    sweep, signal, model, _ = single
    result = sweep.run(score="evidence", data=signal)
    assert result.best == (70.0,)
    boot = sweep.bootstrap(result, noise=model, draws=5, compare=False)
    assert boot.observed[0] == pytest.approx(70.0, abs=0.01)


def test_the_bootstrap_interval_covers_the_truth_and_the_check_stays_quiet(single):
    """Exact forward model, correlated noise: ~95% coverage and z of order one.

    Over 40 seeds: estimates 70.34 +- 0.78, coverage 0.90, z sd 0.96.
    Twelve seeds keep the test quick; the bounds leave room for that scatter.
    """
    sweep, signal, model, factors = single
    covered, z = [], []
    for seed in range(12):
        result = sweep.run(score="evidence", data=_noisy(sweep, signal, factors, 100 + seed))
        boot = sweep.bootstrap(result, noise=model, draws=20, seed=seed)
        row = boot.intervals().iloc[0]
        assert row.sd > 0.0
        covered.append(row.normal_lo < 70.0 < row.normal_hi)
        z.append(boot.consistency().iloc[0].z)
    assert np.mean(covered) >= 9 / 12
    assert np.all(np.isfinite(z)) and np.max(np.abs(z)) < 4.0


def test_the_bootstrap_fits_its_own_noise_by_reml(single):
    sweep, signal, _, factors = single
    result = sweep.run(score="evidence", data=_noisy(sweep, signal, factors, 7))
    boot = sweep.bootstrap(result, draws=8, seed=1, compare=False)
    assert set(boot.noise) == {"asc", "desc"}
    assert all(isinstance(m, CovarianceModel) for m in boot.noise.values())
    with pytest.raises(ValueError, match="compare=False"):
        boot.consistency()


def test_the_bootstrap_needs_an_evidence_result(setup):
    _, _, _, sweep = setup
    with pytest.raises(ValueError, match="score='evidence'"):
        sweep.bootstrap(sweep.run(rank=RANK, fix={0: 90.0, 1: 90.0}))


# -- continuous profiles -----------------------------------------------------------
def test_optimize_finds_planted_dips_between_the_grid_values(setup):
    """Off-grid planar dips on two segments, found from the grid's best to a hundredth of a degree."""
    _, _, _, sweep = setup
    planted = (74.0, 103.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    fit = sweep.optimize(data=data, bounds=(60.0, 120.0))
    assert fit.converged
    np.testing.assert_allclose(fit.table.dip, planted, atol=0.01)
    assert fit.log_evidence == pytest.approx(sweep.evidence(fit.dips, data=data)["log_evidence"],
                                             rel=1e-12)
    grid = sweep.run(score="evidence", data=data, method="exhaustive")
    assert fit.log_evidence > grid.table.log_evidence.max()


def test_optimize_refines_a_planted_profile_one_knot_at_a_time(setup):
    """Planar first, then two knots on the first segment: the dip that changes with depth comes back."""
    _, _, _, sweep = setup
    planted = ((76.0, 58.0), 100.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    planar = sweep.optimize(data=data, bounds=(40.0, 130.0))
    fit = sweep.optimize(knots=[2, 1], start=planar, data=data, bounds=(40.0, 130.0))
    assert isinstance(fit.dips[0], tuple) and isinstance(fit.dips[1], float)
    np.testing.assert_allclose(fit.table.dip, [76.0, 58.0, 100.0], atol=0.01)
    np.testing.assert_allclose(fit.table.depth_km, [0.0, sweep.depths.max() / 1e3, 0.0])
    assert fit.log_evidence > planar.log_evidence


def test_a_knot_depth_holds_the_deepest_dip_below_it(setup):
    """Knots spread down to `knot_depth`; below it every node keeps the deepest dip."""
    trace, frame, obs, _ = setup
    sweep = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, knot_depth=9e3, **KW)
    profile = (75.0, 55.0)
    bent, upright = sweep._mesh(profile), sweep._mesh(90.0)
    depth = -bent.params[:, 1]
    shift = np.hypot(*(bent.nodes[:, :2] - upright.nodes[:, :2]).T)
    np.testing.assert_allclose(shift, np.abs(dip_profile_offsets(depth, profile, 0.0, 9e3)),
                               rtol=1e-6, atol=1e-3)
    below = depth > 9e3
    assert below.any()
    at_knot = dip_profile_offsets(np.array([9e3]), profile, 0.0, 9e3)[0]
    np.testing.assert_allclose(shift[below], at_knot + (depth[below] - 9e3) / np.tan(np.radians(55.0)),
                               rtol=1e-6, atol=1e-3)
    planted = (profile, 100.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    fit = sweep.optimize(knots=[2, 1], start=(70.0, 100.0), data=data, bounds=(40.0, 130.0))
    np.testing.assert_allclose(fit.table.dip, [75.0, 55.0, 100.0], atol=0.01)
    np.testing.assert_allclose(fit.table.depth_km, [0.0, 9.0, 0.0])
    for bad in (0.0, 2 * KW["max_depth"]):
        with pytest.raises(ValueError, match="knot_depth"):
            DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, knot_depth=bad, **KW)


def test_optimize_holds_fixed_segments_and_checks_its_arguments(setup):
    _, _, _, sweep = setup
    planted = (74.0, 103.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    fit = sweep.optimize(fix={0: 74.0}, start=(74.0, 110.0), data=data, bounds=(60.0, 120.0))
    assert fit.dips[0] == 74.0 and list(fit.table.segment) == [2]
    assert fit.table.dip.iloc[0] == pytest.approx(103.0, abs=0.01)
    with pytest.warns(UserWarning, match="on a bound"):
        pinned = sweep.optimize(start=(74.0, 100.0), data=data, bounds=(60.0, 100.0))
    assert pinned.table.on_bound.tolist() == [False, True]
    for bad in (dict(knots=0), dict(knots=[1, 1, 1]), dict(bounds=(0.0, 90.0)),
                dict(fix={0: 70.0, 1: 90.0}), dict(data=data[:-1]), dict(step=2.0)):
        with pytest.raises(ValueError):
            sweep.optimize(start=(70.0, 110.0), **bad)


def test_the_profile_bootstrap_flags_a_knot_on_its_bound(setup):
    """A knot held on its bound has no spread to report: no sd, and a one-sided interval."""
    _, _, _, sweep = setup
    planted = (74.0, 103.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    with pytest.warns(UserWarning, match="on a bound"):
        fit = sweep.optimize(start=(74.0, 100.0), data=data, bounds=(60.0, 100.0))
    model = CovarianceModel(sill=(5e-3) ** 2, range_m=15e3, nugget=(1e-3) ** 2)
    with pytest.warns(UserWarning, match="one-sided"):
        boot = sweep.bootstrap(fit, noise=model, draws=20, compare=False)
    free, held = boot.intervals().itertuples(index=False)
    assert not free.on_bound and free.sd > 0.0 and np.isfinite([free.normal_lo, free.basic_hi]).all()
    assert held.on_bound and held.pinned > 0.5
    assert np.isnan([held.sd, held.normal_lo, held.normal_hi, held.basic_hi]).all()
    assert held.basic_lo <= 100.0 and held.estimate == pytest.approx(100.0)
    assert "(on a bound)" in repr(boot)


def test_optimize_continuity_closes_the_tear_between_segments(setup):
    """The tear is the two segments' offset difference; `continuity` trades evidence for closing it."""
    _, _, _, sweep = setup
    planted = (74.0, 103.0)
    expected = dip_offset(sweep.depths, 74.0) - dip_offset(sweep.depths, 103.0)
    np.testing.assert_allclose(sweep.tears(planted), expected[None, :], rtol=1e-12)
    assert np.abs(sweep.tears((80.0, 80.0))).max() == 0.0
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    loose = sweep.optimize(data=data, bounds=(60.0, 120.0))
    assert loose.penalty == 0.0 and loose.continuity is None
    assert loose.tears.rms_m.iloc[0] == pytest.approx(np.sqrt(np.mean(expected ** 2)), rel=0.01)
    tied = sweep.optimize(data=data, bounds=(60.0, 120.0), continuity=20.0)
    assert tied.continuity == 20.0 and "penalty" in tied.history
    assert tied.tears.rms_m.iloc[0] < 0.1 * loose.tears.rms_m.iloc[0]
    assert tied.log_evidence < loose.log_evidence
    # it maximised the penalised objective: the loose optimum scores worse under the same prior
    loose_cost = 0.5 * (loose.tears.rms_m.iloc[0] / 20.0) ** 2
    assert tied.log_evidence - tied.penalty > loose.log_evidence - loose_cost
    boot = sweep.bootstrap(tied, noise=CovarianceModel(sill=(5e-3) ** 2, range_m=15e3, nugget=(1e-3) ** 2),
                           draws=10, compare=False)
    assert np.isfinite(boot.intervals().sd).all()
    with pytest.raises(ValueError, match="continuity"):
        sweep.optimize(data=data, bounds=(60.0, 120.0), continuity=0.0)


def test_the_profile_bootstrap_covers_the_truth_like_the_grid_bootstrap(single):
    """Exact forward model, correlated noise: the continuous fit and its one-step bootstrap.

    Over 40 seeds: the continuous estimate is the grid's splined one to 0.023 deg rms (0.094
    worst), the bootstrap sd 0.74 against the grid bootstrap's 0.80 and an actual spread of
    0.85, coverage 0.90 for both, consistency z sd 1.08. Twelve seeds keep the test quick.
    """
    from slipsolve.sweep import _refine

    sweep, signal, model, factors = single
    covered, z = [], []
    for seed in range(12):
        data = _noisy(sweep, signal, factors, 100 + seed)
        grid = sweep.run(score="evidence", data=data)
        fit = sweep.optimize(start=grid, data=data)
        splined = _refine(list(grid.table.dip_1), grid.table.log_evidence.to_numpy())[0]
        assert fit.table.dip.iloc[0] == pytest.approx(splined, abs=0.1)
        boot = sweep.bootstrap(fit, noise=model, draws=20, seed=seed)
        row = boot.intervals().iloc[0]
        assert row.sd > 0.0 and row.estimate == pytest.approx(fit.table.dip.iloc[0], abs=0.01)
        covered.append(row.normal_lo < 70.0 < row.normal_hi)
        z.append(boot.consistency().iloc[0].z)
    assert np.mean(covered) >= 9 / 12
    assert np.all(np.isfinite(z)) and np.max(np.abs(z)) < 4.0


def test_smoothed_normals_embed_the_layout_as_a_production_sheet_does(setup):
    """`normal_smoothing` moves each node as `Sheet` does, so the sweep scores the surface a run will mesh."""
    from slipsolve.production import Sheet

    trace, frame, obs, sweep = setup
    smoothed = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, normal_smoothing=10e3, **KW)
    knots = np.linspace(0.0, smoothed.knot_depth, 2)
    cases = [(70.0, dict(dip=70.0), 1e-6), (110.0, dict(dip=110.0), 1e-6),
             # a profile, to the sheet's own quadrature of cot(dip)
             ((75.0, 55.0), dict(profile=lambda s, z: np.interp(z, knots, (75.0, 55.0))), 0.1)]
    for geometry, shape, tol in cases:
        sheet = Sheet("s", trace, frame, depth=KW["max_depth"], normal_smoothing=10e3,
                      spacing=KW["edge_length"], **shape)
        reference = sheet.mesh(smoothed.layout).nodes
        for j in range(smoothed.n_segments):
            nodes = smoothed.segment_nodes[j]
            assert np.abs(smoothed.segment_mesh(j, geometry).nodes - reference[nodes]).max() < tol
    # a different embedding from the local normal's on this bending trace, and the blocks still sum
    assert np.abs(smoothed.segment_mesh(0, 70.0).nodes - sweep.segment_mesh(0, 70.0).nodes).max() > 10.0
    direct = SlipInversion(smoothed._mesh(70.0), obs, basis="node").g
    np.testing.assert_allclose(smoothed.design((70.0, 70.0)), direct,
                               rtol=1e-10, atol=1e-12 * np.abs(direct).max())
    with pytest.raises(ValueError, match="normal_smoothing"):
        DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, normal_smoothing=-1.0, **KW)


def test_a_fit_hands_its_profile_to_a_production_sheet(setup):
    """`ProfileFit.profile` is each segment's dip blended across the breaks; `sheet` builds the run's sheet on it."""
    _, _, _, sweep = setup
    planted = ((76.0, 58.0), 100.0)
    data = sweep.predict(planted, sweep.evaluate(planted, rank=RANK)["x"])
    fit = sweep.optimize(knots=[2, 1], start=planted, data=data, bounds=(40.0, 130.0))
    b, z = sweep.breaks[1], sweep.depths
    step = fit.profile(blend=0.0)
    left, right = step(np.full(z.shape, b - 1.0), z), step(np.full(z.shape, b + 1.0), z)
    np.testing.assert_allclose(left, np.interp(z, [0.0, sweep.knot_depth], fit.dips[0]))
    np.testing.assert_allclose(right, fit.dips[1])
    blended = fit.profile(blend=5e3)
    np.testing.assert_allclose(blended(np.full(z.shape, b), z), 0.5 * (left + right))
    np.testing.assert_allclose(blended(b - 5e3, z), left)
    np.testing.assert_allclose(blended(b + 5e3, z), right)
    # on the sweep's own resampling of the trace, each segment's surface away from the break
    sheet = fit.sheet("main", blend=0.0, spacing=KW["edge_length"])
    assert sheet.depth == sweep.depths.max()
    mesh = sheet.mesh(sweep.layout)
    for j in range(sweep.n_segments):
        nodes = sweep.segment_nodes[j]
        far = np.abs(sweep.layout.params[nodes, 0] - b) > KW["edge_length"]
        assert np.abs(mesh.nodes[nodes][far] - sweep.segment_mesh(j, fit.dips[j]).nodes[far]).max() < 0.5
    # on a finer one the arc lengths differ, and the break is carried by position: to the nearest point
    fine = fit.sheet("main", blend=0.0)
    assert fine.length > sweep.length
    s = np.linspace(b - 2e3, b + 2e3, 4001)
    dips = fine.profile(s, np.zeros_like(s))
    switch = s[np.argmax(dips != dips[0])]
    point = np.interp(b, sweep._s, sweep._rx), np.interp(b, sweep._s, sweep._ry)
    track = fine.position(s, 0.0)
    gap = np.hypot(track[0] - point[0], track[1] - point[1])
    assert abs(switch - s[np.argmin(gap)]) <= 2.0
    with pytest.raises(ValueError, match="breaks"):
        fit.profile(breaks=[b, b + 1.0])


def test_blocks_from_workers_and_from_the_cache_are_the_serial_ones(setup, tmp_path):
    """`workers` splits each block's rows over processes; `cache` keeps the blocks, keyed by mesh, samples and engine."""
    import pickle

    trace, frame, obs, sweep = setup
    combo = (70.0, 90.0)
    serial = sweep.design(combo)
    pooled = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, workers=2, cache=tmp_path, **KW)
    try:
        np.testing.assert_allclose(pooled.design(combo), serial, rtol=1e-12, atol=0.0)
        pickle.dumps(pooled)
    finally:
        pooled.close()
    assert len(list(tmp_path.glob("g_*.npy"))) == 2
    # a block in the cache is the one used: doubled on disk, it comes back doubled
    again = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, cache=tmp_path, **KW)
    [path] = tmp_path.glob(f"g_{again.segment_mesh(0, 70.0).digest()[:12]}_*.npy")
    np.save(path, 2.0 * np.load(path))
    np.testing.assert_array_equal(again._segment_block(0, 70.0), 2.0 * sweep._segment_block(0, 70.0))
    # another engine reads none of them
    other = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, cache=tmp_path, engine=HalfSpaceTDE(0.3), **KW)
    other.design(combo)
    assert len(list(tmp_path.glob("g_*.npy"))) == 4
    with pytest.raises(ValueError, match="workers"):
        DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, workers=0, **KW)


def test_the_combination_count_does_not_wrap():
    from slipsolve.sweep import _combinations

    assert _combinations([[60.0, 75.0, 90.0]] * 40) == 3 ** 40 > 2 ** 63
    assert _combinations([[1, 2, 3, 4]] * 32) == 4 ** 32
