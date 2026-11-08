"""Bounded Monte Carlo and planted recovery.

The oracle for the spread is the closed form: with no limit active the solve is linear, so
the copies' spread must match the propagated covariance. The oracle for the plumbing is that a
copy with no noise is the model itself, in every mode.
"""

import copy

import numpy as np
import pytest

from slipsolve import (CovarianceModel, EulerPole, FaultMesh, FaultTrace, HalfSpaceTDE, InterseismicInversion,
                       Observations, SlipInversion, TimeBasis, TimeDependentInversion, TrackSeries, checkerboard,
                       gnss_observations, monte_carlo, recovery, side_labels)
from slipsolve.calibrate import recovery_score
from slipsolve.interseismic import _axes_at
from slipsolve.regularize import laplace_beltrami, scale_smoothing
from slipsolve.inversion import _boundary_rows
from slipsolve.uncertainty import _sign_blocks

_LOOK = np.array([-0.56, -0.10, 0.82])
_LOOK = _LOOK / np.linalg.norm(_LOOK)
_SIGMA = 0.004


@pytest.fixture(scope="module")
def plain():
    """A small coseismic problem: two looks, a patch, known white noise."""
    trace = FaultTrace(np.array([-68.1, -67.4]), np.array([10.55, 10.60]), name="toy")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=7e3)
    rng = np.random.default_rng(3)
    x = rng.uniform(-70e3, 70e3, 300)
    y = rng.uniform(-50e3, 50e3, 300)
    keep = trace.distance(x, y, frame) > 3e3
    x, y = x[keep], y[keep]
    looks = {"A": _LOOK, "D": np.array([0.60, -0.10, 0.79]) / np.linalg.norm([0.60, -0.10, 0.79])}
    parts = [Observations.from_points(x, y, np.zeros(x.size), v, _SIGMA, name=k, frame=frame, epsg=None)
             for k, v in looks.items()]
    probe = SlipInversion(mesh, Observations.concat(parts, normalize="none"), basis="node")
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    amp = -1.0 * np.exp(-((s - s.mean()) / 18e3) ** 2) * np.exp(-((z + 6e3) / 5e3) ** 2)
    truth = np.r_[amp, np.zeros(mesh.n_nodes)]
    values = probe.g @ truth + rng.normal(0.0, _SIGMA, probe.obs.n)
    obs = Observations(probe.obs.ds.assign(los=("obs", values)))
    inv = SlipInversion(mesh, obs, basis="node", ramp="offset")
    return dict(mesh=mesh, frame=frame, inv=inv, truth=truth, amp=amp, n=mesh.n_nodes)


_SOLVE = dict(smoothing=3e3, method="active_set", boundary_ratio=3e3)


def test_a_noiseless_copy_is_the_model(plain):
    model = plain["inv"].solve(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0), **_SOLVE)
    mc = monte_carlo(model, n=2, noise=lambda rng: np.zeros(plain["inv"].obs.n))
    assert np.abs(mc.x - model.x).max() < 1e-7
    assert mc.failed == 0 and np.abs(mc.bias).max() < 1e-7


def test_spread_matches_the_closed_form_with_no_limit_active(plain):
    """Wide limits make the solve linear: M^-1 A'A M^-1 is the variance, A whitened."""
    inv = plain["inv"]
    model = inv.solve(strike=(-50.0, 50.0), dip=(-50.0, 50.0), **_SOLVE)
    mc = monte_carlo(model, n=400, noise="weights", seed=1)
    A = np.hstack([inv.g, inv.ramp]) * inv.w[:, None]
    smooth = scale_smoothing(np.hstack([laplace_beltrami(inv.mesh, 1.0, 3.0).toarray(), np.zeros((laplace_beltrami(inv.mesh, 1.0, 3.0).shape[0], inv.n_ramp))]), _SOLVE["smoothing"])
    edge = _boundary_rows(inv.basis, ("bottom", "left", "right"), _SOLVE["boundary_ratio"]).toarray()
    edge = np.hstack([edge, np.zeros((edge.shape[0], inv.n_ramp))])
    M = A.T @ A + smooth.T @ smooth + edge.T @ edge
    inverse = np.linalg.inv(M)
    sigma = np.sqrt(np.diag(inverse @ (A.T @ A) @ inverse))
    well = sigma > 0.2 * sigma.max()
    ratio = mc.std[well] / sigma[well]
    assert abs(np.median(ratio) - 1.0) < 0.05, np.median(ratio)
    assert ratio.min() > 0.8 and ratio.max() < 1.2


def test_limits_hold_in_every_copy_and_clip_the_mean(plain):
    inv = plain["inv"]
    model = inv.solve(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0), **_SOLVE)
    mc = monte_carlo(model, n=60, noise="white", seed=2)
    n = plain["n"]
    assert mc.x[:, :n].max() <= 1e-9 and mc.x[:, :n].min() >= -5.0 - 1e-9
    # slip near zero can only move one way, so the copies' mean moves off the limit
    near_zero = model.strike_slip > -1e-6
    assert near_zero.any() and np.all(mc.mean[:n][near_zero] <= 1e-9)
    moments = mc.apply(lambda m: m.moment())
    assert moments.shape == (len(mc),) and moments.std() > 0.0
    assert abs(moments.mean() / model.moment() - 1.0) < 0.15


def test_noise_kinds(plain):
    inv = plain["inv"]
    model = inv.solve(strike=(-50.0, 50.0), dip=(-50.0, 50.0), **_SOLVE)
    rms = np.sqrt(np.mean(model.residual ** 2))
    rng = np.random.default_rng(5)
    from slipsolve.uncertainty import _noise_drawer

    blocks = _noise_drawer(model, "blocks", 30e3)(rng)
    assert np.allclose(np.abs(blocks), np.abs(model.residual)), "a block's draw is its residual, signed"
    assert abs(np.sqrt(np.mean(_noise_drawer(model, "white", None)(rng) ** 2)) / rms - 1.0) < 0.15
    assert abs(np.std(_noise_drawer(model, "weights", None)(rng)) / _SIGMA - 1.0) < 0.15
    drawn = _noise_drawer(model, CovarianceModel(sill=_SIGMA ** 2, range_m=20e3, nugget=0.0), None)(rng)
    assert drawn.shape == model.residual.shape
    with pytest.raises(ValueError, match="noise must be"):
        monte_carlo(model, n=1, noise="nonsense")


def test_sign_blocks_share_a_sign_inside_a_block(plain):
    ds = plain["inv"].obs.ds
    signs = _sign_blocks(ds, 40e3, np.random.default_rng(0))
    x, y = ds["x"].values, ds["y"].values
    track = np.asarray(ds["track"].values, dtype=str)
    key = np.floor(x / 40e3) * 1000 + np.floor(y / 40e3)
    for name in np.unique(track):
        for cell in np.unique(key[track == name]):
            assert np.ptp(signs[(track == name) & (key == cell)]) == 0.0
    assert set(np.unique(signs)) == {-1.0, 1.0}


def test_noiseless_recovery_is_a_direct_solve_of_the_planted_data(plain):
    """The oracle: plant, make the data by hand, solve it the ordinary way; recovery must return that."""
    inv = plain["inv"]
    limits = dict(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0))
    model = inv.solve(**limits, **_SOLVE)
    truth = np.r_[plain["truth"], model.ramp]
    back = recovery(model, truth, n=2, noise=lambda rng: np.zeros(inv.obs.n))
    direct = copy.copy(inv)
    direct.d = inv.g @ plain["truth"] + inv.ramp @ model.ramp
    assert np.abs(back.x[0] - direct.solve(**limits, **_SOLVE).x).max() < 1e-7
    assert np.array_equal(back.centre, truth)


def test_recovery_of_a_planted_patch(plain):
    inv = plain["inv"]
    model = inv.solve(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0), **{**_SOLVE, "smoothing": 1e3})
    truth = np.r_[plain["truth"], model.ramp]
    back = recovery(model, truth, n=8, noise="weights", seed=4)
    n = plain["n"]
    score = recovery_score(back.mean[:n], plain["amp"])
    assert score["correlation"] > 0.7 and 0.6 < score["amplitude_ratio"] < 1.2
    # the peak comes back low, and the averaging over noise does not cure it: that is smoothing's bias
    assert back.mean[:n].min() > plain["amp"].min() + 3.0 * back.std[:n].max() / np.sqrt(len(back))
    with pytest.raises(ValueError, match="truth has"):
        recovery(model, truth[:-1])


def test_recovery_at_another_smoothing_is_smoother(plain):
    inv = plain["inv"]
    model = inv.solve(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0), **_SOLVE)
    truth = np.r_[plain["truth"], model.ramp]
    n = plain["n"]
    sharp = recovery(model, truth, n=4, noise="weights", seed=5).mean[:n]
    blunt = recovery(model, truth, n=4, noise="weights", seed=5, smoothing=3e5).mean[:n]
    assert np.abs(blunt).max() < np.abs(sharp).max()


def test_checkerboard(plain):
    mesh = plain["mesh"]
    field = checkerboard(mesh, 20e3, 10e3, low=0.1, high=0.9)
    assert field.shape == (mesh.n_nodes,) and set(np.unique(field)) == {0.1, 0.9}
    assert checkerboard(mesh, 20e3, 10e3, nodes=False).shape == (mesh.n_elements,)


# -- the modes ----------------------------------------------------------------------------------
def test_a_noiseless_copy_is_the_model_in_time(plain):
    mesh, frame = plain["mesh"], plain["frame"]
    rng = np.random.default_rng(7)
    x = rng.uniform(-70e3, 70e3, 200)
    y = rng.uniform(-50e3, 50e3, 200)
    keep = FaultTrace(np.array([-68.1, -67.4]), np.array([10.55, 10.60])).distance(x, y, frame) > 3e3
    x, y = x[keep], y[keep]
    probe = SlipInversion(mesh, Observations.from_points(x, y, np.zeros(x.size), _LOOK, 1.0, name="A", frame=frame, epsg=None),
                          basis="node")
    basis = TimeBasis.epochs([0.0, 30.0, 90.0])
    amp = plain["truth"]
    t0, t1 = np.full(5, 1.0), np.r_[10.0, 25.0, 45.0, 60.0, 85.0]
    values = np.array([basis.weights([a], [b]).sum() * (probe.g @ amp) for a, b in zip(t0, t1)])
    values = values + rng.normal(0.0, 0.002, values.shape)
    series = TrackSeries(x, y, _LOOK, values, t0, t1, 0.002, name="A", ramp="offset", frame=frame)
    inv = TimeDependentInversion(mesh, series, basis, frame_looks=False)
    model = inv.solve(smoothing=1e3, polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0))
    mc = monte_carlo(model, n=2, noise=lambda r: np.zeros(inv.obs.n))
    assert np.abs(mc.x - model.x).max() < 1e-7
    noisy = monte_carlo(model, n=8, noise="white", seed=1)
    windows = noisy.apply(lambda m: m.slip_between(1.0, 85.0).moment())
    assert windows.std() > 0.0
    assert noisy.apply(lambda m: m.coefficients).shape == (8, 2, 2, mesh.n_nodes)


@pytest.fixture(scope="module")
def inter():
    trace = FaultTrace(np.array([-121.0, -119.0]), np.array([36.0, 36.05]), name="f")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=7e3)
    rng = np.random.default_rng(11)
    x = rng.uniform(-90e3, 90e3, 240)
    y = rng.uniform(-70e3, 70e3, 240)
    keep = trace.distance(x, y, frame) > 3e3
    x, y = x[keep][:80], y[keep][:80]
    lon, lat = frame.to_lonlat(x, y)
    obs = gnss_observations(lon, lat, np.zeros(80), np.zeros(80), np.zeros(80), 1e-3, 1e-3, 3e-3, frame, components="en")
    obs = Observations.concat([obs], normalize="none")
    blocks = side_labels(obs, trace, "north", "south")
    south = EulerPole.from_pole(-110.0, 60.0, 0.5)
    north = EulerPole(south.omega + _axes_at([frame.origin_lon, frame.origin_lat])[0] * 0.03)
    poles = {"north": north, "south": south}
    probe = InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"))
    v_s, v_d = probe.long_term_slip(poles)
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    coupling = np.clip((13e3 + z) / 5e3, 0.0, 1.0) * np.exp(-((s - s.mean()) / 70e3) ** 2)
    back_slip = np.r_[-coupling * v_s, -coupling * v_d]
    olon, olat = frame.to_lonlat(obs.ds["x"].values, obs.ds["y"].values)
    rigid = np.zeros(obs.n)
    for name, pole in poles.items():
        rows = blocks == name
        e, n, _ = pole.velocity(olon[rows], olat[rows])
        rigid[rows] = e * obs.ds["look_e"].values[rows] + n * obs.ds["look_n"].values[rows]
    values = rigid + probe.g @ back_slip + rng.normal(size=obs.n) / obs.ds["weight"].values
    obs = Observations(obs.ds.assign(los=("obs", values)))
    inv = InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"))
    return dict(inv=inv, coupling=coupling, mesh=mesh)


def test_interseismic_copies_resettle_their_rate(inter):
    model = inter["inv"].solve(smoothing=3e5)
    assert model.options["rate_fixed"] is False
    still = monte_carlo(model, n=2, noise=lambda r: np.zeros(inter["inv"].obs.n))
    assert np.abs(still.x - model.x).max() < 1e-6
    mc = monte_carlo(model, n=24, noise="white", seed=3)
    coupling = mc.apply(lambda m: m.coupling)
    assert coupling.shape == (24, inter["mesh"].n_nodes)
    assert np.nanmin(coupling) > -1e-6 and np.nanmax(coupling) < 1.0 + 1e-6, "the bound moves with each copy's rate"
    rates = mc.apply(lambda m: m.long_term_slip[0].mean())
    assert rates.std() > 0.0 and np.abs(rates.mean() - model.long_term_slip[0].mean()) < 2e-3
    held = monte_carlo(model, n=6, noise="white", seed=3, resettle=False)
    assert np.ptp(held.apply(lambda m: m.long_term_slip[0].mean())) == 0.0


def test_fixed_rate_models_do_not_resettle(inter):
    inv = inter["inv"]
    settled = inv.solve(smoothing=3e5)
    fixed = inv.solve(smoothing=3e5, rate=settled.long_term_slip)
    assert fixed.options["rate_fixed"] is True
    mc = monte_carlo(fixed, n=4, noise="white", seed=4)
    assert np.ptp(mc.apply(lambda m: m.long_term_slip[0].mean())) == 0.0


def test_planted_coupling_comes_back(inter):
    inv, mesh = inter["inv"], inter["mesh"]
    model = inv.solve(smoothing=3e5)
    v_s, v_d = model.long_term_slip
    planted = np.clip((8e3 + mesh.params[:, 1]) / 4e3, 0.0, 1.0)
    truth = np.r_[-planted * v_s, -planted * v_d, model.ramp]
    back = recovery(model, truth, n=6, noise="white", seed=5)
    got = back.apply(lambda m: m.coupling).mean(axis=0)
    assert recovery_score(got, planted)["correlation"] > 0.6
    assert np.nanmin(got) >= -1e-6 and np.nanmax(got) <= 1.0 + 1e-6
