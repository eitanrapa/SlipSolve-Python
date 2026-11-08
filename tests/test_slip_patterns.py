"""Signals of known shape with a free amplitude, beside the slip."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from test_slip_temporal import _LAM, _SOLVE, _poly, _series, setup  # noqa: F401  (a fixture)

from slipsolve import (FaultMesh, FaultTrace, InterseismicInversion, Observations, Pattern, TimeBasis, TimeDependentInversion,
                       TimeDependentModel, blocked_cv_curve, gnss_observations, monte_carlo, poroelastic_pattern, separate_history,
                       side_labels)
from slipsolve.basis import make_basis
from slipsolve.inversion import _boundary_rows
from slipsolve.regularize import laplace_beltrami, scale_smoothing, slip_bounds


def _hump(width=60e3):
    """A broad up-and-east signal with no fault in it, one metre at the origin."""
    def enu(x, y):
        shape = np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / width ** 2)
        return 0.3 * shape, 0.0 * shape, shape

    return enu


def _with_pattern(setup, track, t0, t1, field, pattern, amplitude, **kw):  # noqa: F811
    s = _series(setup, track, t0, t1, field, **kw)
    weights = pattern.time.weights(s.t0, s.t1)[:, 0]
    los = pattern.values(s.x, s.y, s.look[:, 0], s.look[:, 1], s.look[:, 2])[:, 0]
    s.values = s.values + amplitude * np.outer(weights, los)
    return s


def test_tabulated_time_functions():
    basis = TimeBasis.tabulated([0.0, 10.0, 30.0], np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 3.0]]))
    assert len(basis) == 2 and basis.names == ["table of 3", "table of 3"]
    assert np.allclose(basis.cumulative([-5.0, 5.0, 20.0, 99.0]), [[0.0, 0.0], [0.5, 1.0], [1.25, 2.5], [1.5, 3.0]])
    again = TimeBasis.from_dict(basis.to_dict())
    assert np.array_equal(again.cumulative([3.0]), basis.cumulative([3.0]))
    with pytest.raises(ValueError, match="increasing"):
        TimeBasis.tabulated([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="rows"):
        TimeBasis.tabulated([0.0, 1.0, 2.0], [1.0, 2.0])


def test_pattern_values_are_checked():
    x, y = np.zeros(3), np.zeros(3)
    look = (np.ones(3), np.zeros(3), np.zeros(3))
    two = Pattern("two", lambda *a: np.ones((3, 2)), TimeBasis.exp_decay(10.0))
    with pytest.raises(ValueError, match="2 columns"):
        two.values(x, y, *look)
    bad = Pattern("bad", lambda *a: np.array([1.0, np.nan, 1.0]), TimeBasis.exp_decay(10.0))
    with pytest.raises(ValueError, match="not finite"):
        bad.values(x, y, *look)
    enu = Pattern.from_enu("enu", lambda x, y: (np.ones(3), 2.0 * np.ones(3), 3.0 * np.ones(3)))
    assert np.allclose(enu.values(x, y, np.full(3, 0.5), np.full(3, 0.25), np.full(3, 0.1)), 0.5 + 0.5 + 0.3)


def test_a_pattern_matches_the_whole_system(setup):  # noqa: F811
    """A redundant network, two slip fields, a pattern on its own time function: the reduction stays exact."""
    n = setup["n"]
    m1 = np.r_[-0.4 * setup["patch"](-15e3, 25e3, 7e3), np.zeros(n)]
    m2 = np.r_[-0.3 * setup["patch"](15e3, 25e3, 11e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 40.0, 100.0])
    field = lambda a, b: basis.weights([a], [b])[0] @ np.array([m1, m2])
    pattern = Pattern.from_enu("hump", _hump(), TimeBasis.exp_decay(60.0))
    dates = np.array([4.0, 16.0, 28.0, 40.0, 52.0, 76.0, 100.0])
    t0, t1 = [], []
    for i in range(dates.size):
        for j in range(i + 1, min(i + 3, dates.size)):
            t0.append(dates[i])
            t1.append(dates[j])
    t0, t1 = np.array(t0), np.array(t1)
    series = [_with_pattern(setup, "A", t0, t1, field, pattern, 0.05, seed=3),
              _with_pattern(setup, "D", t0 + 1.0, t1 + 1.0, field, pattern, 0.05, seed=4)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False, patterns=[pattern])
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    assert inv.obs.n == 3 * sum(s.n_cells for s in series), "two fields and a pattern: three virtual pairs a track"

    K = len(basis)
    blocks = []
    for s in series:
        T = basis.weights(s.t0, s.t1)
        P = pattern.time.weights(s.t0, s.t1)[:, 0]
        values, vectors = np.linalg.eigh(s.time_covariance())
        keep = values > 1e-10 * values.max()
        W = (vectors[:, keep] / np.sqrt(values[keep])).T
        g = setup["greens"][s.name]
        los = pattern.values(s.x, s.y, s.look[:, 0], s.look[:, 1], s.look[:, 2])[:, 0]
        slip = np.vstack([np.hstack([np.hstack([(W[q] @ T[:, k]) * g[:, :n] for k in range(K)]),
                                     np.hstack([(W[q] @ T[:, k]) * g[:, n:] for k in range(K)])]) for q in range(W.shape[0])])
        amplitude = np.concatenate([(W[q] @ P) * los for q in range(W.shape[0])])
        plane = _poly(s.x, s.y)
        ramps = np.vstack([np.kron(W[q], plane) for q in range(W.shape[0])])
        blocks.append((slip, amplitude, ramps, (W @ s.values).ravel()))
    n_ramp = sum(b[2].shape[1] for b in blocks)
    rows = sum(b[0].shape[0] for b in blocks)
    A = np.zeros((rows, 2 * K * n + 1 + n_ramp))
    row = col = 0
    for slip, amplitude, ramps, _ in blocks:
        m = slip.shape[0]
        A[row:row + m, :2 * K * n] = slip
        A[row:row + m, 2 * K * n] = amplitude
        A[row:row + m, 2 * K * n + 1 + col:2 * K * n + 1 + col + ramps.shape[1]] = ramps
        row, col = row + m, col + ramps.shape[1]
    joined = FaultMesh.join([setup["mesh"]] * K)
    smooth = scale_smoothing(laplace_beltrami(joined, 1.0, 3.0), _LAM)
    edge = _boundary_rows(make_basis(joined, "node"), ("bottom", "left", "right"), inv.boundary_scale)
    extra = sp.csr_matrix((smooth.shape[0], 1 + n_ramp))
    reg = sp.vstack([sp.hstack([smooth, extra]), sp.hstack([edge, sp.csr_matrix((edge.shape[0], 1 + n_ramp))])]).toarray()
    lo, hi = slip_bounds(K * n, _SOLVE["strike"], _SOLVE["dip"], _SOLVE["polarity"])
    lo, hi = np.r_[lo, -np.inf * np.ones(1 + n_ramp)], np.r_[hi, np.inf * np.ones(1 + n_ramp)]
    brute = lsq_linear(np.vstack([A, reg]), np.r_[np.concatenate([b[3] for b in blocks]), np.zeros(reg.shape[0])],
                       bounds=(lo, hi), method="bvls", tol=1e-13, max_iter=5000).x
    assert np.abs(model.x[:4 * n] - brute[:4 * n]).max() < 1e-7
    assert model.pattern_amplitudes["hump"] == pytest.approx(brute[4 * n], abs=1e-7)


def test_a_planted_amplitude_comes_back_beside_the_slip(setup):  # noqa: F811
    n = setup["n"]
    amp = np.r_[-0.5 * setup["patch"](0.0, 30e3, 8e3), np.zeros(n)]
    basis = TimeBasis.log_decay(10.0)
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp
    pattern = Pattern.from_enu("hump", _hump(), TimeBasis.exp_decay(200.0))
    t0, t1 = np.full(12, 1.0), 1.0 + 30.0 * np.arange(1, 13)
    series = [_with_pattern(setup, "A", t0, t1, field, pattern, 0.08, ramp="none", seed=1),
              _with_pattern(setup, "D", t0, t1, field, pattern, 0.08, ramp="none", seed=2)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], patterns=[pattern])
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    assert model.pattern_amplitudes["hump"] == pytest.approx(0.08, abs=0.006)
    without = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], _greens=inv.cell_greens)
    plain = without.solve(smoothing=_LAM, **_SOLVE)
    fit, unfit = inv.series_fit(model), without.series_fit(plain)
    assert fit["A"]["rms_residual"] < 0.75 * unfit["A"]["rms_residual"], "left out, the hump stays in the residual"
    assert plain.pattern_amplitudes == {}

    # the amplitude travels with what reads a model
    copies = monte_carlo(model, n=3, noise="weights", seed=1)
    assert copies.apply(lambda m: m.pattern_amplitudes["hump"]).shape == (3,)
    other = inv.with_time_basis(TimeBasis.log_decay(30.0))
    assert other.solve(smoothing=_LAM, **_SOLVE).pattern_amplitudes["hump"] == pytest.approx(0.08, abs=0.02)
    assert other.pattern_cells is inv.pattern_cells


def test_a_pattern_survives_saving_and_cross_validation(setup, tmp_path):  # noqa: F811
    n = setup["n"]
    amp = np.r_[-0.5 * setup["patch"](0.0, 30e3, 8e3), np.zeros(n)]
    basis = TimeBasis.log_decay(10.0)
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp
    pattern = Pattern.from_enu("hump", _hump(), TimeBasis.exp_decay(200.0))
    t0, t1 = np.full(6, 1.0), 1.0 + 40.0 * np.arange(1, 7)
    series = [_with_pattern(setup, "A", t0, t1, field, pattern, 0.08, ramp="none", seed=1),
              _with_pattern(setup, "D", t0, t1, field, pattern, 0.08, ramp="none", seed=2)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], patterns=[pattern])
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    path = tmp_path / "m.slip.zip"
    model.save(path)
    assert TimeDependentModel.load(path).pattern_amplitudes == pytest.approx(model.pattern_amplitudes)
    curve = blocked_cv_curve(inv, [_LAM, 10 * _LAM], n_folds=4, **_SOLVE)
    assert np.isfinite(curve["cv_rms"].values).all()
    with pytest.raises(ValueError, match="different names"):
        TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], patterns=[pattern, pattern])
    with pytest.raises(ValueError, match="no time function"):
        TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], patterns=[Pattern.from_enu("x", _hump())])


def test_a_shape_that_changes_with_time(setup):  # noqa: F811
    """A signal not separable in one (shape, function), kept as a few sharing one amplitude."""
    n = setup["n"]
    amp = np.r_[-0.3 * setup["patch"](0.0, 30e3, 8e3), np.zeros(n)]
    basis = TimeBasis.log_decay(10.0)
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp
    times = np.linspace(0.0, 400.0, 81)

    def history(x, y):
        """A hump that widens as it grows."""
        return np.array([(1.0 - np.exp(-t / 150.0)) * np.exp(-(x ** 2 + y ** 2) / (30e3 + 100.0 * t) ** 2) for t in times])

    t0, t1 = np.full(10, 1.0), 1.0 + 35.0 * np.arange(1, 11)
    x, y = np.r_[setup["sites"]["A"][0], setup["sites"]["D"][0]], np.r_[setup["sites"]["A"][1], setup["sites"]["D"][1]]
    modes, functions = separate_history(times, history(x, y), tolerance=1e-3)
    assert 2 <= modes.shape[0] <= 6
    assert np.abs(functions @ modes - history(x, y)).max() < 0.02
    series = []
    for track, seed in (("A", 5), ("D", 6)):
        s = _series(setup, track, t0, t1, field, ramp="none", seed=seed)
        look = s.look
        truth = history(s.x, s.y)
        at = lambda t: np.array([np.interp(t, times, truth[:, k]) for k in range(s.n_cells)])
        s.values = s.values + 0.07 * look[:, 2] * (at(t1).T - at(t0).T) * 1.0
        series.append(s)

    # Modes must come from one SVD: evaluate at the union of points, then look each point up.
    table = {(a, b): k for k, (a, b) in enumerate(zip(x, y))}

    def los_from_table(px, py, le, ln, lu):
        rows = [table[(a, b)] for a, b in zip(px, py)]
        return lu[:, None] * modes[:, rows].T

    pattern = Pattern("widening", los_from_table, TimeBasis.tabulated(times, functions))
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False, patterns=[pattern])
    assert pattern.n_modes == modes.shape[0]
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    assert model.pattern_amplitudes["widening"] == pytest.approx(0.07, abs=0.006)


def test_poroelastic_rebound_is_the_oracles_difference():
    """Uniform slip on a rectangle: the pattern against Okada (1985) at the two ratios, an independent solution."""
    from types import SimpleNamespace

    from slipsolve._okada import okada_disp_surface

    length, depth = 60e3, 15e3
    frame_trace = FaultTrace(np.array([-120.33, -119.67]), np.array([36.0, 36.0]))
    frame = frame_trace.local_frame()
    x_trace, y_trace = frame_trace.to_local(frame)
    length = frame_trace.length(frame)
    mesh = FaultMesh.vertical(frame_trace, frame, max_depth=depth, edge_length=5e3)
    basis = make_basis(mesh, "element")
    model = SimpleNamespace(mesh=mesh, basis=basis, x=np.r_[np.ones(mesh.n_elements), np.zeros(mesh.n_elements)])
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-30e3, length + 30e3, 150), rng.uniform(-50e3, 50e3, 150)
    keep = np.abs(v) > 2e3
    u, v = u[keep], v[keep]
    x, y = x_trace[0] + u, y_trace[0] + v
    pattern = poroelastic_pattern(model, TimeBasis.exp_decay(100.0), nu_drained=0.25, nu_undrained=0.31)
    expected = (np.array(okada_disp_surface(u, v, length, depth, depth, 90.0, 1.0, 0.0, 0.0, 0.25))
                - np.array(okada_disp_surface(u, v, length, depth, depth, 90.0, 1.0, 0.0, 0.0, 0.31)))
    zero = np.zeros(x.size)
    for k, look in enumerate(((np.ones(x.size), zero, zero), (zero, np.ones(x.size), zero), (zero, zero, np.ones(x.size)))):
        got = pattern.values(x, y, *look)[:, 0]
        # the straight line and the frame agree to a few metres over 60 km, so this is the oracle's scale
        assert np.abs(got - expected[k]).max() < 0.01 * np.abs(expected).max(), "ENU"[k]
    same = poroelastic_pattern(model, nu_drained=0.3, nu_undrained=0.3)
    assert np.abs(same.values(x, y, zero, zero, np.ones(x.size))).max() < 1e-12
    assert pattern.name == "poroelastic"
    with pytest.raises(ValueError, match="one time function"):
        poroelastic_pattern(model, TimeBasis.exp_decay(10.0, 20.0))


def test_a_velocity_pattern_comes_back_in_interseismic_mode():
    trace = FaultTrace(np.array([-120.0, -119.0]), np.array([36.0, 36.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(-121.0, -118.0, 90), rng.uniform(35.2, 36.8, 90)
    x, y = frame.to_local(lon, lat)
    keep = trace.distance(x, y, frame) > 5e3
    lon, lat = lon[keep], lat[keep]
    zero = np.zeros(lon.size)
    obs = Observations.concat([gnss_observations(lon, lat, zero, zero, None, 1e-3, 1e-3, None, frame, components="en")], normalize="none")
    pattern = Pattern.from_enu("hump", _hump(100e3))
    blocks = side_labels(obs, trace, "north", "south")
    inv = InterseismicInversion(mesh, obs, blocks=blocks, patterns=[pattern], faults=("north", "south"))
    assert inv.ramp_labels[-1] == "pattern:hump"
    truth = np.zeros(inv.n_ramp)
    truth[-1] = 0.004
    truth[inv.ramp_labels.index("block:north:ve")] = 0.002
    inv.d = inv.ramp @ truth
    model = inv.solve(smoothing=1e3, sides=("bottom",))
    assert model.pattern_amplitudes["hump"] == pytest.approx(0.004, abs=2e-4)
    assert not {"hump"} & set(model.poles)


def test_a_pattern_from_another_frame_is_refused_and_can_be_moved():
    from types import SimpleNamespace

    from slipsolve import LocalFrame, TrackSeries

    trace = FaultTrace(np.array([-118.5, -117.5]), np.array([35.5, 35.9]))
    frame = trace.local_frame()
    other = LocalFrame(frame.origin_lon + 0.3, frame.origin_lat - 0.2)
    mesh = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=6e3)
    moved = mesh.in_frame(other)
    assert mesh.in_frame(frame) is mesh and moved.frame is other
    assert moved.n_nodes == mesh.n_nodes and np.array_equal(moved.params, mesh.params) and np.array_equal(moved.triangles, mesh.triangles)
    lon, lat = other.to_lonlat(moved.nodes[:, 0], moved.nodes[:, 1])
    back = np.column_stack(frame.to_local(lon, lat))
    assert np.abs(back - mesh.nodes[:, :2]).max() < 1e-6
    tops = moved.nodes[moved.params[:, 1] == 0.0]
    below = moved.nodes[moved.params[:, 1] == moved.params[:, 1].max()]
    assert np.array_equal(np.sort(tops[:, 0]), np.sort(below[:, 0])), "a vertical column stays vertical"

    # The field at one place is the same in either frame (the looks here along z, where the frames share an axis).
    basis = make_basis(mesh, "node")
    slip = np.r_[np.ones(basis.n_basis), np.zeros(basis.n_basis)]
    model = SimpleNamespace(mesh=mesh, basis=basis, x=slip)
    time = TimeBasis.exp_decay(100.0)
    here = poroelastic_pattern(model, time)
    there = poroelastic_pattern(model, time, frame=other)
    assert here.frame is frame and there.frame is other
    x, y = np.array([6e3, -9e3]), np.array([5e3, 11e3])
    lon, lat = frame.to_lonlat(x, y)
    xo, yo = other.to_local(lon, lat)
    up = (np.zeros(2), np.zeros(2), np.ones(2))
    assert np.abs(here.values(x, y, *up) - there.values(xo, yo, *up)).max() < 1e-3 * np.abs(here.values(x, y, *up)).max()

    series = [TrackSeries([x[0]], [y[0]], (0.0, 0.0, 1.0), np.array([[0.0]]), [1.0], [2.0], 1e-3, frame=frame, name="gnss_u")]
    with pytest.raises(ValueError, match="different LocalFrame"):
        TimeDependentInversion(mesh, series, TimeBasis.log_decay(1.0), patterns=[there])
