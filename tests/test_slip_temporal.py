"""The postseismic mode: slip fields of a time basis, from series of any length.

The reductions are identities, so they are tested against the system they replace. The
recoveries plant shallow slip well inside what the data can see: what they test is that
the time structure lands in the right fields, not how deep afterslip can be seen.
"""

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers .rio)
import scipy.sparse as sp
import xarray as xr
from scipy.optimize import lsq_linear

from slipsolve import (FaultMesh, FaultTrace, HalfSpaceTDE, LosGrid, Observations, SlipInversion, SlipSnapshot,
                       TimeBasis, TimeDependentInversion, TimeDependentModel, TrackSeries, gnss_series,
                       resolution_diagonal, blocked_cv_curve)
from slipsolve.inversion import _boundary_rows
from slipsolve.regularize import laplace_beltrami, scale_smoothing, slip_bounds

_LOOK = {"A": np.array([-0.56, -0.10, 0.82]), "D": np.array([0.60, -0.10, 0.79])}
_LOOK = {k: v / np.linalg.norm(v) for k, v in _LOOK.items()}
_SIGMA = 0.004
_LAM = 2e4
_SOLVE = dict(polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0))


def _poly(x, y, degree=1):
    span = max(np.ptp(x), np.ptp(y), 1.0)
    a, b = (x - x.mean()) / span, (y - y.mean()) / span
    return np.column_stack([a ** (d - i) * b ** i for d in range(degree + 1) for i in range(d + 1)])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(np.array([-68.1, -67.0]), np.array([10.55, 10.62]), name="toy")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=24e3, edge_length=6e3)
    rng = np.random.default_rng(2)

    def scatter(n):
        x = rng.uniform(-110e3, 110e3, n)
        y = rng.uniform(-70e3, 70e3, n)
        keep = trace.distance(x, y, frame) > 2.5e3
        return x[keep], y[keep]

    sites = {k: scatter(320) for k in _LOOK}
    obs = Observations.concat(
        [Observations.from_points(*sites[k], np.zeros(sites[k][0].size), _LOOK[k], _SIGMA, name=k,
                                  frame=frame, epsg=None) for k in _LOOK], normalize="none")
    engine = HalfSpaceTDE()
    base = SlipInversion(mesh, obs, engine=engine, basis="node")
    greens = {k: base.g[obs.track_mask(k)] for k in _LOOK}
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    patch = (lambda centre, width, depth: np.exp(-((s - s.mean() - centre) / width) ** 2)
             * np.exp(-((z + depth) / 5e3) ** 2))
    return dict(mesh=mesh, frame=frame, sites=sites, engine=engine, greens=greens, patch=patch, rng=rng, n=mesh.n_nodes)


def _series(setup, track, t0, t1, field, ramp="linear", noise=_SIGMA, seed=0, sigma=None, sigma_kind="dates"):
    """Pairs on one set of cells: `field(t0, t1)` of the slip vector, plus a free plane on each, plus date noise."""
    x, y = setup["sites"][track]
    rng = np.random.default_rng(seed)
    dates = np.unique(np.concatenate([t0, t1]))
    date_noise = {d: rng.normal(0.0, noise, x.size) for d in dates}
    plane = _poly(x, y)
    values = []
    for a, b in zip(t0, t1):
        v = setup["greens"][track] @ field(a, b) + date_noise[b] - date_noise[a]
        if ramp != "none":
            v = v + plane @ rng.normal(0.0, 0.02, plane.shape[1])
        values.append(v)
    return TrackSeries(x, y, _LOOK[track], np.array(values), t0, t1, noise if sigma is None else sigma,
                       name=track, ramp=ramp, frame=setup["frame"], noise=sigma_kind)


# -- the time basis -------------------------------------------------------------------------
def test_basis_shapes():
    t = np.array([-5.0, 0.0, 10.0, 100.0])
    assert np.allclose(TimeBasis.log_decay(30.0).cumulative(t)[:, 0], np.log1p(np.array([0, 0, 10, 100]) / 30.0))
    assert np.allclose(TimeBasis.exp_decay(30.0).cumulative(t)[:, 0], 1 - np.exp(-np.array([0, 0, 10, 100]) / 30.0))
    assert np.allclose(TimeBasis.linear().cumulative(365.25)[:, 0], 1.0)
    assert np.array_equal(TimeBasis.step().cumulative(t)[:, 0], [0.0, 1.0, 1.0, 1.0])


def test_epochs_are_increments_with_fractional_pairs():
    basis = TimeBasis.epochs([0.0, 10.0, 30.0])
    assert len(basis) == 2
    # a pair inside the second window sees its share; one across both sees all of each
    assert np.allclose(basis.weights([15.0], [25.0]), [[0.0, 0.5]])
    assert np.allclose(basis.weights([0.0], [30.0]), [[1.0, 1.0]])
    assert np.allclose(basis.weights([5.0], [20.0]), [[0.5, 0.5]])
    with pytest.raises(ValueError):
        TimeBasis.epochs([0.0, 0.0])


def test_bases_add_and_round_trip():
    basis = TimeBasis.step() + TimeBasis.log_decay(10.0, 100.0) + TimeBasis.epochs([0.0, 5.0])
    assert len(basis) == 4
    again = TimeBasis.from_dict(basis.to_dict())
    t = np.linspace(-3.0, 300.0, 7)
    assert np.array_equal(again.cumulative(t), basis.cumulative(t))


# -- the covariance between pairs ------------------------------------------------------------
def test_time_covariance_follows_the_dates(setup):
    x, y = setup["sites"]["A"]
    one = np.zeros((3, x.size))
    common = TrackSeries(x, y, _LOOK["A"], one, [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 2.0, frame=setup["frame"])
    assert np.allclose(common.time_covariance(), 4.0 * (np.eye(3) + np.ones((3, 3))))
    chain = TrackSeries(x, y, _LOOK["A"], one, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 2.0, frame=setup["frame"])
    assert np.allclose(chain.time_covariance(), 4.0 * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]))
    pairs = TrackSeries(x, y, _LOOK["A"], one, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 2.0, frame=setup["frame"], noise="pairs")
    assert np.allclose(pairs.time_covariance(), 4.0 * np.eye(3))


# -- the reductions are identities ------------------------------------------------------------
def _brute_force(setup, series_list, basis, lam, edge_scale, solve=_SOLVE):
    """The whole system, every pair a block of rows, whitened by Ct, with a free ramp on every pair."""
    mesh, n = setup["mesh"], setup["n"]
    K = len(basis)
    blocks_a, blocks_b = [], []
    for s in series_list:
        T = basis.weights(s.t0, s.t1)
        C = s.time_covariance()
        values, vectors = np.linalg.eigh(C)
        keep = values > 1e-10 * values.max()
        W = (vectors[:, keep] / np.sqrt(values[keep])).T
        g = setup["greens"][s.name]
        # rows (whitened pair q, cell): sum_p W[q,p] [ sum_k T[p,k] g m_k + ramp ]
        slip_rows = np.vstack([np.hstack([np.hstack([(W[q] @ T[:, k]) * g[:, :n] for k in range(K)]),
                                          np.hstack([(W[q] @ T[:, k]) * g[:, n:] for k in range(K)])])
                               for q in range(W.shape[0])])
        plane = _poly(s.x, s.y) if s.ramp != "none" else np.zeros((s.n_cells, 0))
        ramp_rows = np.vstack([np.kron(W[q], plane) for q in range(W.shape[0])]) if plane.size else np.zeros((slip_rows.shape[0], 0))
        blocks_a.append((slip_rows, ramp_rows))
        blocks_b.append((W @ s.values).ravel())
    n_ramp = sum(r.shape[1] for _, r in blocks_a)
    A = np.zeros((sum(a.shape[0] for a, _ in blocks_a), 2 * K * n + n_ramp))
    row = col = 0
    for slip_rows, ramp_rows in blocks_a:
        m = slip_rows.shape[0]
        A[row:row + m, :2 * K * n] = slip_rows
        A[row:row + m, 2 * K * n + col:2 * K * n + col + ramp_rows.shape[1]] = ramp_rows
        row, col = row + m, col + ramp_rows.shape[1]
    b = np.concatenate(blocks_b)
    joined = FaultMesh.join([mesh] * K)
    from slipsolve.basis import make_basis

    smooth = scale_smoothing(laplace_beltrami(joined, 1.0, 3.0), lam)
    edge = _boundary_rows(make_basis(joined, "node"), ("bottom", "left", "right"), edge_scale)
    pad = sp.csr_matrix((smooth.shape[0], n_ramp))
    rows = sp.vstack([sp.hstack([smooth, pad]), sp.hstack([edge, sp.csr_matrix((edge.shape[0], n_ramp))])]).toarray()
    lo, hi = slip_bounds(K * n, solve["strike"], solve["dip"], solve["polarity"])
    lo, hi = np.r_[lo, -np.inf * np.ones(n_ramp)], np.r_[hi, np.inf * np.ones(n_ramp)]
    stacked = np.vstack([A, rows])
    rhs = np.r_[b, np.zeros(rows.shape[0])]
    return lsq_linear(stacked, rhs, bounds=(lo, hi), method="bvls", tol=1e-13, max_iter=5000).x


def test_one_shape_matches_the_brute_force_system(setup):
    """Eight dates per track on one reference date, one decay shape: the stack is exact."""
    n = setup["n"]
    amp = np.r_[-0.5 * setup["patch"](0.0, 30e3, 8e3), np.zeros(n)]
    basis = TimeBasis.log_decay(30.0)
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp
    t_a = 12.0 * np.arange(1, 9)
    t_d = 5.0 + 12.0 * np.arange(1, 9)
    series = [_series(setup, "A", np.full(8, 1.0), t_a, field, seed=1),
              _series(setup, "D", np.full(8, 5.0), t_d, field, seed=2)]
    # frame_looks=False, so this assembles the same G the brute force is given
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    x_brute = _brute_force(setup, series, basis, _LAM, inv.boundary_scale)
    assert inv.obs.n == sum(s.n_cells for s in series), "one shape must reduce each track to one virtual pair"
    assert np.abs(model.x[:2 * n] - x_brute[:2 * n]).max() < 1e-7


def test_two_fields_on_a_network_match_the_brute_force_system(setup):
    """A redundant pair network, two fields, shared dates: virtual pairs with free ramps are exact."""
    n = setup["n"]
    m1 = np.r_[-0.4 * setup["patch"](-15e3, 25e3, 7e3), np.zeros(n)]
    m2 = np.r_[-0.3 * setup["patch"](15e3, 25e3, 11e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 40.0, 100.0])
    field = lambda a, b: basis.weights([a], [b])[0] @ np.array([m1, m2])
    dates = np.array([4.0, 16.0, 28.0, 40.0, 52.0, 76.0, 100.0])
    t0, t1 = [], []
    for i in range(dates.size):
        for j in range(i + 1, min(i + 3, dates.size)):
            t0.append(dates[i])
            t1.append(dates[j])
    series = [_series(setup, "A", np.array(t0), np.array(t1), field, seed=3),
              _series(setup, "D", np.array(t0) + 1.0, np.array(t1) + 1.0, field, seed=4)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    model = inv.solve(smoothing=_LAM, **_SOLVE)
    x_brute = _brute_force(setup, series, basis, _LAM, inv.boundary_scale)
    assert inv.obs.n == 2 * sum(s.n_cells for s in series), "two fields: two virtual pairs a track"
    assert np.abs(model.x[:4 * n] - x_brute[:4 * n]).max() < 1e-7


def test_one_engine_call_per_set_of_cells(setup):
    class Counting(HalfSpaceTDE):
        rows = 0
        calls = 0

        def los_matrix(self, mesh, x, y, *args, **kwargs):
            Counting.rows += np.size(x)
            Counting.calls += 1
            return super().los_matrix(mesh, x, y, *args, **kwargs)

    n = setup["n"]
    amp = np.r_[-0.5 * setup["patch"](0.0, 30e3, 8e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 30.0, 60.0, 120.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * amp
    t = 12.0 * np.arange(1, 11)
    series = [_series(setup, "A", np.full(10, 1.0), t, field, seed=5)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=Counting(), frame_looks=False)
    assert Counting.calls == 1 and Counting.rows == series[0].n_cells
    assert inv.g.shape == (3 * series[0].n_cells, 2 * 3 * n)
    # a new tau needs no engine at all
    other = TimeDependentInversion(setup["mesh"], series, TimeBasis.log_decay(30.0), engine=Counting(), frame_looks=False)
    calls = Counting.calls
    again = other.with_time_basis(TimeBasis.log_decay(60.0))
    assert Counting.calls == calls and again.g.shape[1] == 2 * n


# -- recovery ----------------------------------------------------------------------------------
def test_epochs_recover_a_migrating_increment(setup):
    """Three increments in three places, seen through overlapping pairs with their own dates."""
    n = setup["n"]
    truth = [np.r_[-0.8 * setup["patch"](-20e3, 18e3, 4e3), np.zeros(n)],
             np.r_[-0.6 * setup["patch"](0.0, 18e3, 6e3), np.zeros(n)],
             np.r_[-0.4 * setup["patch"](20e3, 18e3, 8e3), np.zeros(n)]]
    basis = TimeBasis.epochs([0.0, 20.0, 60.0, 180.0])
    field = lambda a, b: basis.weights([a], [b])[0] @ np.array(truth)

    def make(start, key, seed):                 # consecutive 12-day pairs, each track its own dates
        dates = start + 12.0 * np.arange(15)
        return _series(setup, key, dates[:-1], dates[1:], field, seed=seed, noise=0.002)

    series = [make(1.0, "A", 6), make(5.0, "D", 7)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    model = inv.solve(smoothing=3e3, **_SOLVE)
    assert isinstance(model, TimeDependentModel)
    for k, want in enumerate(truth):
        got = model.field(k).strike_slip
        assert np.corrcoef(got, want[:n])[0, 1] > 0.75, k
        assert 0.8 < got.sum() / want[:n].sum() < 1.3, k
    # each window's slip, as an ordinary model on the original mesh
    snapshot = model.slip_between(0.0, 180.0)
    assert isinstance(snapshot, SlipSnapshot)
    assert snapshot.mesh is setup["mesh"] and snapshot.strike_slip.size == n
    assert np.allclose(snapshot.strike_slip, sum(model.field(k).strike_slip for k in range(3)))
    fit = inv.series_fit(model)
    assert fit["A"]["rms_residual"] < fit["A"]["rms_data"]
    # the pair noise is sqrt(2) x 2 mm: the residual must reach about it and not much further
    assert 0.5 * 0.002 * np.sqrt(2) < fit["A"]["rms_residual"] < 2.5 * 0.002 * np.sqrt(2)


def test_tau_search_finds_the_shape(setup):
    """Misfit on the real pairs, over tau, must prefer the tau the data were made with."""
    n = setup["n"]
    amp = np.r_[-1.0 * setup["patch"](0.0, 30e3, 6e3), np.zeros(n)]
    planted = TimeBasis.log_decay(10.0)
    field = lambda a, b: planted.weights([a], [b])[0, 0] * amp
    t = np.r_[3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
    series = [_series(setup, "A", np.full(8, 1.0), t, field, seed=8, noise=0.001),
              _series(setup, "D", np.full(8, 2.0), t + 1.0, field, seed=9, noise=0.001)]
    inv = TimeDependentInversion(setup["mesh"], series, planted, engine=setup["engine"], frame_looks=False)
    taus = [1.0, 10.0, 100.0, 1000.0]
    misfit = []
    for tau in taus:
        other = inv.with_time_basis(TimeBasis.log_decay(tau))
        fit = other.series_fit(other.solve(smoothing=1e3, **_SOLVE))
        misfit.append(fit["A"]["rms_residual"] + fit["D"]["rms_residual"])
    assert int(np.argmin(misfit)) == 1


def test_coseismic_step_and_decay_are_separated(setup):
    """Pairs spanning the event for different times after it tell the step from the decay."""
    n = setup["n"]
    step, decay = (np.r_[-1.5 * setup["patch"](0.0, 25e3, 5e3), np.zeros(n)],
                   np.r_[-0.5 * setup["patch"](15e3, 25e3, 5e3), np.zeros(n)])
    basis = TimeBasis.step() + TimeBasis.log_decay(20.0)
    field = lambda a, b: basis.weights([a], [b])[0] @ np.array([step, decay])
    t1 = np.r_[1.0, 6.0, 12.0, 24.0, 48.0, 96.0]
    series = [_series(setup, "A", np.full(6, -6.0), t1, field, seed=10, noise=0.001),
              _series(setup, "D", np.full(6, -2.0), t1 + 3.0, field, seed=11, noise=0.001)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    model = inv.solve(smoothing=2e3, **_SOLVE)
    assert np.corrcoef(model.field(0).strike_slip, step[:n])[0, 1] > 0.8
    assert np.corrcoef(model.field(1).strike_slip, decay[:n])[0, 1] > 0.7


@pytest.mark.filterwarnings("ignore:the active-set solve was refused")
def test_temporal_rows_tie_consecutive_fields(setup):
    n = setup["n"]
    truth = np.r_[-0.5 * setup["patch"](0.0, 25e3, 6e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 30.0, 60.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * truth
    series = [_series(setup, "A", np.r_[1.0, 31.0], np.r_[29.0, 59.0], field, seed=12)]
    free = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    tied = free.with_temporal(1e9)
    assert tied.g is free.g
    a, b = free.solve(smoothing=_LAM, **_SOLVE), tied.solve(smoothing=_LAM, **_SOLVE)
    gap = lambda m: np.abs(m.coefficients[0] - m.coefficients[1]).max()
    assert gap(b) < 1e-3 * max(gap(a), 1e-6)
    assert free.prior_rows() is None
    assert tied.prior_rows().shape == (2 * n, tied.n_param)
    # resolution reads the same rows: the rank must fall when the fields are tied
    r_free = resolution_diagonal(free, _LAM).attrs["effective_rank"]
    r_tied = resolution_diagonal(tied, _LAM).attrs["effective_rank"]
    assert r_tied < r_free


def test_polarity_makes_epochs_monotonic(setup):
    n = setup["n"]
    truth = np.r_[-0.4 * setup["patch"](0.0, 25e3, 6e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 50.0, 100.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * truth
    series = [_series(setup, "A", np.r_[1.0, 51.0], np.r_[49.0, 99.0], field, seed=13)]
    model = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"],
                                   frame_looks=False).solve(smoothing=_LAM, **_SOLVE)
    assert model.coefficients[:, 0].max() <= 1e-9


def test_limits_per_parameter_of_one_field(setup):
    n = setup["n"]
    amp = np.r_[-0.4 * setup["patch"](0.0, 25e3, 6e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 50.0, 100.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * amp
    series = [_series(setup, "A", np.r_[1.0, 51.0], np.r_[49.0, 99.0], field, seed=14)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    lo = np.full(n, -5.0)
    lo[::2] = -1e-3
    model = inv.solve(smoothing=_LAM, strike=(lo, np.zeros(n)), dip=(-1.0, 1.0))
    assert np.all(model.coefficients[:, 0, ::2] >= -1e-3 - 1e-9)


def test_gnss_series_and_shared_track(setup):
    """Stations with their own days, sharing one 'gnss' track, no ramp, beside InSAR."""
    n, frame = setup["n"], setup["frame"]
    amp = np.r_[-1.0 * setup["patch"](0.0, 30e3, 6e3), np.zeros(n)]
    basis = TimeBasis.log_decay(20.0)
    engine = setup["engine"]
    stations = np.array([[-40e3, 25e3], [-20e3, -30e3], [0.0, 20e3], [25e3, -15e3], [50e3, 35e3], [-70e3, -40e3]])
    rng = np.random.default_rng(15)
    series = []
    lon, lat = frame.to_lonlat(stations[:, 0], stations[:, 1])
    from slipsolve.basis import make_basis

    for i, (x, y) in enumerate(stations):
        t = np.arange(1.0, 200.0, 2.0 + i)
        g = np.vstack([engine.los_matrix(setup["mesh"], np.array([x]), np.array([y]), *np.eye(3)[c][:, None])
                       for c in range(3)])
        p = make_basis(setup["mesh"], "node").projection()
        slip = (p @ amp[:n], p @ amp[n:])
        m = g[:, :slip[0].size] @ slip[0] + g[:, slip[0].size:] @ slip[1]
        f = basis.cumulative(t)[:, 0]
        e, nn, u = (m[c] * f + rng.normal(0.0, s, t.size) for c, s in enumerate((0.001, 0.001, 0.003)))
        nn = nn.copy()
        nn[3] = np.nan                                        # a missing day on one component
        series += gnss_series(lon[i], lat[i], t, e, nn, u, 0.001, 0.001, 0.003, frame, name="gnss")
    assert all(s.name.startswith("gnss_") and s.n_cells == 1 for s in series)
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=engine, frame_looks=False)
    assert inv.obs.tracks == ["gnss_e", "gnss_n", "gnss_u"] and inv.n_ramp == 0
    assert inv.obs.n == len(series), "one shape: one virtual row a component"
    model = inv.solve(smoothing=1e3, **_SOLVE)
    fit = inv.series_fit(model)["gnss_e"]
    assert fit["rms_residual"] < 0.5 * fit["rms_data"]
    assert 0.5e-3 < fit["rms_residual"] < 3e-3, "reaches about its pair noise, sqrt(2) x 1 mm"


# -- a real raster, and a series into the interseismic mode ------------------------------------
def _stack_of(setup, field, t0, t1, noise=0.003, hole=True):
    """A `(pair, y, x)` LOS stack on a local lattice: `field(a, b)` of the slip, a plane and noise per pair."""
    frame = setup["frame"]
    xs = np.arange(-110e3, 110e3, 4e3)
    ys = np.arange(-70e3, 70e3, 4e3)
    gx, gy = np.meshgrid(xs, ys)
    look = _LOOK["A"]
    engine = setup["engine"]
    # the mesh's nodal forward model at every pixel, at once
    g = engine.los_matrix(setup["mesh"], gx.ravel(), gy.ravel(), *look[:, None] * np.ones((3, gx.size)))
    from slipsolve.basis import make_basis

    p = make_basis(setup["mesh"], "node").projection()
    n = setup["n"]
    m = g.shape[1] // 2
    nodal = np.hstack([g[:, :m] @ p, g[:, m:] @ p])
    rng = np.random.default_rng(21)
    pairs = []
    for a, b in zip(t0, t1):
        v = nodal @ field(a, b) + rng.normal(0.0, noise, gx.size)
        pairs.append(v.reshape(gx.shape))
    los = np.array(pairs)
    if hole:
        los[1, 3:6, 3:6] = np.nan        # one pair's missing patch must leave every pair
    ds = xr.Dataset({"los": (("pair", "y", "x"), los),
                     "los_east": (("y", "x"), np.full(gx.shape, look[0])),
                     "los_north": (("y", "x"), np.full(gx.shape, look[1])),
                     "los_up": (("y", "x"), np.full(gx.shape, look[2]))},
                    coords={"x": xs, "y": ys, "pair": np.arange(len(t0))})
    ds = ds.rio.write_crs(frame.local_crs)
    ds.attrs.update(frame=frame.to_dict(), sign=1, direction="ascending")
    return LosGrid(ds), n


def test_from_los_samples_every_pair_on_the_same_cells(setup):
    amp = lambda n: np.r_[-1.0 * setup["patch"](0.0, 30e3, 5e3), np.zeros(n)]
    basis = TimeBasis.log_decay(20.0)
    t0, t1 = np.full(6, 1.0), np.r_[5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp(setup["n"])
    grid, n = _stack_of(setup, field, t0, t1)
    trace = FaultTrace(np.array([-68.1, -67.0]), np.array([10.55, 10.62]), name="toy")
    series = TrackSeries.from_los(grid, t0, t1, 0.002, name="A", frame=setup["frame"], trace=trace, ramp="offset",
                                  rms_min=0.002, width_min=4e3, width_max=30e3, exclude_within=3e3)
    assert series.n_pairs == 6 and series.values.shape == (6, series.n_cells)
    assert np.isfinite(series.values).all(), "the missing patch left every pair"
    assert series.n_cells > 30
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"])
    assert inv.obs.n == series.n_cells and inv.obs.tracks == ["A"]
    model = inv.solve(smoothing=3e3, **_SOLVE)
    got = model.field(0).strike_slip
    assert np.corrcoef(got, amp(n)[:n])[0, 1] > 0.8
    fit = inv.series_fit(model)["A"]
    assert fit["rms_residual"] < 0.5 * fit["rms_data"]


def test_velocity_of_a_series(setup):
    """A rate, seasons and date noise: least squares returns the rate, with the sigma it should have."""
    x, y = setup["sites"]["A"]
    rng = np.random.default_rng(22)
    dates = np.arange(0.0, 6.0 * 365.25, 12.0)
    true = rng.uniform(-0.03, 0.03, x.size)
    year = dates[:, None] / 365.25
    positions = true * year + 0.004 * np.sin(2 * np.pi * year + 1.0) + rng.normal(0.0, 0.003, (dates.size, x.size))
    series = TrackSeries(x, y, _LOOK["A"], positions[1:] - positions[0], np.full(dates.size - 1, 0.0), dates[1:], 0.003,
                         name="A", frame=setup["frame"], ramp="linear")
    rate, sigma = series.velocity()
    expected = 0.003 / np.sqrt(np.sum((year - year.mean()) ** 2))       # a line through independent dates
    assert np.allclose(sigma, expected, rtol=0.05)
    assert abs(np.std(rate - true) / expected - 1.0) < 0.2
    assert np.std(series.velocity(seasonal=False)[0] - true) < 2.0 * expected
    obs = series.velocity_observations(sigma_floor=1e-3)
    assert obs.n == x.size and np.allclose(obs.ds["los"].values, rate) and np.allclose(obs.ds["std"].values, 1e-3)
    # the same dates as consecutive pairs carry the same information
    chain = TrackSeries(x, y, _LOOK["A"], np.diff(positions, axis=0), dates[:-1], dates[1:], 0.003, name="A",
                        frame=setup["frame"])
    assert np.allclose(chain.velocity()[0], rate, atol=1e-9)
    short = TrackSeries(x, y, _LOOK["A"], positions[1:3] - positions[0], [0.0, 0.0], dates[1:3], 0.003, name="A",
                        frame=setup["frame"])
    with pytest.raises(ValueError, match="cannot separate"):
        short.velocity()


def test_plots_read_a_window(setup):
    import matplotlib

    matplotlib.use("Agg")
    from slipsolve.plot import plot_slip

    n = setup["n"]
    amp = np.r_[-0.6 * setup["patch"](0.0, 25e3, 6e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 40.0, 120.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * amp
    series = [_series(setup, "A", np.r_[1.0, 41.0], np.r_[39.0, 119.0], field, seed=23)]
    model = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"],
                                   frame_looks=False).solve(smoothing=_LAM, **_SOLVE)
    assert plot_slip(model.slip_between(0.0, 120.0)) is not None
    matplotlib.pyplot.close("all")


# -- what the rest of the package reads -------------------------------------------------------
def test_blocked_cv_runs_on_a_time_dependent_problem(setup):
    n = setup["n"]
    amp = np.r_[-1.0 * setup["patch"](0.0, 30e3, 6e3), np.zeros(n)]
    basis = TimeBasis.log_decay(20.0) + TimeBasis.log_decay(200.0)
    field = lambda a, b: basis.weights([a], [b])[0, 0] * amp
    t = np.r_[6.0, 12.0, 24.0, 48.0, 96.0]
    series = [_series(setup, "A", np.full(5, 1.0), t, field, seed=16, noise=0.002),
              _series(setup, "D", np.full(5, 2.0), t + 1.0, field, seed=17, noise=0.002)]
    inv = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"], frame_looks=False)
    curve = blocked_cv_curve(inv, [1e3, 1e4, 1e5], n_folds=3, direction=(1.0, 0.0), **_SOLVE)
    assert np.isfinite(curve["cv_rms"].values).all()
    # in units of each row's noise: of order 1 at a sensible weight, whatever the number of dates
    assert 0.5 < curve["cv_rms"].values.min() < 5.0
    assert np.all(curve["effective_rank"].values[:-1] > curve["effective_rank"].values[1:])


def test_save_load_and_snapshot_text(setup, tmp_path):
    n = setup["n"]
    amp = np.r_[-0.6 * setup["patch"](0.0, 25e3, 6e3), np.zeros(n)]
    basis = TimeBasis.epochs([0.0, 40.0, 120.0])
    field = lambda a, b: basis.weights([a], [b]).sum() * amp
    series = [_series(setup, "A", np.r_[1.0, 41.0], np.r_[39.0, 119.0], field, seed=18)]
    model = TimeDependentInversion(setup["mesh"], series, basis, engine=setup["engine"],
                                   frame_looks=False).solve(smoothing=_LAM, **_SOLVE)
    path = model.save(tmp_path / "time.slip.zip")
    back = TimeDependentModel.load(path)
    assert np.array_equal(back.x, model.x)
    assert back.time_basis.names == basis.names
    assert np.allclose(back.slip_between(5.0, 100.0).strike_slip, model.slip_between(5.0, 100.0).strike_slip)
    assert np.isclose(back.slip_between(0.0, 120.0).moment(), model.slip_between(0.0, 120.0).moment())
    history = model.moment_history([0.0, 40.0, 120.0])
    assert history[0] == 0.0 and history[2] >= history[1]
    text = model.slip_between(0.0, 120.0).to_text(tmp_path / "window.txt")
    assert np.loadtxt(text, skiprows=1).shape == (setup["mesh"].n_elements, 10)


def test_refusals(setup):
    x, y = setup["sites"]["A"]
    frame = setup["frame"]
    nan = np.zeros((2, x.size))
    nan[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        TrackSeries(x, y, _LOOK["A"], nan, [0.0, 0.0], [1.0, 2.0], 0.01, frame=frame)
    with pytest.raises(ValueError, match="same"):
        TrackSeries(x, y, _LOOK["A"], np.zeros((1, x.size)), [1.0], [1.0], 0.01, frame=frame)
    a = TrackSeries(x, y, _LOOK["A"], np.zeros((1, x.size)), [0.0], [1.0], 0.01, name="A", ramp="linear", frame=frame)
    with pytest.raises(ValueError, match="share one ramp"):
        TimeDependentInversion(setup["mesh"], [a, a], TimeBasis.log_decay(1.0), engine=setup["engine"])
    before = TrackSeries(x, y, _LOOK["A"], np.zeros((1, x.size)), [-9.0], [-1.0], 0.01, name="A", frame=frame)
    with pytest.raises(ValueError, match="changes across"):
        TimeDependentInversion(setup["mesh"], [before], TimeBasis.log_decay(1.0), engine=setup["engine"])


def test_frame_looks_turn_with_the_meridian():
    """Off the central meridian grid north is not true north; on it, they are the same."""
    from slipsolve.sampling import frame_looks

    trace = FaultTrace(np.array([-118.0, -117.0]), np.array([36.0, 36.0]), name="ca")
    frame = trace.local_frame()
    x = np.array([0.0, 500e3])
    y = np.zeros(2)
    lx, ly, lu = frame_looks(frame, x, y, np.zeros(2), np.ones(2), np.array([0.5, 0.5]))
    assert abs(np.degrees(np.arctan2(lx[0], ly[0]))) < 0.05
    lon, lat = frame.to_lonlat(x[1], y[1])
    expected = (lon - frame.ref_lon) * np.sin(np.radians(lat))         # the convergence, to first order
    assert abs(abs(np.degrees(np.arctan2(lx[1], ly[1]))) - abs(expected)) < 0.1
    assert abs(expected) > 2.5
    assert np.allclose(np.hypot(lx, ly), 1.0)
    assert np.array_equal(lu, [0.5, 0.5])
