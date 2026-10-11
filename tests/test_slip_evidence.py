"""Pins the geometry evidence: the marginal likelihood with slip, ramp, scale and weight integrated out.

The eigen-decomposed formula is only worth having if it is the marginal likelihood
it claims to be, so the first test computes that likelihood the slow way -- the
dense ``N(d; Rb, sigma^2 (C + G (alpha^2 Q)^-1 G'))`` with the ramp integrated
under a flat prior -- and demands the two agree up to a constant that depends on
neither the geometry nor ``alpha^2``. That constant (``|C|``, ``|R'C^-1R|``,
``|Q|``) cancels in every comparison the sweep makes.
"""

import numpy as np
import pytest
import scipy.linalg as sla
import xarray as xr

from slipsolve import FaultMesh, FaultTrace, GeometryEvidence, Observations, smoothing_precision
from slipsolve.evidence import TrackWhitener, pinned_nodes, track_blocks


def _observations(n_per_track=(40, 30), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i, n in enumerate(n_per_track):
        name = f"t{i}"
        ds = xr.Dataset({
            "x": ("obs", rng.uniform(-50e3, 50e3, n)), "y": ("obs", rng.uniform(-30e3, 30e3, n)),
            "los": ("obs", rng.normal(size=n)), "look_e": ("obs", np.full(n, 0.6)),
            "look_n": ("obs", np.full(n, -0.1)), "look_u": ("obs", np.full(n, 0.79)),
            "std": ("obs", np.zeros(n)), "cell_size": ("obs", np.full(n, 2e3)),
            "count": ("obs", rng.integers(4, 60, n).astype(float)),
            "weight": ("obs", np.ones(n)), "track": ("obs", np.full(n, name, dtype=object))})
        ds.attrs.update(frame={"lon0": 0.0}, tracks=[name])
        parts.append(Observations(ds))
    return Observations.concat(parts, normalize="none")


def _dense(c, r, g, q, d, alpha2, jeffreys):
    """The marginal likelihood the slow way, up to a geometry-free constant."""
    sig = c + g @ np.linalg.solve(alpha2 * q, g.T)
    si = np.linalg.inv(sig)
    rsr = r.T @ si @ r
    pi = si - si @ r @ np.linalg.solve(rsr, r.T @ si)
    quad = float(d @ pi @ d)
    base = -0.5 * np.linalg.slogdet(sig)[1] - 0.5 * np.linalg.slogdet(rsr)[1]
    n, k = d.size, r.shape[1]
    return base - (0.5 * (n - k) * np.log(quad) if jeffreys else 0.5 * quad)


def test_the_evidence_is_the_dense_marginal_likelihood():
    obs = _observations()
    n = obs.ds.sizes["obs"]
    rng = np.random.default_rng(1)
    names, rows, _ = track_blocks(obs, None)
    blocks = {}
    for t, r in zip(names, rows):
        a = rng.normal(size=(r.size, r.size))
        blocks[t] = a @ a.T / r.size + 0.5 * np.eye(r.size)
    c = sla.block_diag(*[blocks[t] for t in names])
    ramp = np.zeros((n, 4))
    for i, r in enumerate(rows):
        ramp[r, 2 * i] = 1.0
        ramp[r, 2 * i + 1] = obs.ds["x"].values[r] / 1e5
    p = 8
    lq = rng.normal(size=(p + 3, p))
    q = lq.T @ lq + 0.1 * np.eye(p)
    free = np.arange(p)
    d = rng.normal(size=n)
    for key, jeffreys in (("log_evidence", True), ("log_evidence_fixed", False)):
        offsets = []
        for _ in range(3):
            g = rng.normal(size=(n, p))
            for alpha2 in (0.3, 2.0, 15.0):
                ev = GeometryEvidence(obs, q, free, ramp, covariance=blocks)
                ev.alpha2, ev.dlog = np.array([alpha2]), 1.0      # one grid point: no integral
                dh, s0 = ev.prepare(d)
                mine = ev.score(ev.decompose(g), dh, s0)[key][0]
                offsets.append(_dense(c, ramp, g, q, d, alpha2, jeffreys) - mine)
        assert np.ptp(offsets) < 1e-8, f"{key}: the difference must be one constant"


def test_the_whitener_applies_each_tracks_own_factor():
    rng = np.random.default_rng(2)
    rows = [np.arange(0, 5), np.arange(5, 9), np.arange(9, 12)]
    full = rng.normal(size=(4, 4))
    blocks = [None, full @ full.T + np.eye(4), np.array([0.5, 2.0, 4.0])]
    a = rng.normal(size=(12, 3))
    out = TrackWhitener(rows, blocks)(a)
    np.testing.assert_array_equal(out[rows[0]], a[rows[0]])
    np.testing.assert_allclose(sla.cholesky(blocks[1], lower=True) @ out[rows[1]], a[rows[1]])
    np.testing.assert_allclose(out[rows[2]], a[rows[2]] / np.sqrt(blocks[2])[:, None])


def test_the_prior_drops_the_pinned_nodes_and_is_positive_definite():
    trace = FaultTrace.from_lonlat(np.array([-69.0, -68.6, -68.2]), np.array([10.5, 10.52, 10.54]))
    mesh = FaultMesh.vertical(trace, trace.local_frame(), max_depth=15e3, edge_length=5e3)
    q, free = smoothing_precision(mesh, ss_ratio=1.0, ds_ratio=3.0)
    pinned = pinned_nodes(mesh, ("bottom", "left", "right"))
    assert free.size == 2 * int((~pinned).sum())
    assert not np.isin(np.flatnonzero(pinned), free).any()
    assert np.linalg.eigvalsh(q).min() > 0.0


def test_the_posterior_mean_and_ramp_are_the_penalised_least_squares_solution():
    obs = _observations(seed=3)
    n = obs.ds.sizes["obs"]
    rng = np.random.default_rng(4)
    ramp = np.column_stack([np.ones(n), obs.ds["x"].values / 1e5])
    p = 6
    lq = rng.normal(size=(p + 2, p))
    q = lq.T @ lq + 0.2 * np.eye(p)
    g = rng.normal(size=(n, p))
    d = rng.normal(size=n)
    ev = GeometryEvidence(obs, q, np.arange(p), ramp)
    dh, s0 = ev.prepare(d)
    dec = ev.decompose(g)
    alpha2 = np.array([0.7])
    m = ev.posterior_mean(dec, dh, alpha2)[:, 0]
    b = ev.ramp_coefficients(d - g @ m)[:, 0]
    # The same objective, solved directly over slip and ramp together.
    a = np.hstack([g, ramp])
    reg = np.zeros((p + 2, p + 2))
    reg[:p, :p] = alpha2[0] * q
    direct = np.linalg.solve(a.T @ a + reg, a.T @ d)
    np.testing.assert_allclose(m, direct[:p], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(b, direct[p:], rtol=1e-8, atol=1e-10)


def test_the_alpha_grid_is_fixed_by_the_first_geometry_and_brackets_the_mode():
    obs = _observations(seed=5)
    n = obs.ds.sizes["obs"]
    rng = np.random.default_rng(6)
    p = 6
    q = np.eye(p)
    g = rng.normal(size=(n, p))
    d = g @ rng.normal(size=p) + 0.1 * rng.normal(size=n)
    ev = GeometryEvidence(obs, q, np.arange(p), np.ones((n, 1)))
    dh, s0 = ev.prepare(d)
    first = ev.score(ev.decompose(g), dh, s0)
    grid = ev.alpha2.copy()
    ev.score(ev.decompose(3.0 * g), dh, s0)
    np.testing.assert_array_equal(ev.alpha2, grid)
    assert not first["alpha_edge"][0]
    assert 0.0 < first["rank"][0] < p


def test_an_unknown_covariance_spelling_is_refused():
    obs = _observations()
    with pytest.raises(ValueError, match="covariance must be"):
        track_blocks(obs, "diagonal")
