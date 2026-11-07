"""Pins regularization, the solver, and end-to-end slip recovery.

The acceptance test is :func:`test_recovers_a_planted_slip_patch`: plant a known
slip distribution on the real fault geometry, forward-model it through two
viewing geometries, add noise, quadtree it, invert, and require the answer back.
Everything else here guards a specific way that can go quietly wrong.
"""

import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

from slipsolve import (
    FaultMesh,
    FaultTrace,
    Observations,
    SlipInversion,
    ramp_content,
)
from slipsolve.inversion import _StackedOperator
from slipsolve.regularize import (
    neighbor_smoothing,
    ramp_columns,
    slip_bounds,
    zero_slip_boundary,
)

# The real San Sebastian trace, decimated: nearly east-west, which is the
# orientation that stresses the winding and boundary logic.
_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    return trace, frame, mesh


@pytest.fixture(scope="module")
def recovery(setup):
    """One shared forward-model-and-invert run; the rasters are slow to build."""
    trace, frame, mesh = setup
    truth = tapered_slip(mesh, peak=-2.0)
    obs = Observations.concat(
        [
            Observations.from_los(
                forward_los_stack(mesh, truth, trace, frame, geometry=g,
                                  spacing=5000.0, noise=0.004, nan_rows=3, seed=1),
                name=g, frame=frame, trace=trace,
                rms_min=0.006, width_min=6000.0, width_max=30000.0,
                exclude_within=5000.0,
            )
            for g in ("asc", "desc")
        ],
        normalize="sqrt_count",
    )
    inversion = SlipInversion(mesh, obs)
    return mesh, truth, obs, inversion


# -- regularization ----------------------------------------------------------

def test_smoothing_has_one_row_per_undirected_edge(setup):
    """Two rows per edge would be harmless but rescales the weight by sqrt(2)."""
    _, _, mesh = setup
    nb = mesh.neighbors
    n_edges = len({(min(i, int(j)), max(i, int(j)))
                   for i in range(mesh.n_elements) for j in nb[i] if j >= 0})
    smooth = neighbor_smoothing(nb)
    assert smooth.shape == (2 * n_edges, 2 * mesh.n_elements)
    # Each row is a single +1 / -1 difference.
    dense = smooth.toarray()
    np.testing.assert_allclose(dense.sum(axis=1), 0.0, atol=1e-12)
    assert np.all((dense != 0).sum(axis=1) == 2)


def test_smoothing_annihilates_uniform_slip(setup):
    """A constant slip field is perfectly smooth, so it must cost nothing."""
    _, _, mesh = setup
    smooth = neighbor_smoothing(mesh.neighbors, ss_ratio=1.0, ds_ratio=3.0)
    uniform = np.concatenate([np.full(mesh.n_elements, 2.0),
                              np.full(mesh.n_elements, -0.5)])
    np.testing.assert_allclose(smooth @ uniform, 0.0, atol=1e-12)


def test_dip_ratio_scales_only_the_dip_block(setup):
    _, _, mesh = setup
    smooth = neighbor_smoothing(mesh.neighbors, ss_ratio=1.0, ds_ratio=4.0)
    n_edges = smooth.shape[0] // 2
    ss_only = np.concatenate([np.arange(mesh.n_elements, dtype=float),
                              np.zeros(mesh.n_elements)])
    ds_only = np.concatenate([np.zeros(mesh.n_elements),
                              np.arange(mesh.n_elements, dtype=float)])
    a = smooth @ ss_only
    b = smooth @ ds_only
    np.testing.assert_allclose(b[n_edges:], 4.0 * a[:n_edges], atol=1e-12)


def test_zero_slip_boundary_is_sparse_and_targeted(setup):
    """Only the flagged elements get rows -- not a dense 2m x 2m diagonal per side."""
    _, _, mesh = setup
    w = zero_slip_boundary(mesh, sides=("bottom",), ratio=0.5)
    flagged = mesh.boundary_elements("bottom")
    assert w.shape == (2 * flagged.sum(), 2 * mesh.n_elements)
    dense = w.toarray()
    assert np.all((dense != 0).sum(axis=1) == 1)
    np.testing.assert_allclose(dense[dense != 0], 0.5)

    # Every constrained column belongs to a flagged element.
    touched = np.unique(np.nonzero(dense)[1]) % mesh.n_elements
    assert set(touched) == set(np.nonzero(flagged)[0])


def test_slip_bounds_polarity(setup):
    _, _, mesh = setup
    n = mesh.n_elements
    lo, hi = slip_bounds(n, strike=(-6.0, 6.0), dip=(-1.0, 1.0), polarity=(-1, 0, 0))
    # Right-lateral: strike-slip pinned non-positive, dip-slip untouched.
    np.testing.assert_allclose(hi[:n], 0.0)
    np.testing.assert_allclose(lo[:n], -6.0)
    np.testing.assert_allclose(lo[n:], -1.0)
    np.testing.assert_allclose(hi[n:], 1.0)

    lo, hi = slip_bounds(n, polarity=(1, 0))
    np.testing.assert_allclose(lo[:n], 0.0)

    with pytest.raises(ValueError, match="Bounds are empty"):
        slip_bounds(n, strike=(1.0, 6.0), polarity=(-1, 0))


def test_ramp_columns_are_per_track_and_scaled(recovery):
    """Nuisance columns must not leak between tracks or dominate conditioning."""
    _, _, obs, _ = recovery
    matrix, labels = ramp_columns(obs, "linear")
    assert matrix.shape == (obs.n, 3 * len(obs.tracks))
    assert labels[0].endswith(":offset")

    for i, name in enumerate(obs.tracks):
        block = matrix[:, 3 * i:3 * (i + 1)]
        other = ~obs.track_mask(name)
        np.testing.assert_allclose(block[other], 0.0)
    assert np.abs(matrix).max() <= 1.0

    assert ramp_columns(obs, "none")[0].shape == (obs.n, 0)
    with pytest.raises(ValueError, match="kind must be"):
        ramp_columns(obs, "quartic")


@pytest.mark.parametrize("kind, per_track, terms", [
    ("quadratic", 6, ["offset", "dx", "dy", "dxx", "dxy", "dyy"]),
    ("cubic", 10, ["offset", "dx", "dy", "dxx", "dxy", "dyy",
                   "dxxx", "dxxy", "dxyy", "dyyy"]),
])
def test_higher_degree_ramps_stay_per_track_and_nest(recovery, kind, per_track,
                                                     terms):
    """Total-degree columns, still confined to their track, still O(1)."""
    _, _, obs, _ = recovery
    matrix, labels = ramp_columns(obs, kind)
    n_track = len(obs.tracks)
    assert matrix.shape == (obs.n, per_track * n_track)
    assert [l.split(":")[1] for l in labels[:per_track]] == terms

    # No leakage across tracks. The products are zero off-track only because
    # every monomial carries at least one factor of cx or cy -- `cx**0` is one
    # *everywhere*, so a degree-0 term inside the loop would break this.
    for i, name in enumerate(obs.tracks):
        block = matrix[:, per_track * i:per_track * (i + 1)]
        np.testing.assert_allclose(block[~obs.track_mask(name)], 0.0)
    assert np.abs(matrix).max() <= 1.0
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]

    # Each degree strictly contains the one below: the lower columns are
    # identical, so raising `kind` cannot change what an existing coefficient
    # means or silently reorder a saved run's labels.
    lower_kind = {"quadratic": "linear", "cubic": "quadratic"}[kind]
    lower_per_track = {"linear": 3, "quadratic": 6}[lower_kind]
    lower, lower_labels = ramp_columns(obs, lower_kind)
    for i in range(n_track):
        hi = slice(per_track * i, per_track * i + lower_per_track)
        lo = slice(lower_per_track * i, lower_per_track * (i + 1))
        np.testing.assert_allclose(matrix[:, hi], lower[:, lo])
        assert labels[hi] == lower_labels[lo]


def test_higher_ramp_absorbs_a_curvature_that_linear_cannot(recovery):
    """The point of the option: a real quadratic nuisance has somewhere to go.

    A deep copy, because ``recovery`` is module-scoped and ``Dataset.copy()`` is
    shallow -- editing ``los`` in place would hand every later test a different
    scene than the one its numbers were measured on.
    """
    mesh, _, obs, _ = recovery
    curved = Observations(obs.ds.copy(deep=True))
    x = np.asarray(curved.ds["x"].values, dtype=float)
    cx = (x - x.mean()) / max(np.ptp(x), 1.0)
    # Centred so the curvature is not mostly a disguised offset, which the
    # offset column would absorb on its own and prove nothing.
    curved.ds["los"].values[:] += 0.30 * (cx ** 2 - float((cx ** 2).mean()))

    content = ramp_content(curved)
    assert content["curvature_only"] > 5.0
    assert (content["quadratic"]["variance_reduction"]
            > content["linear"]["variance_reduction"])

    quad = SlipInversion(mesh, curved, ramp="quadratic")
    lin = SlipInversion(mesh, curved, ramp="linear")
    assert quad.n_ramp == 2 * lin.n_ramp
    m_quad = quad.solve(smoothing=0.3, polarity=(-1, 0, 0))
    m_lin = lin.solve(smoothing=0.3, polarity=(-1, 0, 0))
    assert m_quad.variance_reduction > m_lin.variance_reduction


# -- the acceptance test -----------------------------------------------------

def test_recovers_a_planted_slip_patch(recovery):
    """Forward-model a known slip distribution and get it back.

    This is the test the whole module exists to pass. It exercises the mesh, the
    dislocation solution, the quadtree, the two-track combination, the
    regularization and the solver together -- and it would fail on any of the sign
    errors the other tests guard individually.
    """
    mesh, truth, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-1.0, 1.0))

    assert model.converged, f"solver status {model.result.status}"
    assert model.variance_reduction > 95.0

    truth_ss = truth[:mesh.n_elements]
    assert np.corrcoef(model.strike_slip, truth_ss)[0, 1] > 0.9

    # Right-lateral sense preserved, and the polarity bound respected.
    assert np.all(model.strike_slip <= 1e-9)
    assert model.strike_slip.min() < 0.6 * truth_ss.min()

    # Moment is the robust summary statistic; regularization biases peak slip more.
    truth_moment = 30e9 * np.sum(mesh.areas * np.abs(truth_ss))
    assert 0.7 < model.moment() / truth_moment < 1.4

    # No spurious slip at the far ends, where the planted patch has none.
    s = mesh.element_params[:, 0]
    far = np.abs(s - s.mean()) > 0.4 * (s.max() - s.min())
    assert np.abs(model.strike_slip[far]).mean() < 0.1


def test_a_noise_only_inversion_stays_near_zero(recovery):
    """Given nothing to explain, the model must not invent slip.

    The bound is regularization, not the slip limits -- a solution pressed against
    ``strike=(-6, 6)`` would mean the null space is not being controlled.
    """
    mesh, _, obs, _ = recovery
    noisy = Observations(obs.ds.copy(deep=True))
    rng = np.random.default_rng(7)
    noisy.ds["los"] = ("obs", rng.normal(0.0, 0.004, obs.n))

    model = SlipInversion(mesh, noisy).solve(
        smoothing=0.3, polarity=(-1, 0, 0), strike=(-6.0, 6.0), dip=(-1.0, 1.0)
    )
    assert model.max_slip < 0.5


# -- solver behaviour --------------------------------------------------------

def test_more_smoothing_gives_a_smoother_model(recovery):
    _, _, _, inversion = recovery
    rough = inversion.solve(smoothing=0.02, polarity=(-1, 0, 0))
    smooth = inversion.solve(smoothing=3.0, polarity=(-1, 0, 0))
    assert smooth.roughness < rough.roughness
    assert smooth.variance_reduction <= rough.variance_reduction + 1e-6


def test_l_curve_is_monotone_in_roughness(recovery):
    """Roughness must rise as smoothing falls, and convergence must be reported.

    The 0.05 point is deliberately included: on this problem it is the awkward
    one, needing ~236 trust-region iterations where 2.0, 0.3 and 0.01 each need
    ~30. It converges under the default cap and would be silently truncated under
    a tighter one.
    """
    _, _, _, inversion = recovery
    ds, models = inversion.l_curve([0.05, 0.3, 2.0], polarity=(-1, 0, 0))

    # Returned largest-smoothing-first, which is also the cheap-to-converge order.
    assert list(ds["smoothing"].values) == [2.0, 0.3, 0.05]
    assert len(models) == 3
    r = ds["roughness"].values
    assert np.all(np.diff(r) > 0), r          # rougher as smoothing decreases
    assert np.all(ds["converged"].values), ds["iterations"].values
    assert ds.attrs["n_parameters"] == inversion.n_param


def test_l_curve_takes_no_worker_pool(recovery):
    """`l_curve` is serial on purpose -- see its docstring.

    A `workers=` argument was written, verified to give identical results, and
    removed: a sweep is dominated by the one weight that runs to the iteration
    cap (67% of an 8-weight sweep, capping any scheduler at 1.50x) and scipy's
    pure-Python `lsmr` holds the GIL anyway, so it measured 1.02x at best and
    slower above. This pins that it stays out of the signature rather than
    getting reintroduced on the same reasoning.
    """
    _, _, _, inversion = recovery
    with pytest.raises(TypeError, match="workers"):
        inversion.l_curve([0.3, 2.0], workers=2, polarity=(-1, 0, 0))


def _count_matvecs(monkeypatch):
    """Tally products through the stacked operator; returns a one-element list."""
    n = [0]
    for name in ("_matvec", "_rmatvec"):
        original = getattr(_StackedOperator, name)

        def counted(self, v, _original=original, _n=n):
            _n[0] += 1
            return _original(self, v)

        monkeypatch.setattr(_StackedOperator, name, counted)
    return n


def test_lsmr_tol_auto_is_the_default_and_cuts_work(recovery, monkeypatch):
    """The shipped default must be the adaptive inner tolerance, and must reduce
    work versus scipy's ``None`` -- for the same model.

    Counted in **matrix-vector products, not outer iterations**. ``"auto"`` solves
    each sub-problem loosely, so it can take *more* outer steps while doing far
    less total work; ``nit`` is the wrong meter, and pinning it pinned a fixture
    accident instead of the property -- it broke when the fixtures moved to the
    left-looking geometry NISAR actually flies, with no change to the solver.

    Summed over two weights, because the win is **not uniform in the weight**.
    Measured on this fixture, ``None``/``"auto"`` matvecs are 2.18x at lam=1.0,
    1.93x at 0.5 and 1.91x at 0.3, but **0.70x at lam=2.0 and 0.78x at 0.05** --
    the adaptive tolerance loses at both ends. Over a whole sweep it still comes
    out ahead (1.17x here), which is how :meth:`l_curve` uses it.
    """
    _, _, _, inversion = recovery
    weights = (1.0, 0.3)

    n = _count_matvecs(monkeypatch)
    work, models = {}, {}
    for tol in ("auto", None):
        n[0] = 0
        models[tol] = [inversion.solve(smoothing=lam, polarity=(-1, 0, 0),
                                       lsmr_tol=tol) for lam in weights]
        work[tol] = n[0]

    assert models["auto"][0].options["lsmr_tol"] == "auto"
    assert all(m.converged for ms in models.values() for m in ms)
    assert work[None] > 1.2 * work["auto"], work

    # Same answer: the speed-up is in how the sub-problem is solved, not what for.
    for fast, slow in zip(models["auto"], models[None]):
        assert fast.variance_reduction == pytest.approx(slow.variance_reduction,
                                                        abs=0.5)
        np.testing.assert_allclose(fast.strike_slip, slow.strike_slip, atol=0.05)


# -- the active-set solver ---------------------------------------------------

def _stacked(inversion, smoothing, polarity=(-1, 0, 0)):
    """The solve's stacked system, dense, and its bounds: for an independent referee."""
    import scipy.sparse as sp

    from slipsolve.inversion import _boundary_rows
    from slipsolve.regularize import laplace_beltrami, scale_smoothing

    if inversion.basis.name == "element":
        smooth = neighbor_smoothing(inversion.mesh.neighbors, 1.0, 3.0,
                                    weighting="uniform", mesh=inversion.mesh)
    else:
        smooth = laplace_beltrami(inversion.mesh, 1.0, 3.0)
    boundary = _boundary_rows(inversion.basis, ("bottom", "left", "right"), 1.0)
    lo, hi = slip_bounds(inversion.basis.n_basis, (-10.0, 10.0), (-10.0, 10.0), polarity)
    k = inversion.n_ramp
    if k:
        smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], k))], format="csr")
        boundary = sp.hstack([boundary, sp.csr_matrix((boundary.shape[0], k))], format="csr")
        lo = np.concatenate([lo, np.full(k, -np.inf)])
        hi = np.concatenate([hi, np.full(k, np.inf)])
    design = np.hstack([inversion.g, inversion.ramp]) if k else inversion.g
    a = np.vstack([inversion.whiten(design), scale_smoothing(smooth, smoothing).toarray(),
                   boundary.toarray()])
    b = np.concatenate([inversion.whiten(inversion.d), np.zeros(a.shape[0] - inversion.obs.n)])
    return a, b, lo, hi


def _bvls(inversion, smoothing):
    """BVLS on the dense stacked system: exact, and without squaring the condition number."""
    from scipy.optimize import lsq_linear

    a, b, lo, hi = _stacked(inversion, smoothing)
    ref = lsq_linear(a, b, bounds=(lo, hi), method="bvls", tol=1e-13, max_iter=5000)
    # BVLS can also stop on a small change in cost (status 2), which is not exact.
    assert ref.status == 1, f"the referee did not converge exactly: {ref.message}"
    return ref.x


def test_box_qp_escapes_a_cycling_active_set():
    """Kernels with most variables on a bound make primal-dual active set cycle; the answer stays exact."""
    from scipy.optimize import lsq_linear

    from slipsolve.inversion import box_qp

    rng = np.random.default_rng(1)
    escaped = 0
    while escaped < 4:
        n = int(rng.integers(6, 20))
        m = int(rng.integers(n, 3 * n))
        scale = rng.uniform(0.5, 6.0) * n / 10
        design = 1.0 / (1.0 + ((rng.uniform(0, n, m)[:, None] - np.arange(n)[None, :]) / scale) ** 2)
        weight = 10.0 ** rng.uniform(-4, 0)
        smooth = np.diff(np.eye(n), 2, axis=0)
        data = design @ (rng.normal(size=n) * 2.0) + rng.normal(size=m) * 0.3
        lo, hi = np.zeros(n), np.ones(n)
        qp = box_qp(design.T @ design + weight * smooth.T @ smooth, design.T @ data, lo, hi, max_iter=2000)
        if not qp.escaped:
            continue
        escaped += 1
        referee = lsq_linear(np.vstack([design, np.sqrt(weight) * smooth]), np.r_[data, np.zeros(smooth.shape[0])],
                             bounds=(lo, hi), method="bvls", tol=1e-14, max_iter=20000)
        assert np.abs(qp.x - referee.x).max() < 1e-9
        assert qp.kkt < 1e-10


def test_box_qp_is_the_bounded_minimiser():
    """Against BVLS on the Cholesky factor, an independent exact method."""
    from scipy.linalg import cholesky
    from scipy.optimize import lsq_linear

    from slipsolve.inversion import box_qp

    rng = np.random.default_rng(3)
    a = rng.normal(size=(60, 25))
    N = a.T @ a + 0.1 * np.eye(25)
    g = 3.0 * a.T @ rng.normal(size=60)
    lo, hi = np.full(25, -0.5), np.full(25, 0.5)
    lo[:5], hi[:5] = -np.inf, np.inf        # unbounded, as the ramp columns are
    hi[5:10] = 0.0                          # one-signed, as under a polarity

    qp = box_qp(N, g, lo, hi)
    r = cholesky(N)                         # x'Nx/2 - g'x = |Rx - R^-T g|^2 / 2 + const
    ref = lsq_linear(r, np.linalg.solve(r.T, g), bounds=(lo, hi), method="bvls", tol=1e-14)
    assert (qp.at_lo | qp.at_hi).sum() >= 3, "the bounds must be doing something"
    np.testing.assert_allclose(qp.x, ref.x, atol=1e-10)
    assert qp.kkt < 1e-12 and 0.0 < qp.rcond <= 1.0
    # Started from its own answer, the active set is already right.
    assert box_qp(N, g, lo, hi, x0=qp.x).nit == 1


def test_active_set_is_the_exact_minimiser(setup, recovery):
    """The active-set answer is the minimiser; ``lsq_linear`` stops short of it.

    Measured on this fixture against BVLS on the stacked system: the active set
    agrees with it to 3e-13 m (element basis, lam 0.3) and 1e-10 m (nodes and a
    quadratic ramp, lam 0.002), while ``lsq_linear`` sits 0.36 mm and 48 mm away
    at a cost 1.1e-5 and 3.1e-6 higher, though it reports convergence.
    """
    _, _, mesh = setup
    _, _, obs, inversion = recovery
    nodes = SlipInversion(mesh, obs, basis="node", ramp="quadratic")
    for inv, lam in ((inversion, 0.3), (nodes, 0.002)):
        got = inv.solve(smoothing=lam, polarity=(-1, 0, 0), method="active_set")
        assert got.options["method"] == "active_set" and got.converged
        np.testing.assert_allclose(got.x, _bvls(inv, lam), atol=1e-8)


def test_active_set_agrees_with_lsq_linear(recovery, tmp_path):
    """The same problem: never a higher cost, the same slip, and the bounds held exactly."""
    _, _, _, inversion = recovery
    for lam in (2.0, 0.3):
        ref = inversion.solve(smoothing=lam, polarity=(-1, 0, 0))
        got = inversion.solve(smoothing=lam, polarity=(-1, 0, 0), method="active_set")
        assert got.converged and got.options["method"] == "active_set"
        assert got.result.cost <= ref.result.cost
        # Measured 0.04 / 0.36 mm, all of it lsq_linear stopping short.
        np.testing.assert_allclose(got.strike_slip, ref.strike_slip, atol=1e-3)
        np.testing.assert_allclose(got.dip_slip, ref.dip_slip, atol=1e-3)
        assert got.moment_magnitude == pytest.approx(ref.moment_magnitude, abs=1e-4)
        assert np.all(got.strike_slip <= 0.0)
    assert got.to_dataset().attrs["options"]["method"] == "active_set"


def test_a_refused_active_set_answer_falls_back_to_lsq_linear(recovery, monkeypatch):
    """A cycling or badly conditioned active set: the dense solve on small problems, ``lsq_linear`` on big ones.

    Small (up to ``EXACT_FALLBACK_MAX_PARAMS``): ``"lsq_linear_dense"``, at the minimiser to
    within its approach to the bounds (5e-7 m from BVLS here, 2e-11 m on the point test's
    problem; a trust region nears a bound asymptotically). Above it: the trust-region answer,
    bit for bit.
    """
    import slipsolve.inversion as module

    _, _, _, inversion = recovery
    ref = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    exact = _bvls(inversion, 0.3)

    def cycles(*_, **__):
        raise RuntimeError("the active set cycled after 3 steps")

    with monkeypatch.context() as patch:
        patch.setattr(module, "box_qp", cycles)
        with pytest.warns(RuntimeWarning, match="refused .*cycled.*lsq_linear_dense"):
            model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
        assert model.options["method"] == "lsq_linear_dense" and model.converged
        np.testing.assert_allclose(model.x, exact, atol=2e-6)
        patch.setattr(module, "EXACT_FALLBACK_MAX_PARAMS", 0)
        with pytest.warns(RuntimeWarning, match="refused .*cycled.*with lsq_linear instead"):
            model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
        assert model.options["method"] == "lsq_linear"
        np.testing.assert_array_equal(model.x, ref.x)

    with monkeypatch.context() as patch:
        patch.setattr(module, "ACTIVE_SET_MAX_COND", 1.0)
        with pytest.warns(RuntimeWarning, match="condition number"):
            model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
    assert model.options["method"] == "lsq_linear_dense"

    # Asked for by name, the dense solve is the minimiser too.
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="lsq_linear_dense")
    assert model.options["method"] == "lsq_linear_dense"
    np.testing.assert_allclose(model.x, exact, atol=2e-6)

    with pytest.raises(ValueError, match="method must be one of"):
        inversion.solve(smoothing=0.3, method="lbfgs")


def test_the_data_block_is_kept_and_follows_the_data(recovery):
    """Formed once across weights; formed again when the data change in place or the ramp does."""
    import copy

    _, _, _, fixture = recovery
    inversion = copy.copy(fixture)
    inversion.d = fixture.d.copy()
    inversion._normal_cache = None

    first = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
    block = inversion._normal_cache
    inversion.solve(smoothing=2.0, polarity=(-1, 0, 0), method="active_set")
    assert inversion._normal_cache is block

    # In place, as an edit of the observations' own `los` buffer would be.
    inversion.d += 0.01
    shifted = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
    assert inversion._normal_cache is not block
    ref = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    np.testing.assert_allclose(shifted.x, ref.x, atol=1e-3)
    assert np.abs(shifted.x - first.x).max() > 0.01

    # A copy with another ramp shares G but not the block.
    other = inversion.with_ramp("offset")
    got = other.solve(smoothing=0.3, polarity=(-1, 0, 0), method="active_set")
    assert got.ramp.size == 2 and other._normal_cache is not inversion._normal_cache
    ref = other.solve(smoothing=0.3, polarity=(-1, 0, 0))
    np.testing.assert_allclose(got.x, ref.x, atol=1e-3)


def test_l_curve_with_the_active_set(recovery):
    _, _, _, inversion = recovery
    ref, _ = inversion.l_curve([0.3, 2.0], polarity=(-1, 0, 0))
    got, models = inversion.l_curve([0.3, 2.0], polarity=(-1, 0, 0), method="active_set")
    assert all(m.options["method"] == "active_set" for m in models)
    assert np.all(got["converged"].values)
    np.testing.assert_allclose(got["rms_misfit"].values, ref["rms_misfit"].values, rtol=1e-4)
    np.testing.assert_allclose(got["roughness"].values, ref["roughness"].values, rtol=1e-3)


def test_ramp_absorbs_an_added_offset(recovery):
    """A constant added to one track must land in the nuisance column, not in slip."""
    mesh, _, obs, plain = recovery
    shifted = Observations(obs.ds.copy(deep=True))
    mask = shifted.track_mask("asc")
    shifted.ds["los"].values[mask] += 0.05

    without = SlipInversion(mesh, shifted).solve(smoothing=0.3, polarity=(-1, 0, 0))
    with_ramp = SlipInversion(mesh, shifted, ramp="offset").solve(
        smoothing=0.3, polarity=(-1, 0, 0)
    )
    assert with_ramp.variance_reduction > without.variance_reduction
    # The recovered offset should be close to what was injected.
    offsets = dict(zip(with_ramp.ramp_labels, with_ramp.ramp))
    assert abs(offsets["asc:offset"] - 0.05) < 0.03


def test_frame_mismatch_is_refused(setup, recovery):
    from slipsolve import LocalFrame

    _, _, mesh = setup
    _, _, obs, _ = recovery
    wrong = Observations(obs.ds.copy(deep=True))
    wrong.ds.attrs["frame"] = LocalFrame(0.0, 0.0).to_dict()
    with pytest.raises(ValueError, match="different LocalFrame"):
        SlipInversion(mesh, wrong)


# -- outputs -----------------------------------------------------------------

def test_to_dataset_and_text(recovery, tmp_path):
    mesh, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))

    ds = model.to_dataset()
    assert ds.sizes["element"] == mesh.n_elements
    assert ds.attrs["converged"] == 1
    np.testing.assert_allclose(ds["slip"].values,
                               np.hypot(model.strike_slip, model.dip_slip))

    path = model.to_text(tmp_path / "model.txt")
    table = np.loadtxt(path, skiprows=1)
    assert table.shape == (mesh.n_elements, 10)
    np.testing.assert_allclose(table[:, 6], model.strike_slip, rtol=1e-6)
    np.testing.assert_allclose(table[:, 8], mesh.areas, rtol=1e-9)
    # Longitude and latitude, not local metres.
    assert -180.0 < table[:, 1].min() and table[:, 1].max() < 180.0


def test_vertex_tables_rebuild_the_element_table(recovery, tmp_path):
    """The load-bearing check on the vertex pair: the join has to close.

    Averaging each triangle's three nodal slips must give back the element value
    ``slip_model.txt`` reports. That single assertion pins the connectivity
    indexing, the 1-based offset and the column order all at once -- an off-by-one
    in any of them scatters slip onto the wrong corners and still writes a
    plausible-looking file.

    A **nodal** model, because there the identity is exact by construction: the
    element value simply *is* the mean of its three nodes. For an element model the
    node values come from a lossy scatter and the round trip smooths, so what
    survives there is the componentwise integral, tested separately below.
    """
    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    node_path, element_path = model.to_vertex_text(tmp_path)
    assert node_path.name == "vert_nodes.txt"
    assert element_path.name == "vert_elements.txt"

    nodes = np.loadtxt(node_path, skiprows=1)
    elements = np.loadtxt(element_path, skiprows=1, dtype=int)
    assert nodes.shape == (mesh.n_nodes, 9)
    assert elements.shape == (mesh.n_elements, 4)

    # 1-based throughout, and every index addresses a row of the node table.
    np.testing.assert_array_equal(elements[:, 0], np.arange(1, mesh.n_elements + 1))
    assert elements[:, 1:].min() == 1
    assert elements[:, 1:].max() == mesh.n_nodes

    connectivity = elements[:, 1:] - 1
    for column, component in ((5, 0), (6, 1)):
        np.testing.assert_allclose(nodes[:, column][connectivity].mean(axis=1),
                                   model.element_slip[:, component], atol=1e-9)

    # Element ids join row for row with the element table.
    table = np.loadtxt(model.to_text(tmp_path / "slip_model.txt"), skiprows=1)
    np.testing.assert_array_equal(elements[:, 0], table[:, 0].astype(int))
    # Geographic, not local metres; depth negative-down like `to_text`'s column 4.
    assert -180.0 < nodes[:, 1].min() and nodes[:, 1].max() < 180.0
    assert nodes[:, 3].max() == 0.0 and nodes[:, 3].min() < 0.0

    # The rows partition the LUMPED moment exactly -- the rule the file's columns
    # are. The default moment integrates over the tent field instead.
    table_moment = np.sum(nodes[:, 8] * nodes[:, 7] * np.hypot(nodes[:, 5], nodes[:, 6]))
    assert table_moment == pytest.approx(model.moment(rule="lumped"), rel=1e-9)


def test_nodal_slip_conserves_the_slip_integral_componentwise(setup, recovery):
    """The element -> node scatter is area-weighted, so it conserves each component.

    Exactly, because ``P``'s rows sum to one and ``lumped_areas == P.T @ areas``.
    It does **not** conserve the sum of slip *magnitudes* -- an element value is
    the mean of three vectors, which is shorter than the mean of their lengths --
    so the two differ by about a percent and neither is a check on the other.
    """
    from slipsolve.basis import NodeBasis

    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, ramp="linear", basis="element").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    nodal = model.nodal_slip()
    assert nodal.shape == (mesh.n_nodes, 2)
    lumped = NodeBasis(mesh).lumped_areas()
    for component in (0, 1):
        np.testing.assert_allclose((lumped * nodal[:, component]).sum(),
                                   (mesh.areas * model.element_slip[:, component]).sum(),
                                   rtol=1e-12)


def test_nodal_slip_is_the_parameter_vector_for_a_nodal_model(recovery):
    """No transformation at all when the nodes are what was solved for."""
    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, ramp="linear", basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    nodal = model.nodal_slip()
    np.testing.assert_array_equal(nodal[:, 0], model.strike_slip)
    np.testing.assert_array_equal(nodal[:, 1], model.dip_slip)


def test_unconverged_models_refuse_to_export(recovery, tmp_path):
    """An iteration-capped solve has a meaningless fit; it must not be written out."""
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), max_iter=1)
    if model.converged:
        pytest.skip("solver converged in a single iteration; nothing to assert")
    assert "UNCONVERGED" in repr(model)
    with pytest.raises(ValueError, match="unconverged"):
        model.to_text(tmp_path / "bad.txt")
    with pytest.raises(ValueError, match="unconverged"):
        model.to_vertex_text(tmp_path)


def test_persist_round_trip(recovery, tmp_path):
    from slip_synthetic import Workspace

    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    ws = Workspace(tmp_path / "ws")
    stored = model.persist(ws, "slip")
    np.testing.assert_allclose(stored["strike_slip"].values, model.strike_slip)
    assert stored.attrs["mesh_digest"] == inversion.mesh.digest()


def test_save_load_keeps_per_row_smoothing(recovery, tmp_path):
    """``solve`` takes one smoothing weight per row; the saved model must reopen with them (``load`` read a float)."""
    from slipsolve.inversion import SlipModel

    _, _, _, inversion = recovery
    rows = neighbor_smoothing(inversion.mesh.neighbors, 1.0, 3.0, weighting="uniform",
                              mesh=inversion.mesh).shape[0]
    weights = np.full(rows, 0.3)
    weights[: rows // 2] = 0.1
    model = inversion.solve(smoothing=weights, polarity=(-1, 0, 0))
    path = tmp_path / "per_row.slip.zip"
    model.save(path)
    back = SlipModel.load(path)
    np.testing.assert_allclose(back.smoothing, weights)
    np.testing.assert_allclose(back.strike_slip, model.strike_slip)


# -- recovery on a dipping fault ---------------------------------------------

@pytest.fixture(scope="module")
def dipping_recovery():
    """The phase-two acceptance run: a 70-degree fault with genuine dip-slip.

    Two things are new relative to :func:`recovery`, and both matter. The mesh
    dips, so every element's Green's function comes out of the general branch of
    the triangular-dislocation solution rather than the exactly-vertical one; and
    the planted patch has ``rake=25``, so the dip-slip half of the design matrix
    is actually excited. With a pure strike-slip patch on a vertical fault -- the
    phase-one fixture -- half the columns could be wrong and the recovery test
    would still pass.
    """
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                            max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0, rake=25.0)
    obs = Observations.concat(
        [
            Observations.from_los(
                forward_los_stack(mesh, truth, trace, frame, geometry=g,
                                  spacing=5000.0, noise=0.004, nan_rows=3, seed=1),
                name=g, frame=frame, trace=trace,
                rms_min=0.006, width_min=6000.0, width_max=30000.0,
                exclude_within=5000.0,
            )
            for g in ("asc", "desc")
        ],
        normalize="sqrt_count",
    )
    return mesh, truth, SlipInversion(mesh, obs)


def test_recovers_a_planted_patch_on_a_dipping_fault(dipping_recovery):
    """The phase-two gate, held to the same bar as the vertical one."""
    mesh, truth, inversion = dipping_recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-3.0, 3.0))

    assert model.converged, f"solver status {model.result.status}"
    assert model.variance_reduction > 95.0

    n = mesh.n_elements
    assert np.corrcoef(model.strike_slip, truth[:n])[0, 1] > 0.9
    truth_moment = 30e9 * np.sum(mesh.areas * np.hypot(truth[:n], truth[n:]))
    assert 0.7 < model.moment() / truth_moment < 1.4

    s = mesh.element_params[:, 0]
    far = np.abs(s - s.mean()) > 0.4 * (s.max() - s.min())
    assert np.abs(model.slip_magnitude[far]).mean() < 0.1


def test_dip_slip_is_recovered_not_just_absorbed(dipping_recovery):
    """The half of the model a vertical, pure-strike-slip fixture never tests.

    Correlation, not amplitude: regularization biases dip-slip more than
    strike-slip because it is the weaker of the two signals here and carries
    ``ds_ratio`` times the smoothing.
    """
    mesh, truth, inversion = dipping_recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-3.0, 3.0))

    n = mesh.n_elements
    truth_ds = truth[n:]
    assert np.abs(truth_ds).max() > 0.5, "the fixture must actually plant dip-slip"
    assert np.corrcoef(model.dip_slip, truth_ds)[0, 1] > 0.8
    # Sign preserved: rake 25 on a negative peak puts dip-slip negative too.
    assert model.dip_slip[np.argmax(np.abs(truth_ds))] < 0


# -- surface displacement export ---------------------------------------------

def test_surface_displacement_grid(recovery):
    """The full three-component field, on the mesh's own frame.

    Line of sight collapses three components into one; what an inversion actually
    recovers is the vector, so being able to look at it is the point.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=5000.0, pad=40e3)

    assert set(grid.data_vars) == {"ux", "uy", "uz"}
    assert grid.rio.crs is not None, "the grid must be georeferenced to be exportable"
    assert grid.y[0] > grid.y[-1], "north-up, like every other raster here"
    for name in ("ux", "uy", "uz"):
        assert np.isfinite(grid[name].values).mean() > 0.8
        assert grid[name].attrs["units"] == "m"


def test_surface_displacement_is_nan_on_the_trace(recovery):
    """A dislocation solution is singular where the fault meets the surface.

    Returning a very large number there would be worse than returning nothing:
    it would set the colour scale of every plot of the field.
    """
    mesh, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=4000.0, pad=20e3, exclude_within=8000.0)

    top = mesh.nodes[mesh.params[:, 1] == mesh.params[:, 1].max()]
    xx, yy = np.meshgrid(grid.x.values, grid.y.values)
    distance = np.sqrt(np.min((xx.ravel()[:, None] - top[None, :, 0]) ** 2
                              + (yy.ravel()[:, None] - top[None, :, 1]) ** 2, axis=1))
    near = (distance <= 8000.0).reshape(xx.shape)

    assert near.any(), "the test needs some grid points on the fault"
    assert np.all(np.isnan(grid["ux"].values[near]))
    assert np.isfinite(grid["ux"].values[~near]).all()


def test_surface_displacement_matches_forward_point_by_point(recovery):
    """The gridded field and ``forward`` must be the same calculation.

    They are not the same code path -- the grid evaluates in blocks and masks the
    near field -- so this is what pins the blocking as free.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=8000.0, pad=30e3, block=64)

    xx, yy = np.meshgrid(grid.x.values, grid.y.values)
    finite = np.isfinite(grid["ux"].values)
    pick = np.nonzero(finite.ravel())[0][::37][:20]
    expected = model.forward(xx.ravel()[pick], yy.ravel()[pick])

    for i, name in enumerate(("ux", "uy", "uz")):
        np.testing.assert_allclose(grid[name].values.ravel()[pick], expected[:, i],
                                   rtol=1e-5, atol=1e-9)


def test_strike_slip_moves_the_ground_sideways_not_up(recovery):
    """Physical sanity, and a check the components are not swapped.

    A pure strike-slip patch on this near-east-west fault produces horizontal
    displacement an order of magnitude larger than vertical, and mostly east-west.
    If ``ux``/``uy``/``uz`` were permuted this would fail immediately.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), dip=(-0.01, 0.01))
    grid = model.surface_displacement(spacing=5000.0, pad=40e3)

    east = np.nanmax(np.abs(grid["ux"].values))
    north = np.nanmax(np.abs(grid["uy"].values))
    up = np.nanmax(np.abs(grid["uz"].values))
    assert east > 3 * north, "an east-west fault slips mostly east-west"
    assert min(east, north) > 3 * up, "strike-slip is not a vertical motion"


def test_to_grd_writes_readable_single_variable_grids(recovery, tmp_path):
    import xarray as xr

    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    paths = model.to_grd(tmp_path, spacing=8000.0, pad=30e3)

    assert [p.name for p in paths] == ["ux.grd", "uy.grd", "uz.grd"]
    for path in paths:
        field = xr.open_dataarray(path)
        assert field.name == "z"
        assert field.dims == ("lat", "lon")
        assert np.isfinite(field.values).any()
        field.close()

    with pytest.raises(KeyError, match="Unknown field"):
        model.to_grd(tmp_path, fields=("ue",), spacing=20000.0, pad=20e3)
