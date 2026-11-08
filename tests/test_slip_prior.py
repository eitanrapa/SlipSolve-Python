"""A nuisance prior with a mean.

The oracle: on an unbounded nuisance column, a prior `(mean, sigma)` is the zero-mean prior on
the data less `mean` times that column, with `mean` added back to the coefficient. That holds
in each solve method, in blocked cross-validation, in GCV and in the bounded re-solves.
"""

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace, Observations, SlipInversion, monte_carlo
from slipsolve.resolution import blocked_cross_validation, generalized_cross_validation

_SIGMA = 0.004
_MEAN = 0.03          # metres: the offset the prior ties the first track to
_SOLVE = dict(smoothing=3e3, boundary_ratio=3e3, polarity=(-1, 0, 0), strike=(-5.0, 5.0), dip=(-1.0, 1.0))


@pytest.fixture(scope="module")
def pair():
    """The same problem twice: a prior with a mean, and the zero-mean prior on shifted data."""
    trace = FaultTrace(np.array([-68.1, -67.4]), np.array([10.55, 10.60]), name="toy")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=8e3)
    rng = np.random.default_rng(7)
    x, y = rng.uniform(-70e3, 70e3, 400), rng.uniform(-50e3, 50e3, 400)
    keep = trace.distance(x, y, frame) > 3e3
    x, y = x[keep], y[keep]
    parts = []
    for name, look in (("A", (-0.56, -0.10, 0.82)), ("D", (0.60, -0.10, 0.79))):
        look = np.asarray(look) / np.linalg.norm(look)
        parts.append(Observations.from_points(x, y, np.zeros(x.size), look, _SIGMA, name=name, frame=frame, epsg=None))
    probe = SlipInversion(mesh, Observations.concat(parts, normalize="none"), basis="node", ramp="linear")
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    patch = -1.0 * np.exp(-((s - s.mean()) / 18e3) ** 2) * np.exp(-((z + 6e3) / 5e3) ** 2)
    planes = rng.normal(0.0, 0.02, probe.n_ramp)
    values = (probe.g @ np.r_[patch, np.zeros(mesh.n_nodes)] + probe.ramp @ planes
              + rng.normal(0.0, _SIGMA, probe.obs.n))
    j = probe.ramp_labels.index("A:offset")
    observed = Observations(probe.obs.ds.assign(los=("obs", values)))
    shifted = Observations(probe.obs.ds.assign(los=("obs", values - _MEAN * probe.ramp[:, j])))
    common = dict(basis="node", ramp="linear", g=probe.g)
    return dict(
        with_mean=SlipInversion(mesh, observed, nuisance_prior={"A:offset": (_MEAN, 5e-4)}, **common),
        zero_mean=SlipInversion(mesh, shifted, nuisance_prior={"A:offset": 5e-4}, **common),
        unshifted=SlipInversion(mesh, observed, nuisance_prior={"A:offset": 5e-4}, **common),
        j=probe.n_slip + j, mesh=mesh, common=common)


def _shift(x, j):
    out = np.array(x, dtype=float)
    out[..., j] += _MEAN
    return out


@pytest.mark.parametrize("method, tolerance", [("active_set", 1e-9), ("lsq_linear_dense", 1e-6), ("lsq_linear", 1e-4)])
def test_a_mean_is_the_zero_mean_prior_on_shifted_data(pair, method, tolerance):
    a = pair["with_mean"].solve(method=method, **_SOLVE)
    b = pair["zero_mean"].solve(method=method, **_SOLVE)
    assert a.options["method"] == method
    assert np.abs(a.x - _shift(b.x, pair["j"])).max() < tolerance
    plain = pair["unshifted"].solve(method=method, **_SOLVE)
    assert abs(a.x[pair["j"]] - plain.x[pair["j"]]) > 2e-3, "the mean moves the coefficient"


def test_a_tight_prior_holds_the_coefficient_at_its_mean(pair):
    inv = pair["with_mean"]
    tight = SlipInversion(inv.mesh, inv.obs, nuisance_prior={"A:offset": (_MEAN, 1e-7)}, **pair["common"])
    assert abs(tight.solve(method="active_set", **_SOLVE).x[pair["j"]] - _MEAN) < 1e-6


def test_cross_validation_and_gcv_score_the_shifted_problem(pair):
    kwargs = dict(method="active_set", **{k: v for k, v in _SOLVE.items() if k != "smoothing"})
    a = blocked_cross_validation(pair["with_mean"], _SOLVE["smoothing"], n_folds=4, **kwargs)
    b = blocked_cross_validation(pair["zero_mean"], _SOLVE["smoothing"], n_folds=4, **kwargs)
    for key in ("cv_rms", "cv_rms_profiled"):
        assert a[key] == pytest.approx(b[key], rel=1e-9)
    ga = generalized_cross_validation(pair["with_mean"], _SOLVE["smoothing"], boundary_ratio=_SOLVE["boundary_ratio"])
    gb = generalized_cross_validation(pair["zero_mean"], _SOLVE["smoothing"], boundary_ratio=_SOLVE["boundary_ratio"])
    gu = generalized_cross_validation(pair["unshifted"], _SOLVE["smoothing"], boundary_ratio=_SOLVE["boundary_ratio"])
    assert ga["gcv"] == pytest.approx(gb["gcv"], rel=1e-9)
    assert abs(ga["gcv"] - gu["gcv"]) > 1e-6 * gu["gcv"], "GCV sees the mean"


def test_bounded_re_solves_keep_the_mean(pair):
    a = pair["with_mean"].solve(method="active_set", **_SOLVE)
    b = pair["zero_mean"].solve(method="active_set", **_SOLVE)
    copies_a = monte_carlo(a, n=3, noise="white", seed=2)
    copies_b = monte_carlo(b, n=3, noise="white", seed=2)
    assert np.abs(copies_a.x - _shift(copies_b.x, pair["j"])).max() < 1e-7


def test_prior_values(pair):
    inv, common, j = pair["with_mean"], pair["common"], pair["j"]

    def make(prior):
        return SlipInversion(inv.mesh, inv.obs, nuisance_prior=prior, **common)

    assert make({"A:offset": 0.01}).nuisance_prior == {"A:offset": 0.01}
    assert make({"A:offset": [0.0, 0.01]}).nuisance_prior == {"A:offset": 0.01}, "a zero mean is a bare sigma"
    listed = make({"A:offset": [0.02, 0.01]})                    # as a saved model's options give it back
    assert listed.nuisance_prior == {"A:offset": (0.02, 0.01)}
    mean = listed.prior_mean()
    assert mean[j] == 0.02 and np.count_nonzero(mean) == 1
    # The first pattern that matches a label sets its prior.
    first = make({"A:off*": (0.02, 0.01), "A:*": 0.5})
    assert first.prior_rows().shape[0] == 3
    assert np.count_nonzero(first.prior_mean()) == 1 and first.prior_mean()[j] == 0.02
    assert sorted(first.prior_rows().data) == [2.0, 2.0, 100.0]
    for bad in ((0.02, 0.0), (0.02, -1.0), (0.0, 1.0, 2.0), (np.nan, 1.0), (0.0, np.inf)):
        with pytest.raises(ValueError):
            make({"A:offset": bad})
