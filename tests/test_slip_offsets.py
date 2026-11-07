"""Pixel offsets as inversion data: a ramp kind per track, and a horizontal look vector.

A slant-range offset is the same projection as an interferogram's LOS, without the
2 pi ambiguity; an along-track offset projects the displacement on the flight
direction, which is horizontal. Either is a track like any other -- the forward
model is a dot product with whatever unit vector the stack carries -- but its
nuisance is an orbit or timing constant, not an interferogram's cubic, so the
ramp must be choosable per track.
"""

import numpy as np
import pytest
from slip_synthetic import _grid, _stack, forward_los_stack, tapered_slip

from slipsolve import FaultMesh, FaultTrace, Observations, SlipInversion, SlipModel
from slipsolve.greens import HalfSpaceTDE
from slipsolve.regularize import ramp_columns, ramp_spec

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])
_ALONG = np.array([np.sin(np.radians(-12.0)), np.cos(np.radians(-12.0)), 0.0])  # NNW, ascending


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0)
    insar = forward_los_stack(mesh, truth, trace, frame, geometry="asc", spacing=5000.0,
                              noise=0.004, seed=1)
    # an along-track offset track: the displacement on the flight direction, metres of
    # noise-free signal plus offset-like noise and an orbit constant
    x, y = _grid(frame, 32619, 140e3, 90e3, 5000.0)
    xx, yy = np.meshgrid(x, y)
    fx, fy = frame.from_epsg(xx.ravel(), yy.ravel(), 32619)
    along = (HalfSpaceTDE(0.25).forward(mesh, truth, fx, fy) @ _ALONG).reshape(xx.shape)
    along = along + 0.8 + np.random.default_rng(2).normal(0.0, 0.02, along.shape)
    offsets = _stack(along, x, y, tuple(_ALONG), 32619, "az", incidence=90.0)
    return trace, frame, mesh, truth, insar, offsets


def _obs(setup, **kwargs):
    trace, frame, _, _, insar, offsets = setup
    kw = dict(frame=frame, trace=trace, rms_min=0.006, width_min=6000.0, width_max=30000.0,
              exclude_within=5000.0)
    return Observations.concat([Observations.from_los(insar, name="asc", **kw),
                                Observations.from_los(offsets, name="az", **kw)], **kwargs)


def test_a_ramp_kind_per_track(setup):
    obs = _obs(setup)
    _, labels = ramp_columns(obs, {"asc": "quadratic", "az": "offset"})
    assert [l for l in labels if l.startswith("asc:")] == [
        "asc:offset", "asc:dx", "asc:dy", "asc:dxx", "asc:dxy", "asc:dyy"]
    assert [l for l in labels if l.startswith("az:")] == ["az:offset"]
    # "*" stands for the tracks a mapping does not name; "none" leaves a track bare
    assert ramp_columns(obs, {"*": "linear", "az": "none"})[1] == ["asc:offset", "asc:dx", "asc:dy"]
    # a single kind means that kind on every track, exactly
    np.testing.assert_array_equal(ramp_columns(obs, "linear")[0],
                                  ramp_columns(obs, {"*": "linear"})[0])


def test_a_ramp_mapping_is_checked(setup):
    obs = _obs(setup)
    mesh = setup[2]
    with pytest.raises(ValueError, match="do not have"):
        SlipInversion(mesh, obs, ramp={"asc": "linear", "az": "offset", "nope": "offset"})
    with pytest.raises(ValueError, match="no kind"):
        SlipInversion(mesh, obs, ramp={"asc": "linear"})
    with pytest.raises(ValueError, match="kind must be"):
        ramp_spec({"asc": "quartic"})


def test_an_along_track_track_inverts_beside_insar(setup, tmp_path):
    """A horizontal-look track with its own constant: the constant lands in its offset."""
    _, _, mesh, truth, _, _ = setup
    obs = _obs(setup, weights={"az": 0.2})
    inversion = SlipInversion(mesh, obs, ramp={"asc": "quadratic", "az": "offset"})
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-1.0, 1.0))
    assert model.converged
    offsets = dict(zip(model.ramp_labels, model.ramp))
    assert abs(offsets["az:offset"] - 0.8) < 0.02
    assert np.corrcoef(model.strike_slip, truth[:mesh.n_elements])[0, 1] > 0.9
    assert np.all(np.isfinite(obs.ds["look_u"].values))
    np.testing.assert_allclose(obs.ds["look_u"].values[obs.track_mask("az")], 0.0, atol=1e-12)

    path = model.save(tmp_path / "offsets.slip.zip")
    loaded = SlipModel.load(path)
    assert loaded.options["ramp"] == {"asc": "quadratic", "az": "offset"}
    np.testing.assert_array_equal(loaded.ramp_design, inversion.ramp)
