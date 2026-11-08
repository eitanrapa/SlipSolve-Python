"""Production stages for the modes: model-driven sampling and adaptive meshes on an interseismic fit and on a series."""

import numpy as np
import pytest
from slip_synthetic import _grid, look_vector

from slipsolve import (EulerPole, FaultMesh, FaultTrace, HalfSpaceTDE, InterseismicInversion, Observations, TimeBasis, TimeDependentInversion, TrackSeries,
                       iterate_sampling, predicted_los, side_labels)
from slipsolve.interseismic import _axes_at
from slipsolve.trace import TraceSet

EPSG = 32611
RATE = 30e-3
LOCKED = 15e3


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(np.array([-119.5, -118.5]), np.array([36.0, 36.0]), name="f")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=30e3, edge_length=8e3)
    # Two translating blocks, locked to 15 km in the middle third, creeping outside it.
    axis = _axes_at([frame.origin_lon, frame.origin_lat])[0]
    poles = {"north": EulerPole(axis * RATE / 2.0), "south": EulerPole(axis * -RATE / 2.0)}
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    length = s.max()
    coupling = ((s > length / 3.0) & (s < 2.0 * length / 3.0) & (z > -LOCKED)).astype(float)
    return dict(trace=trace, frame=frame, mesh=mesh, poles=poles, coupling=coupling, length=length)


def _velocity_field(setup, x, y, look):
    """The planted field along `look` at points: rigid blocks less the elastic effect of the locked part."""
    trace, frame, mesh, poles = (setup[k] for k in ("trace", "frame", "mesh", "poles"))
    lon, lat = frame.to_lonlat(x, y)
    north = trace.side(x, y, frame) > 0
    east = np.where(north, poles["north"].velocity(lon, lat)[0], poles["south"].velocity(lon, lat)[0])
    northward = np.where(north, poles["north"].velocity(lon, lat)[1], poles["south"].velocity(lon, lat)[1])
    # The block on the trace's left moves east, which is right-lateral on this west-to-east trace: negative in the
    # package's sign, so the back-slip is positive.
    backslip = RATE * setup["coupling"][mesh.triangles].mean(axis=1)
    enu = HalfSpaceTDE(0.25).forward(mesh, np.r_[backslip, np.zeros(backslip.size)], x, y)
    return (east + enu[:, 0]) * look[0] + (northward + enu[:, 1]) * look[1] + enu[:, 2] * look[2]


@pytest.fixture(scope="module")
def grid(setup):
    from slip_synthetic import _stack

    frame = setup["frame"]
    x, y = _grid(frame, EPSG, 90e3, 60e3, 2000.0)
    xx, yy = np.meshgrid(x, y)
    fx, fy = frame.from_epsg(xx.ravel(), yy.ravel(), EPSG)
    look = np.array(look_vector(35.0, -12.0))
    keep = setup["trace"].distance(fx, fy, frame) > 500.0
    values = np.full(fx.size, np.nan)
    values[keep] = _velocity_field(setup, fx[keep], fy[keep], look)
    return _stack(values.reshape(xx.shape), x, y, look, EPSG, "S1")


def _gnss(setup):
    frame = setup["frame"]
    rng = np.random.default_rng(4)
    x, y = rng.uniform(-90e3, 90e3, 30), rng.uniform(-60e3, 60e3, 30)
    keep = setup["trace"].distance(x, y, frame) > 3e3
    x, y = x[keep], y[keep]
    rows = []
    for name, look in (("gnss_e", (1.0, 0.0, 0.0)), ("gnss_n", (0.0, 1.0, 0.0))):
        values = _velocity_field(setup, x, y, look)
        rows.append(Observations.from_points(x, y, values, np.tile(look, (x.size, 1)), 1e-3, name=name, frame=frame, epsg=None))
    return Observations.concat(rows, normalize="none")


def _make(setup):
    def make(mesh, obs):
        blocks = side_labels(obs, setup["trace"], "north", "south")
        return InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"), engine=HalfSpaceTDE(0.25), ramp={"gnss_e": "none", "gnss_n": "none", "*": "linear"})
    return make


KWARGS = dict(rms_min=1e-3, width_min=4e3, width_max=30e3, exclude_within=2e3)


def test_the_interseismic_mode_goes_round_the_loop(setup, grid):
    gnss = _gnss(setup)
    obs, model, history = iterate_sampling({"S1": grid}, setup["mesh"], setup["trace"], setup["frame"], {"S1": KWARGS}, make_inversion=_make(setup),
                                           fixed=[gnss], normalize="none", solve_kwargs=dict(smoothing=1e4, rate=(-RATE, 0.0)), max_rounds=2, verbose=False, tol=0.0)
    assert "gnss_e" in obs.tracks and "S1" in obs.tracks
    assert len(history) == 3 and "fields" in history[-1] and "fields" not in history[0] and "fields" not in history[-2]
    # GNSS goes through untouched.
    assert int(np.sum(np.asarray(obs.ds["track"].values).astype(str) == "gnss_e")) == gnss.ds.sizes["obs"] // 2
    # The first round's samples follow the data, which rotate, and the later ones the back-slip: closer to the locked part.
    first, last = history[0], history[-1]
    assert first["n"] != last["n"]
    locked = (setup["coupling"] > 0.5)
    got = model.coupling[locked]
    assert np.nanmedian(got) > 0.6 and np.nanmedian(model.coupling[(~locked) & (setup["mesh"].params[:, 1] > -10e3)]) < 0.4


def test_inversion_kwargs_and_make_inversion_exclude_each_other(setup, grid):
    with pytest.raises(ValueError, match="not both"):
        iterate_sampling({"S1": grid}, setup["mesh"], setup["trace"], setup["frame"], {"S1": KWARGS}, make_inversion=_make(setup), inversion_kwargs={"ramp": "linear"})


def test_a_network_adapts_sheet_by_sheet(setup, grid):
    """Two sheets, each remeshed on its own trace, each keeping its own budget, and samples kept off both traces."""
    frame = setup["frame"]
    other = FaultTrace(np.array([-119.3, -118.7]), np.array([36.45, 36.45]), name="g")
    meshes = [FaultMesh.vertical(setup["trace"], frame, max_depth=30e3, edge_length=8e3), FaultMesh.vertical(other, frame, max_depth=30e3, edge_length=8e3)]
    joined = FaultMesh.join(meshes)
    sheets = joined.sheets()
    assert [s.n_nodes for s in sheets] == [m.n_nodes for m in meshes] and np.array_equal(sheets[1].nodes, meshes[1].nodes) and np.array_equal(sheets[1].triangles, meshes[1].triangles)
    assert meshes[0].sheets() == [meshes[0]]

    traces = TraceSet([setup["trace"], other])
    assert np.array_equal(traces.distance(np.array([0.0]), np.array([1e3]), frame), np.minimum(setup["trace"].distance(np.array([0.0]), np.array([1e3]), frame), other.distance(np.array([0.0]), np.array([1e3]), frame)))
    with pytest.raises(ValueError, match="needs a trace"):
        TraceSet([])

    def make(mesh, obs):
        blocks = np.where(setup["trace"].side(obs.ds["x"].values, obs.ds["y"].values, frame) > 0, "north", "south")
        return InterseismicInversion(mesh, obs, blocks=blocks, faults=[("north", "south"), ("north", "north")], engine=HalfSpaceTDE(0.25), ramp="linear")

    vertical = dict(uniform_dip=90.0)
    adapt = dict(sheets=[dict(trace=setup["trace"], geometry=vertical), dict(trace=other, geometry=vertical)], size_from="kernel", rounds=1)
    with pytest.raises(ValueError, match="sheets"):
        iterate_sampling({"S1": grid}, joined, traces, frame, {"S1": KWARGS}, make_inversion=make, normalize="none", solve_kwargs=dict(smoothing=1e4), mesh_adapt=dict(sheets=[dict(trace=other)]), max_rounds=1, verbose=False)
    obs, model, history = iterate_sampling({"S1": grid}, joined, traces, frame, {"S1": KWARGS}, make_inversion=make, normalize="none", solve_kwargs=dict(smoothing=1e4, rate=(-RATE, 0.0)),
                                           mesh_adapt=adapt, max_rounds=2, verbose=False, tol=0.0)
    new = model.mesh.sheets()
    assert len(new) == 2 and model.mesh.attrs["n_faults"] == 2
    assert not np.array_equal(new[0].nodes, meshes[0].nodes), "the sheet was remeshed"
    for sheet, start in zip(new, meshes):
        assert abs(sheet.n_elements - start.n_elements) <= 0.25 * start.n_elements, "each keeps its budget"
    x, y = obs.ds["x"].values, obs.ds["y"].values
    assert traces.distance(x, y, frame).min() > 2e3
    assert np.isfinite(history[-1]["max_change"])


def test_a_series_is_sampled_from_the_last_rounds_field(setup, grid):
    """What the loop ends with splits a dated stack's cells, the same on every date."""
    frame = setup["frame"]
    _, model, history = iterate_sampling({"S1": grid}, setup["mesh"], setup["trace"], frame, {"S1": KWARGS}, make_inversion=_make(setup), fixed=[_gnss(setup)], normalize="none",
                                         solve_kwargs=dict(smoothing=1e4, rate=(-RATE, 0.0)), max_rounds=1, verbose=False, tol=0.0)
    field = history[-1]["fields"]["S1"]
    assert field.shape == grid.ds["los"].shape[-2:] and np.array_equal(field, history[-1]["fields"]["S1"])
    assert np.isfinite(predicted_los(model, grid, frame)).any()
    days = np.array([100.0, 200.0, 365.0])
    stack = type(grid)(grid.ds.isel(pair=[0, 0, 0]).assign_coords(pair=[0, 1, 2]).pipe(lambda ds: ds.assign(los=ds["los"] * xr_days(days, ds))))
    series = TrackSeries.from_los(stack, np.zeros(3), days, 1e-3, name="S1", frame=frame, trace=setup["trace"], ramp="linear", field=field, **KWARGS)
    assert series.n_pairs == 3 and series.n_cells > 20
    result = TimeDependentInversion(setup["mesh"], [series], TimeBasis.linear(), engine=HalfSpaceTDE(0.25))
    assert result.solve(smoothing=1e4).converged


def xr_days(days, ds):
    import xarray as xr

    return xr.DataArray(days / 365.25, dims="pair", coords={"pair": ds["pair"]})
