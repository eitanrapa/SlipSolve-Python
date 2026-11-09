"""Topography by the elevation shift (Williams & Wadge 1998): heights on the observations, and the Green's functions.

Oracles: the same rows computed with the fault itself lowered by each row's own height, one row at a time; and the
sampler's averages over a known DEM.
"""

import numpy as np
import pytest
from slip_synthetic import analytic_los_stack

from slipsolve import FaultMesh, FaultTrace, HalfSpaceTDE, Observations, SlipInversion, gnss_observations, gnss_series
from slipsolve.basis import make_basis
from slipsolve.inversion import TOPOGRAPHY_STEP, assemble_design
from slipsolve.temporal import TimeDependentInversion, TrackSeries

LOOK = np.array([0.6, -0.1, 0.7937])


@pytest.fixture
def fault():
    trace = FaultTrace(np.array([-120.0, -120.0]), np.array([35.0, 35.3]), name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=3e3)
    return trace, frame, mesh


def _points(n, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-30e3, 30e3, n)
    x = np.where(np.abs(x) < 2e3, np.sign(x) * 2e3 + (x == 0) * 2e3, x)
    y = rng.uniform(-5e3, 40e3, n)
    look = np.tile(LOOK / np.linalg.norm(LOOK), (n, 1)).T
    return x, y, look, rng.uniform(0.0, 2000.0, n)


def _row_by_row(engine, mesh, basis, x, y, look, heights):
    return np.vstack([assemble_design(engine, mesh.lowered(h), basis, (x[i:i + 1], y[i:i + 1], *look[:, i:i + 1]))
                      for i, h in enumerate(heights)])


def test_a_lowered_mesh_is_the_same_fault_deeper(fault):
    _, _, mesh = fault
    assert mesh.lowered(0.0) is mesh
    down = mesh.lowered(400.0)
    assert np.allclose(down.nodes[:, 2], mesh.nodes[:, 2] - 400.0)
    assert np.allclose(down.params[:, 1], mesh.params[:, 1] - 400.0)
    assert np.array_equal(down.triangles, mesh.triangles) and np.allclose(down.nodes[:, :2], mesh.nodes[:, :2])
    with pytest.raises(ValueError, match="above the free surface"):
        mesh.lowered(-10.0)                                                # its top is at the surface


def test_the_levels_match_a_shift_row_by_row(fault):
    _, _, mesh = fault
    basis, engine = make_basis(mesh, "node"), HalfSpaceTDE()
    x, y, look, heights = _points(40)
    exact = _row_by_row(engine, mesh, basis, x, y, look, heights)
    flat = assemble_design(engine, mesh, basis, (x, y, *look))
    levels = assemble_design(engine, mesh, basis, (x, y, *look), elevation=heights)
    assert np.abs(exact - flat).max() > 0.05 * np.abs(exact).max()        # the heights matter here
    assert np.abs(levels - exact).max() < 5e-4 * np.abs(exact).max()      # second order between levels
    on_levels = np.round(heights / TOPOGRAPHY_STEP) * TOPOGRAPHY_STEP
    assert np.allclose(assemble_design(engine, mesh, basis, (x, y, *look), elevation=on_levels),
                       _row_by_row(engine, mesh, basis, x, y, look, on_levels), rtol=0.0, atol=1e-12 * np.abs(exact).max())
    assert np.array_equal(assemble_design(engine, mesh, basis, (x, y, *look), elevation=np.zeros(x.size)), flat)


def test_the_sampler_averages_the_dem(fault):
    trace, frame, _ = fault
    stack = analytic_los_stack(trace, frame, spacing=1000.0, half_x=40e3, half_y=40e3)
    shape = stack.ds["los"].isel(pair=0).shape
    dem = np.full(shape, 800.0)
    dem[:, : shape[1] // 3] = np.nan                                       # no heights over a third of the scene
    dem[:, shape[1] // 3: shape[1] // 3 + 2] = 1400.0
    with pytest.raises(ValueError, match="no finite elevation"):
        Observations.from_los(stack, frame=frame, trace=trace, rms_min=0.008, width_min=2500.0, width_max=20000.0,
                              exclude_within=3000.0, elevation=dem, datum=100.0)
    dem[:, : shape[1] // 3] = 800.0
    obs = Observations.from_los(stack, frame=frame, trace=trace, rms_min=0.008, width_min=2500.0, width_max=20000.0,
                                exclude_within=3000.0, elevation=dem, datum=100.0)
    heights = obs.ds["elevation"].values
    assert obs.ds.attrs["datum"] == 100.0
    assert np.all((heights >= 700.0 - 1e-9) & (heights <= 1300.0 + 1e-9))
    assert np.isclose(np.median(heights), 700.0) and heights.max() > 700.0  # the ridge shows in the cells over it
    with pytest.raises(ValueError, match="shape"):
        Observations.from_los(stack, frame=frame, trace=trace, exclude_within=3000.0, elevation=dem[1:])


def test_concat_wants_heights_on_every_track_and_one_datum(fault):
    _, frame, _ = fault
    x, y, _, heights = _points(5)
    lon, lat = frame.to_lonlat(x, y)
    with_heights = Observations.from_points(lon, lat, np.zeros(5), LOOK / np.linalg.norm(LOOK), 0.01, name="a",
                                            frame=frame, elevation=heights, datum=10.0)
    without = Observations.from_points(lon, lat, np.zeros(5), LOOK / np.linalg.norm(LOOK), 0.01, name="b", frame=frame)
    other_datum = Observations.from_points(lon, lat, np.zeros(5), LOOK / np.linalg.norm(LOOK), 0.01, name="c",
                                           frame=frame, elevation=heights, datum=0.0)
    assert np.allclose(with_heights.ds["elevation"].values, heights - 10.0)
    with pytest.raises(ValueError, match="give every one"):
        Observations.concat([with_heights, without])
    with pytest.raises(ValueError, match="different datums"):
        Observations.concat([with_heights, other_datum])
    both = Observations.concat([with_heights, with_heights], normalize="none")
    assert both.ds["elevation"].size == 10 and both.ds.attrs["datum"] == 10.0


def test_an_inversion_uses_the_heights(fault):
    _, frame, mesh = fault
    x, y, look, heights = _points(60, seed=2)
    lon, lat = frame.to_lonlat(x, y)
    obs = Observations.from_points(lon, lat, np.zeros(60), look.T, 0.01, frame=frame, elevation=heights)
    inversion = SlipInversion(mesh, obs, basis="node")
    expected = assemble_design(HalfSpaceTDE(), mesh, inversion.basis, (obs.ds["x"].values, obs.ds["y"].values, *look),
                               elevation=heights)
    assert np.allclose(inversion.g, expected, rtol=0.0, atol=1e-14)
    flat = SlipInversion(mesh, Observations(obs.ds.drop_vars("elevation")), basis="node")
    assert np.abs(inversion.g - flat.g).max() > 0.01 * np.abs(flat.g).max()


def test_gnss_heights_reach_the_series_greens_functions(fault):
    _, frame, mesh = fault
    x, y, _, heights = _points(3, seed=3)
    lon, lat = frame.to_lonlat(x, y)
    obs = gnss_observations(lon, lat, np.zeros(3), np.zeros(3), np.zeros(3), 1e-3, 1e-3, 3e-3, frame,
                            elevation=heights, datum=200.0)
    assert np.allclose(obs.ds["elevation"].values, np.tile(heights - 200.0, 3))
    t = np.array([0.0, 10.0, 30.0])
    series = [s for i in range(3) for s in gnss_series(lon[i], lat[i], t, np.zeros(3), np.zeros(3), np.zeros(3), 1e-3,
                                                       1e-3, 3e-3, frame, name=f"s{i}", elevation=heights[i])]
    assert all(np.isclose(s.elevation[0], heights[i // 3]) for i, s in enumerate(series))
    basis = make_basis(mesh, "node")
    greens = TimeDependentInversion._cell_greens(series, HalfSpaceTDE(), mesh, basis, False)
    looks = np.array([s.look[0] for s in series]).T
    expected = assemble_design(HalfSpaceTDE(), mesh, basis, (np.array([s.x[0] for s in series]),
                                                             np.array([s.y[0] for s in series]), *looks),
                               elevation=np.array([s.elevation[0] for s in series]))
    assert np.allclose(greens, expected, rtol=0.0, atol=1e-14)
    bare = TrackSeries([x[0]], [y[0]], [1.0, 0.0, 0.0], [[0.0]], [0.0], [10.0], 1e-3, frame=frame)
    with pytest.raises(ValueError, match="give every one"):
        TimeDependentInversion._cell_greens(series + [bare], HalfSpaceTDE(), mesh, basis, False)
