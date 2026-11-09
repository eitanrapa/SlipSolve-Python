"""Look vectors rotated into the local frame's axes (`frame_looks`) in a coseismic `SlipInversion`.

Oracles: the rotation itself (`frame_looks` and its inverse `geographic_enu` are one rotation, by the grid convergence
`d_lon * sin(lat)`), the design assembled from rotated looks, and a solved model that forwards in the axes it was
solved in, before and after `save`.
"""

import numpy as np
import pytest

from slipsolve import DipSweep, FaultMesh, FaultTrace, Observations, SlipInversion, SlipModel
from slipsolve.basis import make_basis
from slipsolve.frame import LocalFrame
from slipsolve.greens import HalfSpaceTDE
from slipsolve.inversion import assemble_design
from slipsolve.sampling import frame_looks, geographic_enu

LON, LAT = 37.0, 37.0


@pytest.fixture(scope="module")
def setup():
    """A fault 200 km east of its frame's central meridian at 37 N, where grid north turns by about 1.4 degrees."""
    trace = FaultTrace(np.array([LON, LON + 0.05]), np.array([LAT, LAT + 0.3]), name="t")
    frame = LocalFrame(LON, LAT, ref_lon=LON - 2.25)
    mesh = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=4e3)
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-60e3, 60e3, 90), rng.uniform(-40e3, 80e3, 90)
    x = np.where(np.abs(x) < 4e3, 4e3, x)
    looks = np.tile([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, -0.1, 0.7937]], (30, 1))
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    lon, lat = frame.to_lonlat(x, y)
    obs = Observations.from_points(lon, lat, np.zeros(x.size), looks, 0.002, name="a", frame=frame)
    return trace, frame, mesh, obs


def test_the_rotation_and_its_inverse(setup):
    _, frame, _, obs = setup
    x, y = obs.ds["x"].values, obs.ds["y"].values
    ex, ey, _ = frame_looks(frame, x, y, np.ones(x.size), np.zeros(x.size), np.zeros(x.size))
    lon, lat = frame.to_lonlat(x, y)
    expected = np.degrees(np.arctan(np.tan(np.radians(lon - (LON - 2.25))) * np.sin(np.radians(lat))))
    assert np.allclose(np.degrees(np.arctan2(ey, ex)), expected, atol=0.02)     # the grid convergence, about 1.4 deg
    u = np.random.default_rng(1).normal(size=(3, x.size))
    back = geographic_enu(frame, x, y, *frame_looks(frame, x, y, *u))
    assert np.allclose(np.stack(back), u, rtol=0.0, atol=1e-10)


def test_the_design_uses_rotated_looks(setup):
    _, _, mesh, obs = setup
    rotated = SlipInversion(mesh, obs, basis="node", frame_looks=True)
    plain = SlipInversion(mesh, obs, basis="node")
    ds = obs.ds
    expected = assemble_design(HalfSpaceTDE(), mesh, make_basis(mesh, "node"),
                               (ds["x"].values, ds["y"].values, *obs.frame_looks()))
    assert np.allclose(rotated.g, expected, rtol=0.0, atol=1e-15)
    assert np.abs(rotated.g - plain.g).max() > 0.01 * np.abs(plain.g).max()  # a turn of a degree and a half shows


def test_a_rotated_model_forwards_in_its_own_axes(setup, tmp_path):
    _, frame, mesh, obs = setup
    truth = np.zeros(2 * mesh.n_nodes)
    truth[: mesh.n_nodes] = -1.0
    inversion = SlipInversion(mesh, obs, basis="node", frame_looks=True)
    data = Observations(obs.ds.assign(los=("obs", inversion.g @ truth)))
    inversion = SlipInversion(mesh, data, basis="node", frame_looks=True)
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    assert model.frame_rotated and model.options["frame_looks"] is True
    ds = data.ds
    x, y = ds["x"].values, ds["y"].values
    look = tuple(ds[k].values for k in ("look_e", "look_n", "look_u"))
    fit = inversion.g @ model.x[: inversion.n_slip]
    assert np.allclose(model.forward(x, y, look=look), fit, rtol=0.0, atol=1e-12)
    enu = model.forward(x, y)                                               # geographic east, north, up
    assert np.allclose(np.einsum("pc,cp->p", enu, np.stack(look)), fit, rtol=0.0, atol=1e-12)
    grid = model.surface_displacement(spacing=5e3, bounds=(-40e3, 40e3, -20e3, 60e3))
    assert grid.attrs["horizontal_axes"] == "geographic"
    gx, gy = np.meshgrid(grid["x"].values, grid["y"].values)
    on = np.isfinite(grid["ux"].values)
    at = np.column_stack([grid[c].values[on] for c in ("ux", "uy", "uz")])
    assert np.allclose(at, model.forward(gx[on], gy[on]), rtol=0.0, atol=1e-6)
    model.save(tmp_path / "m.slip.zip")
    back = SlipModel.load(tmp_path / "m.slip.zip")
    assert back.frame_rotated
    assert np.allclose(back.forward(x, y, look=look), model.forward(x, y, look=look), rtol=0.0, atol=1e-12)
    plain = SlipInversion(mesh, data, basis="node").solve(smoothing=0.3, polarity=(-1, 0, 0))
    assert not plain.frame_rotated and plain.options["frame_looks"] is False


def test_a_dip_sweep_rotates_too(setup):
    trace, frame, _, obs = setup
    sweep = DipSweep(trace, obs, frame=frame, segments=2, dips=(90.0,), max_depth=12e3, edge_length=4e3,
                     frame_looks=True)
    expected = SlipInversion(sweep._mesh(90.0), obs, basis="node", frame_looks=True).g
    assert np.allclose(sweep.design((90.0, 90.0)), expected, rtol=0.0, atol=1e-14)
