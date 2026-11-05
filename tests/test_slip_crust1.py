"""CRUST1.0 and gridded models as voxel moduli (``slipsolve.crust1``, ``VoxelModel.from_crust1 / from_grid / keep_clear``)."""

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace
from slipsolve import wavenumber as wn
from slipsolve.crust1 import Crust1
from slipsolve.edgrn import VelocityModel
from slipsolve.heterogeneity import VoxelModel

TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])


@pytest.fixture(scope="module")
def coarse_tables():
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=30e3, first=20.0, growth=2.0, spacing=1500.0, deep_spacing=6000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=8), n_k=1024,
                                      surface_z=z, surface_r=wn.distance_grid(per_decade=8), strain=True)


def _crust(water=1.0, layers=None):
    """Every cell: ``water`` km of water over a column given as [(thickness km, vp, vs, rho), ...] for sediments 1-3 and
    crust 1-3 (six rows; zero thickness allowed), then the mantle (vp, vs, rho)."""
    if layers is None:
        layers = [(2.0, 3.0, 1.5, 2.2), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                  (10.0, 6.0, 3.5, 2.7), (7.0, 6.5, 3.7, 2.8), (10.0, 7.0, 3.9, 2.9), (8.1, 4.6, 3.35)]
    bounds = np.zeros((64800, 9))
    vp, vs, rho = (np.zeros((64800, 9)) for _ in range(3))
    top = 0.0
    bounds[:, 0] = top
    top -= water
    bounds[:, 1] = top                                   # ice, zero thickness
    bounds[:, 2] = top
    vp[:, 0], vs[:, 0], rho[:, 0] = 1.5, 0.0, 1.02
    for i, (thick, p, s, r) in enumerate(layers[:6]):
        bounds[:, 2 + i] = top
        vp[:, 2 + i], vs[:, 2 + i], rho[:, 2 + i] = p, s, r
        top -= thick
    bounds[:, 8] = top
    vp[:, 8], vs[:, 8], rho[:, 8] = layers[6]
    return Crust1(bounds, vp, vs, rho)


def test_cells_columns_and_exact_depth_averages():
    crust = _crust()
    assert Crust1.cell(-68.5, 10.5) == 79 * 360 + 111
    assert Crust1.cell(179.9, -89.9) == 179 * 360 + 359 and Crust1.cell(-180.0, 89.99) == 0
    top, bottom, lam, mu = crust.column(1234)
    np.testing.assert_allclose(top, [0.0, 2000.0, 12000.0, 19000.0, 29000.0])     # water, ice, empty layers gone
    assert np.isinf(bottom[-1])
    np.testing.assert_allclose(mu[0], 2200.0 * 1500.0 ** 2)
    np.testing.assert_allclose(lam[1], 2700.0 * 6000.0 ** 2 - 2.0 * 2700.0 * 3500.0 ** 2)
    # a range across the sediment/crust boundary takes each layer's share of it
    got_lam, got_mu = crust.average(-79.5, 8.5, 1000.0, 3000.0)
    np.testing.assert_allclose(got_mu, 0.5 * (mu[0] + mu[1]))
    np.testing.assert_allclose(got_lam, 0.5 * (lam[0] + lam[1]))
    got_lam, got_mu = crust.average(-79.5, 8.5, 25000.0, 31000.0)                   # into the mantle
    np.testing.assert_allclose(got_mu, (4.0 * mu[3] + 2.0 * mu[4]) / 6.0)
    with pytest.raises(ValueError, match="top < bottom"):
        crust.average(-79.5, 8.5, 3000.0, 1000.0)


def _frame_and_mesh():
    trace = FaultTrace.from_lonlat(np.array([-79.6, -79.4]), np.array([8.5, 8.5]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=6e3, edge_length=2e3)
    return trace, frame, mesh


def _reference_crust(tables, cell_override=None):
    """A CRUST1.0 equal to the tables' reference: 2 km of the top layer over the second, which runs into the mantle."""
    (p0, s0, r0), (p1, s1, r1) = ((3.75, 1.95, 2.37), (6.1, 3.5, 2.75))
    layers = [(2.0, p0, s0, r0), (0.0, 0, 0, 0), (0.0, 0, 0, 0), (20.0, p1, s1, r1), (10.0, p1, s1, r1),
              (10.0, p1, s1, r1), (p1, s1, r1)]
    crust = _crust(water=0.0, layers=layers)
    if cell_override is not None:
        k, vs_factor = cell_override
        crust.vs[k, 5:9] *= vs_factor                     # the crust and mantle of one cell slower
    return crust


def test_crust1_equal_to_the_reference_departs_nowhere_and_one_cell_departs_alone(coarse_tables):
    trace, frame, mesh = _frame_and_mesh()
    x_edges, y_edges = np.arange(-6e3, 6001.0, 2e3), np.arange(3e3, 9001.0, 2e3)
    z_edges = np.array([0.0, 1000.0, 3000.0, 5000.0])
    vox = VoxelModel.from_crust1(coarse_tables, x_edges, y_edges, z_edges, _reference_crust(coarse_tables), frame)
    assert vox.n_active == 0 and np.allclose(vox.z_edges, [0.0, 1000.0, 2000.0, 3000.0, 5000.0])  # interface inserted
    centre_lon, centre_lat = (float(np.asarray(v)) for v in frame.to_lonlat(0.0, 6e3))
    k = int(Crust1.cell(centre_lon, centre_lat))
    slower = VoxelModel.from_crust1(coarse_tables, x_edges, y_edges, z_edges,
                                    _reference_crust(coarse_tables, (k, 0.9)), frame)
    # every column is in that cell (the grid spans 12 km of a 110 km cell); the top 2 km keep the reference
    assert slower.n_active == (x_edges.size - 1) * (y_edges.size - 1) * 2
    assert np.all(slower.index[:, 2] >= 2)
    np.testing.assert_allclose(slower.mu[slower.active], 0.81 * 2750.0 * 3500.0 ** 2)


def test_a_gridded_model_is_averaged_over_each_voxels_depths(coarse_tables):
    trace, frame, mesh = _frame_and_mesh()
    x_edges, y_edges = np.arange(-4e3, 4001.0, 2e3), np.arange(3e3, 7001.0, 2e3)
    z_edges = np.array([0.0, 2000.0, 4000.0])
    lon, lat = np.linspace(-80.0, -79.0, 11), np.linspace(8.0, 9.0, 11)
    depth = np.linspace(0.0, 10e3, 21)
    vs = np.broadcast_to(3000.0 + 0.1 * depth, (11, 11, 21))                       # linear in depth
    vp = np.sqrt(3.0) * vs
    rho = np.full((11, 11, 21), 2700.0)
    vox = VoxelModel.from_grid(coarse_tables, x_edges, y_edges, z_edges, lon, lat, depth, vp, vs, rho, frame, samples=6)
    t = (np.arange(6) + 0.5) / 6
    for iz, (a, b) in enumerate(((0.0, 2000.0), (2000.0, 4000.0))):
        s = 3000.0 + 0.1 * (a + (b - a) * t)
        np.testing.assert_allclose(vox.mu[:, :, iz], 2700.0 * np.mean(s ** 2), rtol=1e-12)
        np.testing.assert_allclose(vox.lam[:, :, iz], 2700.0 * np.mean(s ** 2), rtol=1e-12)        # vp^2 = 3 vs^2
    with pytest.raises(ValueError, match="outside the model's grid"):
        VoxelModel.from_grid(coarse_tables, x_edges, y_edges, np.array([0.0, 12e3]), lon, lat, depth, vp, vs, rho, frame)


def test_keep_clear_returns_the_voxels_by_the_fault_to_the_reference(coarse_tables):
    trace, frame, mesh = _frame_and_mesh()
    _, ty = (np.asarray(v) for v in trace.to_local(frame))
    y0 = float(np.mean(ty))
    lam0, mu0 = float(coarse_tables.lam[0]), float(coarse_tables.mu[0])
    vox = VoxelModel(coarse_tables, np.arange(-6e3, 6001.0, 2e3), y0 + np.arange(-6e3, 6001.0, 2e3), [0.0, 2000.0],
                     lam=0.5 * lam0, mu=0.5 * mu0)
    before = vox.n_active
    with pytest.raises(ValueError, match="from the fault"):
        vox.check_clearance(mesh)
    dropped = vox.keep_clear(mesh)
    assert 0 < dropped < before and vox.n_active == before - dropped
    assert vox.check_clearance(mesh)
    off = ~vox.active
    np.testing.assert_allclose(vox.mu[off], vox.mu0[off])
    assert np.all(np.abs(vox.centres[:, 1] - y0) > 2e3)               # what is left is a voxel or more from the trace


def test_a_dataset_in_km_any_order_is_the_grid(coarse_tables):
    """``from_dataset``: km, km/s and g/cm^3, dims (depth, lat, lon), latitudes decreasing and 0-360 longitudes give
    ``from_grid``'s model; Brocher's relations stand in for missing Vs and density."""
    import xarray as xr

    from slipsolve.heterogeneity import brocher_density, brocher_vs

    trace, frame, mesh = _frame_and_mesh()
    x_edges, y_edges = np.arange(-4e3, 4001.0, 2e3), np.arange(3e3, 7001.0, 2e3)
    z_edges = np.array([0.0, 2000.0, 4000.0])
    lon, lat = np.linspace(-80.0, -79.0, 11), np.linspace(8.0, 9.0, 11)
    depth = np.linspace(0.0, 10e3, 21)
    vs = np.broadcast_to(3000.0 + 0.1 * depth, (11, 11, 21)).copy()
    vp = np.sqrt(3.0) * vs
    rho = np.full((11, 11, 21), 2700.0)
    want = VoxelModel.from_grid(coarse_tables, x_edges, y_edges, z_edges, lon, lat, depth, vp, vs, rho, frame, samples=6)
    ds = xr.Dataset({k: (("depth", "latitude", "longitude"), np.transpose(v, (2, 1, 0))[:, ::-1, :] / 1e3)
                     for k, v in (("vp", vp), ("vs", vs), ("rho", rho))},
                    coords={"depth": depth / 1e3, "latitude": lat[::-1], "longitude": lon + 360.0})
    got = VoxelModel.from_dataset(coarse_tables, x_edges, y_edges, z_edges, ds, frame, samples=6)
    np.testing.assert_allclose(got.mu, want.mu, rtol=1e-12)
    np.testing.assert_allclose(got.lam, want.lam, rtol=1e-12)
    assert np.isclose(brocher_density(6.0), 2.7167, atol=1e-4) and np.isclose(brocher_vs(6.0), 3.5494, atol=1e-4)
    derived = VoxelModel.from_dataset(coarse_tables, x_edges, y_edges, z_edges, ds.drop_vars(["vs", "rho"]), frame,
                                      vs=None, rho=None, samples=6)
    assert np.all(derived.mu > 0) and np.all(np.isfinite(derived.lam))
