"""The voxels' strain equation by FFT and GMRES (`slipsolve.voxel_solve`) against the dense path."""

import numpy as np
import pytest

from slipsolve import wavenumber as wn
from slipsolve.edgrn import VelocityModel
from slipsolve.heterogeneity import HeterogeneousEngine, VoxelModel, strain_operator
from slipsolve.layered_tde import LayeredTDE
from slipsolve.voxel_solve import ConvolutionGamma, StrainSystem, gmres

TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])


@pytest.fixture(scope="module")
def coarse_tables():
    """Quick two-layer tables with strain (as ``test_slip_heterogeneous``'s): bookkeeping, not accuracy."""
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=30e3, first=20.0, growth=2.0, spacing=1500.0, deep_spacing=6000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=8), n_k=1024,
                                      surface_z=z, surface_r=wn.distance_grid(per_decade=8), strain=True)


def _model(tables, factor=0.5, holes=True):
    """A 5 x 3 x 3 grid of 1 km voxels crossing the 2 km interface, softer by ``factor``; three of them inactive."""
    def moduli(gx, gy, gz):
        layer = tables.layer(gz)
        lam, mu = factor * tables.lam[layer], factor * tables.mu[layer]
        if holes:
            off = ((gx > 3e3) & (gy < 4e3) & (gz < 1e3)) | ((gx < 1e3) & (gy > 5e3) & (gz > 2e3))
            lam, mu = np.where(off, tables.lam[layer], lam), np.where(off, tables.mu[layer], mu)
        return lam, mu

    return VoxelModel(tables, np.arange(0.0, 5001.0, 1e3), np.arange(3e3, 6001.0, 1e3), [0.0, 1e3, 2e3, 3e3],
                      lam=moduli)


def test_the_convolution_is_the_dense_gamma(coarse_tables):
    """Every offset's kernel is the dense path's number, so the FFT product is ``Gamma @ m`` to rounding."""
    engine = LayeredTDE(coarse_tables)
    vox = _model(coarse_tables)
    assert vox.n_active == 3 * 5 * 3 - 3
    conv = ConvolutionGamma(vox, engine)
    dense = strain_operator(vox, engine)
    np.testing.assert_allclose(conv.dense(), dense, rtol=0, atol=1e-12 * np.abs(dense).max())
    blocks = dense.reshape(vox.n_active, 6, vox.n_active, 6)[np.arange(vox.n_active), :, np.arange(vox.n_active)]
    np.testing.assert_allclose(conv.self_blocks, blocks, rtol=1e-12, atol=0)


def test_gmres_is_the_lu(coarse_tables):
    """``StrainSystem.solve`` against ``lu_solve`` of the dense ``I + Gamma D``, softer and stiffer voxels."""
    import scipy.linalg as sla

    engine = LayeredTDE(coarse_tables)
    rng = np.random.default_rng(2)
    for factor in (0.3, 2.5):
        vox = _model(coarse_tables, factor=factor)
        conv = ConvolutionGamma(vox, engine)
        system = StrainSystem(conv)
        dense = np.eye(conv.size) + strain_operator(vox, engine) @ vox.stiffness().toarray()
        rhs = rng.normal(size=(conv.size, 5))
        want = sla.solve(dense, rhs)
        got = system.solve(rhs, tol=1e-10, block=3)
        assert np.abs(got - want).max() < 1e-8 * np.abs(want).max()
        assert max(system.iterations) < 60


def test_gmres_restarts_converges_and_refuses():
    """On a nonsymmetric system: restarts reach the answer, a zero column stays zero, a cap raises."""
    rng = np.random.default_rng(0)
    n = 120
    a = np.eye(n) + 0.9 * rng.normal(size=(n, n)) / np.sqrt(n)
    b = rng.normal(size=(n, 4))
    b[:, 2] = 0.0
    x, products = gmres(lambda v: a @ v, b, tol=1e-10, restart=15, max_iter=2000)
    np.testing.assert_allclose(a @ x, b, atol=1e-9 * np.abs(b).max())
    assert products[2] == 0 and np.all(x[:, 2] == 0.0)
    assert products.max() > 15                                   # it had to restart
    with pytest.raises(RuntimeError, match="GMRES: column"):
        gmres(lambda v: a @ v, b, tol=1e-12, restart=5, max_iter=10)


def test_the_fft_engine_is_the_dense_engine(coarse_tables, tmp_path):
    """``HeterogeneousEngine(solver="fft")``: the dense engine's G to the solve's tolerance, H streamed in blocks."""
    from slipsolve import FaultMesh, FaultTrace

    tables = wn.RemainderTables.load(coarse_tables.save(tmp_path / "coarse.npz"))
    base = LayeredTDE(tables)
    trace = FaultTrace.from_lonlat(np.array([-80.05, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=6e3, edge_length=2e3)
    _, ty = (np.asarray(v) for v in trace.to_local(frame))
    y0 = float(np.mean(ty))
    lam0, mu0 = float(tables.lam[0]), float(tables.mu[0])
    vox = VoxelModel(tables, [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0], [y0 + 2500.0, y0 + 3500.0, y0 + 4500.0],
                     [0.0, 1000.0, 2000.0], lam=np.full((5, 2, 2), 0.5 * lam0), mu=np.full((5, 2, 2), 0.5 * mu0))
    vox.check_clearance(mesh)
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-9e3, 9e3, 25), y0 + rng.uniform(-9e3, 9e3, 25)
    look = (np.full(25, 0.6), np.full(25, -0.1), np.full(25, 0.79))
    reference = HeterogeneousEngine(base, vox)
    dense = reference.los_matrix(mesh, x, y, *look)
    g0 = base.los_matrix(mesh, x, y, *look)
    for fft in (HeterogeneousEngine(base, vox, solver="fft", h_chunk=7),
                HeterogeneousEngine(base, vox, solver="fft", h_chunk=6, workers=2)):
        got = fft.los_matrix(mesh, x, y, *look)
        assert np.abs(got - dense).max() < 1e-5 * np.abs(dense - g0).max()
        enu, enu_dense = fft.displacement(mesh, x, y), reference.displacement(mesh, x, y)
        assert np.abs(enu - enu_dense).max() < 1e-5 * np.abs(enu_dense - base.displacement(mesh, x, y)).max()
        assert not fft._h and fft.describe()["solver"] == "fft"
        fft.close()
    with pytest.raises(ValueError, match="solver"):
        HeterogeneousEngine(base, vox, solver="cg")


def test_a_subset_of_the_surface_operator_is_its_columns(coarse_tables):
    """`los_operator(subset=)` gives those voxels' columns of the whole operator, to rounding: a row's voxels go
    through one reference voxel at shifted receivers, and which voxel is the reference, and how many receivers share
    a call, depend on the voxels asked for."""
    from slipsolve.heterogeneity import los_operator

    engine = LayeredTDE(coarse_tables)
    vox = _model(coarse_tables)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-8e3, 12e3, 15), rng.uniform(-5e3, 14e3, 15)
    look = (np.full(15, 0.6), np.full(15, -0.1), np.full(15, 0.79))
    whole = los_operator(vox, engine, x, y, look)
    subset = np.array([2, 5, 6, 20, 41])
    cols = (6 * subset[:, None] + np.arange(6)).ravel()
    want = whole[:, cols]
    np.testing.assert_allclose(los_operator(vox, engine, x, y, look, subset=subset), want, rtol=1e-12,
                               atol=1e-12 * np.abs(want).max())


def test_the_threaded_batched_product_is_the_serial_one():
    """``batched_matmul`` over a thread pool equals numpy's serial batched product, and leaves BLAS's threads as found."""
    from slipsolve.voxel_solve import _blas_control, batched_matmul

    rng = np.random.default_rng(5)
    a = rng.normal(size=(40, 9, 12, 12)) + 1j * rng.normal(size=(40, 9, 12, 12))
    b = rng.normal(size=(40, 9, 12, 5)) + 1j * rng.normal(size=(40, 9, 12, 5))
    control = _blas_control()
    before = None if control is None else control[0]()
    np.testing.assert_allclose(batched_matmul(a, b, threads=4), np.matmul(a, b), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(batched_matmul(a, b, threads=1), np.matmul(a, b))
    if control is not None:
        assert control[0]() == before
        seen = []
        import slipsolve.voxel_solve as vs

        real = vs.ThreadPoolExecutor

        class Spy(real):
            def __init__(self, n, *args, **kwargs):
                seen.append(n)
                super().__init__(n, *args, **kwargs)

        vs.ThreadPoolExecutor = Spy
        try:
            batched_matmul(a, b, threads=3)
            batched_matmul(a, b, threads=1)
        finally:
            vs.ThreadPoolExecutor = real
        assert seen == [3]                     # the count asked for; one thread means no pool
