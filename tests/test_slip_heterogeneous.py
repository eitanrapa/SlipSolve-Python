"""3-D elastic heterogeneity on the layered reference (``slipsolve.heterogeneity``).

The plan and its measurements are in ``HETEROGENEOUS_3D_PLAN.local.md``; the strain kernels
the engine stands on are tested in ``test_slip_strain.py``.
"""

import numpy as np
import pytest

from slipsolve import wavenumber as wn
from slipsolve.edgrn import VelocityModel
from slipsolve.heterogeneity import (VoxelModel, VoxelSource, compliance, from_voigt,
                                     stiffness_block, to_voigt, unit_moments)
from slipsolve.layered_tde import LayeredTDE

TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])


@pytest.fixture(scope="module")
def coarse_tables():
    """Quick two-layer tables with strain, for geometry and bookkeeping (not accuracy)."""
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=30e3, first=20.0, growth=2.0, spacing=1500.0, deep_spacing=6000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=8), n_k=1024,
                                      surface_z=z, surface_r=wn.distance_grid(per_decade=8), strain=True)


# -- conventions ---------------------------------------------------------------------------
def test_the_unit_moments_span_the_symmetric_tensors():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 3))
    m = m + m.T
    components = to_voigt(m)
    assert np.allclose(np.einsum("k,kab->ab", components, unit_moments()), m)
    assert np.allclose(from_voigt(components), m)


def test_the_stiffness_block_is_dc_contracted_with_the_strain():
    """``stiffness_block(dlam, dmu) @ eps`` = the components of ``dlam tr(eps) I + 2 dmu eps``."""
    rng = np.random.default_rng(1)
    eps = rng.normal(size=(3, 3))
    eps = 0.5 * (eps + eps.T)
    dlam, dmu = -3.0e9, 2.0e9
    want = to_voigt(dlam * np.trace(eps) * np.eye(3) + 2.0 * dmu * eps)
    assert np.allclose(stiffness_block(dlam, dmu)[0] @ to_voigt(eps), want)


def test_compliance_inverts_the_stiffness():
    lam, mu = 3.2e10, 2.9e10
    eps = compliance(unit_moments(), lam, mu)
    back = lam * np.trace(eps, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 2.0 * mu * eps
    assert np.allclose(back, unit_moments(), rtol=0, atol=1e-14)


# -- one voxel -------------------------------------------------------------------------------
def test_a_voxel_has_outward_faces_and_drops_the_free_surface(coarse_tables):
    engine = LayeredTDE(coarse_tables)
    buried = VoxelSource(engine, (-500.0, 700.0), (-300.0, 600.0), (3000.0, 4000.0))
    surface = VoxelSource(engine, (-500.0, 700.0), (-300.0, 600.0), (0.0, 800.0))
    assert buried.faces.shape == (12, 3, 3) and surface.faces.shape == (10, 3, 3)
    for v in (buried, surface):
        normals = np.cross(v.faces[:, 1] - v.faces[:, 0], v.faces[:, 2] - v.faces[:, 0])
        outward = v.faces.mean(axis=1) - v.centre
        assert np.all(np.einsum("ij,ij->i", normals, outward) > 0)
        area = 0.5 * np.linalg.norm(normals, axis=1).sum()
        dx, dy, dz = v.hi - v.lo
        full = 2.0 * (dx * dy + dx * dz + dy * dz)
        assert np.isclose(area, full - (dx * dy if v.free_surface else 0.0))
    assert buried.layer == 1 and surface.layer == 0
    assert np.isclose(buried.volume, 1200.0 * 900.0 * 1000.0)


def test_a_voxel_must_lie_in_one_layer(coarse_tables):
    engine = LayeredTDE(coarse_tables)
    with pytest.raises(ValueError, match="one layer"):
        VoxelSource(engine, (0.0, 1e3), (0.0, 1e3), (1500.0, 2500.0))
    on_interface = VoxelSource(engine, (0.0, 1e3), (0.0, 1e3), (1000.0, 2000.0))   # base on it: fine
    assert on_interface.layer == 0


# -- the voxel model --------------------------------------------------------------------------
def test_the_model_cuts_at_interfaces_and_finds_its_departure(coarse_tables):
    t = coarse_tables

    def moduli(x, y, depth):
        layer = t.layer(depth)
        soft = (np.abs(x) < 3e3) & (depth < 4e3)
        return t.lam[layer] * np.where(soft, 0.7, 1.0), t.mu[layer] * np.where(soft, 0.6, 1.0)

    v = VoxelModel(t, np.arange(-6e3, 6001.0, 2e3), np.arange(-4e3, 4001.0, 2e3), [0.0, 3e3, 6e3], moduli)
    assert v.inserted == [2000.0] and np.allclose(v.z_edges, [0.0, 2e3, 3e3, 6e3])
    assert v.shape == (6, 4, 3)
    assert v.n_active == 2 * 4 * 2                     # |x| < 3 km and depth < 4 km: two columns, two rows
    assert np.all(-v.centres[:, 2] < 4e3)
    assert np.allclose(v.dmu[v.active] / v.mu0[v.active], -0.4)
    d = v.stiffness()
    assert d.shape == (6 * v.n_active, 6 * v.n_active)
    with pytest.raises(ValueError, match="callable"):
        VoxelModel(t, [0.0, 1e3], [0.0, 1e3], [0.0, 3e3], lam=1e10, mu=1e10)
    # rigidity: a voxel's inside the grid, the reference's outside it
    mu = v.rigidity_at(np.array([[0.0, 0.0, -1000.0], [5e3, 0.0, -1000.0], [0.0, 0.0, -7000.0]]))
    assert np.allclose(mu, [0.6 * t.mu[0], t.mu[0], t.mu[1]])


def test_voxels_keep_clear_of_the_fault(coarse_tables):
    from slipsolve import FaultMesh, FaultTrace

    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.92]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, max_depth=4e3, edge_length=2e3)
    px, py = trace.to_local(frame)
    cy = 0.5 * (py[0] + py[-1])
    t = coarse_tables

    def block(y0):
        def moduli(x, y, depth):
            layer = t.layer(depth)
            here = (y > cy + y0) & (y < cy + y0 + 2e3)
            return t.lam[layer] * np.where(here, 0.8, 1.0), t.mu[layer] * np.where(here, 0.8, 1.0)
        return VoxelModel(t, np.arange(px.min() - 2e3, px.max() + 2001.0, 1e3),
                          np.arange(cy - 8e3, cy + 8001.0, 1e3), [0.0, 2e3, 4e3], moduli)

    assert block(3e3).check_clearance(mesh)                         # 3 km off, voxels 1 km
    with pytest.raises(ValueError, match="clear"):
        block(0.2e3).check_clearance(mesh)


# -- the operators and the engine ---------------------------------------------------------
from slipsolve.heterogeneity import HeterogeneousEngine, strain_operator  # noqa: E402


def _scaled(tables, factor_lam, factor_mu, where):
    """Moduli for :class:`VoxelModel`: the reference's, scaled where ``where(x, y, depth)``."""
    def moduli(x, y, depth):
        layer = tables.layer(depth)
        here = where(x, y, depth)
        return (tables.lam[layer] * np.where(here, factor_lam, 1.0),
                tables.mu[layer] * np.where(here, factor_mu, 1.0))
    return moduli


def test_gamma_by_translation_is_gamma_pair_by_pair(coarse_tables):
    """The translation-invariant assembly against every source voxel evaluated at every centre."""
    t = coarse_tables
    v = VoxelModel(t, [0.0, 1e3, 2e3, 3e3], [0.0, 1e3, 2e3], [3e3, 4e3, 5e3],
                   _scaled(t, 0.9, 0.8, lambda x, y, d: np.ones_like(x, bool)))
    engine = LayeredTDE(t)
    gamma = strain_operator(v, engine)
    want = np.zeros_like(gamma)
    for s in range(v.n_active):
        field = VoxelSource(engine, *v.bounds(s)).strain(*v.centres.T)       # (n, 6 moment, 6 strain)
        for r in range(v.n_active):
            want[6 * r:6 * r + 6, 6 * s:6 * s + 6] = field[r].T
    assert v.n_active == 12
    assert np.abs(gamma - want).max() < 1e-9 * np.abs(want).max()


def _small_problem(tables):
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve.basis import make_basis

    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, max_depth=3e3, edge_length=2e3)
    px, py = trace.to_local(frame)
    cx, cy = 0.5 * (px[0] + px[-1]), 0.5 * (py[0] + py[-1])
    rng = np.random.default_rng(3)
    x = cx + rng.uniform(-12e3, 12e3, 20)
    y = cy + rng.uniform(-12e3, 12e3, 20)
    look = (np.full(20, 0.6), np.full(20, -0.1), np.full(20, 0.79))
    return mesh, make_basis(mesh, "node"), (cx, cy), x, y, look


def test_no_departure_is_the_reference(coarse_tables):
    t = coarse_tables
    mesh, basis, (cx, cy), x, y, look = _small_problem(t)
    v = VoxelModel(t, cx + np.arange(-4e3, 4001.0, 2e3), cy + np.arange(4e3, 8001.0, 2e3), [6e3, 8e3],
                   _scaled(t, 1.0, 1.0, lambda x, y, d: np.ones_like(x, bool)))
    assert v.n_active == 0
    base = LayeredTDE(t)
    assert np.array_equal(HeterogeneousEngine(base, v).los_matrix(mesh, x, y, *look, basis=basis),
                          base.los_matrix(mesh, x, y, *look, basis=basis))


def test_born_is_first_order_in_the_contrast(coarse_tables):
    """The full solve departs from Born-1 as the contrast squared: ``Gamma``, ``E`` and ``H`` are shared."""
    t = coarse_tables
    mesh, basis, (cx, cy), x, y, look = _small_problem(t)
    base = LayeredTDE(t)
    corrections = {}
    shared = None
    for c in (0.05, 0.10):
        v = VoxelModel(t, cx + np.arange(-6e3, 6001.0, 2e3), cy + np.arange(4e3, 10001.0, 2e3),
                       [5e3, 7e3], _scaled(t, 1.0 - c, 1.0 - c, lambda x, y, d: np.ones_like(x, bool)))
        g0 = base.los_matrix(mesh, x, y, *look, basis=basis)
        for order in ("born1", "full"):
            engine = HeterogeneousEngine(base, v, order=order)
            if shared is not None:
                engine._gamma, engine._e, engine._h = shared
            corrections[(c, order)] = engine.los_matrix(mesh, x, y, *look, basis=basis) - g0
            shared = (engine.gamma, engine._e, engine._h)
    born_small, born_large = corrections[(0.05, "born1")], corrections[(0.10, "born1")]
    assert np.allclose(born_large, 2.0 * born_small, rtol=0, atol=1e-12 * np.abs(born_large).max())
    gap_small = np.abs(corrections[(0.05, "full")] - born_small).max()
    gap_large = np.abs(corrections[(0.10, "full")] - born_large).max()
    assert 3.5 < gap_large / gap_small < 4.5
    assert gap_small < 0.2 * np.abs(born_small).max()
