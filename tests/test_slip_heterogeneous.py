"""3-D elastic heterogeneity on the layered reference (`slipsolve.heterogeneity`).

The strain kernels the engine stands on are tested in `test_slip_strain.py`.
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
    """Moduli for `VoxelModel`: the reference's, scaled where `where(x, y, depth)`."""
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


def _small_problem(tables, max_depth=3e3, reach=12e3, n=20):
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve.basis import make_basis

    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, max_depth=max_depth, edge_length=2e3)
    px, py = trace.to_local(frame)
    cx, cy = 0.5 * (px[0] + px[-1]), 0.5 * (py[0] + py[-1])
    rng = np.random.default_rng(3)
    x = cx + rng.uniform(-reach, reach, n)
    y = cy + rng.uniform(-reach, reach, n)
    look = (np.full(n, 0.6), np.full(n, -0.1), np.full(n, 0.79))
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


def test_a_split_level_integral_resolves_receivers_next_to_a_face(monkeypatch):
    """Receivers 1-3% of a size from a triangle: the capped rule misses, the split matches 400 points.

    Against a 400-point rule, displacement / strain, at 1% of a size: 1.4e-2 / 0.26 capped, 8e-5 / 4.5e-3
    split; at 3%: 3.7e-3 / 5.6e-2 and 4e-6 / 1.4e-4.
    """
    import slipsolve.layered_tde as lt
    from slipsolve._tde import tde_disp_hs_unit, tde_strain_hs_unit

    tri = np.array([[0.0, 0.0, -3000.0], [1200.0, 0.0, -3000.0], [0.0, 0.0, -4000.0]])
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    size = np.sqrt(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
    rng = np.random.default_rng(5)
    pts = rng.dirichlet(np.ones(3), size=3) @ tri + 0.02 * size * normal

    def poisson(d):
        return np.full(np.shape(d), 0.25)

    for kernel, width, bar in ((tde_disp_hs_unit, 3, 1e-3), (tde_strain_hs_unit, 6, 1e-2)):
        out = {}
        for label, cap, split in (("capped", 24, False), ("split", 24, True), ("reference", 400, False)):
            monkeypatch.setattr(lt, "TENT_MAX_POINTS", cap)
            field = np.zeros((3, 3, 3, width))
            lt.tent_levelset(tri, *pts.T, np.arange(3), field, [0, 1, 2], kernel, 4.0, 3, (), poisson,
                             split=split)
            out[label] = field
        scale = np.abs(out["reference"]).max()
        assert np.abs(out["split"] - out["reference"]).max() < bar * scale
        assert np.abs(out["capped"] - out["reference"]).max() > 3 * np.abs(out["split"] - out["reference"]).max()


# -- one voxel against exact answers ---------------------------------------------------------------
ONE_LAYER = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)


@pytest.fixture(scope="module")
def uniform_tables():
    """A uniform crust: the remainder is zero, so a voxel is its faces; fine surface tables for the far path."""
    tops, _, _ = wn.layer_stack(ONE_LAYER)
    z = wn.depth_grid(tops, z_max=40e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    return wn.RemainderTables.compute(ONE_LAYER, z=z, r=wn.distance_grid(per_decade=8), n_k=512,
                                      z_max=40e3,
                                      surface_z=wn.surface_depth_grid(tops, fine_to=20e3, z_max=40e3,
                                                                      spacing=100.0),
                                      strain=True)


def test_a_voxel_split_in_two_is_the_voxel(uniform_tables):
    """Outside both halves their sum is the whole: each half's extra Burgers vector is uniform on a closed surface."""
    engine = LayeredTDE(uniform_tables)
    whole = VoxelSource(engine, (0.0, 2000.0), (0.0, 1000.0), (3000.0, 4000.0))
    left = VoxelSource(engine, (0.0, 1000.0), (0.0, 1000.0), (3000.0, 4000.0))
    right = VoxelSource(engine, (1000.0, 2000.0), (0.0, 1000.0), (3000.0, 4000.0))
    pts = np.array([[4500.0, 300.0, -2000.0], [-1500.0, 2500.0, -4200.0], [1200.0, -2200.0, -3500.0],
                    [800.0, 400.0, 0.0]])
    for field in ("displacement", "strain"):
        a = getattr(whole, field)(*pts.T)
        b = getattr(left, field)(*pts.T) + getattr(right, field)(*pts.T)
        assert np.abs(a - b).max() < 3e-4 * np.abs(a).max(), field


def test_a_voxel_is_its_point_moments(uniform_tables):
    """Against a volume integral of exact point moment sources (direct transforms, no tables).

    Receivers a voxel size or more above or below it, where the whole kernel decays in
    wavenumber. The errors are 2e-5 to 8e-5 in displacement and 1e-4 to 1.8e-4 in strain.
    """
    from slipsolve.layered_tde import _combine, _gauss01, combine_strain, tensor_packing

    engine = LayeredTDE(uniform_tables)
    v = VoxelSource(engine, (-500.0, 500.0), (-400.0, 600.0), (3000.0, 4000.0))
    pts = np.array([[700.0, -900.0, -1500.0], [-1600.0, 1200.0, -1800.0], [300.0, 2000.0, -6500.0]])
    tops, lam, mu = wn.layer_stack(ONE_LAYER)
    k, dln = wn.wavenumbers(n=2048)
    packing = tensor_packing(unit_moments(), v.lam, v.mu)
    t, w = _gauss01(5)
    grid = np.stack(np.meshgrid(*[v.lo[i] + (v.hi[i] - v.lo[i]) * t for i in range(3)], indexing="ij"), -1)
    weights = np.einsum("i,j,k->ijk", w, w, w).ravel() * v.volume
    disp = np.zeros((3, 6, 3))
    strain = np.zeros((3, 6, 6))
    for q, wq in zip(grid.reshape(-1, 3), weights):
        kern = wn.kernels(tops, lam, mu, -q[2], -pts[:, 2], k, remainder=False, derivatives=True)
        for i, (x, y, z) in enumerate(pts):
            r = np.array([np.hypot(x - q[0], y - q[1])])
            az = np.array([np.arctan2(x - q[0], y - q[1])])
            one = tuple(a[i:i + 1] for a in kern)
            values = wn.transform(*one[:3], k, dln, r)[0]
            disp[i] += wq * _combine(values, az, packing)[0]
            strain[i] += wq * combine_strain(values, wn.strain_columns(*one, k, dln, r)[0], az, packing)[0]
    for got, want in ((v.displacement(*pts.T), disp), (v.strain(*pts.T), strain)):
        error = np.abs(got - want).max(axis=(1, 2)) / np.abs(want).max(axis=(1, 2))
        assert error.max() < 1e-3


def test_far_surface_receivers_take_point_sources_of_the_same_field(uniform_tables):
    """A surface voxel seen from 2.1-12 sizes: the whole-table point sources against its faces (the four-point band
    inside 4 sizes from the centre, three inside 6, two inside 15)."""
    engine = LayeredTDE(uniform_tables)
    near = VoxelSource(engine, (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), far=None)
    far = VoxelSource(engine, (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0))
    angle = np.linspace(0.0, 2 * np.pi, 7)[:-1]
    radii = [2.8, 3.4, 4.2, 5.0, 8.0, 12.0]
    reach = np.repeat(radii, angle.size) * 1e3
    x = 500.0 + reach * np.cos(np.tile(angle, len(radii)))
    y = 500.0 + reach * np.sin(np.tile(angle, len(radii)))
    z = np.zeros(x.size)
    assert far._far(x, y, z).all()
    a, b = near.displacement(x, y, z), far.displacement(x, y, z)
    for i in range(x.size):
        assert np.abs(a[i] - b[i]).max() < 5e-3 * np.abs(a[i]).max()


def test_the_engine_inverts_and_a_saved_model_refuses_to_forward(coarse_tables, tmp_path):
    """``SlipInversion`` takes the engine like any other; a reloaded model cannot rebuild it (no tables saved)."""
    from slipsolve import FaultTrace, SlipInversion, SlipModel
    from slipsolve.sampling import Observations
    import xarray as xr

    t = coarse_tables
    mesh, basis, (cx, cy), x, y, look = _small_problem(t)
    v = VoxelModel(t, cx + np.arange(-6e3, 6001.0, 3e3), cy + np.arange(4e3, 10001.0, 3e3), [5e3, 8e3],
                   _scaled(t, 0.7, 0.7, lambda x, y, d: np.ones_like(x, bool)))
    engine = HeterogeneousEngine(LayeredTDE(t), v)
    truth = np.concatenate([np.full(basis.n_basis, -1.0), np.zeros(basis.n_basis)])
    los = engine.los_matrix(mesh, x, y, *look, basis=basis) @ truth
    ds = xr.Dataset({"los": ("obs", los), "weight": ("obs", np.ones(x.size)),
                     "x": ("obs", x), "y": ("obs", y), "look_e": ("obs", look[0]),
                     "look_n": ("obs", look[1]), "look_u": ("obs", look[2]),
                     "track": ("obs", np.array(["T"] * x.size))})
    obs = Observations(ds)
    inv = SlipInversion(mesh, obs, engine=engine, basis="node")
    assert np.allclose(inv.g @ truth, los)
    model = inv.solve(smoothing=1e-6, method="active_set")
    assert model.options["bimaterial"]["kind"] == "heterogeneous"
    assert model.rigidity_factor is None
    path = model.save(tmp_path / "het.slip.zip")
    back = SlipModel.load(path)
    assert back.rigidity_factor is None
    with pytest.raises(Exception):
        back.forward(x, y)


# -- the arbiter: a layered band recovered from the reference plus voxels ---------------------------
def _crust(band=None):
    """``TWO_LAYERS``, or with a band ``(top, bottom, mu factor, lam factor)`` cut into its half-space."""
    if band is None:
        return TWO_LAYERS
    top, bottom, f_mu, f_lam = band
    vp2, vs2, rho2 = 6100.0, 3500.0, 2750.0
    mu2, lam2 = rho2 * vs2 ** 2, rho2 * (vp2 ** 2 - 2.0 * vs2 ** 2)
    vs3 = np.sqrt(f_mu * mu2 / rho2)
    vp3 = np.sqrt((f_lam * lam2 + 2.0 * f_mu * mu2) / rho2)
    return VelocityModel([0.0, 2000.0, 2000.0, top, top, bottom, bottom, 60e3],
                         [3750.0, 3750.0, vp2, vp2, vp3, vp3, vp2, vp2],
                         [1950.0, 1950.0, vs2, vs2, vs3, vs3, vs2, vs2],
                         [2370.0, 2370.0, rho2, rho2, rho2, rho2, rho2, rho2])


def _slab_tables(model, strain):
    tops, _, _ = wn.layer_stack(model)
    z = wn.depth_grid(tops, z_max=40e3, first=10.0, growth=1.6, spacing=400.0, deep_spacing=1000.0)
    return wn.RemainderTables.compute(model, z=z, r=wn.distance_grid(per_decade=20), n_k=2048, z_max=40e3,
                                      surface_z=wn.surface_depth_grid(tops, fine_to=20e3, z_max=40e3),
                                      strain=strain)


def test_a_layered_band_is_recovered_from_the_reference_and_voxels():
    """Crust B (a soft band in the half-space) solved exactly by its own tables, against crust A plus voxels.

    The band (5-9 km, `mu x 0.5`, `lambda x 0.6`: Poisson's ratio changes too) is laterally
    infinite in B and +-16 km of 2 km voxels here, with the samples within +-8 km: samples
    over band material the voxels lack lose their local effect (with samples to +-25 km the
    full solve stays 14-17% off, whatever the band and the voxel size). This pins the sign of
    the moment density, the stiffness block, the packing and the operators together: a sign
    error is a 2x error.
    """
    band = (5000.0, 9000.0, 0.5, 0.6)
    A, B = _slab_tables(_crust(), True), _slab_tables(_crust(band), False)
    mesh, basis, (cx, cy), x, y, look = _small_problem(A, max_depth=2.9e3, reach=8e3, n=30)
    base = LayeredTDE(A)
    g_ref = base.los_matrix(mesh, x, y, *look, basis=basis)
    effect = LayeredTDE(B).los_matrix(mesh, x, y, *look, basis=basis) - g_ref
    lam_b, mu_b = wn.layer_stack(_crust(band))[1][2], wn.layer_stack(_crust(band))[2][2]
    v = VoxelModel(A, cx + np.arange(-16e3, 16001.0, 2e3), cy + np.arange(-16e3, 16001.0, 2e3),
                   [5e3, 7e3, 9e3], lambda xx, yy, dd: (np.full(xx.shape, lam_b), np.full(xx.shape, mu_b)))
    v.check_clearance(mesh)
    rms = {}
    shared = None
    for order in ("full", "born1"):
        engine = HeterogeneousEngine(base, v, order=order)
        if shared is not None:
            engine._gamma, engine._e, engine._h = shared
        correction = engine.los_matrix(mesh, x, y, *look, basis=basis) - g_ref
        shared = (engine.gamma, engine._e, engine._h)
        rms[order] = (np.sqrt(np.mean(correction ** 2)) / np.sqrt(np.mean(effect ** 2)),
                      np.sqrt(np.mean((correction - effect) ** 2)) / np.sqrt(np.mean(effect ** 2)))
    print({k: tuple(round(float(a), 3) for a in val) for k, val in rms.items()})
    ratio, error = rms["full"]
    assert 0.8 < ratio < 1.2 and error < 0.25
    assert rms["born1"][1] > error


def test_bodies_set_the_voxels_whose_centres_they_hold(coarse_tables):
    from slipsolve.heterogeneity import Box, Prism

    t = coarse_tables
    basin = Prism([[0.0, 0.0], [4e3, 0.0], [4e3, 4e3], [0.0, 4e3]], (0.0, 2e3))
    block = Box((2e3, 6e3), (2e3, 6e3), (2e3, 6e3))
    v = VoxelModel.from_bodies(t, np.arange(0.0, 8001.0, 2e3), np.arange(0.0, 8001.0, 2e3), [0.0, 2e3, 4e3, 6e3],
                               [(basin, (5e9, 4e9)), (block, "scale", (1.2, 0.8))])
    assert v.n_active == 4 + 4 * 2
    soft = v.mu[:2, :2, 0]
    assert np.all(soft == 4e9) and np.all(v.lam[:2, :2, 0] == 5e9)
    assert np.allclose(v.mu[1:3, 1:3, 1:3], 0.8 * t.mu[1])
    assert np.all(~v.active[3, 3, :])


def test_a_surface_voxel_leaves_the_free_surface_traction_free(uniform_tables):
    """At `z = 0` over a surface voxel (its eigenstrain subtracted) and around it, `sigma . e_z = 0`.

    The top face is dropped and the interior term carries the rest. The traction is about 3e-8 of
    the peak stress, inside the footprint and out, in uniform and two-layer crusts.
    """
    v = VoxelSource(LayeredTDE(uniform_tables), (-500.0, 700.0), (-300.0, 600.0), (0.0, 800.0))
    xs = np.array([0.0, 300.0, -250.0, 2000.0, -1800.0, 400.0])
    ys = np.array([100.0, 350.0, -100.0, 900.0, -2000.0, 2600.0])
    z = np.zeros(xs.size)
    inside = v.inside(xs, ys, z)
    assert inside[:3].all() and not inside[3:].any()
    elastic = v.strain(xs, ys, z) - np.where(inside[:, None, None], to_voigt(v.eigenstrain)[None], 0.0)
    t = from_voigt(elastic)
    sigma = v.lam * np.trace(t, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2.0 * v.mu * t
    assert np.abs(sigma[..., :, 2]).max() < 1e-6 * np.abs(sigma).max()


def test_a_long_thin_voxel_is_the_sum_of_its_cubes(uniform_tables):
    """Past `MAX_ASPECT` a voxel is computed as near-cubic pieces: its field near itself is the cubes' sum.

    Two triangles a face and a volume rule with as many points along every side under-resolve a long
    thin voxel next to itself: computed whole, a 7.84 x 1 x 1 km voxel is 3.8% of the peak off 16
    cubes' sum at its neighbours' centres, where 16 and 32 cubes agree to 7e-4.
    """
    from slipsolve.heterogeneity import MAX_ASPECT

    eng = LayeredTDE(uniform_tables)
    lx, h = 7840.0, 1000.0
    x0, y0, z0 = -lx / 2, 2000.0, 3000.0
    long = VoxelSource(eng, (x0, x0 + lx), (y0, y0 + h), (z0, z0 + h))
    assert long.pieces is not None and len(long.pieces) % 2 == 1
    assert max(np.ptp([p.lo, p.hi], axis=0).max() / np.ptp([p.lo, p.hi], axis=0).min() for p in long.pieces) <= MAX_ASPECT
    cube = VoxelSource(eng, (0.0, h), (y0, y0 + h), (z0, z0 + h))
    assert cube.pieces is None
    xs = np.linspace(x0, x0 + lx, 17)
    cubes = [VoxelSource(eng, (xs[i], xs[i + 1]), (y0, y0 + h), (z0, z0 + h)) for i in range(16)]
    pts = np.array([(dx, y0 + (0.5 + dy) * h, -(z0 + (0.5 + dz) * h))
                    for dx in (0.0, 0.45 * lx) for dy, dz in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.0))])
    got = long.strain(*pts.T)
    want = sum(c.strain(*pts.T) for c in cubes)
    assert (np.abs(got - want).max(axis=(1, 2)) / np.abs(want).max(axis=(1, 2))).max() < 3e-3


def test_points_on_voxel_faces_are_found_and_refused(uniform_tables):
    """A point on a voxel face is singular. `on_faces` finds it -- a voxel's own faces, the faces between a long
    voxel's pieces, a horizontal face between rows -- and the operator refuses it, naming it. A surface point over a
    surface voxel is not on a face (the top face is the free surface's)."""
    from slipsolve.heterogeneity import los_operator

    lam0, mu0 = float(uniform_tables.lam[0]), float(uniform_tables.mu[0])
    lam = np.full((3, 1, 2), lam0)
    mu = np.full((3, 1, 2), mu0)
    lam[[0, 2]], mu[[0, 2]] = 0.5 * lam0, 0.5 * mu0            # columns 0 and 2 soft, column 1 the reference's
    vox = VoxelModel(uniform_tables, [0.0, 1000.0, 2000.0, 3000.0], [0.0, 3000.0], [0.0, 1000.0, 2000.0],
                     lam=lam, mu=mu)                         # 1 x 3 x 1 km voxels: three pieces along y
    cases = [((0.0, 1500.0, 0.0), True),                     # a soft voxel's outer face, at the surface
             ((1000.0, 1500.0, 0.0), True),                  # its face against the reference column
             ((1500.0, 1500.0, 0.0), False),                 # inside the reference column
             ((500.0, 1000.0, 0.0), True),                   # between two of its pieces
             ((500.0, 1500.0, 0.0), False),                  # over it, mid-piece: the top is the free surface
             ((500.0, 1500.0, -1000.0), True),               # the face between its two rows
             ((500.0, 1001.0, 0.0), False),                  # 1 m off the piece face
             ((3000.0001, 1500.0, -500.0), True)]            # within rounding of the last soft voxel's face
    pts = np.array([c[0] for c in cases])
    np.testing.assert_array_equal(vox.on_faces(*pts.T), [c[1] for c in cases])
    with pytest.raises(ValueError, match="lie on voxel faces"):
        los_operator(vox, LayeredTDE(uniform_tables), [1000.0, 9000.0], [1500.0, 9000.0])


def test_voxels_may_end_on_the_fault_with_contact_but_not_straddle_it(uniform_tables):
    """`check_clearance(contact=True)`: voxels on one side of a planar fault, a face on its plane, pass; voxels the
    fault runs through are refused; without `contact` the first are refused too (they lie within a voxel of it)."""
    from slipsolve import FaultMesh, FaultTrace

    trace = FaultTrace.from_lonlat(np.array([-80.05, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=6e3, edge_length=2e3)
    tx, ty = (np.asarray(v) for v in trace.to_local(frame))
    y0 = float(np.mean(ty))                                       # the fault's plane, y = const in this frame

    def model(y_lo):
        lam0, mu0 = float(uniform_tables.lam[0]), float(uniform_tables.mu[0])
        return VoxelModel(uniform_tables, np.linspace(-3000.0, 3000.0, 4), [y_lo, y_lo + 1000.0], [0.0, 1000.0, 2000.0],
                          lam=np.full((3, 1, 2), 0.5 * lam0), mu=np.full((3, 1, 2), 0.5 * mu0))

    assert abs(ty.max() - ty.min()) < 1e-6
    ends_on_it = model(y0)
    assert ends_on_it.check_clearance(mesh, contact=True)
    with pytest.raises(ValueError, match="within"):
        ends_on_it.check_clearance(mesh)
    with pytest.raises(ValueError, match="passes through"):
        model(y0 - 500.0).check_clearance(mesh, contact=True)


def test_workers_give_the_serial_operators(uniform_tables, tmp_path):
    """`workers` > 1: processes that each load the tables from their file give the serial E, H and Gamma to rounding."""
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve import wavenumber as wn
    from slipsolve.basis import make_basis
    from slipsolve.heterogeneity import los_operator, strain_operator

    tables = wn.RemainderTables.load(uniform_tables.save(tmp_path / "uniform.npz"))
    engine = LayeredTDE(tables)
    trace = FaultTrace.from_lonlat(np.array([-80.05, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=6e3, edge_length=2e3)
    _, ty = (np.asarray(v) for v in trace.to_local(frame))
    y0 = float(np.mean(ty))
    lam0, mu0 = float(tables.lam[0]), float(tables.mu[0])
    vox = VoxelModel(tables, [-4000.0, -2000.0, 0.0, 2000.0], [y0 + 3000.0, y0 + 5000.0], [0.0, 2000.0, 4000.0],
                     lam=np.full((3, 1, 2), 0.5 * lam0), mu=np.full((3, 1, 2), 0.5 * mu0))
    c = vox.centres
    for basis in (None, make_basis(mesh, "node")):
        serial = engine.strain(mesh, c[:, 0], c[:, 1], c[:, 2], basis=basis)
        pooled = engine.strain(mesh, c[:, 0], c[:, 1], c[:, 2], basis=basis, workers=2)
        np.testing.assert_allclose(pooled, serial, rtol=1e-12, atol=1e-12 * np.abs(serial).max())
    rng = np.random.default_rng(4)
    x, y = rng.uniform(-9e3, 9e3, 30), y0 + rng.uniform(-9e3, 9e3, 30)
    look = (np.full(30, 0.6), np.full(30, -0.1), np.full(30, 0.79))
    np.testing.assert_allclose(los_operator(vox, engine, x, y, look, workers=2), los_operator(vox, engine, x, y, look),
                               rtol=1e-12, atol=0)
    serial = strain_operator(vox, engine)
    np.testing.assert_allclose(strain_operator(vox, engine, workers=2), serial, rtol=1e-12, atol=1e-12 * np.abs(serial).max())


def test_one_pool_serves_the_engine_operators(uniform_tables, tmp_path):
    """`HeterogeneousEngine(workers=2)`: one pool for H, E and Gamma, the serial engine's G to rounding."""
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve import wavenumber as wn

    tables = wn.RemainderTables.load(uniform_tables.save(tmp_path / "uniform.npz"))
    base = LayeredTDE(tables)
    trace = FaultTrace.from_lonlat(np.array([-80.05, -79.95]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=6e3, edge_length=2e3)
    _, ty = (np.asarray(v) for v in trace.to_local(frame))
    y0 = float(np.mean(ty))
    lam0, mu0 = float(tables.lam[0]), float(tables.mu[0])
    vox = VoxelModel(tables, [-4000.0, -2000.0, 0.0, 2000.0], [y0 + 3000.0, y0 + 5000.0], [0.0, 2000.0, 4000.0],
                     lam=np.full((3, 1, 2), 0.5 * lam0), mu=np.full((3, 1, 2), 0.5 * mu0))
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-9e3, 9e3, 25), y0 + rng.uniform(-9e3, 9e3, 25)
    look = (np.full(25, 0.6), np.full(25, -0.1), np.full(25, 0.79))
    serial = HeterogeneousEngine(base, vox).los_matrix(mesh, x, y, *look)
    pooled_engine = HeterogeneousEngine(base, vox, workers=2)
    pooled = pooled_engine.los_matrix(mesh, x, y, *look)
    assert pooled_engine._pool is not None
    pooled_engine.close()
    np.testing.assert_allclose(pooled, serial, rtol=1e-12, atol=1e-12 * np.abs(serial).max())


# -- the strain far path: point moments by a volume rule -----------------------------------------
def _around_voxel(v, n, lo, hi, seed, z_max=30e3):
    """`n` points at a box distance of `lo`-`hi` voxel sizes from `v`, at or below the surface."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        p = v.centre + rng.uniform(0.0, 2.0 * hi) * v.size * d
        gap = v._gap(*(np.array([c]) for c in p))[0] / v.size
        if lo < gap <= hi and -z_max < p[2] <= 0.0:
            pts.append(p)
    return np.array(pts).T


def test_moment_sources_make_the_unit_moments():
    """Openings and double couples of unit potency, combined, are the six unit moment densities."""
    from slipsolve.heterogeneity import moment_sources

    lam, mu = 2.7e10, 3.3e10
    sources, coefficients = moment_sources(lam, mu)
    moments = np.array([mu * (np.outer(n, d) + np.outer(d, n)) + lam * float(n @ d) * np.eye(3)
                        for n, slips in sources for d in slips])
    assert np.allclose(np.einsum("ks,sab->kab", coefficients, moments), unit_moments(), rtol=0, atol=1e-12)


def test_the_far_strain_is_the_faces_in_a_uniform_crust(uniform_tables):
    """Beyond `strain_far` sizes, Okada point moments by a Gauss volume rule against the faces (no remainder).

    At box distances of 2-20 sizes they agree to 4e-4 (buried) and 1.2e-3 (at the surface) of each receiver's
    peak. On layered tables, where the remainder is not zero, the two paths agree as well, except across an
    interface from the source, where both sit on the tables' trilinear floor.
    """
    engine = LayeredTDE(uniform_tables)
    for seed, box in enumerate((((0.0, 1e3), (0.0, 1e3), (3e3, 4e3)), ((0.0, 1e3), (0.0, 1e3), (0.0, 1e3)),
                                ((0.0, 2e3), (0.0, 1e3), (5e3, 6e3)))):
        far, near = VoxelSource(engine, *box), VoxelSource(engine, *box, strain_far=None)
        x, y, z = _around_voxel(far, 25, 2.0, 15.0, seed)
        assert far._far_strain(x, y, z).all() and not near._far_strain(x, y, z).any()
        a, b = far.strain(x, y, z), near.strain(x, y, z)
        error = np.abs(a - b).max(axis=(1, 2)) / np.abs(b).max(axis=(1, 2))
        assert error.max() < 3e-3


def test_the_far_path_reads_the_remainder_at_the_near_paths_points(coarse_tables, monkeypatch):
    """With the half-space part removed, the far path's sum is the near path's remainder rule at the same order."""
    import slipsolve.heterogeneity as het

    engine = LayeredTDE(coarse_tables)
    v = VoxelSource(engine, (0.0, 1e3), (0.0, 1e3), (500.0, 1500.0))
    x, y, z = _around_voxel(v, 12, 2.0, 10.0, seed=3, z_max=20e3)
    monkeypatch.setattr(het, "point_source_enu",
                        lambda nu, normal, slips, px, *rest: (None, np.zeros((np.size(px), len(slips), 6))))
    for order in (1, 2, 3):
        monkeypatch.setattr(v, "_orders", lambda x, y, z, whole, n=order: np.full(np.size(x), n))
        far = v._point_moments(x, y, z, orders=order)
        near = v._remainder(x, y, z, True, whole=False)
        np.testing.assert_allclose(far, near, rtol=1e-12, atol=1e-12 * np.abs(near).max())
