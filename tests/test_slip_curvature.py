"""Earth's curvature and gravity as a correction to the flat engines (`slipsolve.curvature`).

Oracles: the sphere becomes the flat layers as its radius grows, the difference falling as `1 / R`; Betti's reciprocal
theorem degree by degree, against the same sphere loaded at its surface, which never uses the source's jumps; a
deviatoric source changes no volume, so a uniform sphere's degree-0 response vanishes and a layered one's is Betti's
with a radial load solved here from scratch; the engine wrapper adds exactly the tables' correction.
"""

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace, HalfSpaceTDE
from slipsolve import curvature as cv
from slipsolve import wavenumber as wn
from slipsolve.basis import make_basis
from slipsolve.crust import VelocityModel
from slipsolve.inversion import TOPOGRAPHY_STEP, _blocked, assemble_design

R = 6371e3
CRUST = VelocityModel([0.0, 16e3, 16e3, 30e3, 30e3], [6000.0, 6000.0, 6700.0, 6700.0, 8000.0],
                      [3460.0, 3460.0, 3870.0, 3870.0, 4620.0], [2600.0, 2600.0, 2800.0, 2800.0, 3400.0])
DEEP = VelocityModel([0.0, 16e3, 16e3, 100e3, 100e3, 200e3, 200e3],
                     [6000.0, 6000.0, 8000.0, 8000.0, 8500.0, 8500.0, 9000.0],
                     [3460.0, 3460.0, 4500.0, 4500.0, 4700.0, 4700.0, 5000.0],
                     [2600.0, 2600.0, 3300.0, 3300.0, 3400.0, 3400.0, 3500.0])
UNIFORM = VelocityModel([0.0, 50e3], [6000.0, 6000.0], [3464.1016, 3464.1016], [2700.0, 2700.0])


def test_the_sphere_becomes_the_flat_layers():
    tops, lam, mu = wn.layer_stack(CRUST)
    depth, r = 8e3, np.array([5e3, 10e3, 20e3, 50e3])
    flat = cv._flat_field(tops, lam, mu, depth, r, 1e-3)
    scale = np.abs(flat).max(axis=0)
    difference = {}
    for radius in (R / 2, 2 * R):
        sphere = cv._sphere_field(radius, tops, lam, mu, depth, r / radius, int(4e-3 * radius), 1e-3)
        difference[radius] = sphere - flat
    assert np.all(np.abs(difference[2 * R]) < 0.01 * scale)                 # the sphere is the flat layers, nearly
    big = np.abs(difference[2 * R]) > 0.02 * np.abs(difference[2 * R]).max(axis=0)
    ratio = difference[R / 2][big] / difference[2 * R][big]
    assert np.all(np.abs(ratio - 4.0) < 0.35) and abs(np.median(ratio) - 4.0) < 0.03   # the rest goes as 1 / R


def _load_solution(n, bounds, lams, mus, src, scale, kind):
    """The sphere loaded at its surface by a unit normal (`"s"`) or toroidal (`"t"`) traction, no source:
    `y` at the source radius, from the shell above it."""
    per = 4 if kind == "s" else 2
    A, load, _, _ = cv._global(n, R, bounds, lams, mus, src, np.zeros((n.size, per, 1)), scale, kind)
    load[:, 0, 0] = 1.0 / scale
    i = src - 1
    coef = np.linalg.solve(A, load)[:, per * i:per * (i + 1), 0]
    top, bottom = bounds[i], bounds[i + 1]
    if kind == "s":
        cols = cv._sph_columns(n, bounds[src], [top, top, bottom, bottom], lams[i], mus[i], [0, 1, 2, 3], scale)
    else:
        cols = cv._tor_columns(n, bounds[src], [top, bottom], mus[i], [0, 1], scale)
    y = np.einsum("dri,di->dr", cols, coef)
    y[:, per // 2:] *= scale[:, None]
    return y


@pytest.mark.parametrize("depth", [3e3, 20e3])
def test_reciprocity_degree_by_degree(depth):
    tops, lam, mu = wn.layer_stack(CRUST)
    n = np.array([2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    L = n * (n + 1.0)
    r_s = R - depth
    j = int(np.searchsorted(tops, depth, side="right") - 1)
    U, V, W = cv._degrees(R, tops, lam, mu, depth, 1000)
    U, V, W = U[n.astype(int)], V[n.astype(int)], W[n.astype(int)]
    bounds, lams, mus, src = cv._shells(R, tops, lam, mu, r_s)
    scale = wn.MU_SCALE * n / R
    U1, V1, P1, S1 = _load_solution(n, bounds, lams, mus, src, scale, "s").T
    W1, T1 = _load_solution(n, bounds, lams, mus, src, scale, "t").T
    c = (2.0 * n + 1.0) / (4.0 * np.pi * R ** 2)
    m_, l_ = mu[j], lam[j]
    e_rr = (P1 - l_ * (2.0 * U1 - L * V1) / r_s) / (l_ + 2.0 * m_)
    e_tt = (U1 - L * V1 / 2.0) / r_s
    for ours, betti in ((U[:, 0], -m_ * V1 * c / r_s), (W[:, 0], m_ * W1 * c / (L * r_s)),     # strike-slip
                        (U[:, 1], S1 * c), (W[:, 1], T1 * c / L),                              # dip-slip
                        (U[:, 2], m_ * (e_rr - e_tt) * c)):                                    # CLVD
        assert np.allclose(ours, betti, rtol=1e-8, atol=0.0)


def test_the_deep_layers_are_dropped_only_where_they_cannot_reach():
    tops, lam, mu = wn.layer_stack(DEEP)
    depth = 10e3
    U, V, W = cv._degrees(R, tops, lam, mu, depth, 6000)                     # three chunks, the deep layers cut
    n = np.array([2500.0, 4500.0, 6000.0])
    sph, tor = cv._jumps(n, lam[0], mu[0], R - depth)
    bounds, lams, mus, src = cv._shells(R, tops, lam, mu, R - depth)
    scale = wn.MU_SCALE * n / R
    for kind, jumps, per, ours in (("s", sph, 4, U[n.astype(int)]), ("t", tor, 2, W[n.astype(int)])):
        A, rhs, top, first = cv._global(n, R, bounds, lams, mus, src, jumps, scale, kind)
        full = np.einsum("dri,dik->drk", top, cv._solve(A, rhs, n, first)[:, 0:per, :])[:, 0]
        assert np.allclose(ours, full, rtol=1e-6, atol=0.0)                    # rounding of the shell solve


def test_degree_zero():
    tops, lam, mu = wn.layer_stack(UNIFORM)
    U, _, _ = cv._degrees(R, tops, lam, mu, 10e3, 3)
    assert abs(U[0, 2]) < 1e-12 * abs(U[2, 2])                               # no volume change, no mean motion
    tops, lam, mu = wn.layer_stack(CRUST)
    depth = 10e3
    U, _, _ = cv._degrees(R, tops, lam, mu, depth, 3)
    # A uniform surface pressure: U = a r + b / r^2 in each shell, U = a r in the ball, solved from scratch.
    radii = R - tops
    m = tops.size
    A, rhs = np.zeros((2 * m - 1, 2 * m - 1)), np.zeros(2 * m - 1)

    def rows(i, r):
        k = 3.0 * lam[i] + 2.0 * mu[i]
        if i == m - 1:
            return np.array([[r], [k]])
        return np.array([[r, r ** -2.0], [k, -4.0 * mu[i] * r ** -3.0]])

    A[0, 0:2], rhs[0] = rows(0, R)[1], 1.0
    for i in range(m - 1):
        A[1 + 2 * i:3 + 2 * i, 2 * i:2 * i + 2] = rows(i, radii[i + 1])
        A[1 + 2 * i:3 + 2 * i, 2 * i + 2:2 * i + 4 if i + 1 < m - 1 else 2 * i + 3] = -rows(i + 1, radii[i + 1])
    coef = np.linalg.solve(A, rhs)
    j = int(np.searchsorted(tops, depth, side="right") - 1)
    a, b = coef[2 * j], coef[2 * j + 1]
    r_s = R - depth
    e_rr, e_tt = a - 2.0 * b / r_s ** 3, a + b / r_s ** 3
    assert np.isclose(U[0, 2], mu[j] * (e_rr - e_tt) / (4.0 * np.pi * R ** 2), rtol=1e-8, atol=0.0)


@pytest.fixture(scope="module")
def tables():
    return cv.CurvatureTables.compute(UNIFORM, r_max=200e3, nr=201, z_min=500.0, z_max=15500.0, nz=16, k_taper=5e-4)


def test_tables_round_trip_and_refuse_an_interface(tables, tmp_path):
    tables.save(tmp_path / "c.npz")
    back = cv.CurvatureTables.load(tmp_path / "c.npz")
    assert back.attrs == tables.attrs and np.array_equal(back.r, tables.r) and np.array_equal(back.z, tables.z)
    assert all(np.array_equal(back.values[k][c], tables.values[k][c]) for k, cs in cv.COMPONENTS.items() for c in cs)
    with pytest.raises(ValueError, match="interface"):
        cv.CurvatureTables.compute(CRUST, r_max=50e3, nr=11, z_min=12e3, z_max=20e3, nz=3)


def test_gravity_is_the_flat_layers_buoyancy(tables):
    curvature_only = cv.CurvatureTables.compute(UNIFORM, r_max=200e3, nr=201, z_min=500.0, z_max=15500.0, nz=16,
                                                k_taper=5e-4, gravity=0.0)
    tops, lam, mu = wn.layer_stack(UNIFORM)
    rho = np.full(tops.size, 2700.0)
    depth, row = 7.5e3, 7
    flat = cv._flat_field(tops, lam, mu, depth, tables.r, 5e-4)
    buoyant = cv._flat_field(tops, lam, mu, depth, tables.r, 5e-4, buoyancy=rho * cv.GRAVITY)
    for c, (kind, comp) in enumerate((k, c) for k, cs in cv.COMPONENTS.items() for c in cs):
        gained = tables.values[kind][comp][:, row] - curvature_only.values[kind][comp][:, row]
        tol = 1e-12 * np.abs(tables.values[kind][comp]).max()
        assert np.allclose(gained, buoyant[:, c] - flat[:, c], rtol=0.0, atol=tol)


@pytest.fixture(scope="module")
def fault():
    trace = FaultTrace(np.array([-120.0, -120.0]), np.array([35.0, 35.25]), name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=4e3)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-120e3, 120e3, 60), rng.uniform(-80e3, 100e3, 60)
    x = np.where(np.abs(x) < 3e3, 3e3, x)
    look = np.tile(np.array([0.6, -0.1, 0.7937]) / np.linalg.norm([0.6, -0.1, 0.7937]), (x.size, 1)).T
    return mesh, x, y, look


def test_the_engine_adds_the_tables(tables, fault):
    mesh, x, y, look = fault
    base, engine = HalfSpaceTDE(), cv.Curvature(HalfSpaceTDE(), tables)
    for kind in ("element", "node"):
        basis = make_basis(mesh, kind)
        g = assemble_design(engine, mesh, basis, (x, y, *look))
        tol = 1e-12 * np.abs(g).max()
        correction = engine.correction(mesh, x, y, look, basis=basis)
        assert np.abs(correction).max() > 1e-4 * np.abs(g).max()             # there is something to add
        assert np.allclose(g, assemble_design(base, mesh, basis, (x, y, *look)) + correction, rtol=0.0, atol=tol)
        enu = engine.displacement(mesh, x, y, basis=basis)
        assert np.allclose(np.einsum("pcn,cp->pn", enu, look), g, rtol=0.0, atol=tol)
        slip = np.linspace(0.0, 1.0, g.shape[1])
        assert np.allclose(engine.forward(mesh, slip, x, y, look=look, basis=basis), g @ slip, rtol=0.0,
                           atol=tol * g.shape[1])
    basis = make_basis(mesh, "node")
    tents = engine.correction(mesh, x, y, look, basis=basis)
    means = engine.correction(mesh, x, y, look) @ _blocked(basis.projection())
    assert np.abs(tents - means).max() < 0.05 * np.abs(tents).max()        # tents or element means: a smooth field
    with pytest.raises(ValueError, match="surface"):
        engine.displacement(mesh, x, y, z=np.full(x.size, -100.0))


def test_topography_lowers_the_correction_with_the_fault(tables, fault):
    mesh, x, y, look = fault
    engine, basis = cv.Curvature(HalfSpaceTDE(), tables), make_basis(mesh, "node")
    heights = TOPOGRAPHY_STEP * np.arange(x.size) % 600.0
    g = assemble_design(engine, mesh, basis, (x, y, *look), elevation=heights)
    rows = [assemble_design(engine, mesh.lowered(h), basis, (x[i:i + 1], y[i:i + 1], *look[:, i:i + 1]))
            for i, h in enumerate(heights)]
    assert np.allclose(g, np.vstack(rows), rtol=0.0, atol=1e-12 * np.abs(g).max())
    native = cv.Curvature(_NativeTopography(), tables)                   # a base that takes the elevations itself
    assert np.allclose(assemble_design(native, mesh, basis, (x, y, *look), elevation=heights), g, rtol=0.0,
                       atol=1e-12 * np.abs(g).max())


class _NativeTopography:
    """The half-space engine, taking the elevations itself as the voxel engine does."""

    supports_basis = False
    native_topography = True
    name = "native_half_space"

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, elevation=None, step=None):
        points, element = (x, y, look_e, look_n, look_u), make_basis(mesh, "element")
        return assemble_design(HalfSpaceTDE(), mesh, element, points, elevation=elevation, step=step or TOPOGRAPHY_STEP)
