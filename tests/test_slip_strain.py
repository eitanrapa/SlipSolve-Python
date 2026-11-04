"""Strain kernels for the 3-D heterogeneity engine: the half-space TDE and the layered remainder.

The heterogeneous engine (``HETEROGENEOUS_3D_PLAN.local.md``) needs the strain of every
fault element at every voxel centre, at depth, and the strain of point moment sources in
the layered reference. These pin the strain against independent references: ``cutde``'s
analytic triangular-dislocation strain, the free surface's traction-free condition, and
finite differences of the displacement kernels already validated elsewhere.
"""

import numpy as np
import pytest

from slipsolve._tde import STRAIN_COMPONENTS, tde_disp_hs_unit, tde_strain_hs_unit

NU = 0.25

#: Triangles that exercise the TDE's branches: a dipping one, a vertical one reaching the
#: surface (horizontal top edge in the free surface), a buried vertical one, a shallow dip.
TRIANGLES = {
    "dipping": np.array([[0.0, 0.0, -5e3], [8e3, 1e3, -5e3], [4e3, -2e3, -12e3]]),
    "vertical": np.array([[0.0, 0.0, 0.0], [4e3, 0.0, 0.0], [0.0, 0.0, -4e3]]),
    "vertical_buried": np.array([[0.0, 0.0, -2e3], [4e3, 0.0, -2e3], [4e3, 0.0, -6e3]]),
    "shallow_dip": np.array([[0.0, 0.0, -1e3], [5e3, 0.0, -1e3], [2e3, 4e3, -2.5e3]]),
}


def _size(tri):
    return np.sqrt(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def _receivers(tri, n=120, near=0.03, far=60.0, seed=0):
    """Points at depth from ``near`` to ``far`` triangle sizes, plus a row at the surface."""
    rng = np.random.default_rng(seed)
    size, centre = _size(tri), tri.mean(axis=0)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    pts = centre + dirs * (size * np.geomspace(near, far, n))[:, None]
    pts[:, 2] = -np.abs(pts[:, 2]) - 1.0
    surface = np.column_stack([centre[0] + size * np.geomspace(0.05, 20.0, 15),
                               np.full(15, centre[1] + 0.3 * size), np.zeros(15)])
    return np.vstack([pts, surface])


@pytest.mark.parametrize("name", list(TRIANGLES))
def test_the_strain_matches_cutde(name):
    """Central differences against ``cutde``'s analytic strain, receivers at depth and at the surface.

    ``cutde`` orders strain ``(xx, yy, zz, xy, xz, yz)`` with tensor shears and slip
    ``(strike, dip, tensile)``, as :data:`STRAIN_COMPONENTS` and the TDE do. Measured
    worst 1.4e-7 of each point's peak within 10 sizes (2026-11-04); the bar is 1e-6.
    """
    hs = pytest.importorskip("cutde.halfspace")
    tri = TRIANGLES[name]
    pts = _receivers(tri)
    mine = tde_strain_hs_unit(*pts.T, *tri, NU)
    theirs = np.transpose(hs.strain_matrix(pts, tri[None], NU)[:, :, 0, :], (2, 1, 0))
    scale = np.abs(theirs).max(axis=(0, 1))
    error = np.abs(mine - theirs).max(axis=(0, 1)) / scale
    assert STRAIN_COMPONENTS == ("xx", "yy", "zz", "xy", "xz", "yz")
    assert error.max() < 1e-6, f"worst {error.max():.2e} at point {error.argmax()}"


@pytest.mark.parametrize("name", ["dipping", "vertical", "shallow_dip"])
def test_the_free_surface_is_traction_free(name):
    """At ``z = 0`` (the one-sided branch) the strain carries no traction on a horizontal plane.

    ``sigma_xz = sigma_yz = 0`` and ``sigma_zz = lam tr(eps) + 2 mu eps_zz = 0``: a check of
    the one-sided difference against physics, not against another code.
    """
    tri = TRIANGLES[name]
    size, centre = _size(tri), tri.mean(axis=0)
    rng = np.random.default_rng(3)
    xy = centre[:2] + size * rng.uniform(-6.0, 6.0, size=(60, 2))
    eps = tde_strain_hs_unit(xy[:, 0], xy[:, 1], np.zeros(60), *tri, NU)
    lam_over_mu = 2.0 * NU / (1.0 - 2.0 * NU)
    trace = eps[:, 0] + eps[:, 1] + eps[:, 2]
    szz_over_mu = lam_over_mu * trace + 2.0 * eps[:, 2]
    scale = np.abs(eps).max(axis=1)
    for traction in (szz_over_mu, 2.0 * eps[:, 4], 2.0 * eps[:, 5]):
        assert np.all(np.abs(traction) < 1e-5 * scale)


def test_the_step_does_not_matter():
    """Steps a decade apart agree to the truncation error of the larger one."""
    tri = TRIANGLES["dipping"]
    pts = _receivers(tri, n=60, far=20.0)
    a = tde_strain_hs_unit(*pts.T, *tri, NU, step=1e-4)
    b = tde_strain_hs_unit(*pts.T, *tri, NU, step=1e-3)
    scale = np.abs(a).max(axis=(0, 1))
    assert (np.abs(a - b).max(axis=(0, 1)) / scale).max() < 5e-5


def test_the_strain_is_the_symmetric_gradient_of_the_displacement():
    """A wide difference of the displacement along a diagonal agrees with the strain's projection.

    ``d/ds (u . e) = e . eps . e`` for a unit direction ``e``: a check of the component
    order and the symmetrisation independent of ``cutde``.
    """
    tri = TRIANGLES["vertical_buried"]
    e = np.array([0.48, -0.6, 0.64])
    e /= np.linalg.norm(e)
    h = 2e-4 * _size(tri)
    pts = _receivers(tri, n=40, near=1.0, far=10.0, seed=5)
    pts = pts[pts[:, 2] < -2.0 * h]                  # the stencil stays below the surface
    eps = tde_strain_hs_unit(*pts.T, *tri, NU)
    plus = tde_disp_hs_unit(*(pts + h * e).T, *tri, NU)
    minus = tde_disp_hs_unit(*(pts - h * e).T, *tri, NU)
    want = np.einsum("scp,c->sp", plus - minus, e) / (2.0 * h)
    full = np.empty((3, 3, 3, pts.shape[0]))
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        full[:, i, j] = full[:, j, i] = eps[:, k]
    got = np.einsum("sijp,i,j->sp", full, e, e)
    assert np.allclose(got, want, rtol=0, atol=1e-5 * np.abs(want).max())


def test_a_point_on_the_triangle_has_no_strain():
    """On the face (here a rounding error off it, the centroid) or an edge: NaN, not a zero."""
    tri = TRIANGLES["dipping"]
    centroid = tri.mean(axis=0)
    edge = 0.5 * (tri[0] + tri[1])
    x, y, z = np.column_stack([centroid, edge, [30e3, 0.0, -3e3]])
    eps = tde_strain_hs_unit(x, y, z, *tri, NU)
    assert np.all(np.isnan(eps[:, :, :2])) and np.all(np.isfinite(eps[:, :, 2]))


# -- the layered remainder's derivatives (slipsolve.wavenumber) -------------------------
from scipy.special import gamma  # noqa: E402

from slipsolve import wavenumber as wn  # noqa: E402
from slipsolve.edgrn import VelocityModel  # noqa: E402

#: The test crusts of ``test_slip_wavenumber.py``: a soft 2 km layer (Poisson 0.315) over a
#: half-space (0.255), and a uniform one.
TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])
ONE_LAYER = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)


@pytest.mark.parametrize("remainder", [False, True], ids=["whole", "remainder"])
@pytest.mark.parametrize("z_s", [1200.0, 6000.0])
def test_the_depth_derivatives_are_differences_of_the_kernels(remainder, z_s):
    """``U'``, ``V'``, ``W'`` from the tractions against central differences of ``U``, ``V``, ``W``.

    Receivers in both layers, off the source and the interface; all four sources. This
    pins the traction rows' scaling and the receiver layer's constants.
    """
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, _ = wn.wavenumbers(n=1024)
    z_r = np.array([300.0, 1500.0, 3500.0, 9000.0])
    delta = 1e-3
    got = wn.kernels(tops, lam, mu, z_s, z_r, k, remainder=remainder, derivatives=True)
    plain = wn.kernels(tops, lam, mu, z_s, z_r, k, remainder=remainder)
    plus = wn.kernels(tops, lam, mu, z_s, z_r + delta, k, remainder=remainder)
    minus = wn.kernels(tops, lam, mu, z_s, z_r - delta, k, remainder=remainder)
    for a, b in zip(got[:3], plain):
        assert np.array_equal(a, b)
    for name, d, p, m in zip(("U'", "V'", "W'"), got[3:], plus, minus):
        fd = (p - m) / (2.0 * delta)
        scale = np.abs(d).max(axis=1, keepdims=True)
        assert np.all(np.abs(d - fd) <= 1e-6 * scale + 1e-300), name


def test_the_remainders_depth_derivatives_are_continuous_at_the_source():
    """Both fields jump alike at the source, so the remainder's derivatives do not jump; the whole field's do."""
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, _ = wn.wavenumbers(n=1024)
    z_s, delta = 6000.0, 1e-3
    for remainder in (True, False):
        above = wn.kernels(tops, lam, mu, z_s, [z_s - delta], k, remainder=remainder, derivatives=True)
        below = wn.kernels(tops, lam, mu, z_s, [z_s + delta], k, remainder=remainder, derivatives=True)
        jumps = [np.abs(a - b).max() / np.abs(a).max() for a, b in zip(above, below)
                 if np.abs(a).max() > 0]
        if remainder:
            assert max(jumps) < 1e-5
        else:
            assert max(jumps) > 1e-2


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_the_radial_derivative_of_a_hankel_transform(order):
    """``int k^n e^{-ak} J_n(kr) k dk = 2a (2r)^n Gamma(n + 3/2) / (sqrt(pi) (a^2 + r^2)^(n + 3/2))``.

    Gradshteyn & Ryzhik 6.623.2. Its ``r``-derivative tests the ``J_{n-1}``, ``J_{n+1}``
    recipe, ``J_{-1} = -J_1`` included, on the solver's own wavenumber grid. At FFTLog's
    own nodes the transform is exact to 5e-14; what is left (4e-6 to 4e-5 of the peak,
    measured 2026-11-04) is the linear interpolation in ``log r`` between them, which every
    table shares and which the tables' own 25-50 nodes a decade dwarf.
    """
    k, dln = wn.wavenumbers()
    a = 1000.0
    f = k ** order * np.exp(-a * k)
    r = np.geomspace(20.0, 50e3, 40)
    c = 2.0 * a * 2.0 ** order * gamma(order + 1.5) / np.sqrt(np.pi)
    q = a * a + r * r
    value = c * r ** order * q ** -(order + 1.5)
    slope = c * (order * r ** (order - 1.0) * q ** -(order + 1.5)
                 - (2 * order + 3) * r ** (order + 1.0) * q ** -(order + 2.5))
    got_value = wn._hankel_at(f, k, dln, order, np.log(r))
    got_slope = wn._hankel_dr_at(f, k, dln, order, np.log(r))
    assert np.abs(got_value - value).max() < 1e-4 * np.abs(value).max()
    assert np.abs(got_slope - slope).max() < 1e-4 * np.abs(slope).max()
    if order >= 1:
        # Over r by the recurrence, down to 1 cm from the axis, where the division is ill-posed.
        near = np.geomspace(0.01, 50e3, 60)
        qn = a * a + near * near
        over = c * near ** (order - 1.0) * qn ** -(order + 1.5)
        got_over = wn._hankel_over_r_at(f, k, dln, order, np.log(near))
        assert np.abs(got_over - over).max() < 1e-4 * np.abs(over).max()


@pytest.mark.parametrize("remainder", [False, True], ids=["whole", "remainder"])
def test_the_radial_columns_are_the_derivatives_of_the_columns(remainder):
    """``transform(..., radial=True)`` against a difference of ``transform`` in ``r``.

    The step is 2.5 FFTLog nodes, so the linear interpolation in ``log r`` does not
    dominate; the bar (3e-3 of each column's peak) is that interpolation's.
    """
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers()
    z_r = [0.0, 1500.0, 7000.0] if remainder else [0.0, 1500.0]
    U, V, W = wn.kernels(tops, lam, mu, 3000.0, z_r, k, remainder=remainder)
    r = np.geomspace(300.0, 60e3, 30)
    step = 2.5 * dln
    got = wn.transform(U, V, W, k, dln, r, radial=True)
    fd = (wn.transform(U, V, W, k, dln, r * np.exp(step))
          - wn.transform(U, V, W, k, dln, r * np.exp(-step))) / (2.0 * r[:, None] * np.sinh(step))
    for col, key in enumerate(wn.TABLE_KEYS):
        scale = np.abs(got[..., col]).max(axis=-1, keepdims=True)
        assert np.all(np.abs(got[..., col] - fd[..., col]) <= 3e-3 * scale + 1e-300), key


# -- the strain of a point moment source (slipsolve.layered_tde.combine_strain) ----------
from slipsolve.edgrn import _point_dislocation  # noqa: E402
from slipsolve.layered_tde import combine_strain, moment_packing, tensor_packing  # noqa: E402

#: (normal, slip direction) in east/north/up; ``None`` slip is an opening.
POINT_SOURCES = {"ss": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                 "ds": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
                 "oblique": ((0.3, 0.8, 0.52), (0.85, -0.4, 0.3)),
                 "opening": ((0.2, -0.5, 0.84), None)}


def _unit_pair(name):
    n, d = POINT_SOURCES[name]
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    if d is None:
        return n, n.copy()
    d = np.asarray(d, dtype=float)
    d = d - (d @ n) * n
    return n, d / np.linalg.norm(d)


def _cutde_point_strain(n, d, z_s, x, y, z, size=1.0):
    """The strain per unit potency of a 1 m triangle, by ``cutde``'s closed form, ``(npts, 6)``."""
    hs = pytest.importorskip("cutde.halfspace")
    p1, p2, p3, comp = _point_dislocation(n, d, z_s, size)
    tri = np.array([p1, p2, p3])
    matrix = hs.strain_matrix(np.column_stack([x, y, z]), tri[None], float(ONE_LAYER.poisson()))
    return np.einsum("pcs,s->pc", matrix[:, :, 0, :], np.array(comp)) / (0.5 * size ** 2)


def _whole_columns(z_s, z_r, r):
    """A uniform crust's whole-field columns and its 27 strain columns, by direct transform."""
    tops, lam, mu = wn.layer_stack(ONE_LAYER)
    k, dln = wn.wavenumbers()
    U, V, W, dU, dV, dW = wn.kernels(tops, lam, mu, z_s, [z_r], k, remainder=False,
                                     derivatives=True)
    return (wn.transform(U, V, W, k, dln, r)[0],
            wn.strain_columns(U, V, W, dU, dV, dW, k, dln, r)[0])


@pytest.mark.parametrize("z_r", [200.0, 3000.0, 8000.0])
@pytest.mark.parametrize("name", list(POINT_SOURCES))
def test_a_uniform_crust_is_the_analytic_point_source_strain(name, z_r):
    """The cylindrical-to-Cartesian assembly against a tiny triangle's analytic strain.

    Source at 5 km, receivers above it, below it and near the surface, 300 m - 30 km,
    six azimuths. Measured (2026-11-04) at most 2.7e-5 of the peak, the displacement
    path's own level; insensitive to the triangle's size (1 or 5 m), so the point-source
    approximation is not what is measured. Finite differences of the 1 m triangle are no
    reference here (5e-3: it is 30,000 sizes away).
    """
    z_s = 5000.0
    r = np.geomspace(300.0, 30e3, 13)
    phi = np.radians([0.0, 23.0, 77.0, 131.0, 200.0, 290.0])
    values, columns = _whole_columns(z_s, z_r, r)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    at = np.repeat(np.arange(r.size), phi.size)
    n, d = _unit_pair(name)
    ours = combine_strain(values[at], columns[at], pp.ravel(), moment_packing(n, d)[None])[:, 0]
    x, y = (rr * np.sin(pp)).ravel(), (rr * np.cos(pp)).ravel()
    want = _cutde_point_strain(n, d, z_s, x, y, np.full(x.size, -z_r))
    assert np.abs(ours - want).max() < 1e-4 * np.abs(want).max()


def test_on_the_axis_the_strain_has_its_limit():
    """Directly above and below the source, and centimetres to metres off it.

    Here an explicit ``1/r`` multiplied FFTLog's absolute error floor: a strike-slip
    source read 1000x its strain on the axis, O(1) at 10 cm, 4.5e-3 at 1 m (2026-11-04).
    The ``1/r`` columns (Bessel recurrence) have no division: measured 5e-6 to 3.3e-5 of
    the peak at ``r = 0`` (which takes the transform's first node, a few cm out) and about
    1e-7 from 1 cm to 10 m.
    The bar is relative to the field's peak over a sphere at the same distance, since
    on the axis a double couple's own strain can nearly vanish.
    """
    z_s = 5000.0
    r = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    phi = np.radians([0.0, 140.0, 35.0, 250.0, 300.0])
    for z_r in (2000.0, 7500.0):
        values, columns = _whole_columns(z_s, z_r, r)
        ring = np.radians(np.arange(0.0, 360.0, 30.0))
        ring_r = abs(z_r - z_s) * np.ones(ring.size)
        ring_values, ring_columns = _whole_columns(z_s, z_r, ring_r[:1])
        for name in POINT_SOURCES:
            n, d = _unit_pair(name)
            pack = moment_packing(n, d)[None]
            ours = combine_strain(values, columns, phi, pack)[:, 0]
            want = _cutde_point_strain(n, d, z_s, r * np.sin(phi), r * np.cos(phi), np.full(r.size, -z_r))
            ring_peak = np.abs(combine_strain(np.repeat(ring_values, ring.size, 0),
                                              np.repeat(ring_columns, ring.size, 0), ring, pack)).max()
            assert np.all(np.isfinite(ours))
            assert np.abs(ours - want).max() < 1e-4 * max(np.abs(want).max(), ring_peak), name


def test_a_moment_tensor_packs_like_its_dislocation():
    """``tensor_packing(mu (n d + d n) + lam (n . d) I)`` is ``moment_packing(n, d)``, opening included."""
    lam, mu = 3.2e10, 2.9e10
    for name in POINT_SOURCES:
        n, d = _unit_pair(name)
        moment = mu * (np.outer(n, d) + np.outer(d, n)) + lam * (n @ d) * np.eye(3)
        assert np.allclose(tensor_packing(moment, lam, mu), moment_packing(n, d), rtol=0, atol=1e-12)
    stack = np.stack([np.eye(3), np.diag([1.0, -1.0, 0.0])])
    assert tensor_packing(stack, lam, mu).shape == (2, 6)


# -- the stored strain tables (RemainderTables(strain=True)) ---------------------------------
@pytest.fixture(scope="module")
def strain_tables():
    """The two-layer crust on ``test_slip_layered_tde.py``'s grids with strain, but 1 km apart at depth.

    The strain needs the finer deep spacing: at that file's 3 km the remainder's strain
    was 8e-2 of itself off at the median (2026-11-04), second order in the spacing.
    """
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=40e3, first=10.0, growth=1.6, spacing=400.0, deep_spacing=1000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=20), n_k=2048,
                                      z_max=40e3, surface_z=wn.surface_depth_grid(tops, fine_to=20e3,
                                                                                  z_max=40e3),
                                      strain=True)


def test_the_strain_tables_hold_the_transformed_derivatives(strain_tables):
    """At the grid's own nodes, the tables are the direct transforms; the displacement is unchanged."""
    t = strain_tables
    tops, lam, mu = t.tops, t.lam, t.mu
    k, dln = wn.wavenumbers(n=2048)
    j = int(np.searchsorted(t.z, 6000.0))
    U, V, W, dU, dV, dW = wn.kernels(tops, lam, mu, t.z[j], t.z, k, derivatives=True)
    columns = wn.strain_columns(U, V, W, dU, dV, dW, k, dln, t.r)
    assert np.allclose(t.strain[:, j], np.moveaxis(columns, 1, 0), rtol=1e-12, atol=0)
    plain = wn.RemainderTables.compute(TWO_LAYERS, z=t.z, r=t.r, n_k=2048, z_max=40e3,
                                       surface_z=t.surface["z"], surface_r=t.surface["r"])
    assert np.array_equal(plain.values, t.values) and not plain.has_strain
    assert len(wn.STRAIN_KEYS) == 27 and wn.STRAIN_KEYS[0] == ("dr", "ss", "uz")
    assert wn.STRAIN_KEYS[20] == ("/r", "ss", "ur-2ut")
    with pytest.raises(ValueError, match="no strain"):
        plain.gather_strain(plain.distance_stencil(np.array([1e3])),
                            plain.depth_stencil(np.array([5e3])), plain.depth_stencil(np.array([4e3])))


def test_the_strain_lookup_is_the_displacement_lookup_on_the_derivatives(strain_tables):
    t = strain_tables
    distance = np.array([350.0, 4200.0, 17e3])
    source = t.depth_stencil(np.array([5100.0, 900.0, 12e3]))
    receiver = t.depth_stencil(np.array([7300.0, 1700.0, 2600.0]))
    got = t.gather_strain(t.distance_stencil(distance), source, receiver)
    want = t.gather(t.distance_stencil(distance), source, receiver, table=t.strain)
    assert got.shape == (3, 27) and np.array_equal(got, want)


def test_strain_tables_survive_a_round_trip(strain_tables, tmp_path):
    back = wn.RemainderTables.load(strain_tables.save(tmp_path / "tables.npz"))
    assert back.has_strain and np.array_equal(back.strain, strain_tables.strain)
    assert "strain" in repr(back)


# -- the strain of fault elements and tents at depth (LayeredTDE.strain): the operator E ----
from slipsolve._tde import _element_basis  # noqa: E402
from slipsolve.layered import _triangle_quadrature  # noqa: E402
from slipsolve.layered_tde import LayeredTDE, split_by_layers  # noqa: E402

#: ``test_slip_layered_tde.py``'s triangles: one crossing the 2 km interface, one below it.
CROSSING = np.array([[0.0, 0.0, -500.0], [3000.0, 300.0, -800.0], [1200.0, 1500.0, -4200.0]])
BELOW = np.array([[0.0, 0.0, -4000.0], [1000.0, 100.0, -4200.0], [400.0, 600.0, -5100.0]])


def _around(vertices, n, near, far, seed):
    """Points at depth from ``near`` to ``far`` triangle sizes of a triangle's centroid."""
    rng = np.random.default_rng(seed)
    size = _size(vertices)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    pts = vertices.mean(axis=0) + dirs * size * np.geomspace(near, far, n)[:, None]
    pts[:, 2] = -np.abs(pts[:, 2]) - 100.0
    return pts


def _point_source_strain(vertices, x, y, z, remainder, order):
    """Strain of unit strike-, dip- and tensile-slip by point sources of exact kernels: ``(npts, 3, 6)``.

    Each piece of the triangle (split at the interface, as the engine splits it) on its own
    rule; every (point, receiver) pair transformed at its own distance, no tables.
    """
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers(n=2048)
    vnorm, vstrike, vdip = _element_basis(*vertices)
    packing = np.array([moment_packing(vnorm, d) for d in (vstrike, vdip, vnorm)])
    out = np.zeros((x.size, 3, 6))
    for tri, _ in split_by_layers(vertices, tops[1:]):
        points, share = _triangle_quadrature(*tri, order)
        area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        for p, w in zip(points, share * area):
            for i in range(x.size):
                r = np.array([np.hypot(x[i] - p[0], y[i] - p[1])])
                kern = wn.kernels(tops, lam, mu, -p[2], [-z[i]], k, remainder=remainder,
                                  derivatives=True)
                values = wn.transform(*kern[:3], k, dln, r)[0]
                columns = wn.strain_columns(*kern, k, dln, r)[0]
                azimuth = np.array([np.arctan2(x[i] - p[0], y[i] - p[1])])
                out[i] += w * combine_strain(values, columns, azimuth, packing)[0]
    return out


def test_a_one_layer_crust_is_the_half_space_strain():
    """The remainder of a uniform crust is zero, strain included: the TDE strain alone (near path everywhere)."""
    tops, _, _ = wn.layer_stack(ONE_LAYER)
    z = wn.depth_grid(tops, z_max=40e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    tables = wn.RemainderTables.compute(ONE_LAYER, z=z, r=wn.distance_grid(per_decade=5), n_k=256,
                                        surface_z=z, surface_r=wn.distance_grid(per_decade=5),
                                        strain=True)
    assert np.all(tables.strain == 0.0)
    pts = _around(BELOW, 10, 0.5, 20.0, seed=1)
    got = LayeredTDE(tables, far=None).element_strain(BELOW, *pts.T)
    want = np.moveaxis(tde_strain_hs_unit(*pts.T, *BELOW, float(ONE_LAYER.poisson())), -1, 0)
    assert np.allclose(got, want, rtol=1e-12, atol=0)


@pytest.mark.parametrize("vertices", [BELOW, CROSSING], ids=["below", "crossing"])
def test_the_three_tents_add_up_to_the_element_strain(strain_tables, vertices):
    """``sum_j tent_j = 1`` on the triangle, so the tents' strains add up to the element's.

    This is the level-set rule's accuracy (:data:`~slipsolve.layered_tde.STRAIN_TENT_GAIN`):
    measured at most 9e-5 of each point's peak from 0.6 to 30 sizes (2026-11-04), where
    displacement's rule (two points far away) left 5e-3 at 3 sizes.
    """
    engine = LayeredTDE(strain_tables)
    pts = _around(vertices, 40, 0.6, 30.0, seed=2)
    tents = engine.tent_strain(vertices, *pts.T, components=(0, 1, 2))
    whole = engine.element_strain(vertices, *pts.T)
    error = np.abs(tents.sum(axis=1) - whole).max(axis=(1, 2)) / np.abs(whole).max(axis=(1, 2))
    assert error.max() < 3e-4


@pytest.mark.parametrize("vertices", [BELOW, CROSSING], ids=["below", "crossing"])
def test_far_away_the_strain_is_the_point_source_sum(strain_tables, vertices):
    """Half-space TDE strain plus the tabulated remainder against the whole layered kernel's point sources.

    ⚠️ Receivers are kept 1.5 km or more in depth from every point of the triangle: the
    whole kernel at nearly the source's depth barely decays in wavenumber, and the
    strain's extra power of ``k`` spoils FFTLog's cancellation at 25 km (a receiver 1 km
    deep beside the crossing triangle read 100x its neighbours, 2026-11-04). The engine
    never meets this: its remainder decays by the images. The near path's construction, so
    the far path is off here (it has its own test).

    The bar is the coarse test tables': against references converged in their order (4, 8
    and 12 agree to 7e-4), both the centroid-rule remainder (``ef53e31``) and the
    quadratic rule measure 1e-3 to 5.0e-3 on the crossing triangle (2026-11-04); the old
    5e-3 bar sat on that floor, and the order-4 reference reads 5.1e-3 for the new rule.
    """
    engine = LayeredTDE(strain_tables, far=None)
    top, bottom = -vertices[:, 2].min(), -vertices[:, 2].max()
    depths = [d for d in (500.0, 6500.0, 9000.0) if d < top - 1500.0 or d > bottom + 1500.0]
    x = np.array([15e3, -12e3, 3e3])[:len(depths)]
    y = np.array([4e3, 9e3, -18e3])[:len(depths)]
    z = -np.array(depths)
    got = engine.element_strain(vertices, x, y, z)
    want = _point_source_strain(vertices, x, y, z, remainder=False, order=4)
    for c in range(3):
        assert np.abs(got[:, c] - want[:, c]).max() < 7e-3 * np.abs(want[:, c]).max()


@pytest.mark.parametrize("vertices", [BELOW, CROSSING], ids=["below", "crossing"])
def test_far_away_the_point_source_path_is_the_dislocation_path(strain_tables, vertices):
    """Beyond ``far`` sizes: Okada point sources plus the remainder, against the level-set path.

    Both paths converge to the same field: with tolerance 1e-7 they agree to 3e-5 / 1e-4
    (below / crossing the interface) of each receiver's peak at 4-40 sizes (2026-11-04).
    At the defaults the far path is the closer of the two to that limit (tents 3e-5 /
    1.2e-4 against the near path's 3e-4 / 5e-4), so this bar is the near path's.
    """
    pts = _around(vertices, 12, 4.5, 30.0, seed=8)
    far, near = LayeredTDE(strain_tables), LayeredTDE(strain_tables, far=None)
    assert far._far_strain(vertices, *pts.T).all()
    for f in (lambda e: e.tent_strain(vertices, *pts.T, components=(0, 1, 2)),
              lambda e: e.element_strain(vertices, *pts.T)):
        a, b = f(far), f(near)
        axes = tuple(range(1, a.ndim))
        assert (np.abs(a - b).max(axis=axes) / np.abs(b).max(axis=axes)).max() < 1.5e-3


def test_the_strain_needs_strain_tables():
    tops, _, _ = wn.layer_stack(ONE_LAYER)
    z = wn.depth_grid(tops, z_max=40e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    tables = wn.RemainderTables.compute(ONE_LAYER, z=z, r=wn.distance_grid(per_decade=5), n_k=256,
                                        surface_z=z, surface_r=wn.distance_grid(per_decade=5))
    with pytest.raises(ValueError, match="strain=True"):
        LayeredTDE(tables).element_strain(BELOW, [5e3], [5e3], [-3e3])


def test_the_strain_matrix_has_element_and_node_columns(strain_tables):
    """``LayeredTDE.strain``: strike columns then dip columns, an element's own or a node's tents."""
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve.basis import make_basis

    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.92]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=6e3, edge_length=3e3)
    engine = LayeredTDE(strain_tables)
    xs, ys = trace.from_curvilinear(np.array([2e3, 4.5e3, 7e3]), np.array([-5e3, 6e3, 9e3]), frame)
    zs = np.array([-3000.0, -1200.0, -7500.0])
    n_el = mesh.n_elements
    element = engine.strain(mesh, xs, ys, zs)
    assert element.shape == (3, 6, 2 * n_el)
    k = n_el // 2
    own = engine.element_strain(mesh.vertices[k], xs, ys, zs, components=(0, 1))
    assert np.array_equal(element[:, :, k], own[:, 0]) and np.array_equal(element[:, :, n_el + k], own[:, 1])
    basis = make_basis(mesh, "node")
    nodal = engine.strain(mesh, xs, ys, zs, basis=basis)
    n = basis.n_basis
    assert nodal.shape == (3, 6, 2 * n)
    # The tents sum to one on every element, so all nodes' columns sum to all elements'.
    for half_n, half_e in ((slice(0, n), slice(0, n_el)), (slice(n, 2 * n), slice(n_el, 2 * n_el))):
        total_n, total_e = nodal[:, :, half_n].sum(axis=-1), element[:, :, half_e].sum(axis=-1)
        assert np.abs(total_n - total_e).max() < 3e-4 * np.abs(total_e).max()


@pytest.mark.parametrize("vertices", [BELOW, CROSSING], ids=["below", "crossing"])
def test_the_tabulated_remainder_strain_is_the_remainders(strain_tables, vertices):
    """The remainder alone -- the element's strain less its pieces' half-space strain -- against exact kernels.

    This is all the tables touch, and their grids bound it, at second order in the
    spacing. Measured at 1-12 sizes (2026-11-04), median / max of the remainder's own
    peak: 31 depths and 20 distances a decade 8e-2 / 0.27; 66 and 30, 1.6e-2 / 0.1; 133
    and 40, 2.6e-3 / 1.7e-2. The quadrature tolerance (3e-3 or 3e-4) changed nothing.
    On this fixture 0.05-0.10.
    """
    engine = LayeredTDE(strain_tables, far=None)
    pts = _around(vertices, 4, 1.5, 6.0, seed=3)
    pieces = split_by_layers(vertices, strain_tables.tops[1:])
    nu = strain_tables.poisson(np.array([mid for _, mid in pieces]))
    halfspace = sum(np.moveaxis(tde_strain_hs_unit(*pts.T, *tri, n), -1, 0)
                    for (tri, _), n in zip(pieces, nu))
    got = engine.element_strain(vertices, *pts.T) - halfspace
    want = _point_source_strain(vertices, *pts.T, remainder=True, order=4)
    error = np.abs(got - want).max(axis=(1, 2)) / np.abs(want).max(axis=(1, 2))
    # This fixture's grids give 0.05-0.10 (below the interface); an error in the strain
    # path (a sign, a pattern, a frame) is O(1).
    assert error.max() < 0.2


def test_many_triangles_strain_matches_cutde():
    """The batched strain (one difference stencil a point for all triangles) against ``cutde``.

    Measured 3e-8 to 7e-8 of each point's peak (2026-11-04), as good as the scalar kernel.
    """
    hs = pytest.importorskip("cutde.halfspace")
    from slipsolve._tde import tde_strain_hs_unit_many

    tris = np.array([TRIANGLES["dipping"], TRIANGLES["vertical"], TRIANGLES["shallow_dip"]])
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-20e3, 20e3, 150), rng.uniform(-20e3, 20e3, 150),
                           -np.abs(rng.uniform(0.0, 15e3, 150))])
    pts[:25, 2] = 0.0
    many = tde_strain_hs_unit_many(*pts.T, tris, NU)
    for i, tri in enumerate(tris):
        theirs = np.transpose(hs.strain_matrix(pts, tri[None], NU)[:, :, 0, :], (2, 1, 0))
        scale = np.abs(theirs).max(axis=(0, 1))
        assert (np.abs(many[i] - theirs).max(axis=(0, 1)) / scale).max() < 1e-6
