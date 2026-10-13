"""Static layered Green's functions by wavenumber integration (``slipsolve.wavenumber``)."""

import numpy as np
import pytest

from slipsolve import wavenumber as wn
from slipsolve._tde import tde_disp_hs
from slipsolve.edgrn import VelocityModel, _point_dislocation, _project, _SOURCE_GEOMETRY

#: A two-layer crust with a strong shallow contrast and different Poisson ratios,
#: like the top of the Venezuela model.
TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])
ONE_LAYER = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)


def _tde_tables(nu, r, z_s, z_r, n_az=16, size=1.0):
    """Point-source tables of a homogeneous half-space from tiny triangular dislocations."""
    phi = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    ox, oy = (rr * np.sin(pp)).ravel(), (rr * np.cos(pp)).ravel()
    oz = np.full(ox.size, -float(z_r))
    f = {}
    for label, normal, slip in _SOURCE_GEOMETRY:
        p1, p2, p3, comp = _point_dislocation(normal, slip, z_s, size)
        f[label] = np.stack(tde_disp_hs(ox, oy, oz, p1, p2, p3, *comp, nu)) / (0.5 * size ** 2)
    sources = {"ss": f["ss"], "ds": f["ds"], "cl": 0.5 * (f["tz"] - 0.5 * (f["tx"] + f["ty"])),
               "ep": (f["tx"] + f["ty"] + f["tz"]) / 3.0}
    out = np.empty((r.size, len(wn.TABLE_KEYS)))
    for col, (kind, comp) in enumerate(wn.TABLE_KEYS):
        ue, un, uu = (c.reshape(r.size, phi.size) for c in sources[kind])
        north, east, down = un, ue, -uu
        values = {"uz": down, "ur": north * np.cos(phi) + east * np.sin(phi),
                  "ut": -north * np.sin(phi) + east * np.cos(phi)}[comp]
        out[:, col] = _project(values, phi, "cl" if kind == "ep" else kind, comp)
    return out


@pytest.mark.parametrize("z_r", [0.0, 3000.0, 8000.0])
def test_a_homogeneous_half_space_is_the_analytic_point_source(z_r):
    """All four sources, all three components, at the surface and at depth."""
    tops, lam, mu = wn.layer_stack(ONE_LAYER)
    k, dln = wn.wavenumbers()
    r = np.geomspace(200.0, 100e3, 25)
    U, V, W = wn.kernels(tops, lam, mu, 5000.0, np.array([z_r]), k, remainder=False)
    ours = wn.transform(U, V, W, k, dln, r)[0]
    ref = _tde_tables(float(ONE_LAYER.poisson()), r, 5000.0, z_r)
    for col, key in enumerate(wn.TABLE_KEYS):
        peak = np.abs(ref[:, col]).max()
        if peak < 1e-6 * np.abs(ref).max():           # ds.ut vanishes at the free surface
            assert np.abs(ours[:, col]).max() < 1e-6 * np.abs(ref).max(), key
            continue
        assert np.abs(ours[:, col] - ref[:, col]).max() < 1e-4 * peak, key


def test_the_remainder_of_a_one_layer_crust_is_zero():
    tops, lam, mu = wn.layer_stack(ONE_LAYER)
    k, _ = wn.wavenumbers(n=512)
    for a in wn.kernels(tops, lam, mu, 5000.0, np.array([0.0, 5000.0, 9000.0]), k):
        assert np.all(a == 0.0)


def _force(tops, lam, mu, z_src, z_rec, k, dln, r):
    """Point-force fields: ``G_zz``, ``G_xx``, ``G_yy``, and ``u_r`` / ``u_z`` for ``G_xz`` / ``G_zx``."""
    psv = np.zeros((2, 4))
    sh = np.zeros((2, 2))
    psv[0, 2] = -1.0 / (2 * np.pi * wn.MU_SCALE)       # F_z: a jump in R, over k
    psv[1, 3] = -1.0 / (2 * np.pi * wn.MU_SCALE)       # F_x: jumps in S and T, over k
    sh[1, 1] = -1.0 / (2 * np.pi * wn.MU_SCALE)
    U, V, W = (a[0] / k[:, None] for a in
               wn._Solution(k, tops, lam, mu, z_src, psv, sh).at(np.array([z_rec])))

    def h(f, order):
        grid, values = wn.hankel(f, k, dln, order)
        return np.interp(np.log(r), np.log(grid), values)

    plus, minus = h(V[:, 1] + W[:, 1], 0), h(V[:, 1] - W[:, 1], 2)
    return {"zz": h(U[:, 0], 0), "xz": -h(V[:, 0], 1), "zx": -h(U[:, 1], 1),
            "xx": 0.5 * plus - 0.5 * minus, "yy": 0.5 * plus + 0.5 * minus}


@pytest.mark.parametrize("z1, z2", [(500.0, 5000.0), (2500.0, 1900.0), (1500.0, 40e3)])
def test_point_forces_are_reciprocal_across_layers(z1, z2):
    """``G_ij(x, xi) = G_ji(xi, x)``: a wrong layer matrix or interface condition breaks it."""
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers()
    r = np.array([500.0, 2e3, 5e3, 20e3, 80e3])
    a = _force(tops, lam, mu, z1, z2, k, dln, r)
    b = _force(tops, lam, mu, z2, z1, k, dln, r)
    for one, other in (("zz", "zz"), ("xx", "xx"), ("yy", "yy"), ("xz", "zx")):
        assert np.abs(a[one] - b[other]).max() < 1e-10 * np.abs(a[one]).max()


def test_the_remainder_is_smooth_at_the_source():
    """Mid-layer, source and receiver at one depth: flat in ``r`` at the source."""
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers()
    r = np.array([0.5, 5.0, 50.0])
    U, V, W = wn.kernels(tops, lam, mu, 6000.0, np.array([6000.0]), k)
    values = wn.transform(U, V, W, k, dln, r)[0]
    cl_uz = values[:, wn.TABLE_KEYS.index(("cl", "uz"))]
    ds_ur = values[:, wn.TABLE_KEYS.index(("ds", "ur"))]
    assert np.all(np.isfinite(values))
    assert np.abs(cl_uz - cl_uz[0]).max() < 1e-3 * np.abs(cl_uz[0])
    assert np.abs(ds_ur - ds_ur[0]).max() < 1e-3 * np.abs(ds_ur[0])


@pytest.fixture(scope="module")
def small_tables():
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=30e3, first=20.0, growth=2.0, spacing=1500.0,
                      deep_spacing=8000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=10),
                                      n_k=2048, surface_z=z, surface_r=wn.distance_grid(per_decade=10))


def test_lookups_never_interpolate_across_an_interface(small_tables):
    t = small_tables
    above, below = t.z[t.z < 2000.0].max(), t.z[t.z > 2000.0].min()
    # On the nodes the lookup is the table itself.
    j = int(np.flatnonzero(t.z == above)[0])
    i = 30
    got = t.interpolate(t.r[i], above, above)
    assert np.allclose(got, t.values[i, j, j], rtol=0, atol=1e-14 * np.abs(t.values).max())
    # Between the last node above and the interface the lookup extrapolates
    # from nodes above it, not toward the node below.
    lo, hi, w = t._depth_stencil(np.array([0.5 * (above + 2000.0)]))
    assert t.z[lo[0]] < 2000.0 and t.z[hi[0]] < 2000.0 and w[0] > 1.0
    lo, hi, w = t._depth_stencil(np.array([0.5 * (below + 2000.0)]))
    assert t.z[lo[0]] > 2000.0 and t.z[hi[0]] > 2000.0 and w[0] < 0.0


def test_workers_change_nothing(small_tables):
    z, r = small_tables.z[::4], small_tables.r[::5]
    one = wn.RemainderTables.compute(TWO_LAYERS, z=z, r=r, n_k=1024, surface_z=z, surface_r=r)
    two = wn.RemainderTables.compute(TWO_LAYERS, z=z, r=r, n_k=1024, surface_z=z, surface_r=r,
                                     workers=2)
    assert np.array_equal(one.values, two.values)
    assert np.array_equal(one.surface["whole"], two.surface["whole"])


def test_tables_survive_a_round_trip(small_tables, tmp_path):
    path = small_tables.save(tmp_path / "remainder.npz")
    back = wn.RemainderTables.load(path)
    assert np.array_equal(back.values, small_tables.values)
    assert np.array_equal(back.surface["remainder"], small_tables.surface["remainder"])
    assert np.array_equal(back.tops, small_tables.tops)
    assert back.attrs["n_k"] == 2048


def test_a_gradient_is_refused():
    gradient = VelocityModel([0.0, 5000.0], [5000.0, 6000.0], [2900.0, 3460.0], [2600.0, 2700.0])
    with pytest.raises(NotImplementedError, match="gradient"):
        wn.layer_stack(gradient)
