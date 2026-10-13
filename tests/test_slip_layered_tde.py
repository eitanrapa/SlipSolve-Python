"""Triangular dislocations in a layered half-space (``slipsolve.layered_tde``)."""

import numpy as np
import pytest

from slipsolve import wavenumber as wn
from slipsolve._tde import tde_disp_hs
from slipsolve.edgrn import VelocityModel
from slipsolve.layered import _triangle_quadrature
from slipsolve.layered_tde import LayeredTDE, moment_packing, split_by_layers

TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])
ONE_LAYER = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)

#: A dipping triangle that crosses the 2 km interface, and one wholly below it.
CROSSING = np.array([[0.0, 0.0, -500.0], [3000.0, 300.0, -800.0], [1200.0, 1500.0, -4200.0]])
BELOW = np.array([[0.0, 0.0, -4000.0], [1000.0, 100.0, -4200.0], [400.0, 600.0, -5100.0]])


@pytest.fixture(scope="module")
def two_layer_tables():
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=40e3, first=10.0, growth=1.6, spacing=400.0,
                      deep_spacing=3000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=20),
                                      n_k=2048, z_max=40e3,
                                      surface_z=wn.surface_depth_grid(tops, fine_to=20e3, z_max=40e3))


@pytest.fixture(scope="module")
def one_layer_tables():
    tops, _, _ = wn.layer_stack(ONE_LAYER)
    z = wn.depth_grid(tops, z_max=40e3, first=50.0, growth=2.0, spacing=2000.0,
                      deep_spacing=8000.0)
    return wn.RemainderTables.compute(ONE_LAYER, z=z, r=wn.distance_grid(per_decade=5),
                                      n_k=256, surface_z=z, surface_r=wn.distance_grid(per_decade=5))


def test_split_by_layers_keeps_area_normal_and_vertical_edges():
    pieces = split_by_layers(CROSSING, [2000.0])
    area = 0.5 * np.linalg.norm(np.cross(CROSSING[1] - CROSSING[0], CROSSING[2] - CROSSING[0]))
    normal = np.cross(CROSSING[1] - CROSSING[0], CROSSING[2] - CROSSING[0])
    total = 0.0
    for tri, mid in pieces:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        total += 0.5 * np.linalg.norm(n)
        assert n @ normal > 0                      # the parent's winding
        assert np.isclose(np.linalg.norm(np.cross(n, normal)), 0.0, atol=1e-6 * area)
        depth = -tri[:, 2]
        assert depth.max() <= 2000.0 + 1e-9 or depth.min() >= 2000.0 - 1e-9
    assert np.isclose(total, area, rtol=1e-12)
    # A vertical triangle's cut points keep its x and y exactly.
    vertical = np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0], [1000.0, 0.0, -5000.0]])
    for tri, _ in split_by_layers(vertical, [2000.0]):
        assert np.all(tri[:, 1] == 0.0)
    assert len(split_by_layers(BELOW, [2000.0])) == 1


def test_an_opening_packs_as_an_explosion_plus_a_clvd():
    n = np.array([0.0, 0.0, 1.0])                  # a horizontal crack
    m1, m2, m3, m4, m5, e = moment_packing(n, n)
    assert e == 1.0 and m1 == m2 == m4 == m5 == 0.0
    assert np.isclose(m3, 2.0 - 2.0 / 3.0)
    shear = moment_packing(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert shear[-1] == 0.0 and np.isclose(shear[0], 1.0)


def test_a_one_layer_crust_is_the_half_space(one_layer_tables):
    engine = LayeredTDE(one_layer_tables, far=None)
    nu = float(ONE_LAYER.poisson())
    x = np.array([500.0, 3000.0, -2000.0, 1000.0])
    y = np.array([200.0, -1500.0, 4000.0, 700.0])
    z = np.array([0.0, 0.0, -3000.0, -4500.0])
    got = engine.element_field(BELOW, x, y, z)
    for c in range(3):
        slip = [0.0, 0.0, 0.0]
        slip[c] = 1.0
        want = np.column_stack(tde_disp_hs(x, y, z, *BELOW, *slip, nu))
        assert np.allclose(got[:, c], want, rtol=1e-12, atol=1e-15)


def _point_source_sum(vertices, slip_index, x, y, z, order=4):
    """The layered field by brute force: point sources of the full layered kernel.

    Each piece of the triangle is integrated on its own: the kernel per unit
    potency steps by the rigidity ratio at an interface, and one rule across the
    step would be first order.
    """
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers(n=2048)
    from slipsolve._tde import _element_basis
    vnorm, vstrike, vdip = _element_basis(*vertices)
    packing = moment_packing(vnorm, (vstrike, vdip, vnorm)[slip_index])
    points, weights = [], []
    for tri, _ in split_by_layers(vertices, tops[1:]):
        p, share = _triangle_quadrature(*tri, order)
        points.append(p)
        weights.append(share * 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
    out = np.zeros((x.size, 3))
    for p, w in zip(np.concatenate(points), np.concatenate(weights)):
        dn, de = y - p[1], x - p[0]
        r = np.hypot(dn, de)
        phi = np.arctan2(de, dn)
        for i in range(x.size):
            U, V, W = wn.kernels(tops, lam, mu, -p[2], np.array([-z[i]]), k, remainder=False)
            values = wn.transform(U, V, W, k, dln, np.array([r[i]]))[0, 0]
            m1, m2, m3, m4, m5, e = packing
            c1, s1, c2, s2 = np.cos(phi[i]), np.sin(phi[i]), np.cos(2 * phi[i]), np.sin(2 * phi[i])
            ss_uz, ss_ur, ss_ut, ds_uz, ds_ur, ds_ut, cl_uz, cl_ur, ep_uz, ep_ur = values
            uz = ((m1 * s2 + m4 * c2) * ss_uz + (m2 * c1 + m5 * s1) * ds_uz + m3 * cl_uz
                  + e * ep_uz)
            ur = ((m1 * s2 + m4 * c2) * ss_ur + (m2 * c1 + m5 * s1) * ds_ur + m3 * cl_ur
                  + e * ep_ur)
            ut = (m1 * c2 - m4 * s2) * ss_ut + (m2 * s1 - m5 * c1) * ds_ut
            out[i] += w * np.array([ur * s1 + ut * c1, ur * c1 - ut * s1, -uz])
    return out


@pytest.mark.parametrize("vertices", [BELOW, CROSSING], ids=["below", "crossing"])
def test_far_away_it_is_the_point_source_sum(two_layer_tables, vertices):
    """Half-space TDE plus the remainder equals the whole layered kernel's point sources.

    The bar is the coarse test tables'. Measured on the crossing triangle against
    an order-8 reference (itself 1.3e-4 from order 4): 1.6e-3 of the peak on these
    tables (36 depths, 20 distances a decade) and 5e-4 on 60 depths and 40 a decade,
    whatever the quadrature tolerance -- it is the tables' resolution.
    """
    engine = LayeredTDE(two_layer_tables)
    x = np.array([15e3, -12e3, 3e3, 20e3])
    y = np.array([4e3, 9e3, -18e3, 15e3])
    z = np.array([0.0, 0.0, -6000.0, -1000.0])
    got = engine.element_field(vertices, x, y, z)
    for c in range(3):
        want = _point_source_sum(vertices, c, x, y, z)
        assert np.abs(got[:, c] - want).max() < 5e-3 * np.abs(want).max()


def test_the_field_is_continuous_across_an_interface(two_layer_tables):
    """Receivers a metre either side of the 2 km interface, away from the element."""
    engine = LayeredTDE(two_layer_tables)
    x = np.array([6e3, -4e3, 9e3])
    y = np.array([3e3, 5e3, -6e3])
    above = engine.element_field(CROSSING, x, y, np.full(3, -1999.0))
    below = engine.element_field(CROSSING, x, y, np.full(3, -2001.0))
    assert np.abs(above - below).max() < 3e-3 * np.abs(above).max()


# -- nodal (tent) columns -------------------------------------------------------------------
def test_the_three_tents_add_up_to_the_element(two_layer_tables):
    """sum_j tent_j = 1 on the triangle, so the tents' fields add up to the element's."""
    engine = LayeredTDE(two_layer_tables)
    x = np.array([600.0, 2500.0, -3000.0, 15e3, 4e3])
    y = np.array([-400.0, 2500.0, 1200.0, -9e3, 3e3])
    z = np.array([0.0, 0.0, -1500.0, 0.0, -7000.0])
    tents = engine.tent_field(CROSSING, x, y, z, components=(0, 1, 2))
    whole = engine.element_field(CROSSING, x, y, z)
    np.testing.assert_allclose(tents.sum(axis=1), whole, rtol=0, atol=3e-3 * np.abs(whole).max())


def test_a_tent_is_its_finely_subdivided_triangle(one_layer_tables):
    """Against oriented constant-slip sub-triangles carrying the tent's value at their centroids."""
    from slipsolve._tde import tde_disp_hs_unit
    from slipsolve.layered import _sub_triangles

    engine = LayeredTDE(one_layer_tables, far=None)
    nu = float(ONE_LAYER.poisson())
    tri = np.array([[0.0, 0.0, 0.0], [3000.0, 0.0, 0.0], [1500.0, 0.0, -2600.0]])
    x = np.full(4, 200.0)
    y = np.array([300.0, 1000.0, 3000.0, 20e3])
    z = np.zeros(4)
    want = np.zeros((4, 3, 2, 3))
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    for sub in _sub_triangles(48):
        verts = sub @ tri
        if np.cross(verts[1] - verts[0], verts[2] - verts[0]) @ normal < 0:
            verts = verts[[0, 2, 1]]
        u = np.moveaxis(tde_disp_hs_unit(x, y, z, *verts, nu)[[0, 1]], -1, 0)
        for j, value in enumerate(sub.mean(axis=0)):
            want[:, j] += value * u
    got = engine.tent_field(tri, x, y, z)
    for i in range(4):
        assert np.abs(got[i] - want[i]).max() < 3e-3 * np.abs(want[i]).max()


def test_nodal_columns_match_point_sources_away_from_the_fault(two_layer_tables):
    """Two independent tent integrations of one crust, a few km and more from the fault.

    ``LayeredPointSource`` on tables of the full layered kernel from the same solver
    (remainder plus half-space, at the surface), integrating each tent with point sources.
    Measured: 2-5e-3 of each receiver's largest column, the same at point-source
    tolerance 1e-3 and 1e-4 -- the point sources' own 100 m tables. Projecting
    element means instead differs from these tents by 9-14% on this coarse mesh.
    """
    from slipsolve import FaultMesh, FaultTrace
    from slipsolve.basis import make_basis
    from slipsolve.edgrn import EdgrnTables
    from slipsolve.layered import LayeredPointSource

    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.8]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=8e3, edge_length=4e3)
    basis = make_basis(mesh, "node")
    tops, lam, mu = wn.layer_stack(TWO_LAYERS)
    k, dln = wn.wavenumbers(n=2048)
    r = np.linspace(0.0, 120e3, 1201)
    z = np.linspace(50.0, 12050.0, 121)                  # nodes off the 2 km interface
    tables = {kind: {c: np.zeros((r.size, z.size)) for c in ("uz", "ur", "ut")} for kind in ("ss", "ds", "cl")}
    for j, z_s in enumerate(z):
        U, V, W = wn.kernels(tops, lam, mu, z_s, np.array([0.0]), k, remainder=False)
        values = wn.transform(U, V, W, k, dln, np.maximum(r, 1.0))[0]
        for col, (kind, comp) in enumerate(wn.TABLE_KEYS[:8]):
            tables[kind][comp][:, j] = values[:, col]
    point = LayeredPointSource(EdgrnTables(r, z, tables, interfaces=tops[1:]), tolerance=1e-3)
    s0, s1 = trace.to_curvilinear(*trace.from_curvilinear(np.array([5e3, 15e3]), np.zeros(2), frame), frame)[0]
    xs, ys = trace.from_curvilinear(np.array([5e3, 10e3, 15e3, 8e3, 12e3]),
                                    np.array([6e3, -7e3, 12e3, -15e3, 25e3]), frame)
    look = (np.full(5, 0.6), np.full(5, -0.1), np.full(5, 0.79))
    g_tde = LayeredTDE(two_layer_tables).los_matrix(mesh, xs, ys, *look, basis=basis)
    g_pts = point.los_matrix(mesh, xs, ys, *look, basis=basis)
    assert np.abs(g_tde - g_pts).max() < 1e-2 * np.abs(g_pts).max()


def test_far_surface_receivers_take_point_sources_of_the_same_field(two_layer_tables):
    """The far path (whole surface field, point sources) against the near one, 4-30 sizes out.

    Against tent-weighted point sources of the exact layered kernel (converged to 1e-4), the
    crossing triangle's tents measured 1.3e-3 (near) and 2.9e-3 (far) at 10 sizes, and up to
    7e-3 elsewhere on these coarse test tables; element fields 1e-3 for both.
    """
    near = LayeredTDE(two_layer_tables, far=None)
    far = LayeredTDE(two_layer_tables, far=4.0)
    size = np.sqrt(np.linalg.norm(np.cross(BELOW[1] - BELOW[0], BELOW[2] - BELOW[0])))
    angle = np.linspace(0.0, 2 * np.pi, 9)[:-1]
    reach = np.repeat([5.0, 10.0, 30.0], angle.size) * size
    x = 500.0 + reach * np.cos(np.tile(angle, 3))
    y = 300.0 + reach * np.sin(np.tile(angle, 3))
    z = np.zeros(x.size)
    assert far._far(BELOW, x, y, z).all()
    a, b = near.element_field(BELOW, x, y, z), far.element_field(BELOW, x, y, z)
    for i in range(x.size):
        assert np.abs(a[i] - b[i]).max() < 5e-3 * np.abs(a[i]).max()
    ta, tb = near.tent_field(CROSSING, x, y, z), far.tent_field(CROSSING, x, y, z)
    for i in range(x.size):
        assert np.abs(ta[i] - tb[i]).max() < 1e-2 * np.abs(ta[i]).max()
