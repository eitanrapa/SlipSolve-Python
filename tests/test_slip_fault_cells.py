"""Cells that follow the fault (``slipsolve.fault_cells``): prisms extruded from the fault's triangles."""

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace
from slipsolve import wavenumber as wn
from slipsolve.edgrn import VelocityModel
from slipsolve.fault_cells import FaultCells, PrismSource
from slipsolve.heterogeneity import HeterogeneousEngine, VoxelSource, from_voigt, to_voigt
from slipsolve.layered_tde import LayeredTDE

ONE_LAYER = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)
TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3],
                           [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0],
                           [2370.0, 2370.0, 2750.0, 2750.0])


@pytest.fixture(scope="module")
def uniform_tables():
    tops, _, _ = wn.layer_stack(ONE_LAYER)
    z = wn.depth_grid(tops, z_max=40e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    return wn.RemainderTables.compute(ONE_LAYER, z=z, r=wn.distance_grid(per_decade=8), n_k=512, z_max=40e3,
                                      surface_z=wn.surface_depth_grid(tops, fine_to=20e3, z_max=40e3, spacing=100.0),
                                      strain=True)


@pytest.fixture(scope="module")
def coarse_tables():
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=30e3, first=20.0, growth=2.0, spacing=1500.0, deep_spacing=6000.0)
    return wn.RemainderTables.compute(TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=8), n_k=1024,
                                      surface_z=z, surface_r=wn.distance_grid(per_decade=8), strain=True)


def test_two_prisms_make_the_box(uniform_tables):
    """A box cut on its vertical diagonal plane: the two prisms' fields sum to the box's, outside and inside.

    Eshelby's construction is linear and the shared diagonal face cancels; the box's faces are triangulated
    differently, which a linear Burgers vector does not see. Measured (2026-11-05): displacement 1.7e-5 / 1.1e-6
    (outside / inside), strain 1.9e-4 / 1.6e-5 of each point's peak.
    """
    engine = LayeredTDE(uniform_tables)
    z0, z1 = -3000.0, -2000.0
    box = VoxelSource(engine, (0.0, 1000.0), (0.0, 1000.0), (-z1, -z0))
    a = PrismSource(engine, [[0, 0, z0], [1000, 0, z0], [1000, 1000, z0]], [[0, 0, z1], [1000, 0, z1], [1000, 1000, z1]])
    b = PrismSource(engine, [[0, 0, z0], [1000, 1000, z0], [0, 1000, z0]], [[0, 0, z1], [1000, 1000, z1], [0, 1000, z1]])
    assert np.isclose(a.volume + b.volume, box.volume) and a.pieces is None
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-3e3, 4e3, 30), rng.uniform(-3e3, 4e3, 30), rng.uniform(-6e3, 0.0, 30)])
    pts = np.vstack([pts[box._gap(*pts.T) > 150.0], [[300.0, 600.0, -2500.0], [700.0, 200.0, -2800.0]]])
    assert a.inside(*pts[-1:].T).all() and b.inside(*pts[-2:-1].T).all()
    for which, bar in (("displacement", 1e-4), ("strain", 1e-3)):
        want = getattr(box, which)(*pts.T)
        got = getattr(a, which)(*pts.T) + getattr(b, which)(*pts.T)
        assert (np.abs(got - want).max(axis=(1, 2)) / np.abs(want).max(axis=(1, 2))).max() < bar


def test_a_prism_turned_about_the_vertical_turns_its_field(uniform_tables):
    """In a uniform half-space: a prism and its moment turned by R give R times the field (R eps R^T for strain)."""
    engine = LayeredTDE(uniform_tables)
    inner = np.array([[0.0, 0.0, -3000.0], [1500.0, 200.0, -3000.0], [600.0, 0.0, -4200.0]])
    outer = inner + np.array([0.0, 900.0, 0.0])
    th = np.radians(37.0)
    rot = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    p, q = PrismSource(engine, inner, outer), PrismSource(engine, inner @ rot.T, outer @ rot.T)
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(-6e3, 6e3, 25), rng.uniform(-6e3, 6e3, 25), rng.uniform(-8e3, 0.0, 25)])
    pts = pts[p._gap(*pts.T) > 300.0]
    moment = np.array([[1.0, 0.3, -0.2], [0.3, -0.5, 0.4], [-0.2, 0.4, 0.7]])
    m, m_rot = to_voigt(moment), to_voigt(rot @ moment @ rot.T)
    u = np.einsum("k,pkc->pc", m, p.displacement(*pts.T))
    u_rot = np.einsum("k,pkc->pc", m_rot, q.displacement(*(pts @ rot.T).T))
    assert (np.abs(u_rot - u @ rot.T).max(axis=1) / np.abs(u).max(axis=1)).max() < 2e-4
    eps = from_voigt(np.einsum("k,pka->pa", m, p.strain(*pts.T)))
    eps_rot = from_voigt(np.einsum("k,pka->pa", m_rot, q.strain(*(pts @ rot.T).T)))
    want = np.einsum("ij,pjk,lk->pil", rot, eps, rot)
    assert (np.abs(eps_rot - want).max(axis=(1, 2)) / np.abs(want).max(axis=(1, 2))).max() < 2e-4


def test_a_flat_prism_is_summed_as_near_cubic_pieces(uniform_tables):
    """A prism 4 km across and 0.5 km thick: midpoint pieces, whose volumes sum to its own, and whose fields are its
    own inside its longest side (beyond it the whole cell's rule takes over, the next test)."""
    engine = LayeredTDE(uniform_tables)
    inner = np.array([[0.0, 0.0, -3000.0], [4000.0, 0.0, -3000.0], [2000.0, 0.0, -6400.0]])
    p = PrismSource(engine, inner, inner + np.array([0.0, 500.0, 0.0]))
    assert p.pieces is not None and len(p.pieces) == 4
    assert np.isclose(sum(q.volume for q in p.pieces), p.volume)
    pts = np.array([[2000.0, 1500.0, -4000.0], [-1000.0, 250.0, -3500.0]])
    assert not p._far_whole(*pts.T).any()
    whole = sum(q.strain(*pts.T) for q in p.pieces)
    np.testing.assert_allclose(p.strain(*pts.T), whole, rtol=1e-12, atol=0)


def _vertical_mesh(max_depth=6e3, edge=2e3):
    trace = FaultTrace.from_lonlat(np.array([-80.05, -79.95]), np.array([8.0, 8.02]))
    frame = trace.local_frame()
    return trace, FaultMesh.vertical(trace, frame, max_depth=max_depth, edge_length=edge)


def test_fault_cells_follow_the_mesh_and_feed_the_engine(coarse_tables):
    """Cells per side and layer and element, ending on the fault; the dense engine takes them; no departure is G0."""
    tables = coarse_tables
    base = LayeredTDE(tables)
    trace, mesh = _vertical_mesh()

    def soft(x, y, depth, distance, side):
        layer = tables.layer(depth)
        f = np.where((side > 0) & (depth < 4e3), 0.6, 1.0)
        return f * tables.lam[layer], f * tables.mu[layer]

    cells = FaultCells(tables, mesh, [800.0, 1200.0], soft, sides=(-1, 1))
    n = mesh.n_elements
    assert cells.inner.shape == (2 * 2 * n, 3, 3)
    np.testing.assert_allclose(cells.inner[:n], np.asarray(mesh.vertices))              # the innermost on the fault
    np.testing.assert_allclose(cells.inner[n:2 * n], cells.outer[:n])                   # layers share faces
    assert np.all(cells.side[cells.active] == 1) and cells.n_active > 0
    assert cells.check_clearance(mesh)
    _, other = _vertical_mesh(max_depth=4e3)
    with pytest.raises(ValueError, match="another mesh"):
        cells.check_clearance(other)
    rng = np.random.default_rng(6)
    _, ty = (np.asarray(v) for v in trace.to_local(mesh.frame))
    x, y = rng.uniform(-6e3, 6e3, 20), float(np.mean(ty)) + rng.uniform(-8e3, 8e3, 20)
    look = (np.full(20, 0.6), np.full(20, -0.1), np.full(20, 0.79))
    g0 = base.los_matrix(mesh, x, y, *look)
    eng = HeterogeneousEngine(base, cells)
    g = eng.los_matrix(mesh, x, y, *look)
    assert np.isfinite(g).all() and np.abs(g - g0).max() > 1e-3 * np.abs(g0).max()
    born = HeterogeneousEngine(base, cells, order="born1").los_matrix(mesh, x, y, *look)
    assert np.abs(born - g0).max() > np.abs(g - born).max()                             # the first order dominates
    none = FaultCells(tables, mesh, [800.0], lambda x, y, d, s, side: (tables.lam[tables.layer(d)],
                                                                        tables.mu[tables.layer(d)]))
    assert none.n_active == 0
    np.testing.assert_array_equal(HeterogeneousEngine(base, none).los_matrix(mesh, x, y, *look), g0)
    with pytest.raises(ValueError, match="fault cells are solved densely"):
        HeterogeneousEngine(base, cells, solver="fft")


def test_fault_cells_across_an_interface_are_refused(coarse_tables):
    """Elements crossing the 2 km interface give cells in two layers: refused, naming the interfaces."""
    tables = coarse_tables
    _, mesh = _vertical_mesh(max_depth=6e3, edge=3e3)
    with pytest.raises(ValueError, match="cross a layer interface"):
        FaultCells(tables, mesh, [1000.0], lambda x, y, d, s, side: (0.5 * tables.lam[tables.layer(d)],
                                                                     0.5 * tables.mu[tables.layer(d)]))


def test_beyond_its_longest_side_a_cell_in_pieces_takes_its_whole_volume_rule(uniform_tables):
    """The strain of a 15 km prism in pieces, beyond its longest side: its own volume rule = the sum of its pieces.

    Measured (2026-11-05): 5e-5 of each point's peak (a 3.9 km voxel: 1.5e-4); the pieces are for the near field.
    """
    engine = LayeredTDE(uniform_tables)
    inner = np.array([[0.0, 0.0, -3000.0], [15000.0, 0.0, -3000.0], [7000.0, 0.0, -4000.0]])
    cell = PrismSource(engine, inner, inner + np.array([0.0, 1000.0, 0.0]))
    pts = np.array([[7000.0, 31000.0, -3500.0], [-20000.0, 4000.0, -1000.0], [30000.0, -20000.0, -12000.0]])
    assert cell.pieces is not None and cell._far_whole(*pts.T).all()
    whole = cell.strain(*pts.T)
    pieces = sum(q.strain(*pts.T) for q in cell.pieces)
    assert (np.abs(whole - pieces).max(axis=(1, 2)) / np.abs(pieces).max(axis=(1, 2))).max() < 5e-4
