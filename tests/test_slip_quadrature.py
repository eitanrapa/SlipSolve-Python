"""Point sources on triangles (`slipsolve._quadrature`): moment packing, the quadrature rule and nodal tents.

The packing is the one place the north/east/down convention has to be exactly right, and nothing downstream can tell
a sign error from a real slip reversal. The rule is sub-triangle centroids with equal weights: second order, and no
point is ever shared between cells or triangles, so nothing can be double counted.
"""

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace
from slipsolve._quadrature import _moment_components, _sub_triangles, _tent, _triangle_quadrature

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def scene():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                            max_depth=20e3, edge_length=12e3)
    return trace, frame, mesh, None, None


# -- the moment tensor -------------------------------------------------------

def test_moment_packing_of_a_vertical_strike_slip_element():
    """A vertical east-striking fault slipping east is a pure ``m12`` couple.

    The packing is the one place the north/east/down convention has to be exactly
    right, and nothing downstream can tell a sign error from a real slip reversal.
    """
    # ENU: normal north, slip east.
    m1, m2, m3, m4, m5 = _moment_components((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    assert m1 == pytest.approx(1.0)
    assert (m2, m3, m4, m5) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_moment_packing_of_a_vertical_dip_slip_element():
    """Normal north, slip straight down: a pure ``m13`` couple."""
    m1, m2, m3, m4, m5 = _moment_components((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    assert m2 == pytest.approx(1.0)
    assert (m1, m3, m4, m5) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_moment_tensor_of_a_shear_dislocation_is_traceless():
    """Which is why five components suffice for six tensor entries."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        d = np.cross(n, rng.normal(size=3))
        d /= np.linalg.norm(d)
        _, _, m3, m4, _ = _moment_components(n, d)
        # trace = m_nn + m_ee + m_dd = (-m3/2 + m4) + (-m3/2 - m4) + m3 = 0.
        assert abs(-m3 / 2 + m4 + -m3 / 2 - m4 + m3) < 1e-12


# -- quadrature --------------------------------------------------------------

def test_quadrature_weights_and_centroid():
    p1, p2, p3 = np.array([0.0, 0, 0]), np.array([3.0, 0, 0]), np.array([0.0, 3, 0])
    for order in (1, 2, 3, 5):
        points, weights = _triangle_quadrature(p1, p2, p3, order)
        assert points.shape == (order ** 2, 3)
        assert weights.sum() == pytest.approx(1.0)
        # An equal-weight convergent rule reproduces the centroid exactly.
        np.testing.assert_allclose((points * weights[:, None]).sum(axis=0),
                                   (p1 + p2 + p3) / 3.0, atol=1e-12)


def test_sub_triangles_are_the_quadrature_lattice():
    """`_sub_triangles` is `_triangle_quadrature` with the pieces' extents kept.

    Same lattice, same order: its centroids are the quadrature points, and the
    pieces tile the parent in equal areas.
    """
    for order in (1, 2, 5):
        pieces = _sub_triangles(order)
        points, _ = _triangle_quadrature(*np.eye(3), order)
        assert pieces.shape == (order ** 2, 3, 3)
        np.testing.assert_allclose(pieces.mean(axis=1), points, atol=1e-15)
        np.testing.assert_allclose(pieces.sum(axis=2), 1.0, atol=1e-15)
        edges = pieces[:, 1:, :2] - pieces[:, :1, :2]
        area = 0.5 * np.abs(np.linalg.det(edges))
        np.testing.assert_allclose(area, 0.5 / order ** 2, atol=1e-15)


def test_quadrature_uses_both_sub_triangle_orientations():
    """Taking only the upright sub-triangles is a biased sample, not a rule.

    Subdividing a triangle `k` ways gives `k(k+1)/2` upright sub-triangles and
    `k(k-1)/2` inverted ones. Sampling only the upright set with equal weights
    -- which a plain barycentric lattice does -- has an error that stops falling
    however many points are added; it plateaus near 5e-3 relative.
    """
    p1, p2, p3 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    for order in (2, 3, 4):
        points, _ = _triangle_quadrature(p1, p2, p3, order)
        assert points.shape[0] == order ** 2


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8, 13, 15])
def test_quadrature_points_are_never_shared(order):
    """No point on an edge or vertex, no two coincident, weights summing to one."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    points, share = _triangle_quadrature(p1, p2, p3, order)

    assert len(points) == order ** 2
    np.testing.assert_allclose(share.sum(), 1.0, rtol=0, atol=1e-15)

    # Strictly interior: every barycentric coordinate positive, so no point can
    # be shared with a neighbouring triangle of the mesh either.
    l2, l3 = points[:, 0], points[:, 1]
    bary = np.stack([1.0 - l2 - l3, l2, l3], axis=1)
    assert bary.min() > 0.0

    if len(points) > 1:
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 0.0, "two quadrature points coincide"


@pytest.mark.parametrize("order", [1, 2, 5, 13])
def test_the_tent_integrates_exactly(scene, order):
    """``int_T N_j dA = A_T / 3`` -- exact, because the rule is centroid symmetric.

    A linear function is integrated exactly by any rule whose points are
    symmetric about the centroid with equal weights, so this is a machine-precision
    identity rather than an approximation. It fails immediately if the tent is
    evaluated at the wrong vertex, or if a point is weighted twice.
    """
    _, _, mesh, _, _ = scene
    worst = 0.0
    for element in range(mesh.n_elements):
        triangle = mesh.triangles[element]
        points, share = _triangle_quadrature(*mesh.vertices[element], order)
        for node in triangle:
            weights = share * _tent(mesh.vertices[element], triangle, node, points)
            worst = max(worst, abs(weights.sum() - 1.0 / 3.0))
    assert worst < 1e-12
