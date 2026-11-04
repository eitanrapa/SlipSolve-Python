"""``slipsolve._okada92``: Okada's (1992) half-space point source, the strain far path's half-space part."""
import numpy as np
import pytest

from slipsolve._okada92 import point_source_enu, point_source_unit
from slipsolve._tde import _element_basis, tde_disp_hs_unit, tde_strain_hs_unit
from slipsolve.layered_tde import _rule

NU = 0.27


def _triangle(strike, dip, size=600.0, depth=4000.0):
    """A triangle of about ``size`` metres centred at ``depth``, strike as an azimuth, dipping to its right."""
    s, d = np.radians(strike), np.radians(dip)
    c = np.array([0.0, 0.0, -depth])
    e_s = np.array([np.sin(s), np.cos(s), 0.0])
    e_d = np.array([np.cos(s) * np.cos(d), -np.sin(s) * np.cos(d), -np.sin(d)])
    return np.array([c + size * e_s, c - 0.5 * size * e_s + 0.8 * size * e_d, c - 0.5 * size * e_s - 0.8 * size * e_d])


def _receivers(n, seed):
    rng = np.random.default_rng(seed)
    r, az = rng.uniform(4000.0, 15000.0, n), rng.uniform(0.0, 2.0 * np.pi, n)
    return r * np.sin(az), r * np.cos(az), -rng.choice([0.0, 1500.0, 4000.0, 7000.0], n)


def _integrated(tri, x, y, z, order=40):
    """Unit strike-, dip- and tensile-slip on ``tri`` as point sources at ``order**2`` sub-triangle centroids."""
    vnorm, vstrike, vdip = _element_basis(*tri)
    bary, share = _rule(order)
    pts = bary @ tri
    m, n = pts.shape[0], x.size
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    u, e = point_source_enu(NU, vnorm, np.stack([vstrike, vdip, vnorm]), np.tile(pts[:, 0], n),
                            np.tile(pts[:, 1], n), np.tile(pts[:, 2], n), np.repeat(x, m), np.repeat(y, m),
                            np.repeat(z, m))
    w = (np.tile(share, n) * area)[:, None, None]
    return (u * w).reshape(n, m, 3, 3).sum(axis=1), (e * w).reshape(n, m, 3, 6).sum(axis=1)


@pytest.mark.parametrize("strike", [0.0, 140.0])
@pytest.mark.parametrize("dip", [25.0, 60.0, 80.0, 90.0, 115.0])
def test_point_sources_integrate_to_the_dislocation(strike, dip):
    """Over a 600 m triangle, 6.7-25 sizes away, surface to 7 km: the TDE's field, every slip.

    Measured (2026-11-04, 4 strikes, 7 dips): displacement 8e-6, strain 1.4e-4 of the peak
    away from vertical (the rule's 1600 points and the TDE's differences); the sign
    conventions (Okada's hanging wall against the TDE's normal) need no fudge.
    """
    tri = _triangle(strike, dip)
    x, y, z = _receivers(24, seed=int(dip))
    u, e = _integrated(tri, x, y, z)
    ref_u = np.moveaxis(tde_disp_hs_unit(x, y, z, *tri, NU), -1, 0)
    ref_e = np.moveaxis(tde_strain_hs_unit(x, y, z, *tri, NU), -1, 0)
    for k in range(3):
        assert np.abs(u[:, k] - ref_u[:, k]).max() < 2e-5 * np.abs(ref_u[:, k]).max()
        assert np.abs(e[:, k] - ref_e[:, k]).max() < 3e-4 * np.abs(ref_e[:, k]).max()


def test_near_vertical_it_matches_cutde():
    """At 89.9 deg the TDE's differences are 2e-4 off (its near-vertical band); ``cutde`` and the port agree to 1e-5."""
    hs = pytest.importorskip("cutde.halfspace")
    tri = _triangle(140.0, 89.9)
    x, y, z = _receivers(24, seed=5)
    _, e = _integrated(tri, x, y, z)
    cut = np.moveaxis(hs.strain_matrix(np.column_stack([x, y, z]), tri[None], NU)[:, :, 0, :], 2, 1)
    assert np.abs(e - cut).max() < 3e-5 * np.abs(cut).max()


def test_the_free_surface_is_traction_free():
    """At ``z = 0`` the point source's stress has no vertical traction: Okada's images, checked without the TDE."""
    rng = np.random.default_rng(2)
    n = 40
    x, y = rng.uniform(-8e3, 8e3, n), rng.uniform(-8e3, 8e3, n)
    for dip in (30.0, 75.0, 90.0):
        s, c = np.sin(np.radians(dip)), np.cos(np.radians(dip))
        unit = point_source_unit(1.0 / (2.0 * (1.0 - NU)), x, y, np.zeros(n), 3000.0, s, c)   # (3, n, 12)
        grad = np.stack([unit[..., 3:6], unit[..., 6:9], unit[..., 9:12]], axis=-1)           # [i, j] = du_i/dx_j
        strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
        lam = 2.0 * NU / (1.0 - 2.0 * NU)                                                   # mu = 1
        stress = lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2.0 * strain
        assert np.abs(stress[..., :, 2]).max() < 1e-12 * np.abs(stress).max()


def test_its_edge_cases():
    """On the source it returns zeros, as DC3D0 does; above the surface it refuses."""
    out = point_source_unit(0.68, [0.0, 1000.0], [0.0, 0.0], [-3000.0, -3000.0], 3000.0, 1.0, 0.0)
    assert np.all(out[:, 0] == 0.0) and np.all(np.isfinite(out[:, 1])) and np.any(out[:, 1] != 0.0)
    with pytest.raises(ValueError, match="z <= 0"):
        point_source_unit(0.68, [0.0], [0.0], [10.0], 3000.0, 1.0, 0.0)
