"""Pins faults between blocks of different rigidity (:mod:`slipsolve.bimaterial`).

The scaled mode is a row and column scaling of any engine, so it is tested
against the plain engine's own matrix. The exact mode is tested on what defines
it, without trusting any of its pieces:

- its field jumps by the slip across the fault and by nothing anywhere else on
  the interface, including across a change of ratio along strike;
- in the middle of a long fault it reduces to the 2-D antiplane solution, solved
  here independently with screw dislocations and images.
"""

import numpy as np
import pytest

from slipsolve import (Bimaterial, DipSweep, FaultMesh, FaultTrace, HalfSpaceTDE,
                       Observations, RigidityContrast, SlipInversion, SlipModel)
from slipsolve import wavenumber as wn
from slipsolve.edgrn import VelocityModel
from slipsolve.layered_tde import LayeredTDE

import slip_synthetic as ss

_LON = np.array([-68.9, -68.6, -68.3, -68.0, -67.7])
_LAT = np.array([10.50, 10.52, 10.53, 10.55, 10.56])


@pytest.fixture(scope="module")
def setup():
    """A gently curving fault, data on both sides from two tracks."""
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="bimaterial")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=10e3)
    s, z = mesh.element_params[:, 0], mesh.element_params[:, 1]
    amp = np.exp(-((s - 60e3) / 35e3) ** 2) * np.clip(1.0 + z / 12e3, 0.0, 1.0)
    slip = np.concatenate([-2.0 * amp, 0.3 * amp])
    obs = Observations.concat([
        Observations.from_los(
            ss.forward_los_stack(mesh, slip, trace, frame, geometry=geometry, spacing=6000.0,
                                 half_x=90e3, half_y=50e3, noise=0.001, seed=seed),
            name=geometry, frame=frame, trace=trace, rms_min=0.002, width_min=6000.0,
            width_max=40000.0, exclude_within=3000.0)
        for seed, geometry in enumerate(("asc", "desc"))])
    return trace, frame, mesh, obs


def _points(obs):
    ds = obs.ds
    return (ds["x"].values, ds["y"].values, ds["look_e"].values, ds["look_n"].values,
            ds["look_u"].values)


# -- the contrast ---------------------------------------------------------------
def test_a_compass_side_is_resolved_against_the_trace(setup):
    trace, frame, _, _ = setup
    assert RigidityContrast(trace, side="north", frame=frame).side == "left"
    assert RigidityContrast(trace, side="south", frame=frame).side == "right"
    backwards = FaultTrace.from_lonlat(_LON[::-1], _LAT[::-1])
    assert RigidityContrast(backwards, side="north", frame=frame).side == "right"
    with pytest.raises(ValueError, match="does not name one block"):
        RigidityContrast(trace, side="east", frame=frame)


def test_ratios_and_breaks_are_checked(setup):
    trace, frame, _, _ = setup
    contrast = RigidityContrast(trace, ratio=(2.0, 1.0), breaks=(55e3,), frame=frame)
    np.testing.assert_array_equal(contrast.interval([0.0, 54e3, 56e3, 1e6]), [0, 0, 1, 1])
    with pytest.raises(ValueError, match="intervals"):
        RigidityContrast(trace, ratio=(2.0, 1.0, 1.5), breaks=(55e3,), frame=frame)
    with pytest.raises(ValueError, match="inside the trace"):
        RigidityContrast(trace, ratio=2.0, breaks=(-5.0,), frame=frame)
    with pytest.raises(ValueError, match="positive"):
        RigidityContrast(trace, ratio=0.0, frame=frame)


# -- the scaled mode --------------------------------------------------------------
def test_the_scaled_mode_is_the_plain_matrix_scaled(setup):
    """Rows 2r/(1+r) on the named side and 2/(1+r) on the other, whatever the basis."""
    trace, frame, mesh, obs = setup
    r = 2.0
    contrast = RigidityContrast(trace, side="north", ratio=r, frame=frame)
    x, y, *look = _points(obs)
    plain = HalfSpaceTDE().los_matrix(mesh, x, y, *look)
    north = trace.side(x, y, frame) > 0
    expected = np.where(north, 2 * r / (1 + r), 2 / (1 + r))[:, None] * plain
    np.testing.assert_allclose(Bimaterial(contrast).los_matrix(mesh, x, y, *look), expected,
                               rtol=1e-12, atol=0)
    nodal = SlipInversion(mesh, obs, engine=Bimaterial(contrast), basis="node").g
    direct = SlipInversion(mesh, obs, basis="node").g
    np.testing.assert_allclose(nodal, np.where(north, 2 * r / (1 + r), 2 / (1 + r))[:, None]
                               * direct, rtol=1e-12, atol=1e-15)


def test_no_contrast_changes_nothing(setup):
    trace, frame, mesh, obs = setup
    x, y, *look = _points(obs)
    plain = HalfSpaceTDE().los_matrix(mesh, x[:60], y[:60], *(v[:60] for v in look))
    for exact in (False, True):
        engine = Bimaterial(RigidityContrast(trace, ratio=1.0, frame=frame), exact=exact)
        np.testing.assert_allclose(engine.los_matrix(mesh, x[:60], y[:60], *(v[:60] for v in look)),
                                   plain, rtol=0, atol=1e-12 * np.abs(plain).max())


def test_the_moment_takes_the_harmonic_mean(setup):
    """Moment rigidity = 2 mu+ mu- / (mu+ + mu-), per interval, from either reference block."""
    trace, frame, mesh, obs = setup
    r = (2.0, 1.25)
    for reference, factor in (("other", lambda q: 2 / (1 + q)), ("side", lambda q: 2 * q / (1 + q))):
        contrast = RigidityContrast(trace, side="north", ratio=r, breaks=(60e3,),
                                    reference=reference, frame=frame)
        model = SlipInversion(mesh, obs, engine=Bimaterial(contrast)).solve(smoothing=0.3)
        interval = contrast.interval(mesh.element_params[:, 0])
        expected = np.sum(30e9 * factor(np.asarray(r))[interval] * mesh.areas
                          * np.hypot(*model.element_slip.T))
        assert model.moment(shear_modulus=30e9) == pytest.approx(expected, rel=1e-12)


def test_a_saved_model_keeps_its_contrast(setup, tmp_path):
    trace, frame, mesh, obs = setup
    contrast = RigidityContrast(trace, side="north", ratio=(2.0, 1.0), breaks=(60e3,), frame=frame)
    model = SlipInversion(mesh, obs, engine=Bimaterial(contrast), basis="node").solve(smoothing=0.3)
    loaded = SlipModel.load(model.save(tmp_path / "bimaterial.slip.zip"))
    assert loaded.moment() == pytest.approx(model.moment(), rel=1e-12)
    x, y, *look = _points(obs)
    np.testing.assert_allclose(loaded.forward(x, y, look=look), model.forward(x, y, look=look),
                               rtol=1e-12, atol=1e-15)


# -- the exact mode -----------------------------------------------------------------
def test_the_exact_mode_needs_the_half_space_engine(setup):
    trace, frame, _, _ = setup

    class Layered:
        name, nu, supports_basis = "layered_point_source", 0.25, True

    with pytest.raises(NotImplementedError, match="opening"):
        Bimaterial(RigidityContrast(trace, ratio=2.0, frame=frame), engine=Layered(), exact=True)


def test_the_exact_field_jumps_by_the_slip_and_nowhere_else():
    """Across the fault by the slip; across its continuation and a step along strike, by nothing."""
    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.55]), np.array([8.0, 8.05]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=60.0, max_depth=10e3, edge_length=5e3)
    contrast = RigidityContrast(trace, side="left", ratio=(2.0, 1.25), breaks=(25e3,), frame=frame)
    solve = Bimaterial(contrast, exact=True, extend=dict(depth=1.0, strike=1.0, cross=1.0)
                       ).interface(mesh)
    assert {"fault", "extension", "break"} <= set(solve.kind)
    s, z = mesh.element_params[:, 0], mesh.element_params[:, 1]
    slip = np.concatenate([np.sin(np.pi * s / s.max()) * (1 + z / 12e3), 0.3 * (1 + z / 12e3)])
    d = solve.discontinuity(slip)
    rng = np.random.default_rng(0)
    for kind in ("fault", "extension", "break"):
        picks = rng.choice(np.flatnonzero(solve.kind == kind), 12, replace=False)
        offset = 1e-4 * np.sqrt(solve.areas[picks])[:, None] * solve.normals[picks]
        above = solve.field(d, *(solve.centroids[picks] + offset).T)
        below = solve.field(d, *(solve.centroids[picks] - offset).T)
        jump = solve.a_plus[picks, None] * above - solve.a_minus[picks, None] * below
        local = np.einsum("eki,ei->ek", solve.frames[picks], jump)
        target = np.zeros_like(local)
        if kind == "fault":
            target[:, 0] = slip[picks]
            target[:, 1] = slip[mesh.n_elements + picks]
        np.testing.assert_allclose(local, target, atol=2e-3 * np.abs(slip).max())


def test_a_new_ratio_reuses_the_interface():
    """``with_contrast`` and ``response_matrix`` give what a fresh solve at that ratio gives."""
    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.6]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=10e3, edge_length=5e3)
    reach = dict(depth=1.0, strike=1.0, cross=1.0)
    two = RigidityContrast(trace, side="left", ratio=(2.0, 1.2), breaks=(20e3,), frame=frame)
    other = two.with_ratio((1.5, 0.8))
    built = Bimaterial(two, exact=True, extend=reach).interface(mesh)
    fresh = Bimaterial(other, exact=True, extend=reach).interface(mesh)
    reused = built.with_contrast(other)
    np.testing.assert_array_equal(reused.a_plus, fresh.a_plus)
    np.testing.assert_allclose(reused.solution, fresh.solution, rtol=0,
                               atol=1e-12 * np.abs(fresh.solution).max())
    x, y = trace.from_curvilinear(np.array([5e3, 20e3, 35e3]), np.array([4e3, -6e3, 9e3]), frame)
    look = (np.full(3, 0.6), np.full(3, -0.1), np.full(3, 0.79))
    g = other.observation_factor(x, y)[:, None] * (reused.response_matrix(x, y, look=look)
                                                   @ reused.solution)
    np.testing.assert_allclose(g, fresh.los(x, y, look=look) * other.observation_factor(x, y)[:, None],
                               rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="same"):
        built.with_contrast(RigidityContrast(trace, side="right", ratio=(2.0, 1.2), breaks=(20e3,),
                                             frame=frame))


def test_the_exact_field_is_continuous_across_a_step_along_strike():
    """The scaled mode steps by the ratio of the two intervals' ``a``; the exact one hardly does.

    Element-constant discontinuities leave the exact field a few percent off within
    about an element of the boundary (measured 1-7% just across it, at the surface
    and at depth, against the scaled mode's 50%), so this asks for most of the
    step to go, not all of it.
    """
    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.45]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=4e3)
    contrast = RigidityContrast(trace, side="north", ratio=(2.0, 1.0), breaks=(30e3,), frame=frame)
    slip = np.zeros((mesh.n_elements, 2))
    slip[:, 0] = np.clip(1 + mesh.element_params[:, 1] / 15e3, 0, 1)
    jumps = {}
    for exact in (False, True):
        engine = Bimaterial(contrast, exact=exact, extend=dict(depth=1.0, strike=1.0, cross=1.5))
        north = np.array([2e3, 4e3, 8e3, 16e3])
        x0, y0 = trace.from_curvilinear(np.full(north.size, 30e3), north, frame)
        u_west = engine.forward(mesh, slip, x0 - 20.0, y0)
        u_east = engine.forward(mesh, slip, x0 + 20.0, y0)
        jumps[exact] = np.linalg.norm(u_east - u_west, axis=1) / np.linalg.norm(u_west, axis=1)
    assert np.all(jumps[False] > 0.3)
    assert np.all(jumps[True] < 0.2 * jumps[False])


def _antiplane(xs, dip, depth, r):
    """2-D strike-slip, uniform slip from the trace to ``depth`` (km), dipping toward +x.

    The fault plane, continued to 200 depths, is the boundary between a block
    with ``mu`` on the +x side and ``r mu`` on the -x side. Screw dislocations with
    their free-surface images; the discontinuity on the boundary solves
    ``D + 2k Fbar(D) = s``, ``Fbar`` coming from the images alone (a straight line
    sees nothing of itself). Returns the homogeneous and the bimaterial surface
    displacement per unit slip.
    """
    def omega(px, pz, ax, az, bx, bz):
        ux, uz, vx, vz = ax - px, az - pz, bx - px, bz - pz
        return np.arctan2(ux * vz - uz * vx, ux * vx + uz * vz)

    k = (r - 1) / (r + 1)
    cot = 1.0 / np.tan(np.radians(dip))
    z = np.r_[np.linspace(0.0, depth, 401), np.geomspace(depth, 200 * depth, 300)[1:]]
    x = cot * z
    zm, xm = 0.5 * (z[1:] + z[:-1]), cot * 0.5 * (z[1:] + z[:-1])
    s = np.where(zm < depth, 1.0, 0.0)
    # Field = (fault - image) / 2 pi; the image term is all that survives on the line.
    fbar = -omega(xm[:, None], zm[:, None], x[None, :-1], -z[None, :-1], x[None, 1:],
                  -z[None, 1:]) / (2 * np.pi)
    # Jump of the fault term across the line, +x side minus -x side, is -1 per unit D.
    d = np.linalg.solve(np.eye(zm.size) - 2 * k * fbar, s)
    surface = (omega(xs[:, None], 0 * xs[:, None], x[None, :-1], z[None, :-1], x[None, 1:],
                     z[None, 1:]) / np.pi)
    plus = xs > 0
    return surface @ s, np.where(plus, 1 + k, 1 - k) * (surface @ d)


def test_the_exact_mode_reduces_to_the_2d_solution():
    """Mid-length of a long 60 degree fault: the contrast's effect is the 2-D one.

    Measured at full resolution (a 400 km fault at 50 degrees, 5 km elements, the
    default reach): correlation 0.998 and 94% of the 2-D amplitude. This coarse
    version keeps the test to a minute.
    """
    dip, r, depth = 60.0, 2.0, 15.0
    trace = FaultTrace.from_lonlat(np.array([-82.0, -79.3]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=dip, max_depth=depth * 1e3, edge_length=5e3)
    contrast = RigidityContrast(trace, side="left", ratio=r, frame=frame)
    xs = np.r_[-np.arange(45.0, 2.0, -3.0), np.arange(3.0, 45.5, 3.0)]
    length = trace.length(frame)
    px, py = trace.from_curvilinear(np.full(xs.size, length / 2), xs * 1e3, frame)
    slip = np.zeros((mesh.n_elements, 2))
    slip[:, 0] = 1.0
    scaled = Bimaterial(contrast).forward(mesh, slip, px, py)[:, 0]
    reach = dict(depth=3.0, strike=1.0, cross=1.0)        # the ends are far from mid-length
    exact = Bimaterial(contrast, exact=True, extend=reach).forward(mesh, slip, px, py)[:, 0]
    homogeneous, bimaterial = _antiplane(xs, dip, depth, r)
    k = (r - 1) / (r + 1)
    sign = np.sign(np.sum(scaled * homogeneous))
    effect_3d = sign * (exact - scaled)
    effect_2d = bimaterial - np.where(xs > 0, 1 + k, 1 - k) * homogeneous
    assert np.corrcoef(effect_3d, effect_2d)[0, 1] > 0.98
    assert 0.85 < np.linalg.norm(effect_3d) / np.linalg.norm(effect_2d) < 1.05


# -- a layered crust ---------------------------------------------------------------
#: Two layers with a strong shallow contrast, like the top of the Venezuela crust.
TWO_LAYERS = VelocityModel([0.0, 2000.0, 2000.0, 60e3], [3750.0, 3750.0, 6100.0, 6100.0],
                           [1950.0, 1950.0, 3500.0, 3500.0], [2370.0, 2370.0, 2750.0, 2750.0])


@pytest.fixture(scope="module")
def layered():
    tops, _, _ = wn.layer_stack(TWO_LAYERS)
    z = wn.depth_grid(tops, z_max=80e3, first=10.0, growth=1.6, spacing=400.0,
                      deep_spacing=4000.0)
    return LayeredTDE(wn.RemainderTables.compute(
        TWO_LAYERS, z=z, r=wn.distance_grid(per_decade=20), n_k=2048,
        surface_z=wn.surface_depth_grid(tops, fine_to=25e3, z_max=80e3, spacing=150.0)))


def _antiplane_fem(dip, depth, tops, mu, ys, c_plus=1.0, c_minus=1.0, h0=200.0, growth=1.08):
    """The same 2-D problem by finite elements, which need no Green's function at all.

    P1 triangles on a grid sheared along the fault, ``u = y - z cot(dip)``, so the
    fault and its continuation are the grid line ``u = 0``. The slit's nodes are
    doubled for ``z <= depth`` with ``w(+) = w(-) + 1``. ``mu`` is ``mu_ref(z)`` (the
    layers' moduli from ``tops``) times ``c_plus`` for ``u > 0`` (the dip side) and
    ``c_minus`` for ``u < 0`` -- either a function of depth. Free surface at ``z = 0``,
    ``w = 0`` 2e4 km away.
    Returns the surface displacement at ``ys`` (metres from the trace).
    """
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla

    def graded(extent):
        out, step = [0.0], h0
        while out[-1] < extent:
            out.append(out[-1] + step)
            step *= growth
        return np.array(out)

    cot = 1.0 / np.tan(np.radians(dip))
    half = graded(2e7)
    u = np.concatenate([-half[:0:-1], half])
    uniform = depth + 20e3
    z = np.unique(np.concatenate([np.arange(0.0, uniform + 0.5 * h0, h0),
                                  uniform + half[1:], np.asarray(tops)[1:], [depth]]))
    nu, nz = u.size, z.size
    grid_u, grid_z = np.meshgrid(u, z, indexing="ij")
    idx = np.arange(nu * nz).reshape(nu, nz)
    i0 = int(np.flatnonzero(u == 0.0)[0])
    slit = np.flatnonzero(z <= depth + 1e-9)
    n = nu * nz
    double = np.full(nz, -1)
    double[slit] = n + np.arange(slit.size)
    tris, plus = [], []
    for i in range(nu - 1):
        a, b, c, d = idx[i, :-1], idx[i + 1, :-1], idx[i + 1, 1:], idx[i, 1:]
        if i == i0:                                  # the + side's first column uses the doubles
            a = np.where(double[:-1] >= 0, double[:-1], a)
            d = np.where(double[1:] >= 0, double[1:], d)
        tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
        plus += [np.full(a.size, u[i] >= 0.0)] * 2
    tris, plus = np.concatenate(tris), np.concatenate(plus)
    py = np.concatenate([(grid_u + grid_z * cot).ravel(), cot * z[slit]])
    pz = np.concatenate([grid_z.ravel(), z[slit]])
    zc = pz[tris].mean(axis=1)
    layer = np.searchsorted(tops, zc, side="right") - 1
    scale = lambda c: c(zc) if callable(c) else c  # noqa: E731
    modulus = np.asarray(mu)[layer] * np.where(plus, scale(c_plus), scale(c_minus))
    (y1, y2, y3), (z1, z2, z3) = py[tris].T, pz[tris].T
    area2 = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    gb = np.stack([z2 - z3, z3 - z1, z1 - z2], 1) / area2[:, None]
    gc = np.stack([y3 - y2, y1 - y3, y2 - y1], 1) / area2[:, None]
    ke = (0.5 * np.abs(area2) * modulus)[:, None, None] * (
        gb[:, :, None] * gb[:, None, :] + gc[:, :, None] * gc[:, None, :])
    total = n + slit.size
    k = sps.csr_matrix((ke.ravel(), (np.repeat(tris, 3, 1).ravel(), np.tile(tris, (1, 3)).ravel())),
                       shape=(total, total))
    fixed = np.zeros(total, bool)
    fixed[np.concatenate([idx[0], idx[-1], idx[:, -1]])] = True
    free = np.flatnonzero(~fixed & (np.arange(total) < n))
    column = -np.ones(total, int)
    column[free] = np.arange(free.size)
    rows = np.concatenate([free, double[slit]])
    cols = np.concatenate([np.arange(free.size), column[idx[i0, slit]]])
    t = sps.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(total, free.size))
    g = np.zeros(total)
    g[double[slit]] = 1.0
    w = t @ spla.spsolve((t.T @ k @ t).tocsc(), -(t.T @ (k @ g))) + g
    surface = w[idx[:, 0]]
    plus_side = np.concatenate([[w[double[0]]], surface[i0 + 1:]])
    return np.where(ys > 0, np.interp(ys, np.concatenate([[0.0], u[i0 + 1:]]), plus_side),
                    np.interp(ys, u[:i0 + 1], surface[:i0 + 1]))


def test_the_finite_elements_reproduce_the_image_solution():
    """The 2-D reference below is itself checked: homogeneous, against images."""
    dip, depth, r = 60.0, 15.0, 2.0
    xs = np.r_[-np.arange(45.0, 2.0, -3.0), np.arange(3.0, 45.5, 3.0)]
    homogeneous, bimaterial = _antiplane(xs, dip, depth, r)
    one = _antiplane_fem(dip, depth * 1e3, [0.0], [3e10], xs * 1e3)
    two = _antiplane_fem(dip, depth * 1e3, [0.0], [3e10], xs * 1e3, c_minus=r)
    sign = np.sign(np.sum(one * homogeneous))
    assert np.abs(sign * one - homogeneous).max() < 5e-3 * np.abs(homogeneous).max()
    assert np.abs(sign * two - bimaterial).max() < 5e-3 * np.abs(bimaterial).max()


def test_a_one_layer_crust_is_the_half_space_exact_mode(setup):
    trace, frame, mesh, obs = setup
    one = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)
    tops, _, _ = wn.layer_stack(one)
    z = wn.depth_grid(tops, z_max=80e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    tables = wn.RemainderTables.compute(one, z=z, r=wn.distance_grid(per_decade=5), n_k=256,
                                        surface_z=z, surface_r=wn.distance_grid(per_decade=5))
    contrast = RigidityContrast(trace, side="north", ratio=2.0, frame=frame)
    reach = dict(depth=1.0, strike=1.0, cross=1.0)
    x, y, le, ln, lu = _points(obs)
    layered = Bimaterial(contrast, engine=LayeredTDE(tables, far=None), exact=True, extend=reach)
    half = Bimaterial(contrast, engine=HalfSpaceTDE(float(one.poisson())), exact=True,
                      extend=reach)
    g = layered.los_matrix(mesh, x, y, le, ln, lu)
    np.testing.assert_allclose(g, half.los_matrix(mesh, x, y, le, ln, lu), rtol=0,
                               atol=1e-10 * np.abs(g).max())


def test_the_layered_exact_field_meets_the_interface_conditions(layered):
    """As the half-space test above, with the field of the layered crust throughout."""
    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.6]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=10e3, edge_length=5e3)
    contrast = RigidityContrast(trace, side="left", ratio=2.0, frame=frame)
    solve = Bimaterial(contrast, engine=layered, exact=True,
                       extend=dict(depth=1.0, strike=1.0, cross=1.0)).interface(mesh)
    slip = np.concatenate([np.ones(mesh.n_elements), 0.3 * np.ones(mesh.n_elements)])
    d = solve.discontinuity(slip)
    for kind in ("fault", "extension"):
        picks = np.flatnonzero(solve.kind == kind)[::3]
        offset = 1e-4 * np.sqrt(solve.areas[picks])[:, None] * solve.normals[picks]
        above = solve.field(d, *(solve.centroids[picks] + offset).T)
        below = solve.field(d, *(solve.centroids[picks] - offset).T)
        jump = solve.a_plus[picks, None] * above - solve.a_minus[picks, None] * below
        local = np.einsum("eki,ei->ek", solve.frames[picks], jump)
        target = np.zeros_like(local)
        if kind == "fault":
            target[:, 0], target[:, 1] = 1.0, 0.3
        np.testing.assert_allclose(local, target, atol=2e-3)


def test_the_layered_exact_mode_reduces_to_the_2d_layered_solution(layered):
    """Mid-length of a long 60 degree fault in a layered crust, against finite elements.

    The finite elements share nothing with the 3-D path: no Green's function, no
    remainder tables, no triangular dislocation. Measured at 5 km elements on a
    298 km fault with the Venezuela crust: contrast effect 0.998 of the 2-D one,
    correlation 0.9992 (the half-space version of the same check: 0.995, 0.9967).
    This coarse version (7.5 km elements, two layers) measured 0.994 and 0.997.
    """
    dip, depth, r = 60.0, 15.0, 2.0
    trace = FaultTrace.from_lonlat(np.array([-82.0, -79.3]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=dip, max_depth=depth * 1e3,
                            edge_length=7.5e3)
    contrast = RigidityContrast(trace, side="left", ratio=r, frame=frame)
    xs = np.r_[-np.arange(45.0, 2.0, -3.0), np.arange(3.0, 45.5, 3.0)]
    px, py = trace.from_curvilinear(np.full(xs.size, trace.length(frame) / 2), xs * 1e3, frame)
    slip = np.zeros((mesh.n_elements, 2))
    slip[:, 0] = 1.0
    plain = layered.forward(mesh, slip, px, py)[:, 0]
    exact = Bimaterial(contrast, engine=layered, exact=True,
                       extend=dict(depth=3.0, strike=1.0, cross=1.0)).forward(mesh, slip, px, py)[:, 0]
    tops, mu = layered.tables.tops, layered.tables.mu
    one = _antiplane_fem(dip, depth * 1e3, tops, mu, xs * 1e3)
    two = _antiplane_fem(dip, depth * 1e3, tops, mu, xs * 1e3, c_plus=1.0 / r)
    sign = np.sign(np.sum(plain * one))
    assert np.abs(plain - sign * one).max() < 0.03 * np.abs(one).max()
    effect_3d, effect_2d = exact - plain, sign * (two - one)
    assert np.corrcoef(effect_3d, effect_2d)[0, 1] > 0.99
    assert 0.97 < np.linalg.norm(effect_3d) / np.linalg.norm(effect_2d) < 1.02


# -- a contrast that changes with depth ------------------------------------------------
def test_a_contrast_can_change_with_depth(setup):
    trace, frame, mesh, _ = setup
    c = RigidityContrast(trace, side="north", ratio=[[2.0, 1.0]], depth_breaks=(6e3,), frame=frame)
    assert (c.n_intervals, c.n_bands) == (1, 2)
    np.testing.assert_array_equal(c.band([0.0, 5999.0, 6000.0, 20e3]), [0, 0, 1, 1])
    np.testing.assert_array_equal(c.a("side", [0, 0], [0, 1]), [2.0, 1.0])
    back = RigidityContrast.from_dict(c.to_dict())
    np.testing.assert_array_equal(back.ratios, c.ratios)
    np.testing.assert_array_equal(back.depth_breaks, c.depth_breaks)
    # each element its own band: the moment's harmonic mean steps at the break
    f = c.element_factor(mesh)
    shallow = -np.asarray(mesh.centroids)[:, 2] < 6e3
    np.testing.assert_allclose(f[shallow], 2.0 / 3.0)          # r 2: 2 / (a_side + a_other) = 2 / 3
    np.testing.assert_allclose(f[~shallow], 1.0)
    assert "to 6 km" in repr(c)
    with pytest.raises(ValueError, match="bands"):
        RigidityContrast(trace, ratio=[2.0, 1.0, 1.5], depth_breaks=(6e3,), frame=frame)


def test_the_exact_field_meets_its_interface_conditions_at_a_depth_break():
    """The jump is the slip on the fault and nothing on any boundary, the horizontal one included."""
    trace = FaultTrace.from_lonlat(np.array([-80.0, -79.6]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=12e3, edge_length=4e3)
    levels = np.unique(np.round(-np.asarray(mesh.nodes)[:, 2]))
    cut = float(levels[(levels > 2e3) & (levels < 10e3)][0])    # a row of the mesh
    contrast = RigidityContrast(trace, side="left", ratio=[[2.0, 1.0]], depth_breaks=(cut,),
                                frame=frame)
    solve = Bimaterial(contrast, exact=True, extend=dict(depth=1.0, strike=1.0, cross=1.0)).interface(mesh)
    assert (solve.kind == "depth break").sum() > 0
    slip = np.concatenate([np.ones(mesh.n_elements), 0.3 * np.ones(mesh.n_elements)])
    d = solve.discontinuity(slip)
    for kind in ("fault", "extension", "depth break"):
        picks = np.flatnonzero(solve.kind == kind)[::4]
        offset = 1e-4 * np.sqrt(solve.areas[picks])[:, None] * solve.normals[picks]
        above = solve.field(d, *(solve.centroids[picks] + offset).T)
        below = solve.field(d, *(solve.centroids[picks] - offset).T)
        jump = solve.a_plus[picks, None] * above - solve.a_minus[picks, None] * below
        local = np.einsum("eki,ei->ek", solve.frames[picks], jump)
        target = np.zeros_like(local)
        if kind == "fault":
            target[:, 0], target[:, 1] = 1.0, 0.3
        np.testing.assert_allclose(local, target, atol=2e-3)


def test_a_shallow_contrast_reduces_to_the_2d_solution():
    """Mid-length of a long 60 degree fault, the hanging wall softer above 6 km only, against FEM.

    ⚠️ The horizontal boundary must reach past the data, as the across-strike ones must:
    cut 1 depth out (15 km, receivers to 45 km) it read correlation -0.41. At 4 depths, on a
    400 km fault: shallow-only 1.03 of the 2-D effect (corr 0.997), uniform 1.00, deep-only
    (below 7.5 km) 0.93 -- 0.83 on 165 km, so a deep contrast is the slower to reach 2-D.
    """
    dip, depth, r, cut = 60.0, 15.0, 2.0, 6e3
    trace = FaultTrace.from_lonlat(np.array([-82.0, -79.3]), np.array([8.0, 8.0]))
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=dip, max_depth=depth * 1e3, edge_length=5e3)
    levels = np.unique(np.round(-np.asarray(mesh.nodes)[:, 2]))
    cut = float(levels[np.argmin(np.abs(levels - cut))])
    contrast = RigidityContrast(trace, side="left", ratio=[[r, 1.0]], depth_breaks=(cut,), frame=frame)
    xs = np.r_[-np.arange(45.0, 2.0, -3.0), np.arange(3.0, 45.5, 3.0)]
    px, py = trace.from_curvilinear(np.full(xs.size, trace.length(frame) / 2), xs * 1e3, frame)
    slip = np.zeros((mesh.n_elements, 2))
    slip[:, 0] = 1.0
    plain = HalfSpaceTDE().forward(mesh, slip, px, py)[:, 0]
    exact = Bimaterial(contrast, exact=True, extend=dict(depth=3.0, strike=1.0, cross=4.0)
                       ).forward(mesh, slip, px, py)[:, 0]
    one = _antiplane_fem(dip, depth * 1e3, [0.0], [3e10], xs * 1e3)
    two = _antiplane_fem(dip, depth * 1e3, [0.0], [3e10], xs * 1e3,
                         c_plus=lambda z: np.where(z < cut, 1.0 / r, 1.0))
    sign = np.sign(np.sum(plain * one))
    effect_3d, effect_2d = exact - plain, sign * (two - one)
    assert np.corrcoef(effect_3d, effect_2d)[0, 1] > 0.98
    assert 0.9 < np.linalg.norm(effect_3d) / np.linalg.norm(effect_2d) < 1.1


# -- the sweep ---------------------------------------------------------------------
SWEEP = dict(segments=1, max_depth=15e3, edge_length=10e3, down_dip_levels=5, ramp="offset",
             polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0))


def test_the_sweep_refuses_a_bimaterial_engine(setup):
    trace, frame, _, obs = setup
    engine = Bimaterial(RigidityContrast(trace, ratio=2.0, frame=frame))
    with pytest.raises(ValueError, match="contrast="):
        DipSweep(trace, obs, frame=frame, dips=(90.0,), engine=engine, **SWEEP)


def test_a_unit_contrast_is_the_plain_sweep(setup):
    trace, frame, _, obs = setup
    plain = DipSweep(trace, obs, frame=frame, dips=(70.0, 90.0), **SWEEP)
    unit = DipSweep(trace, obs, frame=frame, dips=(70.0, 90.0), **SWEEP,
                    contrast=RigidityContrast(trace, ratio=1.0, frame=frame))
    np.testing.assert_allclose(unit.design((70.0,)), plain.design((70.0,)), rtol=1e-13, atol=0)
    assert unit.evidence((70.0,))["log_evidence"] == pytest.approx(
        plain.evidence((70.0,))["log_evidence"], rel=1e-12)
    assert unit._combination((70.0,)) == (70.0, 1.0)


def test_the_sweep_matches_a_scaled_inversion(setup):
    """With a contrast, the sweep's matrix is the one `SlipInversion` builds with `Bimaterial`."""
    trace, frame, _, obs = setup
    contrast = RigidityContrast(trace, side="north", ratio=(2.0, 1.25), breaks=(60e3,), frame=frame)
    sweep = DipSweep(trace, obs, frame=frame, dips=(90.0,), **SWEEP, contrast=contrast)
    direct = SlipInversion(sweep.reference, obs, engine=Bimaterial(contrast), basis="node").g
    np.testing.assert_allclose(sweep.design((90.0,)), direct, rtol=1e-10,
                               atol=1e-12 * np.abs(direct).max())


def test_a_planted_contrast_and_dip_are_recovered(setup):
    """Two-sided data separate the ratio from the dip, by the evidence and at a fixed rank."""
    trace, frame, _, obs = setup
    contrast = RigidityContrast(trace, side="north", ratio=1.0, frame=frame)
    sweep = DipSweep(trace, obs, frame=frame, dips=(70.0, 90.0, 110.0), **SWEEP,
                     contrast=contrast, ratios=(0.5, 1.0, 2.0, 4.0))
    planted = (110.0, 2.0)
    x = sweep.evaluate(planted, rank=25.0)["x"]
    data = sweep.predict(planted, x)
    result = sweep.run(score="evidence", data=data, method="exhaustive")
    assert result.best == planted
    assert len(result.table) == 12 and "ratio_1" in result.table
    profile = result.profiles()
    ratios = profile[profile.parameter == "ratio"]
    assert ratios.loc[ratios.score.idxmin(), "ratio"] == 2.0
    assert sweep.run(rank=25.0, data=data, method="exhaustive").best == planted
    assert sweep.label(planted) == "110 | r 2"


def test_the_bootstrap_gives_the_ratio_an_interval(setup):
    """Dip and ratio trade off even with two-sided data; with both grids wide enough it covers.

    Measured over 6 seeds (5 mm noise, 602 samples): ratio 2 recovered as
    1.94-2.06, sd 0.054-0.067, every interval covering, |z| under 1. With the dip
    grid only 88-92 a dip pinned on its edge made the ratio's interval 2-3x too
    narrow and it missed twice, which is what the edge warning is for.
    """
    trace, frame, _, obs = setup
    contrast = RigidityContrast(trace, side="north", ratio=1.0, frame=frame)
    sweep = DipSweep(trace, obs, frame=frame, dips=(86.0, 88.0, 90.0, 92.0, 94.0), **SWEEP,
                     contrast=contrast, ratios=(1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15))
    x = sweep.evaluate((90.0, 2.0), rank=25.0)["x"]
    signal = sweep.predict((90.0, 2.0), x)
    data = signal + 5e-3 * np.random.default_rng(0).standard_normal(signal.size)
    result = sweep.run(score="evidence", data=data, method="exhaustive")
    boot = sweep.bootstrap(result, draws=20, seed=0, compare=False)
    table = boot.intervals()
    row = table[table.parameter == "ratio"].iloc[0]
    assert row.interval == 1 and np.isnan(row.segment)
    assert row.normal_lo < 2.0 < row.normal_hi
    assert 0.02 < row.sd < 0.15
    assert table[table.parameter == "dip"].iloc[0].segment == 1
    narrow = DipSweep(trace, obs, frame=frame, dips=(88.0, 90.0, 92.0), **SWEEP,
                      contrast=contrast, ratios=(1.2, 1.4, 1.6))
    pinned = narrow.run(score="evidence", data=data, method="exhaustive")
    assert pinned.best[1] == 1.6
    with pytest.warns(UserWarning, match="edge of its grid"):
        narrow.bootstrap(pinned, draws=3, seed=0, compare=False)


def test_the_exact_interface_assembles_the_same_in_worker_processes(setup, tmp_path):
    """``InterfaceSolve(workers=2)``: Fbar and the response matrix equal the serial ones (layered kernel, depth break)."""
    from slipsolve.bimaterial import InterfaceSolve

    trace, frame, mesh, obs = setup
    one = VelocityModel.uniform(vp=6000.0, vs=3464.1016, rho=2700.0)
    tops, _, _ = wn.layer_stack(one)
    z = wn.depth_grid(tops, z_max=80e3, first=50.0, growth=2.0, spacing=2000.0, deep_spacing=8000.0)
    tables = wn.RemainderTables.compute(one, z=z, r=wn.distance_grid(per_decade=5), n_k=256,
                                        surface_z=z, surface_r=wn.distance_grid(per_decade=5))
    kernel = LayeredTDE(wn.RemainderTables.load(tables.save(tmp_path / "one.npz")), far=None)
    contrast = RigidityContrast(trace, side="north", ratio=[[2.0, 1.2]], frame=frame, depth_breaks=(6e3,))
    reach = dict(depth=1.0, strike=1.0, cross=1.0)
    serial = InterfaceSolve(mesh, contrast, kernel=kernel, extend=reach)
    pooled = InterfaceSolve(mesh, contrast, kernel=kernel, extend=reach, workers=2)
    np.testing.assert_allclose(pooled.fbar, serial.fbar, rtol=0, atol=1e-12 * np.abs(serial.fbar).max())
    x, y, le, ln, lu = _points(obs)
    a = serial.response_matrix(x[:40], y[:40], look=(le[:40], ln[:40], lu[:40]))
    b = pooled.response_matrix(x[:40], y[:40], look=(le[:40], ln[:40], lu[:40]))
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * np.abs(a).max())
    assert pooled.with_contrast(contrast.with_ratio([[1.5, 1.5]])).workers == 2
