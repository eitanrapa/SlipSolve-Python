"""`slipsolve.relaxation`: our own viscoelastic tables, by the correspondence principle and Talbot's method.

Oracles: Talbot's method on transforms whose inverses are known; the fully relaxed state of a Standard Linear Solid,
which our elastic code computes directly, with and without gravity; Savage & Prescott's (1978) solution for a long
strike-slip fault in an elastic layer over a Maxwell half-space; the exact scaling of time with viscosity; and, for
gravity, the layer equations themselves and the elastic limit as `g` vanishes.
"""

import types

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace, HalfSpaceTDE
from slipsolve import wavenumber as wn
from slipsolve.basis import make_basis
from slipsolve.edgrn import VelocityModel
from slipsolve.inversion import assemble_design
from slipsolve.relaxation import (COMPONENTS, GRAVITY, RELAXATION_K_RANGE, RELAXATION_N_K, SECONDS_PER_DAY,
                                  RelaxationTables, Rheology, _layers, _talbot, relaxation_history, relaxation_pattern,
                                  shear_modulus)

DAYS_PER_YEAR = 365.25


def crust():
    """Two elastic layers (0-16, 16-30 km) over a half-space from 30 km."""
    return VelocityModel([0.0, 16e3, 16e3, 30e3, 30e3], [6000.0, 6000.0, 6700.0, 6700.0, 8000.0],
                         [3460.0, 3460.0, 3870.0, 3870.0, 4620.0], [2600.0, 2600.0, 2800.0, 2800.0, 3400.0])


GRID = dict(r_max=60e3, nr=61, z_min=2000.0, z_max=14000.0, nz=4)


def test_talbot_inverts_known_transforms():
    for a in (1e-9, 1e-7, 1e-5):
        for t in (1e3, 1e5, 1e7):
            nodes, weights = _talbot(t)
            step = np.real(np.sum(weights / (nodes * (nodes + a))))
            assert step == pytest.approx((1.0 - np.exp(-a * t)) / a, rel=1e-9)
            decay = np.real(np.sum(weights / (nodes + a)))
            assert abs(decay - np.exp(-a * t)) < 1e-9


def test_shear_modulus_limits():
    mu = 3e10
    big, small = np.array([1e6]), np.array([1e-20])
    maxwell = Rheology(0.0, None, eta_maxwell=1e19)
    sls = Rheology(0.0, None, eta_kelvin=1e19, alpha=0.4)
    burgers = Rheology(0.0, None, eta_maxwell=1e20, eta_kelvin=1e18, alpha=0.4)
    assert shear_modulus(None, mu, big)[0] == pytest.approx(mu)
    for flow in (maxwell, sls, burgers):
        assert shear_modulus(flow, mu, big)[0].real == pytest.approx(mu, rel=1e-6), "unrelaxed"
    assert abs(shear_modulus(maxwell, mu, small)[0]) < 1e-3 * mu
    assert shear_modulus(sls, mu, small)[0].real == pytest.approx(0.4 * mu, rel=1e-6), "relaxed to alpha mu"
    assert abs(shear_modulus(burgers, mu, small)[0]) < 1e-3 * mu
    # A Maxwell body's stress relaxes with eta / mu: its modulus has the pole there.
    s = -mu / 1e19
    assert abs(1.0 / shear_modulus(maxwell, mu, np.array([s * (1 + 1e-9)]))[0]) < 1e-6 / mu


@pytest.mark.parametrize("gravity", [0.0, GRAVITY])
def test_relaxed_standard_linear_solid_is_the_relaxed_elastic_field(gravity):
    """Long after its relaxation time a Standard Linear Solid is an elastic body with the modulus alpha mu."""
    alpha, eta = 0.5, 1e19
    late = 1e4 * eta * (1 - alpha) / (3400.0 * 4620.0 ** 2) / SECONDS_PER_DAY        # 1e4 relaxation times, days
    tables = RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_kelvin=eta, alpha=alpha)], times=[late],
                                      gravity=gravity, **GRID)
    vp, vs, rho = 8000.0, 4620.0, 3400.0
    mu = rho * vs ** 2
    kappa = rho * vp ** 2 - 4.0 / 3.0 * mu
    relaxed = VelocityModel([0.0, 16e3, 16e3, 30e3, 30e3],
                            [6000.0, 6000.0, 6700.0, 6700.0, np.sqrt((kappa + 4.0 / 3.0 * alpha * mu) / rho)],
                            [3460.0, 3460.0, 3870.0, 3870.0, np.sqrt(alpha * mu / rho)],
                            [2600.0, 2600.0, 2800.0, 2800.0, rho])
    k, dln = wn.wavenumbers(RELAXATION_K_RANGE, RELAXATION_N_K)

    def whole(model, zs, r):
        tops, lam, mu_, rho_, _ = _layers(model)
        j = int(wn._layer_index(tops, zs))
        psv, sh = wn.source_jumps(lam[j], mu_[j])
        buoyancy = rho_ * gravity if gravity else None
        return wn.transform(*wn._Solution(k, tops, lam, mu_, zs, psv, sh, buoyancy=buoyancy).at(np.array([0.0])),
                            k, dln, r)[0]

    keys = [(kind, comp) for kind, comps in COMPONENTS.items() for comp in comps]
    r = tables.r[1:]
    for j, zs in enumerate(tables.z):
        expected = whole(relaxed, zs, r) - whole(crust(), zs, r)
        for i, (kind, comp) in enumerate(keys):
            ours = tables.postseismic[kind][comp][1][1:, j]
            assert np.abs(ours - expected[:, i]).max() < 1e-7 * np.abs(expected[:, :8]).max(), (zs, kind, comp)


def savage_prescott(x, t_years, layer, slip=1.0, depth=None, eta=None, mu=None, terms=200):
    """Savage & Prescott (1978): a long strike-slip fault in an elastic layer over a Maxwell half-space.

    The fault slips `slip` from the surface to `depth` (default the whole `layer`) at `t = 0`. Fault-parallel
    displacement at a distance `x` from the trace, `(n_times, n_x)`:
    `u = (s / pi) [atan(D / x) + sum_n G_n(t) (atan((2 n H + D) / x) - atan((2 n H - D) / x))]` with
    `G_n(t) = 1 - exp(-t / tau) sum_{m<n} (t / tau)^m / m!` and `tau = 2 eta / mu`.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t_years, dtype=float))
    depth = float(layer if depth is None else depth)
    tau = 2.0 * float(eta) / float(mu) / (DAYS_PER_YEAR * SECONDS_PER_DAY)
    ratio = t / tau
    out = np.tile(np.arctan2(depth, x), (t.size, 1))
    total = np.zeros(t.size)
    term = np.exp(-ratio)
    for n in range(1, int(terms) + 1):
        total = total + term
        term = term * ratio / n
        out = out + (1.0 - total)[:, None] * (np.arctan2(2 * n * layer + depth, x) - np.arctan2(2 * n * layer - depth, x))
    return float(slip) / np.pi * out


def test_savage_prescott_long_fault():
    """A 4800 km vertical fault through a 20 km elastic layer over a Maxwell half-space, read at its middle.

    The finite length is what is left: at 80 km it is 1.1% at 2400 km and 0.3% at 4800 km; near the fault 0.1-0.2%.
    """
    vp, vs, rho, layer, eta = 6000.0, 3464.0, 2700.0, 20e3, 1e19
    mu = rho * vs ** 2
    uniform = VelocityModel([0.0, layer, layer], [vp, vp, vp], [vs, vs, vs], [rho, rho, rho])
    tau_days = 2.0 * eta / mu / SECONDS_PER_DAY
    ratios = np.array([0.5, 1.0, 2.0])
    tables = RelaxationTables.compute(uniform, [Rheology(layer, None, eta_maxwell=eta)], times=ratios * tau_days,
                                      r_max=2500e3, nr=2501, z_min=250.0, z_max=19750.0, nz=40, gravity=0.0)
    half_length = 2400e3 / 111.2e3
    trace = FaultTrace(np.array([30.0, 30.0]), np.array([-half_length, half_length]), name="long")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=layer, edge_length=10e3)
    basis = make_basis(mesh, "element")
    model = types.SimpleNamespace(mesh=mesh, basis=basis, strike_slip=np.ones(mesh.n_elements),
                                  dip_slip=np.zeros(mesh.n_elements))
    x0, y0 = frame.to_local(np.array([30.0]), np.array([0.0]))
    distance = np.array([5e3, 10e3, 20e3, 40e3, 80e3])
    x, y = x0[0] + np.r_[distance, -distance], np.full(2 * distance.size, y0[0])
    _, history = relaxation_history(model, tables, x, y, order=4)
    # The coseismic field from the same mesh fixes the sign on each side; the relaxation follows it.
    looks = (np.zeros(x.size), np.ones(x.size), np.zeros(x.size))
    coseismic = assemble_design(HalfSpaceTDE(), mesh, basis, (x, y, *looks)) @ np.r_[model.strike_slip, model.dip_slip]
    post = savage_prescott(distance, ratios * tau_days / DAYS_PER_YEAR, layer, eta=eta, mu=mu) - \
        savage_prescott(distance, [0.0], layer, eta=eta, mu=mu)
    sign = np.sign(coseismic)
    north = history[1:, :, 1] * sign[None, :]
    for side in (slice(0, distance.size), slice(distance.size, None)):
        assert np.abs(north[:, side] - post).max() < 0.005 * post.max(), (north[:, side], post)


def test_time_scales_with_viscosity():
    times = np.array([30.0, 300.0, 3000.0])
    one = RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_maxwell=1e19)], times=times, **GRID)
    two = RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_maxwell=2e19)], times=2.0 * times, **GRID)
    for kind, comps in COMPONENTS.items():
        for comp in comps:
            a, b = one.postseismic[kind][comp], two.postseismic[kind][comp]
            assert np.abs(a - b).max() < 1e-8 * np.abs(a).max()
    assert np.allclose(one.rescaled(2.0).times, two.times)


def test_pattern_rebuilds_the_history():
    tables = RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_maxwell=1e19)], window_days=3650.0, nt=10,
                                      r_max=80e3, nr=41, z_min=500.0, z_max=15500.0, nz=6)
    assert tables.times[0] == 0.0 and np.all(np.diff(tables.times) > 0)
    trace = FaultTrace(np.array([-118.17, -118.0]), np.array([35.7, 35.7]), name="test")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=5e3)
    basis = make_basis(mesh, "node")
    nodes = mesh.nodes
    along = (nodes[:, 0] - nodes[:, 0].min()) / np.ptp(nodes[:, 0])
    taper = np.sin(np.pi * along) * np.exp(-((nodes[:, 2] + 6e3) / 6e3) ** 2)
    model = types.SimpleNamespace(mesh=mesh, basis=basis, strike_slip=-1.0 * taper, dip_slip=0.3 * taper)
    x, y = np.array([-20e3, 0.0, 15e3, 40e3]), np.array([12e3, -25e3, 30e3, -10e3])
    look = (np.full(4, 0.6), np.full(4, -0.1), np.full(4, 0.79))
    _, history = relaxation_history(model, tables, x, y)
    los = history[..., 0] * look[0] + history[..., 1] * look[1] + history[..., 2] * look[2]
    pattern = relaxation_pattern(model, tables, tolerance=1e-6)
    _, functions, _ = tables.modes(tolerance=1e-6)
    assert np.abs(functions @ pattern.values(x, y, *look).T - los).max() < 1e-5 * np.abs(los).max()


def test_sources_must_be_elastic_and_off_interfaces():
    with pytest.raises(ValueError, match="viscous layer"):
        RelaxationTables.compute(crust(), [Rheology(10e3, None, eta_maxwell=1e19)], times=[1.0], **GRID)
    with pytest.raises(ValueError, match="interface"):
        RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_maxwell=1e19)], times=[1.0],
                                 r_max=10e3, nr=11, z_min=8e3, z_max=16e3, nz=2)


def _layer_system(k, lam, mu, buoyancy):
    """The P-SV layer equations with buoyancy, `Y' = A Y` for `Y = (U, V, R, S)`, written out here."""
    alpha, c, kappa = lam + 2 * mu, lam + mu, lam + 2 * mu / 3
    A = np.zeros((4, 4), dtype=complex)
    A[0, 1], A[0, 2] = lam * k / alpha, 1 / alpha
    A[1, 0], A[1, 3] = -k, 1 / mu
    A[2, 0], A[2, 1], A[2, 3] = buoyancy ** 2 / kappa, -buoyancy * k, k
    A[3, 0], A[3, 1], A[3, 2] = -buoyancy * k, 4 * k * k * mu * c / alpha, -k * lam / alpha
    return A


@pytest.mark.parametrize("lam, mu", [(3e10, 3e10), (3e10 + 2e10j, 1e8 - 3e8j), (5e10 - 1e7j, 1e3 + 1e3j)])
def test_gravity_layer_solutions_satisfy_the_layer_equations(lam, mu):
    """Every column, growing and decaying, elastic and nearly relaxed, against fourth-order finite differences."""
    buoyancy = 3300.0 * GRAVITY
    for k in (1e-9, 3e-7, 1e-6, 1e-4, 1e-2):
        A = _layer_system(k, lam, mu, buoyancy)
        for z0 in (0.3 / k, 2.0 / k):
            def columns(z):
                e = wn._psv_columns_gravity(np.array([k]), np.array([z]), np.array([z - 5.0 / k]), lam, mu, buoyancy,
                                            False)[0]
                e[2:4] *= k * wn.MU_SCALE                                    # undo the traction rows' scaling
                return e
            h = 1e-2 / k
            slope = (-columns(z0 + 2 * h) + 8 * columns(z0 + h) - 8 * columns(z0 - h) + columns(z0 - 2 * h)) / (12 * h)
            expected = A @ columns(z0)
            assert np.abs(slope - expected).max() < 1e-7 * (np.abs(expected).max() + np.abs(slope).max()), (k, z0)


def test_gravity_vanishes_into_the_elastic_field():
    """Without gravity the propagator is the elastic one; with it the coseismic field moves little, linearly in `g`."""
    tops, lam, mu, rho, _ = _layers(crust())
    k, dln = wn.wavenumbers(RELAXATION_K_RANGE, 1024)
    r = np.array([1e3, 5e3, 20e3, 50e3, 100e3])
    psv, sh = wn.source_jumps(lam[0], mu[0])

    def field(g):
        buoyancy = rho * g if g else None
        return wn.transform(*(a[0] for a in wn._Solution(k, tops, lam, mu, 8e3, psv, sh, buoyancy=buoyancy)
                              .at(np.array([0.0]))), k, dln, r)

    plain = field(0.0)
    small, full = field(1e-3) - plain, field(GRAVITY) - plain
    assert np.abs(full).max() < 5e-3 * np.abs(plain).max()
    scaled = full * 1e-3 / GRAVITY
    assert np.abs(small - scaled).max() < 1e-2 * np.abs(small).max()


def test_gravity_resists_dip_slip_relaxation():
    """Buoyancy holds back the vertical relaxation of a dip-slip source and barely touches a strike-slip one."""
    rheo = [Rheology(30e3, None, eta_maxwell=1e19)]
    late = [10 * DAYS_PER_YEAR]
    on = RelaxationTables.compute(crust(), rheo, times=late, **GRID)
    off = RelaxationTables.compute(crust(), rheo, times=late, gravity=0.0, **GRID)
    uplift_on, uplift_off = (abs(t.postseismic["ds"]["uz"][1][-1, -1]) for t in (on, off))     # 60 km away
    assert uplift_on < 0.97 * uplift_off
    strike_on, strike_off = (t.postseismic["ss"]["ur"][1] for t in (on, off))
    assert np.abs(strike_on - strike_off).max() < 0.02 * np.abs(strike_off).max()


def test_the_axis_is_exact():
    """At `r = 0` a strike-slip source's displacement vanishes, and the other columns join their neighbours."""
    tables = RelaxationTables.compute(crust(), [Rheology(30e3, None, eta_maxwell=1e19)], times=[10 * DAYS_PER_YEAR],
                                      r_max=6e3, nr=61, z_min=2000.0, z_max=14000.0, nz=4)
    for table in (tables.coseismic, {kind: {c: v[1] for c, v in comps.items()} for kind, comps in tables.postseismic.items()}):
        for comp in ("uz", "ur", "ut"):
            assert np.all(table["ss"][comp][0] == 0.0)
        for kind, comp in (("ds", "ur"), ("ds", "ut"), ("cl", "uz")):
            values = table[kind][comp]
            step = np.abs(values[1] - values[2]).max()                       # the change over 100 m
            assert np.abs(values[0] - values[1]).max() < 2.0 * step + 1e-6 * np.abs(values).max(), (kind, comp)
