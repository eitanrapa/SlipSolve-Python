"""`slipsolve.psgrn`: PSGRN's input, its tables, and their integration over a slip model.

The integration is checked against `slipsolve.layered.LayeredPointSource` on the same synthetic tables, with no
PSGRN. The tests that run PSGRN need an executable (`$SLIPSOLVE_PSGRN`, or `psgrn2020` / `psgrn` on `PATH`) and are
skipped without one.
"""

import types

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace
from slipsolve.basis import make_basis
from slipsolve.edgrn import EdgrnTables, VelocityModel
from slipsolve.layered import LayeredPointSource
from slipsolve.psgrn import (COMPONENTS, PsgrnTables, Rheology, _interfaces, _read_table, find_psgrn, psgrn_layers,
                             relaxation_history, relaxation_pattern, run_psgrn, write_psgrn_input)


def crust():
    """Two elastic layers (0-16, 16-30 km) over a half-space from 30 km."""
    return VelocityModel([0.0, 16e3, 16e3, 30e3, 30e3], [6000.0, 6000.0, 6700.0, 6700.0, 8000.0],
                         [3460.0, 3460.0, 3870.0, 3870.0, 4620.0], [2600.0, 2600.0, 2800.0, 2800.0, 3400.0])


def small_fault(length=30e3, depth=15e3, edge=5e3):
    trace = FaultTrace(np.array([-118.17, -118.0]), np.array([35.7, 35.7]), name="test")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=depth, edge_length=edge)
    basis = make_basis(mesh, "node")
    nodes = mesh.nodes
    along = (nodes[:, 0] - nodes[:, 0].min()) / np.ptp(nodes[:, 0])
    taper = np.sin(np.pi * along) * np.exp(-((nodes[:, 2] + 6e3) / 6e3) ** 2)
    model = types.SimpleNamespace(mesh=mesh, basis=basis, strike_slip=-1.0 * taper, dip_slip=0.3 * taper)
    return model, frame


def test_layers_from_a_velocity_model_and_rheologies():
    rows = psgrn_layers(crust(), [Rheology(30e3, None, eta_maxwell=1e19)])
    assert len(rows) == 5
    assert rows[0][:4] == (0.0, 6.0, 3.46, 2600.0)
    assert rows[-1] == (30.0, 8.0, 4.62, 3400.0, 0.0, 1e19, 1.0)
    # a rheology boundary inside a layer is an interface where only the viscosity changes
    rows = psgrn_layers(crust(), [Rheology(20e3, 30e3, eta_kelvin=1e18, alpha=0.5)])
    depths = [row[0] for row in rows]
    assert depths.count(20.0) == 2
    assert rows[depths.index(20.0) + 1][4:] == (1e18, 0.0, 0.5)
    assert rows[-1][4:] == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="overlap"):
        psgrn_layers(crust(), [Rheology(10e3, 30e3, eta_maxwell=1e19), Rheology(20e3, None, eta_maxwell=1e19)])


def test_input_file_depth_edges_and_interfaces(tmp_path):
    path = write_psgrn_input(tmp_path / "psgrn.inp", crust(), [Rheology(20e3, 30e3, eta_kelvin=1e18, alpha=0.5)],
                             z_min=500.0, z_max=29500.0, nz=30)
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    # PSGRN2020 takes the two depths as cell edges: 0 and 30 km give sources at 0.5, 1.5, ..., 29.5 km
    assert [float(v) for v in lines[2].split()[1:]] == [0.0, 30.0]
    np.testing.assert_allclose(_interfaces(text), [16e3, 20e3, 30e3])


def _fake_table(path, r, z, window_s, co, post):
    """A table file in PSGRN's layout: header lines, then per depth its layer line, the coseismic floats, one decimal
    exponent per distance and `nt - 1` lines of integers."""
    nt = post.shape[0]
    lines = ["# fake", f"{r.size:7d}{r[0]:14.6E}{r[-1]:14.6E}{1.0:10.4f}", "# site", " ".join(["1.0"] * 7),
             f"{z.size:7d}{z[0]:14.6E}{z[-1]:14.6E}", f"{nt:7d}{window_s:24.16E}"]
    for j in range(z.size):
        lines.append(f"{z[j]:13.6E} " + " ".join(["1.0"] * 6))
        lines.append(" ".join(f"{v:14.6E}" for v in co[:, j]))
        am = np.abs(post[1:, :, j]).max(axis=0)
        idec = np.where(am > 0, np.floor(np.log10(np.where(am > 0, am, 1.0))).astype(int) - 5, 0)
        lines.append(" ".join(str(int(v)) for v in idec))
        for k in range(1, nt):
            lines.append(" ".join(str(int(round(v))) for v in post[k, :, j] * 10.0 ** (-idec)))
    path.write_text("\n".join(lines) + "\n")


def test_table_reader(tmp_path):
    rng = np.random.default_rng(3)
    r = np.linspace(0.0, 4e3, 5)
    z = np.array([1e3, 2e3, 3e3])
    co = rng.normal(size=(5, 3)) * 1e-9
    post = np.concatenate([np.zeros((1, 5, 3)), rng.normal(size=(3, 5, 3)) * 1e-11])
    _fake_table(tmp_path / "uz.ss", r, z, 3 * 86400.0, co, post)
    rr, zz, t, c, p = _read_table(tmp_path / "uz.ss")
    np.testing.assert_allclose(rr, r)
    np.testing.assert_allclose(zz, z)
    np.testing.assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(c, co, rtol=1e-6)
    np.testing.assert_allclose(p, post, rtol=1e-5, atol=1e-16)


def _synthetic_tables(functions, shapes, r, z, times):
    post = {kind: {comp: np.einsum("tm,mrz->trz", functions, shapes[(kind, comp)]) for comp in comps}
            for kind, comps in COMPONENTS.items()}
    co = {kind: {comp: np.zeros((r.size, z.size)) for comp in comps} for kind, comps in COMPONENTS.items()}
    return PsgrnTables(r, z, times, co, post)


def test_modes_recover_a_separable_history():
    rng = np.random.default_rng(1)
    r, z = np.linspace(0.0, 10e3, 6), np.linspace(1e3, 4e3, 4)
    times = np.linspace(0.0, 100.0, 9)
    functions = np.column_stack([1 - np.exp(-times / 20.0), times / 100.0])
    shapes = {(k, c): rng.normal(size=(2, r.size, z.size)) for k, comps in COMPONENTS.items() for c in comps}
    tables = _synthetic_tables(functions, shapes, r, z, times)
    t, f, stacks = tables.modes(tolerance=1e-10)
    assert f.shape == (times.size, 2)
    assert np.abs(f[0]).max() < 1e-12
    for (kind, comp), shape in shapes.items():
        rebuilt = np.einsum("tm,rzm->trz", f, stacks[kind][comp])
        np.testing.assert_allclose(rebuilt, tables.postseismic[kind][comp], atol=1e-10)


def test_integration_matches_the_layered_engine():
    """The same tables through our summation and through `LayeredPointSource`: azimuthal factors, signs, tents."""
    r, z = np.linspace(0.0, 120e3, 121), np.linspace(250.0, 20250.0, 41)
    static = EdgrnTables.homogeneous(nu=0.25, r=r, z=z)
    co = {kind: {comp: static.tables[kind][comp] for comp in comps} for kind, comps in COMPONENTS.items()}
    post = {kind: {comp: np.stack([np.zeros_like(static.tables[kind][comp]), static.tables[kind][comp]])
                   for comp in comps} for kind, comps in COMPONENTS.items()}
    tables = PsgrnTables(r, z, np.array([0.0, 1.0]), co, post)
    model, frame = small_fault()
    x = np.array([-20e3, 0.0, 15e3, 40e3, 10e3])
    y = np.array([12e3, -25e3, 30e3, -10e3, 50e3])
    _, history = relaxation_history(model, tables, x, y, order=10)
    reference = LayeredPointSource(static, tolerance=1e-4).forward(
        model.mesh, np.r_[model.strike_slip, model.dip_slip], x, y, basis=model.basis)
    assert np.abs(history[1] - reference).max() < 2e-3 * np.abs(reference).max()


def _psgrn():
    try:
        return find_psgrn()
    except RuntimeError:
        pytest.skip("no PSGRN executable")


GRID = dict(r_max=100e3, nr=101, z_min=1e3, z_max=11e3, nz=6, accuracy=0.025, gravity=0.0)


def test_psgrn_homogeneous_half_space_has_no_remainder(tmp_path):
    _psgrn()
    uniform = VelocityModel.uniform(max_depth=60e3)
    tables = run_psgrn(uniform, (), tmp_path, nt=1, window_days=1.0, **GRID)
    np.testing.assert_allclose(tables.z, np.linspace(1e3, 11e3, 6), atol=1e-3)
    biggest = max(float(np.abs(tables.coseismic[k][c]).max()) for k, comps in COMPONENTS.items() for c in comps)
    assert biggest < 1e-15


def test_psgrn_relaxed_state_matches_our_layered_code(tmp_path):
    """A Standard Linear Solid half-space fully relaxed (nt = 2), against `slipsolve.wavenumber`'s whole field with
    the relaxed modulus less the unrelaxed one: an independent layered code, per source type to 5e-3 of its largest
    component. (PSGRN's own elastic path is the weaker reference: for the CLVD's vertical from a 1 km source it reads
    2.4% low where its time path and this code agree to 0.1%.)"""
    from slipsolve import wavenumber as wn

    _psgrn()
    alpha = 0.5
    sls = run_psgrn(crust(), [Rheology(30e3, None, eta_kelvin=1e19, alpha=alpha)], tmp_path, nt=2,
                    window_days=1.0, **GRID)
    vp, vs, rho = 8000.0, 4620.0, 3400.0
    mu = rho * vs ** 2
    kappa = rho * vp ** 2 - 4.0 / 3.0 * mu
    relaxed = VelocityModel([0.0, 16e3, 16e3, 30e3, 30e3],
                            [6000.0, 6000.0, 6700.0, 6700.0, np.sqrt((kappa + 4.0 / 3.0 * alpha * mu) / rho)],
                            [3460.0, 3460.0, 3870.0, 3870.0, np.sqrt(alpha * mu / rho)],
                            [2600.0, 2600.0, 2800.0, 2800.0, rho])
    k, dln = wn.wavenumbers()

    def whole(model, zs, r):
        tops, lam, mu_ = wn.layer_stack(model)
        j = int(wn._layer_index(tops, zs))
        psv, sh = wn.source_jumps(lam[j], mu_[j])
        return wn.transform(*wn._Solution(k, tops, lam, mu_, zs, psv, sh).at(np.array([0.0])), k, dln, r)[0]

    r = sls.r[1:]                                  # r = 0 is exact zeros in PSGRN and FFTLog's first node in ours
    keys = [(kind, comp) for kind, comps in COMPONENTS.items() for comp in comps]
    for j, zs in enumerate(sls.z):
        ours = whole(relaxed, zs, r) - whole(crust(), zs, r)       # TABLE_KEYS order; the first eight are these keys
        for kind in COMPONENTS:
            cols = [i for i, key in enumerate(keys) if key[0] == kind]
            scale = np.abs(ours[:, cols]).max()
            for i in cols:
                comp = keys[i][1]
                theirs = sls.postseismic[kind][comp][1][1:, j]
                assert np.abs(theirs - ours[:, i]).max() < 5e-3 * scale, (zs, kind, comp)


def test_pattern_reproduces_the_history(tmp_path):
    _psgrn()
    grid = dict(GRID, z_min=500.0, z_max=15500.0, nz=16)
    tables = run_psgrn(crust(), [Rheology(30e3, None, eta_maxwell=1e19)], tmp_path, nt=64, window_days=20 * 365.25,
                       **grid)
    model, frame = small_fault()
    x = np.array([-20e3, 0.0, 15e3, 40e3])
    y = np.array([12e3, -25e3, 30e3, -10e3])
    look = (np.full(4, 0.6), np.full(4, -0.1), np.full(4, 0.79))
    times, history = relaxation_history(model, tables, x, y, order=4)
    los = history[..., 0] * look[0] + history[..., 1] * look[1] + history[..., 2] * look[2]
    pattern = relaxation_pattern(model, tables, tolerance=1e-5, order=4)
    _, functions, _ = tables.modes(tolerance=1e-5)
    rebuilt = functions @ pattern.values(x, y, *look).T
    assert np.abs(rebuilt - los).max() < 1e-4 * np.abs(los).max()
    assert pattern.frame is model.mesh.frame
