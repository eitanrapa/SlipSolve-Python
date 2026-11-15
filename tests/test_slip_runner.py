"""Pins the production runner end to end on a small planted fault.

Two phase rasters and a few GNSS stations from a known slip patch go through every stage:
start meshes, re-meshing, resampling from the model, the cross-validated rank, the final
mesh, its check and the report. The moment must come back, a second run must resume from
the checkpoints without solving again, and the outputs must be complete.
"""

import json

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace, Observations, SlipModel, resample_all
from slipsolve.greens import HalfSpaceTDE
from slipsolve.production import STAGES, DataSeries, Production, Recipe, Sheet

import slip_synthetic as ss

_LON = np.array([-68.9, -68.65, -68.4, -68.15])
_LAT = np.array([10.50, 10.52, 10.53, 10.55])


def _mw(m0):
    return (2.0 / 3.0) * (np.log10(m0) - 9.1)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="planted")
    frame = trace.local_frame()
    truth = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=3e3)
    slip = ss.tapered_slip(truth, peak=-2.0, along_width=25e3, locking_depth=10e3)
    stacks = {g: ss.forward_los_stack(truth, slip, trace, frame, geometry=g, spacing=2500.0, half_x=70e3,
                                      half_y=40e3, noise=0.002, seed=k)
              for k, g in enumerate(("asc", "desc"))}
    grids = resample_all(stacks, frame, spacing=2000.0, bounds=(-60e3, 60e3, -32e3, 32e3))
    rng = np.random.default_rng(3)
    px, py = rng.uniform(-50e3, 50e3, 8), rng.uniform(-30e3, 30e3, 8)
    px, py = px[np.abs(py) > 3e3], py[np.abs(py) > 3e3]
    enu = HalfSpaceTDE(0.25).forward(truth, slip, px, py)
    lon, lat = frame.to_lonlat(px, py)
    look = np.tile(np.eye(3)[:2], (px.size, 1))
    gnss = Observations.from_points(np.repeat(lon, 2), np.repeat(lat, 2), enu[:, :2].ravel(), look,
                                    np.full(2 * px.size, 0.002), name="gnss", frame=frame)
    area = np.array([np.linalg.norm(np.cross(*np.diff(truth.nodes[t], axis=0))) / 2 for t in truth.triangles])
    m0 = 30e9 * np.sum(area * np.hypot(slip[:truth.n_elements], slip[truth.n_elements:]))
    sheets = [Sheet("main", trace, frame, dip=90.0, depth=15e3)]
    series = [DataSeries(g, grids[g], kind="phase", ramp="linear", near=2e3) for g in ("asc", "desc")]
    recipe = Recipe(rank0=15.0, start_edge=6e3, size_min=4e3, size_max=12e3, maps_b=2, maps_d=2, rounds=2,
                    field_spacing=4e3, width_min=3e3, width_max=30e3, folds=6, shifts=2, cv_ranks=(3.0, 80.0),
                    min_region_km2=50.0, sigma_far=15e3,
                    round0={"phase": dict(rms_min=0.004, sigma=0.003), "offset": dict(rms_min=0.2, sigma=None)},
                    solve=dict(boundary_ratio=1.0, polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0)))
    out = tmp_path_factory.mktemp("production") / "run"
    run = Production(out, sheets, series, points=[gnss], recipe=recipe, sees={"gnss": [0]}, echo=False)
    model = run.run()
    return run, model, _mw(m0), out


def test_every_stage_runs_and_writes_its_outputs(planted):
    run, model, _, out = planted
    assert isinstance(model, SlipModel)
    for name in ("A.pkl", "B.pkl", "C0.pkl", "C.pkl", "final.pkl", "final.slip.zip", "model.slip.zip",
                 "summary.json", "converge_B.csv", "converge_D.csv", "sampling_C0.csv", "cv_C.csv", "cv_E.csv"):
        assert (out / name).exists(), name
    record = json.loads((out / "stages.json").read_text())
    assert set(record) == {"A", "B", "C0", "C", "D", "E"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rows_by_track"]["gnss"] == model.obs.n - sum(
        v for k, v in summary["rows_by_track"].items() if k != "gnss")
    assert {"asc", "desc", "gnss"} <= set(summary["stats"])
    assert summary["recipe"]["folds"] == 6 and summary["sheets"][0]["name"] == "main"


def test_the_moment_and_the_sense_of_slip_come_back(planted):
    _, model, mw_truth, out = planted
    summary = json.loads((out / "summary.json").read_text())
    assert summary["Mw"]["total"] == pytest.approx(mw_truth, abs=0.1)
    assert summary["Mw"]["main"] == pytest.approx(summary["Mw"]["total"], abs=1e-9)
    ss_slip = np.asarray(model.strike_slip)
    assert ss_slip.min() < -0.5 and ss_slip.max() < 0.05, "right-lateral, as planted"


def test_a_second_run_resumes_from_the_checkpoints(planted):
    run, model, _, out = planted
    before = {p.name: p.stat().st_mtime for p in out.glob("*.pkl")}
    again = run.run(stages=STAGES)
    after = {p.name: p.stat().st_mtime for p in out.glob("*.pkl")}
    assert before == after, "no stage was recomputed"
    np.testing.assert_allclose(again.x, model.x, atol=1e-10)


def test_bad_inputs_are_refused(planted):
    run, _, _, _ = planted
    with pytest.raises(ValueError, match="unknown stages"):
        run.run(stages=("A", "Z"))
    with pytest.raises(ValueError, match="sigma"):
        Recipe(sigma="median")
    with pytest.raises(ValueError, match="kind"):
        DataSeries("x", None, kind="gnss")
    with pytest.raises(ValueError, match="first="):
        DataSeries("x", None, first=dict(rms=0.1))


def _branch_pair(dip=135.0):
    """A vertical host along x and a branch 5 km north of it dipping toward it: they meet at 5 km for dip 135."""
    host = FaultTrace.from_lonlat(np.array([-68.9, -68.5]), np.array([10.5, 10.5]), name="host")
    frame = host.local_frame()
    x = np.array([-15e3, 15e3])
    lon, lat = frame.to_lonlat(x, np.array([5e3, 5e3]))
    branch = FaultTrace.from_lonlat(lon, lat, name="branch")
    main = Sheet("host", host, frame, dip=90.0, depth=12e3)
    side = Sheet("branch", branch, frame, dip=dip, depth=12e3, ends_on=main,
                 bounds=dict(strike=(0.0, 0.0), polarity=(0, 1)))
    return frame, main, side


def test_a_sheet_ending_on_another_is_clipped_and_pinned(tmp_path):
    frame, main, side = _branch_pair()
    s, depth = side.meeting
    inner = (s > 2e3) & (s < side.length - 2e3)
    np.testing.assert_allclose(depth[inner], 5e3, atol=60.0, err_msg="a 45-degree branch 5 km away meets at 5 km")
    meshes = [main.start(3e3), side.start(3e3)]
    m = meshes[1]
    node_depth = -m.nodes[:, 2]
    assert node_depth.max() <= 5e3 + 60.0, "nothing of the branch reaches below the host"
    edge = side.lower_edge(m)
    assert edge.sum() >= 4 and np.allclose(node_depth[edge], 5e3, atol=60.0)
    assert m.attrs["layout"] != "" and not main.lower_edge(meshes[0]).any()

    rng = np.random.default_rng(0)
    px, py = rng.uniform(-30e3, 30e3, 40), rng.uniform(-25e3, 25e3, 40)
    px, py = px[np.abs(py) > 2e3], py[np.abs(py) > 2e3]
    lon, lat = frame.to_lonlat(px, py)
    look = np.tile(np.eye(3), (px.size, 1))
    obs = Observations.from_points(np.repeat(lon, 3), np.repeat(lat, 3), rng.normal(0.0, 0.05, 3 * px.size), look,
                                   np.full(3 * px.size, 0.01), name="gnss", frame=frame)
    run = Production(tmp_path / "run", [main, side], [], points=[obs], echo=False,
                     recipe=Recipe(solve=dict(boundary_ratio=1e3, strike=(-5.0, 5.0), dip=(-5.0, 5.0))))
    inv, system = run.system(meshes, obs)
    n_host = meshes[0].n_nodes
    assert np.array_equal(inv.pinned, np.r_[np.zeros(n_host, bool), edge])
    kw, fixed = run.solve_options(inv.mesh)
    n = inv.mesh.n_nodes
    assert fixed[n_host:n].all() and not fixed[:n_host].any() and not fixed[n:].any()
    assert np.all(kw["dip"][0][n_host:] == 0.0) and np.all(kw["dip"][0][:n_host] == -5.0)
    assert np.all(inv.g[:, n_host:n] == 0.0), "a fixed component leaves the fit and the rank"
    model = run.solve(inv, system.lam_for_rank(10.0))
    x = np.asarray(model.x, float)
    assert np.all(x[n_host:n] == 0.0) and np.all(x[n + n_host:2 * n] >= 0.0)
    assert np.abs(x[n_host:n][edge]).max() == 0.0 and np.abs(x[n + n_host:2 * n][edge]).max() < 1e-3


def test_a_clip_the_layout_never_reaches_leaves_it_alone():
    from slipsolve.layout import clip_layout, lloyd_layout

    lay = lloyd_layout(30e3, 10e3, 3e3)
    s = np.linspace(0.0, 30e3, 61)
    assert clip_layout(lay, s, np.full(s.size, 10e3)) is lay
    cut = clip_layout(lay, s, 6e3 + 2e3 * np.sin(s / 5e3))
    depth = -cut.params[:, 1]
    assert depth.max() <= 8e3 + 30.0 and cut.attrs["clipped"]
    assert np.all(depth <= 6e3 + 2e3 * np.sin(cut.params[:, 0] / 5e3) + 30.0)


def test_series_regions_are_used_as_given(planted, tmp_path):
    run, _, _, _ = planted
    grid = run.series["asc"].grid
    ny, nx = grid.ds["los"].shape[1:]
    labels = np.zeros((ny, nx), int)
    labels[:, : nx // 2] = 1
    labels[:, nx // 2:] = 2                  # one connected area split in two by hand
    labels[:3, :3] = 7                       # 36 km2, below the planted recipe's 50 km2: kept, not dropped
    s = DataSeries("asc", grid, kind="phase", ramp="linear", near=2e3, regions=labels)
    prod = Production(tmp_path / "regions", run.sheets, [s], recipe=run.recipe, echo=False)
    prod._setup_lattice()
    los, valid, extra = prod._usable(s)
    assert np.array_equal(valid, np.isfinite(los) & (prod._lattice["dist"] >= 2e3) & (labels > 0))
    assert np.array_equal(extra["regions"][valid], labels[valid]) and 7 in extra["regions"][valid]
    default = DataSeries("asc", grid, kind="phase", ramp="linear", near=2e3)
    plain = Production(tmp_path / "default", run.sheets, [default], recipe=run.recipe, echo=False)
    plain._setup_lattice()
    _, _, connected = plain._usable(default)
    assert len(np.unique(connected["regions"][connected["regions"] > 0])) < 3, "the default relabels: no hand split"
    with pytest.raises(ValueError, match="regions are for phase"):
        DataSeries("rg", grid, kind="offset", ramp="offset", regions=labels)


def test_a_curved_sheet_is_faultmesh_curveds_surface():
    from slipsolve.layout import lloyd_layout
    from slipsolve.production import CurvedSheet

    trace = FaultTrace.from_lonlat(_LON, _LAT, name="planted")
    frame = trace.local_frame()
    dips = [90.0, 80.0, 70.0]
    sheet = CurvedSheet("main", trace, frame, depth=15e3, edge_length=4000.0, segments=3, dips=dips)
    x, y = trace.resample(4000.0, frame).to_local(frame)
    assert sheet.length == pytest.approx(np.sum(np.hypot(np.diff(x), np.diff(y))), rel=1e-12)
    lay = lloyd_layout(sheet.length, 15e3, 4000.0)
    ref = FaultMesh.curved(trace, frame, layout=lay, edge_length=4000.0, segments=3, dips=dips)
    np.testing.assert_array_equal(sheet.mesh(lay).nodes, ref.nodes)
    assert sheet.dip == pytest.approx(80.0) and sheet.start(5e3).n_elements > 0
    with pytest.raises(NotImplementedError):
        Sheet("branch", trace, frame, ends_on=sheet)


def test_stage_x_adds_the_correction_and_quotes_it(planted, tmp_path):
    import shutil

    from slipsolve import SlipInversion

    run, _, _, out = planted
    copy = tmp_path / "run"
    shutil.copytree(out, copy)

    def halve(mesh, obs):                    # dG = -G/2: the same data need twice the slip
        return -0.5 * np.asarray(SlipInversion(mesh, obs, engine=run.engine, basis="node", ramp="none").g, float)

    again = Production(copy, run.sheets, list(run.series.values()), points=run.points, recipe=run.recipe,
                       sees=run.sees, correction=halve, echo=False)
    again.run(stages=["X", "report"])
    stages = json.loads((copy / "stages.json").read_text())
    summary = json.loads((copy / "summary.json").read_text())
    assert "X" in stages and summary["corrected"] and (copy / "cv_X.csv").exists()
    corrected = SlipModel.load(copy / "model.slip.zip")
    plain = SlipModel.load(copy / "model_uncorrected.slip.zip")
    gain = _mw(corrected.moment()) - _mw(plain.moment())
    assert abs(gain - (2.0 / 3.0) * np.log10(2.0)) < 0.02, gain


def _field_case(edge, lattice_offset):
    """A planted model on a hand-set 500 m lattice, offset from the frame origin by `lattice_offset` m."""
    from types import SimpleNamespace

    from slipsolve.basis import make_basis

    trace = FaultTrace.from_lonlat(_LON, _LAT, name="planted")
    frame = trace.local_frame()
    sheet = Sheet("main", trace, frame, dip=90.0, depth=15e3)
    mesh = sheet.start(edge)
    basis = make_basis(mesh, "node")
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    amp = np.sin(np.pi * np.clip(s / sheet.length, 0, 1)) * np.clip(1 + z / 12e3, 0, 1)
    amp[np.asarray(basis.boundary(("bottom", "left", "right")), bool)] = 0
    slip = np.column_stack([basis.element_values(-3.0 * amp), np.zeros(mesh.n_elements)])
    model = SimpleNamespace(mesh=mesh, element_slip=slip)
    x = np.arange(-60e3, 60e3 + 1, 500.0) + lattice_offset
    y = np.arange(-25e3, 25e3 + 1, 500.0) + lattice_offset
    X, Y = np.meshgrid(x, y)
    run = Production("unused", [sheet], [], echo=False)
    dist = trace.distance(X.ravel(), Y.ravel(), frame).reshape(X.shape)
    run._lattice = dict(x=x, y=y, X=X, Y=Y, dist=dist, spacing=500.0)
    return run, model, X, Y, dist


def test_the_surface_field_is_exact_beside_the_fault_and_never_crosses_it():
    run, model, X, Y, dist = _field_case(3e3, 137.0)
    look = ss.look_vector(35.0, -10.0)
    field = sum(c * v for c, v in zip(run.field(model), look))
    sel = (dist >= 2e3) & (dist < 12e3)
    exact = np.full(X.shape, np.nan)
    exact[sel] = HalfSpaceTDE(0.25).forward(model.mesh, model.element_slip, X[sel], Y[sel]) @ look
    err = np.abs(field - exact)
    near = sel & (dist < 5e3)                      # within two field_spacing: forwarded at each pixel
    assert np.nanmax(err[near]) < 1e-6, "no value interpolated across the fault"
    assert np.sqrt(np.nanmean(err[sel & ~near] ** 2)) < 3e-3, "the grid's error beyond: a few mm on metres"

    # The frame origin is on the trace; a lattice snapped to the field grid holds it, and no top node is near it
    run, model, X, Y, dist = _field_case(12e3, 0.0)
    east, north, up = run.field(model)
    on = dist < 1.0
    assert on.any() and np.isnan(east[on]).all() and np.isfinite(east[~on]).all()
    need = dist > 20e3
    partial = run.field(model, need)[0]
    assert np.isnan(partial[~need]).all() and np.allclose(partial[need], east[need], atol=1e-9)


class _NotAHalfSpace:
    """A half-space of Poisson's ratio 0.4 that is not a `HalfSpaceTDE`, as a layered engine is not."""

    name = "not_a_half_space"

    def __init__(self):
        self._inner = HalfSpaceTDE(0.4)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def test_the_sigma_is_measured_against_the_models_own_engine(planted, tmp_path):
    run, _, _, _ = planted
    prod = Production(tmp_path / "engine", run.sheets, list(run.series.values()), engine=_NotAHalfSpace(),
                      recipe=run.recipe, echo=False)
    prod._setup_lattice()
    obs, _ = prod.sample()
    inv, system = prod.system([run.sheets[0].start(6e3)], obs)
    model = prod.solve(inv, system.lam_for_rank(15.0))
    seen, measure = {}, prod._sigma

    def spy(s, los, valid, regions, prediction, model):
        seen[s.name] = (valid, prediction)
        return measure(s, los, valid, regions, prediction, model)

    prod._sigma = spy
    prod.sample(model)
    valid, prediction = seen["asc"]
    ds, lat = run.series["asc"].grid.ds, prod._lattice
    look = [np.asarray(ds[f"los_{c}"].values, float) for c in ("east", "north", "up")]
    exact = model.forward(lat["X"][valid], lat["Y"][valid], look=[v[valid] for v in look])
    half = sum(c * v for c, v in zip(prod.field(model, valid), look))[valid]

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2)))

    print(f"signal {rms(exact):.4f} m; half-space field off by {rms(half - exact):.5f}, "
          f"with the engine's difference {rms(prediction[valid] - exact):.5f}")
    assert rms(prediction[valid] - exact) < 0.15 * rms(half - exact)


def test_redoing_a_stage_redoes_the_later_ones(planted, tmp_path):
    import shutil

    from slipsolve.production import _file_digest

    run, _, _, out = planted
    copy = tmp_path / "run"
    shutil.copytree(out, copy)
    again = Production(copy, run.sheets, list(run.series.values()), points=run.points, recipe=run.recipe,
                       sees=run.sees, echo=False)
    meshes, obs, lam, rank, extra = again.load("C.pkl")
    again._save("C.pkl", meshes, obs, 1.5 * lam, rank + 1.0, extra)          # as a redone stage C leaves it
    with pytest.raises(ValueError, match="earlier C.pkl"):
        again.run(stages=["report"])
    again.run()
    assert again.load("final.pkl")[3] == pytest.approx(rank + 1.0), "D was redone from the new C"
    record = json.loads((copy / "stages.json").read_text())
    assert record["E"]["final"] == _file_digest(copy / "final.pkl"), "and so was E"


def test_the_greens_function_cache_tells_engines_apart(planted, tmp_path):
    from slipsolve import SlipInversion

    run, model, _, _ = planted
    series = list(run.series.values())
    plain = Production(tmp_path / "g", run.sheets, series, echo=False)._green(model.mesh, model.obs)
    other = Production(tmp_path / "g", run.sheets, series, engine=HalfSpaceTDE(0.45), echo=False)
    got = other._green(model.mesh, model.obs)
    want = SlipInversion(model.mesh, model.obs, engine=HalfSpaceTDE(0.45), ramp="none", basis="node").g
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
    assert not np.allclose(got, plain)


def test_a_curved_sheets_geometry_goes_into_the_record():
    from slipsolve.production import CurvedSheet, _jsonable

    trace = FaultTrace.from_lonlat(_LON, _LAT, name="planted")
    frame = trace.local_frame()
    x, y = trace.to_local(frame)
    nx, ny = trace.normals(frame)
    bottom = FaultTrace.from_lonlat(*frame.to_lonlat(x + 10e3 * nx, y + 10e3 * ny), name="base")
    sheet = CurvedSheet("main", trace, frame, depth=15e3, edge_length=4000.0, bottom_trace=bottom, bottom_depth=15e3)
    record = json.loads(json.dumps(_jsonable(sheet.curved), default=float))
    assert record["bottom_trace"]["name"] == "base" and len(record["bottom_trace"]["lon"]) == len(_LON)


def test_the_slip_change_between_rounds_is_relative_to_the_slip():
    from slipsolve.production import _slip_change

    rng = np.random.default_rng(0)
    slip = rng.normal(size=200)                    # strike then dip, 100 nodes
    moved = slip * (1.0 + 0.0244 * rng.choice([-1.0, 1.0], 200))
    assert _slip_change(moved, slip) == pytest.approx(0.0244, rel=0.05), "not 0.0244 / sqrt(2)"
