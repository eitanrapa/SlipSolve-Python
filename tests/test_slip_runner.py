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
