"""The 2023 Kahramanmaraş (Türkiye) slip model: one run on slipsolve's production recipe.

    python run.py [stages...] [name=prod] [workers=6]

Reads the data `get_data.py` downloaded, builds the CRUST1.0 tables if they are missing, and runs
`slipsolve.production.Production` into runs/<name>/. Stages: A B C0 C D E report (all by default); each one is
checkpointed, so an interrupted run resumes.

  fault   four sheets 26 km deep: the East Anatolian fault (south) and the Narli splay of event 1, the Cardak fault
          (north, 65 N) and a N-S strand of event 2; the rest vertical
  data    Sentinel-1 range and azimuth offsets on four tracks (one constant each, from 2 km of every trace; 116A
          without the 35 km around the Feb 20 Mw 6.4 its pair spans), LiCSAR phase on six frames (coherence >= 0.25,
          from 15 km of every trace, one offset per region + a plane), GNSS of each event (horizontal, sigma x 3,
          2-300 km from a trace); event 1's GNSS sees the south and Narli sheets, event 2's the north and N-S
  engine  LayeredTDE on CRUST1.0 averaged over the 1-degree cells the traces cross
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ARGV = sys.argv[1:]
OPT = dict(tok.split("=", 1) for tok in _ARGV if "=" in tok)
STAGES = [tok for tok in _ARGV if "=" not in tok] or ["A", "B", "C0", "C", "D", "E", "report"]

from slipsolve import (FaultTrace, LayeredTDE, gnss_points, on_lattice, read_gnss_offsets, read_licsar,  # noqa: E402
                       read_offsets)
from slipsolve.production import DataSeries, Production, Recipe, Sheet  # noqa: E402

import crust  # noqa: E402
import geometry  # noqa: E402

CEDA = HERE / "ceda_ou2024" / "inputs"
LICSAR = HERE / "licsar"
OFFSET_PAIRS = {"014A": ("014A_05232_242525", "20230128_20230209"), "021D": ("021D_05266_252525", "20230129_20230210"),
                "116A": ("116A_05207_252525", "20230204_20230228"), "123D": ("123D_05292_242525", "20230205_20230217")}
PHASE_PAIRS = {"014A": ("014A_05232_242525", "20230128_20230209"), "021D": ("021D_05266_252525", "20230129_20230210"),
               "116A": ("116A_05207_252525", "20230204_20230216"), "123D": ("123D_05292_242525", "20230205_20230217"),
               "094Ds": ("094D_05288_130913", "20230203_20230215"), "094Dn": ("094D_05100_131313", "20230203_20230215")}
GNSS = {"gnss_e1": ("us6000jllz_final5min.txt", ["CYL2"]), "gnss_e2": ("us6000jlqa_final5min.txt", ["CYL2", "EKZ1"])}
MW64 = (36.063, 36.214)                       # the Feb 20 2023 Mw 6.4 (Lomax 2023), inside the 116A offsets' pair
DIPS = {"south": 90.0, "narli": 90.0, "north": 65.0, "ns": 90.0}
DEPTH = 26e3
LATTICE = 250.0
# the first, data-driven samples of the offsets: their outliers need large cells and loose thresholds
FIRST = {"rng": dict(rms_min=0.6, sigma=0.15, width_min=6000.0), "azi": dict(rms_min=2.4, sigma=0.6, width_min=6000.0)}
RECIPE = Recipe(rank0=78.0, sigma="whole", folds=24, shifts=1, slab_span="all", min_region_km2=100.0,
                solve=dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=3e3, sides=("bottom", "left", "right"),
                           polarity=(1, 0, 0), strike=(-20.0, 20.0), dip=(-5.0, 5.0)),
                workers=int(OPT.get("workers", 6)))


def lattice_bounds(frame):
    """The frame box covering every product's footprint (34.0-41.3 E, 34.6-40.3 N)."""
    x, y = frame.to_local(np.array([34.0, 41.3, 34.0, 41.3]), np.array([34.6, 34.6, 40.3, 40.3]))
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def build():
    lines = geometry.traces()
    traces = {k: FaultTrace(v[:, 0], v[:, 1], name=k) for k, v in lines.items()}
    frame = traces["south"].local_frame()
    sheets = [Sheet(k, traces[k], frame, dip=DIPS[k], depth=DEPTH) for k in DIPS]

    grids = {}
    for track, (folder, pair) in OFFSET_PAIRS.items():
        for comp in ("rng", "azi"):
            grids[f"{track}_{comp}"] = read_offsets(
                CEDA / "sentinel-1" / folder / f"{pair}.geo.{comp}.tif",
                *(CEDA / "component_vector_coefficients" / f"{track}_{comp}_{c}.tif" for c in "ENU"),
                sigma=CEDA / "empirical_offset_uncertainties" / f"{track}_{comp}_sig.tif", name=f"{track}_{comp}")
    for track, (folder, pair) in PHASE_PAIRS.items():
        grids[f"{track}_phase"] = read_licsar(LICSAR / folder, pair, name=f"{track}_phase")
    # each lattice cell the median of its pixels (the symmetric median: the mean of the two middle values)
    grids = on_lattice(grids, frame, spacing=LATTICE, bounds=lattice_bounds(frame), resampling="median",
                       cache=HERE / "work")

    g0 = next(iter(grids.values())).ds
    X, Y = np.meshgrid(g0["x"].values, g0["y"].values)
    x64, y64 = frame.to_local(*MW64)
    away_from_mw64 = np.hypot(X - x64, Y - y64) >= 35e3
    series = []
    for name, grid in grids.items():
        if name.endswith("_phase"):
            series.append(DataSeries(name, grid, kind="phase", ramp="linear", near=15e3,
                                     valid=grid.ds["coherence"].values >= 0.25))
        else:
            series.append(DataSeries(name, grid, kind="offset", ramp="offset", near=2e3,
                                     valid=away_from_mw64 if name.startswith("116A") else None,
                                     first=FIRST[name.split("_")[1]]))

    points = [gnss_points(read_gnss_offsets(HERE / "gnss" / path), frame, components="en", scale=3.0,
                          traces=list(traces.values()), max_distance=300e3, min_distance=2e3, drop=drop, name=name)
              for name, (path, drop) in GNSS.items()]

    model = crust.velocity_model(list(lines.values()), HERE / "crust1.0", HERE / "crust1_turkiye.json",
                                 "CRUST1.0, Turkiye (mean over the rupture's cells)")
    tables = crust.tables(model, HERE / "remainder_turkiye_crust1.npz")
    return Production(HERE / "runs" / OPT.get("name", "prod"), sheets, series, points=points,
                      engine=LayeredTDE(tables), velocity_model=model, recipe=RECIPE,
                      sees={"gnss_e1": [0, 1], "gnss_e2": [2, 3]}, events={"e1": [0, 1], "e2": [2, 3]},
                      traces=list(traces.values()))


if __name__ == "__main__":
    build().run(stages=STAGES)
