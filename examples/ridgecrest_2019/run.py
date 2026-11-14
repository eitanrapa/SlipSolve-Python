"""The 2019 Ridgecrest (California) slip model: the Mw 6.4 and Mw 7.1 earthquakes in one run of slipsolve's
production recipe.

    python run.py [stages...] [name=prod] [workers=5]

Reads the data `get_data.py` downloaded, builds the CRUST1.0 tables if they are missing, and runs
`slipsolve.production.Production` into runs/<name>/. Stages: A B C0 C D E report (all by default); each one is
checkpointed, so an interrupted run resumes.

  fault   two vertical sheets 20 km deep: the Mw 7.1's NW-striking fault and the Mw 6.4's NE-striking fault, which
          crosses it
  data    on a 500 m lattice: ALOS-2 tracks 65 and 66 (a cubic each) and Sentinel-1 tracks 64 and 71 (a plane each),
          from 2 km of the traces; Sentinel-2 east and north offsets 2-30 km from the traces, a plane each; GNSS
          (east, north, up) within 120 km of the epicentres, sigma x 3 and at least 2 cm
  engine  LayeredTDE on CRUST1.0 averaged over the 1-degree cells the traces cross
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ARGV = sys.argv[1:]
OPT = dict(tok.split("=", 1) for tok in _ARGV if "=" in tok)
STAGES = [tok for tok in _ARGV if "=" not in tok] or ["A", "B", "C0", "C", "D", "E", "report"]

from slipsolve import FaultTrace, LayeredTDE, gnss_points, on_lattice, read_gnss_offsets  # noqa: E402
from slipsolve.production import DataSeries, Production, Recipe, Sheet  # noqa: E402

import crust  # noqa: E402
import products  # noqa: E402

DEPTH = 20e3
LATTICE = 500.0
BOX = (-118.1, -117.0, 35.3, 36.2)         # lon / lat covering every product
#: series -> (UCSD set, ramp); every phase series is masked within 2 km of the traces
PHASE = {"A2T065_phase": ("ALOS2_los_T065", "cubic"), "A2T066_phase": ("ALOS2_los_T066", "cubic"),
         "S1T064_phase": ("S1_los_T064", "linear"), "S1T071_phase": ("S1_los_T071", "linear")}
RECIPE = Recipe(rank0=40.0, start_edge=3000.0, sigma="whole", slab_span="all", shifts=1,
                solve=dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=3e3, sides=("bottom", "left", "right"),
                           polarity=(0, 0, 0), strike=(-20.0, 20.0), dip=(-15.0, 15.0)),
                workers=int(OPT.get("workers", 5)))


def build():
    lines = json.loads((HERE / "traces.json").read_text())
    traces = {k: FaultTrace.from_lonlat(v["lon"], v["lat"], name=k) for k, v in lines.items()}
    frame = traces["m71"].local_frame()
    sheets = [Sheet(k, traces[k], frame, dip=90.0, depth=DEPTH) for k in ("m71", "m64")]

    grids = {name: products.ucsd(src) for name, (src, _) in PHASE.items()}
    grids.update({f"s2_{c}": products.sentinel2(c) for c in "en"})
    x, y = frame.to_local(np.array([BOX[0], BOX[1], BOX[0], BOX[1]]), np.array([BOX[2], BOX[2], BOX[3], BOX[3]]))
    # each lattice cell the median of its pixels (the symmetric median: the mean of the two middle values)
    grids = on_lattice(grids, frame, spacing=LATTICE, bounds=(x.min(), x.max(), y.min(), y.max()), resampling="median",
                       cache=HERE / "work")
    series = [DataSeries(name, grids[name], kind="phase", ramp=ramp, near=2e3) for name, (_, ramp) in PHASE.items()]
    series += [DataSeries(f"s2_{c}", grids[f"s2_{c}"], kind="offset", ramp="linear", near=2e3, reach=30e3)
               for c in "en"]
    gnss = gnss_points(read_gnss_offsets(HERE / "gnss" / "ngl_offsets.csv"), frame, components="enu", scale=3.0,
                       floor=0.02, name="gnss")

    model = crust.velocity_model([list(zip(v["lon"], v["lat"])) for v in lines.values()], HERE / "crust1.0",
                                 HERE / "crust1_ridgecrest.json", "CRUST1.0, Ridgecrest (mean over the rupture's cells)")
    tables = crust.tables(model, HERE / "remainder_ridgecrest_crust1.npz")
    return Production(HERE / "runs" / OPT.get("name", "prod"), sheets, series, points=[gnss],
                      engine=LayeredTDE(tables), velocity_model=model, recipe=RECIPE, traces=list(traces.values()))


if __name__ == "__main__":
    build().run(stages=STAGES)
