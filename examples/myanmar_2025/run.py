"""The 2025 Mandalay (Myanmar) slip model: the Mw 7.7 rupture of the Sagaing fault on slipsolve's production recipe,
with a rigidity contrast across the fault.

    python run.py [stages...] [name=prod] [workers=5]

Reads the data `get_data.py` downloaded, builds the CRUST1.0 tables if they are missing, and runs
`slipsolve.production.Production` into runs/<name>/. Stages: A B C0 C D E X report (all by default); each one is
checkpointed, so an interrupted run resumes.

  fault   the rupture trace (He & Yun 2025), 30 km deep: one sheet on `FaultMesh.curved`'s surface with a dip per
          eighth of the trace, south to north 90 90 95 100 110 120 120 120 (above 90 it leans east); right-lateral
  data    on a 500 m lattice: the two ALOS-2 interferograms (coherence >= 0.3, from 5 km of the trace, a cubic and one
          offset per region each) and the Sentinel-1 range and azimuth offsets of four tracks (from 2 km of the trace, a
          plane each); no GNSS
  recipe  rank0 80; start meshes of 5 km; sigma from every pixel; 24 slabs; the first offsets samples split at 0.6 m
          (range) or 2.4 m (azimuth) in cells of at least 6 km, weighted by the median of each product's own sigma
  X       the final mesh and samples solved again with the west block 1.3 times softer above 10 km (`contrast.py`),
          at the final rank: the model the report quotes
  engine  LayeredTDE on CRUST1.0 averaged over the 1-degree cells the trace crosses
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ARGV = sys.argv[1:]
OPT = dict(tok.split("=", 1) for tok in _ARGV if "=" in tok)
STAGES = [tok for tok in _ARGV if "=" not in tok] or ["A", "B", "C0", "C", "D", "E", "X", "report"]

from slipsolve import FaultTrace, LayeredTDE, on_lattice  # noqa: E402
from slipsolve.production import CurvedSheet, DataSeries, Production, Recipe  # noqa: E402

import contrast  # noqa: E402
import crust  # noqa: E402
import products  # noqa: E402

DEPTH = 30e3
LATTICE = 500.0
BOX = (94.0, 98.0, 16.5, 24.0)             # lon / lat of the products' grid
DIPS = [90.0, 90.0, 95.0, 100.0, 110.0, 120.0, 120.0, 120.0]
DIP_EDGE = 4000.0                          # the trace is resampled at this spacing to fit the dipping surface on
COHERENCE, NEAR, EXCLUDE = 0.3, 5e3, 2e3
FIRST = {"range": dict(rms_min=0.6, width_min=6000.0, sigma="sigma"),
         "azimuth": dict(rms_min=2.4, width_min=6000.0, sigma="sigma")}
RECIPE = Recipe(rank0=80.0, start_edge=5000.0, sigma="whole", folds=24, shifts=1, slab_span="all",
                solve=dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=3e3, sides=("bottom", "left", "right"),
                           polarity=(-1, 0, 0), strike=(-20.0, 20.0), dip=(-5.0, 5.0)),
                workers=int(OPT.get("workers", 5)))


def trace_lonlat():
    """(lon, lat) of the mapped rupture, south to north."""
    a = np.loadtxt(HERE / "trace" / "fault_trace_from_NS.txt")
    order = np.argsort(a[:, 1])
    return a[order, 0], a[order, 1]


def fault_trace(step=1000.0):
    """The rupture as a `FaultTrace`, its vertices thinned to about `step` metres apart along it."""
    from pyproj import Geod

    lon, lat = trace_lonlat()
    d = np.r_[0.0, np.cumsum(Geod(ellps="WGS84").inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])]
    keep = np.unique(np.r_[np.searchsorted(d, np.arange(0.0, d[-1], step)), lon.size - 1])
    return FaultTrace.from_lonlat(lon[keep], lat[keep], name="sagaing_2025")


def build():
    trace = fault_trace()
    frame = trace.local_frame()
    sheet = CurvedSheet("sagaing", trace, frame, depth=DEPTH, edge_length=DIP_EDGE, segments=len(DIPS), dips=DIPS)

    x, y = frame.to_local(np.array([BOX[0], BOX[1], BOX[0], BOX[1]]), np.array([BOX[2], BOX[2], BOX[3], BOX[3]]))
    # each lattice cell the median of its pixels (the symmetric median: the mean of the two middle values)
    grids = on_lattice({products.series_name(p): products.grid(p) for p in products.PRODUCTS}, frame,
                       spacing=LATTICE, bounds=(x.min(), x.max(), y.min(), y.max()), resampling="median",
                       cache=HERE / "work")
    series = []
    for product, (kind, _) in products.PRODUCTS.items():
        name = products.series_name(product)
        g = grids[name]
        if kind == "phase":
            series.append(DataSeries(name, g, kind="phase", ramp="cubic", near=NEAR,
                                     valid=g.ds["coherence"].values >= COHERENCE))
        else:
            series.append(DataSeries(name, g, kind="offset", ramp="linear", near=EXCLUDE, first=FIRST[kind]))

    model = crust.velocity_model([list(zip(*trace_lonlat()))], HERE / "crust1.0", HERE / "crust1_myanmar.json",
                                 "CRUST1.0, Myanmar (mean over the rupture's cells)")
    tables = crust.tables(model, HERE / "remainder_myanmar_crust1.npz")
    return Production(HERE / "runs" / OPT.get("name", "prod"), [sheet], series, engine=LayeredTDE(tables),
                      velocity_model=model, recipe=RECIPE,
                      correction=contrast.correction(contrast.softer_side(trace), tables))


if __name__ == "__main__":
    build().run(stages=STAGES)
