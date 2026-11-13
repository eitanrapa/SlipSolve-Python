# Production runs

A single inversion is easy to set up; a careful one has many steps. The mesh should be
as fine as the data resolve and no finer. The samples should be dense where the model
varies and sparse where it does not. Each data set should be weighted by its own noise,
and the smoothing should be chosen by how well the model predicts data it was not fitted
to. [`Production`](../api/production.md#slipsolve.production.Production) runs those steps
as one recipe, in checkpointed stages, so a run can be interrupted and resumed.

## What a run needs

```python
import numpy as np
from slipsolve import FaultTrace, LayeredTDE, Observations, RemainderTables, VelocityModel, frame_lattice, resample_all
from slipsolve.production import DataSeries, Production, Recipe, Sheet

trace = FaultTrace.from_kml("rupture.kml")
frame = trace.local_frame()

# every raster on one lattice in the frame (median resampling suits offsets with outliers)
grids = resample_all({"S1_asc": asc, "S1_dsc": dsc, "S1_asc_rg": asc_rg}, frame, spacing=500.0,
                     bounds=(-150e3, 150e3, -100e3, 100e3), resampling="med")

sheets = [Sheet("main", trace, frame, dip=70.0, depth=20e3)]
series = [
    DataSeries("S1_asc", grids["S1_asc"], kind="phase", ramp="linear", near=3e3,
               valid=coherence_asc >= 0.25),
    DataSeries("S1_dsc", grids["S1_dsc"], kind="phase", ramp="linear", near=3e3),
    DataSeries("S1_asc_rg", grids["S1_asc_rg"], kind="offset", ramp="linear", near=2e3, reach=30e3),
]
gnss = Observations.from_points(lon, lat, values, look, sigma, name="gnss", frame=frame)

tables = RemainderTables.load("crust_tables.npz")
run = Production(
    "runs/prod1", sheets, series, points=[gnss],
    engine=LayeredTDE(tables), velocity_model=crust,
    recipe=Recipe(rank0=50.0, workers=6,
                  solve=dict(polarity=(1, 0, 0), strike=(-20.0, 20.0), dip=(-5.0, 5.0))),
)

if __name__ == "__main__":
    model = run.run()
```

- A [`Sheet`](../api/production.md#slipsolve.production.Sheet) is one fault surface: a
  trace, a dip and a depth. Each node at depth \(z\) is moved \(z \cot(\text{dip})\)
  along the trace's left-hand normal. The normal is smoothed along strike (10 km by
  default) so a wiggly trace does not fold the sheet at depth. A dip below 90 degrees dips
  to the left of the drawing direction. `profile=`, a function of along-strike distance
  and depth returning a dip, makes a listric or twisted sheet.
- A [`DataSeries`](../api/production.md#slipsolve.production.DataSeries) is one raster on
  the run's lattice, with `los` positive toward the sensor.
  - `kind="phase"`: masked within `near` of every trace. Every connected region of at
    least `Recipe.min_region_km2` gets its own offset, and the series gets the polynomial
    `ramp`.
  - `kind="offset"`: pixel offsets, used from `near` to `reach` of the traces, with
    `ramp` as their nuisance terms.
  - `valid=` masks pixels (a coherence threshold, a zone to leave out).
  - `regions=` (phase only) gives integer labels on the lattice, each with its own
    offset, used as given in place of the connected regions: labels drawn on the native
    product, or a region split by hand. The `near` mask applies on top.
- `points` are used as they are, with their own weights, for instance GNSS.
- `sees={"gnss_e1": [0, 1]}` restricts which sheets a track sees, so GNSS that resolves
  two events can be given to each event's faults. `events={"e1": [0, 1]}` reports a
  moment per group of sheets.

The run writes everything to its folder: `run.log`, a checkpoint per stage, the
Green's-function cache, the convergence and cross-validation tables, `final.slip.zip`,
`model.slip.zip` and `summary.json` (the moments per sheet, the fit per series, and the
recipe).

## The stages

| stage | what | output |
|---|---|---|
| `A` | start meshes (`start_edge`), first samples, solve at `rank0` | `A.pkl` |
| `B` | re-mesh from the resolution at `rank0` until the element count settles | `B.pkl`, `converge_B.csv` |
| `C0` | redraw every series' samples from the model until they settle | `C0.pkl`, `sampling_C0.csv` |
| `C` | the rank by blocked cross-validation on mesh B | `C.pkl`, `cv_C.csv`, `cv_C.json` |
| `D` | re-mesh at that rank: the final model | `final.pkl`, `final.slip.zip`, `converge_D.csv` |
| `E` | cross-validation on the final mesh: is the rank inside its 1% band? | `cv_E.csv` |
| `report` | the final solve and its summary | `model.slip.zip`, `summary.json` |

`run(stages=...)` runs a subset. A stage whose checkpoint exists is skipped, so delete a
checkpoint to redo its stage and everything after it.

**Meshes.** Each re-mesh computes the resolution matrix at the stage's rank. It fits a
Gaussian to every parameter's resolving kernel and asks for elements a `fraction` of that
width, within `size_min` to `size_max`. It then builds an unstructured layout with that
size field (see [Regularization and resolution](regularization.md)). Re-meshing repeats
until the element count changes by less than `count_tol` twice running. If the count
reverses direction, the size field is under-relaxed.

**Samples.** The first samples come from the data alone, or pass `samples=` to start from
your own. From then on each series' quadtree splits while the model varies by more than
`split[kind]` times the series' sigma. This puts samples where the model has structure
rather than where the data are noisy, which matters for offsets with heavy-tailed
outliers. Each series is weighted one over its sigma, re-measured from the latest model
every round:

- `sigma="far"` (the default): the residual beyond `sigma_far` of every trace, after the
  series' own nuisance terms, as block medians `sigma_block` wide. The near field's model
  error does not count as noise.
- `sigma="whole"`: every pixel of a phase series after region offsets and a plane, and
  every sample of an offsets series.

Rounds continue until the row count and the slip both change by less than `round_tol`.

## Choosing the rank

Stage `C` chooses the smoothing by blocked cross-validation (see
[Regularization and resolution](regularization.md)). The data are cut into `folds` equal
slabs along the first sheet's trace, and each slab is predicted from a model fitted to
the others. The score is the held-out residual over sigma, summed over a ladder of
weights from rank `cv_ranks[1]` down to `cv_ranks[0]`.

- **Shifted partitions.** `shifts` partitions, each with its slab edges moved by a
  fraction of a slab, are averaged before the minimum is chosen. One partition's choice
  depends on where its edges happen to fall, and the average removes most of that.
- **Slab width.** Keep slabs wider than the noise's correlation length, or held-out data
  share noise with their neighbours and the chosen rank climbs.
- **Slab extent.** With `slab_span="data"` the slabs divide the extent of the series'
  samples, and point data far away join the end slabs instead of stretching them.

Every fold is solved from the full normal equations minus its own rows, so a curve costs
seconds, not a re-inversion per fold. The same machinery is available directly:
[`NormalSystem`](../api/production.md#slipsolve.selection.NormalSystem) holds an
inversion's normal equations. It solves them exactly at any weight, gives the effective
rank there, and finds the weight that gives a chosen rank.
[`choose_rank`](../api/production.md#slipsolve.selection.choose_rank) scores
[`slab_folds`](../api/production.md#slipsolve.selection.slab_folds).
[`heldout_residuals`](../api/production.md#slipsolve.selection.heldout_residuals) shows
which rows a cross-validation gain comes from.

```python
from slipsolve.selection import NormalSystem, choose_rank, slab_folds

system = NormalSystem(inversion, boundary_ratio=1e3, polarity=(-1, 0, 0))
folds = [slab_folds(obs, 12, direction, shift=k / 4)[0] for k in range(4)]
choice = choose_rank(system, folds)
print(choice.rank, choice.bands)
```

Stage `E` repeats the cross-validation on the final mesh and samples, and reports
whether the rank chosen on mesh B lies inside the final mesh's 1% band.

## Sheets that meet, and per-sheet bounds

A sheet can end where it meets another one, as an antithetic fault does against the main
fault at depth:

```python
main = Sheet("main", main_trace, frame, dip=50.0, depth=20e3)
west = Sheet("west", west_trace, frame, dip=135.0, depth=20e3, ends_on=main,
             bounds=dict(strike=(0.0, 0.0), polarity=(0, 1)))
```

- `ends_on=main` finds, for every column of `west`, the depth where it first crosses
  `main` (`Sheet.meeting_depth`). Every mesh of `west` is then cut off below that curve
  (`slipsolve.layout.clip_layout`), so nothing reaches through the main sheet, and its
  nodes on the curve are pinned to zero slip, as a bottom edge is
  (`SlipInversion.pinned`). The log reports how much of the sheet ends on the other and
  at what depths.
- `bounds=` overrides `Recipe.solve`'s `strike`, `dip` and `polarity` on that sheet's
  nodes alone; here no strike-slip and positive dip-slip only. A component fixed at zero
  also loses its Green's functions, so it counts neither in the fit nor in the rank.

## Constraints beyond the recipe

`constrain=` takes a function `(inversion, meshes) -> inversion`, applied to every
system the run assembles. Use it for what the recipe and the sheets do not cover, such
as another zero-slip condition. Bounds, polarity, the zero-slip sides and the smoothing
ratios go in `Recipe.solve`, which is passed to both `SlipInversion.solve` and
`NormalSystem`; `Sheet(bounds=)` overrides the bounds on one sheet.

## Workers

With `workers` above one, the Green's functions and the surface field are computed in
that many spawned processes. Each process loads the layered tables from their file, so
`LayeredTDE` tables must have been loaded from, or saved to, a file. Spawned processes
re-import the calling script, so keep the run under `if __name__ == "__main__":`.
