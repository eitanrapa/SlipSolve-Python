# slipsolve-python

Resolution-driven geodetic slip inversion on curved triangular fault meshes,
from InSAR line-of-sight displacement.

> **Lineage.** This is a Python translation of, and extension to,
> [**SlipSolve-Curve**](https://github.com/x3zou/SlipSolve-Curve) (Xiaoyu Zou,
> MATLAB, MIT), which is itself descended from **SlipSolve** (ZeyuJin). It is
> *not* either of those. The `-python` in the distribution name is there to say
> which one you are installing; the import name is `slipsolve`.

```python
import slipsolve
```

## What it does

Given line-of-sight displacement from one or more InSAR scenes, a mapped fault
trace and a dip model, it solves for the slip distribution on a triangulated
fault surface:

- **curved fault meshes** built from a surface trace, with element-constant or
  nodal (P1) slip;
- **layered elastic Green's functions** from the package's own wavenumber-domain
  solver (`RemainderTables`, `LayeredTDE`): exact near the fault, at depth too.
  EDGRN/EDCMP tables (`LayeredPointSource`) and a closed-form homogeneous
  half-space (Nikkhoo & Walter triangular dislocations) remain available;
- **model-based quadtree subsampling** of each scene (Lohman & Simons 2005,
  Wang & Fialko 2015);
- **Laplace–Beltrami regularisation** with bound constraints, and per-track
  ramp and offset terms solved *jointly* with slip;
- **resolution analysis** — the model resolution matrix, effective rank,
  anisotropic averaging kernels, posterior covariance and regularisation bias;
- **measured data covariance** — an exponential variogram fitted per track from
  the far-field residual, or a REML fit on the samples themselves
  (`fit_reml`), with block-diagonal whitening;
- **geometry sweeps** — per-segment dips, or dips that change with depth,
  scored by the marginal likelihood with slip integrated out, with bootstrap
  intervals and a check for forward-model error (`DipSweep`).

## What is new here relative to the MATLAB

Two things, both benchmarked:

**Adaptive quadrature for layered Green's functions.** The reference
implementations — EDCMP, and SlipSolve-Curve itself — integrate each
`(triangle, observation)` pair at a fixed order, 91 points in EDCMP's case. The
required order actually depends on how far the observation is from the element
*in units of the element's own size*, so it is chosen per pair from a measured
error model. Result: **48× fewer quadrature points**, and layered assembly at
**0.86× the cost of the closed-form homogeneous engine**. There is no longer a
computational reason to accept a homogeneous half-space, which matters because a
uniform crust biases the shallow/deep slip partitioning.

**A joint resolution loop.** Barnhart & Lohman (2010) size a mesh from
resolution; Lohman & Simons (2005) subsample data from a predicted field. Both
papers keep the two steps sequential and separate. Here one resolution matrix
drives both, alternating. Measured honestly, this is an **efficiency** method
rather than an accuracy one — see the findings below.

## Installing

```bash
conda env create -f environment.yml && conda activate slipsolve
pip install -e .
```

The dependency list is deliberately short — numpy, scipy, xarray, zarr, pyproj,
rioxarray, matplotlib. This package works on points and meshes rather than
rasters, is eager throughout and uses no dask, so it does not inherit a raster
pipeline's dependencies.

| extra | what for |
|---|---|
| `[layered]` | `pygrnwang`, which bundles and builds Wang's EDGRN/EDCMP Fortran. Only needed if you want the package to *generate* layered tables; you can equally run EDGRN yourself and hand over the tables. |
| `[dev]` | `pytest`, plus `cutde` as an independent oracle for the triangular dislocation. |

Reading NISAR products is **not** part of this package. Feed it any object
exposing an xarray `.ds` with `los`, `los_east/north/up`, `x`, `y` — a `.grd`
pair, a plain Dataset, or [`nisar_tools`](https://github.com/eitanrapa/nisar_tools)'
`LOSStack`.

## What the method actually buys — measured

Run on six scenes (NISAR + ALOS) over the San Sebastián fault, Venezuela, and
against synthetic targets with known truth:

- **λ dominates.** Four mesh designs at matched effective rank agree on misfit to
  0.3% and on Mw to 0.005. Row-structured and unstructured layouts, scored
  against planted truth, agree to within 3% at every damping. Changing λ moves
  the answer roughly **thirty times** as far as changing the mesh does.
- **The L-curve corner is not the minimum-error point.** Scored against truth
  under measured noise, total error minimises at effective rank ~79 while the
  corner sits at 21 — the corner is **40% worse**.
- **More data does not buy resolution unless λ is relaxed to spend it.**
  Removing every coverage mask (2.4× the observations) made recovery *worse* at
  matched rank.
- **Data covariance does not improve the point estimate** (0.147 m rms against
  0.069, five seeds out of five). GLS is minimum-variance only among *unbiased*
  estimators, and this one is bound-constrained and ~97% regulariser. It is kept
  for honest error bars and a meaningful χ², not for the estimate.
- **The moment integrates the rigidity exactly.** `SlipModel.moment()` averages a
  layered crust's rigidity over each element (steps included) and samples only
  the slip magnitude. The node-lumped rule it replaced on 2026-09-25, still
  available as `moment(rule="lumped")`, read Mw **0.002–0.006 high** at
  production smoothing on the Venezuela meshes and **0.011** at effective rank
  340, by an amount that varies with the mesh and the slip — so Mw values
  recorded before then are not corrected by a constant.

The full write-up, with the mathematics, is in `docs/`.

## Calibrating it for your own data

`slipsolve.calibrate` runs a standardised checkerboard recovery test on *your*
geometry, coverage and measured noise, and reports what element size, layout and
damping to use, and what wavelength you can claim to have resolved. Start with
the free part:

```python
from slipsolve.calibrate import discretisation_ceiling
for row in discretisation_ceiling(mesh, cells=(20e3, 40e3, 60e3)):
    print(row)          # pure geometry, milliseconds, no data needed
```

That tells you which wavelengths your mesh can carry *at all*, so you never
mistake a parameterisation failure for a resolution limit.

## Sweeping the dip, segment by segment

`slipsolve.DipSweep` asks the data which dip each stretch of the fault wants. It
cuts the trace into along-strike segments, gives each a dip from a grid, and
scores every combination. Every combination uses the same samples and the same
regularisation, so the geometry is the only thing that changes:

```python
sweep = DipSweep(trace, obs, segments=6, dips=range(60, 121, 10), engine=engine,
                 ramp="cubic", polarity=(-1, 0, 0), max_depth=40e3, edge_length=8e3)
result = sweep.run(score="evidence")   # or sweep.run(rank=62) for misfit at a fixed rank
result.profiles()                      # each segment's score against its dip, the others at the best
boot = sweep.bootstrap(result)         # per-segment dips with an honest spread
boot.intervals(); boot.consistency()
```

**Score by the evidence.** `score="evidence"` integrates the slip, the ramp, the
noise scale and the smoothing weight out analytically (`slipsolve.evidence`), so
there is no rank to choose. A fixed rank is not neutral for geometry: on a
synthetic Mw 7.7 with data on both sides, rms at rank 60 put a 60° fault at 57–58°
even with an exact forward model, because the data supported rank ~190. The
evidence put it within 0.3°. The `rank=` path is kept for comparison with older
results.

**Take the uncertainty from `bootstrap`, not from the evidence.** That synthetic
was run over 100 noise draws (`geometry_calibration/`, record of 2026-10-11).

- **iid evidence:** found the dip well, but the posterior it implies was 4–7×
  too narrow.
- **Evidence weighted by the true noise covariance:** exact when the forward
  model was exact. Against a 2 km truth, a 4 km mesh moved the dip +2.2° (0%
  coverage) and an 8 km mesh +12.6°.
- **The bootstrap:** re-estimates on synthetic data, the best model plus noise
  from a fitted model. Its intervals covered the truth 82–97% of the time in every
  case.

What the bootstrap reports:

- **`intervals()`:** each segment's continuous estimate (splined between grid
  points), its sd, and normal and basic 95% intervals.
- **`consistency()`:** the weighted estimate minus the iid one, against its
  spread under the noise model. `|z|` was of order one with an exact forward model
  and 5.7–47 when mesh error had biased the weighted posterior. A large `z` means
  the forward model is not good enough to weight by the covariance; trust the iid
  estimate and refine the mesh.
- **`noise="reml"`** (the default) fits each track's covariance to the best
  model's residual by restricted maximum likelihood (`covariance.fit_reml`). It
  fits on the samples, with the ramp projected out as the inversion projects it.
  On the synthetic it calibrated like the true covariance (96% against 95%),
  where the variogram route gave 83%.
- **Not in the interval:** mesh and other forward-model error. That is what the
  consistency check is for.

**A dip that changes with depth** is a tuple in the grid: `(70, 50)` runs from 70°
at the top of the fault to 50° at its base, linear in depth, and `(60, 80, 100)`
passes through 80° halfway down. The offset integrates `cot(dip)` exactly.
Candidates mix freely with planar dips (`dips=(70, (70, 50), (70, 60), 80)`). They
cost one assembly each, and each segment still hangs from the trace, so the block
assembly is unchanged. The bootstrap reports the share of replicates choosing
each profile rather than an interval. Whether data can resolve a change of dip
with depth has not been calibrated; plant one and recover it (`predict`,
`data=`) on your own coverage first.

What to know before reading a result:

- **It costs one Green's-function assembly per dip value**, not one per
  combination. Every engine builds a node's column from the elements around it,
  and a node column on each segment boundary keeps elements from straddling one,
  so any combination's design matrix is a sum of per-segment blocks. Measured at
  Venezuela scale (16,130 samples, 720 elements, layered tables), each assembly
  took 2.5–3.6 min and each combination 3 s. Memory is about one design matrix
  per dip.
- **Dips follow `FaultMesh.curved(uniform_dip=)`**: below 90 the fault dips
  toward the trace's left-hand side, above 90 toward its right. Check
  `mesh.dip_direction` on your trace.
- **The geometry steps but the slip does not.** Segments meet on the trace and
  step apart at depth (3.5 km at 20 km depth for 80° against 90°), while the
  boundary nodes are shared, so the slip stays continuous. Rebuild a chosen
  combination as one smooth surface before reporting a model on it.
- **The smoothing operator is built once**, on `reference_dip`, and used for
  every combination. With `rank=`, compare at a fixed rank, since one smoothing
  weight resolves a different number of parameters on a different geometry. The
  evidence needs neither: it integrates the weight out.
- **Give the grid a step finer than the spread you expect near the best.** The
  bootstrap splines between grid points, and warns when the spread is under a
  tenth of the step. A coarse sweep followed by a fine grid around its best is
  the cheap way: each new dip value costs one assembly.
- **The evidence has no bounds.** Its prior is Gaussian on every free
  parameter, and the pinned edges are removed outright. Polarity and slip bounds
  act only in the `rank=`/`smoothing=` path. The posterior-mean `Mw` and `peak`
  the evidence table reports are a guide, not a model.
- **Read the profile, not the winner.** Dip preferences are often shallow. The
  Venezuela sweeps rejected one dip direction clearly and barely ranked the
  angles within the other.
- **Test the coverage first.** With one-sided data a dip preference can be an
  artefact. `predict(dips, x)` and `run(data=...)` re-run the sweep on a planted
  geometry through the same samples; if it does not recover what was planted,
  the real result means little.
- `method="coordinate"` (automatic above 1000 combinations) re-chooses one
  segment at a time. It can stop at a combination that no single segment can
  improve but that is not the global best, so check it against an exhaustive run
  on a coarser grid when segments trade off.

## License

MIT. See `LICENSE`, which retains Xiaoyu Zou's original copyright as that
license requires.
