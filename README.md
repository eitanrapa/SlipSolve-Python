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
  ramp and offset terms solved *jointly* with slip. A scene unwrapped in
  separate regions (islands, or a block cut off along the rupture so the
  unwrapper cannot cross it) gets one offset per region:
  `Observations.from_los(..., regions="connected")`, and no sample mixes two
  regions;
- **two solvers for the bounded problem**: scipy's `lsq_linear` on the stacked
  system (the default), or `solve(method="active_set")`, which solves its normal
  equations exactly by primal-dual active set. The second was 100-1,700x faster
  on the problems measured, and it reaches the minimum where `lsq_linear` can
  stop short of it. It hands the problem back to `lsq_linear`, with a warning,
  when its own answer cannot be trusted (an ill-conditioned block, a cycling
  active set);
- **pixel offsets as data** beside InSAR. A slant-range or along-track offset map
  is a track like any other, carrying its own unit vector (an along-track one is
  horizontal). Each track can take its own nuisance ramp, e.g.
  `ramp={"*": "cubic", "A061_az": "offset"}`, and should be weighted by its noise
  (`covariance={track: model}`), not by sample count. nisar_tools'
  `roff.offset_los_stacks` makes both stacks from a ROFF;
- **resolution analysis** — the model resolution matrix, effective rank,
  anisotropic averaging kernels, posterior covariance and regularisation bias;
- **measured data covariance** — an exponential variogram fitted per track from
  the far-field residual, or a REML fit on the samples themselves
  (`fit_reml`), with block-diagonal whitening;
- **geometry sweeps** — per-segment dips, or dips that change with depth,
  scored by the marginal likelihood with slip integrated out, with bootstrap
  intervals and a check for forward-model error (`DipSweep`);
- **bimaterial faults** — a fault between blocks of different rigidity, with
  the ratio changing along strike if need be. There is a cheap scaled mode and
  an exact interface solve (`Bimaterial`), and the ratio can be a sweep axis.

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

## Faults between blocks of different rigidity

A fault often separates two different rocks. The softer block moves more for the
same slip, so the contrast shows in the data as an asymmetry across the trace,
the same asymmetry a dip makes. `RigidityContrast` says which block is softer and
by how much, and where along strike that changes. `Bimaterial` wraps an engine
with it:

```python
contrast = RigidityContrast(trace, side="north", ratio=(1.45, 1.0), breaks=(55e3,))
model = SlipInversion(mesh, obs, engine=Bimaterial(contrast), basis="node").solve(...)
exact = Bimaterial(contrast, exact=True)               # homogeneous half-space
tables = RemainderTables.compute(velocity_model)        # once per crust, ~6 min
exact = Bimaterial(contrast, engine=LayeredTDE(tables), exact=True)   # layered
sweep = DipSweep(trace, obs, ..., contrast=contrast, ratios=(1, 1.25, 1.5, 2, 3))
```

`ratio` is `mu_other / mu_side`, so above 1 the named block is softer. `side` is
`"left"`/`"right"` of the trace as it runs, or a compass word. With `breaks`
(arc lengths in metres) there is one ratio per interval. The example has a
contrast west of 55 km and none east of it.

**The model.** The material boundary is the fault surface itself, continued down
dip and past the trace's ends. The two blocks are one rigidity model scaled by a
constant on each side: the same layering and Poisson's ratio, a different level.
`reference` names the block the velocity model describes (by default the other
one).

**A contrast that changes with depth.** `depth_breaks` (metres) cut the stepping
block into bands from the surface down, with one ratio per interval and band:

```python
shallow = RigidityContrast(trace, side="north", ratio=[[2.0, 1.0]], depth_breaks=(6e3,))
```

That is the north block twice as soft in the top 6 km only. Each band is still a
scaled copy of the reference, so the exact mode holds: it meshes a horizontal
boundary at each break, from the fault out across the stepping block. In the
scaled mode, and in the moment, each element takes its own band's ratio.

- ⚠️ **The horizontal boundary must reach past the data**, as the across-strike
  ones must (`extend["cross"]`, 4 depths by default). Cut 1 depth out with data to
  3, it read correlation -0.41 against the 2-D solution.
- **Lay the fault's rows on the breaks.** An element crossing one takes its
  centroid's band, with a warning.
- ⚠️ **Keep a break off a layer interface.** The boundary would then lie on a
  material interface, where the layered remainder is singular.
- **`DipSweep` refuses it.** It sweeps one ratio per interval along strike.

Checked against 2-D finite elements at mid-length of a 400 km fault dipping 60°
(4 depths of reach): the contrast's effect is 1.03 of the 2-D one for the top
7.5 km only (correlation 0.997), 1.00 everywhere, 0.93 below 7.5 km only. That last
one was 0.83 on a 165 km fault: a deep contrast is the slower to reach 2-D.

Not built: more than two blocks across strike (it would fit the exact form the
same way). Outside the exact form: different Poisson's ratios on the two sides (the
moduli are no longer proportional), and a contrast varying smoothly with depth,
except as steps.

**Two modes.** Within that model the field has an exact form: each block's own
scale times the field of a discontinuity on the boundary, which the exact mode
solves for.

- **Scaled (`exact=False`)** is that form's first term. Each sample's row is
  scaled by its block, and each slip column by the harmonic mean of the two
  rigidities, which gives `2r/(1+r)` on the named side and `2/(1+r)` on the other.
  It costs nothing, works with any engine, and is what the Venezuela scripts used
  (2026-10-07). It is exact only for strike-slip on a vertical fault in 2-D.
- **Exact (`exact=True`)** solves for the discontinuity on the whole boundary. That
  means the fault, its continuation, and at each break an across-strike boundary
  in the block that steps. It needs every element's field at depth, with opening
  as well as slip. The half-space engine gives that, and so does `LayeredTDE` for
  a layered crust (below). EDGRN's tables (`LayeredPointSource`) give neither.

What the scaled mode gets wrong:

- **In 3-D, even on a vertical fault.** The exact discontinuity opens the fault
  near the rupture ends, and an opening lifts the ground, which LOS sees. Measured
  on the Panama synthetic (Mw 7.7, two-sided, ratio 2, noise-free, scored by a
  scaled-mode sweep given exact-mode data; `geometry_calibration/`):

  | fault | scaled-mode error | sweep's dip | sweep's ratio | potency |
  |---|---|---|---|---|
  | vertical | 30 mm rms, 11% of the contrast's effect | 95.0 (true 90) | 2.24 | x1.05 |
  | 60° | 96 mm rms, 44% | 64.6 (true 60) | 2.24 | x1.06 |

  The noise there was 17 mm. Given its own data, the same sweep found the truth
  exactly. On the layered Venezuela crust (the same synthetic, layered exact data,
  a layered scaled sweep): 21 mm (8%) and dip 94.2, ratio 2.26, potency x1.075 on
  the vertical fault; 101 mm (50%) at 60°, where the sweep happened to read the
  truth (60.07, 1.995, x0.975). ⚠️ So its bias is not an offset you can correct
  for: it moves with the crust and the geometry.
- **Near a change along strike, everything.** With the ratio stepping from 2 to 1
  at mid-length:
  - on a vertical fault its error was 112% of the contrast's effect (273 mm rms
    within 20 km of the step, 1.2 m at worst), and the sweep read the two ratios
    as 1.48 and 1.29 (true 2 and 1), potency x1.18;
  - at 60° its error was 177%, and the sweep read dip 65.8 and ratios 1.70 / 1.50,
    with twice the true potency.
  - On the layered crust: 100% and 175% of the effect. The sweep put the vertical
    fault's dip on its grid's edge (84) with ratios 1.18 / 0.97 and 108x the
    potency, and the 60° one at 65.0 with ratios 1.73 / 1.50 and 3.5x. With the
    smoothing weak, the slip oscillates to fit what the model cannot represent.

  On a smaller test fault its error was 73-120% of the effect out to 30 km from
  the step. A contrast that steps along strike needs the exact mode.
- **On a dipping fault in 2-D, the slip.** In 2-D, with the dip known, it
  recovered the true ratio at 80 to 50 degrees, but its slip is an effective one:
  - potency x0.95 at 80° and x0.83 at 50° (ratio 2, hanging wall softer);
  - x1.06 / x1.23 with the footwall softer (Mw -0.05 / +0.06 at 50°);
  - correcting the slip afterwards recovered only about two thirds of that.
- **A vertical boundary under a dipping fault** (the Venezuela 80° south runs'
  assumption) is a different model again. Side-scaling it reads a ratio of 2 as
  1.82 at 80° and 1.52 at 50°.

So the scaled mode is for exploring. Quote a dip, a ratio or a moment from the
exact mode.

What the exact mode was checked against:

- **The 2-D solution.** At mid-length of a 400 km fault dipping 50°, the contrast's
  effect came out 94% of the 2-D exact one, with correlation 0.998. The 2-D
  solution is screw dislocations and images, solved independently.
- **Its own interface conditions.** The field jumps by the slip across the fault
  and by nothing elsewhere.
- **Convergence.** At the default discretisation it is within 1-3% of the
  contrast's effect of one with 5.6x the elements, from 2 to 30 km from a step.
- **Close to a boundary** (within about an element of where it meets the surface)
  the field is a few percent off, because the discontinuity is constant per
  element.
- **Cost.** The interface has 2.5-3x the fault's elements. It is assembled densely,
  `(3N)^2` for `N` of them, and solved once per mesh, after which each Green's
  matrix costs one engine matrix over the interface. On the Panama synthetic (688
  fault elements, 2,996 samples) a half-space geometry took 4-8 min on the Intel
  Mac. Since 2026-10-12 the three slip components share one dislocation pass, which
  measured 2.5x faster on a 284-element interface.

**A layered crust.** `LayeredTDE` gives each element's field anywhere, at depth
included, on a layered crust:

```python
tables = RemainderTables.compute(velocity_model, workers=8)   # once per crust
exact = Bimaterial(contrast, engine=LayeredTDE(tables), exact=True)
```

- **How.** Near the fault, a triangle's layered field is the closed-form half-space
  dislocation of each piece of it, cut at the interfaces, with that layer's Poisson
  ratio. To that it adds the remainder: the layered point source minus the
  half-space one, tabulated by the package's own wavenumber-domain solver for
  strike-slip, dip-slip, CLVD and explosion sources at any depth. The subtraction
  happens before the Hankel transform, so the remainder is smooth at the source.
  The crust must be in homogeneous layers (a gradient raises).
- **Far from it, point sources.** Beyond 4 element sizes (`far=`), a surface
  receiver takes point sources of the whole tabulated layered field, by the same
  adaptive quadrature `LayeredPointSource` uses. Only the tables differ.
- **Nodal slip is integrated exactly.** A node's tent is a 1-D integral over its
  level sets, each a smaller triangle with constant slip. Projecting element means
  instead put single tents 9–60% off near the fault.
- ⚠️ **Surface receivers need fine tables in depth.** A point source's surface field
  needs about 100 m in source depth to interpolate to 1e-3 at a few km. So the
  tables carry a 2-D surface grid (100 m, 50 distances a decade) beside the coarser
  3-D one used at depth. Without it, both paths were 0.5–1.3% off at 5 element
  sizes.
- **Checked.** The solver is the analytic half-space to 2e-5 and reciprocal across
  layers to 1e-14. It agrees with EDGRN to 1e-6–1e-3 of the peak where EDGRN is
  reliable, at the surface and at 8 km depth. The layered exact mode is a 2-D
  finite-element solution (sharing no code with it) to 0.998 of the contrast's
  effect, correlation 0.9992. The half-space version of that check gives 0.995.
- **The half-space exact mode is no stand-in.** Layering moved the contrast's
  effect by 12–19% of itself on the Panama synthetic: 18–30 mm rms against the
  17 mm noise.
- **Cost.** Tables: 135 s with 6 workers for the Venezuela crust (5 layers; 3-D
  part 172 depths to 200 km, 0.4 GB; surface part 734 depths). A Panama geometry,
  with four runs and other jobs sharing the machine: interface 261–445 s for
  1,730–2,098 elements, then 136–185 s per forward. Venezuela's model 3 west-vertical
  interface (5,234 elements): assembly 40 min, then 11 min for the reference field
  at 16,458 samples with the far path. Each element crossing an interface is cut
  into pieces, each a dislocation of its own.
- ⚠️ **Least accurate along an interface.** Next to one, the remainder grows like
  the source's image across it. The depth grid is graded toward every interface,
  but an element edge lying along one is the weakest place in the model.
- **Near the surface trace it is better than EDGRN's tables.** They read 0.36 m
  100 m from a vertical fault where this reads 0.91 m. The tables collapse inside
  about 1 km and this does not.

**Read with care:**

- **Where only one block has data, the ratio there cannot be measured.** It
  rescales the slip by up to `(1+r)/2`. On Venezuela east of the coast it is an
  assumption, and `breaks` is how to state it.
- **Dip and contrast trade off.** On Venezuela's one-sided data they were
  substitutes, and even with two-sided data a ratio's estimate moves with the
  dip's. Sweep them together (`DipSweep(contrast=, ratios=)`): a ratio costs no
  extra assembly, but the sweep uses the scaled mode, so its answer is a place to
  look, not a result. Give both grids room. A dip pinned on its grid's edge made the
  ratio's bootstrap interval 2-3x too narrow, and the bootstrap now warns when any
  coordinate's best is on an edge.
- **The moment uses the harmonic-mean rigidity** of the two blocks.
  `SlipModel.moment()` applies it, and a saved model keeps its contrast.

## License

MIT. See `LICENSE`, which retains Xiaoyu Zou's original copyright as that
license requires.
