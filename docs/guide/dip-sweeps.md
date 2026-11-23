# Dip sweeps

A wrong fault geometry is absorbed into the slip while the fit stays good, so a
geometry has to be chosen by comparing geometries on the same data.
[`DipSweep`](../api/geometry-search.md#slipsolve.sweep.DipSweep) cuts the trace into
along-strike segments, gives each a dip from a grid, and scores every combination
on the same samples with the same regularization, so the geometry is the only thing
that changes.

```python
from slipsolve import DipSweep

sweep = DipSweep(trace, obs, segments=4, dips=range(60, 121, 10),
                 max_depth=20e3, edge_length=4e3, engine=engine,
                 ramp="quadratic", polarity=(-1, 0, 0))
result = sweep.run(score="evidence")
print(result.table.head())        # one row per combination, best first
result.profiles()                 # each segment's score against its dip, others at the best
```

Dips follow `FaultMesh.curved(uniform_dip=)`: below 90 the fault dips toward the
trace's left-hand side, above 90 toward its right. Check `mesh.dip_direction` on
your trace (for example on `sweep.segment_mesh(0, 70)`). The keywords `engine`, `ramp`,
`velocity_model` and `frame_looks` mean what they mean to
[`SlipInversion`](inversion.md).

## How it stays affordable

Every engine builds a node's Green's function from the elements around it, and the
sweep puts a node column on every segment boundary, so no element straddles two
segments. The design matrix of any combination is then a sum of per-segment blocks,
and the sweep costs one Green's-function assembly per dip value, not one per
combination. Memory is about one design matrix per dip value used. Slip is nodal
and the boundary nodes are shared, so the slip stays continuous along strike while
the geometry steps from one segment to the next at depth.

Everything but the geometry is held fixed: the samples, the \((s, z)\) node layout,
the bounds, the ramp, the zero-slip edges and the smoothing operator, which is
built once on `reference_dip` (vertical by default).

With an expensive engine the assemblies are the cost. `DipSweep(workers=4)` splits
each block's rows over four processes, each with its own copy of the engine; a
`LayeredTDE` needs its tables loaded from a file, and `sweep.close()` stops the
processes. `cache=` names a directory that keeps every block assembled, in a file
named by the segment's mesh, the samples and the engine, so a repeated run reads it
back. That includes `optimize`'s finite-difference geometries, which is most of
the files: one block per segment and geometry.

## Two ways to score

- **`score="evidence"`** ranks geometries by their marginal likelihood (Yabuki &
  Matsu'ura 1992; Fukahata & Wright 2008): the slip, the ramp, the noise scale and
  the smoothing weight are integrated out analytically
  ([`GeometryEvidence`](../api/geometry-search.md#slipsolve.evidence.GeometryEvidence)),
  so no smoothing weight or rank has to be chosen. `covariance=` weights it by a
  noise model.
- **`rank=` or `smoothing=`** (with `score="rms"` or `"weighted_rms"`) ranks the
  bounded solve's misfit at a fixed effective rank or weight. One smoothing weight
  resolves a different number of parameters on a different geometry, so compare at
  fixed rank. A fixed rank is not neutral either: well below the rank the data
  support, it can bias the preferred dip by degrees even with an exact forward model.

The evidence knows nothing of bounds or polarity: its prior is Gaussian on every
free parameter, and the zero-slip edges are removed outright. The posterior-mean
\(M_w\) and peak slip in its table are a guide, not a model.

## Uncertainty from the bootstrap

The evidence ranks geometries well, but the posterior width it implies is too
narrow to use as an interval when the noise is treated as independent. Take the
uncertainty from a parametric bootstrap instead:

```python
boot = sweep.bootstrap(result, noise="reml", draws=100)
boot.intervals()       # per segment: estimate, sd, normal and basic 95% intervals
boot.consistency()     # noise-weighted minus iid estimate, against its spread
```

The best combination's model predicts the data, `draws` synthetic data sets add
noise to it (fitted by REML on the residual per track by default, or a
`CovarianceModel` or `{track: model}` you supply), and every evaluated combination
is re-scored on each. A segment's estimate is splined between grid points.

- `consistency()` compares the noise-weighted and the iid estimates. A difference
  large against its spread means the forward model (usually the mesh) is not good
  enough to weight by the covariance; trust the iid estimate and refine the mesh.
- Mesh and other forward-model errors are not in the intervals.
- Give the grid a step finer than the spread you expect near the best; the bootstrap
  warns when the spread is under a tenth of the step, and when a best value sits on
  the edge of its grid. A coarse sweep followed by a fine grid around its best is
  cheap, since each new dip value costs one assembly.

## Dips that change with depth

A tuple in the grid is a dip that varies linearly with depth: `(70, 50)` runs from
70° at the top of the fault to 50° at its base, and `(60, 80, 100)` passes through 80°
halfway down. `DipSweep(knot_depth=)` moves the deepest dip up from the base; below it
the dip holds. The offset integrates \(\cot(\text{dip})\) over depth exactly.
Profiles and planar dips mix freely, `dips=(70, (70, 50), (70, 60), 80)`, at one
assembly each. The bootstrap then reports the share of replicates choosing each
profile instead of an interval.

## Continuous profiles

[`DipSweep.optimize`](../api/geometry-search.md#slipsolve.sweep.DipSweep.optimize) refines a
sweep's answer without a grid, and lets a segment's dip change with depth without listing
profiles:

```python
result = sweep.run(score="evidence")
planar = sweep.optimize(start=result)             # one dip per segment, between grid values
curved = sweep.optimize(knots=2, start=planar)    # each segment's dip linear in depth
curved.table          # segment, knot, depth_km, dip, gradient, on_bound
curved.tears          # per break between segments: how far apart they lie at depth
boot = sweep.bootstrap(curved, noise="reml", draws=100)
boot.intervals()      # per knot: estimate, sd, normal and basic 95% intervals, on_bound, pinned
```

A segment with `knots=k` has `k` dips at evenly spaced depths from the top of the fault to
`knot_depth`, linear in depth between them, the same geometry as a `k`-entry tuple in the
grid.
`knots` can also be a list, one count per segment, and `fix={segment: geometry}` holds
segments.

The search is a trust region on the design linearised in the knots. A knot's derivative
comes from two assemblies of its own segment, and the evidence of the linearised design is
maximised with its analytic gradient. A step is kept when the assembled design confirms the
gain. An iteration costs about two segment assemblies per knot, so a profile with a few
knots costs a few dozen assemblies in all.

- It is local: start it from a sweep (`start=result`, the default runs one), and add knots
  one at a time (`start=` the previous fit).
- Judge a knot by its bootstrap interval, not by the gain in `log_evidence`. Every knot
  gains something, and with correlated noise treated as independent the gains are large
  even on a planar fault.
- Put the deepest knot near the base of the slip with `DipSweep(knot_depth=)`. A knot
  below the slip has little to fit: noise moves it, the knots above trade off with it,
  and its bootstrap interval can come out too narrow.
- `bounds=(lo, hi)` limits every knot, by default to the grid's span; a knot that ends on
  a bound is flagged and warned about. Its bootstrap replicates stop on the bound too, so
  `intervals()` gives it no sd and a one-sided interval, open on the bound's side
  (`on_bound`, and `pinned`, the share of replicates stopped there). The other knots'
  intervals hold it on the bound.
- The bootstrap re-estimates every knot on each replicate by one Newton step on the
  linearised design, so replicates cost no assembly.
- Each segment is meshed at its own geometry, so two neighbours with different
  geometries tear apart below the surface where they meet. `fit.tears` reports each break's
  rms and largest gap over depth, and `sweep.tears(dips)` reports them for any combination.
  The evidence does nothing to keep the gap small. `optimize(continuity=2e3)` adds a prior
  that each break's rms tear is of order 2 km: the search maximises the log evidence less
  `0.5 * sum((rms tear / continuity) ** 2)`, and the bootstrap keeps the same prior.
- Set `continuity` against the mesh: it is the tear the prior lets through. In one
  real application with 4 km elements, without the prior most of the gain in evidence
  came from the tears. The optimum under a 1 km scale tied the reference geometry in
  cross-validation, and the one under 2 km lost to it. Start near a quarter of
  `edge_length`.

**How well a profile is resolved.** We tested this on a two-sided synthetic of a large
strike-slip rupture, with correlated noise and a dip shallowing from 75° at the surface to 45°
at 30 km:

- A planar dip per segment came back to about ±0.4°, with honest intervals.
- With two knots, the deepest near the base of the slip, and the iid evidence, the dip came
  back to ±2–3° per knot. It was biased by about 2° toward more curvature, so its 95% intervals
  held the truth two times in three.
- Weighting the evidence by noise models fitted to the planar fit's residual (`covariance=`),
  on a mesh as fine as the truth's, narrowed that to ±0.6° per knot, with intervals of the
  right width. About 1° of extra curvature remained, part of it from holding the dip below the
  deepest knot. The intervals held the truth at 10 of 16 knots, and the best two-knot profile
  at 13 of 16.
- Weighted on a mesh twice as coarse, the intervals were as narrow but about 3° off. The
  planar fit's `consistency()` flagged that mesh in most noise draws and the finer one in none,
  so check it before trusting weighted intervals.
- The dip at mid-depth and the fault's position at depth were better determined than the
  knots themselves.
- Three knots locked onto kinked geometries that fit the correlated noise, with misleadingly
  narrow intervals. Weighting the evidence prevented the lock, but even on the fine mesh the
  three knots stayed uncertain by 1–5°.

## From a fit to a production sheet

`fit.sheet(name)` builds the
[`Sheet`](../api/production.md#slipsolve.production.Sheet) a production run takes,
on the sweep's trace, with the fitted dips. Each segment keeps its profile between
its breaks, and across each break the dip changes linearly over `blend` metres
either side (3 km by default), so the run meshes one surface where the sweep's
segments tore. `fit.profile()` is the same function of arc length and depth, for a
sheet built by hand.

```python
sweep = DipSweep(trace, obs, segments=5, dips=range(90, 161, 10), engine=engine,
                 normal_smoothing=10e3, workers=4, cache="blocks")
fit = sweep.optimize(knots=2, start=sweep.run(score="evidence"), continuity=1e3)
sheet = fit.sheet("main", blend=3e3)       # into Production(..., [sheet, ...], ...)
```

- Sweep the surface the run will mesh. A `Sheet` moves its nodes along the trace's
  normal smoothed over `normal_smoothing` (10 km by default), and `FaultMesh.curved`
  along the local normal; give `DipSweep` the sheets' value. `fit.sheet` keeps the
  sweep's, and carries the breaks onto the sheet's own arc length by position.
- Score what you adopt on that surface. The evidence was computed for the stepped
  segments, and a gain there can come from the tears. Rebuild the proposal with
  `fit.sheet` and cross-validate it one part of the fault at a time: swap each
  part's fitted profile into the reference geometry, and keep the parts that win on
  their own.

## One sheet of several

A sweep has one trace. To sweep one sheet of a model with several, hold the others
at a model's slip: take their prediction off the data and pass it as `data=` to
`run`, `evidence` and `optimize`. The bootstrap reuses the fit's data.

```python
others = SlipInversion(other_mesh, obs, engine=engine, basis="node")
held = obs.ds["los"].values - others.g @ other_slip     # strike-slip then dip-slip, per node
fit = sweep.optimize(knots=2, start=sweep.run(score="evidence", data=held), data=held)
boot = sweep.bootstrap(fit)
```

- Holding is a plug-in. Neither the other sheets' uncertainty nor their trade-off
  with the swept one enters the evidence, so where the sheets are close the
  intervals come out too narrow.
- Estimating them with the swept sheet is no better: the evidence has no slip
  bounds, and neighbouring sheets can trade large, opposite slips.
- The held slip was fit to all of the data, so a cross-validation of the result
  leaks through it.

## Reading the result

- **Read the profile, not the winner.** Dip preferences are often shallow near the
  minimum; a sweep can reject one dip direction clearly while barely ranking the
  angles on the other side.
- **Test the coverage first.** With data on one side of the fault, a dip preference
  can be an artefact of the coverage. Plant a geometry and recover it through the
  same samples: `sweep.predict(dips, x)` predicts data from parameters `x` on a
  combination, and `sweep.run(data=...)` re-runs the sweep on them. If the planted
  geometry is not recovered, the real result means little.
- **`method="coordinate"`** (automatic above `max_evaluations` combinations)
  re-chooses one segment at a time from the best uniform dip, and can stop at a
  combination no single segment can improve that is not the global best. Compare it
  with an exhaustive run on a coarser grid when segments trade off. `fix={segment:
  dip}` holds segments at a geometry.
- **The swept fault steps** at segment boundaries below the surface. Rebuild the
  chosen geometry as one smooth fault (`fit.sheet` for a production run,
  `FaultMesh.curved` with `segments=` and `dips=`, or `surface=`) and solve it
  before reporting a model.

## A rigidity contrast as a sweep axis

`DipSweep(..., contrast=contrast, ratios=(1, 1.25, 1.5, 2))` adds the ratio of a
[`RigidityContrast`](bimaterial.md) as a coordinate after the dips, one per interval
of the contrast. Dip and contrast both make the two sides of the fault move
differently and trade off, so sweep them together. A ratio costs no assembly
because the sweep uses the contrast's scaled mode, which is an approximation the
data can see: read the sweep's answer as where to look, and take the dip, the ratio
and the moment from the exact mode.
