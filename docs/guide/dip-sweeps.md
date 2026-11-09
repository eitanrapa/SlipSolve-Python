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
halfway down. The offset integrates \(\cot(\text{dip})\) over depth exactly.
Profiles and planar dips mix freely, `dips=(70, (70, 50), (70, 60), 80)`, at one
assembly each. The bootstrap then reports the share of replicates choosing each
profile instead of an interval.

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
  chosen geometry as one smooth fault (`FaultMesh.curved` with `segments=` and
  `dips=`, or `surface=`) and solve it with `SlipInversion` before reporting a
  model.

## A rigidity contrast as a sweep axis

`DipSweep(..., contrast=contrast, ratios=(1, 1.25, 1.5, 2))` adds the ratio of a
[`RigidityContrast`](bimaterial.md) as a coordinate after the dips, one per interval
of the contrast. Dip and contrast both make the two sides of the fault move
differently and trade off, so sweep them together. A ratio costs no assembly
because the sweep uses the contrast's scaled mode, which is an approximation the
data can see: read the sweep's answer as where to look, and take the dip, the ratio
and the moment from the exact mode.
