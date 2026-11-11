# slipsolve

**slipsolve** inverts geodetic surface displacement for the distribution of slip
on a fault. It takes gridded InSAR line-of-sight (LOS) products, pixel offsets and
GNSS displacements, a mapped fault trace and a model of the fault's dip, and solves
a bounded, regularized least-squares problem for slip on a triangulated fault
surface. It is written for geodesists and seismologists who model earthquakes from
geodetic data and want the choices that shape a slip model (mesh, sampling,
smoothing, nuisance ramps, elastic structure) to be explicit and testable.

The package is a Python translation of, and extension to,
[SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve) by Xiaoyu Zou (MATLAB),
which descends from [SlipSolve](https://github.com/ZeyuJin/geodetic_inversion) by
Zeyu Jin. The distribution is called `slipsolve-python`; the import name is
`slipsolve`. See [Relation to SlipSolve-Curve](slipsolve-curve.md).

## The workflow

```text
FaultTrace ──> LocalFrame ──> FaultMesh ──┐
                                          ├──> SlipInversion ──> SlipModel
LOS grids, offsets, GNSS ──> Observations ┘
```

1. A [`FaultTrace`](guide/faults-and-meshes.md#the-fault-trace) is read from KML or
   a lon/lat table.
2. A [`LocalFrame`](guide/conventions.md#one-local-frame) puts every object in one
   metric east/north/up frame.
3. A [`FaultMesh`](guide/faults-and-meshes.md#building-a-mesh) triangulates the fault
   surface: vertical, or dipping and curved, with slip per element or per node.
4. [`Observations`](guide/observations.md) quadtree-sample each LOS grid, and take
   pixel offsets and GNSS as further tracks.
5. A [`SlipInversion`](guide/inversion.md) builds the Green's functions once, then
   solves with smoothing, zero-slip edges, bounds, polarity and per-track ramps.
6. The resulting [`SlipModel`](guide/inversion.md#the-slip-model) reports the fit and
   the moment, predicts displacement anywhere, exports tables and grids, and plots.

## Features

- **Fault geometry.** Vertical or dipping faults built from a surface trace; a dip
  per segment, a digitised bottom trace, or a fitted surface; structured, graded or
  fully unstructured meshes; several faults in one inversion.
- **Elastic models.** Triangular dislocations in a homogeneous half-space (Nikkhoo
  & Walter 2015); a layered crust through the package's own wavenumber-domain tables
  (`RemainderTables`, `LayeredTDE`), exact near the fault and at depth; fully 3-D
  heterogeneity on voxels
  (`HeterogeneousEngine`); a rigidity contrast across the fault (`Bimaterial`);
  topography by the elevation shift; and Earth's curvature and gravity as a correction
  for the far field (`Curvature`).
- **Data.** Quadtree sampling of gridded LOS products, optionally driven by a
  model's predicted field (Wang & Fialko 2015) or by model resolution; one offset
  per separately unwrapped region; pixel offsets and GNSS as tracks of their own;
  spatially correlated noise models fitted by variogram or REML.
- **Regularization.** Smoothing (element differences, or a Laplace–Beltrami
  operator for nodal slip), zero-slip edges, bounds and slip polarity, and
  unbounded per-track ramps (offset to cubic) solved jointly with the slip.
- **Choosing the smoothing weight.** Effective rank, blocked cross-validation,
  GCV and the L-curve, plus checkerboard recovery tests on your own coverage and
  noise (`slipsolve.calibrate`).
- **Resolution and uncertainty.** The model resolution matrix and averaging-kernel
  widths, data importance, regularization bias and posterior covariance; a joint
  loop that adapts the mesh and the sampling to the resolution.
- **Geometry search.** Per-segment dip sweeps, including dips that change with
  depth, scored by misfit at fixed effective rank or by the marginal likelihood,
  with bootstrap intervals (`DipSweep`, `GeometryEvidence`).
- **Production runs.** The whole recipe in checkpointed stages: meshes adapted to the
  resolution, samples redrawn from the model, each data set weighted by its own noise, and
  the rank chosen by blocked cross-validation averaged over shifted partitions
  (`Production`, `NormalSystem`, `choose_rank`).
- **Postseismic and interseismic.** Slip that changes with time from dated series
  (`TimeDependentInversion`); rigid blocks plus back-slip with coupling bounds from
  velocities (`InterseismicInversion`); and signals beside the slip: viscoelastic
  relaxation of a layered crust (`RelaxationTables`) and the far field of a regional
  block model (`RegionalModel`).
- **Output.** Seismic moment with depth-dependent rigidity, surface displacement
  grids (GMT `.grd`), element and node tables, a single-file save format, and
  figures.

## Where to go next

- [Installation](installation.md)
- [Getting started](getting-started.md): a complete synthetic inversion that runs in
  under a minute.
- [Conventions](guide/conventions.md): signs, frames and units. Read this before
  feeding the package real data.
- The [user guide](guide/faults-and-meshes.md) and the [API reference](api/index.md).
- Source code and issues:
  [github.com/eitanrapa/SlipSolve-Python](https://github.com/eitanrapa/SlipSolve-Python).
