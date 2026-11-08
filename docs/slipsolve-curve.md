# Relation to SlipSolve-Curve

slipsolve is a Python translation of, and extension to,
[SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve), Xiaoyu Zou's MATLAB
package for slip inversion on curved faults, which descends from
[SlipSolve](https://github.com/ZeyuJin/geodetic_inversion) by Zeyu Jin. It is
neither of those: the `-python` in the distribution name says which one you are
installing. SlipSolve-Curve is MIT licensed, and so is this package, whose licence
keeps SlipSolve-Curve's copyright notice.

## What carries over

The design follows SlipSolve-Curve: a fault built from a surface trace and a dip
per deep segment, joined by a smooth fitted surface (a port of the `gridfit`
gridder it uses); triangular dislocations (Nikkhoo & Walter 2015); layered Green's
functions from EDGRN tables, superposed as EDCMP does; quadtree sampling; smoothing,
zero-slip edges and bounds stacked into one bounded least-squares problem. Several
of its conventions can be reproduced exactly:

- `LocalFrame(..., ellps="clrk66")` reproduces its `ll2xy.m` projection;
- `FaultMesh.curved(segments=..., dips=...)` and `bias_w` correspond to its
  segment files, segment dips and `biasW`;
- `Observations.concat(normalize="count")` reproduces its track weighting;
- `LayeredPointSource(tables, tolerance=None, max_points=91)` reproduces its fixed
  91-point quadrature (`slipsolve.layered.REFERENCE_QUADRATURE_POINTS`);
- `SlipModel.to_text` writes its ten-column element table.

## What is new

**Adaptive quadrature for layered Green's functions.** The reference
implementations (EDCMP, and SlipSolve-Curve itself) integrate each triangle at a
fixed order, 91 points in every case. The order a point source needs depends on
how far the observation is from the element in units of the element's own size,
so here it is chosen per triangle and observation from an error model. Layered
assembly then costs about as much as the closed-form half-space, so there is no
computational reason to accept a uniform crust, which biases the split between
shallow and deep slip. The layered Green's functions are also evaluated at the
observations themselves, rather than on an intermediate grid and interpolated.

**A joint resolution loop.** Barnhart & Lohman (2010) size a mesh from model
resolution, and Lohman & Simons (2005) place samples by resolution; both keep the
two steps separate. Here one resolution matrix drives both, alternating
(`iterate_sampling(mesh_adapt=...)`). It is an efficiency method rather than an
accuracy one: at matched effective rank the smoothing weight, not the mesh,
decides the model.

**Beyond the original.** The package's own layered tables and engine
(`RemainderTables`, `LayeredTDE`), exact at the fault and at depth; nodal slip with
a Laplace–Beltrami smoother; per-track ramps solved jointly with the slip, and
offsets per separately unwrapped region; pixel offsets and GNSS as data; noise
covariance by variogram or REML; resolution, posterior covariance and blocked
cross-validation; checkerboard calibration; per-segment dip sweeps scored by the
marginal likelihood, with a bootstrap; bimaterial faults; 3-D heterogeneity on
voxels; and an exact active-set solver.

## Deliberate departures

These change numbers you might compare against a MATLAB run.

- **Parameterization by arc length.** The mesh is laid out in arc length along the
  trace and depth. The reference lays it out along a map axis (northing, or the
  axis of larger extent), which degenerates for a fault striking near 45° and
  spaces nodes unevenly along a trace that curves.
- **Winding from the trace.** Every triangle's normal is wound toward the trace's
  left-hand side, fixed in the \((s, z)\) plane. The reference tests each normal
  against a fixed axis (`orientTriFromLeft.m`), which has almost no margin on a
  fault striking near east-west. A triangle wound the wrong way flips its dip-slip
  column and leaves strike-slip unchanged.
- **Exact boundaries.** The edges of the fault (for zero-slip rows) are found by
  exact equality in \((s, z)\), which also works on unstructured meshes, rather than
  by layer indexing or by extreme map coordinates.
- **The smoothing operator.** For element slip, one row per shared edge, where the
  reference has one per ordered pair of neighbours, so every edge twice. Both divide
  the operator by its row count, so the same weight smooths \(\sqrt{2}\) times more
  strongly here. For nodal slip the operator is a Laplace–Beltrami operator on the
  fault surface. Smoothing weights are not directly comparable with a MATLAB run;
  compare effective ranks instead.
- **Track weighting.** The default `normalize="sqrt_count"` gives each track equal
  total influence; the reference's per-count weighting gives a track with twice the
  samples half the influence.
- **The quadtree** runs on the raster's own index rectangles with summed-area
  tables, so the look vectors are averaged over exactly the pixels of the data, and
  it splits on the within-cell standard deviation rather than a curvature measure.
- **Nuisance ramps.** The reference reserves columns for them (`add_col`) and never
  uses them; here they are solved jointly with the slip.
- **The surface row of a nodal model.** The reference's final nodal inversion
  represents the top row of elements by constant-slip half-space dislocations tied
  to the second level of nodes; here the nodal slip field, layered or not, reaches
  the surface.

Not ported: the reference's mode that reloads its own legacy mesh and point
products.
