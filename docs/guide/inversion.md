# The inversion

[`SlipInversion`](../api/inversion.md#slipsolve.inversion.SlipInversion) pairs a mesh
with a set of observations, builds the Green's matrix once, and solves for slip at
any number of smoothing weights. Each solve returns a
[`SlipModel`](../api/inversion.md#slipsolve.inversion.SlipModel).

```python
from slipsolve import SlipInversion

inversion = SlipInversion(mesh, obs, engine=engine, ramp="quadratic", basis="node",
                          velocity_model=crust)
model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
```

| Argument | Meaning |
|---|---|
| `engine` | the Green's functions; [`HalfSpaceTDE()`](elastic-models.md) by default |
| `ramp` | per-track nuisance columns, below |
| `basis` | `"element"` (constant slip per triangle, default) or `"node"` (continuous piecewise-linear slip) |
| `velocity_model` | the rigidity used for the moment; 30 GPa without it |
| `covariance` | a [noise model](observations.md#noise), one for every track or `{track: model}` |
| `sees` | for a [joined mesh](faults-and-meshes.md#several-faults), which faults each track measures |

## The problem solved

The slip vector \(\mathbf{m}\) (strike-slip, then dip-slip, then any ramp
coefficients) solves the bounded least-squares problem

\[
\min_{\mathbf{m}} \;
\bigl\| \mathbf{W} (\mathbf{G}\mathbf{m} - \mathbf{d}) \bigr\|^2
+ \bigl\| (\lambda / n_L)\, \mathbf{L}\mathbf{m} \bigr\|^2
+ \bigl\| \mathbf{B}\mathbf{m} \bigr\|^2
\quad \text{subject to} \quad \mathbf{l} \le \mathbf{m} \le \mathbf{u},
\]

where \(\mathbf{G}\) is the Green's matrix with the ramp columns beside it,
\(\mathbf{W}\) the observation weights (or \(\mathbf{C}_d^{-1/2}\) with a covariance
model), \(\mathbf{L}\) the smoothing operator with \(n_L\) rows, \(\lambda\) the
smoothing weight, \(\mathbf{B}\) the zero-slip edge rows, and \(\mathbf{l}\),
\(\mathbf{u}\) the bounds. Dividing by \(n_L\) keeps a weight's meaning as the mesh
is refined.

## Smoothing

`smoothing` is \(\lambda\): a scalar, or one weight per row of the smoothing
operator for a spatially variable prior (see `epic_prior` in
[Regularization](regularization.md)). `ss_ratio` and `ds_ratio` scale the
strike-slip and dip-slip blocks separately; the default `ds_ratio=3` smooths
dip-slip harder, the usual choice on a strike-slip fault.

- With the element basis, \(\mathbf{L}\) penalises the slip difference across each
  shared edge, one row per edge. `smooth_weighting="length"` divides each row by the
  distance between the two centroids, a gradient rather than a difference, which a
  mesh with widely varying element sizes needs.
- With the node basis, \(\mathbf{L}\) is a Laplace–Beltrami operator on the fault
  surface, already weighted by area and shape.

How to choose \(\lambda\) is the subject of
[Regularization and resolution](regularization.md). It matters more than any other
choice: the smoothing weight moves a slip model much further than the mesh does.

## Zero-slip edges

`sides` (default `("bottom", "left", "right")`) pulls slip toward zero on the named
edges of the mesh, with weight `boundary_ratio`. A mesh is a finite piece of a fault
that continues past it, and without these rows the inversion tends to put slip on
the edges, where the data constrain it least. The free surface (`"top"`) is
normally left out, since surface rupture is a real outcome.

## Bounds and polarity

`strike=(-10, 10)` and `dip=(-10, 10)` bound each component in metres. `polarity`
fixes the sense of slip: `(-1, 0, 0)` keeps strike-slip non-positive, which is
right-lateral (see [Conventions](conventions.md#the-sign-of-slip)). Fixing the
sense of a fault whose sense is known removes a large part of the null space and
the alternating-sense solutions that smoothing alone still allows. Ramp
coefficients are never bounded.

## Ramps and offsets

An unwrapped interferogram has an arbitrary constant and usually an orbital or
ionospheric surface on top. Without somewhere to put them the inversion absorbs
them as broad, deep, fictitious slip. `ramp` adds unbounded, unregularized columns
per track:

| `ramp` | Columns per track |
|---|---|
| `"none"` | 0 |
| `"offset"` | 1 |
| `"linear"` | 3 (offset and two gradients) |
| `"quadratic"` | 6 |
| `"cubic"` | 10 |

A dict gives each track its own kind, with `"*"` for the rest, for example
`{"*": "cubic", "asc_rg": "offset"}` for interferograms beside pixel offsets.
A track sampled with `regions=` gets one offset per region, sharing the higher
terms.

The ramp is solved jointly with the slip, not removed beforehand, so the fault model
competes for the near-field structure and the polynomial never has to extrapolate
under a masked signal. The competition is real, though: a cubic across strike looks
much like the far field of a long strike-slip fault, and a ramp that has absorbed
part of the earthquake still reports an excellent fit. Three diagnostics help:

- `ramp_content(obs)` reports how much of the *data* each degree explains. It has no
  fault model, so it overstates the ramp; read a large value as a prompt to compare
  degrees, not as a ramp.
- `ramp_leakage(inversion)` reports how much of the *fault's* signal each degree
  could absorb.
- `ramp_sweep(inversion, kinds=("linear", "quadratic", "cubic"), **solve_kwargs)`
  solves at each degree and warns when the moment falls while the fit improves,
  which is the signature of a ramp eating the earthquake.
  `inversion.with_ramp(kind)` shares the Green's matrix, so each degree costs one
  solve.

The ramp is reported separately from the slip: `model.ramp_prediction` is the
nuisance surface and `model.fault_prediction` what the slip alone explains.
Compare `data - ramp_prediction` with `fault_prediction`, not the data with the full
prediction.

## Solvers

`solve(method=...)` offers three solvers of the same bounded problem:

- `"lsq_linear"` (default): scipy's trust-region solver on the stacked system,
  applied as an operator, so the sparse smoothing rows are never densified.
- `"active_set"`: the normal equations solved exactly by a primal-dual active-set
  method (`box_qp`). The data block of the normal equations is formed once per
  inversion and each solve starts from the last, so a sweep over weights is much
  faster. It reaches the minimum where `lsq_linear` can stop slightly short of it.
  Because the normal equations square the condition number, an answer that is badly
  conditioned or fails its optimality check is refused with a warning and the
  problem is re-solved by `lsq_linear` (by `"lsq_linear_dense"` for up to 1000
  parameters). `model.options["method"]` records what ran.
- `"lsq_linear_dense"`: the stacked system as a dense matrix with exact inner
  solves. Accurate, but its cost grows with rows times parameters squared, so it is
  for small problems.

Check `model.converged`. A solve stopped by the iteration cap returns a plausible
model whose statistics mean nothing, and the text exports refuse it; raise
`max_iter` or the smoothing weight.

`inversion.l_curve(weights, **solve_kwargs)` solves at each weight and returns a
Dataset of misfit, roughness, variance reduction, peak slip and \(M_w\) per weight,
with the models.

## The slip model

A [`SlipModel`](../api/inversion.md#slipsolve.inversion.SlipModel) holds the
solution and its fit:

| Attribute | Meaning |
|---|---|
| `strike_slip`, `dip_slip` | the slip parameters, per element or per node (metres) |
| `element_slip`, `nodal_slip()` | slip per element `(n_elements, 2)` or per node `(n_nodes, 2)`, whatever the basis |
| `slip_magnitude`, `max_slip` | \(\sqrt{s_s^2 + s_d^2}\) per parameter, and its maximum |
| `ramp`, `ramp_labels` | the nuisance coefficients and their names (`"asc:offset"`, `"asc:dx"`, ...) |
| `data`, `prediction`, `residual` | per sample; the residual is observed minus modelled |
| `ramp_prediction`, `fault_prediction` | the prediction split into the nuisance surface and the slip's part |
| `variance_reduction`, `rms_misfit` | fit, on the unweighted residual |
| `chi_square`, `reduced_chi_square` | fit in the metric the solver minimised; `None` without a covariance model |
| `roughness` | rms of the smoothing operator applied to the model |
| `converged` | whether the solver stopped on its tolerance |
| `track_residual(name)` | one track's residual |

### Moment

`model.moment()` is \(\int \mu\, |\mathbf{s}|\, dA\) in N·m, and
`model.moment_magnitude` the corresponding \(M_w\). The rigidity is the inversion's
velocity model, averaged exactly over each element (layer steps included), or 30 GPa
without one; `moment(shear_modulus=...)` overrides it with a scalar, one value per
parameter, or a `VelocityModel`. For nodal slip the magnitude is integrated over
sub-triangles of each element. `moment(rule="lumped")` gives the simpler sum of
rigidity times lumped area times slip at each parameter's depth, which is what the
node table written by `to_vertex_text` partitions; on a layered crust it reads
somewhat high. With a [rigidity contrast](bimaterial.md) the moment uses the
harmonic mean of the two blocks' rigidities.

### Forward predictions

```python
los = model.forward(x, y, look=(look_e, look_n, look_u))   # LOS at any points
enu = model.forward(x, y)                                   # (n, 3) east, north, up
grid = model.surface_displacement(spacing=1000.0)           # ux, uy, uz on a grid
```

`surface_displacement` returns an xarray Dataset of east, north and up displacement
on a regular grid in the local frame, NaN within about an element size of the trace,
where a discretised fault means nothing.

### Saving and exporting

```python
model.save("model.slip.zip")                 # one file: model, mesh, observations
model = SlipModel.load("model.slip.zip")

model.to_text("slip_model.txt")              # SlipSolve-Curve's ten-column element table
model.to_vertex_text("tables/")              # vert_nodes.txt and vert_elements.txt
model.to_grd("grids/", spacing=1000.0)       # ux.grd, uy.grd, uz.grd in lon/lat
ds = model.to_dataset()                      # xarray, per element and per sample
```

`save` writes a zipped Zarr store. The Green's matrix and any layered tables are
not saved, so a loaded model can be read, exported, plotted and forward-modelled
with the half-space engine, but not re-solved; build a new `SlipInversion` for that.
The element table has one row per element: id, lon, lat, depth, strike, dip,
strike-slip, dip-slip, area and shear modulus. The vertex tables describe the mesh
by its nodes and connectivity and report slip at the nodes. Text tables are refused
for an unconverged model.

### Figures

[`slipsolve.plot`](../api/plotting.md) draws the slip on the unrolled fault
(`plot_slip`, `plot_slip_edges`) or in 3-D (`plot_slip_3d`), the mesh (`plot_mesh`),
data, model and residual per track (`plot_fit`), the samples (`plot_samples`), the
L-curve (`plot_l_curve`) and coverage (`plot_coverage`). Each returns `(fig, ax)`.
`plot_slip(model, kernels=report)` adds the resolving length from a
`resolution_report`, so a map drawn at element scale does not suggest more detail
than the data support.
