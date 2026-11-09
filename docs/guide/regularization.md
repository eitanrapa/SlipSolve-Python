# Regularization and resolution

Surface displacement constrains slip on a fault about as well as a shadow
constrains a shape: deep elements barely register at the surface, neighbouring ones
produce nearly the same signal, and there are usually more parameters than
independent data. The smoothing weight decides how much of each estimate comes from
the data and how much from the regularizer, and it moves a slip model further than
the mesh, the sampling or the noise weighting do. This page covers how to choose
it, how to read what the data resolve, and how to let resolution size the mesh and
the sampling.

## Effective rank

The model resolution matrix of the damped problem,

\[
\mathbf{R} = \bigl(\mathbf{A}^\top\mathbf{A} + \mathbf{S}^\top\mathbf{S} + \mathbf{B}^\top\mathbf{B}\bigr)^{-1} \mathbf{A}^\top\mathbf{A},
\]

with \(\mathbf{A}\) the weighted design matrix and \(\mathbf{S}\), \(\mathbf{B}\) the
scaled smoothing and boundary rows, says how much of each estimate is data. Its
trace over the slip parameters is the **effective rank**: the number of slip
parameters the data pay for.

```python
from slipsolve import resolution_diagonal

diag = resolution_diagonal(inversion, smoothing=0.3)
print(diag.attrs["effective_rank"])
```

A smoothing weight means different things on different meshes, under different
weightings and with or without a covariance model; an effective rank does not. Quote
and compare choices in effective rank, and compare two meshes or two geometries at
matched effective rank, never at matched weight. (Pass the same `ss_ratio`,
`ds_ratio`, `boundary_ratio`, `sides` and `smooth_weighting` to the resolution
functions as to `solve`, so they describe the same problem.)

## Choosing the smoothing weight

Several criteria are available. On real data they can disagree by an order of
magnitude in effective rank, so look at more than one.

**Blocked cross-validation** (`blocked_cv_curve`) holds out contiguous blocks of
samples along strike, solves on the rest, and scores the prediction of each held-out
block. Leave-one-out criteria assume the held-out datum is independent of its
neighbours; InSAR noise is correlated over kilometres, so a held-out point is
predicted by its correlated neighbours and the criterion rewards models rough
enough to follow the noise. A block wider than the correlation length restores the
independence without relying on a fitted noise model.

```python
from slipsolve import blocked_cv_curve

curve = blocked_cv_curve(inversion, [0.03, 0.1, 0.3, 1.0, 3.0], n_folds=8,
                         polarity=(-1, 0, 0))
print(curve.attrs["best_smoothing"], curve.attrs["best_rank"], curve.attrs["interior"])
```

- Extra keyword arguments go to `solve`; use the same bounds and ratios as the
  final solve. The default solver (`method="active_set"`) makes the many solves cheap.
- **Check `interior`.** A minimum at the end of the sweep has run out of weights,
  not chosen one; extend the sweep.
- Folds are cut at equal widths along the samples' principal axis (`direction=` to
  set it), so their counts can be unequal; `attrs["counts"]` shows them. Too many
  folds leave a track with too few training samples for its ramp, which is refused.
  A warning says when blocks are narrower than a supplied covariance's range.
- With ramps, `cv_rms_profiled` re-fits each held-out block's own ramp before
  scoring, removing the ramp's extrapolation error from the score; its minimum is
  in `attrs["profiled_smoothing"]`.
- A fold that did not converge makes its weight's score meaningless (`converged`).
- One distant block can decide a close call. Before trusting a small margin, look
  at `cv_rms_spread` and `cv_rms_worst`, and try another fold count.

**Generalized cross-validation** (`gcv_curve`) is the closed-form approximation to
leave-one-out (Craven & Wahba 1979), and tends to under-smooth on correlated noise for
the reason above. **The L-curve** (`inversion.l_curve`, `plot_l_curve`) plots
roughness against misfit; its corner tends to over-smooth. **Checkerboard recovery**
on your own coverage and noise ([below](#calibration-on-your-own-data)) scores
against a known truth.

**A spatially variable prior.** `epic_prior(inversion, sigma_target)` returns
per-row smoothing weights that give every slip parameter the same posterior
standard deviation `sigma_target` (metres), the EPIC prior of Ortega-Culaciati et
al. (2021). It needs an inversion with a covariance model and the node basis; pass
its result as `solve(smoothing=...)`.

## Resolution

| Function | What it gives |
|---|---|
| `resolution_matrix(inversion, smoothing)` | the full \(\mathbf{R}\); row \(k\) is the averaging kernel of parameter \(k\) |
| `resolution_diagonal(inversion, smoothing)` | its diagonal per parameter, and the effective rank |
| `resolution_kernels(inversion, R)` | the width of each averaging kernel along strike and down dip, a length |
| `resolution_report(inversion, smoothing)` | the diagonal, the kernel widths and the data importance from one factorization |
| `data_importance(inversion, smoothing)` | the diagonal of the data resolution matrix: which samples pay for the model |
| `resolution_bias(R, model)` | \((\mathbf{R} - \mathbf{I})\,\mathbf{m}\): the smoothing's pull on this model |
| `bound_fraction(model)` | the fraction of slip parameters on a bound |

The diagonal largely says how hard the problem is damped; the kernel width is the
quantity that is a length, the shortest wavelength an estimate can separate. Draw
it on the slip map with `plot_slip(model, kernels=report)`.

All of these describe the **unbounded** damped problem, while `solve` is bounded.
They are a discretisation guide and an approximation to the solved model's
uncertainty, which is good only when few parameters sit on a bound
(`bound_fraction`).

## Uncertainty

```python
from slipsolve import posterior_covariance

sigma = posterior_covariance(inversion, smoothing=0.3,
                             noise={"asc": cov_asc, "desc": cov_desc}, model=model)
```

`kind="propagated"` (the default) is the estimator's variance under data noise:
what repeated observation would scatter. `kind="posterior"` is the Bayesian
posterior if the smoothing is taken as a prior you believe; it adds the prior's
own contribution and is much wider. A noise model is required: either the
inversion's own `covariance`, or `noise={track: CovarianceModel}`, which gives the
sandwich form for a solve weighted without one. With `model=`, the function refuses
if too many parameters sit on their bounds. Summarise sigma by its mean or maximum:
at light damping it spans orders of magnitude, and its median describes only the
well-resolved bulk. The covariance does not include the regularization bias
(`resolution_bias`), which is usually the larger error.

## Adapting the mesh and the sampling

Resolution can size the mesh (Barnhart & Lohman 2010) and place the samples
(after Lohman & Simons 2005):

- `slipsolve.resolution.kernel_element_size(inversion, smoothing, fraction=0.5)`
  sizes each element at a fraction of its averaging-kernel width (two elements per
  resolvable feature at 0.5). `target_element_size(mesh, resolution)` sizes from the
  resolution diagonal instead.
- `adapt_layout(mesh, size, structure="rows")` turns a per-element size into a new
  layout, with depth levels (`"rows"`) or fully unstructured
  (`"unstructured"`); `FaultMesh.from_layout` embeds it.
- `resolution_gain(model, los_stack, resolution)` is a field marking where a sample
  would most improve resolution, usable as `Observations.from_los(field=...)`.

`iterate_sampling` runs the loop of Wang & Fialko (2015): sample coarsely, invert,
predict each scene from the model, re-sample on the prediction, and repeat until
the slip stops changing. With `mesh_adapt` it also re-sizes the mesh from the
resolution each round, so one resolution matrix drives both the mesh and the
sampling:

```python
from slipsolve import iterate_sampling

obs, model, history = iterate_sampling(
    scenes, mesh, trace, frame,
    sample_kwargs={name: settings for name in scenes},
    inversion_kwargs={"ramp": "quadratic", "basis": "node"},
    solve_kwargs={"smoothing": 0.3, "polarity": (-1, 0, 0)},
    mesh_adapt={"geometry": {"uniform_dip": 90, "edge_length": 3e3},
                "size_from": "kernel", "fraction": 0.5, "size_max": 10e3},
    sampling_field="model")
```

`sample_kwargs` holds the `Observations.from_los` settings per scene; round 0
samples the data coarsely (`width_min` tripled, or `coarse_kwargs`). `mesh_adapt`
must say what the fault is: `geometry` holds the keywords `FaultMesh.from_layout`
needs (`uniform_dip=`, `bottom_trace=`, `edge_length=`, ...). Its other keys choose
the element sizes: `size_from="kernel"` with `fraction` (`kernel_element_size`), or
the default `"diagonal"` with `target`, `gain` and `max_ratio`
(`target_element_size`), clamped to `size_min` and `size_max`. The element budget
stays at the starting mesh's count unless `budget` says otherwise, and `structure`
chooses rows or an unstructured layout. Bound the element size: a loop free to
spend its budget along strike can collapse the depth levels. The loop finishes on a
fixed mesh by default (`rounds`), because convergence cannot be measured while the
mesh is still moving. `sampling_field` chooses what the quadtree splits on: the
predicted field (`"model"`), the resolution gain (`"resolution"`) or both
(`"both"`). `history` records the sample count, fit and slip change per round.

The smoothing weight still has to be chosen at each mesh; refine the mesh only as
far as the effective rank you can justify.

## Calibration on your own data

[`slipsolve.calibrate`](../api/resolution.md#slipsolve.calibrate) runs a
checkerboard recovery test on your own geometry, coverage and measured noise: a
checkerboard is planted on the mesh, forward-modelled through the real look vectors
and coverage masks, given noise drawn from the measured covariance, and inverted
with the same estimator as the real data. It reports what element size, layout and
damping to use, and what wavelength of slip you can claim to have resolved.

Start with the free part, which is pure geometry:

```python
from slipsolve.calibrate import discretisation_ceiling

for row in discretisation_ceiling(mesh, cells=(20e3, 40e3, 60e3)):
    print(row)
```

It says which checkerboard wavelengths the mesh can carry at all, so that a
discretisation failure is never mistaken for a resolution limit. Then
`calibration_plan(mesh)` lays out a staged search (wavelengths and an effective-rank
ladder first, then noise seeds, then mesh budgets and layouts) and returns its cost
before anything runs: `plan["cases"]` lists the loops, `plan["n_loops"]` and
`plan["n_solves"]` count them. `run_case` runs one case: plant, forward-model,
invert through `iterate_sampling`, and score every rung of the rank ladder. The
protocol's rules are in the module's documentation; the main ones are that the
target alternates in magnitude within the polarity bound rather than in sign, that
noise is mandatory and drawn from the measured covariance, that recoveries are
compared at matched effective rank, and that the coverage mask reaches the forward
model through the look vectors.
