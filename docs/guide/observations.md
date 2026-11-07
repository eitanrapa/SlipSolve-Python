# Observations

An unwrapped interferogram has millions of pixels and a fault model thousands of
parameters, so the inversion works on a decimated set of samples.
[`Observations`](../api/observations.md#slipsolve.sampling.Observations) holds them:
one row per sample, with its position in the local frame, its value, its unit
vector, its within-cell scatter, its weight and the name of the track it came from.
Gridded LOS products are quadtree-sampled into it; pixel offsets and GNSS enter as
further tracks.

## Gridded line-of-sight products

Any object with an xarray Dataset at `.ds` is accepted as a gridded product, if
the Dataset holds:

| Variable | Dimensions | Meaning |
|---|---|---|
| `los` | `(pair, y, x)` | displacement in metres, positive toward the sensor |
| `los_east`, `los_north`, `los_up` | `(y, x)` | the target-to-sensor unit vector |
| `x`, `y` | coordinates | a regular grid |

and either `attrs["epsg"]` (the projected CRS of `x` and `y`) or `attrs["frame"]`
(the grid is already in local-frame metres). Pixels that are NaN in `los` or in
the look vectors are treated as missing. [`LosGrid`](../api/observations.md#slipsolve.LosGrid)
wraps such a Dataset, and builds one from GMT grids:

```python
from slipsolve import LosGrid

asc = LosGrid.from_grd("los_asc.grd", "look_e_asc.grd", "look_n_asc.grd", "look_u_asc.grd",
                       units="m", sign=1, direction="ascending", look_direction="right")
```

`units` (`"m"`, `"cm"` or `"mm"`), `sign` and `look_convention` must be declared,
because no `.grd` file records them; see
[Conventions](conventions.md#line-of-sight). A geographic grid is reprojected to
the UTM zone of its centre, because the quadtree measures cell sizes in metres.

**Put every scene on one lattice first.** Quadtree cells are whole pixels halved at
their midpoint, so the cell sizes a scene can reach depend on its pixel size, and
two scenes at different postings sample differently under the same settings.
`resample_all` warps a set of scenes onto one lattice in the local frame:

```python
from slipsolve import resample_all

scenes = resample_all({"asc": asc, "desc": desc}, frame)   # 10 arc-second spacing
```

## Quadtree sampling

```python
from slipsolve import Observations

obs_asc = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                rms_min=0.006, width_min=1e3, width_max=20e3,
                                exclude_within=2e3)
```

A cell splits while the standard deviation of the displacement inside it exceeds
`rms_min` (metres), within the bounds `width_min` and `width_max` (metres). Each
surviving cell becomes one sample: the mean (or, with `stat="median"`, the median)
of its valid pixels, with the look vector averaged over the same pixels and
renormalised. Cells with less than `min_valid_fraction` of valid pixels are
dropped.

`exclude_within` drops samples closer than that distance to the trace. It is
required in practice: a dislocation solution is singular on the fault, and a
sample on the trace gives a non-finite Green's function, which the inversion
refuses with a message naming the offending samples. A cell should not straddle
the trace, so use at least `width_min / 2`.

### Measuring the sampling parameters

Each parameter is a statement about the data, and a wrong value produces a symptom
that looks like something else. In particular, an `rms_min` below the scene's noise
floor cannot stop the recursion: the scatter inside a cell does not shrink as the
cell shrinks, so every cell runs down to `width_min` and the samples mostly resolve
atmosphere. Measure them instead:

```python
from slipsolve import scene_report

report = scene_report(scenes["asc"], trace, frame, mesh=mesh)
settings = {k: report.attrs[k] for k in ("rms_min", "width_min", "exclude_within")}
obs_asc = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                **settings)
```

`scene_report` returns the coverage on each side of the trace along strike (the
quantity that most limits what the data can resolve, and which aggregate numbers
hide), and carries in `attrs` the far-field noise floor (`noise_floor`), the
settings derived from it, and a check of the look vectors' east component against
the declared pass direction. The noise floor is measured on blocks farther than
`min_distance` from the trace (four times the mesh's depth when a mesh is given),
so that they carry no deformation; pass a smaller `min_distance` for a scene that
does not reach that far. `cell_size_ladder` lists the cell sizes a scene's
quadtree can actually reach; [`plot_coverage`](../api/plotting.md#slipsolve.plot.plot_coverage)
and [`plot_samples`](../api/plotting.md#slipsolve.plot.plot_samples) show the
coverage and the samples.

### Sampling on a model

On real interferograms the observed field's scatter is mostly atmosphere. Wang &
Fialko (2015) quadtree a *predicted* interferogram instead, computed from a
preliminary slip model, and fill the cells from the data. `field=` does this:

```python
from slipsolve import model_rms_min, predicted_los

field = predicted_los(model, scenes["asc"])      # the model's LOS on the scene's grid
obs_asc = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                field=field, rms_min=model_rms_min(field),
                                width_min=1e3, exclude_within=2e3, refine_within=6e3)
```

Under `field=`, `rms_min` is a threshold on the model's own variation, not a
noise level, so take it from `model_rms_min` rather than from `scene_report`.
`refine_within` keeps cells near the trace at `width_min` whatever the model
predicts there, so a first model with little shallow slip does not coarsen the
sampling exactly where shallow slip would be constrained.
[`iterate_sampling`](regularization.md#adapting-the-mesh-and-the-sampling) runs the
whole loop: sample, invert, predict, re-sample, until the slip stops changing.

### Regions unwrapped apart

Each separately unwrapped region of an interferogram (an island, or a block cut
off along the rupture so the unwrapper cannot cross it) carries its own arbitrary
constant. With `regions="connected"`, or an integer label map of the grid's shape,
no cell mixes two regions, each sample records its `region`, and the inversion
solves an offset for each region instead of assuming the regions are tied.

## Combining tracks

```python
obs = Observations.concat([obs_asc, obs_desc])
```

`normalize` sets how a track's sample count affects its influence:

- `"sqrt_count"` (default): each row is scaled by \(1/\sqrt{n}\), so each track's
  total contribution to the normal equations is independent of how densely it was
  sampled;
- `"count"`: each row is scaled by \(1/n\), as SlipSolve-Curve does, which gives a
  track with twice the samples half the influence;
- `"none"`: rows keep their own weights. Use it with a noise covariance (below), and
  for GNSS, whose rows are weighted by their standard deviations.

`weights={track: factor}` scales named tracks on top of that.

## Pixel offsets

A pixel-offset map is a track like any other: its `los` is the displacement along
the offset's own unit vector, and its look vector is that unit vector. A slant-range
offset uses the line-of-sight vector; an along-track offset uses the horizontal
flight direction. Build the Dataset with that vector in `los_east`, `los_north`,
`los_up` and sample it with `Observations.from_los`.

Offsets are much noisier than interferometric phase and carry a different nuisance
(an orbit or timing constant rather than an ionospheric or orbital surface), so
give each track its own ramp and its own weight:

```python
inversion = SlipInversion(mesh, obs, ramp={"*": "quadratic", "asc_az": "offset"})
```

## GNSS and other point data

```python
import numpy as np

gnss = Observations.from_points(
    x=np.repeat(lon, 3), y=np.repeat(lat, 3),                 # lon/lat (epsg=4326)
    values=np.column_stack([east, north, up]).ravel(),         # metres
    look=np.tile(np.eye(3), (len(lon), 1)),                    # east, north, up rows
    std=np.column_stack([sig_e, sig_n, sig_u]).ravel(),        # metres
    name="gnss", frame=frame)
obs = Observations.concat([obs_asc, obs_desc, gnss], normalize="none")
```

Each row measures the displacement along its own unit vector, so a three-component
station is three rows with looks \((1, 0, 0)\), \((0, 1, 0)\) and \((0, 0, 1)\). The
weight is \(1/\sigma\). Coordinates are lon/lat by default, any EPSG code with
`epsg=`, or local metres with `epsg=None`. With `normalize="none"` the InSAR rows
keep weight 1, so either weight them by their noise as well (a covariance model, or
`weights=`) or expect the GNSS to count for little.

## Noise

InSAR noise is dominated by atmospheric delay, which is correlated over kilometres
to tens of kilometres, so samples within one atmospheric cell are not independent
(Lohman & Simons 2005). [`slipsolve.covariance`](../api/observations.md#slipsolve.covariance)
fits a stationary noise model, a [`CovarianceModel`](../api/observations.md#slipsolve.covariance.CovarianceModel)
with a correlated `sill`, a `range_m` and an uncorrelated `nugget`
(exponential, Gaussian or Matérn 3/2):

```python
from slipsolve import fit_reml, scene_covariance

cov_asc, variogram = scene_covariance(scenes["asc"], trace, frame)   # from the far field
models = fit_reml(obs, model.residual, ramp="quadratic")             # or on the samples
```

`scene_covariance` fits a variogram of the scene's far field, away from the
deformation; `fit_reml` fits restricted maximum likelihood on the samples
themselves, with the ramp space projected out as the inversion projects it.

Passing the models to the inversion solves the whitened system, with a block per
track (tracks are independent):

```python
obs = Observations.concat([obs_asc, obs_desc], normalize="none")
inversion = SlipInversion(mesh, obs, covariance={"asc": cov_asc, "desc": cov_desc})
```

Whitening states the noise instead of assuming it away: it makes
`SlipModel.chi_square` meaningful and makes the resolution and covariance
estimates describe the system that was solved. It does not, in general, give a
better slip estimate. The estimator is bounded and heavily regularized, so the
smoothing rather than the data weighting sets most of its error, and whitening
up-weights the short wavelengths the smoothing exists to suppress. Smoothing
weights do not carry over between whitened and unwhitened problems.

`posterior_covariance(..., noise={track: model})` propagates a noise model through a
solve made without one; see [Uncertainty](regularization.md#uncertainty).
