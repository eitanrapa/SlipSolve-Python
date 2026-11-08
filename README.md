# slipsolve-python

Geodetic coseismic slip inversion on triangular fault meshes. Given gridded InSAR
line-of-sight displacement, pixel offsets and GNSS, a mapped fault trace and a model
of its dip, `slipsolve` solves a bounded, regularized least-squares problem for the
slip on a vertical, dipping or curved fault, with Green's functions for a
homogeneous half-space, a layered crust or a 3-D heterogeneous one. It is a Python
translation of, and extension to, the MATLAB package
[SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve).

**Documentation: <https://eitanrapa.github.io/SlipSolve-Python/>**

## Features

- Faults built from a surface trace: vertical, a dip per segment, a digitised bottom
  edge or a fitted surface; slip per element or per node; graded or unstructured meshes.
- Triangular dislocations (Nikkhoo & Walter 2015), layered Green's functions from the
  package's own wavenumber-domain tables or from EDGRN, 3-D heterogeneity on voxels,
  and rigidity contrasts across the fault.
- Quadtree sampling of LOS grids, driven by the data, a model's prediction or model
  resolution; pixel offsets and GNSS as further tracks; correlated noise models.
- Smoothing, zero-slip edges, bounds and slip polarity, and per-track ramps solved
  jointly with the slip.
- The smoothing weight chosen by effective rank, blocked cross-validation, GCV or the
  L-curve; resolution, posterior covariance and checkerboard calibration.
- Per-segment dip sweeps scored by the marginal likelihood, with bootstrap intervals.
- Postseismic and interseismic modes: slip that changes with time from dated series
  (`TimeDependentInversion`), rigid blocks plus back-slip with coupling bounds from velocities
  (`InterseismicInversion`), and signals of known shape beside the slip (`Pattern`). See
  [the guide](https://eitanrapa.github.io/SlipSolve-Python/guide/time-dependent/).

## Install

```bash
pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"
```

or, with every optional dependency, `conda env create -f environment.yml` and
`pip install -e .` in the clone. Extras: `[layered]` (run EDGRN through `pygrnwang`),
`[dev]` (tests), `[docs]` (this documentation). The import name is `slipsolve`.

## Example

```python
from slipsolve import FaultMesh, FaultTrace, LosGrid, Observations, SlipInversion

trace = FaultTrace.from_file("fault.kml")
frame = trace.local_frame()                      # one local frame for every object
mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

asc = LosGrid.from_grd("los_asc.grd", "look_e.grd", "look_n.grd", "look_u.grd")
obs = Observations.from_los(asc, name="asc", frame=frame, trace=trace,
                            rms_min=0.006, width_min=1e3, exclude_within=2e3)

inversion = SlipInversion(mesh, obs, ramp="linear")
model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))   # right-lateral
print(model.moment_magnitude, model.variance_reduction)
model.save("model.slip.zip")
```

Positive strike-slip is left-lateral, LOS is positive toward the sensor and look
vectors point from the ground to the sensor; see the
[conventions](https://eitanrapa.github.io/SlipSolve-Python/guide/conventions/). The
[getting-started page](https://eitanrapa.github.io/SlipSolve-Python/getting-started/)
has a complete synthetic example that runs as it is.

## Credits and licence

slipsolve is a Python translation of, and extension to,
[SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve) by Xiaoyu Zou (MATLAB,
MIT), which is itself descended from
[SlipSolve](https://github.com/ZeyuJin/geodetic_inversion) by Zeyu Jin. It is neither
of those; the `-python` in the distribution name says which one you are installing.

MIT. See [`LICENSE`](LICENSE), which retains Xiaoyu Zou's original copyright notice as
that licence requires.
