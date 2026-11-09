# Getting started

This example runs a complete inversion on synthetic data, with the public API only:
it builds a fault, plants a right-lateral slip patch, forward-models two
interferograms with noise and an arbitrary offset, samples them, inverts, and checks
the answer against what was planted. It needs nothing beyond the package and runs
in well under a minute.

```python
import numpy as np
import xarray as xr

from slipsolve import (FaultMesh, FaultTrace, HalfSpaceTDE, LosGrid,
                       Observations, SlipInversion, SlipModel)
from slipsolve.plot import plot_slip

# 1. The fault: a surface trace in lon/lat. Vertex order fixes the
#    along-strike direction and which side of the fault is "left".
trace = FaultTrace.from_lonlat(lon=[20.00, 20.20, 20.40, 20.60],
                               lat=[40.00, 40.04, 40.06, 40.10], name="example")
frame = trace.local_frame()            # one local metric frame for everything
mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=4e3)

# 2. A planted slip patch. Parameters are [strike-slip on every element,
#    dip-slip on every element]; negative strike-slip is right-lateral.
s, z = mesh.element_params.T           # arc length and depth (z < 0), metres
patch = (np.exp(-(((s - s.mean()) / 12e3) ** 2))
         * np.clip(1.0 + z / 12e3, 0.0, 1.0))
truth = np.concatenate([-2.0 * patch, np.zeros(mesh.n_elements)])

# 3. Two synthetic interferograms on a 1 km grid in the local frame. A look
#    vector points from the ground to the satellite (east, north, up), and
#    LOS displacement is positive toward the satellite.
x = np.arange(-60e3, 60.1e3, 1e3)
y = np.arange(50e3, -50.1e3, -1e3)                # north-up rows
xx, yy = np.meshgrid(x, y)
far = trace.distance(xx.ravel(), yy.ravel(), frame) > 1e3
enu = np.full((xx.size, 3), np.nan)              # no data on the trace itself
enu[far] = HalfSpaceTDE().forward(mesh, truth, xx.ravel()[far], yy.ravel()[far])

rng = np.random.default_rng(0)
looks = {"asc": (0.565, 0.100, 0.819), "desc": (-0.620, 0.109, 0.777)}
grids = {}
for name, look in looks.items():
    # signal + an arbitrary 1 cm offset + 3 mm of white noise
    los = enu @ np.array(look) + 0.01 + rng.normal(0.0, 0.003, xx.size)
    ds = xr.Dataset(
        {"los": (("pair", "y", "x"), los.reshape(1, *xx.shape)),
         **{f"los_{c}": (("y", "x"), np.full(xx.shape, v))
            for c, v in zip(("east", "north", "up"), look)}},
        coords={"pair": [0], "y": y, "x": x},
        attrs={"frame": frame.to_dict()},     # x and y are local-frame metres
    )
    grids[name] = LosGrid(ds)

# 4. Quadtree sampling, dropping samples within 2 km of the trace.
obs = Observations.concat([
    Observations.from_los(grid, name=name, frame=frame, trace=trace,
                          rms_min=0.004, width_min=2e3, width_max=16e3,
                          exclude_within=2e3)
    for name, grid in grids.items()
])
print(obs)

# 5. Invert. The Green's functions are built once, here; every solve reuses
#    them. A linear ramp per track absorbs the offset, and polarity=(-1, 0, 0)
#    keeps the strike-slip right-lateral.
inversion = SlipInversion(mesh, obs, engine=HalfSpaceTDE(), ramp="linear")
model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
print(model)

true_m0 = 30e9 * np.sum(mesh.areas * np.abs(truth[:mesh.n_elements]))
print(f"converged: {model.converged}")
print(f"variance reduction {model.variance_reduction:.1f}%, "
      f"rms misfit {1e3 * model.rms_misfit:.1f} mm")
print(f"Mw {model.moment_magnitude:.2f} "
      f"(planted {2 / 3 * (np.log10(true_m0) - 9.1):.2f}), "
      f"peak slip {model.max_slip:.2f} m (planted 2.00 m)")
print("correlation with the planted strike-slip: "
      f"{np.corrcoef(model.strike_slip, truth[:mesh.n_elements])[0, 1]:.3f}")
print("ramp (m):", {k: round(float(v), 4) for k, v in zip(model.ramp_labels, model.ramp)})

# 6. Keep the result.
model.save("example.slip.zip")             # model, mesh and samples in one file
model.to_text("slip_model.txt")            # one row per element
fig, ax = plot_slip(model)
fig.savefig("slip.png", dpi=150, bbox_inches="tight")

print(SlipModel.load("example.slip.zip"))
```

It prints:

```text
<Observations n=918 tracks=['asc', 'desc']>
<SlipModel VR=100.0% max_slip=1.79m Mw=6.53 roughness=0.101>
converged: True
variance reduction 100.0%, rms misfit 1.1 mm
Mw 6.53 (planted 6.52), peak slip 1.79 m (planted 2.00 m)
correlation with the planted strike-slip: 0.998
ramp (m): {'asc:offset': 0.0101, 'asc:dx': -0.0002, 'asc:dy': -0.0009, 'desc:offset': 0.01, 'desc:dx': -0.0006, 'desc:dy': 0.0008}
<SlipModel VR=100.0% max_slip=1.79m Mw=6.53 roughness=0.101>
```

The noise here is white and small, so the fit is near perfect; real interferograms
carry correlated atmospheric noise of centimetres. The recovered moment matches the
planted one, the peak slip is lower because the smoothing spreads the patch, and the
1 cm offsets land in the ramp rather than in the slip.

## What each step does

1. **The fault.** [`FaultTrace`](guide/faults-and-meshes.md#the-fault-trace) holds the
   trace; `local_frame()` builds the [local frame](guide/conventions.md#one-local-frame)
   every object shares. `FaultMesh.vertical` extrudes the trace to 20 km in elements
   of about 4 km.
2. **Planted slip.** `mesh.element_params` gives each element's arc length and
   (negative) depth. Negative strike-slip is right-lateral; see
   [the sign of slip](guide/conventions.md#the-sign-of-slip).
3. **Data.** A gridded product is any xarray Dataset with `los` on `(pair, y, x)` and
   the three look-vector components; `attrs["frame"]` says the grid is already in
   local-frame metres. For real data, see
   [Observations](guide/observations.md#gridded-line-of-sight-products).
4. **Sampling.** `Observations.from_los` quadtree-samples each grid, finely where the
   displacement varies and coarsely where it is smooth, and drops samples near the
   trace, where a dislocation solution is singular. `Observations.concat` combines
   the tracks.
5. **Inversion.** `SlipInversion` computes the Green's matrix once; `solve` adds
   smoothing, zero-slip edges on the sides and bottom, the polarity bound and the
   ramp. The weight `smoothing=0.3` is arbitrary here; choosing it is the most
   consequential decision in a real inversion (see
   [Regularization](guide/regularization.md)).
6. **Output.** `save` writes one file that `SlipModel.load` reads back; `to_text`
   writes a table of element slip; `plot_slip` draws the slip on the unrolled fault.

## With real data

The same steps with real inputs, here with placeholder file names:

```python
import numpy as np

from slipsolve import (FaultMesh, FaultTrace, LosGrid, Observations, SlipInversion,
                       blocked_cv_curve, resample_all, scene_report)

trace = FaultTrace.from_file("fault.kml")
frame = trace.local_frame()
mesh = FaultMesh.curved(trace, frame, uniform_dip=80, max_depth=20e3, edge_length=3e3)

asc = LosGrid.from_grd("asc_los.grd", "asc_look_e.grd", "asc_look_n.grd", "asc_look_u.grd",
                       units="m", sign=1, direction="ascending", look_direction="right")
desc = LosGrid.from_grd("desc_los.grd", "desc_look_e.grd", "desc_look_n.grd",
                        "desc_look_u.grd", units="m", sign=1, direction="descending",
                        look_direction="right")
scenes = resample_all({"asc": asc, "desc": desc}, frame)     # one lattice for all

parts = []
for name, scene in scenes.items():
    report = scene_report(scene, trace, frame, mesh=mesh)    # measured settings
    settings = {k: report.attrs[k] for k in ("rms_min", "width_min", "exclude_within")}
    parts.append(Observations.from_los(scene, name=name, frame=frame, trace=trace,
                                       **settings))
obs = Observations.concat(parts)

inversion = SlipInversion(mesh, obs, ramp="quadratic", basis="node")
curve = blocked_cv_curve(inversion, np.geomspace(0.03, 30, 9), polarity=(-1, 0, 0))
model = inversion.solve(smoothing=curve.attrs["best_smoothing"], polarity=(-1, 0, 0))
```

Before trusting the result, check the [conventions](guide/conventions.md) against your
data (the sign of LOS, the direction of the look vectors, the sense of slip), check
that the cross-validation minimum is interior (`curve.attrs["interior"]`), and look at
the fit per track with [`plot_fit`](api/plotting.md#slipsolve.plot.plot_fit).

## Next steps

- [Faults and meshes](guide/faults-and-meshes.md): dipping and curved faults, graded
  and unstructured meshes, nodal slip.
- [Observations](guide/observations.md): real LOS grids, sampling parameters, pixel
  offsets, GNSS and noise models.
- [Elastic models](guide/elastic-models.md): layered crusts.
- [Regularization and resolution](guide/regularization.md): choosing the smoothing
  weight, and what the data resolve.
