# Conventions

Most ways of getting a slip inversion wrong produce a plausible-looking answer
rather than an error: a flipped sign gives a fault slipping the wrong way with a
poor fit, a frame mismatch gives a smooth but displaced Green's matrix. This page
lists the conventions every part of the package uses. Read it before feeding the
package real data.

## Units

Everything is SI.

| Quantity | Unit |
|---|---|
| Coordinates, depths, distances, element sizes, slip, displacement | metres |
| Rigidity (shear modulus), Lamé parameters | pascals |
| Seismic velocities | m/s |
| Density | kg/m³ |
| Seismic moment | N·m |
| Angles (dip, strike, azimuths, incidence) | degrees |

Moment magnitude is \( M_w = \tfrac{2}{3} (\log_{10} M_0 - 9.1) \) with \(M_0\) in N·m.

Kilometres appear only in figure axes and printed summaries. The mesh refuses node
coordinates larger than \(10^7\), which catches a mesh accidentally built in
kilometres.

## One local frame

Every object is expressed in one flat, right-handed frame: \(x\) east, \(y\) north,
\(z\) up, in metres. A [`LocalFrame`](../api/geometry.md#slipsolve.frame.LocalFrame)
is a transverse Mercator projection whose central meridian runs through the study
area, with the projected coordinates of a chosen origin subtracted, so the origin
is \((0, 0)\). `FaultTrace.local_frame()` puts the origin at the trace's midpoint
by arc length.

A UTM grid is not good enough: two tracks over one fault can fall in different UTM
zones, and their coordinates are then metres from two different central meridians.

- **Build one frame and pass it to everything:** the mesh, every call to
  `Observations.from_los`, `Observations.from_points` and the rigidity contrast.
- **Each object records the frame it was built in** (`frame.to_dict()`, kept in the
  object's `attrs`) and checks it when objects meet. `SlipInversion` refuses a mesh
  and observations built in different frames, and `Observations.concat` refuses
  tracks sampled in different frames.
- A grid that is already in local-frame metres (for example one made by
  `resample_to_frame`) says so with `attrs["frame"]`. Otherwise the grid's
  `attrs["epsg"]` names its projected CRS, and its coordinates are transformed into
  the frame.
- To write a raster whose `x`/`y` are local metres, tag it with
  `frame.local_crs`, not `frame.crs`. The latter is the bare projection, offset from
  local metres by the origin and the 500 km false easting.
- **The frame's \(y\) is grid north, not true north.** Away from the central meridian
  they differ by about the longitude offset times the sine of the latitude: 1 degree
  150 km from it at 37 N, which puts 1.7% of the horizontal field in the wrong
  component. Look vectors are geographic. `SlipInversion(frame_looks=True)` rotates
  them into the frame's axes, as the time-dependent modes do by default, and a model
  solved that way reports geographic east and north. Turn it on once the data reach
  a degree or more of longitude from the meridian at mid latitudes.

## Depth and the free surface

The free surface is \(z = 0\) and the fault lies at \(z \le 0\): node and centroid
`z` values are negative below the surface. Arguments that describe a depth
(`max_depth`, `top_depth`, `depths=`, `depth_breaks=`, voxel `z_edges`, the depths
of a `VelocityModel`) are positive metres below the surface. With topography the
observations carry their heights above a datum, the elevation \(z = 0\) stands for
([Elastic models](elastic-models.md#topography)).

The mesh is parameterized by \((s, z)\): \(s\) is arc length along the trace in
metres from its first vertex, and \(z\) is the (negative) vertical coordinate.
`mesh.params` holds them per node and `mesh.element_params` per element centroid.

## Line of sight

- **The `los` value is positive toward the sensor**, the sense of a decreasing
  range.
- **A look vector points from the target (the ground) to the sensor**, as east,
  north and up components (`los_east`, `los_north`, `los_up` on a grid; `look_e`,
  `look_n`, `look_u` on samples). Its up component is \(\cos(\text{incidence})\)
  and positive.

With both conventions the forward model is a plain dot product of the ENU
displacement with the look vector, with no sign to remember.

When importing grids from other software, check both: `LosGrid.from_grd` takes
`sign=-1` for a grid that is positive away from the sensor and
`look_convention="sensor_to_target"` for look vectors that point the other way,
and applies both on load so the stored values are canonical. A common GMT recipe,
`grdmath phase 4 DIV PI DIV wavelength MUL`, gives a grid that is positive away
from the sensor. Do not copy a sign from another product of the same track;
measure it, for example from the sense of near-field motion against the known
sense of slip. `Observations.from_los` records a grid's `sign` attribute but never
applies it again.

The same convention covers other data types: a GNSS component is a sample whose
"look" is the unit vector of that component, \((1, 0, 0)\) for east, and a pixel
offset is a sample along the offset's own unit vector (see
[Observations](observations.md)).

## The sign of slip

Slip parameters are ordered `[strike-slip on every element, dip-slip on every
element]` (per node for a nodal basis), followed by any ramp coefficients.

**Positive `strike_slip` is left-lateral (sinistral)**, at every strike and for
either direction of the trace. A **right-lateral** fault therefore has negative
strike-slip, and its inversion should say so with `polarity=(-1, 0, 0)`:

```python
model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))   # right-lateral
```

`polarity` is `(strike, dip)` flags (a third, tensile entry is accepted and
ignored): `+1` keeps a component non-negative, `-1` non-positive, `0` leaves it
free. A polarity bound of the wrong sign does not fail; it returns near-zero slip
and a poor fit that looks like bad data.

**Positive `dip_slip` moves the block on the trace's left-hand side up-dip relative
to the block on its right.** On a fault that dips toward its left-hand side
(`uniform_dip` below 90) that is reverse (thrust) motion of the hanging wall; on
a fault that dips toward its right-hand side (above 90) it is normal motion. On a
vertical fault the left-hand block moves up.

Left and right are taken looking along the trace from its first vertex to its last
(`FaultTrace.side` returns +1 on the left). Reversing the trace's vertex order
swaps the two sides; it does not change the sense of strike-slip.

These follow from the triangular dislocation's frame: each element's normal is
wound to point to the trace's left-hand side, its strike vector is
\(\hat{e}_z \times \hat{n}\), which is minus the trace's direction, and its dip
vector is \(\hat{n} \times\) strike, which points up-dip. The mesh builds every
triangle with that winding and checks it; a triangle wound the other way would flip
its dip-slip column and leave strike-slip untouched.

## Dip and strike

- `FaultMesh.curved(uniform_dip=...)` and the dips of a [`DipSweep`](dip-sweeps.md)
  are measured from the trace's left-hand side: below 90 the fault dips toward the
  left, above 90 toward the right. Check `mesh.dip_direction` (the azimuth of the
  down-dip direction, clockwise from north) on your own trace.
- `mesh.dip` is folded into \([0, 90]\), as in SlipSolve-Curve's element table, so
  a fault dipping 60° either way reports 60; read it with `dip_direction`.
- `mesh.strike` is the azimuth of each element's own strike vector, which points
  backwards along the trace: 180° from the direction the trace was digitised in.
  It is what the element table written by `SlipModel.to_text` reports.

## Rigidity and moment

The half-space engine's displacements depend only on Poisson's ratio (default
0.25); rigidity enters only through the moment. Without a velocity model the
moment uses a uniform shear modulus of 30 GPa. Pass `velocity_model=` to
`SlipInversion` (or `shear_modulus=` to `SlipModel.moment`) to use the rigidity of
a layered crust at the depth of the slip.
