# Elastic models

An *engine* computes the Green's functions: the displacement each observation would
record for unit slip on each element (or node). Every engine answers to the same
calls (`los_matrix`, `displacement` and `forward`), so `SlipInversion(engine=...)`
takes any of them and nothing else changes.

| Engine | Medium | Notes |
|---|---|---|
| [`HalfSpaceTDE`](../api/elastic-models.md#slipsolve.greens.HalfSpaceTDE) | homogeneous half-space | closed form; the default |
| [`LayeredTDE`](../api/elastic-models.md#slipsolve.layered_tde.LayeredTDE) | layered half-space | the package's own tables; exact near the fault, fields at depth |
| [`HeterogeneousEngine`](heterogeneity.md) | any \(\lambda(\mathbf{x}), \mu(\mathbf{x})\) | voxels on a layered reference |
| [`Bimaterial`](bimaterial.md) | two blocks of different rigidity | wraps another engine |
| [`Curvature`](#earths-curvature-and-gravity) | the same layers on a sphere, with gravity | wraps another engine; the far field |

## Homogeneous half-space

```python
from slipsolve import HalfSpaceTDE

engine = HalfSpaceTDE(nu=0.25)
```

Triangular dislocations after Nikkhoo & Walter (2015), a vectorised port of their
artefact-free solution. Triangles tile a curved or segmented fault without the gaps
and overlaps rectangles leave at every bend. The displacements depend only on
Poisson's ratio; rigidity cancels and enters only through the moment.

## Layered crust

A uniform crust gives the shallow, soft layers the rigidity of the deep crust, so
the same surface displacement then needs less shallow slip and more deep slip.
`LayeredTDE` computes a layered crust's Green's functions from tables the package
builds itself.

```python
from slipsolve import LayeredTDE, RemainderTables, VelocityModel

crust = VelocityModel.from_file("crust.txt")        # rows of depth vp vs rho (SI)
tables = RemainderTables.compute(crust, workers=6)  # once per crust; minutes
tables.save("crust_tables.npz")                     # RemainderTables.load(...) later
engine = LayeredTDE(tables)
```

Near an element, its layered field is split into the closed-form half-space
dislocation of each piece of the triangle (cut at the layer interfaces, each piece
with its own layer's Poisson's ratio) plus a smooth remainder, the layered point
source minus the half-space one, which the package tabulates by wavenumber
integration. Far from it, the whole tabulated field of point sources is used. So
the engine is exact at the fault, works at depth as well as at the surface, and has
an opening source, which the [exact bimaterial solve](bimaterial.md) and the
[3-D engine](heterogeneity.md) need.

- The crust must be made of homogeneous layers. In a `VelocityModel` a step is two
  rows at the same depth; a gradient (two rows at different depths with different
  values) is refused.
- Nodal slip is integrated over each tent exactly.
- The least accurate place is an element edge lying along a layer interface,
  where the remainder is not smooth. Avoid putting mesh levels exactly on an
  interface; a triangle that crosses one is cut into pieces, one per layer.

Why not tables of the whole field, summed as point sources (EDGRN's, as EDCMP and
SlipSolve-Curve use them)? Point sources are accurate a few element sizes from the
fault and degrade close to it, and such tables are blurred to their own grid spacing
near the source; they also give no field at depth and no opening source.

## Topography

Every engine puts the receivers on a flat free surface at \(z = 0\). Where the ground
stands high, as on a range beside a fault, the observations can carry their heights and
the Green's functions then use the elevation shift of Williams & Wadge (1998): each
observation is computed as if the free surface passed through it, the fault deeper below
it by its height above the model's surface (the datum).

```python
dem = ...                                        # (ny, nx) on the stack's grid, metres above sea level
obs = Observations.from_los(stack, frame=frame, trace=trace, exclude_within=2e3, elevation=dem, datum=0.0)
gnss = gnss_observations(lon, lat, ve, vn, vu, se, sn, su, frame, elevation=heights, datum=0.0)
inversion = SlipInversion(mesh, Observations.concat([obs, gnss]), engine=engine)
```

- **The heights.** `Observations.from_los` averages the DEM over each cell's valid pixels,
  as it averages the look vectors; GNSS rows take their stations' heights
  (`gnss_observations`, `gnss_series`, `velocity_observations`). Each sample stores its
  height above `datum`, the elevation the model's \(z = 0\) stands for: the trace's, or sea
  level on a coastal fault. Give heights to every data set or to none, in one datum (a GNSS
  height is usually ellipsoidal, tens of metres from a DEM's).
- **The Green's functions.** Rows are evaluated on elevation levels 50 m apart
  (`TOPOGRAPHY_STEP`), the fault lowered by each level (`FaultMesh.lowered`), and each row is
  interpolated between the two levels around its height. That is exact on a level and within
  about 1e-4 of the largest entry between levels for receivers 2 km from the fault. Every
  inversion takes them: `SlipInversion`, `InterseismicInversion`, `TimeDependentInversion`
  (each series' cells carry their heights).
- **Which engines.** All of them. A layered engine keeps its layers under the local surface, so
  they move with it. In `HeterogeneousEngine` the voxels, like the fault, are fixed beneath the
  datum, so each point sees both the fault and the voxels' induced moments deeper by its height
  (a voxel lowered across a layer interface is cut there); the moments are solved once beneath
  the datum, so topography's effect on them is left out. It is about topography's own effect
  times the heterogeneity's: beside a 15 km deep fault, with a block 30% softer at 2-8 km and
  every point 0.4-1 km high, 5% of topography's effect (1% for the block at 10-16 km).
- **What it is.** A first-order correction: it moves the surface, not its slope. It is good
  where slopes are moderate and the relief is small against the distance to the slip
  (Williams & Wadge 1998; Cayol & Cornet 1998 compared it with full topography). Steep relief
  next to shallow slip needs a boundary- or finite-element model.
- **What it moves.** On a coseismic model whose samples stand up to 2.3 km high beside the
  fault (median 450 m) it changed the predictions by 5 mm rms (about 3% of the signal above
  100 m), and re-solved it moved Mw by -0.007 and the peak slip by +11%, with a slightly better
  fit.
- Forward maps (`SlipModel.surface_displacement`, `predicted_los`) and the patterns beside the
  slip are computed on the flat surface.

## Earth's curvature and gravity

Every engine is flat and non-gravitating. Near a fault that costs little, but by a few
hundred kilometres both the sphere and gravity change the field by several percent.
`Curvature` wraps any engine and adds the difference, from tables computed once per crust:

```python
from slipsolve import Curvature, CurvatureTables, LayeredTDE

far = CurvatureTables.compute(crust, r_max=800e3, z_max=30.5e3, workers=6)  # once per crust; about a minute
far.save("crust_far_field.npz")                                            # CurvatureTables.load(...) later
engine = Curvature(LayeredTDE(tables), far)
inversion = SlipInversion(mesh, obs, engine=engine, velocity_model=crust)
```

- **What it adds.** The field of the crust's layers on a sphere of radius 6371 km (the deepest
  layer filling the ball beneath) less the same layers flat. Then the flat layers' buoyancy: the
  layers hydrostatically prestressed, as the [relaxation tables](time-dependent.md) have them
  (Wang, Lorenzo-Martín & Roth 2006). `gravity=0` leaves gravity out.
- **How large.** The table gives each effect relative to the local field at 100-500 km, for a source
  at 10 km in a layered crust:

  | One azimuth each | Curvature | Gravity |
  |---|---|---|
  | Vertical strike-slip fault, horizontal | 2-10% | under 0.4% |
  | Vertical strike-slip fault, vertical | 4-12% | 4-9%, of the opposite sign |
  | 45° thrust, horizontal | 0.3-1.7% | 2-11% |
  | 45° thrust, vertical | 5-15% | 6-26%, of the same sign |

  Within 50 km both are about 1% of the horizontal field or less. So correct both or neither.
  Curvature alone over-corrects a strike-slip fault's vertical field, where the two nearly
  cancel, and leaves out most of a thrust's change.
- **How it is computed.** The sphere is solved degree by degree. In each shell the solutions are
  powers of the radius (Love's), handled with the global-matrix method of the flat tables. A point
  source becomes a displacement jump plus a surface force, which carries the curvature of the source
  sphere. The sphere and both flat fields share one Gaussian taper in wavenumber, so their common
  near field cancels. Each difference is then good wherever it is smooth on the taper's length
  (`1 / k_taper`, 1 km by default), which covers everywhere it is not negligible.
- **How it was checked.**
  - As the radius grows, the sphere becomes the flat layers, the difference falling as `1 / R`.
  - Degree by degree, it satisfies Betti's reciprocal theorem against the same sphere loaded at its
    surface, to 1e-8.
  - Its far field agrees with STATIC1D (Pollitz 1996, a non-gravitating sphere) minus the flat engine.
    For a 240 km strike-slip rupture, at 14 GNSS stations 190-800 km away, the difference is 0.06 mm
    rms. The correction there is 0.35 mm rms: 4-8% of the horizontal field within 550 km, 7-18% at
    660-800 km. The differences reach 5% of the field at the farthest stations, more than STATIC1D's
    own 1-2% spread between truncation degrees there. So this comparison bounds the correction's
    error, at about a fifth of the correction, more than it tests it finely.
- **Cost.** A few seconds per source depth for the tables. Adding the correction to 3,300
  elements and 17,000 points took about 4 minutes in one run (`order=2`: four point sources per
  element).
- **With the other options.**
  - Topography lowers the correction with the fault, as for any engine.
  - Wrap the outermost engine: `Curvature(HeterogeneousEngine(...), far)`, or around a
    `Bimaterial`. The correction is computed on the layered reference, so its product with the
    voxels' or the contrast's effect is left out.
  - The product of curvature and gravity is left out too, about 1% of the field at 500 km.
- **Frame.** Distances and azimuths are those of the local frame (a transverse Mercator), whose
  own distortion is a few tenths of a percent at 500 km. Its grid north also turns away from
  true north, by about the longitude offset times the sine of the latitude. Over such distances
  rotate the look vectors for that too: `SlipInversion(frame_looks=True)`, the default of the
  time-dependent modes ([Conventions](conventions.md#one-local-frame)).

## Rigidity in the moment

The engines do not decide the moment's rigidity. Pass the velocity model to the
inversion as well, or every moment assumes 30 GPa:

```python
inversion = SlipInversion(mesh, obs, engine=LayeredTDE(tables), velocity_model=crust)
```

`SlipModel.moment()` then averages the layered rigidity exactly over each element,
interfaces included; see [The inversion](inversion.md#moment).

## What a saved model keeps

A saved `SlipModel` does not store the Green's matrix or any tables (they are large
and nothing downstream of a solved model needs them). A loaded model reports every
statistic and exports, and can forward-model new points with the half-space engine
(or a bimaterial wrapper around it). For a layered model, rebuild the engine from
the same tables and a fresh `SlipInversion`.
