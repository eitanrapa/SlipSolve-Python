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
- **Which engines.** The half-space, layered and bimaterial engines: a layered engine keeps its
  layers under the local surface, so they move with it. A voxel model (`HeterogeneousEngine`)
  refuses heights for now, since lowering the fault alone would move it against its voxels.
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
