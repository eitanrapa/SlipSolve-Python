# 3-D heterogeneity

[`HeterogeneousEngine`](../api/structure.md#slipsolve.heterogeneity.HeterogeneousEngine)
computes Green's functions in a crust with any elastic moduli
\(\lambda(\mathbf{x})\), \(\mu(\mathbf{x})\): a layered reference, solved exactly by
[`LayeredTDE`](elastic-models.md#the-packages-own-tables-layeredtde), plus a
departure from it held on voxels. It is an engine like the others, so
`SlipInversion(engine=...)` takes it with either basis.

## The method

Following Barbot, Fialko & Sandwell (2009), the true field is the reference field
of the slip plus the reference field of an equivalent moment density in the
voxels:

\[
\mathbf{M} = -\Delta\mathbf{C} : \boldsymbol{\varepsilon}, \qquad
\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_0 + \boldsymbol{\Gamma}\mathbf{M}, \qquad
\mathbf{u} = \mathbf{u}_0 + \mathbf{H}\mathbf{M},
\]

with \(\Delta\mathbf{C}\) the departure of the moduli from the reference,
\(\boldsymbol{\varepsilon}_0\) the reference strain of the slip, \(\boldsymbol{\Gamma}\)
the reference medium's moment-to-strain operator and \(\mathbf{H}\) its
moment-to-surface operator. Solving the strain equation gives the Green's matrix of
the heterogeneous crust,

\[
\mathbf{G} = \mathbf{G}_0 - \mathbf{H}\,\Delta\mathbf{C}\,
(\mathbf{I} + \boldsymbol{\Gamma}\,\Delta\mathbf{C})^{-1}\mathbf{E},
\]

with \(\mathbf{E}\) the strain of each slip basis function at the voxels. Each
voxel's field is computed as `LayeredTDE` computes an element's: a closed-form
half-space part (Eshelby's construction, as dislocations on the voxel's faces) plus
the tabulated layered remainder.

## Building the model

The reference tables must carry strain, and the voxels and the base engine must
share one set of tables:

```python
import numpy as np
from slipsolve import (HeterogeneousEngine, LayeredTDE, RemainderTables, SlipInversion,
                       VoxelModel)
from slipsolve.heterogeneity import Box

tables = RemainderTables.compute(crust, workers=6, strain=True)
tables.save("crust_strain.npz")            # worker processes load the tables from file

x_edges = np.arange(-40e3, 40.1e3, 2e3)    # east, local frame metres, uniform
y_edges = np.arange(-20e3, 20.1e3, 2e3)    # north, uniform
z_edges = np.arange(0.0, 10.1e3, 2e3)      # depth below the surface, metres

basin = Box(x=(-30e3, -10e3), y=(2e3, 14e3), depth=(0.0, 4e3))
voxels = VoxelModel.from_bodies(tables, x_edges, y_edges, z_edges,
                                [(basin, "scale", (0.5, 0.5))], frame=frame)
voxels.check_clearance(mesh)               # voxels must keep clear of the fault

engine = HeterogeneousEngine(LayeredTDE(tables), voxels, workers=6)
model = SlipInversion(mesh, obs, engine=engine, basis="node").solve(smoothing=0.3)
```

Ways to fill the voxels:

- `VoxelModel(tables, x_edges, y_edges, z_edges, lam, mu)`: arrays on the grid, or a
  callable `f(x, y, depth) -> (lam, mu)` at the voxel centres;
- `VoxelModel.from_bodies`: `Box` and `Prism` (a map-view polygon between two
  depths) shapes with their own moduli, a callable of depth, or a `"scale"` of the
  reference; later bodies override earlier ones;
- `VoxelModel.from_crust1`: CRUST1.0 (Laske et al. 2013) through
  `slipsolve.crust1.Crust1.load(directory)`;
- `VoxelModel.from_grid` and `VoxelModel.from_dataset`: a tomography on a regular
  lon/lat/depth grid, as arrays or as an xarray Dataset (for example a netCDF model
  file); `slipsolve.heterogeneity.brocher_vs` and `brocher_density` (Brocher 2005)
  fill in missing Vs or density from Vp.

Voxels whose moduli differ from the reference's by less than `tol` are inactive and
cost nothing. Layer interfaces inside the depth range are inserted into `z_edges`,
so no voxel crosses one.

## Limits and costs

- **Voxels must keep clear of the fault.** A voxel's strain is collocated at its
  centre and taken as uniform, so a voxel the fault passes through cannot carry a
  departure. By default voxels keep a voxel size away (`check_clearance`, and
  `keep_clear` returns the offending voxels to the reference); with
  `contact=True` they may end on the fault but not straddle it. A contrast across
  the fault itself is [`Bimaterial`](bimaterial.md)'s job. Cells that follow the
  fault, the fault's own triangles extruded sideways in layers, are
  [`FaultCells`](../api/structure.md#slipsolve.fault_cells.FaultCells), and
  `CombinedCells` solves them together with voxels.
- **Receivers exactly on a voxel face** are singular and refused, by name.
- **Solvers.** `solver="dense"` (default) forms \(\boldsymbol{\Gamma}\) and factors
  it, which limits it to roughly ten thousand voxels on a large machine.
  `solver="fft"` never forms it: on a regular grid \(\boldsymbol{\Gamma}\) is a
  lateral convolution for each pair of depth rows, applied by FFT and solved by
  GMRES (`slipsolve.voxel_solve`), with memory linear in the number of voxels.
- **Cost.** The strain of every slip basis function at every voxel (\(\mathbf{E}\))
  dominates at realistic problem sizes. `workers=` runs \(\mathbf{E}\), \(\mathbf{H}\) and
  \(\boldsymbol{\Gamma}\) in worker processes, each of which loads the tables from
  their file; a script that uses workers must keep its work under
  `if __name__ == "__main__":`. `engine.close()` shuts the pool down.
- The tabulated remainder limits the accuracy, at second order in the tables' depth
  and distance grids; strain is more demanding than displacement.
- `order="born1"` keeps only the first term, \(\mathbf{G}_0 - \mathbf{H}\,\Delta\mathbf{C}\,\mathbf{E}\),
  for a quick look at a weak departure.
