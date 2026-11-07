# API reference

These pages are generated from the docstrings. Every name in `slipsolve.__all__`
can be imported from the package root (`from slipsolve import SlipInversion`); the
pages are organised by module.

| Page | Modules |
|---|---|
| [Geometry](geometry.md) | `slipsolve.trace`, `slipsolve.frame`, `slipsolve.mesh`, `slipsolve.surface`, `slipsolve.layout`, `slipsolve.basis` |
| [Observations](observations.md) | `slipsolve.sampling`, `LosGrid` and `read_grd`, `slipsolve.resample`, `slipsolve.diagnostics`, `slipsolve.covariance` |
| [Elastic models](elastic-models.md) | `slipsolve.greens`, `slipsolve.edgrn`, `slipsolve.layered`, `slipsolve.wavenumber`, `slipsolve.layered_tde` |
| [Contrasts and 3-D structure](structure.md) | `slipsolve.bimaterial`, `slipsolve.heterogeneity`, `slipsolve.fault_cells`, `slipsolve.voxel_solve`, `slipsolve.crust1` |
| [Inversion](inversion.md) | `slipsolve.inversion`, `slipsolve.regularize`, `slipsolve.predict` |
| [Resolution and calibration](resolution.md) | `slipsolve.resolution`, `slipsolve.calibrate` |
| [Geometry search](geometry-search.md) | `slipsolve.sweep`, `slipsolve.evidence` |
| [Plotting](plotting.md) | `slipsolve.plot` |

The [conventions](../guide/conventions.md) (signs, frames and units) apply
throughout.
