# Elastic models

An *engine* computes the Green's functions: the displacement each observation would
record for unit slip on each element (or node). Every engine answers to the same
calls (`los_matrix`, `displacement` and `forward`), so `SlipInversion(engine=...)`
takes any of them and nothing else changes.

| Engine | Medium | Notes |
|---|---|---|
| [`HalfSpaceTDE`](../api/elastic-models.md#slipsolve.greens.HalfSpaceTDE) | homogeneous half-space | closed form; the default |
| [`LayeredTDE`](../api/elastic-models.md#slipsolve.layered_tde.LayeredTDE) | layered half-space | the package's own tables; exact near the fault, fields at depth |
| [`LayeredPointSource`](../api/elastic-models.md#slipsolve.layered.LayeredPointSource) | layered half-space | EDGRN tables, as EDCMP uses them |
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
Two layered engines are available.

### The package's own tables: `LayeredTDE`

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

### EDGRN tables: `LayeredPointSource`

```python
from slipsolve import EdgrnTables, LayeredPointSource, run_edgrn

tables = EdgrnTables.from_input_file("edgrn.inp")    # tables from your own EDGRN run
tables = run_edgrn(crust, "edgrn_work")              # or run EDGRN (needs [layered])
engine = LayeredPointSource(tables)
```

The approach of EDCMP and SlipSolve-Curve: each element is cut into point sources,
each looked up in Rongjiang Wang's EDGRN tables (Wang, Lorenzo-Martin & Roth 2003)
and summed. The quadrature order is chosen per element and observation from the
observation's distance in units of the element's size, rather than fixed. Point
sources are accurate a few element sizes from the fault and degrade close to it, and
the tables give no field at depth and no opening source. `write_edgrn_input` writes
the input file for a velocity model, and `EdgrnTables.homogeneous` synthesises the
tables of a uniform medium for testing without Fortran.

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
