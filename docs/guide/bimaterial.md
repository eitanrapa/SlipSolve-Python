# Bimaterial faults

A fault often separates two different rocks. The softer block moves more for the
same slip, so a rigidity contrast shows in the data as an asymmetry across the
trace, the same kind of asymmetry a dip makes.
[`RigidityContrast`](../api/structure.md#slipsolve.bimaterial.RigidityContrast) says
which block is softer, by how much, and where along strike (or with depth) that
changes. [`Bimaterial`](../api/structure.md#slipsolve.bimaterial.Bimaterial) wraps an
engine with it, and answers to the same calls, so `SlipInversion`, forward fields,
`SlipModel` and the resolution tools take it like any engine.

```python
from slipsolve import Bimaterial, LayeredTDE, RemainderTables, RigidityContrast, SlipInversion

contrast = RigidityContrast(trace, side="north", ratio=(1.5, 1.0), breaks=(50e3,))

scaled = Bimaterial(contrast)                                   # scaled mode, any engine
exact = Bimaterial(contrast, exact=True)                        # exact, half-space
tables = RemainderTables.compute(crust, workers=6)
exact = Bimaterial(contrast, engine=LayeredTDE(tables), exact=True)   # exact, layered

model = SlipInversion(mesh, obs, engine=exact, basis="node").solve(smoothing=0.3)
```

## The contrast

- **`side`** names the block whose rigidity differs from the reference: `"left"` or
  `"right"` of the trace as it runs, or a compass word (`"north"`, `"southwest"`,
  ...) resolved against the trace's mean left-hand normal and refused when more than
  60° from it.
- **`ratio`** is \(\mu_\text{other} / \mu_\text{side}\): above 1 the named block is
  softer.
- **`breaks`** (arc lengths along the trace, metres) cut the fault into intervals
  with one ratio each. In the example, the north block is 1.5 times softer along
  the first 50 km and the same beyond.
- **`reference`** names the block the rigidity model describes (the velocity model,
  or the default 30 GPa): `"other"` (default) or `"side"`. It stays uniform along
  strike; the other block is the reference scaled by each interval's ratio.
- **`depth_breaks`** (metres) cut the stepping block into bands from the surface
  down, with `ratio` then one row per interval and one column per band:
  `RigidityContrast(trace, side="north", ratio=[[2.0, 1.0]], depth_breaks=(6e3,))` is
  a north block twice as soft in the top 6 km only.

The material boundary is the fault surface itself, continued down dip and past the
trace's ends. The two blocks are one reference model scaled by a constant on each
side: the same layering and Poisson's ratio, a different rigidity.

## Two modes

Within that model the field has an exact form: each block's own scale times the
field of a displacement discontinuity on the boundary, which solves a boundary
equation driven by the slip.

**Scaled** (`exact=False`) keeps the first term of that solution: each sample's row
is scaled by its own block, and each slip column by the harmonic mean of the two
rigidities over the reference's, \(2r/(1+r)\) on the named side and \(2/(1+r)\) on
the other. It costs nothing beyond the wrapped engine and works with any engine.
It is exact only for strike-slip on a vertical fault in 2-D. On a dipping fault its
slip is an effective slip, not the slip; in 3-D the exact discontinuity also opens
the fault near the rupture's ends, which lifts the ground and which LOS sees; and
near a change of ratio along strike it is a poor approximation. Use it to explore.

**Exact** (`exact=True`) solves for the discontinuity on the whole interface: the
fault, its continuation (`extend`, in units of the fault's depth extent), the
across-strike boundaries at the breaks and the horizontal boundaries at the depth
breaks. It needs each element's field at depth with opening as well as slip, which
the half-space engine and `LayeredTDE` provide. The
interface solve is dense, \((3N)^2\) for \(N\) interface elements, and is done once
per mesh; after that each Green's matrix costs one engine matrix over the interface.
With `LayeredTDE` both cost several times the half-space's. Quote a dip, a ratio or
a moment from the exact mode.

On a layered crust, use the layered exact mode: layering changes the contrast's
effect enough that the half-space version is not a stand-in.

## Using it well

- **Where only one block has data, the ratio there cannot be measured.** It rescales
  the slip by up to \((1+r)/2\) and is an assumption, not a result; state it with
  `breaks`.
- **Dip and contrast trade off.** Even with data on both sides, a ratio's estimate
  moves with the dip's. Sweep them together with
  [`DipSweep(contrast=..., ratios=...)`](dip-sweeps.md#a-rigidity-contrast-as-a-sweep-axis),
  which uses the scaled mode, and confirm with the exact mode. Give both grids room.
- **The horizontal boundaries of `depth_breaks` must reach past the data**, as the
  across-strike ones do (`extend["cross"]`).
- **Lay the fault's mesh levels on the depth breaks.** An element crossing one takes
  its centroid's band, with a warning.
- **Keep a depth break off a layer interface**, where the layered remainder is
  singular.
- `DipSweep` does not take a contrast with depth breaks; use `Bimaterial(...,
  exact=True)` at fixed ratios.
- **The moment** uses the harmonic-mean rigidity \(2\mu_+\mu_-/(\mu_+ + \mu_-)\) of the
  two blocks; `SlipModel.moment()` applies it, and a saved model keeps its contrast.

Not modelled: more than two blocks across strike, different Poisson's ratios on the
two sides, and a contrast that varies smoothly with depth except as steps.
