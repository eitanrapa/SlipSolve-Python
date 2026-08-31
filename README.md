# slipsolve-python

Resolution-driven geodetic slip inversion on curved triangular fault meshes,
from InSAR line-of-sight displacement.

> **Lineage.** This is a Python translation of, and extension to,
> [**SlipSolve-Curve**](https://github.com/x3zou/SlipSolve-Curve) (Xiaoyu Zou,
> MATLAB, MIT), which is itself descended from **SlipSolve** (ZeyuJin). It is
> *not* either of those. The `-python` in the distribution name is there to say
> which one you are installing; the import name is `slipsolve`.

```python
import slipsolve
```

## What it does

Given line-of-sight displacement from one or more InSAR scenes, a mapped fault
trace and a dip model, it solves for the slip distribution on a triangulated
fault surface:

- **curved fault meshes** built from a surface trace, with element-constant or
  nodal (P1) slip;
- **layered elastic Green's functions** via EDGRN/EDCMP, or a closed-form
  homogeneous half-space (Nikkhoo & Walter triangular dislocations);
- **model-based quadtree subsampling** of each scene (Lohman & Simons 2005,
  Wang & Fialko 2015);
- **Laplace–Beltrami regularisation** with bound constraints, and per-track
  ramp and offset terms solved *jointly* with slip;
- **resolution analysis** — the model resolution matrix, effective rank,
  anisotropic averaging kernels, posterior covariance and regularisation bias;
- **measured data covariance** — an exponential variogram fitted per track from
  the far-field residual, with block-diagonal whitening.

## What is new here relative to the MATLAB

Two things, both benchmarked:

**Adaptive quadrature for layered Green's functions.** The reference
implementations — EDCMP, and SlipSolve-Curve itself — integrate each
`(triangle, observation)` pair at a fixed order, 91 points in EDCMP's case. The
required order actually depends on how far the observation is from the element
*in units of the element's own size*, so it is chosen per pair from a measured
error model. Result: **48× fewer quadrature points**, and layered assembly at
**0.86× the cost of the closed-form homogeneous engine**. There is no longer a
computational reason to accept a homogeneous half-space, which matters because a
uniform crust biases the shallow/deep slip partitioning.

**A joint resolution loop.** Barnhart & Lohman (2010) size a mesh from
resolution; Lohman & Simons (2005) subsample data from a predicted field. Both
papers keep the two steps sequential and separate. Here one resolution matrix
drives both, alternating. Measured honestly, this is an **efficiency** method
rather than an accuracy one — see the findings below.

## Installing

```bash
conda env create -f environment.yml && conda activate slipsolve
pip install -e .
```

The dependency list is deliberately short — numpy, scipy, xarray, zarr, pyproj,
rioxarray, matplotlib. This package works on points and meshes rather than
rasters, is eager throughout and uses no dask, so it does not inherit a raster
pipeline's dependencies.

| extra | what for |
|---|---|
| `[layered]` | `pygrnwang`, which bundles and builds Wang's EDGRN/EDCMP Fortran. Only needed if you want the package to *generate* layered tables; you can equally run EDGRN yourself and hand over the tables. |
| `[dev]` | `pytest`, plus `cutde` as an independent oracle for the triangular dislocation. |

Reading NISAR products is **not** part of this package. Feed it any object
exposing an xarray `.ds` with `los`, `los_east/north/up`, `x`, `y` — a `.grd`
pair, a plain Dataset, or [`nisar_tools`](https://github.com/eitanrapa/nisar_tools)'
`LOSStack`.

## What the method actually buys — measured

Run on six scenes (NISAR + ALOS) over the San Sebastián fault, Venezuela, and
against synthetic targets with known truth:

- **λ dominates.** Four mesh designs at matched effective rank agree on misfit to
  0.3% and on Mw to 0.005. Row-structured and unstructured layouts, scored
  against planted truth, agree to within 3% at every damping. Changing λ moves
  the answer roughly **thirty times** as far as changing the mesh does.
- **The L-curve corner is not the minimum-error point.** Scored against truth
  under measured noise, total error minimises at effective rank ~79 while the
  corner sits at 21 — the corner is **40% worse**.
- **More data does not buy resolution unless λ is relaxed to spend it.**
  Removing every coverage mask (2.4× the observations) made recovery *worse* at
  matched rank.
- **Data covariance does not improve the point estimate** (0.147 m rms against
  0.069, five seeds out of five). GLS is minimum-variance only among *unbiased*
  estimators, and this one is bound-constrained and ~97% regulariser. It is kept
  for honest error bars and a meaningful χ², not for the estimate.

The full write-up, with the mathematics, is in `docs/`.

## Calibrating it for your own data

`slipsolve.calibrate` runs a standardised checkerboard recovery test on *your*
geometry, coverage and measured noise, and reports what element size, layout and
damping to use, and what wavelength you can claim to have resolved. Start with
the free part:

```python
from slipsolve.calibrate import discretisation_ceiling
for row in discretisation_ceiling(mesh, cells=(20e3, 40e3, 60e3)):
    print(row)          # pure geometry, milliseconds, no data needed
```

That tells you which wavelengths your mesh can carry *at all*, so you never
mistake a parameterisation failure for a resolution limit.

## License

MIT. See `LICENSE`, which retains Xiaoyu Zou's original copyright as that
license requires.
