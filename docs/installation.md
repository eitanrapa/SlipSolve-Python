# Installation

slipsolve needs Python 3.10 or later. It is pure Python on top of numpy and
scipy; nothing is compiled at install time.

## With pip, from GitHub

```bash
pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"
```

With optional extras (see below):

```bash
pip install "slipsolve-python[dev] @ git+https://github.com/eitanrapa/SlipSolve-Python.git"
```

The distribution is called `slipsolve-python`; the package you import is
`slipsolve`:

```python
import slipsolve
```

## With conda

The repository carries an environment file with every dependency, including the
optional ones:

```bash
git clone https://github.com/eitanrapa/SlipSolve-Python.git
cd SlipSolve-Python
conda env create -f environment.yml
conda activate slipsolve
pip install -e .
```

## Dependencies

The required dependencies are installed with the package: `numpy`, `scipy`,
`xarray`, `zarr` (2.18 or later, below 3; a saved model is a zipped Zarr store),
`rioxarray` (CRS handling and GMT `.grd` export), `pyproj` (map projections) and
`matplotlib` (figures, and some interpolation). The package works on points and
meshes and is eager throughout: it does not use dask, and it does not need a SAR
processing stack.

## Optional dependencies

| Extra or package | What it enables |
|---|---|
| `h5py` | Reading GMT 6 `.grd` files, which are netCDF-4 (HDF5) files, in `read_grd` and `LosGrid.from_grd`. Included in the conda environment. Other `.grd` flavours are read through whichever xarray backend is installed (`netcdf4`, `h5netcdf`, `scipy` or `rasterio`). |
| `[dev]` (`pytest`, `pytest-xdist`, `cutde`) | The test suite. `cutde` is an independent implementation of the triangular dislocation, used only as a cross-check; the tests that need it are skipped without it. |
| `[docs]` (`mkdocs-material`, `mkdocstrings[python]`) | Building this documentation. |

## Input data

Reading SAR products is not part of this package. A gridded LOS product is any
object with an xarray Dataset at `.ds` holding the displacement and the look
vectors (see [Observations](guide/observations.md#gridded-line-of-sight-products)).
`LosGrid` wraps a plain Dataset, and `LosGrid.from_grd` builds one from GMT `.grd`
grids. The LOS stacks of the companion package
[nisar_tools](https://github.com/eitanrapa/nisar_tools) can be passed directly.
