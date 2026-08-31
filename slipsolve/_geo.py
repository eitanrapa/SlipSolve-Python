"""The small amount of raster plumbing the slip module needs.

⚠️ **Vendored from ``nisar_tools``, deliberately.** This module is otherwise
free of the raster pipeline -- it works on points and meshes, is eager
throughout, and uses no dask at all -- so these two functions were the entire
reason it could not be installed on its own. They are copied rather than
imported so that a slip inversion runs against ALOS ``.grd`` files, a plain
xarray Dataset, or anything else with the right variables, without pulling in
h5py, snaphu, or a NISAR-specific stack class.

Upstream at the time of copying:

    ``open_stage``          nisar_tools/_base.py
    ``write_grd``           nisar_tools/geo.py
    ``project_to_latlon``   nisar_tools/geo.py
    ``as_crs``              nisar_tools/geo.py

Keep them here, together, and diff against upstream rather than re-deriving:
they are small and stable, but ``geo.py`` has been actively worked (the
``.grd`` reader, the h5py/pygmt interaction) and a fix there may belong here.
"""

import numpy as np
import xarray as xr


def as_crs(value):
    """Normalise a CRS argument to something rasterio and pyproj both accept.

    An integer (or a string of digits) means an EPSG code; anything else is
    passed through. The pass-through matters here:
    :class:`~slipsolve.frame.LocalFrame` is a transverse Mercator centred
    on the study area and has **no EPSG code at all**, so a grid may be
    identified by a :class:`pyproj.CRS` object instead.
    """
    if isinstance(value, (int, np.integer)):
        return f"EPSG:{int(value)}"
    if isinstance(value, str) and value.isdigit():
        return f"EPSG:{int(value)}"
    return value


def open_stage(path):
    """Open a Zarr store, restoring the CRS coordinate.

    ⚠️ Zarr does not distinguish coordinates from data variables, so the
    ``spatial_ref`` that ``rio.write_crs`` wrote as a *coordinate* comes back as
    a *variable*. rioxarray then stops recognising the CRS and every field
    reports ``rio.crs is None``, which surfaces far downstream as "Provide a
    CRS-aware DataArray" from anything that reprojects.
    """
    ds = xr.open_zarr(path)
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    return ds


def project_to_latlon(data, x_coords=None, y_coords=None, epsg_code=None):
    """Reproject a 2-D grid from its native CRS to WGS84 lon/lat.

    Accepts a georeferenced :class:`xarray.DataArray`, or a raw 2-D array plus
    ``x_coords``/``y_coords``/``epsg_code``. Eager: computes its input.
    """
    if isinstance(data, xr.DataArray) and data.rio.crs is not None:
        da = data
    else:
        if x_coords is None or y_coords is None or epsg_code is None:
            raise ValueError(
                "Provide a CRS-aware DataArray, or data with x_coords, "
                "y_coords and epsg_code."
            )
        da = xr.DataArray(
            np.asarray(data), coords={"y": y_coords, "x": x_coords},
            dims=["y", "x"]).rio.write_crs(f"EPSG:{epsg_code}")
    return da.rio.reproject("EPSG:4326")


def write_grd(field, path):
    """Reproject a 2-D native-grid field to lon/lat and write a GMT ``.grd``.

    A GMT grid is single-variable NetCDF, so the field is reprojected to WGS84,
    its axes renamed ``lat``/``lon`` and the variable to ``z``. Two rioxarray
    artefacts must be stripped or the file is unusable: the length-1 ``band``
    axis a reproject leaves behind, and the ``spatial_ref`` coordinate, which
    NetCDF would write as a second variable and GMT would then refuse.

    Integer label layers are cast to ``float32`` with NaN outside coverage: a
    GMT grid is float anyway, uint32 overflows classic NetCDF's int32, and a NaN
    edge beats an integer sentinel warped in by the reprojection.
    """
    if np.issubdtype(field.dtype, np.integer):
        field = field.astype(np.float32).rio.write_nodata(np.nan)
    g = project_to_latlon(field)                       # eager; EPSG:4326
    g = g.squeeze(drop=True)                           # drop the band axis
    g = g.drop_vars("spatial_ref", errors="ignore")    # else a 2nd variable
    g = g.rename({"y": "lat", "x": "lon"}).rename("z")
    g.to_netcdf(path)
    return path


#: Sign relating interferometric phase to range change.
#:
#: ⚠️ Interferograms are formed as ``ref * conj(sec)``, and with ISCE3's
#: ``exp(-j 4 pi r / lambda)`` SLC phase that gives
#: ``phi = (4 pi / lambda) (r_sec - r_ref)`` -- so phase *decreases* as the
#: ground approaches the sensor, and the conversion carries a minus sign.
#:
#: This was wrong in every LOS product this package produced before 2026-07-28,
#: and no test caught it: the synthetic fixtures built data as ``G m`` and
#: inverted the same convention, so they were self-consistent under a global
#: flip. It was found because inverting a known **dextral** fault recovered
#: **sinistral** slip with 86% of elements pinned on the polarity bound. Forcing
#: this to ``+1`` drops the recovery test's variance reduction from >95% to 4.9%.
PHASE_RANGE_SIGN = -1


def phase_to_los(unwrapped, wavelength, sign=1):
    """Unwrapped phase (radians) to line-of-sight displacement (metres).

    ``d = -sign * (wavelength / 4pi) * phase``; see :data:`PHASE_RANGE_SIGN`.
    The default makes displacement positive **toward** the sensor. Pass
    ``sign=-1`` for data whose fringe sense is inverted, which brings the
    returned values back into that same convention -- nothing downstream
    re-applies it.

    Vendored alongside the rest of this module so a standalone slip inversion
    owns the convention its correctness rests on, rather than inheriting it.
    """
    return sign * PHASE_RANGE_SIGN * (wavelength / (4.0 * np.pi)) * unwrapped


# ---------------------------------------------------------------------------
# Putting several scenes on one lattice.
#
# ⚠️ The raster package's `warp_onto_lattice` is a dask `map_blocks` kernel,
# written so a hundred-gigabyte stack can be warped lazily. This module is eager
# throughout and a slip inversion sees a handful of scenes, so the loop below is
# the same operation without the dependency -- and without the laziness, which
# was never doing anything here anyway. Upstream: `nisar_tools/stack.py`
# (`warp_onto_lattice`, `_warp_block`) and `nisar_tools/geo.py` (`warp_to_grid`,
# `grid_transform`, `resampling_from_name`, `transform_native_bbox`).
# ---------------------------------------------------------------------------


def resampling_from_name(name):
    """Resolve a rasterio ``Resampling`` mode by name (e.g. ``"bilinear"``)."""
    from rasterio.enums import Resampling

    try:
        return Resampling[name]
    except KeyError:
        valid = ", ".join(r.name for r in Resampling)
        raise ValueError(
            f"Unknown resampling {name!r}; expected one of: {valid}") from None


def grid_transform(x_coords, y_coords):
    """Affine geotransform for a regular grid, from its pixel-centre coords."""
    from affine import Affine

    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    return Affine(dx, 0.0, float(x_coords[0]) - dx / 2,
                  0.0, dy, float(y_coords[0]) - dy / 2)


def transform_native_bbox(x_min, x_max, y_min, y_max, src_epsg, dst_epsg):
    """Transform a projected bbox into another projected CRS.

    Edge-densified, so the result covers the whole warped footprint even though
    a projected rectangle's edges curve in the target CRS. Returns
    ``(x_min, x_max, y_min, y_max)``.
    """
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds(
        as_crs(src_epsg), as_crs(dst_epsg), x_min, y_min, x_max, y_max)
    return left, right, bottom, top


def warp_to_grid(arr, src_transform, src_epsg, dst_transform, dst_epsg,
                 dst_shape, resampling="bilinear"):
    """Warp one 2-D array between projected grids. Eager, whole-frame.

    Pixels with no source coverage come back NaN.
    """
    from rasterio.warp import reproject

    mode = resampling_from_name(resampling)
    src = np.ascontiguousarray(arr, dtype=np.float32)
    dst = np.full(tuple(dst_shape), np.nan, dtype=np.float32)
    reproject(src, dst, src_transform=src_transform, src_crs=as_crs(src_epsg),
              dst_transform=dst_transform, dst_crs=as_crs(dst_epsg),
              src_nodata=np.nan, dst_nodata=np.nan, resampling=mode)
    return dst


def warp_onto_lattice(field, tx, ty, src_epsg, dst_epsg, resampling="bilinear"):
    """Warp a ``(stack, y, x)`` DataArray onto the ``(ty, tx)`` lattice.

    The stack dimension is taken from ``field.dims[0]``, so this serves a
    ``time`` axis and a ``pair`` axis alike.

    ⚠️ **Integer layers are warped as float32 with nearest resampling and
    refilled with 0.** A connected-component or subswath label must never be
    blended -- an interpolated label is not a label -- and the reprojection's
    out-of-coverage fill is NaN, which no integer dtype can hold. float32 is
    exact to 2**24, far above any label these products carry.
    """
    resampling_from_name(resampling)          # fail now, not deep in the loop

    stack_dim = field.dims[0]
    sx, sy = field["x"].values, field["y"].values
    integer = np.issubdtype(field.dtype, np.integer)
    if integer:
        resampling = "nearest"

    src_transform = grid_transform(sx, sy)
    dst_transform = grid_transform(tx, ty)
    planes = [
        warp_to_grid(np.asarray(field.isel({stack_dim: i}).values),
                     src_transform, src_epsg, dst_transform, dst_epsg,
                     (len(ty), len(tx)), resampling)
        for i in range(field.sizes[stack_dim])
    ]
    out = xr.DataArray(
        np.stack(planes), dims=(stack_dim, "y", "x"),
        coords={stack_dim: field[stack_dim].values, "y": ty, "x": tx},
        name=field.name)
    if integer:
        out = out.fillna(0).astype(field.dtype)
    return out.rio.write_crs(as_crs(dst_epsg))
