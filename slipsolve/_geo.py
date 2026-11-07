"""The small amount of raster plumbing the slip module needs.

Copied from `nisar_tools` rather than imported, so that the package installs on
its own: the rest of it works on points and meshes, is eager throughout and uses
no dask, and a slip inversion should run on ALOS `.grd` files, a plain xarray
Dataset, or anything else with the right variables, without pulling in h5py,
snaphu or a NISAR-specific stack class.

Upstream, to diff against rather than re-derive (a fix there may belong here):

    open_stage          nisar_tools/_base.py
    write_grd           nisar_tools/geo.py
    read_grd            nisar_tools/geo.py
    project_to_latlon   nisar_tools/geo.py
    as_crs              nisar_tools/geo.py
    utm_epsg            nisar_tools/geo.py
    LosGrid             nisar_tools/los.py (the parts of LOSStack used here)
"""

import numpy as np
import xarray as xr

# Registers the `.rio` accessor every CRS-aware call below goes through. Imported
# for the side effect: do not prune it as unused, or `read_grd`, `write_grd` and
# `project_to_latlon` raise `AttributeError: 'DataArray' object has no attribute
# 'rio'`.
import rioxarray  # noqa: F401


def as_crs(value):
    """Normalise a CRS argument to something rasterio and pyproj both accept.

    An integer (or a string of digits) means an EPSG code; anything else is
    passed through. The pass-through matters here: `slipsolve.frame.LocalFrame`
    is a transverse Mercator centred on the study area and has no EPSG code, so
    a grid may be identified by a `pyproj.CRS` object instead.
    """
    if isinstance(value, (int, np.integer)):
        return f"EPSG:{int(value)}"
    if isinstance(value, str) and value.isdigit():
        return f"EPSG:{int(value)}"
    return value


def open_stage(path):
    """Open a Zarr store, restoring the CRS coordinate.

    Zarr does not distinguish coordinates from data variables, so the
    `spatial_ref` that `rio.write_crs` wrote as a coordinate comes back as
    a variable. rioxarray then stops recognising the CRS and every field
    reports `rio.crs is None`, which surfaces far downstream as "Provide a
    CRS-aware DataArray" from anything that reprojects.
    """
    ds = xr.open_zarr(path)
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    return ds


def project_to_latlon(data, x_coords=None, y_coords=None, epsg_code=None):
    """Reproject a 2-D grid from its native CRS to WGS84 lon/lat.

    Accepts a georeferenced `xarray.DataArray`, or a raw 2-D array plus
    `x_coords`/`y_coords`/`epsg_code`. Eager: computes its input.
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
    """Reproject a 2-D native-grid field to lon/lat and write a GMT `.grd`.

    A GMT grid is single-variable NetCDF, so the field is reprojected to WGS84,
    its axes renamed `lat`/`lon` and the variable to `z`. Two rioxarray
    artefacts must be stripped or the file is unusable: the length-1 `band`
    axis a reproject leaves behind, and the `spatial_ref` coordinate, which
    NetCDF would write as a second variable and GMT would then refuse.

    Integer label layers are cast to `float32` with NaN outside coverage: a
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


# ---------------------------------------------------------------------------
# Reading a GMT `.grd`, the inverse of `write_grd`: how ALOS and GMTSAR
# products reach the inversion. Upstream: `nisar_tools/geo.py` (`read_grd`,
# `_read_grd_hdf5`, `utm_epsg`).
# ---------------------------------------------------------------------------

#: xarray engines tried in turn by `read_grd`, best first.
#:
#: A `.grd` is single-variable NetCDF, but which NetCDF matters: GMT 6 writes
#: HDF5-backed NetCDF-4 by default, which scipy's classic-only backend cannot
#: read, while `netcdf4`/`h5netcdf` are optional dependencies. NetCDF-4 is
#: handled instead by `_read_grd_hdf5` through `h5py`.
#:
#: `pygmt` registers a `gmt` engine that reads both, and it is deliberately
#: left out: with `h5py` also imported (two HDF5 libraries in one process) it
#: drops a NetCDF-4 grid's coordinate variables and silently returns pixel
#: indices instead. The data comes back right and only the georeferencing is
#: wrong, so nothing downstream would notice.
_GRD_ENGINES = ("netcdf4", "h5netcdf", "scipy", "rasterio")

#: Leading bytes of an HDF5 file, i.e. of a NetCDF-4 `.grd`.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def utm_epsg(lon, lat):
    """EPSG code of the WGS84 UTM zone containing `(lon, lat)`."""
    zone = int((float(lon) + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _read_grd_hdf5(path):
    """Read a NetCDF-4 (HDF5-backed) `.grd` with `h5py`.

    A GMT grid is one 2-D variable plus its two coordinate variables at the file
    root, so the layout can be recovered without a NetCDF library: the data is
    the only 2-D dataset, and its axes are named by the HDF5 dimension scales it
    points at (`DIMENSION_LIST`), falling back to matching 1-D datasets by
    length. The CRS, when present, rides on a `grid_mapping` dataset's
    `spatial_ref`/`crs_wkt` attribute.
    """
    import h5py

    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(path, "r") as f:
        arrays = {k: v for k, v in f.items() if isinstance(v, h5py.Dataset)}
        grids = [k for k, v in arrays.items() if v.ndim == 2]
        if len(grids) != 1:
            raise ValueError(
                f"{path}: expected exactly one 2-D variable, found {grids or 'none'}")
        var = arrays[grids[0]]

        names = []
        for ref_list in var.attrs.get("DIMENSION_LIST", []):
            names.append(f[ref_list[0]].name.lstrip("/"))
        if len(names) != 2:            # no dimension scales; match by length
            names = [
                next(k for k, v in arrays.items() if v.ndim == 1 and v.shape[0] == n)
                for n in var.shape
            ]

        data = np.asarray(var)
        fill = var.attrs.get("_FillValue")
        if fill is not None and np.issubdtype(data.dtype, np.floating):
            data = np.where(data == np.asarray(fill).ravel()[0], np.nan, data)

        da = xr.DataArray(data, dims=names,
                          coords={n: np.asarray(arrays[n]) for n in names},
                          name=grids[0])
        mapping = var.attrs.get("grid_mapping")
        if mapping is not None:
            attrs = arrays.get(text(mapping), None)
            wkt = attrs and (attrs.attrs.get("spatial_ref")
                             or attrs.attrs.get("crs_wkt"))
            if wkt is not None:
                da = da.rio.write_crs(text(wkt))
    return da


def read_grd(path, engine=None):
    """Read a GMT `.grd` as a CRS-aware 2-D DataArray, the inverse of `write_grd`.

    Axes are renamed to `x`/`y`. The CRS comes from the file when it carries
    one; otherwise the grid is taken as geographic if its axes were named
    `lon`/`lat` (GMT's own convention for a geographic grid) or if they span
    a plausible lon/lat range. `engine` forces one backend instead of trying
    `_GRD_ENGINES` in turn; `engine="hdf5"` forces the `h5py` reader.

    Eager: a `.grd` is a whole raster in one variable, with no chunking to
    preserve.
    """
    with open(path, "rb") as fh:
        is_hdf5 = fh.read(8) == _HDF5_MAGIC

    engines = [engine] if engine else (
        ["hdf5"] if is_hdf5
        else [e for e in _GRD_ENGINES if e in xr.backends.list_engines()])
    da, errors = None, []
    for name in engines:
        try:
            da = (_read_grd_hdf5(path) if name == "hdf5"
                  else xr.open_dataarray(path, engine=name))
            break
        except Exception as exc:            # unreadable this way; try the next
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if da is None:
        raise ValueError(f"Could not read {path} (tried {', '.join(engines)}).\n"
                         + "\n".join(errors))

    da = da.squeeze(drop=True)              # rasterio leaves a length-1 band axis
    geographic = "lon" in da.dims or "lat" in da.dims
    da = da.rename({k: v for k, v in (("lon", "x"), ("lat", "y")) if k in da.dims})
    if da.ndim != 2:
        raise ValueError(f"{path} is not a 2-D grid (dims {da.dims})")

    if da.rio.crs is None:
        x, y = da["x"].values, da["y"].values
        if geographic or (abs(x).max() <= 360.0 and abs(y).max() <= 90.0):
            da = da.rio.write_crs("EPSG:4326")
    return da.rio.write_nodata(np.nan)


#: Sign relating interferometric phase to range change.
#:
#: Interferograms are formed as `ref * conj(sec)`, and with ISCE3's
#: `exp(-j 4 pi r / lambda)` SLC phase that gives
#: `phi = (4 pi / lambda) (r_sec - r_ref)`, so phase decreases as the
#: ground approaches the sensor, and the conversion carries a minus sign.
#:
#: A wrong sign here is invisible to synthetics built as `G m` and inverted
#: with the same convention, which are self-consistent under a global flip; it
#: shows only against a fault of known slip sense.
PHASE_RANGE_SIGN = -1


def phase_to_los(unwrapped, wavelength, sign=1):
    """Unwrapped phase (radians) to line-of-sight displacement (metres).

    `d = -sign * (wavelength / 4pi) * phase`; see `PHASE_RANGE_SIGN`.
    The default makes displacement positive toward the sensor. Pass
    `sign=-1` for data whose fringe sense is inverted, which brings the
    returned values back into that same convention; nothing downstream
    re-applies it.
    """
    return sign * PHASE_RANGE_SIGN * (wavelength / (4.0 * np.pi)) * unwrapped


# ---------------------------------------------------------------------------
# Putting several scenes on one lattice.
#
# The same operation as `nisar_tools`' lazy dask `warp_onto_lattice`, as an eager
# loop: a slip inversion sees only a handful of scenes. Upstream:
# `nisar_tools/stack.py` (`warp_onto_lattice`, `_warp_block`) and
# `nisar_tools/geo.py` (`warp_to_grid`, `grid_transform`, `resampling_from_name`,
# `transform_native_bbox`).
# ---------------------------------------------------------------------------


def resampling_from_name(name):
    """Resolve a rasterio `Resampling` mode by name (e.g. `"bilinear"`)."""
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
    `(x_min, x_max, y_min, y_max)`.
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
    """Warp a `(stack, y, x)` DataArray onto the `(ty, tx)` lattice.

    The stack dimension is taken from `field.dims[0]`, so this serves a
    `time` axis and a `pair` axis alike.

    Integer layers are warped as float32 with nearest resampling and
    refilled with 0. A connected-component or subswath label must never be
    blended (an interpolated label is not a label), and the reprojection's
    out-of-coverage fill is NaN, which no integer dtype can hold. float32 is
    exact to `2**24`, far above any label these products carry.
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


# ---------------------------------------------------------------------------
# The stack object the rest of the package expects: `.ds`, `.crs`, `.x`, `.y`
# and a `type(stack)(ds)` constructor (used by `resample.py` and `calibrate.py`).
# `LosGrid` is those five plus a `.grd` ingest and `.crop`; any object with the
# same five, such as `nisar_tools`' `LOSStack`, works as well.
# ---------------------------------------------------------------------------

#: Metres per unit, for the `units` a `.grd` cannot record.
_LOS_UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _match_grid(da, ref, resampling):
    """Put `da` on `ref`'s lattice, resampling only if it is not already."""
    same = (da.dims == ref.dims
            and all(np.array_equal(da[d].values, ref[d].values) for d in ("y", "x"))
            and da.rio.crs == ref.rio.crs)
    if same:
        return da
    return da.rio.reproject_match(ref, resampling=resampling_from_name(resampling))


def _finalise_look(ds):
    """Validate, renormalise and complete the look geometry.

    Reprojecting three components independently shortens the vector slightly, and
    `los_up == cos(incidence)` only holds for a unit vector, so it is
    renormalised, and the incidence angle derived from it. A norm far from 1 to
    begin with means the grids are not a unit vector at all, which is raised on
    rather than quietly rescaled.
    """
    e, n, u = (np.asarray(ds[f"los_{c}"].values, float)
               for c in ("east", "north", "up"))
    norm = np.sqrt(e * e + n * n + u * u)
    finite = np.isfinite(norm) & (norm > 0)
    if not finite.any():
        raise ValueError("The look-vector grids have no pixel in common with data")

    median = float(np.median(norm[finite]))
    if not 0.9 <= median <= 1.1:
        raise ValueError(
            f"look_e/n/u are not a unit vector (median norm {median:.3g}); "
            "check the three grids are the components of the LOS direction.")

    mean_up = float(np.mean(u[finite]))
    if mean_up <= 0:
        raise ValueError(
            f"Mean look_u is {mean_up:+.3f}, but a target->sensor unit vector "
            "points up (cos of the incidence angle). Pass "
            "look_convention='sensor_to_target' if the grids point sensor->target.")

    scale = np.where(finite, norm, np.nan)
    for c, v in (("east", e), ("north", n), ("up", u)):
        ds[f"los_{c}"] = (("y", "x"), (v / scale).astype(np.float32))
    ds["incidence_angle"] = (
        ("y", "x"),
        np.degrees(np.arccos(np.clip(u / scale, -1.0, 1.0))).astype(np.float32))
    ds["los"] = ds["los"].astype(np.float32)
    return ds


class LosGrid:
    """A gridded LOS product: `los` over `(pair, y, x)` plus its look vectors.

    Wraps an `xarray.Dataset` and exposes the small surface the inversion
    uses. The constructor must keep taking a bare Dataset:
    `slipsolve.resample.resample_to_frame` and the synthetic builders in
    `slipsolve.calibrate` rebuild a stack as `type(stack)(ds)`.
    """

    def __init__(self, ds):
        self.ds = ds

    def __repr__(self):
        return f"LosGrid({dict(self.ds.sizes)}, crs={self.ds.rio.crs})"

    @property
    def crs(self):
        """The grid's CRS, whether or not it has an EPSG code.

        A stack resampled into a `slipsolve.frame.LocalFrame` carries a
        proj4 transverse Mercator with no EPSG code, so `f"EPSG:{...}"`
        cannot describe it and this must be preferred to `attrs["epsg"]`.
        """
        crs = self.ds.rio.crs
        return crs if crs is not None else as_crs(self.ds.attrs["epsg"])

    @property
    def x(self):
        return self.ds["x"].values

    @property
    def y(self):
        return self.ds["y"].values

    @property
    def sizes(self):
        return dict(self.ds.sizes)

    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """A new stack trimmed to a lon/lat box, in whatever CRS this one is in."""
        x_min, x_max, y_min, y_max = transform_native_bbox(
            lon_min, lon_max, lat_min, lat_max, "EPSG:4326", self.crs)
        x, y = self.x, self.y
        x_slice = slice(x_min, x_max) if x[0] <= x[-1] else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y[0] <= y[-1] else slice(y_max, y_min)
        out = self.ds.sel(x=x_slice, y=y_slice)
        out.attrs.update(self.ds.attrs)
        return type(self)(out)

    @classmethod
    def from_grd(cls, los, look_e, look_n, look_u, units="m", sign=1,
                 look_convention="target_to_sensor", epsg=None, resolution=None,
                 resampling="bilinear", direction=None, look_direction=None,
                 wavelength=None):
        """Build a stack from GMT `.grd` grids of displacement and look vector.

        `los` is one displacement grid or a sequence of them (one per pair);
        the three look components are shared by every pair and are resampled onto
        `los`'s lattice if they are not already on it.

        Three things have to be declared because no `.grd` records them, and
        each gives plausible-looking but wrong results if mis-set:

        -   `units`: `"m"` (default), `"cm"` or `"mm"`; a file called
            `los_cm.grd` needs `units="cm"`. Everything downstream works in
            metres, so a factor of 100 here becomes a factor of 100 in the slip.
        -   `sign`: `+1` if the grid is already positive toward the sensor (this
            package's convention, and the sense of a decreasing range); `-1` if
            it is positive away. Applied on load, exactly as `phase_to_los`
            applies it, so what is stored is canonical and nothing downstream
            re-applies it.

            A grid built by the usual GMT recipe
            `grdmath phase 4 DIV PI DIV <wavelength> MUL` carries no minus sign,
            so it is positive away and needs `sign=-1`, the opposite of what
            `PHASE_RANGE_SIGN` gives. Never inherit a sign from a previous
            product of the same track; measure it. Two cheap independent checks:
            the sense of the near-field block motion against the fault's known
            slip sense, and the correlation against another track of the same
            pass and look direction.

        -   `look_convention`: `"target_to_sensor"` (default, this package's) or
            `"sensor_to_target"`, which is negated on load. Checked against the
            data: the vertical component of a target->sensor vector is
            `cos(incidence)` and must be positive.

        `direction` (`"ascending"`/`"descending"`) and `look_direction`
        (`"left"`/`"right"`; ALOS is right-looking, NISAR is left) are carried
        as provenance and are what lets `slipsolve.diagnostics.scene_report`
        check `los_east`'s sign against the pass geometry. Both matter: a
        descending right-looking track has `los_east > 0` where a descending
        left-looking one has `los_east < 0`, so the check is meaningless
        without them.

        The grid is reprojected to a metric CRS if it is geographic: the
        quadtree in `slipsolve.sampling.Observations.from_los` measures
        cell widths against `width_min` in metres, so a lon/lat lattice would
        put `width_min=1000` at a million columns.
        """
        import os

        try:
            scale = _LOS_UNITS[units]
        except KeyError:
            raise ValueError(
                f"units must be one of {sorted(_LOS_UNITS)}, not {units!r}") from None
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, not {sign!r}")
        if look_convention not in ("target_to_sensor", "sensor_to_target"):
            raise ValueError(
                "look_convention must be 'target_to_sensor' or 'sensor_to_target'")

        paths = ([los] if isinstance(los, (str, os.PathLike)) else list(los))
        if not paths:
            raise ValueError("Need at least one LOS grid")
        grids = [read_grd(p) for p in paths]
        ref = grids[0]
        for i, g in enumerate(grids[1:], 1):
            grids[i] = _match_grid(g, ref, resampling)

        look = {f"los_{c}": _match_grid(read_grd(p), ref, resampling)
                for c, p in (("east", look_e), ("north", look_n), ("up", look_u))}
        if look_convention == "sensor_to_target":
            look = {k: -v for k, v in look.items()}

        stacked = xr.concat([g * (scale * sign) for g in grids],
                            dim="pair", coords="minimal")
        ds = xr.Dataset({"los": stacked, **look})

        if ds.rio.crs is None:
            raise ValueError(
                f"{paths[0]} carries no CRS and its coordinates are not lon/lat; "
                "pass grids with a CRS, or reproject them first.")
        if ds.rio.crs.is_geographic:
            if epsg is None:
                epsg = utm_epsg(ds["x"].values.mean(), ds["y"].values.mean())
            ds = ds.rio.reproject(f"EPSG:{int(epsg)}", resolution=resolution,
                                  resampling=resampling_from_name(resampling),
                                  nodata=np.nan)
        epsg = int(ds.rio.crs.to_epsg())

        ds = _finalise_look(ds)
        ds.attrs.update(
            epsg=epsg, direction=direction, look_direction=look_direction,
            wavelength=None if wavelength is None else float(wavelength),
            # Applied above, so the stored field is positive toward the sensor
            # whatever was passed -- nothing downstream re-applies it.
            sign=int(sign), granules=[str(p) for p in paths],
            source="grd", units=units)
        return cls(ds)
