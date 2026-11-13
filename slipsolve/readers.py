"""Readers for the data products slip inversions use most.

Rasters become a `LosGrid` on their own geographic grid: `los` in metres, positive toward
the sensor, with look vectors from the ground to the sensor. `on_lattice` then puts
every raster on one lattice in a `LocalFrame`, the form `slipsolve.production` expects.
GNSS offset tables become a `GnssTable`, and `gnss_points` turns one into observation
rows.

| product | reader |
|---|---|
| a LiCSAR interferogram: `<pair>.geo.unw.tif`, `.cc.tif` and the frame's `.geo.{E,N,U}.tif` | `read_licsar` |
| pixel offsets as a GeoTIFF, with east, north and up coefficient rasters and optionally a sigma raster | `read_offsets` |
| any single-band GeoTIFF | `read_geotiff` |
| GNSS offsets: the Nevada Geodetic Laboratory's per-event files, or a CSV with station, longitude, latitude, the three displacements and their sigmas | `read_gnss_offsets` |

```python
from slipsolve import FaultTrace
from slipsolve.readers import gnss_points, on_lattice, read_gnss_offsets, read_licsar

frame = FaultTrace.from_kml("rupture.kml").local_frame()
grids = {"S1_014A": read_licsar("licsar/014A_05232_242525", "20230128_20230209")}
grids = on_lattice(grids, frame, spacing=250.0, bounds=(-200e3, 200e3, -150e3, 150e3), cache="work")
gnss = gnss_points(read_gnss_offsets("gnss/event.txt"), frame, components="en", scale=3.0)
```
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._geo import LosGrid

#: Sentinel-1's C-band wavelength, m.
SENTINEL1_WAVELENGTH = 0.05546576


def read_geotiff(path, band=1, nodata=True):
    """`(values, x, y, crs)`: one band of a GeoTIFF, its pixel-centre coordinates and its CRS (None if it has none).

    The no-data value becomes NaN unless `nodata` is False. Rotated or sheared transforms
    are refused.
    """
    import rasterio

    with rasterio.open(path) as src:
        values = src.read(band).astype(float)
        t, crs, missing = src.transform, src.crs, src.nodata
    if t.b or t.d:
        raise ValueError(f"{path}: a rotated or sheared grid is not supported")
    if nodata and missing is not None and np.isfinite(missing):
        values[values == missing] = np.nan
    ny, nx = values.shape
    x = t.c + (np.arange(nx) + 0.5) * t.a
    y = t.f + (np.arange(ny) + 0.5) * t.e
    return values, x, y, crs


def los_grid(los, x, y, look, crs="EPSG:4326", name="track", extra=None, **attrs):
    """A `LosGrid` from one displacement raster and its three look components.

    `los` is positive toward the sensor and `look` is `(east, north, up)` from the
    ground to the sensor, each on the grid `(y, x)`. Pixels where the displacement or
    a look component is not finite are NaN throughout. `extra` adds variables on the
    same grid (coherence, sigma).
    """
    import rioxarray  # noqa: F401
    import xarray as xr

    valid = np.isfinite(los) & np.all([np.isfinite(v) for v in look], axis=0)
    data = {"los": (("pair", "y", "x"), np.where(valid, los, np.nan)[None].astype(np.float32))}
    for key, v in zip(("los_east", "los_north", "los_up"), look):
        data[key] = (("y", "x"), np.where(valid, v, np.nan).astype(np.float32))
    for key, v in (extra or {}).items():
        data[key] = (("y", "x"), np.where(valid, v, np.nan).astype(np.float32))
    ds = xr.Dataset(data, coords={"pair": [0], "y": np.asarray(y, float), "x": np.asarray(x, float)})
    ds = ds.rio.write_crs(crs)
    epsg = ds.rio.crs.to_epsg() if ds.rio.crs is not None else None
    ds.attrs.update({"sign": 1, "direction": name, **({"epsg": int(epsg)} if epsg else {}), **attrs})
    return LosGrid(ds)


def _same_grid(a, b, what):
    """Two rasters on one grid, allowing the half-pixel offset some products' companion files carry."""
    va, xa, ya = a[:3]
    vb, xb, yb = b[:3]
    if va.shape != vb.shape:
        raise ValueError(f"{what}: shape {vb.shape}, not the data's {va.shape}")
    dx = abs(xa[1] - xa[0]) if xa.size > 1 else 1.0
    dy = abs(ya[1] - ya[0]) if ya.size > 1 else 1.0
    if np.abs(xa - xb).max() > dx or np.abs(ya - yb).max() > dy:
        raise ValueError(f"{what}: more than a pixel off the data's grid")


def _first(*candidates):
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def read_licsar(folder, pair, frame=None, wavelength=SENTINEL1_WAVELENGTH, name=None):
    """A LiCSAR interferogram as a `LosGrid`, with its coherence.

    `folder` holds `<pair>.geo.unw.tif` and optionally `<pair>.geo.cc.tif`, either
    directly or in a `<pair>/` subfolder (LiCSAR's archive layout). The look rasters
    `<frame>.geo.E.tif`, `.N.tif` and `.U.tif` sit in `folder` or its `metadata/`
    subfolder; with `frame` None the one set found there is used.

    The unwrapped phase becomes `los = -phase * wavelength / (4 pi)`, positive toward
    the satellite. LiCSAR's E, N and U point from the ground to the satellite, as this
    package's look vectors do. A zero phase or look value marks a missing pixel.
    Coherence stored as a byte (0-255) is scaled to 0-1, and a zero there is a coherence
    of zero wherever the phase is valid, whatever the raster's no-data tag says. Rasters
    without a CRS are taken as geographic (EPSG:4326). Measure the sign against GNSS or
    the model before trusting it on a new product.
    """
    folder = Path(folder)
    unw_path = _first(folder / f"{pair}.geo.unw.tif", folder / pair / f"{pair}.geo.unw.tif")
    if unw_path is None:
        raise FileNotFoundError(f"no {pair}.geo.unw.tif in {folder} or {folder / pair}")
    cc_path = _first(folder / f"{pair}.geo.cc.tif", folder / pair / f"{pair}.geo.cc.tif")
    if frame is None:
        found = sorted({p.name[:-len(".geo.E.tif")] for d in (folder, folder / "metadata") if d.exists()
                        for p in d.glob("*.geo.E.tif")})
        if len(found) != 1:
            raise FileNotFoundError(f"{folder}: found look rasters for {found or 'no frame'}; pass frame=")
        frame = found[0]
    looks = []
    for c in "ENU":
        path = _first(folder / f"{frame}.geo.{c}.tif", folder / "metadata" / f"{frame}.geo.{c}.tif")
        if path is None:
            raise FileNotFoundError(f"no {frame}.geo.{c}.tif in {folder} or {folder / 'metadata'}")
        looks.append(read_geotiff(path))
    unw = read_geotiff(unw_path)
    for c, look in zip("ENU", looks):
        _same_grid(unw, look, f"{frame}.geo.{c}.tif")
    phase, x, y, crs = unw
    look = [np.where(v[0] == 0, np.nan, v[0]) for v in looks]
    valid = np.isfinite(phase) & (phase != 0)
    los = np.where(valid, -phase * wavelength / (4.0 * np.pi), np.nan)
    extra = {}
    if cc_path is not None:
        cc = read_geotiff(cc_path, nodata=False)
        _same_grid(unw, cc, f"{pair}.geo.cc.tif")
        extra["coherence"] = cc[0] / 255.0 if np.nanmax(cc[0]) > 1.5 else cc[0]
    return los_grid(los, x, y, look, crs=crs or "EPSG:4326", name=name or frame, extra=extra,
                    source="LiCSAR", pair=str(pair), frame=str(frame), wavelength=float(wavelength))


def read_offsets(value, east, north, up, sigma=None, sign=1, scale=1.0, name="offsets"):
    """Pixel offsets (or any displacement) as a `LosGrid`, from a raster and its east, north and up coefficients.

    The coefficients are the unit vector each pixel's value measures: `value = d_east *
    east + d_north * north + d_up * up`. For a range offset that is the look vector from
    the ground to the sensor; for an azimuth offset, the horizontal along-track
    direction. `sign` (+1 or -1) and `scale` (the factor to metres) are applied on load.
    `sigma`, a raster of the offsets' uncertainty in the same units, rides along as a
    variable. Rasters without a CRS are taken as geographic (EPSG:4326).
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, not {sign!r}")
    data = read_geotiff(value)
    looks = [read_geotiff(p) for p in (east, north, up)]
    for label, look in zip(("east", "north", "up"), looks):
        _same_grid(data, look, f"the {label} coefficients")
    extra = {}
    if sigma is not None:
        s = read_geotiff(sigma)
        _same_grid(data, s, "the sigma raster")
        extra["sigma"] = s[0] * float(scale)
    values, x, y, crs = data
    return los_grid(sign * float(scale) * values, x, y, [v[0] for v in looks], crs=crs or "EPSG:4326", name=name,
                    extra=extra, source=str(Path(value).name))


def on_lattice(grids, frame, spacing, bounds, resampling="median", cache=None):
    """`{name: LosGrid}` on one lattice in `frame`, each resampled from its native grid once.

    `bounds` is `(x_min, x_max, y_min, y_max)` in the frame, so the lattice is fixed by
    `bounds` and `spacing` alone and every grid lands on the same one. The default
    `"median"` is the symmetric median, which suits offsets with outliers; rasterio's
    `"med"` is biased low (`slipsolve.resample.resample_all`). With `cache` (a folder), each
    result is kept as a zarr store named by the grid, the frame, the spacing and the
    bounds, and read from there next time.
    """
    import xarray as xr

    from .resample import resample_all

    key = None
    if cache is not None:
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(np.asarray([frame.origin_lon, frame.origin_lat, spacing, *bounds], float).tobytes())
        key = f"{h.hexdigest()[:12]}_{resampling}"
    out, todo = {}, {}
    for name, grid in grids.items():
        path = cache / f"{name}_{key}.zarr" if cache is not None else None
        if path is not None and path.exists():
            out[name] = LosGrid(xr.open_zarr(path).load().rio.write_crs(frame.local_crs))
        else:
            todo[name] = grid
    if todo:
        done = resample_all(todo, frame, spacing=spacing, bounds=tuple(bounds), resampling=resampling)
        for name, grid in done.items():
            grid.ds.attrs["frame"] = frame.to_dict()
            if cache is not None:
                clean = grid.ds.drop_vars("spatial_ref", errors="ignore")
                for v in clean.variables.values():
                    v.encoding.clear()
                clean.to_zarr(cache / f"{name}_{key}.zarr", mode="w")
            out[name] = grid
    return {name: out[name] for name in grids}


@dataclass
class GnssTable:
    """GNSS displacements at stations, in metres.

    Attributes:
        station: The station names.
        lon: Longitudes, degrees.
        lat: Latitudes, degrees.
        east: Eastward displacement, m.
        north: Northward displacement, m.
        up: Upward displacement, m.
        sigma_e: The east component's standard deviation, m.
        sigma_n: The north component's standard deviation, m.
        sigma_u: The up component's standard deviation, m.
    """

    station: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    sigma_e: np.ndarray
    sigma_n: np.ndarray
    sigma_u: np.ndarray

    def __len__(self):
        return int(np.size(self.station))

    def select(self, keep):
        """The stations where the boolean mask `keep` is true."""
        keep = np.asarray(keep, bool)
        return GnssTable(*(np.asarray(getattr(self, f))[keep] for f in self.__dataclass_fields__))


_GNSS_COLUMNS = ("de_m", "dn_m", "du_m", "sde_m", "sdn_m", "sdu_m")


def read_gnss_offsets(path):
    """A `GnssTable` from a GNSS offsets file.

    Two layouts are read. The Nevada Geodetic Laboratory's per-event files are
    whitespace columns `Sta Lon Lat de dn du sde sdn sdu` (m), whose header and rule
    lines are skipped. A comma-separated file needs a header naming `sta`, `lon`, `lat`,
    `de_m`, `dn_m`, `du_m`, `sde_m`, `sdn_m` and `sdu_m`; other columns are ignored.
    """
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if "," in lines[0]:
        header = [h.strip().lower() for h in lines[0].split(",")]
        missing = [c for c in ("sta", "lon", "lat") + _GNSS_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: the header lacks {missing}")
        rows = [line.split(",") for line in lines[1:]]
        col = {c: header.index(c) for c in ("sta", "lon", "lat") + _GNSS_COLUMNS}
        station = np.array([r[col["sta"]].strip() for r in rows])
        numbers = np.array([[float(r[col[c]]) for c in ("lon", "lat") + _GNSS_COLUMNS] for r in rows])
    else:
        names, numbers = [], []
        for line in lines:
            parts = line.split()
            if len(parts) < 9 or parts[0].lower() == "sta" or parts[0].startswith(("=", "-", "#")):
                continue
            try:
                values = [float(v) for v in parts[1:9]]
            except ValueError:
                continue
            names.append(parts[0])
            numbers.append(values)
        station, numbers = np.array(names), np.array(numbers)
    if not len(station):
        raise ValueError(f"{path}: no stations read")
    return GnssTable(station, *numbers.T)


def gnss_points(table, frame, components="en", scale=1.0, floor=0.0, traces=None, max_distance=None,
                min_distance=0.0, drop=(), name="gnss"):
    """Observation rows from a `GnssTable`, one per station and component.

    Stations named in `drop` are left out, and with `traces` (a list of `FaultTrace`)
    so are those farther than `max_distance` or nearer than `min_distance` (m) to every
    trace. Each sigma is multiplied by `scale` and raised to `floor`
    (`slipsolve.gnss.gnss_observations`).
    """
    from .gnss import gnss_observations

    keep = ~np.isin(table.station, list(drop))
    if traces is not None and (max_distance is not None or min_distance):
        x, y = frame.to_local(table.lon, table.lat)
        dist = np.min([np.asarray(t.distance(x, y, frame), float) for t in traces], axis=0)
        if max_distance is not None:
            keep &= dist <= max_distance
        keep &= dist >= min_distance
    t = table.select(keep)
    if not len(t):
        raise ValueError("no station is left after the selection")
    return gnss_observations(t.lon, t.lat, t.east, t.north, t.up, t.sigma_e, t.sigma_n, t.sigma_u, frame, name=name,
                             components=components, scale=scale, floor=floor)
