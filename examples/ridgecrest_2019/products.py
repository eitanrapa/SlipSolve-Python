"""The Ridgecrest data as slipsolve grids: UCSD's line-of-sight sets and the Sentinel-2 offsets."""
import warnings
from pathlib import Path

import numpy as np

from slipsolve.readers import los_grid, read_geotiff

HERE = Path(__file__).resolve().parent
UCSD_STEP = 0.005                          # degrees: the lattice the UCSD sets are written on
S2_BLOCK = 25                              # Sentinel-2 offsets: medians of 25 x 25 pixels of 20 m (500 m)


def ucsd(name):
    """One UCSD set as a longitude/latitude grid: LOS in metres, positive toward the sensor, with its own look.

    The files are whitespace columns: longitude, latitude, elevation (m), the unit look vector (east, north, up, ground
    to sensor), LOS (mm) and a flag. No data is written as NaN or as an exact 0. Each 0.005-degree cell takes the
    median of its points (one point per cell on these files), and the look its normalised median.
    """
    import pandas as pd

    d = pd.read_csv(HERE / "ucsd" / f"{name}.lltnde", sep=r"\s+", header=None, na_values=["nan", "NaN"])
    d = d[np.isfinite(d[6]) & (d[6] != 0.0)]
    lon, lat, look, los = d[0].to_numpy(), d[1].to_numpy(), d[[3, 4, 5]].to_numpy().T, d[6].to_numpy()

    x0, y0 = np.floor(lon.min() / UCSD_STEP) * UCSD_STEP, np.floor(lat.min() / UCSD_STEP) * UCSD_STEP
    i = np.floor((lat - y0) / UCSD_STEP + 1e-6).astype(int)
    j = np.floor((lon - x0) / UCSD_STEP + 1e-6).astype(int)
    ny, nx = i.max() + 1, j.max() + 1
    key = i * nx + j
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
    value = np.full(ny * nx, np.nan)
    unit = np.full((3, ny * nx), np.nan)
    for a, b in zip(starts, np.r_[starts[1:], sorted_key.size]):
        idx = order[a:b]
        value[sorted_key[a]] = np.median(los[idx])
        unit[:, sorted_key[a]] = np.median(look[:, idx], axis=1)
    unit /= np.linalg.norm(unit, axis=0)
    x = x0 + (np.arange(nx) + 0.5) * UCSD_STEP
    y = y0 + (np.arange(ny) + 0.5) * UCSD_STEP
    return los_grid(value.reshape(ny, nx) / 1e3, x, y, unit.reshape(3, ny, nx), name=name)


def sentinel2(component):
    """One component ("e" or "n") of the Sentinel-2 offsets (m), as medians of `S2_BLOCK`-pixel blocks on their UTM grid.

    The look is the unit axis of the component, so the grid's LOS is the offset itself.
    """
    tag = {"e": "EW", "n": "NS"}[component]
    a, x, y, crs = read_geotiff(HERE / "s2" / f"COSI_20190628_20190708_{tag}_DETR.tif")
    f = S2_BLOCK
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    blocks = a[:ny, :nx].reshape(ny // f, f, nx // f, f).transpose(0, 2, 1, 3).reshape(ny // f, nx // f, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # blocks with no data become NaN
        b = np.nanmedian(blocks, axis=2)
    dx, dy = x[1] - x[0], y[1] - y[0]
    xb = x[0] - dx / 2 + dx * (np.arange(nx // f) * f + f / 2)
    yb = y[0] - dy / 2 + dy * (np.arange(ny // f) * f + f / 2)
    if yb[0] > yb[-1]:
        yb, b = yb[::-1], b[::-1]
    axis = {"e": (1.0, 0.0, 0.0), "n": (0.0, 1.0, 0.0)}[component]
    look = [np.full_like(b, v) for v in axis]
    return los_grid(b, xb, yb, look, crs=crs, name=f"s2_{component}")
