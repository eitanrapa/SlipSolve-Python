"""Xu et al.'s (2025) products as slipsolve grids: two ALOS-2 interferograms and the Sentinel-1 pixel offsets.

Every product is an ISCE-2 geocoded binary (float32, band-interleaved by line) on one 10" grid, described by its
`.rsc` header, with `enu.geo` beside it: the unit vector of its measurement direction (east, north, up). That vector
points along a positive value. For phase and range offsets it runs from the satellite to the ground, so the value is a
range increase; slipsolve wants the line of sight positive toward the sensor, so both the value and the vector are
negated there. Azimuth offsets keep both.
"""
from pathlib import Path

import numpy as np

from slipsolve.readers import los_grid

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "xu2025"

#: product -> (kind, value file). The ascending interferogram is corrected for the troposphere (ERA5) and solid-earth
#: tides only; the descending one also for the ionosphere.
PRODUCTS = {
    "ALOS2_A152_20250211_20250408_IFG": ("phase", "geo_filt_msk_ERA5_SET.unw"),
    "ALOS2_D041_20250216_20250330_IFG": ("phase", "geo_filt_msk_ION_ERA5_SET.unw"),
    "S1_A070_20250322_20250403_OFF_RG": ("range", "offRg.geo"),
    "S1_A070_20250322_20250403_OFF_AZ": ("azimuth", "offAz.geo"),
    "S1_A143_20250327_20250408_OFF_RG": ("range", "offRg.geo"),
    "S1_A143_20250327_20250408_OFF_AZ": ("azimuth", "offAz.geo"),
    "S1_D033_20250319_20250331_OFF_RG": ("range", "offRg.geo"),
    "S1_D033_20250319_20250331_OFF_AZ": ("azimuth", "offAz.geo"),
    "S1_D106_20250324_20250405_OFF_RG": ("range", "offRg.geo"),
    "S1_D106_20250324_20250405_OFF_AZ": ("azimuth", "offAz.geo"),
}


def series_name(product):
    """S1_A070_..._OFF_RG -> S1A070_rg; ALOS2_D041_..._IFG -> A2D041_phase."""
    p = product.split("_")
    sensor = {"S1": "S1", "ALOS2": "A2"}[p[0]]
    kind = {"IFG": "phase", "RG": "rg", "AZ": "az"}[p[-1]]
    return f"{sensor}{p[1]}_{kind}"


def header(path):
    """An `.rsc` header as a dict of strings."""
    out = {}
    for line in Path(str(path) + ".rsc").read_text().splitlines():
        p = line.split(None, 1)
        if len(p) == 2:
            out[p[0]] = p[1].strip()
    return out


def read(path):
    """(values (bands, rows, columns) as float, header) of one binary; its no-data value becomes NaN."""
    h = header(path)
    w, n, b = int(h["WIDTH"]), int(h["FILE_LENGTH"]), int(h.get("BANDS", 1))
    a = np.fromfile(path, dtype="<f4")
    if a.size != w * n * b:
        raise ValueError(f"{path}: {a.size} values for {b} x {n} x {w}")
    a = a.reshape(n, b, w).transpose(1, 0, 2).astype(float)
    nodata = h.get("NO_DATA_VALUE", "none")
    if nodata not in ("none", ""):
        a[a == float(nodata)] = np.nan
    return a, h


def grid(product):
    """One product as a longitude/latitude grid in metres, with its coherence (phase) or its sigma (offsets)."""
    kind, stem = PRODUCTS[product]
    folder = FOLDER / product
    a, h = read(folder / stem)
    if kind == "phase":
        value = a[1] * float(h["WAVELENGTH"]) / (4.0 * np.pi)   # band 2, unwrapped radians -> metres of range change
        value[a[1] == 0] = np.nan                                 # masked pixels are written as 0
        extra = {"coherence": read(folder / "geo_filt.cor")[0][-1]}
    else:
        value = a[0]
        extra = {"sigma": read(folder / stem.replace(".geo", "Std.geo"))[0][0]}
    sign = -1.0 if kind in ("phase", "range") else 1.0
    look = sign * read(folder / "enu.geo")[0]
    los = np.where(np.linalg.norm(look, axis=0) > 0.5, sign * value, np.nan)
    x0, dx, y0, dy = (float(h[k]) for k in ("X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP"))
    lon = x0 + (np.arange(int(h["WIDTH"])) + 0.5) * dx                # X_FIRST / Y_FIRST are pixel corners
    lat = y0 + (np.arange(int(h["FILE_LENGTH"])) + 0.5) * dy
    return los_grid(los, lon, lat, look, name=series_name(product), extra=extra)
