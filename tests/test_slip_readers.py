"""Pins the product readers on small rasters and tables written here.

A LiCSAR interferogram must come back as LOS positive toward the satellite (minus the
phase times the wavelength over 4 pi) with its look vectors and coherence, in either of
LiCSAR's folder layouts; offsets must take their coefficient rasters as look vectors;
both GNSS layouts must read the same table; and `on_lattice` must put every grid on one
lattice and read it back from its cache.
"""

import numpy as np
import pytest

from slipsolve import FaultTrace
from slipsolve.readers import (SENTINEL1_WAVELENGTH, GnssTable, gnss_points, on_lattice, read_geotiff,
                               read_gnss_offsets, read_licsar, read_offsets)

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

LON0, LAT0, STEP = -68.9, 10.8, 0.01
SHAPE = (40, 60)


def _write(path, values, crs="EPSG:4326", shift=0.0, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(STEP, 0.0, LON0 + shift * STEP, 0.0, -STEP, LAT0)
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform) as dst:
        dst.write(values.astype(dtype), 1)


def _phase():
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    phase = 0.05 * xx - 0.02 * yy + 1.0
    phase[:3, :] = 0.0                                   # LiCSAR's no-data
    return phase


@pytest.fixture()
def licsar_flat(tmp_path):
    folder = tmp_path / "014A_05232_242525"
    phase = _phase()
    _write(folder / "20230128_20230209.geo.unw.tif", phase)
    _write(folder / "20230128_20230209.geo.cc.tif", np.full(SHAPE, 128), dtype="uint8")
    for c, v in zip("ENU", (-0.6, -0.1, 0.79)):
        _write(folder / f"014A_05232_242525.geo.{c}.tif", np.full(SHAPE, v), shift=0.5)
    return folder, phase


def test_licsar_is_los_toward_the_satellite(licsar_flat):
    folder, phase = licsar_flat
    grid = read_licsar(folder, "20230128_20230209")
    los = grid.ds["los"].values[0]
    np.testing.assert_allclose(los[3:], -phase[3:] * SENTINEL1_WAVELENGTH / (4 * np.pi), rtol=1e-6)
    assert np.isnan(los[:3]).all(), "a zero phase is a missing pixel"
    np.testing.assert_allclose(grid.ds["los_up"].values[5, 5], 0.79, rtol=1e-6)
    np.testing.assert_allclose(grid.ds["coherence"].values[5, 5], 128 / 255, rtol=1e-6)
    assert grid.ds.attrs["frame"] == "014A_05232_242525" and grid.ds.attrs["sign"] == 1
    assert grid.ds.rio.crs.to_epsg() == 4326


def test_licsar_archive_layout_and_missing_files(tmp_path):
    folder = tmp_path / "frame"
    phase = _phase()
    _write(folder / "P" / "P.geo.unw.tif", phase)
    for c, v in zip("ENU", (0.6, -0.1, 0.79)):
        _write(folder / "metadata" / f"F.geo.{c}.tif", np.full(SHAPE, v))
    grid = read_licsar(folder, "P")
    assert "coherence" not in grid.ds and grid.ds.attrs["frame"] == "F"
    with pytest.raises(FileNotFoundError):
        read_licsar(folder, "Q")


def test_a_look_raster_off_the_grid_is_refused(tmp_path):
    folder = tmp_path / "frame"
    _write(folder / "P.geo.unw.tif", _phase())
    for c in "ENU":
        _write(folder / f"F.geo.{c}.tif", np.full(SHAPE, 0.5), shift=3.0)
    with pytest.raises(ValueError, match="pixel off"):
        read_licsar(folder, "P")


def test_offsets_take_their_coefficients_and_sigma(tmp_path):
    values = np.random.default_rng(0).normal(0.0, 1.0, SHAPE)
    paths = {}
    for key, v in dict(value=values, east=np.full(SHAPE, 0.5), north=np.full(SHAPE, -0.1),
                       up=np.full(SHAPE, 0.86), sigma=np.full(SHAPE, 0.2)).items():
        paths[key] = tmp_path / f"{key}.tif"
        _write(paths[key], v, crs=None)                   # as a release whose rasters carry no CRS
    grid = read_offsets(paths["value"], paths["east"], paths["north"], paths["up"], sigma=paths["sigma"],
                        sign=-1, scale=0.01, name="014A_rng")
    np.testing.assert_allclose(grid.ds["los"].values[0], -0.01 * values, rtol=1e-6)
    np.testing.assert_allclose(grid.ds["sigma"].values, 0.002, rtol=1e-6)
    assert grid.ds.rio.crs.to_epsg() == 4326
    with pytest.raises(ValueError, match="sign"):
        read_offsets(paths["value"], paths["east"], paths["north"], paths["up"], sign=2)
    values_back, x, y, crs = read_geotiff(paths["value"])
    assert crs is None and values_back.shape == SHAPE and np.isclose(x[0], LON0 + 0.5 * STEP)


ROWS = [("AAAA", -68.80, 10.70, 0.10, -0.02, 0.003, 0.002, 0.002, 0.004),
        ("BBBB", -68.50, 10.40, -0.05, 0.01, 0.000, 0.001, 0.001, 0.003),
        ("CCCC", -66.00, 10.40, 0.001, 0.000, 0.000, 0.001, 0.001, 0.003)]


def test_both_gnss_layouts_read_the_same_table(tmp_path):
    unr = tmp_path / "event.txt"
    unr.write_text("Sta  Lon  Lat  de  dn  du  sde  sdn  sdu\n" + "=" * 40 + "\n"
                   + "\n".join(" ".join(str(v) for v in r) for r in ROWS) + "\n")
    csv = tmp_path / "ngl_offsets.csv"
    csv.write_text("sta,lon,lat,dist_km,de_m,dn_m,du_m,sde_m,sdn_m,sdu_m,n_pre,n_post\n"
                   + "\n".join(f"{r[0]},{r[1]},{r[2]},1.0," + ",".join(str(v) for v in r[3:]) + ",10,5"
                               for r in ROWS) + "\n")
    a, b = read_gnss_offsets(unr), read_gnss_offsets(csv)
    assert isinstance(a, GnssTable) and len(a) == len(b) == 3
    for f in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    np.testing.assert_allclose(a.east, [0.10, -0.05, 0.001])


def test_gnss_points_select_and_weight(tmp_path):
    table = GnssTable(*(np.array(c) for c in zip(*ROWS)))
    trace = FaultTrace.from_lonlat(np.array([-68.9, -68.4]), np.array([10.5, 10.55]), name="t")
    frame = trace.local_frame()
    obs = gnss_points(table, frame, components="en", scale=3.0, floor=0.004, traces=[trace],
                      max_distance=100e3, drop=["BBBB"])
    assert obs.n == 2, "BBBB dropped by name, CCCC by distance; two components of AAAA"
    np.testing.assert_allclose(sorted(obs.ds["weight"].values), [1 / 0.006, 1 / 0.006])
    with pytest.raises(ValueError, match="no station"):
        gnss_points(table, frame, drop=["AAAA", "BBBB", "CCCC"])


def test_on_lattice_shares_one_lattice_and_caches(licsar_flat, tmp_path):
    folder, _ = licsar_flat
    trace = FaultTrace.from_lonlat(np.array([-68.8, -68.5]), np.array([10.6, 10.65]), name="t")
    frame = trace.local_frame()
    grids = {"a": read_licsar(folder, "20230128_20230209"), "b": read_licsar(folder, "20230128_20230209")}
    bounds = (-20e3, 20e3, -15e3, 15e3)
    first = on_lattice(grids, frame, spacing=2000.0, bounds=bounds, cache=tmp_path / "work")
    assert list(first) == ["a", "b"]
    np.testing.assert_array_equal(first["a"].ds["x"].values, first["b"].ds["x"].values)
    assert len(list((tmp_path / "work").glob("*.zarr"))) == 2
    again = on_lattice({"a": grids["a"]}, frame, spacing=2000.0, bounds=bounds, cache=tmp_path / "work")
    np.testing.assert_allclose(again["a"].ds["los"].values, first["a"].ds["los"].values, equal_nan=True)
    assert "coherence" in again["a"].ds
