"""Pins the shared lattice: :mod:`slipsolve.resample` and ``LocalFrame``'s
two coordinate systems.

The whole point of resampling is that every track ends up on **one** grid, so
most of what is checked here is sameness rather than correctness of any single
value. The rest guards the distinction that made this subtle: a ``LocalFrame`` is
a projection *with an origin subtracted*, so ``frame.crs`` and the local metres
everything else speaks are different numbers -- on the Venezuela frame, 500 km
and 1167 km apart.
"""

import numpy as np
import pytest
from pyproj import Transformer
from slip_synthetic import forward_los_stack, tapered_slip

from slipsolve import (
    FaultMesh,
    FaultTrace,
    LocalFrame,
    Observations,
)
from slipsolve import (
    ARCSEC_10,
    frame_lattice,
    resample_all,
    resample_to_frame,
)

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    return trace, frame, mesh


@pytest.fixture(scope="module")
def scenes(setup):
    trace, frame, mesh = setup
    truth = tapered_slip(mesh, peak=-2.0)
    return {
        g: forward_los_stack(mesh, truth, trace, frame, geometry=g, spacing=5000.0,
                             noise=0.004, nan_rows=3, seed=1)
        for g in ("asc", "desc")
    }


# -- the two coordinate systems -----------------------------------------------

def test_local_crs_puts_the_origin_at_zero(setup):
    """``local_crs`` is the frame's projection with the origin folded in, so a
    local coordinate is directly valid in it."""
    _, frame, _ = setup
    to_lonlat = Transformer.from_crs(frame.local_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(0.0, 0.0)
    assert lon == pytest.approx(frame.origin_lon, abs=1e-6)
    assert lat == pytest.approx(frame.origin_lat, abs=1e-6)


def test_local_crs_agrees_with_to_local_everywhere(setup):
    """The CRS and the frame's own arithmetic must be the same map."""
    _, frame, _ = setup
    lon = np.linspace(-69.5, -65.5, 17)
    lat = np.linspace(9.5, 11.5, 17)
    grid_lon, grid_lat = (a.ravel() for a in np.meshgrid(lon, lat))

    expected_x, expected_y = frame.to_local(grid_lon, grid_lat)
    forward = Transformer.from_crs("EPSG:4326", frame.local_crs, always_xy=True)
    x, y = forward.transform(grid_lon, grid_lat)
    np.testing.assert_allclose(x, expected_x, atol=1e-6)
    np.testing.assert_allclose(y, expected_y, atol=1e-6)


def test_crs_and_local_crs_differ_by_the_origin(setup):
    """Tagging local metres with ``crs`` is the bug ``local_crs`` exists to stop.

    Pinned with the actual magnitude, because "they differ" understates it: on
    this frame the two disagree by a UTM false easting and 1167 km of northing,
    which put an exported grid in the wrong hemisphere.
    """
    _, frame, _ = setup
    x, y = frame.to_projected(0.0, 0.0)
    assert float(x) == pytest.approx(500e3, rel=1e-3)
    assert float(y) > 1000e3
    np.testing.assert_allclose(frame.from_projected(x, y), (0.0, 0.0), atol=1e-6)


def test_from_epsg_and_local_crs_land_in_the_same_place(setup):
    """A UTM stack transformed by hand and a stack resampled into ``local_crs``
    have to agree, or samples from two tracks sit in different worlds."""
    _, frame, _ = setup
    lon, lat = np.array([-68.0, -67.0]), np.array([10.2, 10.9])
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    ux, uy = utm.transform(lon, lat)
    np.testing.assert_allclose(frame.from_epsg(ux, uy, 32619),
                               frame.to_local(lon, lat), atol=1e-3)


# -- the lattice ---------------------------------------------------------------

def test_lattice_is_local_metres_on_whole_multiples(setup, scenes):
    _, frame, _ = setup
    x, y = frame_lattice(list(scenes.values()), frame, spacing=5000.0)
    np.testing.assert_allclose(x % 5000.0, 0.0, atol=1e-6)
    np.testing.assert_allclose(y % 5000.0, 0.0, atol=1e-6)
    assert y[0] > y[-1]                      # north-up, like every grid here


def test_lattice_does_not_depend_on_which_stacks_were_passed(setup, scenes):
    """The grid phase is a property of the frame and the spacing alone, so a
    subset of the tracks must not shift it -- otherwise two runs of the same
    study produce grids that cannot be compared."""
    _, frame, _ = setup
    everything = frame_lattice(list(scenes.values()), frame, spacing=5000.0)
    bounded = frame_lattice([scenes["asc"]], frame, spacing=5000.0)
    for full, part in zip(everything, bounded):
        shared = np.intersect1d(np.round(full, 6), np.round(part, 6))
        assert shared.size > 10
        # Every shared coordinate is on the same rung, not merely nearby.
        np.testing.assert_allclose(shared % 5000.0, 0.0, atol=1e-6)


def test_arcsec_10_is_ten_arcseconds():
    """The default spacing is the ALOS-2 posting. The measured D134 grid is
    0.00277836 deg = 10.0021 arcsec, hence the 3e-4 rather than an exact match."""
    assert ARCSEC_10 == pytest.approx(309.2, abs=0.5)
    assert 0.00277836 * 111320.0 == pytest.approx(ARCSEC_10, rel=3e-4)


# -- resampling ----------------------------------------------------------------

def test_every_track_lands_on_an_identical_grid(setup, scenes):
    _, frame, _ = setup
    out = resample_all(scenes, frame, spacing=5000.0)
    grids = [(o.ds["x"].values, o.ds["y"].values) for o in out.values()]
    for x, y in grids[1:]:
        np.testing.assert_array_equal(x, grids[0][0])
        np.testing.assert_array_equal(y, grids[0][1])


def test_resampled_stack_carries_the_frame_and_no_epsg(setup, scenes):
    """``epsg`` is dropped rather than left stale: the frame's transverse
    Mercator has no EPSG code, and a wrong one is worse than none."""
    _, frame, _ = setup
    out = resample_to_frame(scenes["asc"], frame, spacing=5000.0)
    assert out.ds.attrs["frame"] == frame.to_dict()
    assert "epsg" not in out.ds.attrs
    assert out.crs is not None
    with pytest.raises(AttributeError, match="LocalFrame"):
        out.epsg


def test_resampling_preserves_the_look_vectors(setup, scenes):
    """The look geometry is a direction; warping must not rescale it."""
    _, frame, _ = setup
    out = resample_to_frame(scenes["desc"], frame, spacing=5000.0)
    components = [out.ds[f"los_{c}"].values for c in ("east", "north", "up")]
    norm = np.sqrt(sum(c.astype(float) ** 2 for c in components))
    finite = np.isfinite(norm)
    assert finite.mean() > 0.8
    np.testing.assert_allclose(norm[finite], 1.0, atol=1e-4)
    for c, source in zip(components, ("east", "north", "up")):
        expected = float(scenes["desc"].ds[f"los_{source}"].values.mean())
        assert np.nanmean(c) == pytest.approx(expected, abs=1e-4)


def test_sampling_a_resampled_stack_matches_the_utm_original(setup, scenes):
    """The seam that matters: ``from_los`` must read a frame-gridded raster's
    ``x``/``y`` as local metres. Reading them as projected values put every
    observation ~1200 km from the fault while still inverting to a plausible
    number, which is exactly the kind of failure this pins."""
    trace, frame, _ = setup
    kwargs = dict(name="asc", frame=frame, trace=trace, rms_min=0.006,
                  width_min=6000.0, width_max=40000.0, exclude_within=5000.0)
    native = Observations.from_los(scenes["asc"], **kwargs)
    warped = Observations.from_los(
        resample_to_frame(scenes["asc"], frame, spacing=5000.0), **kwargs
    )
    for axis in ("x", "y"):
        span = native.ds[axis].values
        assert warped.ds[axis].values.min() == pytest.approx(span.min(), abs=1e4)
        assert warped.ds[axis].values.max() == pytest.approx(span.max(), abs=1e4)
    assert warped.frame.matches(frame)


def test_surface_displacement_is_georeferenced_where_the_fault_is(setup, scenes):
    """A regression guard for the ``crs``/``local_crs`` mix-up.

    ``surface_displacement`` returns local metres; tagging them with the frame's
    bare projection placed the exported field at latitude -0.06 instead of +10.5,
    and ``SlipModel.to_grd`` baked that into a lon/lat file.
    """
    from slipsolve import SlipInversion

    trace, frame, mesh = setup
    obs = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                rms_min=0.006, width_min=8000.0, width_max=40000.0,
                                exclude_within=8000.0)
    model = SlipInversion(mesh, obs).solve(smoothing=0.3)
    grid = model.surface_displacement(spacing=10000.0, pad=30e3)

    to_lonlat = Transformer.from_crs(grid.rio.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(float(grid.x.values.mean()),
                                   float(grid.y.values.mean()))
    assert lon == pytest.approx(_LON.mean(), abs=1.0)
    assert lat == pytest.approx(_LAT.mean(), abs=1.0)


def test_to_grd_writes_the_field_where_the_fault_is(setup, scenes, tmp_path):
    """The half of the mix-up the test above did not reach.

    ``surface_displacement`` tagging the grid correctly is not enough: ``to_grd``
    re-tagged it with the frame's bare projection on the way out, which put the
    written file 10.6 degrees of latitude -- about 1180 km -- south of the fault.
    Checking the CRS of the in-memory grid could not catch that; only reading the
    file back can.
    """
    import xarray as _xr
    from slipsolve import SlipInversion

    trace, frame, mesh = setup
    obs = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                rms_min=0.006, width_min=8000.0, width_max=40000.0,
                                exclude_within=8000.0)
    model = SlipInversion(mesh, obs).solve(smoothing=0.3)
    (path, *_) = model.to_grd(tmp_path / "grd", fields=("ux",),
                              spacing=10000.0, pad=30e3)

    grid = _xr.open_dataarray(path)
    y_name, x_name = grid.dims[-2], grid.dims[-1]
    assert float(grid[x_name].mean()) == pytest.approx(_LON.mean(), abs=1.0)
    assert float(grid[y_name].mean()) == pytest.approx(_LAT.mean(), abs=1.0)


def test_frame_round_trips_through_a_dict(setup):
    _, frame, _ = setup
    rebuilt = LocalFrame.from_dict(frame.to_dict())
    np.testing.assert_allclose(rebuilt.to_projected(0.0, 0.0),
                               frame.to_projected(0.0, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# The `.grd` round trip.
#
# `write_grd` came across in the extraction and `read_grd` did not, so for a
# while a standalone slipsolve could write a GMT grid it could not read back --
# and `.grd` is how ALOS and GMTSAR products arrive. These pin both halves and
# the two things a `.grd` cannot record, `units` and `sign`.
# ---------------------------------------------------------------------------


@pytest.fixture
def grd_scene(tmp_path):
    """A geographic `.grd` quadruple: displacement in cm plus a look vector."""
    import xarray as xr

    lon = np.linspace(-68.6, -66.4, 41)
    lat = np.linspace(10.0, 11.0, 23)
    field = np.outer(np.sin(np.linspace(0, 3, lat.size)),
                     np.cos(np.linspace(0, 3, lon.size))).astype("float32")

    def write(name, values):
        path = tmp_path / f"{name}.grd"
        xr.DataArray(values, coords={"lat": lat, "lon": lon},
                     dims=("lat", "lon"), name="z").to_netcdf(path)
        return path

    # A plausible descending right-looking unit vector: east positive, north
    # negative, up positive -- ALOS's geometry, not NISAR's.
    ones = np.ones((lat.size, lon.size), dtype="float32")
    return dict(
        los=write("los_cm", 10.0 * field),
        look_e=write("look_e", 0.62 * ones),
        look_n=write("look_n", -0.11 * ones),
        look_u=write("look_u", np.sqrt(1 - 0.62 ** 2 - 0.11 ** 2) * ones),
        field=field, lon=lon, lat=lat)


def test_read_grd_round_trips_write_grd(tmp_path, grd_scene):
    """`read_grd` is the inverse of `write_grd`, values and CRS alike."""
    from slipsolve._geo import read_grd, write_grd

    original = read_grd(grd_scene["los"])
    assert original.rio.crs is not None and original.rio.crs.is_geographic

    path = write_grd(original.rio.write_crs("EPSG:4326"), tmp_path / "again.grd")
    again = read_grd(path)

    # `write_grd` reprojects, and rioxarray normalises the result to north-up, so
    # a grid written from south-up input comes back with its rows reversed. That
    # is a property of the georeferencing, not a difference in the field --
    # compare on the coordinates rather than on row order.
    assert again.shape == original.shape
    a = again.sortby("y").sortby("x")
    b = original.sortby("y").sortby("x")
    np.testing.assert_allclose(a["y"].values, b["y"].values, atol=1e-6)
    np.testing.assert_allclose(np.asarray(a.values, float),
                               np.asarray(b.values, float),
                               rtol=1e-4, atol=1e-5)


def test_read_grd_names_its_axes_x_and_y(grd_scene):
    """GMT writes `lon`/`lat`; everything downstream indexes `x`/`y`."""
    from slipsolve._geo import read_grd

    da = read_grd(grd_scene["los"])
    assert da.dims == ("y", "x")
    np.testing.assert_allclose(da["x"].values, grd_scene["lon"])


def test_from_grd_applies_units_and_sign(grd_scene):
    """`units` and `sign` are applied on load, so what is stored is canonical.

    A ``.grd`` records neither, and both are silent when wrong -- ``units`` as a
    factor of 100 in the recovered slip, ``sign`` as a fault slipping backwards.
    """
    from slipsolve import LosGrid

    kw = dict(direction="descending", look_direction="right")
    positive = LosGrid.from_grd(grd_scene["los"], grd_scene["look_e"],
                                grd_scene["look_n"], grd_scene["look_u"],
                                units="cm", sign=+1, **kw)
    negative = LosGrid.from_grd(grd_scene["los"], grd_scene["look_e"],
                                grd_scene["look_n"], grd_scene["look_u"],
                                units="cm", sign=-1, **kw)

    a = np.asarray(positive.ds["los"].values, float)
    b = np.asarray(negative.ds["los"].values, float)
    good = np.isfinite(a) & np.isfinite(b)
    # cm -> m, and the sign applied once on load; nothing re-applies it.
    assert np.nanmax(np.abs(a[good])) == pytest.approx(0.1, rel=0.05)
    np.testing.assert_allclose(a[good], -b[good], rtol=1e-6)
    assert positive.ds.attrs["sign"] == +1
    assert negative.ds.attrs["sign"] == -1


def test_from_grd_reprojects_geographic_to_metres(grd_scene):
    """A lon/lat lattice would put `width_min=1000` at a million columns."""
    from slipsolve import LosGrid

    stack = LosGrid.from_grd(grd_scene["los"], grd_scene["look_e"],
                             grd_scene["look_n"], grd_scene["look_u"],
                             units="cm", direction="descending",
                             look_direction="right")
    assert not stack.crs.is_geographic
    assert abs(float(np.diff(stack.x)[0])) > 100.0        # metres, not degrees
    assert stack.ds.attrs["look_direction"] == "right"


def test_from_grd_rejects_an_inverted_look_vector(grd_scene, tmp_path):
    """`los_up` is cos(incidence) and must be positive; a flip is raised on."""
    import xarray as xr

    from slipsolve import LosGrid

    from slipsolve._geo import read_grd

    down = tmp_path / "look_d.grd"
    da = read_grd(grd_scene["look_u"])
    xr.DataArray(-np.asarray(da.values), coords={"lat": grd_scene["lat"],
                                                 "lon": grd_scene["lon"]},
                 dims=("lat", "lon"), name="z").to_netcdf(down)

    with pytest.raises(ValueError, match="sensor_to_target"):
        LosGrid.from_grd(grd_scene["los"], grd_scene["look_e"],
                         grd_scene["look_n"], down, units="cm")


def test_los_grid_rebuilds_from_a_dataset(grd_scene):
    """`type(stack)(ds)` is how `resample` and `calibrate` rebuild a stack."""
    from slipsolve import LosGrid

    stack = LosGrid.from_grd(grd_scene["los"], grd_scene["look_e"],
                             grd_scene["look_n"], grd_scene["look_u"],
                             units="cm", direction="descending",
                             look_direction="right")
    again = type(stack)(stack.ds)
    assert isinstance(again, LosGrid)
    np.testing.assert_array_equal(again.x, stack.x)
    assert again.crs == stack.crs
