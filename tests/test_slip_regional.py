"""`slipsolve.regional`: a regional block model read, evaluated and handed to a local interseismic model.

The oracle is physical: two regional blocks moving past each other on a long locked fault give, as rotation less
back-slip, the arctangent of Savage & Burford (1973), `v = V/2 + (V/pi) atan(x / D)` from the far block's frame.
celeri's files are written here from the synthetic model and read back, so no celeri is needed.
"""

import numpy as np
import pandas as pd
import pytest

from slipsolve import FaultTrace, InterseismicInversion, Observations, gnss_observations
from slipsolve import _okada
from slipsolve.interseismic import EARTH_RADIUS, EulerPole, _axes_at, rotation_columns
from slipsolve.regional import CELERI_RADIUS, RegionalModel, fit_frame_rotation

LON0 = 240.0                   # the fault's meridian, celeri's 0-360 longitudes
RATE = 20.0                    # mm/yr, left-lateral: the east block moves north
DEPTH = 15.0                   # km
_DEG_PER_MYR = np.radians(1.0) * 1e-6


def two_blocks():
    """A fault along the meridian from 10 S to 10 N between a west block (at rest) and an east block moving north."""
    corners = {"sw": (LON0 - 2, -10.0), "s": (LON0, -10.0), "se": (LON0 + 2, -10.0),
               "nw": (LON0 - 2, 10.0), "n": (LON0, 10.0), "ne": (LON0 + 2, 10.0)}
    edges = [("s", "n", RATE), ("sw", "s", 0.0), ("s", "se", 0.0), ("nw", "n", 0.0), ("n", "ne", 0.0),
             ("sw", "nw", 0.0), ("se", "ne", 0.0)]
    rows = []
    for a, b, rate in edges:
        (lon1, lat1), (lon2, lat2) = corners[a], corners[b]
        azimuth = 0.0 if lon1 == lon2 else 90.0
        rows.append(dict(name=f"{a}-{b}", lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2, dip=90.0,
                         locking_depth=DEPTH if rate else 0.1, azimuth=azimuth,
                         length_km=111.2 * (abs(lat2 - lat1) + abs(lon2 - lon1)),
                         strike_slip=rate, dip_slip=0.0, tensile=0.0))
    # The east block's pole: 90 degrees west of the fault on the equator, so it moves north there at RATE.
    rate_deg = RATE * 1e-3 / CELERI_RADIUS / _DEG_PER_MYR
    blocks = pd.DataFrame([dict(name="west", interior_lon=LON0 - 1, interior_lat=0.0, euler_lon=0.0, euler_lat=90.0,
                                euler_rate=0.0),
                           dict(name="east", interior_lon=LON0 + 1, interior_lat=0.0, euler_lon=LON0 - 90.0,
                                euler_lat=0.0, euler_rate=rate_deg)])
    return RegionalModel(pd.DataFrame(rows), blocks)


def test_blocks_are_the_network_faces():
    model = two_blocks()
    labels = model.block_of(np.array([LON0 - 1.5, LON0 - 0.1, LON0 + 0.1, LON0 + 1.9, LON0 + 3.0, LON0]),
                            np.array([5.0, -9.0, 0.0, 9.5, 0.0, 12.0]))
    assert labels.tolist() == [0, 0, 1, 1, -1, -1]
    assert model.ties(np.array([LON0 - 1, LON0 - 1, LON0 + 1]), np.zeros(3), ["a", "a", "a"]) == {"a": 0}
    rest = pd.DataFrame([dict(name="rest", interior_lon=LON0 + 5, interior_lat=0.0, euler_lon=0.0, euler_lat=90.0,
                              euler_rate=0.0)])
    with_rest = RegionalModel(model.segments, pd.concat([model.blocks, rest], ignore_index=True))
    assert with_rest.block_of(np.array([LON0 + 3.0, LON0 - 0.1]), np.array([0.0, 0.0])).tolist() == [2, 0]


def test_a_segment_is_okada_in_its_own_frame():
    model = two_blocks()
    lat = np.array([0.0, 0.0, 0.3, -2.0])
    lon = LON0 + np.array([0.05, -0.2, 0.5, 0.01])
    field = model.segment_velocity(lon, lat, which=[0])
    # The fault runs north along the meridian: its frame's x is north and y west, so the station's y is minus its east.
    east = np.radians(lon - LON0) * EARTH_RADIUS * np.cos(np.radians(lat))
    north = np.radians(lat - (-10.0)) * EARTH_RADIUS
    length = np.radians(20.0) * EARTH_RADIUS
    ux, uy, uz = _okada.okada_disp_surface(north, -east, length, DEPTH * 1e3, DEPTH * 1e3, 90.0, RATE, 0.0, 0.0, 0.25)
    assert np.allclose(field[:, 1], 1e-3 * ux, rtol=1e-2, atol=1e-7)     # the projection is on the ellipsoid
    assert np.allclose(field[:, 0], -1e-3 * uy, rtol=1e-2, atol=1e-7)


def test_rotation_less_back_slip_is_the_arctangent():
    """The far field a one-block local model would need: Savage & Burford's interseismic profile across the fault."""
    model = two_blocks()
    x_km = np.array([-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0])
    lon = LON0 + np.degrees(x_km * 1e3 / CELERI_RADIUS)
    lat = np.zeros(x_km.size)
    trace = FaultTrace(np.array([LON0 - 360.0, LON0 - 360.0]), np.array([-1.0, 1.0]), name="probe")
    frame = trace.local_frame()
    zeros = np.zeros(x_km.size)
    obs = gnss_observations(lon - 360.0, lat, zeros, zeros, zeros, 1e-3, 1e-3, 1e-3, frame, name="gnss",
                            components="en")
    blocks = np.full(obs.n, "local")
    patterns = model.far_field(obs, blocks, {"local": 0}, traces=[], parts=("rotation", "outside"), closure_km=5000.0)
    assert sorted(p.name for p in patterns) == ["regional:outside", "regional:rotation"]
    looks = obs.frame_looks()
    xs, ys = obs.ds["x"].values, obs.ds["y"].values
    total = sum(p.values(xs, ys, *looks)[:, 0] for p in patterns)
    north_rows = obs.ds["look_n"].values > 0.5
    expected = RATE * 1e-3 * (0.5 + np.arctan(x_km / DEPTH) / np.pi)
    assert np.allclose(total[north_rows], expected, atol=0.01 * RATE * 1e-3)


def test_rotation_priors_hold_the_regional_pole():
    model = two_blocks()
    rng = np.random.default_rng(3)
    lon = LON0 - 360.0 + rng.uniform(0.2, 1.8, 30)
    lat = rng.uniform(-5.0, 5.0, 30)
    trace = FaultTrace(np.array([LON0 - 360.0, LON0 - 360.0]), np.array([-5.0, 5.0]), name="probe")
    frame = trace.local_frame()
    ve, vn, _ = model.rotation(1).velocity(lon, lat)
    obs = Observations.concat([gnss_observations(lon, lat, ve, vn, np.zeros(30), 1e-3, 1e-3, 1e-3, frame, name="gnss",
                                                 components="en")], normalize="none")
    blocks = np.full(obs.n, "east")
    shift = EulerPole.from_pole(100.0, 30.0, 0.05)
    priors = model.rotation_priors(obs, blocks, {"east": 1}, frame_rotation=shift, sigma=1e-9, spin_sigma=1e-9)
    assert set(priors) == {"block:east:ve", "block:east:vn", "block:east:spin"}
    _, _, centres = rotation_columns(obs, blocks)
    axes = _axes_at(centres["east"])
    omega = sum(priors[f"block:east:{c}"][0] * axis for c, axis in zip(("ve", "vn", "spin"), axes))
    assert np.allclose(omega, model.rotation(1).omega * CELERI_RADIUS / EARTH_RADIUS - shift.omega, rtol=1e-12, atol=1e-25)
    tied = InterseismicInversion(_probe_mesh(frame), obs, blocks=blocks, nuisance_prior=priors)
    assert np.abs(tied.rigid_poles()["east"].omega - omega).max() < 1e-6 * np.abs(omega).max()


def _probe_mesh(frame):
    from slipsolve import FaultMesh

    trace = FaultTrace(np.array([LON0 - 360.0 + 3.0, LON0 - 360.0 + 3.0]), np.array([-1.0, 1.0]), name="far")
    return FaultMesh.vertical(trace, frame, max_depth=10e3, edge_length=20e3)


def test_frame_rotation_is_recovered():
    rng = np.random.default_rng(5)
    lon, lat = rng.uniform(-123, -115, 80), rng.uniform(33, 40, 80)
    truth = EulerPole.from_pole(-80.0, 5.0, 0.19)
    e0, n0 = rng.normal(0, 0.02, 80), rng.normal(0, 0.02, 80)
    de, dn, _ = truth.velocity(lon, lat)
    e1, n1 = e0 + de + rng.normal(0, 1e-4, 80), n0 + dn + rng.normal(0, 1e-4, 80)
    e1[:3] += 0.01                                                          # three bad stations
    pole, kept = fit_frame_rotation(lon, lat, e0, n0, e1, n1, floor=1e-5)
    assert not kept[:3].any() and kept[3:].all()
    assert np.allclose(pole.omega, truth.omega, rtol=2e-2)


def test_celeri_files_round_trip(tmp_path):
    model = two_blocks()
    s = model.segments
    pd.DataFrame({"name": s["name"], "lon1": s["lon1"], "lat1": s["lat1"], "lon2": s["lon2"], "lat2": s["lat2"],
                  "dip": s["dip"], "locking_depth": s["locking_depth"], "azimuth": s["azimuth"],
                  "length": 1e3 * s["length_km"], "model_strike_slip_rate": s["strike_slip"],
                  "model_dip_slip_rate": s["dip_slip"], "model_tensile_slip_rate": s["tensile"]}).to_csv(
        tmp_path / "model_segment.csv", index=False)
    model.blocks.to_csv(tmp_path / "model_block.csv", index=False)
    (tmp_path / "config.json").write_text('{"material_lambda": 3e10, "material_mu": 3e10}')
    back = RegionalModel.from_celeri(tmp_path)
    lon, lat = np.array([LON0 + 0.1, LON0 - 0.3]), np.array([1.0, -2.0])
    assert np.allclose(back.segment_velocity(lon, lat), model.segment_velocity(lon, lat))
    assert back.poisson == pytest.approx(0.25)
    assert back.block_of(lon, lat).tolist() == [1, 0]
