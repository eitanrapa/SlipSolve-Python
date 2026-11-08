"""Preparing GNSS series: steps, trajectories, common mode, and noise that is not white."""

import numpy as np
import pandas as pd
import pytest

from slipsolve import (FaultTrace, TimeBasis, TrackSeries, common_mode, fit_trajectory, gnss_series, noise_levels,
                       prepare_stations, read_steps, screen_velocities, trajectory_design)
from slipsolve.gnss import _date_to_mjd

_STEPS = """\
00NA  17MAY04  1  Antenna_Type_Changed
00NA  17MAY04  1  Receiver_Make_and_Model_Changed
P595  19JUL04  2   257.040     9.157  6.4 ci38443183
P595  19JUL06  2   575.440    19.454  7.1 ci38457511
CCCC  19JUL06  2   575.440    23.615  7.1 ci38457511
"""


def test_read_steps(tmp_path):
    path = tmp_path / "steps.txt"
    path.write_text(_STEPS)
    table = read_steps(path)
    assert len(table) == 5
    assert _date_to_mjd("19JUL06") == 58670, "the M7.1's day"
    assert _date_to_mjd("00JAN01") == 51544 and _date_to_mjd("99DEC31") == 51543
    row = table[table["station"] == "P595"].iloc[1]
    assert row["kind"] == "earthquake" and row["magnitude"] == 7.1 and row["distance_km"] == 19.454 and row["event"] == "ci38457511"
    assert table.iloc[0]["detail"] == "Antenna_Type_Changed"
    assert read_steps(path, stations=["CCCC"]).shape[0] == 1


def test_trajectory_finds_rate_seasons_and_a_step():
    rng = np.random.default_rng(0)
    t = np.arange(-1500.0, 0.0)
    truth = 0.012 + 0.020 * t / 365.25 + 0.003 * np.sin(2 * np.pi * t / 365.25) + 0.015 * (t >= -700.0)
    values = truth + rng.normal(0.0, 1e-3, t.size)
    values[::97] += 0.05                                       # outliers the clipping must drop
    coefficients, scatter = fit_trajectory(t, values, [-700.0])
    assert abs(coefficients[1] - 0.020) < 3e-4 and abs(coefficients[-1] - 0.015) < 5e-4
    assert abs(scatter - 1e-3) < 2e-4
    residual = values - trajectory_design(t, [-700.0]) @ coefficients
    assert np.median(np.abs(residual)) < 1e-3


def test_a_step_outside_the_data_is_left_out():
    t = np.arange(-800.0, 0.0)
    values = 0.01 * t / 365.25 + np.random.default_rng(1).normal(0.0, 1e-3, t.size)
    coefficients, _ = fit_trajectory(t, values, [-2000.0, -300.0, 50.0])
    assert coefficients[-3] == 0.0 and coefficients[-1] == 0.0 and coefficients.size == 9
    assert (trajectory_design(t, [-2000.0, -300.0, 50.0]) @ coefficients).shape == t.shape


@pytest.mark.parametrize("sigma,walk", [(1e-3, 0.0), (1e-3, 2e-3), (5e-4, 4e-3)])
def test_noise_levels_recover_white_and_random_walk(sigma, walk):
    rng = np.random.default_rng(int(1e6 * sigma + 1e4 * walk))
    t = np.arange(0.0, 2000.0)
    series = rng.normal(0.0, sigma, t.size) + np.cumsum(rng.normal(0.0, walk * np.sqrt(1.0 / 365.25), t.size))
    got_sigma, got_walk = noise_levels(t, series)
    assert abs(got_sigma - sigma) < 0.25 * sigma
    assert abs(got_walk - walk) < 0.4 * walk + 5e-4, "a true zero reads ~0.4 mm/sqrt(yr): the difference of two sampled squares"


def test_noise_levels_take_gaps():
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 2000.0)
    series = rng.normal(0.0, 1e-3, t.size) + np.cumsum(rng.normal(0.0, 2e-3 * np.sqrt(1.0 / 365.25), t.size))
    keep = rng.uniform(size=t.size) > 0.2
    keep[300:500] = False
    got_sigma, got_walk = noise_levels(t[keep], series[keep])
    assert abs(got_sigma - 1e-3) < 3e-4 and abs(got_walk - 2e-3) < 1e-3
    with pytest.raises(ValueError, match="too few"):
        noise_levels(t[:10], series[:10])


def test_noise_levels_never_give_zero():
    """A TrackSeries refuses a sigma of zero, and a flat record would otherwise give one."""
    sigma, walk = noise_levels(np.arange(0.0, 500.0), np.zeros(500))
    assert sigma > 0.0 and walk == 0.0


def test_common_mode_is_each_days_median():
    mjd = np.arange(10)
    a = (mjd, np.ones(10), np.zeros(10), 2.0 * np.ones(10))
    b = (mjd, 3.0 * np.ones(10), np.zeros(10), 2.0 * np.ones(10))
    c = (mjd[:5], 2.0 * np.ones(5), np.ones(5), 2.0 * np.ones(5))
    days, east, north, up = common_mode({"a": a, "b": b, "c": c})
    assert np.array_equal(days, mjd)
    assert np.allclose(east[:5], 2.0) and np.isnan(east[5:]).all(), "fewer than three stations: no correction"
    assert np.allclose(up[:5], 2.0)


# -- the covariance a random walk adds ------------------------------------------------------------
def _series(t0, t1, sigma, walk, frame):
    return TrackSeries([0.0], [0.0], (1.0, 0.0, 0.0), np.zeros((len(t0), 1)), t0, t1, sigma, frame=frame, random_walk=walk)


@pytest.fixture(scope="module")
def frame():
    return FaultTrace(np.array([-118.0, -117.0]), np.array([36.0, 36.0])).local_frame()


def test_random_walk_covariance_from_one_date(frame):
    t = np.array([100.0, 200.0, 400.0])
    C = _series(np.zeros(3), t, 1e-3, 3e-3, frame).time_covariance()
    white = 1e-6 * (np.eye(3) + np.ones((3, 3)))
    walk = (3e-3) ** 2 * np.minimum.outer(t, t) / 365.25
    assert np.allclose(C, white + walk)


def test_random_walk_covariance_of_chained_and_reversed_pairs(frame):
    # consecutive pairs do not overlap: no walk between them, only the shared date's white noise
    chain = _series(np.array([0.0, 100.0]), np.array([100.0, 300.0]), 1e-3, 3e-3, frame).time_covariance()
    assert np.isclose(chain[0, 1], -1e-6)
    assert np.isclose(chain[0, 0], 2e-6 + 9e-6 * 100.0 / 365.25) and np.isclose(chain[1, 1], 2e-6 + 9e-6 * 200.0 / 365.25)
    # a reversed pair overlapping another shares its walk with the opposite sign
    both = _series(np.array([0.0, 200.0]), np.array([300.0, 100.0]), 1e-3, 3e-3, frame).time_covariance()
    assert np.isclose(both[0, 1], -9e-6 * 100.0 / 365.25)


def test_random_walk_covariance_matches_a_simulation(frame):
    rng = np.random.default_rng(4)
    t0, t1 = np.array([0.0, 50.0, 100.0]), np.array([150.0, 200.0, 180.0])
    C = _series(t0, t1, 2e-3, 5e-3, frame).time_covariance()
    days = np.arange(0.0, 201.0)
    draws = []
    for _ in range(40000):
        walk = np.r_[0.0, np.cumsum(rng.normal(0.0, 5e-3 * np.sqrt(1.0 / 365.25), days.size - 1))]
        position = walk + rng.normal(0.0, 2e-3, days.size)
        draws.append(position[t1.astype(int)] - position[t0.astype(int)])
    empirical = np.cov(np.array(draws).T)
    assert np.allclose(empirical, C, rtol=0.05, atol=5e-7)


def test_a_random_walk_changes_the_weights_as_it_should(frame):
    """A walk makes the late dates worth less in a stack: the virtual pair's size falls."""
    t = np.arange(10.0, 310.0, 10.0)
    basis = TimeBasis.linear()
    values = np.zeros((t.size, 1))
    white = TrackSeries([0.0], [0.0], (1.0, 0.0, 0.0), values, np.zeros(t.size), t, 1e-3, frame=frame)
    wander = TrackSeries([0.0], [0.0], (1.0, 0.0, 0.0), values, np.zeros(t.size), t, 1e-3, frame=frame, random_walk=3e-3)
    _, t_white, scale_white = white.reduce(basis)
    _, t_wander, scale_wander = wander.reduce(basis)
    assert t_wander[0, 0] / scale_wander < 0.5 * t_white[0, 0] / scale_white


# -- the whole preparation --------------------------------------------------------------------------
def _table(name, days, lon, lat, signal, rng, antenna=(), walk=0.0, sigma=1e-3, common=None):
    t = days + 0.5
    mjd = 58000 + days.astype(int)
    east = 0.020 * t / 365.25 + 0.002 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0.0, sigma, t.size)
    if walk:
        east = east + np.cumsum(rng.normal(0.0, walk * np.sqrt(1.0 / 365.25), t.size))
    for when, size in antenna:
        east = east + size * (days >= when)
    east = east + signal(t)
    if common is not None:
        east = east + common[days.astype(int) - int(days.min())]
    return pd.DataFrame({"mjd": mjd, "east": 100.0 + east, "north": 50.0 + rng.normal(0.0, sigma, t.size),
                         "up": 10.0 + rng.normal(0.0, 3 * sigma, t.size), "lon": lon, "lat": lat})


def test_prepare_stations_recovers_a_planted_decay(tmp_path):
    rng = np.random.default_rng(5)
    days = np.arange(-1200.0, 400.0)
    common = rng.normal(0.0, 2e-3, days.size)
    decay = lambda t: 0.03 * np.log1p(np.maximum(t, 0.0) / 30.0) + 0.1 * (t >= 0.0)    # a coseismic step and a decay
    none = lambda t: 0.0 * t
    tables = {
        "NEAR": _table("NEAR", days, -117.0, 36.0, decay, rng, common=common, antenna=[(-600.0, 0.02), (200.0, 0.03)]),
        "ALSO": _table("ALSO", days, -117.1, 36.1, decay, rng, common=common, walk=2e-3),
        **{f"FAR{k}": _table(f"FAR{k}", days, -115.0 + k, 37.0, none, rng, common=common) for k in range(4)},
        "NOPRE": _table("NOPRE", days[days > -100], -117.0, 36.2, decay, rng),
        "NOISY": _table("NOISY", days, -117.0, 36.3, decay, rng, sigma=9e-3),
    }
    tables["NOISY"]["north"] = 50.0 + rng.normal(0.0, 9e-3, days.size)
    steps = pd.DataFrame({"station": ["NEAR", "NEAR"], "mjd": [58000 - 600, 58000 + 200], "kind": ["equipment", "equipment"],
                          "detail": ["Antenna_Type_Changed", "Antenna_Type_Changed"], "distance_km": np.nan, "magnitude": np.nan, "event": ""})
    stations, report = prepare_stations(tables, 58000.0, window=(1.0, 380.0), steps=steps, reference=[f"FAR{k}" for k in range(4)])
    status = report.set_index("station")["status"]
    assert "pre-event" in status["NOPRE"] and "noise" in status["NOISY"] and status["NEAR"] == "ok"
    assert set(stations) == {"NEAR", "ALSO", "FAR0", "FAR1", "FAR2", "FAR3"}

    near = stations["NEAR"]
    expected = 0.03 * np.log1p(near.t / 30.0) + 0.1
    misfit = near.east - expected
    assert abs(np.median(misfit)) < 2e-3, "the trend, the seasons, the antenna steps and the common mode are gone"
    assert np.std(misfit) < 2.2e-3
    assert near.sigma[0] < 1.5e-3, "the common mode's 2 mm is not left in the noise"
    assert stations["ALSO"].random_walk[0] > 1e-3 > stations["NEAR"].random_walk[0] - 1e-3
    assert np.std(stations["FAR0"].east) < 2.0e-3

    frame = FaultTrace(np.array([-118.0, -116.0]), np.array([36.0, 36.0])).local_frame()
    series = near.track_series(frame)
    assert [s.name for s in series] == ["gnss_e", "gnss_n", "gnss_u"]
    assert series[0].random_walk == near.random_walk[0] and series[0].n_pairs == near.t.size - 1


def test_gnss_series_passes_the_walk(frame):
    lon, lat = frame.to_lonlat(0.0, 0.0)
    t = np.arange(1.0, 50.0)
    series = gnss_series(lon, lat, t, t * 0.0, t * 0.0, t * 0.0, 1e-3, 1e-3, 3e-3, frame, random_walk=(1e-3, 2e-3, 3e-3))
    assert [s.random_walk for s in series] == [1e-3, 2e-3, 3e-3]


def test_screen_velocities():
    rng = np.random.default_rng(6)
    lon, lat = rng.uniform(-121.0, -119.0, 80), rng.uniform(35.0, 37.0, 80)
    ve, vn, vu = 0.003 * (lon + 120.0), 0.002 * (lat - 36.0), rng.normal(0.0, 1e-3, 80)
    # a block's step, which is not an outlier when the blocks are named
    side = lon > -120.0
    vn = vn + 0.030 * side
    # a landslide: three stations together, agreeing with each other
    lon[:3], lat[:3] = -120.5 + np.array([0.0, 2e-3, 4e-3]), 36.5
    ve[:3] -= 0.05
    # a subsiding aquifer
    vu[3] = -0.15
    table = pd.DataFrame({"lon": lon, "lat": lat, "ve": ve, "vn": vn, "vu": vu})
    keep, reason = screen_velocities(table, groups=side)
    assert list(reason[:4]) == ["neighbours", "neighbours", "neighbours", "vertical"]
    assert keep[4:].all() and not keep[:4].any()
    # without the blocks, the step itself reads as an outlier near the boundary
    assert (~screen_velocities(table)[0][4:]).sum() > 0
    with pytest.raises(ValueError, match="labels"):
        screen_velocities(table, groups=side[:-1])
