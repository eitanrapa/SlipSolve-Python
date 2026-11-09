"""`slipsolve.regional`: a regional block model solved, evaluated and handed to a local interseismic model.

The oracle is physical: two regional blocks moving past each other on a long locked fault give, as rotation less
back-slip, the arctangent of Savage & Burford (1973), `v = V/2 + (V/pi) atan(x / D)` from the far block's frame.
celeri's input files are written here from the synthetic model and read back (`celeri_inputs`), so no celeri is needed.
"""

import numpy as np
import pandas as pd
import pytest

from slipsolve import FaultTrace, InterseismicInversion, Observations, gnss_observations
from slipsolve import _okada
from slipsolve.interseismic import EARTH_RADIUS, EulerPole, _axes_at, rotation_columns
from slipsolve.regional import RegionalModel, fit_frame_rotation

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
    rate_deg = RATE * 1e-3 / EARTH_RADIUS / _DEG_PER_MYR
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
    lon = LON0 + np.degrees(x_km * 1e3 / EARTH_RADIUS)
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
    assert np.allclose(omega, model.rotation(1).omega * model.radius / EARTH_RADIUS - shift.omega, rtol=1e-12, atol=1e-25)
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


# -- solving a block model ---------------------------------------------------------------------------------------

def _planted():
    """`two_blocks` with a third block outside held at rest, and the slip rates its rotations ask for."""
    from slipsolve import regional

    model = two_blocks()
    blocks = pd.concat([model.blocks, pd.DataFrame([dict(name="rest", interior_lon=LON0 + 5, interior_lat=0.0,
                                                         euler_lon=0.0, euler_lat=90.0, euler_rate=0.0)])],
                       ignore_index=True)
    blocks["apriori"] = blocks["name"] == "rest"
    segments = regional.west_first(model.segments)
    segments["azimuth"], segments["length_km"], _, _ = regional._geodesics(segments)
    planted = RegionalModel(segments, blocks)
    omega = np.concatenate([planted.rotation(b).omega for b in range(len(blocks))])
    rates = regional._kinematics(segments, planted.segment_blocks(), len(blocks), planted.radius) @ omega
    for c, name in enumerate(("strike_slip", "dip_slip", "tensile")):
        planted.segments[name] = 1e3 * rates[:, c]
    return planted, omega


def _stations(model, lon, lat, sigma=0.1):
    v = 1e3 * model.velocity(lon, lat)
    return pd.DataFrame(dict(name=[f"s{i}" for i in range(lon.size)], lon=lon, lat=lat, east=v[:, 0], north=v[:, 1],
                             east_sigma=sigma, north_sigma=sigma))


def test_a_block_model_recovers_its_rotations():
    """Stations from a planted model: the profile across its middle is Savage & Burford's, and the solve gets the
    rotations and the fault's slip rate back."""
    planted, omega = _planted()
    assert planted.segments.loc[0, "strike_slip"] == pytest.approx(RATE, rel=1e-6)    # left-lateral, as planted
    x_km = np.array([-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0])
    profile = _stations(planted, LON0 + np.degrees(x_km * 1e3 / EARTH_RADIUS), np.zeros(x_km.size))
    assert np.allclose(profile["north"], RATE * (0.5 + np.arctan(x_km / DEPTH) / np.pi), atol=0.01 * RATE)
    rng = np.random.default_rng(7)
    lon = LON0 + np.r_[-rng.uniform(0.1, 1.9, 30), rng.uniform(0.1, 1.9, 30)]
    stations = pd.concat([profile, _stations(planted, lon, rng.uniform(-8.0, 8.0, 60))], ignore_index=True)
    segments = planted.segments[["name", "lon1", "lat1", "lon2", "lat2", "dip", "locking_depth"]]
    solved = RegionalModel.solve(segments, planted.blocks, stations)
    assert np.allclose(solved.parameters, omega, rtol=0.0, atol=1e-6 * np.abs(omega).max())
    assert solved.segments.loc[0, "strike_slip"] == pytest.approx(RATE, rel=1e-5)
    assert solved.rms < 1e-6
    assert solved.segments.loc[0, "strike_slip_sigma"] > 0


def test_a_slip_rate_constraint_holds_its_rate():
    planted, _ = _planted()
    rng = np.random.default_rng(8)
    lon = LON0 + np.r_[-rng.uniform(0.1, 1.9, 20), rng.uniform(0.1, 1.9, 20)]
    stations = _stations(planted, lon, rng.uniform(-8.0, 8.0, 40), sigma=1.0)
    segments = planted.segments[["name", "lon1", "lat1", "lon2", "lat2", "dip", "locking_depth"]].copy()
    segments["ss_rate"], segments["ss_rate_sigma"] = np.nan, np.nan
    segments.loc[0, ["ss_rate", "ss_rate_sigma"]] = (0.5 * RATE, 1e-4)
    solved = RegionalModel.solve(segments, planted.blocks, stations)
    assert solved.segments.loc[0, "strike_slip"] == pytest.approx(0.5 * RATE, abs=1e-3)


def _grid_mesh(lon0=LON0 + 0.6, nx=6, nz=4, length_deg=0.6, depth_km=20.0):
    """A vertical rectangle along the meridian `lon0`, triangulated on a regular grid: `(points, triangles)`."""
    lats = np.linspace(-0.5 * length_deg, 0.5 * length_deg, nx + 1)
    depths = np.linspace(0.0, -depth_km, nz + 1)
    points = np.array([[lon0, la, de] for de in depths for la in lats])
    triangles = []
    for k in range(nz):
        for i in range(nx):
            a, b = k * (nx + 1) + i, k * (nx + 1) + i + 1
            c, d = a + nx + 1, b + nx + 1
            triangles += [[a, c, b], [b, c, d]]
    return points, np.array(triangles)


def test_mesh_edges_and_the_umbrella_operator():
    from slipsolve import regional

    points, triangles = _grid_mesh()
    edges = regional.mesh_edges(points, triangles)
    assert edges["top"].sum() == 6 and edges["bottom"].sum() == 6 and edges["side"].sum() == 6
    tri, _, centroid = regional._mesh_geometry(points, triangles, EARTH_RADIUS)
    L = regional._umbrella(tri, centroid).toarray()
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-18)
    i = int(np.flatnonzero((L != 0).sum(axis=1) == 4)[0])                   # an element with three neighbours
    neighbours = np.flatnonzero(L[i] > 0)
    d = np.linalg.norm(centroid[neighbours] - centroid[i], axis=1)
    assert L[i, i] == pytest.approx(-2.0 / d.sum() * np.sum(1.0 / d), rel=1e-12)    # Desbrun et al. (1999)
    assert np.allclose(L[i, neighbours], 2.0 / d.sum() / d, rtol=1e-12)


def test_celeri_input_files_are_read(tmp_path):
    """A block model written as celeri's input files, a mesh among them: the same solve as from the tables."""
    import json

    from slipsolve.regional import _read_msh, celeri_inputs

    planted, _ = _planted()
    rng = np.random.default_rng(9)
    lon = LON0 + np.r_[-rng.uniform(0.1, 1.9, 20), rng.uniform(0.1, 1.9, 20)]
    stations = _stations(planted, lon, rng.uniform(-8.0, 8.0, 40))
    s = planted.segments
    pd.DataFrame(dict(name=s["name"], lon1=s["lon1"], lat1=s["lat1"], lon2=s["lon2"], lat2=s["lat2"], dip=s["dip"],
                      locking_depth=s["locking_depth"], locking_depth_flag=0, ss_rate=0.0, ss_rate_sig=0.0, ss_rate_flag=0,
                      ds_rate=0.0, ds_rate_sig=0.0, ds_rate_flag=0, ts_rate=0.0, ts_rate_sig=0.0, ts_rate_flag=0,
                      mesh_flag=0)).to_csv(tmp_path / "segment.csv", index=False)
    b = planted.blocks
    pd.DataFrame(dict(name=b["name"], interior_lon=b["interior_lon"], interior_lat=b["interior_lat"],
                      euler_lon=b["euler_lon"], euler_lat=b["euler_lat"], rotation_rate=b["euler_rate"],
                      rotation_flag=b["apriori"].astype(int))).to_csv(tmp_path / "block.csv", index=False)
    pd.DataFrame(dict(lon=stations["lon"], lat=stations["lat"], name=stations["name"], east_vel=stations["east"],
                      north_vel=stations["north"], east_sig=0.1, north_sig=0.1, flag=1, up_vel=0.0, up_sig=1.0)).to_csv(
        tmp_path / "station.csv", index=False)
    points, triangles = _grid_mesh()
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points))]
    lines += [f"{i + 1} {p[0]} {p[1]} {p[2]}" for i, p in enumerate(points)]
    lines += ["$EndNodes", "$Elements", str(len(triangles))]
    lines += [f"{i + 1} 2 2 0 1 {t[0] + 1} {t[1] + 1} {t[2] + 1}" for i, t in enumerate(triangles)]
    lines += ["$EndElements", ""]
    (tmp_path / "fault.msh").write_text("\n".join(lines))
    back_points, back_triangles = _read_msh(tmp_path / "fault.msh")
    assert np.allclose(back_points, points) and (back_triangles == triangles).all()
    (tmp_path / "mesh.json").write_text(json.dumps([dict(mesh_filename="fault.msh", smoothing_weight=1e16,
                                                         top_slip_rate_constraint=0, bot_slip_rate_constraint=1,
                                                         side_slip_rate_constraint=0)]))
    (tmp_path / "config.json").write_text(json.dumps(dict(
        segment_file_name="segment.csv", block_file_name="block.csv", station_file_name="station.csv",
        mesh_parameters_file_name="mesh.json", slip_constraint_weight=1e5, tri_con_weight=1e6,
        material_lambda=3e10, material_mu=3e10)))
    inputs = celeri_inputs(tmp_path / "config.json")
    assert inputs["poisson"] == pytest.approx(0.25) and len(inputs["meshes"]) == 1
    from_files = RegionalModel.solve(**inputs)
    direct = RegionalModel.solve(planted.segments[["name", "lon1", "lat1", "lon2", "lat2", "dip", "locking_depth"]],
                                 planted.blocks, stations, meshes=[dict(points=points, triangles=triangles,
                                                                        smoothing=1e16, edges={"bottom": 1e-3})])
    assert np.allclose(from_files.parameters, direct.parameters, rtol=1e-9, atol=1e-15)


def test_a_mesh_recovers_uniform_slip():
    """Uniform slip is free under the umbrella operator, so with strong smoothing a planted uniform slip-rate deficit
    on a mesh comes back exactly, with the rotations."""
    from slipsolve import regional

    planted, omega = _planted()
    points, triangles = _grid_mesh()
    _, corners, _ = regional._mesh_geometry(points, triangles, EARTH_RADIUS)
    table = pd.DataFrame({f"{key}{k + 1}": corners[:, k, c] for k in range(3) for c, key in enumerate(("lon", "lat", "dep"))})
    table["strike_slip"], table["dip_slip"] = 5.0, -2.0
    planted.meshes = table
    rng = np.random.default_rng(10)
    lon = LON0 + np.r_[-rng.uniform(0.1, 1.9, 30), rng.uniform(0.1, 1.9, 30), 0.6 + rng.uniform(-0.3, 0.3, 20)]
    stations = _stations(planted, lon, np.r_[rng.uniform(-8.0, 8.0, 60), rng.uniform(-0.4, 0.4, 20)])
    segments = planted.segments[["name", "lon1", "lat1", "lon2", "lat2", "dip", "locking_depth"]]
    solved = RegionalModel.solve(segments, planted.blocks, stations,
                                 meshes=[dict(points=points, triangles=triangles, smoothing=1e24)])
    assert np.allclose(solved.meshes["strike_slip"], 5.0, atol=1e-4)
    assert np.allclose(solved.meshes["dip_slip"], -2.0, atol=1e-4)
    assert np.allclose(solved.parameters[:omega.size], omega, rtol=0.0, atol=1e-6 * np.abs(omega).max())
