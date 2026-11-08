"""The interseismic mode: rigid blocks plus back-slip, from velocities.

Everything rides on three signs, so they are tested against what they mean physically: a
rotation against its Euler pole, slip against the jump it makes across the trace, and
rigid blocks plus back-slip against the arctangent of Savage & Burford (1973).
"""

import warnings

import numpy as np
import pytest

from slipsolve import (EulerPole, FaultMesh, FaultTrace, HalfSpaceTDE, InterseismicInversion, InterseismicModel,
                       Observations, SlipInversion, block_labels, blocked_cv_curve, coupling_bounds,
                       gnss_observations, read_midas, read_tenv3, resolution_diagonal, rotation_columns,
                       side_labels, velocity_observations)
from slipsolve.interseismic import _axes_at, poles_from_coefficients
from slipsolve.regularize import slip_bounds

_MM = 1e-3
_RATE = 30.0 * _MM          # right-lateral, m/yr


def _east_west():
    """A long straight fault running due east, so the left block is the north one."""
    lon = np.array([-122.0, -117.0])
    lat = np.array([36.0, 36.0])
    trace = FaultTrace(lon, lat, name="ew")
    return trace, trace.local_frame()


def _translation_poles(frame, north_east=_RATE / 2, south_east=-_RATE / 2):
    """Poles moving the two blocks due east at the given rates at the frame's origin."""
    axis = _axes_at([frame.origin_lon, frame.origin_lat])[0]       # 1 m/yr east at that point
    return EulerPole(axis * north_east), EulerPole(axis * south_east)


# -- rotations ------------------------------------------------------------------------------
def test_pole_at_the_north_pole_moves_the_equator_east():
    pole = EulerPole.from_pole(0.0, 90.0, 1.0)
    east, north, up = pole.velocity(0.0, 0.0)
    assert np.isclose(east[0], np.radians(1.0) / 1e6 * 6378137.0)
    assert abs(north[0]) < 1e-12 and abs(up[0]) < 1e-12
    assert np.isclose(pole.lat, 90.0) and np.isclose(pole.rate, 1.0)


def test_pole_round_trip_and_difference():
    a = EulerPole.from_pole(-105.0, 48.0, 0.73)
    assert np.isclose(a.lon, -105.0) and np.isclose(a.lat, 48.0) and np.isclose(a.rate, 0.73)
    b = EulerPole.from_pole(-100.0, 40.0, 0.5)
    lon, lat = np.array([-120.0, -118.0]), np.array([35.0, 37.0])
    for got, x, y in zip((a - b).velocity(lon, lat), a.velocity(lon, lat), b.velocity(lon, lat)):
        assert np.allclose(got, x - y)


def test_rotation_columns_recover_planted_poles():
    """Two blocks, three-component rows and radar rows: a noise-free fit returns the poles exactly."""
    trace, frame = _east_west()
    rng = np.random.default_rng(1)
    x = rng.uniform(-200e3, 200e3, 200)
    y = rng.uniform(-120e3, 120e3, 200)
    look = rng.normal(size=(200, 3))
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    truth = {"north": EulerPole.from_pole(-110.0, 50.0, 0.8), "south": EulerPole.from_pole(-90.0, 45.0, 0.3)}
    blocks = np.where(y > 0, "north", "south")
    lon, lat = frame.to_lonlat(x, y)
    values = np.zeros(200)
    for name, pole in truth.items():
        rows = blocks == name
        e, n, u = pole.velocity(lon[rows], lat[rows])
        values[rows] = e * look[rows, 0] + n * look[rows, 1] + u * look[rows, 2]
    obs = Observations.from_points(x, y, values, look, 1e-3, name="any", frame=frame, epsg=None)
    matrix, labels, centres = rotation_columns(obs, blocks)
    assert matrix.shape == (200, 6) and labels[0] == "block:north:ve" and set(centres) == {"north", "south"}
    assert np.all(matrix[blocks == "north", 3:] == 0.0) and np.all(matrix[blocks == "south", :3] == 0.0)
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    poles = poles_from_coefficients(labels, coefficients, centres)
    for name, pole in truth.items():
        assert np.allclose(poles[name].omega, pole.omega, atol=1e-16, rtol=1e-8), name


def test_rotation_coefficients_mean_what_they_say():
    """At a block's centre, ve and vn are its velocity; spin turns it anticlockwise."""
    _, frame = _east_west()
    x = np.array([-10e3, 10e3, 0.0, 0.0])
    y = np.array([0.0, 0.0, -10e3, 10e3])
    obs = Observations.from_points(x, y, np.zeros(4), (1.0, 0.0, 0.0), 1e-3, name="e", frame=frame, epsg=None)
    matrix, labels, _ = rotation_columns(obs, np.array(["b"] * 4))
    assert np.allclose(matrix[:, labels.index("block:b:ve")], 1.0, atol=1e-2)
    assert np.allclose(matrix[:, labels.index("block:b:vn")], 0.0, atol=1e-2)
    # spin of 1 m/yr per 1000 km, anticlockwise: a site 10 km north moves west at 10 mm/yr
    assert np.isclose(matrix[3, labels.index("block:b:spin")], -0.01, atol=2e-4)
    assert np.isclose(matrix[2, labels.index("block:b:spin")], 0.01, atol=2e-4)


def test_rotation_columns_refuse_a_block_with_too_few_rows():
    _, frame = _east_west()
    obs = Observations.from_points(np.arange(4.0), np.arange(4.0), np.zeros(4), (1.0, 0.0, 0.0), 1e-3,
                                   name="e", frame=frame, epsg=None)
    with pytest.raises(ValueError, match="at least three"):
        rotation_columns(obs, np.array(["a", "a", "a", "b"]))


# -- labels ---------------------------------------------------------------------------------
def test_block_labels_and_sides():
    trace, frame = _east_west()
    x = np.array([-50e3, 50e3, 60e3])
    y = np.array([40e3, 40e3, -40e3])
    obs = Observations.from_points(x, y, np.zeros(3), (1.0, 0.0, 0.0), 1e-3, name="e", frame=frame, epsg=None)
    lon, lat = frame.to_lonlat(x, y)
    west = ([-125.0, lon[0] + 0.1, lon[0] + 0.1, -125.0], [30.0, 30.0, 40.0, 40.0])
    assert block_labels(obs, {"west": west}, default="rest").tolist() == ["west", "rest", "rest"]
    with pytest.raises(ValueError, match="no block"):
        block_labels(obs, {"west": west})
    assert side_labels(obs, trace, "north", "south").tolist() == ["north", "north", "south"]


# -- the signs ------------------------------------------------------------------------------
@pytest.mark.parametrize("dip", [90.0, 60.0])
def test_slip_vector_is_left_block_minus_right_block(dip):
    """The jump across the trace of uniform slip is `s Vstrike + d Vdip`, left minus right."""
    trace, frame = _east_west()
    if dip == 90.0:
        mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=20e3)
    else:
        mesh = FaultMesh.curved(trace, frame, uniform_dip=dip, max_depth=20e3, edge_length=20e3)
    n = mesh.n_elements
    normals = mesh.normals
    assert np.allclose(normals[:, 1] > 0, True), "the left of an east-running trace is north"
    strike = np.cross([0.0, 0.0, 1.0], normals)
    strike /= np.linalg.norm(strike, axis=1, keepdims=True)
    along = np.array([1.0, 0.0, 0.0])
    assert np.allclose(strike @ along, -1.0), "the strike vector runs against the trace"
    down_dip_up = np.cross(normals, strike)

    engine = HalfSpaceTDE()
    tx, ty = trace.to_local(frame)
    mid = np.array([0.5 * (tx[0] + tx[-1]) + 10e3, 0.5 * (ty[0] + ty[-1])])
    x, y = np.full(2, mid[0]), mid[1] + np.array([300.0, -300.0])
    for s, d in ((1.0, 0.0), (0.0, 1.0)):
        slip = np.r_[np.full(n, s), np.full(n, d)]
        u = engine.forward(mesh, slip, x, y)
        jump = u[0] - u[1]
        expected = s * strike[0] + d * down_dip_up[0]
        assert np.allclose(jump, expected, atol=0.03), (s, d, jump, expected)


def test_blocks_plus_back_slip_is_the_arctangent():
    """Savage & Burford (1973): rigid blocks + back-slip above D = (V / pi) atan(y / D)."""
    lon = np.array([-132.0, -108.0])            # ~2100 km, so mid-length is 2-D at these distances
    lat = np.array([36.0, 36.0])
    trace = FaultTrace(lon, lat, name="long")
    frame = trace.local_frame()
    depth = 15e3
    mesh = FaultMesh.vertical(trace, frame, max_depth=depth, edge_length=15e3)
    y = np.array([-200e3, -60e3, -20e3, -5e3, 5e3, 20e3, 60e3, 200e3])
    # The trace is a great circle's chord, so measure from it rather than from y = 0.
    tx, ty = trace.to_local(frame)
    y0 = np.interp(0.0, tx, ty)
    x = np.zeros_like(y) + 3e3
    obs = Observations.from_points(x, y0 + y, np.zeros(y.size), (1.0, 0.0, 0.0), 1e-3, name="e", frame=frame, epsg=None)
    blocks = side_labels(obs, trace, "north", "south")
    assert blocks.tolist() == ["south"] * 4 + ["north"] * 4
    inv = InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"), frame_looks=False)

    # Translate the blocks east and west at +-V/2 at this mid-fault site.
    north, south = _translation_poles(frame)
    v_s, v_d = inv.long_term_slip({"north": north, "south": south})
    middle = np.abs(mesh.params[:, 0] - mesh.params[:, 0].mean()) < 300e3
    assert np.allclose(v_s[middle], -_RATE, rtol=2e-3), "right-lateral: negative"
    assert np.all(v_d == 0.0)

    back_slip = -v_s
    surface = inv.g[:, :mesh.n_nodes] @ back_slip
    rigid = np.sign(y) * _RATE / 2
    expected = _RATE / np.pi * np.arctan(y / depth)
    assert np.abs(rigid + surface - expected).max() < 0.012 * _RATE
    # The two halves of the arctangent go the right way: fully locked at the trace.
    assert abs(rigid[3] + surface[3]) < 0.2 * _RATE


def test_left_and_right_swapped_warns():
    trace, frame = _east_west()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=30e3)
    x = np.linspace(-100e3, 100e3, 20)
    y = np.where(np.arange(20) % 2, 50e3, -50e3)
    obs = Observations.from_points(x, y, np.zeros(20), (1.0, 0.0, 0.0), 1e-3, name="e", frame=frame, epsg=None)
    blocks = side_labels(obs, trace, "north", "south")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"))
    with pytest.warns(RuntimeWarning, match="check \\(left, right\\)"):
        InterseismicInversion(mesh, obs, blocks=blocks, faults=("south", "north"))
    with pytest.raises(ValueError, match="no rows"):
        InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "pacific"))


# -- bounds ---------------------------------------------------------------------------------
def test_scalar_bounds_are_unchanged():
    lo, hi = slip_bounds(4, (-10.0, 10.0), (-2.0, 3.0), (-1, 0, 0))
    assert np.array_equal(lo, np.r_[np.full(4, -10.0), np.full(4, -2.0)])
    assert np.array_equal(hi, np.r_[np.zeros(4), np.full(4, 3.0)])


def test_bounds_per_parameter():
    lo, hi = slip_bounds(3, (np.array([-1.0, -2.0, -3.0]), 0.0), (0.0, np.array([1.0, 2.0, 3.0])))
    assert np.array_equal(lo, [-1.0, -2.0, -3.0, 0.0, 0.0, 0.0])
    assert np.array_equal(hi, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        slip_bounds(3, (np.zeros(2), 1.0))


def test_coupling_bounds_have_the_back_slip_sign():
    v_s = np.array([-0.03, -0.03, 0.02])
    v_d = np.array([0.0, 0.01, -0.01])
    limits = coupling_bounds((v_s, v_d))
    assert np.allclose(limits["strike"][0], [0.0, 0.0, -0.02])
    assert np.allclose(limits["strike"][1], [0.03, 0.03, 0.0], atol=1e-8)
    assert np.allclose(limits["dip"][0], [0.0, -0.01, 0.0])
    assert limits["dip"][0][0] == limits["dip"][1][0] == 0.0, "no rate: fixed at zero"
    half = coupling_bounds((v_s, v_d), coupling=(0.25, 0.5))
    assert np.allclose(half["strike"][0][:2], 0.25 * 0.03) and np.allclose(half["strike"][1][:2], 0.5 * 0.03)


# -- a whole inversion ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def problem():
    """Two blocks, GNSS and two radar tracks with planes, a fault locked in two patches."""
    lon = np.array([-122.0, -117.0])
    lat = np.array([36.0, 36.2])
    trace = FaultTrace(lon, lat, name="f")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=24e3, edge_length=6e3)
    s, z = mesh.params[:, 0], mesh.params[:, 1]
    mid = 0.5 * (s.min() + s.max())
    along = (np.exp(-((s - mid + 90e3) / 45e3) ** 4) + np.exp(-((s - mid - 90e3) / 45e3) ** 4))   # a creeping centre
    down = np.clip((16e3 + z) / 6e3, 0.0, 1.0)
    coupling = along * down

    # 30 mm/yr right-lateral: the north block moves east relative to the south, on top of a shared plate motion.
    south = EulerPole.from_pole(-115.0, 62.0, 0.62)
    north = EulerPole(south.omega + _axes_at([frame.origin_lon, frame.origin_lat])[0] * _RATE)
    rng = np.random.default_rng(5)

    def sites(n, half_x, half_y, near):
        x = rng.uniform(-half_x, half_x, 4 * n)
        y = rng.uniform(-half_y, half_y, 4 * n)
        keep = trace.distance(x, y, frame) > near
        return x[keep][:n], y[keep][:n]

    gx, gy = sites(70, 200e3, 130e3, 2e3)
    glon, glat = frame.to_lonlat(gx, gy)
    gnss = gnss_observations(glon, glat, np.zeros(70), np.zeros(70), np.zeros(70), 0.5 * _MM, 0.5 * _MM, 1.5 * _MM,
                             frame, name="gnss")
    tracks = {}
    for name, look in (("asc", (-0.56, -0.10, 0.82)), ("dsc", (0.60, -0.10, 0.79))):
        tx, ty = sites(400, 170e3, 110e3, 2e3)
        look = np.asarray(look) / np.linalg.norm(look)
        tracks[name] = Observations.from_points(tx, ty, np.zeros(tx.size), look, 1.0 * _MM, name=name, frame=frame,
                                                epsg=None)
    obs = Observations.concat([gnss, *tracks.values()], normalize="none")
    labels = side_labels(obs, trace, "north", "south")

    probe = InterseismicInversion(mesh, obs, blocks=labels, faults=("north", "south"))
    truth_poles = {"north": north, "south": south}
    v_s, v_d = probe.long_term_slip(truth_poles)
    back_slip = np.r_[-coupling * v_s, -coupling * v_d]

    rigid = np.zeros(obs.n)
    glook_e, glook_n, glook_u = (obs.ds[f"look_{c}"].values for c in "enu")
    olon, olat = frame.to_lonlat(obs.ds["x"].values, obs.ds["y"].values)
    for name, pole in truth_poles.items():
        rows = labels == name
        e, n, u = pole.velocity(olon[rows], olat[rows])
        rigid[rows] = e * glook_e[rows] + n * glook_n[rows] + u * glook_u[rows]
    track = obs.ds["track"].values.astype(str)
    plane = np.zeros(obs.n)
    for name, gradient in (("asc", (2.0, -1.5, 1.0)), ("dsc", (-1.0, 2.5, -2.0))):
        rows = track == name
        plane[rows] = _MM * (gradient[0] + gradient[1] * obs.ds["x"].values[rows] / 170e3
                             + gradient[2] * obs.ds["y"].values[rows] / 110e3)
    noise = rng.normal(0.0, 1.0, obs.n) / obs.ds["weight"].values
    values = rigid + probe.g @ back_slip + plane + noise
    obs = Observations(obs.ds.assign(los=("obs", values)))
    inv = InterseismicInversion(mesh, obs, blocks=labels, faults=("north", "south"),
                                ramp={"gnss": "none", "*": "linear"})
    return dict(mesh=mesh, frame=frame, obs=obs, inv=inv, coupling=coupling, poles=truth_poles, labels=labels,
                long_term=(v_s, v_d), back_slip=back_slip, trace=trace)


_LAMBDA = 1e5           # rank ~55, where this problem's recovery peaks


def test_inversion_recovers_rates_coupling_and_poles(problem):
    model = problem["inv"].solve(smoothing=_LAMBDA)
    assert isinstance(model, InterseismicModel)
    v_true, _ = problem["long_term"]
    v_got, d_got = model.long_term_slip
    assert np.abs(v_got - v_true).max() < 1.5 * _MM, "the blocks' relative rate"
    coupling = model.coupling
    assert np.nanmin(coupling) >= -1e-6 and np.nanmax(coupling) <= 1.0 + 1e-6, "the bound holds"
    corr = np.corrcoef(coupling, problem["coupling"])[0, 1]
    assert corr > 0.8, corr
    relative = model.poles["north"] - model.poles["south"]
    truth = problem["poles"]["north"] - problem["poles"]["south"]
    lon, lat = problem["frame"].to_lonlat(0.0, 0.0)
    for got, want in zip(relative.velocity(lon, lat)[:2], truth.velocity(lon, lat)[:2]):
        assert abs(got[0] - want[0]) < 1.0 * _MM
    # The rows' misfit reaches the noise.
    assert model.rms_misfit < 1.6 * _MM
    assert model.options["rounds"] >= 1 and model.converged
    assert np.allclose(model.prediction, model.fault_prediction + model.ramp_prediction)
    assert np.allclose(model.elastic_prediction, model.fault_prediction)
    assert abs(np.median(np.abs(model.rotation_prediction))) > 3 * _MM, "the blocks carry the far field"


def test_edges_are_pinned_in_m_per_year(problem):
    """The zero-slip rows must mean the same in m/yr as in m: a tiny fraction of the fault's rate."""
    model = problem["inv"].solve(smoothing=_LAMBDA)
    mesh = problem["mesh"]
    edges = mesh.boundary_nodes("bottom") | mesh.boundary_nodes("left") | mesh.boundary_nodes("right")
    assert np.abs(model.slip_magnitude[edges]).max() < 0.02 * model.max_slip
    assert model.max_slip > 10 * _MM


def test_fixed_rate_is_the_same_as_the_settled_one(problem):
    inv = problem["inv"]
    settled = inv.solve(smoothing=_LAMBDA)
    fixed = inv.solve(smoothing=_LAMBDA, rate=settled.long_term_slip)
    assert fixed.options["rounds"] == 1
    assert np.allclose(fixed.x, settled.x, atol=1e-7)


def test_free_back_slip_and_scalar_rate(problem):
    inv = problem["inv"]
    free = inv.solve(smoothing=_LAMBDA, coupling=None)
    assert free.options["coupling"] is None
    assert np.isfinite(free.coupling).all()
    scalar = inv.solve(smoothing=_LAMBDA, rate=(-_RATE, 0.0))
    assert scalar.strike_slip.max() <= _RATE + 1e-9 and scalar.strike_slip.min() >= -1e-9
    assert np.abs(scalar.dip_slip).max() <= 1e-8


def test_nuisance_prior(problem):
    inv, obs = problem["inv"], problem["obs"]
    kwargs = dict(blocks=problem["labels"], faults=("north", "south"), ramp={"gnss": "none", "*": "linear"})
    tight = InterseismicInversion(problem["mesh"], obs, nuisance_prior={"asc:d*": 1e-6}, **kwargs)
    loose = InterseismicInversion(problem["mesh"], obs, nuisance_prior={"asc:d*": 1e3}, **kwargs)
    free = inv.solve(smoothing=_LAMBDA, rate=problem["long_term"])
    gradient = [i for i, label in enumerate(inv.ramp_labels) if label.startswith("asc:d")]
    assert len(gradient) == 2
    a = tight.solve(smoothing=_LAMBDA, rate=problem["long_term"])
    b = loose.solve(smoothing=_LAMBDA, rate=problem["long_term"])
    assert np.abs(a.ramp[gradient]).max() < 1e-5
    assert np.abs(free.ramp[gradient]).max() > 1e-3
    assert np.allclose(b.x, free.x, atol=1e-6), "a wide prior changes nothing"
    assert tight.prior_rows().shape == (2, tight.n_param)
    with pytest.raises(ValueError, match="matches no nuisance"):
        InterseismicInversion(problem["mesh"], obs, nuisance_prior={"nosuch*": 1.0}, **kwargs)
    with pytest.raises(ValueError, match="positive"):
        InterseismicInversion(problem["mesh"], obs, nuisance_prior={"asc:d*": 0.0}, **kwargs)


def test_a_pattern_held_at_one_is_its_field_removed(problem):
    """A far-field signal held at amplitude 1 by a prior with a mean: the same solve as on data without it."""
    from slipsolve import Pattern

    obs = problem["obs"]

    def far_field(x, y):         # 2 mm/yr of east shear across the area, and some uplift: no block or fault makes it
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return 2.0 * _MM * y / 100e3, np.zeros_like(x), 0.5 * _MM * (x / 150e3) ** 2

    pattern = Pattern.from_enu("far", far_field)
    kwargs = dict(blocks=problem["labels"], faults=("north", "south"), ramp={"gnss": "none", "*": "linear"})
    probe = InterseismicInversion(problem["mesh"], obs, patterns=[pattern], **kwargs)
    column = probe.ramp[:, probe.ramp_labels.index("pattern:far")]
    added = Observations(obs.ds.assign(los=("obs", obs.ds["los"].values + column)))
    held = InterseismicInversion(problem["mesh"], added, patterns=[pattern], nuisance_prior={"pattern:far": (1.0, 1e-6)},
                                 **kwargs)
    model = held.solve(smoothing=_LAMBDA, rate=problem["long_term"])
    assert model.pattern_amplitudes["far"] == pytest.approx(1.0, abs=1e-6)
    plain = problem["inv"].solve(smoothing=_LAMBDA, rate=problem["long_term"])
    keep = [held.ramp_labels.index(label) for label in problem["inv"].ramp_labels]
    assert np.allclose(model.x[:held.n_slip], plain.x[:held.n_slip], atol=1e-6)
    assert np.allclose(model.ramp[keep], plain.ramp, atol=1e-6)
    free = InterseismicInversion(problem["mesh"], added, patterns=[pattern], **kwargs).solve(smoothing=_LAMBDA,
                                                                                          rate=problem["long_term"])
    assert abs(free.pattern_amplitudes["far"] - 1.0) > 1e-3, "free, the amplitude is the data's"


def test_a_rotation_prior_with_a_mean(problem):
    """A block's three coefficients tied to a pole from elsewhere: the solve and the rigid pre-fit both keep it."""
    kwargs = dict(blocks=problem["labels"], faults=("north", "south"), ramp={"gnss": "none", "*": "linear"})
    centres = problem["inv"].block_centres
    target = EulerPole(problem["poles"]["north"].omega + _axes_at(centres["north"])[2] * 3.0 * _MM)    # 3 mm/yr of spin off
    axes = _axes_at(centres["north"])
    coefficients = [float(target.omega @ axis / (axis @ axis)) for axis in axes]
    prior = {f"block:north:{c}": (value, 1e-7) for c, value in zip(("ve", "vn", "spin"), coefficients)}
    tied = InterseismicInversion(problem["mesh"], problem["obs"], nuisance_prior=prior, **kwargs)
    for name, pole in (("solve", tied.solve(smoothing=_LAMBDA).poles["north"]), ("rigid", tied.rigid_poles()["north"])):
        assert np.abs(pole.omega - target.omega).max() < 1e-6 * np.abs(target.omega).max(), name
    untied = problem["inv"].rigid_poles()["north"]
    assert np.abs(untied.omega - target.omega).max() > 1e-3 * np.abs(target.omega).max()


def test_every_solver_respects_a_bound_per_node(problem):
    """The three solvers on a plain inversion (no rotations): per-node limits are limits."""
    obs = problem["obs"]
    mesh = FaultMesh.vertical(problem["trace"], problem["frame"], max_depth=24e3, edge_length=24e3)
    sub = Observations(obs.ds.isel(obs=np.flatnonzero(obs.ds["track"].values.astype(str) == "gnss")))
    inv = SlipInversion(mesh, sub, basis="node")
    n = mesh.n_nodes
    hi = np.full(n, 0.02)
    hi[::3] = 0.001
    for method in ("lsq_linear", "active_set", "lsq_linear_dense"):
        model = inv.solve(smoothing=1e4, strike=(0.0, hi), dip=(-1e-3, 1e-3), method=method)
        assert np.all(model.strike_slip <= hi + 1e-9), method
        assert np.all(model.strike_slip >= -1e-9), method
    # the options stay JSON-able
    assert isinstance(model.options["strike"][1], list) and len(model.options["strike"][1]) == n


def test_given_g_is_used_and_checked(problem):
    mesh, obs = problem["mesh"], problem["obs"]
    g = np.zeros((obs.n, 2 * mesh.n_nodes))
    inv = SlipInversion(mesh, obs, basis="node", g=g)
    assert inv.g is not None and np.all(inv.g == 0.0)
    with pytest.raises(ValueError, match="g has shape"):
        SlipInversion(mesh, obs, basis="node", g=np.zeros((3, 3)))


def test_extra_columns_are_checked(problem):
    mesh, obs = problem["mesh"], problem["obs"]
    g = np.zeros((obs.n, 2 * mesh.n_nodes))
    with pytest.raises(ValueError, match="extra_columns"):
        SlipInversion(mesh, obs, basis="node", g=g, extra_columns=(np.zeros((3, 1)), ["x"]))
    inv = SlipInversion(mesh, obs, basis="node", g=g, extra_columns=(np.ones((obs.n, 1)), ["bias"]), ramp="offset")
    assert inv.ramp_labels[-1] == "bias" and inv.n_ramp == 4
    other = inv.with_ramp("linear")
    assert other.ramp_labels[-1] == "bias" and other.n_ramp == 10
    assert np.array_equal(other.ramp[:, -1], np.ones(obs.n))


def test_resolution_and_cross_validation_read_the_columns(problem):
    inv = problem["inv"]
    d = resolution_diagonal(inv, _LAMBDA)
    assert 1.0 < d.attrs["effective_rank"] < inv.n_slip
    curve = blocked_cv_curve(inv, [1e4, 1e6], n_folds=3, direction=(1.0, 0.0), rate=problem["long_term"])
    assert np.isfinite(curve["cv_rms"].values).all()
    # a fold re-fits its own blocks: the labels are sliced with the rows
    settled = blocked_cv_curve(inv, [1e5, 1e6], n_folds=3, direction=(1.0, 0.0), iterations=2)
    assert np.isfinite(settled["cv_rms"].values).all()


def test_save_load_keeps_columns_rates_and_poles(problem, tmp_path):
    model = problem["inv"].solve(smoothing=_LAMBDA)
    back = InterseismicModel.load(model.save(tmp_path / "inter.slip.zip"))
    assert np.array_equal(back.x, model.x)
    assert np.allclose(back.ramp_prediction, model.ramp_prediction)
    assert np.allclose(back.rotation_prediction, model.rotation_prediction)
    assert np.allclose(back.coupling, model.coupling, equal_nan=True)
    assert back.options["coupling"] == [0.0, 1.0]
    for name, pole in model.poles.items():
        assert np.allclose(back.poles[name].omega, pole.omega)


def test_frame_looks_matter_off_the_meridian():
    """Far from the central meridian the frame's axes are turned, and G must know."""
    trace, frame = _east_west()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=30e3)
    x = np.linspace(-300e3, 300e3, 30)
    y = np.where(np.arange(30) % 2, 40e3, -40e3).astype(float)
    obs = Observations.from_points(x, y, np.zeros(30), (0.0, 1.0, 0.0), 1e-3, name="n", frame=frame, epsg=None)
    blocks = side_labels(obs, trace, "north", "south")
    plain = InterseismicInversion(mesh, obs, blocks=blocks, frame_looks=False)
    turned = InterseismicInversion(mesh, obs, blocks=blocks, frame_looks=True)
    assert not np.allclose(plain.g, turned.g)
    assert np.array_equal(plain.ramp, turned.ramp), "rotations are geographic either way"


# -- NGL's files ----------------------------------------------------------------------------
_MIDAS = """\
00NA MIDAS5 2008.2355 2018.7324 10.4969  3190  2968   5171   0.036213   0.058799  -0.001111  0.000233 0.000262 0.000919   4781.8345  -1378706.2822  104.8496 0.072 0.116 0.072 0.002145 0.002358 0.008469   1 -12.4666386637 -229.1560118921   104.84337
BEPK MIDAS5 2000.6817 2024.7420 24.0603  8536  8142  12863  -0.021344  -0.000329   0.000448  0.000179 0.000133 0.000458   2339.6463   3972048.8581 2471.1027 0.062 0.107 0.070 0.002612 0.001893 0.006657   9  35.8783879304 -118.0740936530  2471.10719
CCCC MIDAS5 2000.6845 2024.7420 24.0575  8490  7643  12025  -0.020668   0.000138  -0.000289  0.000165 0.000141 0.000440   2613.5518   3937311.8945  813.9582 0.068 0.072 0.055 0.002326 0.001987 0.006234  10  35.5653122077 -117.6711733703   813.95500
P595 MIDAS5 2022.8070 2024.7420  1.9350  6908  6515  10844  -0.018814  -0.002481  -0.000707  0.000163 0.000142 0.000401   -256.6010   3951985.0373  644.4279 0.079 0.053 0.056 0.002166 0.001911 0.005390   7  35.6975594464 -117.4028361565   644.42365
"""

_TENV3 = """\
site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)
CCS1 25JAN22 2025.0595 60697 2350 3  -66.9   1338  0.109390   1159955  0.293051   853  0.674488  0.0000 0.001202 0.001022 0.004731 -0.045275 -0.180582  0.227389  10.48911 -66.88833 853.66723
CCS1 25JAN21 2025.0568 60696 2350 2  -66.9   1338  0.109000   1159955  0.293000   853  0.674000  0.0000 0.001202 0.001022 0.004731 -0.045275 -0.180582  0.227389  10.48911 -66.88833 853.66723
"""


def test_read_midas(tmp_path):
    path = tmp_path / "midas.txt"
    path.write_text(_MIDAS)
    table = read_midas(path)
    assert len(table) == 4
    assert np.isclose(table["lon"].values[0], 130.8439881079), "wrapped"
    assert np.isclose(table["ve"].values[1], -0.021344) and np.isclose(table["su"].values[2], 0.000440)
    assert np.isclose(table["lat"].values[1], 35.8783879304) and table["station"].values[1] == "BEPK"
    box = read_midas(path, bbox=(-119.0, -117.0, 35.0, 36.5), min_years=2.5, exclude=("CCCC",))
    assert table["station"].values.tolist()[1:] == ["BEPK", "CCCC", "P595"]
    assert box["station"].values.tolist() == ["BEPK"]


def test_velocity_observations_and_floor(tmp_path):
    path = tmp_path / "midas.txt"
    path.write_text(_MIDAS)
    table = read_midas(path, bbox=(-119.0, -117.0, 35.0, 36.5))
    frame = FaultTrace(np.array([-118.5, -117.5]), np.array([35.5, 35.8])).local_frame()
    obs = velocity_observations(table, frame, components="en", scale=2.0, floor=5e-4)
    assert obs.n == 6 and obs.tracks == ["gnss"]
    assert np.allclose(obs.ds["los"].values[:3], table["ve"].values)
    assert np.allclose(obs.ds["look_e"].values, np.r_[np.ones(3), np.zeros(3)])
    assert np.allclose(obs.ds["std"].values, 5e-4), "2 x 0.2 mm/yr, under the floor"


def test_read_tenv3(tmp_path):
    path = tmp_path / "CCS1.tenv3"
    path.write_text(_TENV3)
    table = read_tenv3(path)
    assert table["mjd"].values.tolist() == [60696, 60697], "sorted"
    assert np.isclose(table["east"].values[1], 1338.109390)
    assert np.isclose(table["north"].values[1], 1159955.293051)
    assert np.isclose(table["up"].values[1], 853.674488)
    assert np.isclose(table["lat"].values[0], 10.48911) and np.isclose(table["lon"].values[0], -66.88833)
    assert np.isclose(table["sigma_u"].values[0], 0.004731)
