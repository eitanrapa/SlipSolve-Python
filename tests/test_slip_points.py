"""Point observations (GNSS) and per-track fault visibility, for an earthquake sequence.

Two events on two faults: a radar pair spans both, so its rows see the sum, while GNSS
resolves each event, so each event's stations must see only the fault that event broke.
``Observations.from_points`` carries the stations; ``SlipInversion(sees=)`` zeroes the
Green's functions of the faults a track does not see.
"""

import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

from slipsolve import FaultMesh, FaultTrace, Observations, SlipInversion, SlipModel
from slipsolve.greens import HalfSpaceTDE

_LON_A = np.array([-68.40, -68.20, -68.00, -67.80, -67.60])
_LAT_A = np.array([10.48, 10.52, 10.55, 10.57, 10.60])
_SHIFT = 0.35                          # fault B ~39 km north of fault A
_SIGMA = np.array([0.002, 0.002, 0.005])
_LOOKS = np.eye(3)


def _stations(frame, traces, n=45, seed=3):
    """GNSS-like sites in a box round both faults, none within 4 km of either trace."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-90e3, 90e3, 4 * n)
    y = rng.uniform(-45e3, 85e3, 4 * n)
    far = np.all([t.distance(x, y, frame) > 4e3 for t in traces], axis=0)
    x, y = x[far][:n], y[far][:n]
    return x, y


def _gnss(frame, x, y, enu, name, seed):
    """Three rows per station from an ``(n, 3)`` ENU field, with noise."""
    rng = np.random.default_rng(seed)
    lon, lat = frame.to_lonlat(x, y)
    values = (enu + rng.normal(0.0, _SIGMA, enu.shape)).ravel()
    return Observations.from_points(np.repeat(lon, 3), np.repeat(lat, 3), values,
                                    np.tile(_LOOKS, (len(x), 1)), np.tile(_SIGMA, len(x)),
                                    name=name, frame=frame)


@pytest.fixture(scope="module")
def setup():
    trace_a = FaultTrace(_LON_A, _LAT_A, name="a")
    trace_b = FaultTrace(_LON_A, _LAT_A + _SHIFT, name="b")
    frame = trace_a.local_frame()
    mesh_a = FaultMesh.vertical(trace_a, frame, max_depth=15e3, edge_length=5e3)
    mesh_b = FaultMesh.vertical(trace_b, frame, max_depth=15e3, edge_length=5e3)
    joined = FaultMesh.join([mesh_a, mesh_b])
    truth_a = tapered_slip(mesh_a, peak=-2.0, along_width=25e3, locking_depth=10e3)
    truth_b = tapered_slip(mesh_b, peak=-1.5, along_centre=0.4, along_width=20e3,
                           locking_depth=10e3)
    na, nb = mesh_a.n_elements, mesh_b.n_elements
    truth = np.concatenate([truth_a[:na], truth_b[:nb], truth_a[na:], truth_b[nb:]])

    engine = HalfSpaceTDE(0.25)
    sx, sy = _stations(frame, (trace_a, trace_b))
    gnss1 = _gnss(frame, sx, sy, engine.forward(mesh_a, truth_a, sx, sy), "gnss1", seed=4)
    gnss2 = _gnss(frame, sx, sy, engine.forward(mesh_b, truth_b, sx, sy), "gnss2", seed=5)

    stack = forward_los_stack(joined, truth, trace_a, frame, geometry="asc", spacing=5000.0,
                              half_x=110e3, half_y=90e3, noise=0.004, seed=6)
    insar = Observations.from_los(stack, name="asc", frame=frame, trace=trace_a,
                                  rms_min=0.006, width_min=5000.0, width_max=25000.0,
                                  exclude_within=3e3)
    keep = trace_b.distance(insar.ds["x"].values, insar.ds["y"].values, frame) > 3e3
    insar = Observations(insar.ds.isel(obs=np.flatnonzero(keep)))
    obs = Observations.concat([insar, gnss1, gnss2], normalize="none",
                              weights={"asc": 1.0 / 0.004})
    return dict(frame=frame, mesh_a=mesh_a, mesh_b=mesh_b, joined=joined, truth=truth,
                truth_a=truth_a, truth_b=truth_b, obs=obs, sx=sx, sy=sy)


def test_from_points_rows_weights_and_frame(setup):
    frame, obs = setup["frame"], setup["obs"]
    g1 = obs.ds.isel(obs=np.flatnonzero(obs.track_mask("gnss1")))
    assert g1.sizes["obs"] == 3 * len(setup["sx"])
    np.testing.assert_allclose(g1["x"].values[::3], setup["sx"], atol=1e-3)
    np.testing.assert_allclose(g1["y"].values[::3], setup["sy"], atol=1e-3)
    np.testing.assert_allclose(g1["weight"].values, 1.0 / np.tile(_SIGMA, len(setup["sx"])))
    np.testing.assert_array_equal(g1["look_n"].values[1::3], 1.0)
    assert np.all(g1["cell_size"].values == 0) and np.all(g1["count"].values == 1)
    # local metres in, no transform
    p = Observations.from_points([1e3], [2e3], [0.1], (1, 0, 0), 0.002, frame=frame, epsg=None)
    assert float(p.ds["x"][0]) == 1e3 and float(p.ds["y"][0]) == 2e3


def test_from_points_refuses_bad_input(setup):
    frame = setup["frame"]
    with pytest.raises(ValueError, match="frame"):
        Observations.from_points([-68.0], [10.5], [0.1], (1, 0, 0), 0.002)
    with pytest.raises(ValueError, match="unit vectors"):
        Observations.from_points([-68.0], [10.5], [0.1], (1, 1, 0), 0.002, frame=frame)
    with pytest.raises(ValueError, match="positive"):
        Observations.from_points([-68.0], [10.5], [0.1], (1, 0, 0), 0.0, frame=frame)
    with pytest.raises(ValueError, match="one entry per row"):
        Observations.from_points([-68.0, -67.9], [10.5], [0.1, 0.2], (1, 0, 0), 0.002,
                                 frame=frame)


def test_sees_zeroes_exactly_the_hidden_faults(setup):
    inv = SlipInversion(setup["joined"], setup["obs"], basis="node",
                        ramp={"asc": "offset", "gnss1": "none", "gnss2": "none"},
                        sees={"gnss1": [0], "gnss2": [1]})
    labels = inv.fault_labels()
    n = inv.basis.n_basis
    assert labels.size == n and set(labels) == {0, 1}
    cols = {f: np.concatenate([np.flatnonzero(labels == f), np.flatnonzero(labels == f) + n])
            for f in (0, 1)}
    obs = setup["obs"]
    rows = {t: np.flatnonzero(obs.track_mask(t)) for t in ("asc", "gnss1", "gnss2")}
    g = inv.g
    assert not g[np.ix_(rows["gnss1"], cols[1])].any()
    assert not g[np.ix_(rows["gnss2"], cols[0])].any()
    assert np.abs(g[np.ix_(rows["gnss1"], cols[0])]).max() > 0
    assert np.abs(g[np.ix_(rows["gnss2"], cols[1])]).max() > 0
    for f in (0, 1):                     # the radar sees both
        assert np.abs(g[np.ix_(rows["asc"], cols[f])]).max() > 0
    # the unrestricted matrix differs only in the hidden blocks
    full = SlipInversion(setup["joined"], setup["obs"], basis="node",
                         ramp={"asc": "offset", "gnss1": "none", "gnss2": "none"}).g
    np.testing.assert_array_equal(g[rows["asc"]], full[rows["asc"]])
    np.testing.assert_array_equal(g[np.ix_(rows["gnss1"], cols[0])],
                                  full[np.ix_(rows["gnss1"], cols[0])])


def test_sees_is_checked(setup):
    kw = dict(basis="node", ramp={"asc": "offset", "gnss1": "none", "gnss2": "none"})
    with pytest.raises(ValueError, match="not in the observations"):
        SlipInversion(setup["joined"], setup["obs"], sees={"gps": [0]}, **kw)
    with pytest.raises(ValueError, match="faults are 0..1"):
        SlipInversion(setup["joined"], setup["obs"], sees={"gnss1": [2]}, **kw)
    # a mesh that was never joined is one fault
    single = SlipInversion(setup["mesh_a"], setup["obs"], **kw)
    assert set(single.fault_labels()) == {0}


def _potency(model, inv, fault):
    labels = inv.fault_labels()
    areas = inv.basis.lumped_areas()
    ss = model.strike_slip
    return float(np.sum(np.abs(ss[labels == fault]) * areas[labels == fault]))


def test_each_event_recovered_and_recorded(setup, tmp_path):
    """With sees, both events' GNSS fit to their noise and each fault keeps its own potency;
    without it, each event's stations are charged with the other event's slip."""
    kw = dict(basis="node", ramp={"asc": "offset", "gnss1": "none", "gnss2": "none"})
    solve = dict(smoothing=0.02, polarity=(-1, 0, 0), method="active_set")
    inv = SlipInversion(setup["joined"], setup["obs"], sees={"gnss1": [0], "gnss2": [1]}, **kw)
    model = inv.solve(**solve)
    blind = SlipInversion(setup["joined"], setup["obs"], **kw).solve(**solve)

    obs = setup["obs"]
    for t in ("gnss1", "gnss2"):
        m = obs.track_mask(t)
        sigma = np.tile(_SIGMA, int(m.sum()) // 3)
        chi = np.sqrt(np.mean((model.residual[m] / sigma) ** 2))
        chi_blind = np.sqrt(np.mean((blind.residual[m] / sigma) ** 2))
        assert chi < 2.5, (t, chi)
        assert chi_blind > 3 * chi, (t, chi, chi_blind)

    # potency per fault against the planted one (element slip x element area)
    for fault, mesh, truth in ((0, setup["mesh_a"], setup["truth_a"]),
                               (1, setup["mesh_b"], setup["truth_b"])):
        planted = float(np.sum(np.abs(truth[:mesh.n_elements]) * mesh.areas))
        assert _potency(model, inv, fault) == pytest.approx(planted, rel=0.2)

    path = model.save(tmp_path / "sees.slip.zip")
    assert SlipModel.load(path).options["sees"] == {"gnss1": [0], "gnss2": [1]}
