"""Pins ``regions=``: one offset per region an unwrapper saw on its own.

An unwrapped interferogram cut along a rupture (so the unwrapper cannot cross it),
or with islands, has an arbitrary constant in every connected region. The sampler
must never average two regions into one sample, and the inversion must give each
region its own offset rather than take a tie between them on trust.
"""

import numpy as np
import pytest
from slip_synthetic import LosLike, analytic_los_stack, forward_los_stack, tapered_slip

from slipsolve import FaultMesh, FaultTrace, Observations, SlipInversion, SlipModel
from slipsolve.regularize import ramp_columns

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture(scope="module")
def frame(trace):
    return trace.local_frame()


def _with_los(stack, los):
    ds = stack.ds.copy(deep=True)
    ds["los"] = (("pair", "y", "x"), np.asarray(los, np.float32)[None])
    return LosLike(ds)


def _two_halves(trace, frame, cut_col=137):
    """A constant 0 left of column ``cut_col`` and 1 right of it, labelled 1 / 2."""
    stack = analytic_los_stack(trace, frame, spacing=1000.0)
    ny, nx = stack.ds["los"].shape[1:]
    col = np.broadcast_to(np.arange(nx), (ny, nx))
    labels = np.where(col < cut_col, 1, 2)
    return _with_los(stack, np.where(col < cut_col, 0.0, 1.0)), labels


def _flat_sample(stack, frame, **kwargs):
    # rms_min far above the field: the only reason left to split is a region mix
    kwargs.setdefault("rms_min", 10.0)
    return Observations.from_los(stack, frame=frame, width_min=3000.0,
                                 width_max=20000.0, **kwargs)


# -- the sampler ---------------------------------------------------------------

def test_no_sample_mixes_two_regions(trace, frame):
    stack, labels = _two_halves(trace, frame)
    obs = _flat_sample(stack, frame, regions=labels)
    los, region = obs.ds["los"].values, obs.ds["region"].values
    # each region is a constant, so any average across the cut would land between
    assert set(np.unique(los)) <= {0.0, 1.0}
    np.testing.assert_array_equal(region, np.where(los == 0.0, 1, 2))
    # without regions the cells straddling the cut average the two constants
    plain = _flat_sample(stack, frame).ds["los"].values
    assert np.any((plain > 0.0) & (plain < 1.0))


def test_a_mixed_cell_at_the_minimum_width_keeps_its_majority(trace, frame):
    """A cut off the cells' lattice ends inside minimum-width cells; one on it does not."""
    stack, labels = _two_halves(trace, frame, cut_col=137)
    total = labels.size                          # every pixel is valid
    used = _flat_sample(stack, frame, regions=labels).ds["count"].values.sum()
    assert used < total                          # the minority columns were dropped
    nx = labels.shape[1]
    aligned, aligned_labels = _two_halves(trace, frame, cut_col=nx // 2)
    used = _flat_sample(aligned, frame, regions=aligned_labels).ds["count"].values.sum()
    assert used == total                         # the first split is on the cut


def test_connected_labels_islands_and_leaves_one_region_alone(trace, frame):
    base = analytic_los_stack(trace, frame, spacing=1000.0)
    one = Observations.from_los(base, frame=frame, rms_min=0.01, width_min=3000.0,
                                width_max=20000.0)
    same = Observations.from_los(base, frame=frame, rms_min=0.01, width_min=3000.0,
                                 width_max=20000.0, regions="connected")
    # one connected region: exactly the old sampler, plus a label
    for name in ("x", "y", "los", "look_e", "count"):
        np.testing.assert_array_equal(one.ds[name].values, same.ds[name].values)
    assert set(same.ds["region"].values) == {1}

    los = base.ds["los"].values[0].copy()
    los[:, 120:123] = np.nan                     # water: two islands
    two = Observations.from_los(_with_los(base, los), frame=frame, rms_min=0.01,
                                width_min=3000.0, width_max=20000.0, regions="connected")
    assert set(two.ds["region"].values) == {1, 2}


def test_region_labels_are_checked(trace, frame):
    stack, labels = _two_halves(trace, frame)
    with pytest.raises(ValueError, match="shape"):
        _flat_sample(stack, frame, regions=labels[:-1])
    with pytest.raises(ValueError, match="integer"):
        _flat_sample(stack, frame, regions=labels.astype(float))
    with pytest.raises(ValueError, match="connected"):
        _flat_sample(stack, frame, regions="islands")


# -- the offset columns ------------------------------------------------------------

def test_one_offset_per_region_and_one_ramp_per_track(trace, frame):
    stack, labels = _two_halves(trace, frame)
    obs = _flat_sample(stack, frame, regions=labels, name="t")
    columns, names = ramp_columns(obs, "linear")
    region = obs.ds["region"].values
    counts = {r: int((region == r).sum()) for r in (1, 2)}
    big, small = sorted(counts, key=lambda r: -counts[r])
    assert names == ["t:offset", f"t:offset@{small}", "t:dx", "t:dy"]
    np.testing.assert_array_equal(columns[:, 0], (region == big).astype(float))
    np.testing.assert_array_equal(columns[:, 0] + columns[:, 1], 1.0)
    assert np.linalg.matrix_rank(columns) == 4


def test_a_plain_track_beside_a_region_track_keeps_one_offset(trace, frame):
    stack, labels = _two_halves(trace, frame)
    with_regions = _flat_sample(stack, frame, regions=labels, name="r")
    plain = _flat_sample(stack, frame, name="p")
    obs = Observations.concat([with_regions, plain])
    assert np.all(obs.ds["region"].values[obs.track_mask("p")] == 0)
    _, names = ramp_columns(obs, "offset")
    assert sorted(names) == sorted(["p:offset", "r:offset", f"r:offset@{names_small(with_regions)}"])


def names_small(obs):
    region = obs.ds["region"].values
    values, counts = np.unique(region, return_counts=True)
    return int(values[np.argmin(counts)])


# -- the inversion -----------------------------------------------------------------

@pytest.fixture(scope="module")
def shifted(trace, frame):
    """Two tracks of a planted patch; on one, the south block is off by 0.37 m."""
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0)
    stacks, label_maps = {}, {}
    for g in ("asc", "desc"):
        stack = forward_los_stack(mesh, truth, trace, frame, geometry=g, spacing=5000.0,
                                  noise=0.004, seed=1)
        xx, yy = np.meshgrid(stack.ds["x"].values, stack.ds["y"].values)
        fx, fy = frame.from_epsg(xx.ravel(), yy.ravel(), int(stack.ds.attrs["epsg"]))
        south = (trace.side(fx, fy, frame) < 0).reshape(xx.shape)
        los = stack.ds["los"].values[0].astype(float)
        if g == "asc":
            los = np.where(south, los + 0.37, los)
        stacks[g] = _with_los(stack, los)
        label_maps[g] = np.where(south, 2, 1)
    return mesh, truth, stacks, label_maps


def _invert(trace, frame, mesh, stacks, label_maps=None):
    obs = Observations.concat([
        Observations.from_los(stacks[g], name=g, frame=frame, trace=trace,
                              rms_min=0.006, width_min=6000.0, width_max=30000.0,
                              exclude_within=5000.0,
                              regions=None if label_maps is None else label_maps[g])
        for g in ("asc", "desc")])
    inversion = SlipInversion(mesh, obs, ramp="offset")
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-1.0, 1.0))
    return obs, inversion, model


def _region_offsets(model, obs, track):
    region = obs.ds["region"].values[obs.track_mask(track)]
    values, counts = np.unique(region, return_counts=True)
    out = {}
    for label, value in zip(model.ramp_labels, model.ramp):
        if label == f"{track}:offset":
            out[int(values[np.argmax(counts)])] = value
        elif label.startswith(f"{track}:offset@"):
            out[int(label.split("@")[1])] = value
    return out


def test_a_region_offset_takes_the_block_constant_not_the_slip(trace, frame, shifted, tmp_path):
    mesh, truth, stacks, label_maps = shifted
    truth_ss = truth[:mesh.n_elements]
    obs, inversion, model = _invert(trace, frame, mesh, stacks, label_maps)
    _, _, plain = _invert(trace, frame, mesh, stacks)

    offsets = _region_offsets(model, obs, "asc")
    assert abs(offsets[2] - offsets[1] - 0.37) < 0.01
    desc = _region_offsets(model, obs, "desc")
    assert abs(desc[2] - desc[1]) < 0.01

    corr = np.corrcoef(model.strike_slip, truth_ss)[0, 1]
    corr_plain = np.corrcoef(plain.strike_slip, truth_ss)[0, 1]
    assert corr > 0.9 and corr > corr_plain
    m0_truth = 30e9 * np.sum(mesh.areas * np.abs(truth_ss))
    assert abs(model.moment() / m0_truth - 1) < abs(plain.moment() / m0_truth - 1)
    assert model.variance_reduction > plain.variance_reduction

    # a saved model rebuilds the same per-region design
    path = model.save(tmp_path / "regions.slip.zip")
    loaded = SlipModel.load(path)
    np.testing.assert_array_equal(loaded.obs.ds["region"].values, obs.ds["region"].values)
    assert loaded.ramp_labels == model.ramp_labels
    np.testing.assert_array_equal(loaded.ramp_design, inversion.ramp)
