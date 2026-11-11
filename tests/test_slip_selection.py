"""Pins the smoothing-weight selection.

`NormalSystem` must be the system `SlipInversion.solve` minimises, so a weight chosen on
one means the same on the other; a weight and an effective rank must invert each other;
the slabs must partition the rows; and the cross-validation must find an interior
minimum on a planted fault, alone and averaged over shifted partitions.
"""

import json

import numpy as np
import pytest

from slipsolve import FaultMesh, FaultTrace, Observations, SlipInversion
from slipsolve.selection import NormalSystem, RankChoice, choose_rank, heldout_residuals, slab_folds

import slip_synthetic as ss

_LON = np.array([-68.9, -68.65, -68.4, -68.15])
_LAT = np.array([10.50, 10.52, 10.53, 10.55])
SOLVE = dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=1.0, sides=("bottom", "left", "right"),
             polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0))


@pytest.fixture(scope="module")
def problem():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="selection")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=6e3)
    slip = ss.tapered_slip(mesh, peak=-2.0, along_width=25e3, locking_depth=10e3)
    obs = Observations.concat([
        Observations.from_los(
            ss.forward_los_stack(mesh, slip, trace, frame, geometry=geometry, spacing=4000.0,
                                 half_x=70e3, half_y=40e3, noise=0.002, seed=seed),
            name=geometry, frame=frame, trace=trace, rms_min=0.003, width_min=4000.0,
            width_max=30000.0, exclude_within=2000.0)
        for seed, geometry in enumerate(("asc", "desc"))])
    inv = SlipInversion(mesh, obs, basis="node", ramp="offset")
    x, y = frame.to_local(_LON, _LAT)
    direction = (float(x[-1] - x[0]), float(y[-1] - y[0]))
    return inv, obs, direction


def test_normal_system_is_the_solvers_system(problem):
    inv, _, _ = problem
    system = NormalSystem(inv, **SOLVE)
    for lam in (0.3, 3.0):
        x = system.solve(lam)
        reference = inv.solve(smoothing=lam, method="active_set", **SOLVE).x
        np.testing.assert_allclose(x, reference, atol=1e-8 * np.abs(reference).max())


def test_weight_and_rank_invert_each_other(problem):
    inv, _, _ = problem
    system = NormalSystem(inv, **SOLVE)
    ranks = [system.rank(lam) for lam in (0.1, 1.0, 10.0)]
    assert ranks[0] > ranks[1] > ranks[2]
    for rank in (8.0, 20.0):
        assert system.rank(system.lam_for_rank(rank)) == pytest.approx(rank, abs=0.05)
    with pytest.raises(ValueError, match="outside"):
        system.lam_for_rank(1e6)


def test_a_full_covariance_is_refused(problem):
    inv, _, _ = problem
    inv._chol = {"asc": None}
    try:
        with pytest.raises(ValueError, match="per-row weights"):
            NormalSystem(inv, **SOLVE)
    finally:
        inv._chol = None


def test_slabs_partition_the_rows_along_the_fault(problem):
    _, obs, direction = problem
    labels, width = slab_folds(obs, 6, direction)
    assert labels.shape == (obs.n,) and width > 0
    assert set(np.unique(labels)) == set(range(labels.max() + 1))
    axis = np.asarray(direction) / np.linalg.norm(direction)
    t = obs.ds["x"].values * axis[0] + obs.ds["y"].values * axis[1]
    means = [t[labels == k].mean() for k in range(labels.max() + 1)]
    assert np.all(np.diff(means) > 0), "slabs are numbered along the fault"
    shifted, _ = slab_folds(obs, 6, direction, shift=0.5)
    assert not np.array_equal(shifted, labels)
    # a span leaves the extent to its rows; the others join the end slabs
    span = np.zeros(obs.n, bool)
    span[: obs.n // 2] = True
    labels_span, width_span = slab_folds(obs, 6, direction, span=span)
    assert width_span != width and labels_span.shape == (obs.n,)


def test_cross_validation_finds_an_interior_minimum(problem):
    inv, obs, direction = problem
    system = NormalSystem(inv, **SOLVE)
    folds, _ = slab_folds(obs, 6, direction)
    choice = choose_rank(system, folds, ranks=(3.0, 120.0))
    assert isinstance(choice, RankChoice)
    assert 3.0 < choice.rank < 120.0
    assert choice.inside(choice.rank)
    assert choice.table.refine.any(), "a single partition is refined around its minimum"
    json.dumps(choice.to_dict(), default=float)
    # every row of the table is reproduced by the held-out residuals at its weight
    row = choice.table.iloc[len(choice.table) // 2]
    r = heldout_residuals(system, folds, row.lam)
    assert np.sqrt(np.mean(r[np.isfinite(r)] ** 2)) == pytest.approx(row.cv_rms, rel=1e-6)


def test_shifted_partitions_are_averaged(problem):
    inv, obs, direction = problem
    system = NormalSystem(inv, **SOLVE)
    partitions = [slab_folds(obs, 6, direction, shift=k / 3)[0] for k in range(3)]
    choice = choose_rank(system, partitions, ranks=(3.0, 120.0))
    assert len(choice.partition_ranks) == 3
    mean = choice.table[choice.table.partition == -1]
    single = choice.table[choice.table.partition == 0]
    assert len(mean) == len(single)
    stacked = np.mean([choice.table[choice.table.partition == k].cv_rms.to_numpy() for k in range(3)], axis=0)
    np.testing.assert_allclose(mean.cv_rms.to_numpy(), stacked)
    assert 3.0 < choice.rank < 120.0 and choice.inside(choice.rank)
