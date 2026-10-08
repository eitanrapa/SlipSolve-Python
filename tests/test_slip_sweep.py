"""Pins the segment-by-segment dip sweep.

The sweep rests on one identity -- the design matrix of any combination of dips is
the sum of per-segment blocks -- and on a solver that must agree with
:meth:`SlipInversion.solve`. Both are tested against the direct path. Recovery is
the acceptance test, as for the inversion itself: a stepped fault with a
different dip on each segment, forward-modelled independently (element-constant
slip on the joined segment meshes, through a raster and a quadtree), must come
back out of the sweep as the dips it was built with.
"""

import numpy as np
import pytest

from slipsolve import DipSweep, FaultMesh, FaultTrace, Observations, SlipInversion
from slipsolve.mesh import _arc_length, _depth_levels
from slipsolve.sweep import segment_breaks, segmented_layout

import slip_synthetic as ss

_LON = np.array([-68.9, -68.6, -68.3, -68.0, -67.7])
_LAT = np.array([10.50, 10.52, 10.53, 10.55, 10.56])

TRUE_DIPS = (70.0, 110.0)
GRID = (70.0, 90.0, 110.0)
RANK = 30.0
KW = dict(max_depth=15e3, edge_length=10e3, down_dip_levels=5, ramp="offset",
          polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0))
SOLVE = dict(ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=1.0,
             sides=("bottom", "left", "right"), polarity=(-1, 0, 0),
             strike=(-8.0, 8.0), dip=(-3.0, 3.0))


def _stepped_truth(trace, frame, dips):
    """The two-segment fault the sweep should find, built WITHOUT the sweep.

    Same layout recipe (``segmented_layout`` on the same breaks and levels), each
    segment cut out of a uniform-dip mesh by its elements' arc length, and the
    pieces joined -- so the truth's geometry does not come from the code whose
    block assembly is under test.
    """
    edge = KW["edge_length"]
    length = _arc_length(*trace.resample(edge, frame).to_local(frame))[-1]
    breaks = segment_breaks(length, count=len(dips))
    layout = segmented_layout(breaks, _depth_levels(0.0, KW["max_depth"],
                                                    KW["down_dip_levels"], 1.0), edge)
    pieces = []
    for j, dip in enumerate(dips):
        mesh = FaultMesh.from_layout(layout, trace, frame, uniform_dip=dip, edge_length=edge)
        s = mesh.element_params[:, 0]
        keep = np.flatnonzero((s > breaks[j]) & (s < breaks[j + 1]))
        nodes = np.unique(mesh.triangles[keep])
        remap = np.full(mesh.n_nodes, -1)
        remap[nodes] = np.arange(nodes.size)
        pieces.append(FaultMesh(mesh.nodes[nodes], remap[mesh.triangles[keep]],
                                mesh.params[nodes], frame=frame, attrs=dict(mesh.attrs)))
    return FaultMesh.join(pieces), length


def _truth_slip(mesh, length):
    """Right-lateral with a reverse component, smooth along strike, tapering with depth."""
    s, z = mesh.element_params[:, 0], mesh.element_params[:, 1]
    amplitude = (np.exp(-((s - 0.5 * length) / (0.3 * length)) ** 2)
                 * np.clip(1.0 + z / 12e3, 0.0, 1.0))
    return np.concatenate([-2.0 * amplitude, 0.8 * amplitude])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="sweep")
    frame = trace.local_frame()
    truth, length = _stepped_truth(trace, frame, TRUE_DIPS)
    slip = _truth_slip(truth, length)
    obs = Observations.concat([
        Observations.from_los(
            ss.forward_los_stack(truth, slip, trace, frame, geometry=geometry,
                                 spacing=6000.0, half_x=90e3, half_y=50e3,
                                 noise=0.001, seed=seed),
            name=geometry, frame=frame, trace=trace, rms_min=0.002,
            width_min=6000.0, width_max=40000.0, exclude_within=3000.0)
        for seed, geometry in enumerate(("asc", "desc"))])
    sweep = DipSweep(trace, obs, frame=frame, segments=2, dips=GRID, **KW)
    return trace, frame, obs, sweep


# -- the pieces --------------------------------------------------------------
def test_segment_breaks():
    np.testing.assert_allclose(segment_breaks(100e3, count=4), [0, 25e3, 50e3, 75e3, 100e3])
    np.testing.assert_allclose(segment_breaks(100e3, size=30e3), np.linspace(0, 100e3, 4))
    np.testing.assert_allclose(segment_breaks(100e3, breaks=[60e3, 20e3]), [0, 20e3, 60e3, 100e3])
    np.testing.assert_allclose(segment_breaks(100e3, breaks=[0.0, 40e3, 100e3]), [0, 40e3, 100e3])
    with pytest.raises(ValueError, match="exactly one"):
        segment_breaks(100e3, count=2, size=10e3)
    with pytest.raises(ValueError, match="on the trace"):
        segment_breaks(100e3, breaks=[120e3])


def test_every_boundary_is_a_node_column_and_no_triangle_straddles_one(setup):
    _, _, _, sweep = setup
    params = sweep.layout.params
    for b in sweep.breaks:
        on = params[params[:, 0] == b]
        assert np.unique(on[:, 1]).size == sweep.depths.size, f"break {b} is not a full column"
    s = params[:, 0][sweep.layout.triangles]
    seg_lo = np.searchsorted(sweep.breaks, s.min(axis=1), side="right") - 1
    seg_hi = np.searchsorted(sweep.breaks, s.max(axis=1), side="left") - 1
    np.testing.assert_array_equal(np.maximum(seg_lo, 0), np.maximum(seg_hi, 0))
    assert sum(e.size for e in sweep.segment_elements) == sweep.reference.n_elements


def test_one_dip_everywhere_is_exactly_the_direct_assembly(setup):
    """The identity the whole sweep rests on, including the shared boundary columns."""
    trace, frame, obs, sweep = setup
    for dip in (70.0, 110.0):
        mesh = FaultMesh.from_layout(sweep.layout, trace, frame, uniform_dip=dip,
                                     edge_length=KW["edge_length"])
        direct = SlipInversion(mesh, obs, basis="node").g
        np.testing.assert_allclose(sweep.design((dip, dip)), direct,
                                   rtol=1e-10, atol=1e-12 * np.abs(direct).max())


def test_the_solver_matches_slipinversion(setup):
    """Same objective, bounds and ramp as `SlipInversion.solve`, same answer."""
    _, _, obs, sweep = setup
    lam = sweep.evaluate((90.0, 90.0), rank=RANK)["lam"]
    result = sweep.evaluate((90.0, 90.0), smoothing=lam)
    model = SlipInversion(sweep.reference, obs, ramp="offset", basis="node").solve(
        smoothing=lam, **SOLVE)
    np.testing.assert_allclose(result["x"], model.x, atol=1e-5 * np.abs(model.x).max())
    assert result["rms"] == pytest.approx(model.rms_misfit, rel=1e-5)


def test_a_rank_target_is_hit(setup):
    _, _, _, sweep = setup
    for dips in ((90.0, 90.0), TRUE_DIPS):
        assert sweep.evaluate(dips, rank=RANK)["rank"] == pytest.approx(RANK, abs=1e-3)


def test_exactly_one_of_rank_or_smoothing(setup):
    _, _, _, sweep = setup
    with pytest.raises(ValueError, match="exactly one"):
        sweep.evaluate((90.0, 90.0))
    with pytest.raises(ValueError, match="one dip per segment"):
        sweep.evaluate((90.0, 90.0, 90.0), rank=RANK)


# -- the search --------------------------------------------------------------
def test_recovers_the_planted_dips(setup):
    """Acceptance: a stepped 70 / 110 fault comes back as 70 / 110."""
    _, _, _, sweep = setup
    result = sweep.run(rank=RANK, method="exhaustive")

    assert len(result.table) == len(GRID) ** 2
    assert result.best == TRUE_DIPS
    profile = result.profiles()
    for j, dip in enumerate(TRUE_DIPS):
        rows = profile[profile.segment == j + 1]
        assert rows.loc[rows.score.idxmin(), "dip"] == dip
    uniform = result.table[(result.table.dip_1 == result.table.dip_2)].score.min()
    assert result.table.score.iloc[0] < 0.9 * uniform, "a stepped truth must beat every uniform dip"


def test_the_coordinate_search_finds_the_exhaustive_optimum(setup):
    _, _, _, sweep = setup
    exhaustive = sweep.run(rank=RANK, method="exhaustive")
    coordinate = sweep.run(rank=RANK, method="coordinate")
    assert coordinate.best == exhaustive.best
    assert len(coordinate.table) <= len(exhaustive.table)


def test_a_fixed_segment_stays_fixed(setup):
    _, _, _, sweep = setup
    result = sweep.run(rank=RANK, fix={0: 90.0})
    assert (result.table.dip_1 == 90.0).all()
    assert len(result.table) == len(GRID)
    assert result.best[0] == 90.0


def test_planted_data_on_the_sweeps_own_geometry_is_found(setup):
    """`predict` + `data=`: the recovery test a real coverage needs, run on synthetic data."""
    _, _, _, sweep = setup
    planted = (110.0, 70.0)
    x = sweep.evaluate(planted, rank=RANK)["x"]
    data = sweep.predict(planted, x)
    result = sweep.run(rank=RANK, data=data, method="exhaustive")
    assert result.best == planted
