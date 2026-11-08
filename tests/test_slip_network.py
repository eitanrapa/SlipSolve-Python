"""Interseismic networks: block pairs that change along a fault, intra-block strain, and imported surfaces."""

import copy
import warnings

import numpy as np
import pytest

from slipsolve import (FaultMesh, FaultTrace, InterseismicInversion, Observations, TSurf, adapt_layout, block_labels, cell_folds, gnss_observations,
                       monte_carlo, read_tsurf, rotation_columns, strain_from_coefficients)
from slipsolve.interseismic import EulerPole, _axes_at, _position, _strain_vectors

# A main fault running north, and a branch leaving it at 36 N to the north-north-east. West of the
# main fault is one block; a wedge lies between its northern half and the branch; the rest is east.
MAIN = np.array([[-120.0, 35.0], [-120.0, 37.5]])
BRANCH = np.array([[-120.0, 36.0], [-119.6, 37.5]])
DEPTH = 20e3
WEDGE = ([-120.0, -119.2, -120.0], [36.0, 39.0, 39.0])
WEST = ([-130.0, -120.0, -120.0, -130.0], [30.0, 30.0, 40.0, 40.0])


def _translation(east, north, centre):
    """The pole that moves a block at `centre` with these velocities there, m/yr."""
    axis_e, axis_n, _ = _axes_at(centre)
    return EulerPole(east * axis_e + north * axis_n)


def _with_data(inversion, data):
    out = copy.copy(inversion)
    out.d = np.asarray(data, dtype=float)
    return out


@pytest.fixture(scope="module")
def network():
    """The Y, with stations around it; data to be supplied."""
    main = FaultTrace(MAIN[:, 0], MAIN[:, 1], name="main")
    branch = FaultTrace(BRANCH[:, 0], BRANCH[:, 1], name="branch")
    frame = main.local_frame()
    mesh = FaultMesh.join([FaultMesh.vertical(main, frame, max_depth=DEPTH, edge_length=20e3),
                           FaultMesh.vertical(branch, frame, max_depth=DEPTH, edge_length=20e3)])
    rng = np.random.default_rng(0)
    lon = rng.uniform(-122.0, -117.5, 220)
    lat = rng.uniform(34.5, 38.0, 220)
    x, y = frame.to_local(lon, lat)
    keep = (main.distance(x, y, frame) > 8e3) & (branch.distance(x, y, frame) > 8e3)
    lon, lat = lon[keep], lat[keep]
    zero = np.zeros(lon.size)
    obs = Observations.concat([gnss_observations(lon, lat, zero, zero, None, 1e-3, 1e-3, None, frame, components="en")], normalize="none")
    blocks = block_labels(obs, {"west": WEST, "wedge": WEDGE}, default="east")
    length = main.length(frame)
    # The main fault runs north, so the west is on its left; the branch too, so the wedge is on its left.
    faults = [[(None, 0.4 * length, "west", "east"), (0.4 * length, None, "west", "wedge")], ("wedge", "east")]
    return dict(main=main, branch=branch, frame=frame, mesh=mesh, obs=obs, lon=lon, lat=lat, blocks=blocks, faults=faults, length=length)


def _rigid_data(net, velocities):
    """`(inversion, poles, rows)`: each block moving with `{block: (east, north)}`, m/yr at its centre; `rows` their velocities."""
    inv = InterseismicInversion(net["mesh"], net["obs"], blocks=net["blocks"], faults=net["faults"])
    poles = {name: _translation(*velocities[name], inv.block_centres[name]) for name in velocities}
    n = net["lon"].size
    ve, vn = np.zeros(n), np.zeros(n)
    for name, pole in poles.items():
        rows = net["blocks"][:n] == name          # all the east rows, then all the north rows
        ve[rows], vn[rows], _ = pole.velocity(net["lon"][rows], net["lat"][rows])
    return inv, poles, np.r_[ve, vn]


VELOCITIES = {"west": (0.0, -0.030), "wedge": (0.0, -0.012), "east": (0.0, 0.0)}


def test_a_segment_change_changes_the_rate_at_it(network):
    inv, poles, _ = _rigid_data(network, VELOCITIES)
    v_s, v_d = inv.long_term_slip(poles)
    labels = inv.fault_labels()
    s = inv.mesh.params[:, 0]
    main = labels == 0
    south_part = main & (s < 0.4 * network["length"] - 30e3)
    north_part = main & (s > 0.4 * network["length"] + 30e3)
    # West moving south past its neighbour, on the trace's left, is left-lateral: positive here.
    assert np.allclose(v_s[south_part], 0.030, atol=1e-3)
    assert np.allclose(v_s[north_part], 0.018, atol=1e-3)
    assert np.all(v_d == 0.0), "vertical faults take no dip-slip rate"
    # Each node is in one segment: the rate steps at the junction, and does not blend.
    assert set(np.round(1e3 * v_s[main], -1)) <= {20.0, 30.0}


def test_rates_close_at_a_junction(network):
    """Every rate is a difference of one set of rotations: south of the Y carries what the two arms do."""
    inv, poles, _ = _rigid_data(network, VELOCITIES)
    v_s, _ = inv.long_term_slip(poles)
    labels = inv.fault_labels()
    s = inv.mesh.params[:, 0]
    length = network["length"]
    just_south = v_s[(labels == 0) & (s < 0.4 * length) & (s > 0.4 * length - 25e3)].mean()
    just_north = v_s[(labels == 0) & (s >= 0.4 * length) & (s < 0.4 * length + 25e3)].mean()
    arm = v_s[(labels == 1) & (s < 25e3)].mean()
    tangent = np.array([(BRANCH[1, 0] - BRANCH[0, 0]) * np.cos(np.radians(36.75)), BRANCH[1, 1] - BRANCH[0, 1]])
    along_north = tangent[1] / np.linalg.norm(tangent)
    assert just_south == pytest.approx(just_north + arm / along_north, abs=2e-4)


def test_the_network_recovers_planted_rotation_and_coupling(network):
    """All three faults locked to 10 km, ends free: the rates, the blocks and the depth come back."""
    inv, poles, rotation = _rigid_data(network, VELOCITIES)
    n = inv.basis.n_basis
    v_s, _ = inv.long_term_slip(poles)
    z = inv.mesh.params[:, 1]
    coupling = np.clip((z + 12e3) / 4e3, 0.0, 1.0)
    data = rotation + inv.g[:, :n] @ (-coupling * v_s)
    solved = _with_data(inv, data).solve(smoothing=3e2, sides=("bottom",))
    table = solved.fault_table()
    assert list(table["left"]) == ["west", "west", "wedge"] and list(table["right"]) == ["east", "wedge", "east"]
    assert table["strike_rate"].iloc[0] == pytest.approx(30.0, abs=1.5)
    assert table["strike_rate"].iloc[1] == pytest.approx(18.0, abs=1.5)
    assert table["strike_rate"].iloc[2] == pytest.approx(1e3 * v_s[inv.fault_labels() == 1].mean(), abs=1.5)
    assert table["moment"].sum() == pytest.approx(solved.moment(), rel=0.02)
    main = inv.fault_labels() == 0
    assert np.nanmedian(solved.coupling[main & (z > -5e3)]) > 0.7
    assert np.nanmedian(solved.coupling[main & (z < -15e3)]) < 0.3
    blocks = solved.block_table()
    assert list(blocks["block"]) == ["east", "wedge", "west"] and (blocks["rms_residual"] < 1.0).all()
    assert (blocks["rms_less_rotation"] < blocks["rms_data"]).any()
    copies = monte_carlo(solved, n=3, noise="weights", seed=1)
    assert len(copies) == 3 and copies.apply(lambda m: m.long_term_slip[0].mean()).shape == (3,)


def test_freeing_the_ends_is_what_sides_does(network):
    inv, poles, rotation = _rigid_data(network, VELOCITIES)
    n = inv.basis.n_basis
    v_s, _ = inv.long_term_slip(poles)
    data = rotation + inv.g[:, :n] @ (-0.8 * v_s)
    solved = _with_data(inv, data).solve(smoothing=3e2, sides=("bottom",))
    pinned = _with_data(inv, data).solve(smoothing=3e2)
    labels, s, z = inv.fault_labels(), inv.mesh.params[:, 0], inv.mesh.params[:, 1]
    end = (labels == 1) & (s == s[labels == 1].min()) & (z > -8e3)
    assert np.abs(pinned.strike_slip[end]).max() < 1e-3 and np.abs(solved.strike_slip[end]).max() > 3e-3


def test_segments_are_checked(network):
    length = network["length"]
    args = (network["mesh"], network["obs"])
    kw = dict(blocks=network["blocks"])
    with pytest.raises(ValueError, match="cover"):
        InterseismicInversion(*args, faults=[[(0.2 * length, None, "west", "east")], ("wedge", "east")], **kw)
    with pytest.raises(ValueError, match="must meet"):
        InterseismicInversion(*args, faults=[[(None, 0.3 * length, "west", "east"), (0.5 * length, None, "west", "wedge")],
                                             ("wedge", "east")], **kw)
    with pytest.raises(ValueError, match="no rows"):
        InterseismicInversion(*args, faults=[("west", "nowhere"), ("wedge", "east")], **kw)
    with pytest.raises(ValueError, match="entries for 2"):
        InterseismicInversion(*args, faults=[("west", "east")], **kw)
    # (left, left) is how a stretch says it does not slip: its limits collapse to zero.
    quiet = InterseismicInversion(*args, faults=[[(None, 0.4 * length, "west", "east"), (0.4 * length, None, "west", "west")],
                                                 ("wedge", "east")], **kw)
    v_s, _ = quiet.long_term_slip(quiet.rigid_poles())
    s, labels = quiet.mesh.params[:, 0], quiet.fault_labels()
    assert np.all(v_s[(labels == 0) & (s > 0.4 * length + 1.0)] == 0.0)


def test_a_swapped_segment_warns(network):
    length = network["length"]
    with pytest.warns(RuntimeWarning, match="other side"):
        InterseismicInversion(network["mesh"], network["obs"], blocks=network["blocks"],
                              faults=[[(None, 0.4 * length, "east", "west"), (0.4 * length, None, "west", "wedge")], ("wedge", "east")])


def test_cell_folds(network):
    obs = network["obs"]
    folds = cell_folds(obs, cell=100e3, n_folds=5, seed=3)
    values = folds.values
    assert values.shape == (obs.n,) and set(np.unique(values)) == set(range(5)) and folds.attrs["direction"] == "cells"
    assert np.array_equal(values, cell_folds(obs, cell=100e3, n_folds=5, seed=3).values)
    assert sum(folds.attrs["counts"]) == obs.n
    n = network["lon"].size
    assert np.array_equal(values[:n], values[n:]), "a station's components stay together"
    ds = obs.ds
    cell = np.column_stack([np.floor(ds["x"].values / 100e3), np.floor(ds["y"].values / 100e3)])
    for corner in np.unique(cell, axis=0):
        assert np.unique(values[(cell == corner).all(axis=1)]).size == 1
    with pytest.raises(ValueError, match="cannot fill"):
        cell_folds(obs, cell=1e7, n_folds=8)
    # Dealt block by block, no fold takes the whole of a block spanning several cells.
    blocks = network["blocks"]
    dealt = cell_folds(obs, cell=30e3, n_folds=4, seed=1, groups=blocks).values
    for name in np.unique(blocks):
        cells = np.unique(np.floor(np.column_stack([ds["x"].values, ds["y"].values])[blocks == name] / 30e3), axis=0)
        if cells.shape[0] > 1:
            assert np.unique(dealt[blocks == name]).size > 1, name
    with pytest.raises(ValueError, match="labels"):
        cell_folds(obs, groups=blocks[:-1])


def test_a_fold_that_empties_a_block_warns(network):
    from slipsolve.resolution import _fold_problems

    inv = InterseismicInversion(network["mesh"], network["obs"], blocks=network["blocks"], faults=network["faults"])
    fold = np.where(network["blocks"] == "wedge", 2, np.arange(network["obs"].n) % 2)
    with pytest.warns(RuntimeWarning, match="fold 2 leaves block.*wedge"):
        _fold_problems(inv, fold)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _fold_problems(inv, np.r_[np.zeros(network["lon"].size, dtype=int), np.ones(network["lon"].size, dtype=int)])


# -- strain -------------------------------------------------------------------------------------
def test_strain_columns_are_the_uniform_field(network):
    obs = network["obs"]
    n = network["lon"].size
    matrix, labels, centres = rotation_columns(obs, np.full(obs.n, "all"), strain=True)
    assert labels[-3:] == ["block:all:exx", "block:all:exy", "block:all:eyy"] and matrix.shape[1] == 6
    centre = centres["all"]
    position = _position(network["lon"], network["lat"])
    lon, lat = np.radians(centre)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]), east)
    dx, dy = position @ east / 1e6, position @ north / 1e6
    # Stretching along x at 1 m/yr per 1000 km moves a station east by its offset, and not north,
    # to the few percent a station's own east axis turns from the centre's.
    assert np.abs(matrix[:n, 3] - dx).max() < 0.05 * np.abs(dx).max()
    assert np.abs(matrix[n:, 3]).max() < 0.05 * np.abs(dx).max()
    # The shear's east part is the north offset and its north part the east offset: no rotation.
    assert np.abs(matrix[:n, 4] - dy).max() < 0.05 * np.abs(dy).max()
    assert np.abs(matrix[n:, 4] - dx).max() < 0.05 * np.abs(dx).max()
    a, b, c = _strain_vectors(centre, position)
    assert np.allclose(a, dx[:, None] * east)
    assert np.allclose(b, dy[:, None] * east + dx[:, None] * north)
    assert np.allclose(c, dy[:, None] * north)


def test_strain_needs_rows_and_known_blocks(network):
    obs = network["obs"]
    with pytest.raises(ValueError, match="no rows"):
        rotation_columns(obs, network["blocks"], strain=("nowhere",))
    few = np.full(obs.n, "main", dtype=object)
    few[:4] = "tiny"
    with pytest.raises(ValueError, match="at least six"):
        rotation_columns(obs, few.astype(str), strain=("tiny",))
    with pytest.raises(ValueError, match="needs blocks"):
        InterseismicInversion(network["mesh"], obs, strain=True)
    assert strain_from_coefficients(["block:a:ve", "block:a:exx", "block:a:exy", "block:a:eyy"], [1.0, 2.0, 3.0, 4.0]) == {"a": (2.0, 3.0, 4.0)}


def test_a_long_term_rate_does_not_depend_on_where_strain_is_referred_to(network):
    """Why strain counts toward the rate: the total field at the fault is defined, the rotation alone is not."""
    obs, blocks = network["obs"], network["blocks"]
    strained = ("west", "east")          # the wedge has too few stations for half of them to fit it
    full = InterseismicInversion(network["mesh"], obs, blocks=blocks, faults=network["faults"], strain=strained)
    label = {name: i for i, name in enumerate(full.ramp_labels)}
    truth = np.zeros(full.n_ramp)
    for name, value in (("block:west:ve", 0.01), ("block:west:exx", 0.02), ("block:west:exy", -0.05), ("block:wedge:vn", 0.003),
                        ("block:east:vn", 0.004), ("block:east:exy", 0.04)):
        truth[label[name]] = value
    data = full.ramp @ truth
    poles, strain = _with_data(full, data).rigid_motion(far=0.0)
    rate = full.long_term_slip(poles, strain)

    # The same field seen by half the stations: other centres, so other rotations and strains' origin.
    n = network["lon"].size
    half = np.arange(0, n, 2)
    rows = np.r_[half, half + n]
    other = InterseismicInversion(network["mesh"], Observations(obs.ds.isel(obs=rows)), blocks=blocks[rows], faults=network["faults"], strain=strained)
    moved = max(float(np.abs(np.subtract(other.block_centres[name], full.block_centres[name])).max()) for name in other.block_centres)
    assert moved > 0.03, "the test needs the centres to move"
    poles_other, strain_other = _with_data(other, data[rows]).rigid_motion(far=0.0)
    again = other.long_term_slip(poles_other, strain_other)
    assert np.abs(again[0] - rate[0]).max() < 2e-5, "the whole field at the fault agrees to 0.02 mm/yr"
    # The rotation alone does not, by much more.
    assert np.abs(other.long_term_slip(poles_other)[0] - full.long_term_slip(poles)[0]).max() > 2e-4


def test_a_planted_strain_comes_back_and_survives_saving(network, tmp_path):
    obs, blocks = network["obs"], network["blocks"]
    inv = InterseismicInversion(network["mesh"], obs, blocks=blocks, faults=network["faults"], strain=("east",))
    truth = np.zeros(inv.n_ramp)
    label = {name: i for i, name in enumerate(inv.ramp_labels)}
    truth[label["block:east:exx"]] = 0.05
    truth[label["block:east:ve"]] = 0.002
    truth[label["block:west:vn"]] = -0.02
    solved = _with_data(inv, inv.ramp @ truth).solve(smoothing=1e3, sides=("bottom",))
    assert solved.strain["east"][0] == pytest.approx(0.05, abs=2e-3) and set(solved.strain) == {"east"}
    assert np.abs(solved.residual).max() < 1e-3
    assert "exx" in solved.block_table().columns
    path = tmp_path / "strained.slip.zip"
    solved.save(path)
    back = type(solved).load(path)
    assert back.strain["east"] == pytest.approx(solved.strain["east"])
    assert back.poles["west"].rate == pytest.approx(solved.poles["west"].rate)


# -- t-surf ---------------------------------------------------------------------------------------
_CRS = "EPSG:32611"


def _surface_cross(s, z, length):
    """A fault that dips east at one end and is vertical at the other, curving as it goes."""
    dip_offset = 0.25 * (1.0 - s / length) + 0.05 * np.sin(np.pi * s / length)
    return dip_offset * z


def _write_tsurf(path, trace, frame, spacing=2e3, depth=18e3, z_positive="Elevation", unit="m", atoms=True, name="test_fault"):
    from pyproj import Transformer

    length = trace.length(frame)
    s = np.arange(0.0, length + 1.0, spacing)
    z = -np.arange(0.0, depth + 1.0, spacing)
    ss, zz = np.meshgrid(s, z)
    x, y = trace.from_curvilinear(ss.ravel(), _surface_cross(ss.ravel(), zz.ravel(), length), frame)
    lon, lat = frame.to_lonlat(x, y)
    easting, northing = Transformer.from_crs("EPSG:4326", _CRS, always_xy=True).transform(lon, lat)
    factor = 1e-3 if unit == "km" else 1.0
    sign = -1.0 if z_positive == "Depth" else 1.0
    lines = ["GOCAD TSurf 1", "HEADER {", f"name:{name}", "*solid*color: 0.5 0.5 0.5 1", "}", "GOCAD_ORIGINAL_COORDINATE_SYSTEM", "NAME Default",
             'AXIS_NAME "X" "Y" "Z"', f'AXIS_UNIT "{unit}" "{unit}" "{unit}"', f"ZPOSITIVE {z_positive}", "END_ORIGINAL_COORDINATE_SYSTEM", "TFACE"]
    n_s, n_z = s.size, z.size
    # Numbered from 10 by twos, as files with holes in their numbering are, and half the rows as PVRTX.
    number = lambda i, j: 10 + 2 * (j * n_s + i)
    k = 0
    for j in range(n_z):
        for i in range(n_s):
            word = "PVRTX" if j % 2 else "VRTX"
            tail = " 1.0 2.0" if j % 2 else ""
            lines.append(f"{word} {number(i, j)} {easting[k] * factor:.3f} {northing[k] * factor:.3f} {sign * zz.ravel()[k] * factor:.3f}{tail}")
            k += 1
    alias = 10 ** 6
    if atoms:
        lines.append(f"ATOM {alias} {number(1, 1)}")
    for j in range(n_z - 1):
        for i in range(n_s - 1):
            a, b, c, d = number(i, j), number(i + 1, j), number(i, j + 1), number(i + 1, j + 1)
            first = alias if atoms and (i, j) == (1, 1) else a
            lines.append(f"TRGL {first} {b} {d}")
            lines.append(f"TRGL {first} {d} {c}")
    lines += [f"BSTONE {number(0, 0)}", f"BORDER {10 ** 7} {number(0, 0)} {number(1, 0)}", "END"]
    path.write_text("\n".join(lines) + "\n")
    return length


@pytest.fixture(scope="module")
def curved(tmp_path_factory):
    trace = FaultTrace(np.array([-118.6, -118.3, -118.0, -117.7]), np.array([35.0, 35.12, 35.18, 35.2]), name="drawn")
    frame = trace.local_frame()
    path = tmp_path_factory.mktemp("ts") / "fault.ts"
    length = _write_tsurf(path, trace, frame)
    return dict(trace=trace, frame=frame, path=path, length=length)


def test_read_tsurf(curved, tmp_path):
    surfaces = read_tsurf(curved["path"], crs=_CRS)
    assert len(surfaces) == 1
    surface = surfaces[0]
    assert surface.name == "test_fault" and surface.crs == _CRS
    assert surface.vertices[:, 2].max() == 0.0 and surface.vertices[:, 2].min() == -18e3
    assert surface.triangles.min() == 0 and surface.triangles.max() == surface.vertices.shape[0] - 1, "numbers re-indexed, and unused vertices dropped"
    n_s = int(curved["length"] // 2e3) + 1
    assert surface.vertices.shape[0] == n_s * 10 and surface.triangles.shape[0] == 2 * (n_s - 1) * 9
    # The same surface however the file says it: depth positive, kilometres.
    other = tmp_path / "depth_km.ts"
    _write_tsurf(other, curved["trace"], curved["frame"], z_positive="Depth", unit="km", atoms=False)
    again = read_tsurf(other, crs=_CRS)[0]
    assert np.abs(again.vertices - surface.vertices).max() < 1.0
    # Two surfaces in a file.
    both = tmp_path / "both.ts"
    both.write_text(curved["path"].read_text() + curved["path"].read_text().replace("test_fault", "second"))
    assert [s.name for s in read_tsurf(both)] == ["test_fault", "second"]
    empty = tmp_path / "nothing.ts"
    empty.write_text("not a surface\n")
    with pytest.raises(ValueError, match="no triangulated surface"):
        read_tsurf(empty)
    with pytest.raises(ValueError, match="no crs"):
        read_tsurf(curved["path"])[0].lonlat()


def test_a_surface_with_no_top_edge_is_refused():
    tetrahedron = np.array([[0.0, 0.0, 0.0], [1e3, 0.0, 0.0], [0.0, 1e3, 0.0], [0.0, 0.0, -1e3]])
    closed = TSurf(tetrahedron, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], name="closed")
    with pytest.raises(ValueError, match="closed"):
        closed.top_edge()


def test_the_trace_follows_the_top_edge(curved):
    surface = read_tsurf(curved["path"], crs=_CRS)[0]
    trace = surface.trace()
    x, y = trace.to_local(curved["frame"])
    assert np.abs(curved["trace"].distance(x, y, curved["frame"])).max() < 5.0
    assert x[-1] > x[0], "west to east"
    flipped = surface.trace(reverse=True)
    assert flipped.to_local(curved["frame"])[0][-1] < x[-1] - 1e3
    assert trace.length(curved["frame"]) == pytest.approx(curved["length"], abs=2.5e3), "the file's columns stop within 2 km of the end"


def test_a_mesh_from_a_surface_lies_on_it(curved):
    surface = read_tsurf(curved["path"], crs=_CRS)[0]
    frame = curved["frame"]
    mesh = FaultMesh.from_tsurf(surface, surface.trace(), frame, edge_length=4e3)
    assert mesh.attrs["tsurf"] == "test_fault" and mesh.attrs["kind"] == "curved"
    assert mesh.nodes[:, 2].max() == 0.0 and mesh.nodes[:, 2].min() == pytest.approx(-18e3)
    assert mesh.attrs["tsurf_misfit_rms"] < 60.0, "on a 4 km mesh of a smooth surface"
    assert mesh.attrs["tsurf_misfit_max"] < 300.0
    s, cross = curved["trace"].to_curvilinear(mesh.nodes[:, 0], mesh.nodes[:, 1], frame)
    inside = (s > 5e3) & (s < curved["length"] - 5e3)
    expected = _surface_cross(s, mesh.nodes[:, 2], curved["length"])
    assert np.sqrt(np.mean((cross[inside] - expected[inside]) ** 2)) < 100.0
    # A mesh that inverts: winding, smoothing edges and the Green's functions all stand.
    assert mesh.n_elements > 50 and np.all(mesh.areas > 0.0)
    deeper = FaultMesh.from_tsurf(surface, frame=frame, edge_length=4e3, max_depth=10e3)
    assert deeper.nodes[:, 2].min() == pytest.approx(-10e3)


def test_a_new_layout_goes_on_the_surface_already_fitted(curved):
    """Remeshing an imported fault keeps the fault: its nodes still lie on it, on any lattice along the same trace."""
    surface = read_tsurf(curved["path"], crs=_CRS)[0]
    frame = curved["frame"]
    trace = surface.trace()
    mesh = FaultMesh.from_tsurf(surface, trace, frame, edge_length=4e3)
    z = mesh.params[:, 1]
    # Finer near the surface than down deep, and not on the old levels.
    sizes = np.interp(-mesh.element_params[:, 1], [0.0, 18e3], [2.5e3, 7e3])
    layout = adapt_layout(mesh, sizes, budget=mesh.n_elements)
    moved = FaultMesh.from_layout(layout, trace, frame, surface=mesh.surface, edge_length=4e3)
    assert not np.array_equal(moved.params, mesh.params) and moved.attrs["kind"] == "curved"
    assert abs(moved.n_elements - mesh.n_elements) <= 0.25 * mesh.n_elements
    assert moved.nodes[:, 2].max() == 0.0 and moved.nodes[:, 2].min() == pytest.approx(z.min())
    along, cross = curved["trace"].to_curvilinear(moved.nodes[:, 0], moved.nodes[:, 1], frame)
    inside = (along > 5e3) & (along < curved["length"] - 5e3)
    expected = _surface_cross(along, moved.nodes[:, 2], curved["length"])
    assert np.sqrt(np.mean((cross[inside] - expected[inside]) ** 2)) < 100.0, "as close as the mesh it came from"
    # Without a layout, a surface on another lattice is still a mistake.
    with pytest.raises(ValueError, match="different lattice"):
        FaultMesh.curved(trace, frame, surface=mesh.surface, edge_length=3e3, max_depth=18e3)


def test_a_network_runs_on_an_imported_surface(curved):
    surface = read_tsurf(curved["path"], crs=_CRS)[0]
    frame = curved["frame"]
    trace = surface.trace()
    mesh = FaultMesh.from_tsurf(surface, trace, frame, edge_length=5e3)
    rng = np.random.default_rng(2)
    lon, lat = rng.uniform(-119.5, -117.0, 70), rng.uniform(34.4, 35.9, 70)
    x, y = frame.to_local(lon, lat)
    keep = trace.distance(x, y, frame) > 8e3
    lon, lat = lon[keep], lat[keep]
    zero = np.zeros(lon.size)
    obs = Observations.concat([gnss_observations(lon, lat, zero, zero, None, 1e-3, 1e-3, None, frame, components="en")], normalize="none")
    side = trace.side(obs.ds["x"].values.astype(float), obs.ds["y"].values.astype(float), frame)
    blocks = np.where(side > 0, "north", "south")
    inv = InterseismicInversion(mesh, obs, blocks=blocks, faults=("north", "south"))
    assert inv.g.shape[1] == 2 * mesh.n_nodes
    poles = {name: _translation(0.0, -0.015 if name == "north" else 0.015, inv.block_centres[name]) for name in ("north", "south")}
    rates = inv.long_term_slip(poles)
    assert np.isfinite(rates[0]).all()


def _lattice_surface(trace, frame, s, depth=18e3, spacing=2e3, name="piece"):
    """A surface on `(s, z)`, as a TSurf in the test's CRS, straight from arrays."""
    from pyproj import Transformer

    length = trace.length(frame)
    z = -np.arange(0.0, depth + 1.0, spacing)
    ss, zz = np.meshgrid(s, z)
    x, y = trace.from_curvilinear(ss.ravel(), _surface_cross(ss.ravel(), zz.ravel(), length), frame)
    lon, lat = frame.to_lonlat(x, y)
    easting, northing = Transformer.from_crs("EPSG:4326", _CRS, always_xy=True).transform(lon, lat)
    n_s = s.size
    triangles = []
    for j in range(z.size - 1):
        for i in range(n_s - 1):
            a, b, c, d = j * n_s + i, j * n_s + i + 1, (j + 1) * n_s + i, (j + 1) * n_s + i + 1
            triangles += [[a, b, d], [a, d, c]]
    return TSurf(np.column_stack([easting, northing, zz.ravel()]), triangles, name=name, crs=_CRS)


def test_merged_sections_make_one_fault(curved):
    trace, frame, length = curved["trace"], curved["frame"], curved["length"]
    # Three sections, a 6 km gap in the second joint. The trace runs through their top edges however they were built.
    a = _lattice_surface(trace, frame, np.arange(0.0, 0.4 * length, 2e3), name="a")
    b = _lattice_surface(trace, frame, np.arange(0.4 * length, 0.7 * length, 2e3)[::-1], name="b")
    c = _lattice_surface(trace, frame, np.arange(0.7 * length + 6e3, length, 2e3), name="c")
    merged = TSurf.merged([a, b, c], name="all")
    assert merged.vertices.shape[0] == a.vertices.shape[0] + b.vertices.shape[0] + c.vertices.shape[0]
    assert merged.triangles.max() == merged.vertices.shape[0] - 1
    through = merged.trace()
    x, y = through.to_local(frame)
    assert x[-1] > x[0] and np.all(np.diff(x) > -1.0), "one way along, however each section was digitised"
    assert np.abs(trace.distance(x, y, frame)).max() < 50.0
    step = np.hypot(np.diff(x), np.diff(y))
    assert 5e3 < step.max() < 9e3, "a straight line bridges the gap"
    mesh = FaultMesh.from_tsurf(merged, through, frame, edge_length=4e3)
    assert mesh.attrs["tsurf"] == "all" and mesh.attrs["tsurf_misfit_rms"] < 150.0
    with pytest.raises(ValueError, match="merged from 3"):
        merged.boundary_loops()
    with pytest.raises(ValueError, match="different coordinate"):
        TSurf.merged([a, TSurf(c.vertices, c.triangles, crs="EPSG:4326")])
    with pytest.raises(ValueError, match="at least one"):
        TSurf.merged([])


def test_the_top_edge_loses_its_lateral_ends_and_spikes():
    # A flat-topped rectangle in plan: 40 km long and 20 km deep, whose lateral edges lean out, and with a
    # vertex on the top edge that doubles back.
    s = np.arange(0.0, 41e3, 2e3)
    z = -np.arange(0.0, 21e3, 1e3)
    ss, zz = np.meshgrid(s, z)
    x = ss + 0.8 * (zz * (ss == 0.0) - zz * (ss == s[-1]))
    y = 0.01 * ss
    vertices = np.column_stack([x.ravel(), y.ravel(), zz.ravel()])
    n_s = s.size
    triangles = []
    for j in range(z.size - 1):
        for i in range(n_s - 1):
            a, b, c, d = j * n_s + i, j * n_s + i + 1, (j + 1) * n_s + i, (j + 1) * n_s + i + 1
            triangles += [[a, b, d], [a, d, c]]
    surface = TSurf(vertices, triangles, name="box")
    edge = surface.top_edge(fraction=0.1)
    assert np.all(surface.vertices[edge, 2] == 0.0), "the two lateral edges' top 2 km are gone"
    assert edge.size == n_s
    # Now a spike on the top edge, 3 km back along the fault and off it: its vertex goes.
    spiked = vertices.copy()
    spiked[10, 0] = s[10] - 3e3
    bad = TSurf(spiked, triangles, name="spike")
    cleaned = bad.top_edge(fraction=0.1)
    steps = np.diff(bad.vertices[cleaned, 0])
    assert cleaned.size < n_s and (np.all(steps > 0.0) or np.all(steps < 0.0)), "one way along, once the kink has gone"
    # A top edge too rough to follow: its high points stand apart.
    rough = zz.copy()
    rough[0, ::4] += 5e3
    with pytest.raises(ValueError, match="raise fraction"):
        TSurf(np.column_stack([x.ravel(), y.ravel(), rough.ravel()]), triangles, name="rough").top_edge(fraction=0.01)


def test_below_the_surface(curved):
    surface = read_tsurf(curved["path"], crs=_CRS)[0]
    frame = curved["frame"]
    trace = surface.trace()
    vertical = FaultMesh.from_tsurf(surface, trace, frame, edge_length=4e3, max_depth=38e3)
    dipping = FaultMesh.from_tsurf(surface, trace, frame, edge_length=4e3, max_depth=38e3, below="dip")

    def offset(mesh, depth):
        rows = np.abs(mesh.nodes[:, 2] + depth) < 2.1e3
        columns = (mesh.params[:, 0] > 8e3) & (mesh.params[:, 0] < curved["length"] - 8e3)
        return np.median(np.abs(trace.to_curvilinear(mesh.nodes[rows & columns, 0], mesh.nodes[rows & columns, 1], frame)[1]))

    assert abs(offset(vertical, 36e3) - offset(vertical, 22e3)) < 0.1 * offset(vertical, 22e3), "straight down from the bottom edge"
    assert offset(dipping, 36e3) > 1.5 * offset(dipping, 22e3), "or carrying on at its dip"
    assert vertical.attrs["tsurf_deepest"] == pytest.approx(18e3)
    with pytest.raises(ValueError, match="below must"):
        FaultMesh.from_tsurf(surface, trace, frame, edge_length=4e3, max_depth=38e3, below="flat")
