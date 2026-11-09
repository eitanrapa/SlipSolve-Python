"""A regional block model as the far field of a local interseismic model.

A local model (`slipsolve.interseismic.InterseismicInversion`) has its own blocks and faults, a piece of a wider system.
A regional block model of that system describes the rest: how the blocks the local ones sit in rotate, and the faults
the local model leaves out, each locked to a depth and slipping as its two blocks' rotations ask (Meade & Loveless
2009). `RegionalModel` holds a solved one, from `RegionalModel.solve`, and evaluates it at the local model's observation
points, in the two forms the local inversion takes:

- `RegionalModel.rotation_priors`: each local block's three rotation coefficients tied to the rotation of the regional
  block it lies in, as `nuisance_prior` entries with a mean;
- `RegionalModel.far_field`: the velocity of what the local model leaves out, as `slipsolve.patterns.Pattern`s to hold
  at 1, in parts:
  - `rotation`: where a regional block is smaller than the local one, its rotation less the one the local block is
    tied to;
  - `inside` and `outside`: the field of the regional faults inside and outside the local blocks, without their tensile
    slip;
  - `inside_tensile` and `outside_tensile`: their tensile slip, which is how rigid blocks take fault-normal motion
    across a vertical segment;
  - `meshes`: the regional model's triangulated faults.

**Fit each part's amplitude freely before holding any at 1.** A regional model's own picture of the local blocks
competes with the local model's, and parts the data reject show up as amplitudes far from 1.

`RegionalModel.solve` fits one to station velocities (Meade & Loveless 2009): block rotations, the segments' slip rates
that follow from them, and slip on triangulated faults, by weighted least squares with slip-rate constraints, poles
held a priori, the umbrella smoothing of each mesh (Desbrun et al. 1999) and zero-slip mesh edges. Its estimator is
celeri's dense one, and `celeri_inputs` reads celeri's input files. Block strain, volume sources, mesh edges tied to the
blocks' motion and coupling bounds are not built.

A regional model's velocity at a point is its block's rotation less the elastic field of every locked segment and mesh.
Segments are rectangles from the surface to their locking depth (Okada 1985, `slipsolve._okada`), each in an oblique
Mercator projection along it; mesh triangles are triangular dislocations (`slipsolve._tde`) in a transverse Mercator
projection at each one's centroid. The blocks are the faces of the segment network, each named by the interior point
the model gives it. The tables follow celeri's conventions (github.com/brendanjmeade/celeri): velocities in mm/yr,
strike-slip positive left-lateral, dip-slip positive thrust (its sign flipped past 90 degrees of dip), tensile positive
opening. Everything this module returns is in m/yr. Rotations and positions use the package's Earth radius,
`slipsolve.interseismic.EARTH_RADIUS` (6371 km); celeri's is 6367.4 km, the mean of WGS84's semi-axes, so pass that as
`radius` to reproduce its numbers exactly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as _Polygon
from pyproj import CRS, Geod, Proj

from . import _okada, _tde
from .interseismic import EARTH_RADIUS, SPIN_LENGTH, EulerPole, _axes_at, rotation_columns
from .patterns import Pattern
from .sampling import frame_looks

__all__ = ["RegionalModel", "celeri_inputs", "fit_frame_rotation", "mesh_edges", "west_first"]

#: A segment longer than this (km) closes the model's blocks far away; `far_field` leaves its field out.
CLOSURE_KM = 1000.0
#: A segment is on one of the local faults when both its ends lie within this (m) of that fault's trace and it runs
#: within `ON_FAULT_ANGLE` (degrees) of the trace there.
ON_FAULT_DISTANCE = 1.5e3
ON_FAULT_ANGLE = 20.0
_MM = 1e-3
_PARTS = ("rotation", "inside", "inside_tensile", "outside", "outside_tensile", "meshes")


def _lon180(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def _unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _east_north(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    return east, north


def _faces(lon1, lat1, lon2, lat2):
    """The faces of the planar network the segments draw in `(lon, lat)`: `(vertices, [(loop, area)], face_of)`.

    Each face is the loop of vertex indices that keeps it on the left (a counter-clockwise loop has a positive signed
    area); the network's outside comes out negative. `face_of[2 i]` is the face on the left of segment `i` run from
    endpoint 1 to endpoint 2, `face_of[2 i + 1]` the one on its right.
    """
    n = lon1.size
    ends = np.round(np.column_stack([np.r_[lon1, lon2], np.r_[lat1, lat2]]), 7)
    vertices, index = np.unique(ends, axis=0, return_inverse=True)
    index = np.asarray(index).ravel()
    tail, head = np.empty(2 * n, dtype=int), np.empty(2 * n, dtype=int)
    tail[0::2], head[0::2] = index[:n], index[n:]
    tail[1::2], head[1::2] = index[n:], index[:n]
    step = vertices[head] - vertices[tail]
    angle = np.arctan2(step[:, 1], step[:, 0])
    outgoing = {}
    for h in np.argsort(angle, kind="stable"):
        outgoing.setdefault(int(tail[h]), []).append(int(h))
    position = {h: p for edges in outgoing.values() for p, h in enumerate(edges)}
    following = np.empty(2 * n, dtype=int)
    for h in range(2 * n):
        around = outgoing[int(head[h])]
        following[h] = around[(position[h ^ 1] - 1) % len(around)]       # the edge just clockwise of the way back
    seen = np.zeros(2 * n, dtype=bool)
    face_of = np.full(2 * n, -1)
    faces = []
    for start in range(2 * n):
        if seen[start]:
            continue
        loop, h = [], start
        while not seen[h]:
            seen[h] = True
            face_of[h] = len(faces)
            loop.append(int(tail[h]))
            h = following[h]
        xy = vertices[loop]
        area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
        faces.append((np.array(loop), area))
    return vertices, faces, face_of


def _segment_fields(s, lon, lat, poisson):
    """`(n, 3, 3)`: east, north, up (axis 1) at points of unit strike-, dip- and tensile slip (axis 2) on segment `s`.

    A rectangle from the surface to the locking depth (Okada 1985), in an oblique Mercator projection along the segment:
    strike from endpoint 1 to endpoint 2, the dip to the right of it below 90 degrees, positive dip-slip reverse on
    either side (its sign flipped past 90 degrees, so a vertical segment carries none), back to east and north by the
    segment's geodesic azimuth.
    """
    out = np.zeros((np.size(lon), 3, 3))
    depth = abs(float(s["locking_depth"])) * 1e3
    if depth == 0.0:
        return out
    lon1, lat1, lon2, lat2 = float(_lon180(s["lon1"])), float(s["lat1"]), float(_lon180(s["lon2"])), float(s["lat2"])
    lat2_proj = lat2 + 0.001 if np.isclose(lat1, lat2) else lat2        # the projection cannot take equal latitudes
    projection = Proj(CRS.from_proj4(f"+proj=omerc +lon_1={lon1} +lat_1={lat1} +lon_2={lon2} "
                                     f"+lat_2={lat2_proj} +ellps=WGS84"))
    x, y = projection(lon, lat)
    x1, y1 = projection(lon1, lat1)
    x2, y2 = projection(lon2, lat2)
    strike = np.arctan2(y2 - y1, x2 - x1)
    length = float(np.hypot(x2 - x1, y2 - y1))
    dip = float(s["dip"])
    width = depth / np.sin(np.radians(dip))
    dx, dy = np.asarray(x) - 0.5 * (x1 + x2), np.asarray(y) - 0.5 * (y1 + y2)
    along = np.cos(strike) * dx + np.sin(strike) * dy + 0.5 * length
    across = -np.sin(strike) * dx + np.cos(strike) * dy + np.cos(np.radians(dip)) * width
    turn = np.radians(90.0 - float(s["azimuth"]))
    for k, unit in enumerate(np.eye(3)):
        ux, uy, uz = _okada.okada_disp_surface(along, across, length, width, depth, dip, unit[0],
                                               unit[1] * np.sign(90.0 - dip), unit[2], poisson)
        out[:, 0, k] = np.cos(turn) * ux - np.sin(turn) * uy
        out[:, 1, k] = np.sin(turn) * ux + np.cos(turn) * uy
        out[:, 2, k] = uz
    return out


def _triangle_fields(corners, lon, lat, poisson):
    """`(n, 3, 2)`: east, north, up at points of unit strike- and dip-slip on a triangle (`corners`: lon, lat, depth in
    km, negative down), in a transverse Mercator projection at its centroid."""
    lons, lats, deps = _lon180(corners[:, 0]), corners[:, 1], corners[:, 2]
    centre = _unit(lons, lats) * (EARTH_RADIUS + 1e3 * deps)[:, None]
    centre = centre.mean(axis=0)
    lon0 = np.degrees(np.arctan2(centre[1], centre[0]))
    lat0 = np.degrees(np.arctan2(centre[2], np.hypot(centre[0], centre[1])))
    projection = Proj(CRS.from_proj4(f"+proj=tmerc +lon_0={lon0} +lat_0={lat0} +ellps=WGS84"))
    x, y = projection(lon, lat)
    px, py = projection(lons, lats)
    vertices = [np.array([px[k], py[k], 1e3 * deps[k]]) for k in range(3)]
    unit = _tde.tde_disp_hs_unit(np.asarray(x), np.asarray(y), np.zeros(np.size(lon)), *vertices, nu=poisson)
    return np.transpose(unit[:2], (2, 1, 0))


def fit_frame_rotation(lon, lat, east, north, east_model, north_model, clip=3.0, rounds=5, floor=3e-4):
    """`(EulerPole, kept)`: the rotation taking velocities into a regional model's frame, `v_model = v + omega x r`.

    Fitted at shared stations, velocities in m/yr, each round dropping stations more than `clip` robust standard
    deviations off; the robust deviation is taken as at least `floor` (m/yr), so a close fit drops nothing. `kept`
    marks the stations of the last fit.
    """
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    position = EARTH_RADIUS * _unit(lon, lat)
    e, nn = _east_north(lon, lat)
    columns = []
    for axis in np.eye(3):
        v = np.cross(axis, position)
        columns.append(np.r_[np.sum(v * e, axis=1), np.sum(v * nn, axis=1)])
    design = np.column_stack(columns)
    target = np.r_[np.asarray(east_model, dtype=float) - east, np.asarray(north_model, dtype=float) - north]
    n = lon.size
    kept = np.ones(n, dtype=bool)
    for _ in range(int(rounds)):
        rows = np.r_[kept, kept]
        omega, *_ = np.linalg.lstsq(design[rows], target[rows], rcond=None)
        residual = target - design @ omega
        size = np.hypot(residual[:n], residual[n:])
        scale = max(1.4826 * float(np.median(size[kept])), float(floor))
        kept = size < clip * scale
    return EulerPole(omega), kept


class RegionalModel:
    """A solved block model of a wider region: blocks with their rotations, and segments locked to a depth.

    Attributes:
        segments: one row per segment: `lon1, lat1, lon2, lat2` (degrees, the western end first), `dip` (degrees),
            `locking_depth` (km), `azimuth` (degrees), `length_km`, and the slip rates `strike_slip`, `dip_slip` and
            `tensile` (mm/yr).
        blocks: one row per block: `name`, `interior_lon`, `interior_lat`, and the rotation as `euler_lon`,
            `euler_lat` (degrees) and `euler_rate` (degrees/Myr).
        meshes: one row per triangle: `lon1, lat1, dep1, lon2, lat2, dep2, lon3, lat3, dep3` (depths in km, negative
            below the surface) and `strike_slip`, `dip_slip` (mm/yr); or `None`.
        stations: the model's own stations (`lon`, `lat`, `east`, `north`, mm/yr, in its frame), or `None`.
        poisson: the half-space's Poisson's ratio.
        radius: the Earth radius (m) its rotation velocities use, `EARTH_RADIUS` by default.
    """

    def __init__(self, segments, blocks, meshes=None, stations=None, poisson=0.25, radius=EARTH_RADIUS):
        self.segments = pd.DataFrame(segments).reset_index(drop=True)
        self.blocks = pd.DataFrame(blocks).reset_index(drop=True)
        self.meshes = None if meshes is None else pd.DataFrame(meshes).reset_index(drop=True)
        self.stations = None if stations is None else pd.DataFrame(stations).reset_index(drop=True)
        self.poisson = float(poisson)
        self.radius = float(radius)
        self._polygons = None

    @classmethod
    def solve(cls, segments, blocks, stations, meshes=(), *, poisson=0.25, radius=EARTH_RADIUS, vertical=False):
        """A block model fitted to station velocities (Meade & Loveless 2009): its rotations, slip rates and mesh slip.

    The tables are in a block model's units (degrees, km, mm/yr), as celeri writes them; `celeri_inputs` reads celeri's
    input files into these arguments.

    - `segments`: `lon1`, `lat1`, `lon2`, `lat2`, `dip` (degrees, to the right of the segment run from its western end,
      `west_first`), `locking_depth` (km; 0 gives no elastic field, as where a mesh takes the segment's place). Optional
      slip-rate constraints `ss_rate`, `ds_rate`, `ts_rate` with `ss_rate_sigma`, ... (NaN where there is none).
    - `blocks`: `name`, `interior_lon`, `interior_lat`; blocks with `apriori` true are held at `euler_lon`, `euler_lat`,
      `euler_rate` (degrees/Myr).
    - `stations`: `lon`, `lat`, `east`, `north` and `east_sigma`, `north_sigma` (and `up`, `up_sigma` with `vertical`).
    - `meshes`: dicts with `points` (`(n, 3)`: lon, lat, depth in km, negative down), `triangles` (`(m, 3)` indices),
      `smoothing` (the weight on `(L s)^2`, `s` in mm/yr and `L` the umbrella operator on element centroids in 1/m^2,
      celeri's `smoothing_weight`) and `edges` (`{"top" | "bottom" | "side": sigma in mm/yr}`, holding those edges'
      slip at zero; `mesh_edges` finds them).

    The velocity at a station is its block's rotation less the elastic field of every segment, locked from the surface
    to its depth and slipping as its two blocks' rotations ask, less that of the meshes' slip. Weighted least squares
    over the stations, the slip-rate constraints, the meshes' smoothing and their edges. Returns a `RegionalModel` with
    the estimates filled in (and `_sigma` columns for the slip rates), `stations` with `model_*` columns, `covariance`
    and `parameters` (rotations in rad/yr, three per block, then each mesh's strike and dip slip in m/yr per element)
    and `rms` (mm/yr).
        """
        return _solve_block_model(cls, segments, blocks, stations, meshes, poisson=poisson, radius=radius,
                                  vertical=vertical)

    # -- blocks -----------------------------------------------------------------------------------------------------

    def rotation(self, block):
        """Block `block`'s rotation, `EulerPole` (rad/yr), in the model's frame."""
        row = self.blocks.iloc[int(block)]
        return EulerPole.from_pole(float(row["euler_lon"]), float(row["euler_lat"]), float(row["euler_rate"]))

    def rotation_velocity(self, block, lon, lat):
        """`(n, 3)` east, north, up (m/yr) of block `block`'s rotation at points, on the model's own sphere."""
        omega = self.rotation(block).omega
        position = self.radius * _unit(lon, lat)
        v = np.cross(omega, position)
        e, n = _east_north(lon, lat)
        return np.column_stack([np.sum(v * e, axis=1), np.sum(v * n, axis=1), np.zeros(len(position))])

    def _block_polygons(self):
        if self._polygons is None:
            s = self.segments
            vertices, faces, face_of = _faces(s["lon1"].values % 360.0, s["lat1"].values, s["lon2"].values % 360.0,
                                              s["lat2"].values)
            keep = [i for i, (_, area) in enumerate(faces) if area > 0]
            bounded = [(_Polygon(vertices[faces[i][0]]), faces[i][1]) for i in keep]
            owner = np.full(len(bounded), -1)
            interior = np.column_stack([self.blocks["interior_lon"].values % 360.0, self.blocks["interior_lat"].values])
            outside = []
            for b, point in enumerate(interior):
                inside = [i for i, (poly, area) in enumerate(bounded) if poly.contains_point(point)]
                if inside:
                    owner[min(inside, key=lambda i: bounded[i][1])] = b
                else:
                    outside.append(b)
            # A block whose interior point lies in no face is the network's outside (the rest of the world).
            exterior = outside[0] if len(outside) == 1 else -1
            block_of_face = np.full(len(faces), exterior)
            block_of_face[keep] = owner
            self._sides = block_of_face[face_of].reshape(-1, 2)
            self._polygons = (bounded, owner, exterior)
        return self._polygons

    def segment_blocks(self):
        """`(n_segments, 2)`: the block on the left and on the right of each segment, from endpoint 1 to endpoint 2."""
        self._block_polygons()
        return self._sides

    def block_of(self, lon, lat):
        """`(n,)`: the block each point lies in, the smallest face around it.

        A point outside every face lies in the block whose interior point does too, if exactly one does; otherwise
        it is -1.
        """
        bounded, owner, exterior = self._block_polygons()
        points = np.column_stack([np.asarray(lon, dtype=float) % 360.0, np.asarray(lat, dtype=float)])
        best = np.full(len(points), np.inf)
        out = np.full(len(points), exterior)
        for (poly, area), b in zip(bounded, owner):
            hit = poly.contains_points(points) & (area < best)
            out[hit], best[hit] = b, area
        return out

    def ties(self, lon, lat, local_blocks):
        """`{local block: regional block}`: the regional block holding most of each local block's points."""
        regional = self.block_of(lon, lat)
        local_blocks = np.asarray(local_blocks, dtype=str)
        out = {}
        for name in sorted(set(local_blocks.tolist())):
            counts = np.bincount(regional[(local_blocks == name) & (regional >= 0)], minlength=len(self.blocks))
            if counts.sum() == 0:
                raise ValueError(f"no point of local block {name!r} lies in a regional block")
            out[name] = int(np.argmax(counts))
        return out

    # -- elastic fields ---------------------------------------------------------------------------------------------

    def segment_velocity(self, lon, lat, which=None, components=("strike_slip", "dip_slip", "tensile")):
        """`(n, 3)` east, north, up (m/yr): the elastic field of the segments `which` (a mask or indices; all by
        default) slipping at their rates, the slip components named in `components`."""
        lon, lat = _lon180(lon), np.asarray(lat, dtype=float)
        if which is None:
            rows = np.arange(len(self.segments))
        elif np.asarray(which).dtype == bool:
            rows = np.flatnonzero(which)
        else:
            rows = np.asarray(which, dtype=int)
        total = np.zeros((lon.size, 3))
        for i in rows:
            s = self.segments.iloc[int(i)]
            slip = np.array([float(s[c]) if c in components else 0.0 for c in ("strike_slip", "dip_slip", "tensile")])
            if not slip.any():
                continue
            total += _segment_fields(s, lon, lat, self.poisson) @ slip
        return _MM * total

    def mesh_velocity(self, lon, lat):
        """`(n, 3)` east, north, up (m/yr): the elastic field of the model's mesh triangles slipping at their rates."""
        lon, lat = _lon180(lon), np.asarray(lat, dtype=float)
        total = np.zeros((lon.size, 3))
        if self.meshes is None:
            return total
        for _, t in self.meshes.iterrows():
            corners = np.array([[t[f"lon{k}"], t[f"lat{k}"], t[f"dep{k}"]] for k in (1, 2, 3)], dtype=float)
            total += _triangle_fields(corners, lon, lat, self.poisson) @ np.array([t["strike_slip"], t["dip_slip"]], dtype=float)
        return _MM * total

    def velocity(self, lon, lat):
        """`(n, 3)` east, north, up (m/yr): the model's whole velocity at points, in its frame.

        Each point's block's rotation, less every segment's and mesh triangle's elastic field.
        """
        block = self.block_of(lon, lat)
        out = np.zeros((len(block), 3))
        for b in np.unique(block[block >= 0]):
            on = block == b
            out[on] = self.rotation_velocity(b, np.asarray(lon)[on], np.asarray(lat)[on])
        out[block < 0] = np.nan
        return out - self.segment_velocity(lon, lat) - self.mesh_velocity(lon, lat)

    # -- segment classes --------------------------------------------------------------------------------------------

    def on_faults(self, traces, distance=ON_FAULT_DISTANCE, angle=ON_FAULT_ANGLE):
        """`(n_segments,)`: the segments on one of `traces` (`slipsolve.trace.FaultTrace`s, the local faults)."""
        s = self.segments
        out = np.zeros(len(s), dtype=bool)
        if not traces:
            return out
        frame = traces[0].local_frame()
        lon1, lon2 = _lon180(s["lon1"]), _lon180(s["lon2"])
        # Only segments near a trace vertex are projected: the local frame is no projection far from its centre.
        vertices = np.concatenate([_unit(_lon180(t.lon), t.lat) for t in traces])
        reach = max(float(np.max(np.hypot(np.diff(t.to_local(frame)[0]), np.diff(t.to_local(frame)[1]))))
                    for t in traces) + distance
        close = np.ones(len(s), dtype=bool)
        for lon, lat in ((lon1, s["lat1"].values), (lon2, s["lat2"].values)):
            chord = np.min(np.linalg.norm(_unit(lon, lat)[:, None, :] - vertices[None, :, :], axis=2), axis=1)
            close &= 2.0 * EARTH_RADIUS * np.arcsin(np.minimum(chord / 2.0, 1.0)) < reach
        if not close.any():
            return out
        rows = np.flatnonzero(close)
        x1, y1 = frame.to_local(lon1[rows], s["lat1"].values[rows])
        x2, y2 = frame.to_local(lon2[rows], s["lat2"].values[rows])
        heading = np.degrees(np.arctan2(x2 - x1, y2 - y1))
        for trace in traces:
            tx, ty = trace.to_local(frame)
            near = []
            for px, py in ((x1, y1), (x2, y2)):
                ax, ay, bx, by = tx[:-1], ty[:-1], tx[1:], ty[1:]
                dx, dy = bx - ax, by - ay
                t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / (dx ** 2 + dy ** 2), 0.0, 1.0)
                d = np.hypot(px[:, None] - (ax + t * dx), py[:, None] - (ay + t * dy))
                k = np.argmin(d, axis=1)
                near.append((d[np.arange(px.size), k], np.degrees(np.arctan2(dx[k], dy[k]))))
            turn = np.abs((heading - near[0][1] + 90.0) % 180.0 - 90.0)
            out[rows] |= (near[0][0] < distance) & (near[1][0] < distance) & (turn < angle)
        return out

    def inside(self, polygons):
        """`(n_segments,)`: the segments whose midpoint lies inside one of `polygons` (`{name: (lons, lats)}`)."""
        s = self.segments
        a, b = np.radians(s["lon1"].values), np.radians(s["lon2"].values)
        mid = np.column_stack([np.degrees(np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))),
                               0.5 * (s["lat1"].values + s["lat2"].values)])
        out = np.zeros(len(s), dtype=bool)
        for lons, lats in polygons.values():
            out |= _Polygon(np.column_stack([_lon180(lons), np.asarray(lats, dtype=float)])).contains_points(mid)
        return out

    # -- what the local inversion takes -----------------------------------------------------------------------------

    def rotation_priors(self, obs, blocks, ties, frame_rotation=None, sigma=1e-3, spin_sigma=1e-2, strain=()):
        """`nuisance_prior` entries tying each local block's rotation to its regional block's: `{label: (mean, sigma)}`.

        `obs` and `blocks` are what the local `InterseismicInversion` is given (its block centres come from them).
        `frame_rotation` is the `EulerPole` taking the local data into the model's frame (`fit_frame_rotation`); the
        tied rotation is the regional one less it, scaled to give the regional model's velocities on the local model's
        sphere. `sigma` is on the velocity at a block's centre (m/yr), `spin_sigma` on its spin (the velocity it gives
        `SPIN_LENGTH` from the centre: the default is the same 1 mm/yr at a tenth of that).
        """
        _, _, centres = rotation_columns(obs, blocks, strain=strain)
        shift = np.zeros(3) if frame_rotation is None else frame_rotation.omega
        out = {}
        for name, block in ties.items():
            omega = self.rotation(block).omega * (self.radius / EARTH_RADIUS) - shift
            for component, axis, width in zip(("ve", "vn", "spin"), _axes_at(centres[name]), (sigma, sigma, spin_sigma)):
                out[f"block:{name}:{component}"] = (float(omega @ axis / (axis @ axis)), float(width))
        return out

    def far_field(self, obs, blocks, ties, traces, polygons=None, parts=_PARTS, closure_km=CLOSURE_KM,
                  prefix="regional"):
        """`[Pattern]`: what the regional model has and the local one does not, at the rows of `obs`, one per part.

        - `blocks` is the local block of each row; `ties` maps each to its regional block (`ties`).
        - `traces` are the local faults' `FaultTrace`s: segments on them (`on_faults`) are the local model's own.
        - `polygons` (`{name: (lons, lats)}`, the local blocks) split the other segments into `inside` and `outside`
          parts; without them every segment is `outside`.
        - Segments longer than `closure_km` close the model's blocks far away and are left out.
        - The patterns are named `"{prefix}:{part}"`; parts with no field are skipped.
        """
        unknown = set(parts) - set(_PARTS)
        if unknown:
            raise ValueError(f"unknown parts {sorted(unknown)}; they are {_PARTS}")
        frame = obs.frame
        x, y = obs.ds["x"].values.astype(float), obs.ds["y"].values.astype(float)
        lon, lat = frame.to_lonlat(x, y)
        blocks = np.asarray(blocks, dtype=str)
        fields = {}
        long_segments = self.segments["length_km"].values > float(closure_km)
        local = self.on_faults(traces)
        within = self.inside(polygons) if polygons else np.zeros(len(self.segments), dtype=bool)
        other = ~local & ~long_segments
        if "rotation" in parts:
            regional = self.block_of(lon, lat)
            field = np.zeros((lon.size, 3))
            for name, tied in ties.items():
                rows = blocks == name
                for b in np.unique(regional[rows]):
                    if b < 0 or b == tied:
                        continue
                    on = rows & (regional == b)
                    field[on] = self.rotation_velocity(b, lon[on], lat[on]) - self.rotation_velocity(tied, lon[on], lat[on])
            fields["rotation"] = field
        shear = ("strike_slip", "dip_slip")
        for part, mask, components in (("inside", other & within, shear), ("inside_tensile", other & within, ("tensile",)),
                                       ("outside", other & ~within, shear), ("outside_tensile", other & ~within, ("tensile",))):
            if part in parts:
                fields[part] = -self.segment_velocity(lon, lat, mask, components)
        if "meshes" in parts:
            fields["meshes"] = -self.mesh_velocity(lon, lat)
        out = []
        for part, field in fields.items():
            if not np.any(field):
                continue
            east, north, up = frame_looks(frame, x, y, field[:, 0], field[:, 1], field[:, 2])
            out.append(_lookup_pattern(f"{prefix}:{part}", frame, x, y, east, north, up))
        return out


def _lookup_pattern(name, frame, x, y, east, north, up):
    """A `Pattern` holding a field at given points, read back by position (the inversion's own rows)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([x, y]))

    def enu(qx, qy):
        distance, k = tree.query(np.column_stack([np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)]))
        if distance.max() > 1.0:
            raise ValueError(f"a row lies {distance.max():.0f} m from every point the pattern was made at")
        return east[k], north[k], up[k]

    pattern = Pattern.from_enu(name, enu)
    pattern.frame = frame
    return pattern


# -- solving a block model -----------------------------------------------------------------------------------------

_GEOD = Geod(ellps="WGS84")
_SLIP = ("ss", "ds", "ts")


def west_first(segments):
    """A copy of `segments` with each one's western endpoint first (by the sign of `cross(p1, p2)`), dips unchanged.

    A segment's dip is read to the right of it from its western end, so a table must be ordered this way before its
    dips mean anything.
    """
    s = pd.DataFrame(segments).reset_index(drop=True).copy()
    swap = np.cross(_unit(s["lon1"], s["lat1"]), _unit(s["lon2"], s["lat2"]))[:, 2] < 0
    for a, b in (("lon1", "lon2"), ("lat1", "lat2")):
        s.loc[swap, [a, b]] = s.loc[swap, [b, a]].values
    return s


def _geodesics(s):
    """Forward azimuth at endpoint 1 (degrees), length (km) and geodesic midpoint of each segment, on WGS84."""
    azimuth, _, length = _GEOD.inv(s["lon1"].values, s["lat1"].values, s["lon2"].values, s["lat2"].values)
    middle = np.array([_GEOD.npts(a, b, c, d, 1)[0] for a, b, c, d in
                       zip(s["lon1"].values, s["lat1"].values, s["lon2"].values, s["lat2"].values)])
    return azimuth, length / 1e3, middle[:, 0] % 360.0, middle[:, 1]


def _rotation_rows(lon, lat, radius):
    """`(n, 3, 3)`: east, north, up velocity (axis 1) of a rotation of 1 rad/yr about each axis (axis 2), at points."""
    position = radius * _unit(lon, lat)
    e, n = _east_north(lon, lat)
    out = np.zeros((position.shape[0], 3, 3))
    for k, axis in enumerate(np.eye(3)):
        v = np.cross(axis, position)
        out[:, 0, k] = np.sum(v * e, axis=1)
        out[:, 1, k] = np.sum(v * n, axis=1)
    return out


def _kinematics(s, sides, n_blocks, radius):
    """`(n_segments, 3, 3 n_blocks)`: strike-, dip- and tensile-slip rates (m/yr) of each segment from the rotations.

    The relative velocity of the block on the right of the segment (from endpoint 1 to 2) against the one on its left,
    at the geodesic midpoint: its component along the segment is the strike slip (positive left-lateral); across it,
    convergence is dip slip divided by `|cos(dip)|` (positive reverse), or minus the tensile slip on a vertical segment
    (positive opening). Meade & Loveless (2009), in celeri's conventions.
    """
    azimuth, _, mid_lon, mid_lat = _geodesics(s)
    jacobian = _rotation_rows(mid_lon, mid_lat, radius)[:, :2, :]
    az = np.radians(azimuth)
    along = np.column_stack([np.sin(az), np.cos(az)])
    left_normal = np.column_stack([-np.cos(az), np.sin(az)])
    out = np.zeros((len(s), 3, 3 * n_blocks))
    dip = s["dip"].values.astype(float)
    vertical = np.isclose(dip, 90.0, rtol=0.0, atol=1e-9)
    for i, (left, right) in enumerate(sides):
        strike = along[i] @ jacobian[i]
        convergence = left_normal[i] @ jacobian[i]
        for block, sign in ((right, 1.0), (left, -1.0)):
            columns = slice(3 * block, 3 * block + 3)
            out[i, 0, columns] += sign * strike
            if vertical[i]:
                out[i, 2, columns] -= sign * convergence
            else:
                out[i, 1, columns] += sign * convergence / abs(np.cos(np.radians(dip[i])))
    return out


def _mesh_geometry(points, triangles, radius):
    """Upward-wound triangles (celeri's rule: an element whose normal points down has its last two corners swapped),
    their corners `(n, 3, 3)` (lon, lat, depth km) and Cartesian centroids (m)."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int).copy()
    corners = points[triangles]
    xyz = _unit(corners[..., 0], corners[..., 1]) * (radius + 1e3 * corners[..., 2])[..., None]
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    centroid = xyz.mean(axis=1)
    down = np.sum(normal * centroid, axis=1) < 0
    triangles[down] = triangles[down][:, [0, 2, 1]]
    return triangles, points[triangles], centroid


def _mesh_neighbours(triangles):
    """`(n, 3)`: the elements sharing each side of each element (-1 where none), and the boundary sides as
    `(element, side)` pairs, side k running from corner k to corner k + 1."""
    sides = np.sort(np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1), axis=2)
    flat = sides.reshape(-1, 2)
    _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    neighbours = np.full(flat.shape[0], -1)
    order = np.argsort(inverse, kind="stable")
    for a, b in zip(order[:-1], order[1:]):
        if inverse[a] == inverse[b]:
            neighbours[a], neighbours[b] = b // 3, a // 3
    boundary = np.flatnonzero(counts[inverse] == 1)
    return neighbours.reshape(-1, 3), np.column_stack([boundary // 3, boundary % 3])


def _umbrella(triangles, centroid):
    """The scale-dependent umbrella operator on element centroids (Desbrun et al. 1999), `(n, n)` in 1/m^2."""
    from scipy import sparse

    neighbours, _ = _mesh_neighbours(triangles)
    n = len(triangles)
    has = neighbours >= 0
    distance = np.where(has, np.linalg.norm(centroid[:, None, :] - centroid[np.where(has, neighbours, 0)], axis=2), np.nan)
    total = np.nansum(distance, axis=1)
    lead = np.where(total > 0, 2.0 / np.where(total > 0, total, 1.0), 0.0)
    rows = np.r_[np.arange(n), np.nonzero(has)[0]]
    columns = np.r_[np.arange(n), neighbours[has]]
    values = np.r_[-lead * np.nansum(1.0 / distance, axis=1), (lead[:, None] / distance)[has]]
    return sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()


def mesh_edges(points, triangles):
    """`{"top", "bottom", "side"}`: boolean masks of a mesh's edge elements.

    An element with a side on the mesh's boundary is on its top if its third corner is deeper than that side by more
    than the side's own change in depth, on its bottom if shallower by more than that, and on a side otherwise.
    """
    triangles = np.asarray(triangles, dtype=int)
    depth = np.abs(np.asarray(points, dtype=float)[:, 2])
    _, boundary = _mesh_neighbours(triangles)
    out = {key: np.zeros(len(triangles), dtype=bool) for key in ("top", "bottom", "side")}
    for element, side in boundary:
        a, b, c = triangles[element, side], triangles[element, (side + 1) % 3], triangles[element, (side + 2) % 3]
        rise, spread = depth[c] - 0.5 * (depth[a] + depth[b]), abs(depth[a] - depth[b])
        out["side"][element] = True
        if rise > spread:
            out["top"][element] = True
        elif rise < -spread:
            out["bottom"][element] = True
    out["side"] &= ~(out["top"] | out["bottom"])
    return out


def _pole(omega):
    """`(lon, lat, rate)`: an Euler pole (degrees, degrees/Myr) from a rotation vector (rad/yr)."""
    rate = float(np.linalg.norm(omega))
    if rate == 0.0:
        return 0.0, 90.0, 0.0
    return (float(np.degrees(np.arctan2(omega[1], omega[0]))), float(np.degrees(np.arcsin(omega[2] / rate))),
            float(np.degrees(rate) * 1e6))


def _solve_block_model(cls, segments, blocks, stations, meshes=(), *, poisson=0.25, radius=EARTH_RADIUS,
                       vertical=False):
    from scipy import sparse
    from scipy.linalg import lstsq

    s = west_first(segments)
    azimuth, length, _, _ = _geodesics(s)
    s["azimuth"], s["length_km"] = azimuth, length
    blocks = pd.DataFrame(blocks).reset_index(drop=True)
    stations = pd.DataFrame(stations).reset_index(drop=True)
    model = cls(s, blocks, poisson=poisson, radius=radius)
    nb = len(blocks)
    sides = model.segment_blocks()
    if (sides < 0).any():
        raise ValueError(f"{int((sides < 0).any(axis=1).sum())} segments border a face with no block's interior point")
    K = _kinematics(s, sides, nb, radius)                                    # (n_seg, 3, 3 nb), m/yr per rad/yr
    lon, lat = stations["lon"].values.astype(float), stations["lat"].values.astype(float)
    labels = model.block_of(lon, lat)
    if (labels < 0).any():
        raise ValueError(f"{int((labels < 0).sum())} stations lie in no block")
    components = 3 if vertical else 2
    n_st = len(stations)
    rotation = np.zeros((n_st, 3, 3 * nb))
    rotation_rows = _rotation_rows(lon, lat, radius)
    for i, b in enumerate(labels):
        rotation[i, :, 3 * b:3 * b + 3] = rotation_rows[i]
    back_slip = np.zeros((n_st * 3, 3 * nb))
    for j in range(len(s)):
        if float(s["locking_depth"].iloc[j]) != 0.0:
            back_slip += _segment_fields(s.iloc[j], lon, lat, poisson).reshape(-1, 3) @ K[j]
    block_columns = (rotation.reshape(n_st * 3, -1) - back_slip).reshape(n_st, 3, -1)[:, :components]
    # Mesh slip (strike and dip per element, m/yr): its elastic field subtracts from the stations' velocities.
    mesh_parts, mesh_columns, smoothing_rows, edge_rows = [], [], [], []
    offset = 3 * nb
    for m, mesh in enumerate(meshes):
        triangles, corners, centroid = _mesh_geometry(mesh["points"], mesh["triangles"], radius)
        n_tri = len(triangles)
        columns = np.zeros((n_st, 3, 2 * n_tri))
        for k in range(n_tri):
            columns[:, :, 2 * k:2 * k + 2] = _triangle_fields(corners[k], lon, lat, poisson)
        mesh_columns.append(-columns[:, :components])
        laplacian = _umbrella(triangles, centroid)
        weight = np.sqrt(float(mesh.get("smoothing", 0.0))) * 1e3          # celeri's weight is on mm/yr
        if weight > 0:
            smoothing_rows.append((offset, weight * sparse.kron(laplacian, sparse.eye(2)).toarray()))
        edges = mesh_edges(mesh["points"], triangles)
        for side, sigma in (mesh.get("edges") or {}).items():
            if sigma is None:
                continue
            chosen = np.flatnonzero(edges[side])
            rows = np.zeros((2 * chosen.size, 2 * n_tri))
            rows[np.arange(2 * chosen.size), (2 * chosen[:, None] + np.arange(2)).ravel()] = 1.0 / (1e-3 * float(sigma))
            edge_rows.append((offset, rows))
        mesh_parts.append(dict(triangles=triangles, corners=corners, offset=offset, n=n_tri))
        offset += 2 * n_tri
    n_param = offset
    # Station rows, weighted by their sigmas (mm/yr in the table).
    names = ("east", "north", "up")[:components]
    data = np.column_stack([1e-3 * stations[c].values.astype(float) for c in names]).ravel()
    sigma = np.column_stack([1e-3 * stations[f"{c}_sigma"].values.astype(float) for c in names]).ravel()
    design = np.zeros((n_st * components, n_param))
    design[:, :3 * nb] = block_columns.reshape(n_st * components, -1)
    for part, columns in zip(mesh_parts, mesh_columns):
        design[:, part["offset"]:part["offset"] + 2 * part["n"]] = columns.reshape(n_st * components, -1)
    rows, targets = [design / sigma[:, None]], [data / sigma]
    # Slip-rate constraints: a rate and its sigma (mm/yr) on any of a segment's three components.
    for c, name in enumerate(_SLIP):
        rate, width = f"{name}_rate", f"{name}_rate_sigma"
        if rate not in s or width not in s:
            continue
        for j in np.flatnonzero(np.isfinite(s[width].values.astype(float))):
            row = np.zeros(n_param)
            row[:3 * nb] = K[j, c] / (1e-3 * float(s[width].iloc[j]))
            rows.append(row[None])
            targets.append(np.array([float(s[rate].iloc[j]) / float(s[width].iloc[j])]))
    for start, block in smoothing_rows + edge_rows:
        row = np.zeros((block.shape[0], n_param))
        row[:, start:start + block.shape[1]] = block
        rows.append(row)
        targets.append(np.zeros(block.shape[0]))
    A, d = np.vstack(rows), np.concatenate(targets)
    # Blocks with an a priori pole are held there; the rest are solved for.
    fixed = np.zeros(n_param, dtype=bool)
    x = np.zeros(n_param)
    if "apriori" in blocks:
        for b in np.flatnonzero(blocks["apriori"].values.astype(bool)):
            row = blocks.iloc[b]
            x[3 * b:3 * b + 3] = EulerPole.from_pole(float(row["euler_lon"]), float(row["euler_lat"]),
                                                     float(row["euler_rate"])).omega
            fixed[3 * b:3 * b + 3] = True
    free = ~fixed
    d = d - A[:, fixed] @ x[fixed]
    scale = np.linalg.norm(A[:, free], axis=0)
    scale[scale == 0] = 1.0
    solution, _, rank, _ = lstsq(A[:, free] / scale, d, lapack_driver="gelsy")
    if rank < free.sum():
        raise ValueError(f"the block model is singular: rank {rank} of {int(free.sum())} parameters")
    x[free] = solution / scale
    normal = (A[:, free] / scale).T @ (A[:, free] / scale)
    covariance = np.zeros((n_param, n_param))
    covariance[np.ix_(free, free)] = np.linalg.inv(normal) / np.outer(scale, scale)
    # The solved model, in the tables' units (mm/yr, degrees/Myr).
    omega = x[:3 * nb]
    rates = K @ omega
    rate_sigma = np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", K, covariance[:3 * nb, :3 * nb], K), 0.0))
    for c, name in enumerate(("strike_slip", "dip_slip", "tensile")):
        s[name] = 1e3 * rates[:, c]
        s[f"{name}_sigma"] = 1e3 * rate_sigma[:, c]
    poles = [_pole(omega[3 * b:3 * b + 3]) for b in range(nb)]
    blocks = blocks.copy()
    blocks["euler_lon"], blocks["euler_lat"], blocks["euler_rate"] = (np.array([p[k] for p in poles]) for k in range(3))
    tables = []
    for m, part in enumerate(mesh_parts):
        slip = x[part["offset"]:part["offset"] + 2 * part["n"]].reshape(-1, 2)
        corners = part["corners"]
        table = pd.DataFrame({f"{key}{k + 1}": corners[:, k, c] for k in range(3) for c, key in enumerate(("lon", "lat", "dep"))})
        table["mesh"] = m
        table["strike_slip"], table["dip_slip"] = 1e3 * slip[:, 0], 1e3 * slip[:, 1]
        tables.append(table)
    out = cls(s, blocks, meshes=pd.concat(tables, ignore_index=True) if tables else None, poisson=poisson, radius=radius)
    predicted = design @ x
    out.stations = stations.assign(**{f"model_{c}": 1e3 * predicted.reshape(n_st, components)[:, k]
                                      for k, c in enumerate(names)})
    out.covariance, out.parameters = covariance, x
    out.rms = float(1e3 * np.sqrt(np.mean((predicted - data) ** 2)))
    return out


def _read_msh(path):
    """`(points, triangles)` from a gmsh mesh file (ASCII, format 2.2 or 4.1)."""
    lines = Path(path).read_text().split("\n")
    version = float(lines[lines.index("$MeshFormat") + 1].split()[0])
    start, stop = lines.index("$Nodes") + 1, lines.index("$EndNodes")
    nodes = {}
    if version >= 4:
        _, n_nodes, _, _ = map(int, lines[start].split())
        i = start + 1
        while i < stop:
            _, _, parametric, count = map(int, lines[i].split())
            tags = [int(v) for v in lines[i + 1:i + 1 + count]]
            for k, tag in enumerate(tags):
                nodes[tag] = [float(v) for v in lines[i + 1 + count + k].split()[:3]]
            i += 1 + 2 * count
    else:
        for line in lines[start + 1:stop]:
            values = line.split()
            nodes[int(values[0])] = [float(v) for v in values[1:4]]
    start, stop = lines.index("$Elements") + 1, lines.index("$EndElements")
    triangles = []
    if version >= 4:
        i = start + 1
        while i < stop:
            _, _, kind, count = map(int, lines[i].split())
            if kind == 2:
                triangles += [[int(v) for v in line.split()[1:4]] for line in lines[i + 1:i + 1 + count]]
            i += 1 + count
    else:
        for line in lines[start + 1:stop]:
            values = [int(v) for v in line.split()]
            if len(values) > 1 and values[1] == 2:
                triangles.append(values[-3:])
    tags = np.array(sorted(nodes))
    index = {tag: k for k, tag in enumerate(tags)}
    points = np.array([nodes[tag] for tag in tags])
    return points, np.array([[index[v] for v in tri] for tri in triangles])


def celeri_inputs(config):
    """The arguments of `RegionalModel.solve` from a celeri configuration file and the input files it names.

    Applies what celeri applies before its solve: stations with flag 0 dropped; the locking-depth flags and override;
    segments replaced by a mesh (`mesh_flag`) given no elastic field; flagged slip rates and poles as constraints, with
    celeri's weights as sigmas (`slip_constraint_weight`, `tri_con_weight`; its a priori poles are held, as its weight
    of 1e24 does). A mesh edge flagged 2 (slip tied to the blocks' motion) is not supported.

    The solve then uses the package's Earth radius. celeri's is the mean of WGS84's semi-axes, 6367.4 km: add
    `radius=0.5 * (6378137.0 + 6356752.314245)` to reproduce celeri's own solution.
    """
    config = Path(config).expanduser()
    options = json.loads(config.read_text())

    def path(name):
        value = Path(str(options[name])).expanduser()
        return value if value.is_absolute() else config.parent / value

    seg = pd.read_csv(path("segment_file_name"))
    seg["name"] = seg["name"].astype(str).str.strip()
    depth = seg["locking_depth"].astype(float).copy()
    for flag in (2, 3, 4, 5):
        depth[seg["locking_depth_flag"] == flag] = float(options.get(f"locking_depth_flag{flag}", depth.mean()))
    if options.get("locking_depth_override_flag"):
        depth[:] = float(options["locking_depth_override_value"])
    if "mesh_flag" in seg:
        depth[seg["mesh_flag"] != 0] = 0.0
    sigma = 1.0 / np.sqrt(float(options.get("slip_constraint_weight", 1e5)))
    segments = pd.DataFrame({"name": seg["name"], "lon1": seg["lon1"], "lat1": seg["lat1"], "lon2": seg["lon2"],
                             "lat2": seg["lat2"], "dip": seg["dip"], "locking_depth": depth})
    for name in _SLIP:
        flagged = seg.get(f"{name}_rate_flag", pd.Series(0, index=seg.index)).values == 1
        segments[f"{name}_rate"] = np.where(flagged, seg.get(f"{name}_rate", 0.0), np.nan)
        segments[f"{name}_rate_sigma"] = np.where(flagged, sigma, np.nan)
    blk = pd.read_csv(path("block_file_name"))
    blocks = pd.DataFrame({"name": blk["name"].astype(str).str.strip(), "interior_lon": blk["interior_lon"],
                           "interior_lat": blk["interior_lat"], "apriori": blk["rotation_flag"].values == 1,
                           "euler_lon": blk["euler_lon"], "euler_lat": blk["euler_lat"], "euler_rate": blk["rotation_rate"]})
    st = pd.read_csv(path("station_file_name"))
    st = st.loc[st["flag"] != 0].reset_index(drop=True)
    stations = pd.DataFrame({"name": st["name"].astype(str).str.strip(), "lon": st["lon"], "lat": st["lat"],
                             "east": st["east_vel"], "north": st["north_vel"], "east_sigma": st["east_sig"],
                             "north_sigma": st["north_sig"], "up": st.get("up_vel", 0.0), "up_sigma": st.get("up_sig", 1.0)})
    meshes = []
    if options.get("mesh_parameters_file_name"):
        edge_sigma = 1.0 / np.sqrt(float(options.get("tri_con_weight", 1e6)))
        for item in json.loads(path("mesh_parameters_file_name").read_text()):
            file = Path(item["mesh_filename"]).expanduser()
            points, triangles = _read_msh(file if file.is_absolute() else config.parent / file)
            points[:, 0] = np.where(points[:, 0] < 0, points[:, 0] + 360.0, points[:, 0])
            edges = {}
            for side, key in (("top", "top"), ("bottom", "bot"), ("side", "side")):
                flag = int(item.get(f"{key}_slip_rate_constraint", 0))
                if flag == 2:
                    raise NotImplementedError("a mesh edge tied to the blocks' motion (flag 2) is not supported")
                edges[side] = edge_sigma if flag == 1 else None
            meshes.append(dict(points=points, triangles=triangles, smoothing=float(item.get("smoothing_weight", 0.0)),
                               edges=edges))
    lam, mu = float(options.get("material_lambda", 3e10)), float(options.get("material_mu", 3e10))
    return dict(segments=segments, blocks=blocks, stations=stations, meshes=meshes,
                poisson=lam / (2.0 * (lam + mu)), vertical=bool(options.get("include_vertical_velocity", False)))
