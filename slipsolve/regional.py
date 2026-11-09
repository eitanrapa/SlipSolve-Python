"""A regional block model as the far field of a local interseismic model.

A local model (`slipsolve.interseismic.InterseismicInversion`) has its own blocks and faults, a piece of a wider system.
A regional block model of that system describes the rest: how the blocks the local ones sit in rotate, and the faults
the local model leaves out, each locked to a depth and slipping as its two blocks' rotations ask (Meade & Loveless
2009). `RegionalModel` reads a solved one and evaluates it at the local model's observation points, in the two forms
the local inversion takes:

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

A regional model's velocity at a point is its block's rotation less the elastic field of every locked segment and mesh.
Segments are rectangles from the surface to their locking depth (Okada 1985, `slipsolve._okada`), each in an oblique
Mercator projection along it; mesh triangles are triangular dislocations (`slipsolve._tde`) in a transverse Mercator
projection at each one's centroid. The blocks are the faces of the segment network, each named by the interior point
the model gives it. `from_celeri` reads a solved run of celeri (github.com/brendanjmeade/celeri), whose conventions
these follow: velocities in mm/yr, strike-slip positive left-lateral, dip-slip positive thrust (its sign flipped past
90 degrees of dip), tensile positive opening. Everything this module returns is in m/yr.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as _Polygon
from pyproj import CRS, Proj

from . import _okada, _tde
from .interseismic import EARTH_RADIUS, SPIN_LENGTH, EulerPole, _axes_at, rotation_columns
from .patterns import Pattern
from .sampling import frame_looks

__all__ = ["RegionalModel", "fit_frame_rotation"]

#: A segment longer than this (km) closes the model's blocks far away; `far_field` leaves its field out.
CLOSURE_KM = 1000.0
#: A segment is on one of the local faults when both its ends lie within this (m) of that fault's trace and it runs
#: within `ON_FAULT_ANGLE` (degrees) of the trace there.
ON_FAULT_DISTANCE = 1.5e3
ON_FAULT_ANGLE = 20.0
#: celeri's Earth radius, the mean of WGS84's semi-axes (m): its rotation velocities use it.
CELERI_RADIUS = 0.5 * (6378137.0 + 6356752.314245)
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
    """The bounded faces of the planar network the segments draw in `(lon, lat)`: `(vertices, [(loop, area)])`.

    Each face is the loop of vertex indices that keeps it on the left (a counter-clockwise loop has a positive signed
    area); the network's outside comes out negative.
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
    faces = []
    for start in range(2 * n):
        if seen[start]:
            continue
        loop, h = [], start
        while not seen[h]:
            seen[h] = True
            loop.append(int(tail[h]))
            h = following[h]
        xy = vertices[loop]
        area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
        faces.append((np.array(loop), area))
    return vertices, faces


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
        radius: the Earth radius (m) its rotation velocities use.
    """

    def __init__(self, segments, blocks, meshes=None, stations=None, poisson=0.25, radius=CELERI_RADIUS):
        self.segments = pd.DataFrame(segments).reset_index(drop=True)
        self.blocks = pd.DataFrame(blocks).reset_index(drop=True)
        self.meshes = None if meshes is None else pd.DataFrame(meshes).reset_index(drop=True)
        self.stations = None if stations is None else pd.DataFrame(stations).reset_index(drop=True)
        self.poisson = float(poisson)
        self.radius = float(radius)
        self._polygons = None

    @classmethod
    def from_celeri(cls, run):
        """A solved celeri run's output folder: `model_segment.csv`, `model_block.csv`, `model_meshes.csv`,
        `model_station.csv` and `config.json`."""
        run = Path(run).expanduser()
        seg = pd.read_csv(run / "model_segment.csv")
        segments = pd.DataFrame({
            "name": seg["name"].astype(str).str.strip(), "lon1": seg["lon1"], "lat1": seg["lat1"],
            "lon2": seg["lon2"], "lat2": seg["lat2"], "dip": seg["dip"], "locking_depth": seg["locking_depth"],
            "azimuth": seg["azimuth"], "length_km": seg["length"] / 1e3,
            "strike_slip": seg["model_strike_slip_rate"], "dip_slip": seg["model_dip_slip_rate"],
            "tensile": seg["model_tensile_slip_rate"]})
        blk = pd.read_csv(run / "model_block.csv")
        blocks = pd.DataFrame({"name": blk["name"].astype(str).str.strip(), "interior_lon": blk["interior_lon"],
                               "interior_lat": blk["interior_lat"], "euler_lon": blk["euler_lon"],
                               "euler_lat": blk["euler_lat"], "euler_rate": blk["euler_rate"]})
        meshes = None
        if (run / "model_meshes.csv").exists():
            mesh = pd.read_csv(run / "model_meshes.csv")
            if len(mesh):
                meshes = mesh[["lon1", "lat1", "dep1", "lon2", "lat2", "dep2", "lon3", "lat3", "dep3"]].copy()
                meshes["strike_slip"] = mesh["strike_slip_rate"]
                meshes["dip_slip"] = mesh["dip_slip_rate"]
        stations = None
        if (run / "model_station.csv").exists():
            st = pd.read_csv(run / "model_station.csv")
            stations = pd.DataFrame({"name": st["name"].astype(str).str.strip(), "lon": st["lon"], "lat": st["lat"],
                                     "east": st["east_vel"], "north": st["north_vel"]})
        poisson = 0.25
        if (run / "config.json").exists():
            config = json.loads((run / "config.json").read_text())
            lam, mu = float(config.get("material_lambda", 3e10)), float(config.get("material_mu", 3e10))
            poisson = lam / (2.0 * (lam + mu))
        return cls(segments, blocks, meshes, stations, poisson=poisson)

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
            vertices, faces = _faces(s["lon1"].values % 360.0, s["lat1"].values, s["lon2"].values % 360.0,
                                     s["lat2"].values)
            bounded = [(_Polygon(vertices[loop]), area) for loop, area in faces if area > 0]
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
            self._polygons = (bounded, owner, exterior)
        return self._polygons

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
            slip = [float(s[c]) if c in components else 0.0 for c in ("strike_slip", "dip_slip", "tensile")]
            if not any(slip):
                continue
            total += self._one_segment(s, lon, lat, *slip)
        return _MM * total

    def _one_segment(self, s, lon, lat, strike_slip, dip_slip, tensile):
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
        depth = abs(float(s["locking_depth"])) * 1e3
        width = depth / np.sin(np.radians(dip))
        dx, dy = np.asarray(x) - 0.5 * (x1 + x2), np.asarray(y) - 0.5 * (y1 + y2)
        along = np.cos(strike) * dx + np.sin(strike) * dy
        across = -np.sin(strike) * dx + np.cos(strike) * dy + np.cos(np.radians(dip)) * width
        ux, uy, uz = _okada.okada_disp_surface(along + 0.5 * length, across, length, width, depth, dip,
                                               strike_slip, dip_slip * np.sign(90.0 - dip), tensile, self.poisson)
        turn = np.radians(90.0 - float(s["azimuth"]))
        return np.column_stack([np.cos(turn) * ux - np.sin(turn) * uy, np.sin(turn) * ux + np.cos(turn) * uy, uz])

    def mesh_velocity(self, lon, lat):
        """`(n, 3)` east, north, up (m/yr): the elastic field of the model's mesh triangles slipping at their rates."""
        lon, lat = _lon180(lon), np.asarray(lat, dtype=float)
        total = np.zeros((lon.size, 3))
        if self.meshes is None:
            return total
        for _, t in self.meshes.iterrows():
            lons = _lon180([t["lon1"], t["lon2"], t["lon3"]])
            lats = np.array([t["lat1"], t["lat2"], t["lat3"]], dtype=float)
            projection = Proj(CRS.from_proj4(f"+proj=tmerc +lon_0={float(np.mean(lons))} "
                                             f"+lat_0={float(np.mean(lats))} +ellps=WGS84"))
            x, y = projection(lon, lat)
            px, py = projection(lons, lats)
            corners = [np.array([px[k], py[k], 1e3 * float(t[f"dep{k + 1}"])]) for k in range(3)]
            ue, un, uu = _tde.tde_disp_hs(np.asarray(x), np.asarray(y), np.zeros(lon.size), *corners,
                                          float(t["strike_slip"]), float(t["dip_slip"]), 0.0, self.poisson)
            total += np.column_stack([ue, un, uu])
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
