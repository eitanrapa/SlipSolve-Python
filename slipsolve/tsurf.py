"""GOCAD t-surf: triangulated fault surfaces, as the Community Fault Model is distributed.

`read_tsurf` reads a file into one `TSurf` per surface it holds. A surface is
then remeshed rather than used as given, by
`FaultMesh.from_tsurf`: community-model triangles
are ~1 km, far finer than geodesy resolves, have no `(s, z)` lattice for the smoothing, edges
and sheets of a mesh to stand on, and cannot wind against a trace. So the surface is read as a
field `cross = f(s, z)` off a trace along its top edge, the form
`FaultSurface` holds every curved fault in. The price: a surface
that folds over itself in `(s, z)` (a strand that reverses along the trace) has no such
field, and only its average comes through. `mesh.attrs["tsurf_misfit_rms"]` says how close
it came.

```python
sections = [read_tsurf(path, crs="EPSG:26711")[0] for path in paths]    # in order along the fault
surface = TSurf.merged(sections, name="fault")
trace = surface.trace()                         # through their top edges, west to east
mesh = FaultMesh.from_tsurf(surface, trace, frame, edge_length=5e3, max_depth=40e3)
```

**Name the CRS.** A t-surf's header does not: it says only the axis names, units and which
way is up. SCEC's CFM is UTM zone 11, NAD27 (`EPSG:26711`), statewide (CFM 7.0).

A community-model surface stops where seismicity stops, and interseismic back-slip needs the
fault to run on below the locking depth. `FaultMesh.from_tsurf` continues it straight down from
each column's deepest vertices (`below="vertical"`, the default) or at about its last dip
(`below="dip"`), and records the surface's deepest point in `mesh.attrs["tsurf_deepest"]`.
"""

import numpy as np

__all__ = ["TSurf", "read_tsurf"]

_UNIT_SCALE = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "km": 1e3, "kilometer": 1e3,
               "kilometers": 1e3, "ft": 0.3048, "feet": 0.3048}

#: The longest run of the boundary near the top must span this share of the surface's extent.
_MIN_TOP_SHARE = 0.5

#: A top edge turning by more than this (cosine of 120 degrees) at a vertex has doubled back: a spike.
_SPIKE_COSINE = -0.5


class TSurf:
    """One triangulated surface: `vertices` `(n, 3)`, metres, z up; `triangles` `(m, 3)` indices.

    Horizontal coordinates are the file's, in `crs`; only vertices the triangles use are kept.
    """

    def __init__(self, vertices, triangles, name="", crs=None, parts=None):
        vertices = np.asarray(vertices, dtype=float)
        triangles = np.asarray(triangles, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"expected (n, 3) vertices and (m, 3) triangles, got {vertices.shape} and {triangles.shape}")
        used, inverse = np.unique(triangles, return_inverse=True)
        self.vertices = vertices[used]
        self.triangles = inverse.reshape(triangles.shape)
        self.name = str(name)
        self.crs = crs
        self.parts = parts

    @classmethod
    def merged(cls, surfaces, name=""):
        """One fault from the surfaces of its sections, given in order along it.

        Their triangles are not stitched, which nothing needs: `FaultMesh.from_tsurf` fits the
        vertices, and `trace` chains the sections' top edges.
        """
        surfaces = list(surfaces)
        if not surfaces:
            raise ValueError("merged needs at least one surface")
        crs = {surface.crs for surface in surfaces}
        if len(crs) != 1:
            raise ValueError(f"the surfaces are in different coordinate systems: {sorted(map(str, crs))}")
        vertices, triangles, offset = [], [], 0
        for surface in surfaces:
            vertices.append(surface.vertices)
            triangles.append(surface.triangles + offset)
            offset += surface.vertices.shape[0]
        return cls(np.vstack(vertices), np.vstack(triangles), name=name or "+".join(s.name for s in surfaces),
                   crs=surfaces[0].crs, parts=surfaces)

    def lonlat(self):
        """Each vertex's longitude and latitude, degrees, from `crs`."""
        if self.crs is None:
            raise ValueError(f"surface {self.name!r} has no crs: give read_tsurf(path, crs=...); a t-surf's header does not name one")
        from pyproj import Transformer

        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return transformer.transform(self.vertices[:, 0], self.vertices[:, 1])

    def boundary_loops(self):
        """The surface's boundary, as closed loops of vertex indices, longest first."""
        if self.parts:
            raise ValueError(f"{self.name!r} is merged from {len(self.parts)} surfaces: ask each of .parts")
        t = self.triangles
        edges = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        if not boundary.size:
            raise ValueError(f"surface {self.name!r} is closed: it has no boundary edges")
        neighbours = {}
        for a, b in boundary:
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
        loops = []
        while neighbours:
            start = next(iter(neighbours))
            loop = [start]
            current = start
            while True:
                options = neighbours.get(current)
                if not options:
                    break
                following = options.pop()
                neighbours[following].remove(current)
                if not neighbours[following]:
                    del neighbours[following]
                if not options and current in neighbours:
                    del neighbours[current]
                if following == start:
                    break
                loop.append(following)
                current = following
            loops.append(np.array(loop))
        length = [float(np.sum(np.linalg.norm(np.diff(self.vertices[np.r_[loop, loop[0]]], axis=0), axis=1))) for loop in loops]
        return [loops[i] for i in np.argsort(length)[::-1]]

    def top_edge(self, fraction=0.1):
        """The vertices along the surface's top edge, in order along it.

        The longest contiguous run of the longest boundary loop that stays within `fraction` of
        the surface's depth range of its highest point. Raises if that run spans less than half
        the surface's map extent, as a top edge that follows rough topography can: raise
        `fraction`, or give a trace of your own.
        """
        loop = self.boundary_loops()[0]
        z = self.vertices[loop, 2]
        top, bottom = float(self.vertices[:, 2].max()), float(self.vertices[:, 2].min())
        near = z >= top - float(fraction) * (top - bottom)
        if near.all():
            raise ValueError(f"surface {self.name!r}: its whole boundary is within {fraction} of the top; is it horizontal?")
        # Rotate so the loop starts outside a run, which leaves every run contiguous.
        start = int(np.flatnonzero(~near)[0])
        order = np.roll(loop, -start)
        near = np.roll(near, -start)
        xy = self.vertices[order, :2]
        best, best_length, k = None, -1.0, 0
        while k < order.size:
            if not near[k]:
                k += 1
                continue
            end = k
            while end + 1 < order.size and near[end + 1]:
                end += 1
            length = float(np.sum(np.hypot(*np.diff(xy[k:end + 1], axis=0).T)))
            if length > best_length:
                best, best_length = (k, end), length
            k = end + 1
        if best is None:
            raise ValueError(f"surface {self.name!r}: no boundary run near its top")
        extent = np.ptp(self.vertices[:, :2], axis=0)
        if best_length < _MIN_TOP_SHARE * float(np.hypot(*extent)):
            raise ValueError(f"surface {self.name!r}: the top edge found spans {best_length / 1e3:.1f} km of a surface "
                             f"{np.hypot(*extent) / 1e3:.1f} km across; raise fraction= or give FaultMesh.from_tsurf a trace")
        return self._clean_edge(order[best[0]:best[1] + 1])

    def _clean_edge(self, edge):
        """A top edge without the start of the lateral edges at its ends, or the spikes a rough boundary leaves.

        Both would be kinks of 90-175 degrees in the trace every position is then measured from.
        """
        xyz = self.vertices[edge]

        def steep(a, b):
            return abs(xyz[b, 2] - xyz[a, 2]) > np.hypot(*(xyz[b, :2] - xyz[a, :2]))

        first, last = 0, edge.size - 1
        while last - first > 2 and steep(first, first + 1):
            first += 1
        while last - first > 2 and steep(last - 1, last):
            last -= 1
        keep = list(range(first, last + 1))
        changed = True
        while changed and len(keep) > 2:
            changed = False
            p = xyz[keep, :2]
            before, after = p[1:-1] - p[:-2], p[2:] - p[1:-1]
            cosine = np.einsum("ij,ij->i", before, after) / np.maximum(
                np.hypot(*before.T) * np.hypot(*after.T), 1e-9)
            reversed_at = np.flatnonzero(cosine < _SPIKE_COSINE)
            if reversed_at.size:
                del keep[int(reversed_at[0]) + 1]
                changed = True
        return edge[keep]

    def trace(self, name=None, reverse=False, fraction=0.1):
        """A `FaultTrace` along the top edge, west to east (south to north if closer to N-S).

        `reverse=True` flips it, which swaps the fault's left and right. On a `merged`
        surface it runs through its sections' top edges in the order given, a straight line
        bridging any gap between them.
        """
        from .trace import FaultTrace

        lon, lat = self._top_lonlat(fraction)
        keep = np.r_[True, np.hypot(np.diff(lon), np.diff(lat)) > 0.0]
        lon, lat = lon[keep], lat[keep]
        d_east = (lon[-1] - lon[0]) * np.cos(np.radians(lat.mean()))
        d_north = lat[-1] - lat[0]
        backwards = (d_east < 0.0) if abs(d_east) >= abs(d_north) else (d_north < 0.0)
        if backwards != bool(reverse):
            lon, lat = lon[::-1], lat[::-1]
        return FaultTrace(lon, lat, name=name or self.name or None)

    def _top_lonlat(self, fraction):
        """The top edge, in order along it; through the sections, each turned to continue the last, if merged."""
        if not self.parts:
            edge = self.top_edge(fraction)
            lon, lat = self.lonlat()
            return np.asarray(lon)[edge], np.asarray(lat)[edge]
        pieces = [part._top_lonlat(fraction) for part in self.parts]

        def gap(a, b):
            return np.hypot((a[0] - b[0]) * np.cos(np.radians(a[1])), a[1] - b[1])

        if len(pieces) > 1:
            first, second = pieces[0], pieces[1]
            ends = [(first[0][k], first[1][k]) for k in (0, -1)]
            near = [min(gap(end, (second[0][j], second[1][j])) for j in (0, -1)) for end in ends]
            if near[0] < near[1]:
                pieces[0] = (first[0][::-1], first[1][::-1])
        lon, lat = list(pieces[0][0]), list(pieces[0][1])
        for piece in pieces[1:]:
            if gap((lon[-1], lat[-1]), (piece[0][-1], piece[1][-1])) < gap((lon[-1], lat[-1]), (piece[0][0], piece[1][0])):
                piece = (piece[0][::-1], piece[1][::-1])
            lon += list(piece[0])
            lat += list(piece[1])
        return np.asarray(lon), np.asarray(lat)

    def local(self, frame):
        """Each vertex in `frame`: `x`, `y`, `z`, metres."""
        lon, lat = self.lonlat()
        x, y = frame.to_local(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y), self.vertices[:, 2]

    def __repr__(self):
        return (f"<TSurf {self.name!r} {self.vertices.shape[0]} vertices, {self.triangles.shape[0]} triangles, "
                f"z {self.vertices[:, 2].min() / 1e3:.1f} to {self.vertices[:, 2].max() / 1e3:.1f} km>")


def read_tsurf(path, crs=None):
    """Every surface in a GOCAD t-surf file: a list of `TSurf`.

    Reads `VRTX` and `PVRTX`, whatever follows the coordinates, `ATOM` and `PATOM` (a
    second name for an earlier vertex), `TRGL` and the header's `name`, across any number of
    `TFACE` sections. `ZPOSITIVE` and `AXIS_UNIT` are honoured, so every surface comes back
    in metres with z up. Comments and border and bstone lines are skipped.
    """
    surfaces = []
    state = None

    def finish():
        if state is not None and state["triangles"]:
            surfaces.append(_assemble(state, crs))

    with open(path, errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            key = parts[0]
            if key == "GOCAD":
                finish()
                state = _new_state()
                state["kind"] = parts[1] if len(parts) > 1 else ""
            elif state is None:
                continue
            elif key.startswith("name:") or key == "name":
                text = line.split(":", 1)[1].strip() if ":" in line else " ".join(parts[1:])
                if not state["name"]:
                    state["name"] = text.strip('"')
            elif key == "ZPOSITIVE" and len(parts) > 1:
                state["z_up"] = parts[1].lower() != "depth"
            elif key == "AXIS_UNIT" and len(parts) > 1:
                units = [p.strip('"').lower() for p in parts[1:]]
                state["scale"] = [_unit(u, path) for u in units[:3]] if len(units) >= 3 else [_unit(units[0], path)] * 3
            elif key in ("VRTX", "PVRTX"):
                state["vertices"][int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif key in ("ATOM", "PATOM"):
                state["atoms"][int(parts[1])] = int(parts[2])
            elif key == "TRGL":
                state["triangles"].append((int(parts[1]), int(parts[2]), int(parts[3])))
            elif key == "END":
                finish()
                state = None
    finish()
    if not surfaces:
        raise ValueError(f"{path}: no triangulated surface found; is it a GOCAD TSurf?")
    return surfaces


def _new_state():
    return {"kind": "", "name": "", "z_up": True, "scale": [1.0, 1.0, 1.0], "vertices": {}, "atoms": {}, "triangles": []}


def _unit(text, path):
    try:
        return _UNIT_SCALE[text]
    except KeyError:
        raise ValueError(f"{path}: unknown AXIS_UNIT {text!r}") from None


def _assemble(state, crs):
    """Vertices re-indexed from the file's numbers, atoms resolved to the vertex they name."""
    ids = sorted(state["vertices"])
    index = {vertex: k for k, vertex in enumerate(ids)}

    def resolve(number):
        seen = set()
        while number in state["atoms"]:
            if number in seen:
                raise ValueError(f"surface {state['name']!r}: ATOM {number} refers to itself")
            seen.add(number)
            number = state["atoms"][number]
        if number not in index:
            raise ValueError(f"surface {state['name']!r}: a triangle or atom names vertex {number}, which is not defined")
        return index[number]

    vertices = np.array([state["vertices"][vertex] for vertex in ids], dtype=float) * np.asarray(state["scale"])
    if not state["z_up"]:
        vertices[:, 2] = -vertices[:, 2]
    triangles = np.array([[resolve(corner) for corner in triangle] for triangle in state["triangles"]], dtype=int)
    return TSurf(vertices, triangles, name=state["name"], crs=crs)
