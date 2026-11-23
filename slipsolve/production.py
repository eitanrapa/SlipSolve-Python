"""A full inversion run: meshes adapted to the resolution, model-driven samples and a cross-validated smoothing.

`Production` runs one recipe in checkpointed stages, so an interrupted run resumes where
it stopped:

| stage | what |
|---|---|
| `A` | start meshes, first samples, solve at `Recipe.rank0` |
| `B` | re-mesh from the resolution at `rank0` until the element count settles |
| `C0` | redraw every series' samples from the model until they settle |
| `C` | choose the rank by blocked cross-validation on mesh B |
| `D` | re-mesh at that rank: the final model |
| `E` | cross-validate on the final mesh: is the rank inside its 1% band? |
| `report` | `model.slip.zip` and `summary.json` |

What a run needs is described, not coded:

- `Sheet`: one fault sheet from a trace, a dip (or a dip that varies with depth and
  along strike) and a depth.
- `DataSeries`: one raster data set on the run's lattice (phase or pixel offsets), with its
  nuisance terms and how near the faults it is used.
- `points`: observations used as they are (GNSS), with their own weights.
- `Recipe`: the settings, with the defaults below.

```python
from slipsolve import FaultTrace, LayeredTDE, RemainderTables
from slipsolve.production import DataSeries, Production, Recipe, Sheet

trace = FaultTrace.from_kml("rupture.kml")
frame = trace.local_frame()
sheets = [Sheet("main", trace, frame, dip=70.0, depth=20e3)]
series = [DataSeries("S1_asc", asc_grid, kind="phase", ramp="linear", near=3e3),
          DataSeries("S1_asc_rg", asc_offsets, kind="offset", ramp="linear", near=2e3)]
run = Production("runs/prod1", sheets, series, points=[gnss],
                 engine=LayeredTDE(RemainderTables.load("tables.npz")), velocity_model=crust,
                 recipe=Recipe(rank0=50, workers=6, solve=dict(polarity=(1, 0, 0))))
if __name__ == "__main__":
    model = run.run()
```

With `Recipe.workers` above one, Green's functions and surface fields are computed in
spawned processes, which re-import the calling script: keep the run under
`if __name__ == "__main__":`.
"""

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve

from . import resolution as R
from ._geo import LosGrid
from .greens import HalfSpaceTDE
from .inversion import SlipInversion, SlipModel, _far_enough, _surface_segments
from .layout import MAX_COUNT_RATIO, clip_layout, lloyd_layout
from .mesh import FaultMesh, _arc_length
from .predict import _fill_nearest
from .regularize import slip_bounds
from .sampling import Observations, _frame_axes
from .sampling import frame_looks as _rotate_looks
from .selection import CV_RANKS, RANK_JITTER, GridEdge, NormalSystem, choose_rank, slab_folds
from .trace import _smoothed_normals

#: The stages `Production.run` knows, in order.
STAGES = ("A", "B", "C0", "C", "D", "E", "X", "report")
#: Each checkpoint's input: a checkpoint is current only while it was built from that file as it is now.
_PARENT = {"B.pkl": "A.pkl", "C0.pkl": "B.pkl", "C.pkl": "C0.pkl", "final.pkl": "C.pkl"}
#: The degree of each nuisance kind's polynomial.
RAMP_DEGREE = {"none": None, "offset": 0, "linear": 1, "quadratic": 2, "cubic": 3}


@dataclass
class Recipe:
    """The settings of a `Production` run.

    Attributes:
        rank0: The effective rank of stages A to C0, before cross-validation chooses one.
        start_edge: The element size of the start meshes, m.
        fraction: Element size as a fraction of the local resolving width.
        size_min: The smallest element size, m.
        size_max: The largest element size, m.
        count_tol: A mesh has settled when its element count changes by less than this
            fraction twice running.
        maps_b: The most re-meshes in stage B.
        maps_d: The most re-meshes in stage D.
        rounds: The most resampling rounds in stage C0.
        round_tol: Sampling has settled when the row count and the slip both change by
            less than this fraction.
        field_spacing: The grid the model's surface field is forwarded on before it is
            interpolated to the lattice, m. Within two of these of the fault's outcrop the
            field is forwarded at each pixel instead.
        width_min: The smallest quadtree cell, m.
        width_max: The largest quadtree cell, m.
        min_valid_fraction: A cell needs at least this fraction of valid pixels.
        split: A cell splits while the model varies across it by more than this times the
            series' sigma, per kind.
        round0: The first samples, drawn from the data alone: `rms_min`, `sigma` and
            optionally `width_min`, per kind (`sigma` None measures it from the data's own
            scatter). `DataSeries.first` overrides them for one series.
        min_region_km2: A connected region of a phase series smaller than this is dropped;
            every larger one gets its own offset.
        min_region_samples: A phase region left with fewer samples than this after its
            quadtree is dropped with them, since its own offset would absorb them (0: keep
            every region).
        sigma: How each series' sigma is measured from the latest model. `"far"`: its
            residual beyond `sigma_far` of every trace, after its own nuisance terms, as
            `sigma_block` block medians. `"whole"`: every pixel of a phase series after
            region offsets and a plane, and every sample of an offsets series.
        sigma_far: See `sigma`, m.
        sigma_block: See `sigma`, m.
        folds: The slabs of the blocked cross-validation.
        shifts: The partitions averaged, each with its slab edges moved by `1 / shifts` of
            a slab.
        cv_ranks: The ranks the cross-validation spans.
        slab_span: `"data"`: the slabs divide the extent of the series' samples, and
            point data join the end slabs; `"all"`: every row's extent.
        solve: Keywords for `SlipInversion.solve` (and `NormalSystem`): ratios, edges,
            polarity and bounds.
        workers: Processes for the Green's functions and the surface field.
    """

    rank0: float = 50.0
    start_edge: float = 4000.0
    fraction: float = 0.35
    size_min: float = 1500.0
    size_max: float = 10000.0
    count_tol: float = 0.02
    maps_b: int = 8
    maps_d: int = 6
    rounds: int = 4
    round_tol: float = 0.02
    field_spacing: float = 2500.0
    width_min: float = 2000.0
    width_max: float = 20000.0
    min_valid_fraction: float = 0.5
    split: dict = field(default_factory=lambda: {"phase": 1.0, "offset": 0.5})
    round0: dict = field(default_factory=lambda: {"phase": dict(rms_min=0.02, sigma=0.02),
                                                  "offset": dict(rms_min=0.2, sigma=None)})
    min_region_km2: float = 100.0
    min_region_samples: int = 0
    sigma: str = "far"
    sigma_far: float = 10e3
    sigma_block: float = 2000.0
    folds: int = 12
    shifts: int = 4
    cv_ranks: tuple = CV_RANKS
    slab_span: str = "data"
    solve: dict = field(default_factory=lambda: dict(
        ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=3e3, sides=("bottom", "left", "right"), polarity=None,
        strike=(-20.0, 20.0), dip=(-15.0, 15.0)))
    workers: int = 1

    def __post_init__(self):
        if self.sigma not in ("far", "whole"):
            raise ValueError(f"sigma={self.sigma!r}: one of 'far', 'whole'")
        if self.slab_span not in ("data", "all"):
            raise ValueError(f"slab_span={self.slab_span!r}: one of 'data', 'all'")
        base = Recipe.__dataclass_fields__["solve"].default_factory()
        self.solve = {**base, **dict(self.solve)}


def _side_of(px, py, cx, cy):
    """Signed distance of the points `(px, py)` from the polyline `(cx, cy)` (+ on its left), and whether
    the nearest point of the polyline is an interior one (not beyond either end)."""
    ax, ay = cx[:-1], cy[:-1]
    vx, vy = np.diff(cx), np.diff(cy)
    l2 = np.maximum(vx * vx + vy * vy, 1e-12)
    t = ((px[:, None] - ax) * vx + (py[:, None] - ay) * vy) / l2
    tc = np.clip(t, 0.0, 1.0)
    dx, dy = px[:, None] - (ax + tc * vx), py[:, None] - (ay + tc * vy)
    d = np.hypot(dx, dy)
    k = np.argmin(d, axis=1)
    r = np.arange(px.size)
    cross = vx[k] * dy[r, k] - vy[k] * dx[r, k]
    inside = ~(((k == 0) & (t[r, k] < 0)) | ((k == vx.size - 1) & (t[r, k] > 1)))
    return np.sign(cross) * d[r, k], inside


def _jsonable(value):
    """`value` as a JSON record can hold it: tuples and arrays as lists, numpy numbers as numbers, a trace as its
    name and vertices, a segment as its four coordinates, and any other object as its repr."""
    from .trace import FaultSegment, FaultTrace

    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, FaultTrace):
        return dict(name=value.name, lon=_jsonable(np.asarray(value.lon)), lat=_jsonable(np.asarray(value.lat)))
    if isinstance(value, FaultSegment):
        return [value.x_begin, value.y_begin, value.x_end, value.y_end]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class Sheet:
    """One fault sheet: a trace, a dip and a depth, meshed on any `(arc length, depth)` layout.

    Each node at depth `z` is moved `z cot(dip)` along the trace's left-hand normal,
    smoothed along strike over `normal_smoothing` so a wiggly trace does not fold the sheet
    at depth. A dip below 90 degrees dips to the left of the drawing direction. `profile`,
    a function of `(s, z)` arrays returning a dip in degrees, replaces the uniform dip: the
    offset is then the integral of `cot(dip)` over depth, which makes a listric sheet.

    `ends_on`, another `Sheet`, makes this one end where it meets that sheet: every mesh is
    cut off below the meeting depth (`slipsolve.layout.clip_layout`), and its nodes on
    that edge are pinned to zero slip, as a bottom edge is. `bounds` overrides the
    recipe's slip bounds on this sheet: any of `polarity`, `strike` and `dip`, as
    `SlipInversion.solve` takes them. `strike=(0, 0)` with `polarity=(0, 1)`, say, allows
    positive dip-slip only.

    Attributes:
        name: The sheet's name.
        trace: Its `FaultTrace`.
        frame: The `LocalFrame` of every sheet and series.
        dip: The uniform dip, degrees.
        depth: The sheet's depth, m.
        length: The trace's arc length in the mesh's parameterisation, m.
        ends_on: The `Sheet` this one ends on, or None.
        meeting: `(s, depth)`: where this sheet meets `ends_on` along its arc length (`depth`
            where it never does), or None.
        bounds: The slip-bound overrides, or None.
    """

    def __init__(self, name, trace, frame, dip=90.0, depth=20e3, profile=None, normal_smoothing=10e3, spacing=500.0,
                 ends_on=None, bounds=None):
        self.name, self.trace, self.frame = str(name), trace, frame
        self.dip, self.depth, self.profile = float(dip), float(depth), profile
        self.bounds = dict(bounds) if bounds else None
        unknown = set(self.bounds or {}) - {"polarity", "strike", "dip"}
        if unknown:
            raise ValueError(f"sheet {name}: bounds take polarity, strike and dip, not {sorted(unknown)}")
        ref = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=[0.0, 2e3], along_spacing=spacing,
                               edge_length=spacing)
        top = np.abs(ref.nodes[:, 2]) < 1.0
        order = np.argsort(ref.params[top, 0])
        self._s = ref.params[top, 0][order]
        self._x, self._y = ref.nodes[top, 0][order], ref.nodes[top, 1][order]
        if abs(self._s[0]) > 1.0:
            raise ValueError(f"sheet {name}: the trace's parameterisation starts at s = {self._s[0]:.1f} m, not 0")
        self._nx, self._ny = _smoothed_normals(self._x, self._y, self._s, normal_smoothing)
        self.length = float(self._s[-1])
        self.ends_on = ends_on
        self.meeting = self.meeting_depth(ends_on) if ends_on is not None else None

    def offset(self, s, depth):
        """How far each point `(s, depth)` lies from the trace along the normal, m."""
        if self.profile is None:
            return depth * (0.0 if self.dip == 90.0 else 1.0 / np.tan(np.radians(self.dip)))
        zz = np.linspace(0.0, 1.0, 61)[None, :] * depth[:, None]
        dips = np.asarray(self.profile(np.repeat(s[:, None], zz.shape[1], axis=1), zz), float)
        cot = np.where(np.abs(dips - 90.0) < 1e-9, 0.0, 1.0 / np.tan(np.radians(dips)))
        return np.trapezoid(cot, zz, axis=1)

    def position(self, s, depth):
        """`(x, y)` in the frame of the sheet's surface at arc lengths `s` and depth `depth` (m)."""
        s = np.asarray(s, float)
        depth = np.broadcast_to(np.asarray(depth, float), s.shape)
        step = self.offset(s.reshape(-1), depth.reshape(-1)).reshape(s.shape)
        return (np.interp(s, self._s, self._x) + step * np.interp(s, self._s, self._nx),
                np.interp(s, self._s, self._y) + step * np.interp(s, self._s, self._ny))

    def meeting_depth(self, host, dz=50.0):
        """`(s, depth)`: where each column of this sheet first crosses `host`'s surface.

        The sheet's columns, one per point of its reference trace, are stepped down by `dz`
        and compared with the host's depth contour at the same depth. A column that never
        crosses, or crosses beyond either end of the host, keeps the sheet's own depth.
        """
        s = self._s
        zend = np.full(s.size, self.depth)
        prev = None
        for z in np.arange(0.0, self.depth + dz / 2, dz):
            bx, by = self.position(s, z)
            hx, hy = host.position(host._s, z)
            side, inside = _side_of(bx, by, hx, hy)
            if prev is not None:
                flip = (np.sign(side) != np.sign(prev[0])) & inside & prev[1] & (zend >= self.depth) & (prev[0] != 0)
                if flip.any():
                    frac = np.abs(prev[0][flip]) / np.maximum(np.abs(prev[0][flip]) + np.abs(side[flip]), 1e-9)
                    zend[flip] = z - dz + frac * dz
            prev = (side, inside)
        return s.copy(), zend

    def lower_edge(self, mesh, tol=30.0):
        """The nodes of one of this sheet's meshes on the edge where it meets `ends_on` (none if it does not)."""
        if self.meeting is None:
            return np.zeros(mesh.n_nodes, bool)
        s, depth = mesh.params[:, 0], -mesh.params[:, 1]
        end = np.interp(s, *self.meeting)
        return (np.abs(depth - end) <= tol) & (end < self.depth - 1.0) & (depth > 0.0)

    def mesh(self, layout):
        """The `FaultMesh` of a layout (`slipsolve.layout.lloyd_layout`, `resolution.adapt_layout`).

        A sheet that ends on another has the layout cut off where it meets it first.
        """
        if self.meeting is not None and (self.meeting[1] < self.depth).any():
            layout = clip_layout(layout, *self.meeting)
        s, depth = layout.params[:, 0], -layout.params[:, 1]
        step = self.offset(s, depth)
        nodes = np.column_stack([np.interp(s, self._s, self._x) + step * np.interp(s, self._s, self._nx),
                                 np.interp(s, self._s, self._y) + step * np.interp(s, self._s, self._ny), -depth])
        mesh = FaultMesh(nodes, layout.triangles, layout.params, frame=self.frame,
                         attrs={"kind": "production", "sheet": self.name, "dip_deg": self.dip,
                                "max_depth": self.depth, "top_depth": 0.0, "trace_name": self.name,
                                "trace_length": self.length, "layout": str(layout.attrs.get("layout", ""))})
        mesh._check_winding((self._s, self._nx, self._ny))
        return mesh

    def start(self, edge):
        """A uniform unstructured mesh of element size `edge` (at least four columns along strike)."""
        return self.mesh(lloyd_layout(self.length, self.depth, float(min(edge, self.length / 4.0))))

    @property
    def direction(self):
        """`(dx, dy)` from the trace's first point to its last."""
        return float(self._x[-1] - self._x[0]), float(self._y[-1] - self._y[0])


class CurvedSheet(Sheet):
    """A sheet whose surface is `FaultMesh.curved`'s, each layout carried onto it.

    `FaultMesh.curved` fits its surface on the trace resampled at `edge_length` from the
    keywords in `curved` (`segments` and `dips`, `uniform_dip`, `bottom_trace`, and their
    companions; `max_depth` there is the depth the dips are fitted to). The layouts set the
    sheet's extent, down to `depth`, along the arc length of that resampled trace, which is
    the sheet's `length`. Use it when the surface must be exactly `FaultMesh.curved`'s, for
    instance a dip per segment of the trace; a `Sheet` with a `profile` gives a smooth
    approximation of it.

    A curved sheet cannot end on another sheet, nor be one another sheet ends on.

    Attributes:
        curved: The `FaultMesh.curved` keywords.
        edge_length: The trace resampling the surface is fitted on, m.
    """

    def __init__(self, name, trace, frame, depth=20e3, edge_length=4000.0, bounds=None, **curved):
        self.name, self.trace, self.frame = str(name), trace, frame
        self.depth, self.profile, self.edge_length = float(depth), None, float(edge_length)
        self.curved = dict(curved)
        dips = self.curved.get("dips")
        self.dip = float(np.mean(dips)) if dips is not None else float(self.curved.get("uniform_dip", 90.0))
        self.bounds = dict(bounds) if bounds else None
        unknown = set(self.bounds or {}) - {"polarity", "strike", "dip"}
        if unknown:
            raise ValueError(f"sheet {name}: bounds take polarity, strike and dip, not {sorted(unknown)}")
        x, y = trace.resample(self.edge_length, frame).to_local(frame)
        self._x, self._y = np.asarray(x, float), np.asarray(y, float)
        self._s = _arc_length(self._x, self._y)
        self.length = float(self._s[-1])
        self.ends_on, self.meeting = None, None

    def offset(self, s, depth):
        raise NotImplementedError("a CurvedSheet's surface is FaultMesh.curved's fit, not an offset from the trace")

    def position(self, s, depth):
        raise NotImplementedError("a CurvedSheet cannot be the sheet another one ends on")

    def meeting_depth(self, host, dz=50.0):
        raise NotImplementedError("a CurvedSheet cannot end on another sheet")

    def mesh(self, layout):
        """The `FaultMesh` of a layout, on `FaultMesh.curved`'s surface."""
        mesh = FaultMesh.curved(self.trace, self.frame, layout=layout, edge_length=self.edge_length, **self.curved)
        mesh.attrs.update(kind="production", sheet=self.name, max_depth=self.depth, top_depth=0.0,
                          trace_name=self.name, trace_length=self.length, layout=str(layout.attrs.get("layout", "")))
        return mesh


@dataclass
class DataSeries:
    """One raster data set on the run's lattice.

    Attributes:
        name: The track name its samples carry (and `sees` keys on).
        grid: A `LosGrid` on the run's lattice: `los` positive toward the sensor, with its
            look vectors.
        kind: `"phase"` or `"offset"`.
        ramp: Its nuisance terms: `"offset"`, `"linear"`, `"quadratic"` or `"cubic"`.
        near: Phase is masked within this distance of every trace; offsets are excluded
            within it, m.
        reach: Offsets are used only within this distance of a trace, m.
        valid: A boolean mask on the lattice of the pixels to use (coherence, a zone to
            leave out), or None.
        first: Overrides of `Recipe.round0` for this series' first, data-driven samples:
            any of `rms_min`, `sigma` and `width_min`. `sigma` may name a variable of the
            grid, a per-pixel uncertainty, whose median over the usable pixels is then used.
        regions: Phase only: integer labels on the lattice (0 for none), each with its own
            offset, used as given in place of the connected regions of at least
            `Recipe.min_region_km2` (labels drawn on the native product, a split by hand).
            The `near` and `valid` masks apply on top; labels are not merged, and dropped
            only by `Recipe.min_region_samples`.
    """

    name: str
    grid: object
    kind: str = "phase"
    ramp: str = "linear"
    near: float = 0.0
    reach: float = np.inf
    valid: object = None
    first: dict = None
    regions: object = None

    def __post_init__(self):
        if self.kind not in ("phase", "offset"):
            raise ValueError(f"series {self.name}: kind={self.kind!r}, one of 'phase', 'offset'")
        if self.ramp not in RAMP_DEGREE or self.ramp == "none":
            raise ValueError(f"series {self.name}: ramp={self.ramp!r}, one of offset, linear, quadratic, cubic")
        if self.regions is not None and self.kind != "phase":
            raise ValueError(f"series {self.name}: regions are for phase series")
        unknown = set(self.first or {}) - {"rms_min", "sigma", "width_min"}
        if unknown:
            raise ValueError(f"series {self.name}: first= takes rms_min, sigma and width_min, not {sorted(unknown)}")


class _Log:
    """Print and append to a file, stamped with the seconds since the run started."""

    def __init__(self, path, echo=True):
        self.path, self.t0, self.echo = Path(path), time.time(), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg=""):
        line = f"[{time.time() - self.t0:7.0f}s] {msg}" if msg else ""
        if self.echo:
            print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


# -- worker processes ------------------------------------------------------------------------------------------------
_WORKER = {}


def _make_engine(spec):
    """The engine `_engine_spec` describes."""
    kind, payload = spec
    if kind == "layered":
        from .layered_tde import LayeredTDE
        from .wavenumber import RemainderTables
        path, options = payload
        return LayeredTDE(RemainderTables.load(path), **options)
    if kind == "curvature":
        from .curvature import Curvature, CurvatureTables
        base, path, order = payload
        return Curvature(_make_engine(base), CurvatureTables.load(path), order=order)
    return HalfSpaceTDE(payload)


def _start_worker(spec):
    """Build the worker's engines once: the run's own, and a half-space one for the surface field."""
    kind, payload = spec
    _WORKER["engine"] = _make_engine(spec)
    _WORKER["halfspace"] = HalfSpaceTDE(payload if kind == "halfspace" else 0.25)


def _green_rows(mesh, obs, frame_looks=False):
    return np.asarray(SlipInversion(mesh, obs, engine=_WORKER["engine"], ramp="none", basis="node",
                                    frame_looks=frame_looks).g, float)


def _worker_pool(engine, workers):
    """`workers` spawned processes, each with its own copy of `engine` (see `_engine_spec`)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    return ProcessPoolExecutor(int(workers), mp_context=get_context("spawn"), initializer=_start_worker,
                               initargs=(_engine_spec(engine),))


def _cached_green(mesh, obs, engine, *, frame_looks=False, cache=None, pool=None, workers=1):
    """`(g, seconds)`: the node-basis Green's matrix of `obs` against `mesh`, and how long it took.

    With `cache` (a directory) the matrix is read from or saved to a file named by what it
    depends on: the mesh's nodes and triangles, the samples' positions and look vectors, the
    engine and `frame_looks`. `seconds` is None when it was read. `pool`, a callable
    returning a process pool or None, is asked only when the matrix has to be assembled; the
    rows are then split into `3 * workers` chunks.
    """
    path = None
    if cache is not None:
        rotated = "_fl" if frame_looks else ""
        path = Path(cache) / f"g_{mesh.digest()[:12]}_{_digest(obs)}_{_engine_key(engine)}{rotated}.npy"
        if path.exists():
            return np.load(path), None
    t0 = time.time()
    executor = pool() if pool is not None else None
    if executor is None:
        g = np.asarray(SlipInversion(mesh, obs, engine=engine, ramp="none", basis="node",
                                     frame_looks=frame_looks).g, float)
    else:
        edges = np.linspace(0, obs.n, 3 * int(workers) + 1).astype(int)
        jobs = [(mesh, Observations(obs.ds.isel(obs=slice(a, b))), frame_looks)
                for a, b in zip(edges[:-1], edges[1:]) if b > a]
        g = np.vstack(list(executor.map(_green_rows, *zip(*jobs))))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, g)
    return g, time.time() - t0


def _forward_rows(mesh, slip, x, y):
    return _blocked_forward(_WORKER["halfspace"], mesh, slip, x, y)


def _blocked_forward(engine, mesh, slip, x, y, block=2048):
    """East, north and up `(n, 3)` of `slip` at the points, a block of points at a time: the engine builds a
    `(points, 3, 2 * n_elements)` array."""
    out = np.empty((np.size(x), 3))
    for a in range(0, np.size(x), block):
        out[a:a + block] = engine.forward(mesh, slip, x[a:a + block], y[a:a + block])
    return out


def _element_faults(mesh):
    """The joined fault of each element (all 0 for a mesh that was not joined)."""
    counts = mesh.attrs.get("joined_elements")
    if counts is None:
        return np.zeros(mesh.n_elements, dtype=int)
    return np.repeat(np.arange(len(counts)), np.asarray(counts, dtype=int))


def _engine_spec(engine):
    from .curvature import Curvature
    from .layered_tde import LayeredTDE
    if isinstance(engine, Curvature):
        path = getattr(engine.tables, "path", None)
        if path is None:
            raise ValueError("workers > 1 with a Curvature engine needs its tables loaded from (or saved to) a file")
        return "curvature", (_engine_spec(engine.base), str(path), engine.order)
    if isinstance(engine, LayeredTDE):
        path = getattr(engine.tables, "path", None)
        if path is None:
            raise ValueError("workers > 1 with a LayeredTDE needs tables loaded from (or saved to) a file")
        return "layered", (str(path), dict(tolerance=engine.tolerance, max_order=engine.max_order,
                                           far=engine.strain_far, far_rule=engine.far_rule))
    if isinstance(engine, HalfSpaceTDE):
        return "halfspace", float(getattr(engine, "nu", 0.25))
    raise ValueError(f"workers > 1 supports LayeredTDE and HalfSpaceTDE engines and Curvature on them, "
                     f"not {type(engine).__name__}")


def _file_digest(path):
    """A checkpoint's identity: its bytes' hash, or None while it does not exist."""
    path = Path(path)
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def _engine_key(engine):
    """What an engine computes, as a short hash: the Green's function cache is shared by every run in a folder.

    A half-space by its Poisson's ratio; a layered engine by its tables' crust and grids and its own settings; a
    `Curvature` by its base engine's key and its tables; any other engine by its type and its scalar settings.
    """
    from .curvature import Curvature
    if isinstance(engine, Curvature):
        h = hashlib.sha1(b"Curvature")
        h.update(_engine_key(engine.base).encode())
        t = engine.tables
        for a in [t.r, t.z, t.interfaces] + [t.values[k][c] for k in sorted(t.values) for c in sorted(t.values[k])]:
            h.update(np.ascontiguousarray(a, dtype=float).tobytes())
        h.update(json.dumps(t.attrs, sort_keys=True, default=str).encode())
        h.update(f"order={engine.order}".encode())
        return h.hexdigest()[:8]
    h = hashlib.sha1(type(engine).__name__.encode())
    tables = getattr(engine, "tables", None)
    if tables is not None:
        for a in (tables.r, tables.z, tables.tops, tables.lam, tables.mu):
            h.update(np.ascontiguousarray(a, dtype=float).tobytes())
        h.update(json.dumps(tables.attrs, sort_keys=True, default=str).encode())
        h.update(str(np.shape(tables.values)).encode())
    for name, value in sorted(vars(engine).items()):
        if isinstance(value, (bool, int, float, str, type(None))):
            h.update(f"{name}={value!r};".encode())
    return h.hexdigest()[:8]


def _digest(obs):
    h = hashlib.sha1()
    for k in ("x", "y", "look_e", "look_n", "look_u"):
        h.update(np.ascontiguousarray(obs.ds[k].values, dtype=float).tobytes())
    return h.hexdigest()[:16]


def _slip_change(new, old):
    """The rms change of the slip from `old` to `new` relative to the rms of `new`, over both components."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(new), 1e-300))


def _nuisance_columns(regions, x, y, degree):
    """One column per region and the bivariate monomials up to `degree` (scaled by 100 km)."""
    cols = [(regions == k).astype(float) for k in np.unique(regions)]
    xs, ys = x / 1e5, y / 1e5
    cols += [xs ** (d - j) * ys ** j for d in range(1, (degree or 0) + 1) for j in range(d + 1)]
    return np.column_stack(cols)


def _block_rms(r, k):
    """The rms of `k x k` block medians of a lattice field (blocks at least half valid)."""
    if k <= 1:
        return float(np.sqrt(np.nanmean(r ** 2)))
    ny, nx = (s // k * k for s in r.shape)
    b = r[:ny, :nx].reshape(ny // k, k, nx // k, k).transpose(0, 2, 1, 3).reshape(ny // k, nx // k, k * k)
    ok = np.sum(np.isfinite(b), axis=2) >= (k * k + 1) // 2
    return float(np.sqrt(np.mean(np.nanmedian(b[ok], axis=1) ** 2)))


def _scatter(los, valid):
    """A raster's noise without a model: 1.4826 MAD about a five-pixel median."""
    filled = np.where(valid, los, np.nan)
    smooth = ndimage.generic_filter(filled, np.nanmedian, size=5, mode="nearest")
    d = (filled - smooth)[valid & np.isfinite(smooth)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


class Production:
    """One inversion run of the production recipe, checkpointed in `out`.

    Args:
        out: The run's folder.
        sheets: The `Sheet`s, in join order (`sees` and `events` index them).
        series: The `DataSeries`, all on one lattice in the sheets' frame.
        points: `Observations` used as they are, weights included (GNSS). Each keeps its
            own track name.
        engine: The elastic engine: a `LayeredTDE` (its tables loaded from a file when
            `workers > 1`), a `HalfSpaceTDE` (the default), or `slipsolve.curvature.Curvature`
            on either (its tables from a file too).
        velocity_model: The `VelocityModel` the moments integrate the rigidity of.
        recipe: The `Recipe`.
        sees: `{track: [sheet indices]}`: which sheets each named track sees
            (`SlipInversion`'s `sees`). A track not named sees every sheet.
        events: `{label: [sheet indices]}`: moments reported per group of sheets.
        traces: The `FaultTrace`s that distances to the faults are measured from; the
            sheets' own traces by default.
        samples: `Observations` to start from instead of the first data-driven samples.
        constrain: A function `(inversion, meshes) -> inversion` applied to every system
            assembled, for constraints beyond the recipe's (pinned edges, rake limits).
        correction: A function `(mesh, obs) -> dG`, an array the shape of the Green's
            functions (`obs.n` rows; strike-slip then dip-slip columns, one per node of the
            joined mesh), added to them in stage X: an effect the engine leaves out, such as
            a rigidity contrast across the fault solved exactly. Without it stage X is skipped.
        frame_looks: Rotate every look vector into the frame's axes, for the Green's functions
            (`SlipInversion`'s `frame_looks`) and for the predictions the samples are drawn
            and the sigmas measured from.
        echo: Print the log as well as writing it.
    """

    def __init__(self, out, sheets, series, points=(), engine=None, velocity_model=None, recipe=None, sees=None,
                 events=None, traces=None, samples=None, constrain=None, correction=None, frame_looks=False,
                 echo=True):
        self.out = Path(out)
        self.sheets = list(sheets)
        self.series = {s.name: s for s in series}
        if len(self.series) != len(series):
            raise ValueError("two series share a name")
        self.points = list(points)
        self.engine = engine if engine is not None else HalfSpaceTDE()
        self.velocity_model = velocity_model
        self.recipe = recipe or Recipe()
        self.sees = dict(sees) if sees else None
        self.events = dict(events) if events else None
        self.traces = list(traces) if traces is not None else [s.trace for s in self.sheets]
        self.samples = samples
        self.constrain = constrain
        self.correction = correction
        self.frame_looks = bool(frame_looks)
        self.echo = echo
        self.frame = self.sheets[0].frame
        self.direction = self.sheets[0].direction
        self._pool = None
        self._lattice = None
        self.log = lambda message: None          # until `run` opens the log

    # -- the lattice ---------------------------------------------------------------------------------------------
    def _setup_lattice(self):
        grids = [s.grid for s in self.series.values()]
        x, y = grids[0].ds["x"].values, grids[0].ds["y"].values
        for s in self.series.values():
            if not (np.array_equal(s.grid.ds["x"].values, x) and np.array_equal(s.grid.ds["y"].values, y)):
                raise ValueError(f"series {s.name} is not on the first series' lattice")
        X, Y = np.meshgrid(x, y)
        # every pixel's distance to the nearest trace, cached in the run's folder: on a large lattice it takes minutes
        h = hashlib.sha1()
        for a in [x, y] + [np.column_stack([t.lon, t.lat]) for t in self.traces]:
            h.update(np.ascontiguousarray(a, dtype=float).tobytes())
        cache = self.out / f"lattice_distance_{h.hexdigest()[:16]}.npy"
        if cache.exists():
            dist = np.load(cache)
        else:
            dist = np.min([np.asarray(t.distance(X.ravel(), Y.ravel(), self.frame), float) for t in self.traces],
                          axis=0).reshape(X.shape)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, dist)
        spacing = float(np.median(np.abs(np.diff(x))))
        axes = None
        if self.frame_looks:                     # true east and north in the frame's axes, for `_lattice_look`
            east, north = _frame_axes(self.frame, X, Y)
            axes = (east.astype(np.float32), north.astype(np.float32))
        self._lattice = dict(x=x, y=y, X=X, Y=Y, dist=dist, spacing=spacing, axes=axes)

    def _lattice_look(self, ds):
        """A series' look vector on the lattice, `(x, y, up)`: rotated into the frame's axes with `frame_looks`."""
        e, n, u = (np.asarray(ds[f"los_{c}"].values, float) for c in ("east", "north", "up"))
        axes = self._lattice.get("axes")
        if axes is None:
            return e, n, u
        east, north = axes
        return e * east[0] + n * north[0], e * east[1] + n * north[1], u

    def _usable(self, s):
        """`(los, valid, kwargs)`: a series' pixels to sample, and its sampler's extra keywords."""
        lat, r = self._lattice, self.recipe
        los = np.asarray(s.grid.ds["los"].values[0], float)
        valid = np.isfinite(los)
        if s.valid is not None:
            valid &= np.asarray(s.valid, bool)
        extra = {}
        if s.kind == "phase" and s.regions is not None:
            regions = np.asarray(s.regions)
            if regions.shape != valid.shape:
                raise ValueError(f"series {s.name}: regions {regions.shape} against the lattice's {valid.shape}")
            valid &= (lat["dist"] >= s.near) & (regions > 0)
            extra["regions"] = np.where(valid, regions, 0).astype(int)
        elif s.kind == "phase":
            valid &= lat["dist"] >= s.near
            labels, n = ndimage.label(valid)
            regions = np.zeros_like(labels)
            if n:
                size = ndimage.sum(valid, labels, index=np.arange(1, n + 1))
                big = np.flatnonzero(size >= r.min_region_km2 * 1e6 / lat["spacing"] ** 2) + 1
                for k, label in enumerate(big, start=1):
                    regions[labels == label] = k
            valid &= regions > 0
            extra["regions"] = np.where(valid, regions, 0)
        else:
            valid &= (lat["dist"] >= s.near) & (lat["dist"] <= s.reach)
            extra["exclude_within"] = float(s.near)
        return los, valid, extra

    # -- workers ---------------------------------------------------------------------------------------------------
    def _workers(self):
        if self.recipe.workers < 2:
            return None
        if self._pool is None:
            self._pool = _worker_pool(self.engine, self.recipe.workers)
        return self._pool

    def _engine_name(self):
        return str(getattr(self.engine, "name", type(self.engine).__name__))

    def close(self):
        """Shut the worker processes down (`run` does at its end)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _green(self, mesh, obs):
        """The node-basis Green's matrix of `obs` against `mesh`, cached in `out/g_cache`."""
        g, seconds = _cached_green(mesh, obs, self.engine, frame_looks=self.frame_looks,
                                   cache=self.out / "g_cache", pool=self._workers, workers=self.recipe.workers)
        if seconds is not None:
            self.log(f"  G {g.shape} in {seconds:.0f} s")
        return g

    # -- systems and solves ------------------------------------------------------------------------------------------
    def _joined(self, meshes):
        return FaultMesh.join(meshes) if len(meshes) > 1 else meshes[0]

    def _ramps(self, obs):
        return {t: (self.series[t].ramp if t in self.series else "none") for t in obs.tracks}

    def solve_options(self, mesh):
        """`(options, fixed)`: the solve keywords for a joined mesh, and its slip parameters fixed at zero.

        The recipe's `solve`, with each sheet's `bounds` applied to its own nodes as
        per-parameter limits. `fixed` is None when no sheet overrides the bounds.
        """
        kw = dict(self.recipe.solve)
        if not any(s.bounds for s in self.sheets):
            return kw, None
        counts = [int(c) for c in np.atleast_1d(mesh.attrs.get("joined_nodes", [mesh.n_nodes]))]
        n = int(sum(counts))
        lo, hi = slip_bounds(n, kw["strike"], kw["dip"], kw.get("polarity"))
        a = 0
        for sheet, c in zip(self.sheets, counts):
            if sheet.bounds:
                b = {"strike": kw["strike"], "dip": kw["dip"], "polarity": kw.get("polarity"), **sheet.bounds}
                l, h = slip_bounds(c, b["strike"], b["dip"], b["polarity"])
                lo[a:a + c], hi[a:a + c] = l[:c], h[:c]
                lo[n + a:n + a + c], hi[n + a:n + a + c] = l[c:], h[c:]
            a += c
        kw.update(strike=(lo[:n], hi[:n]), dip=(lo[n:], hi[n:]), polarity=None)
        return kw, (lo == 0.0) & (hi == 0.0)

    def _correction(self, mesh, obs):
        """`correction`'s `dG` for the mesh and samples, cached in `out/g_cache`."""
        path = self.out / "g_cache" / f"dg_{mesh.digest()[:12]}_{_digest(obs)}.npy"
        if path.exists():
            return np.load(path)
        t0 = time.time()
        dg = np.asarray(self.correction(mesh, obs), float)
        expect = (obs.n, 2 * mesh.n_nodes)
        if dg.shape != expect:
            raise ValueError(f"the correction is {dg.shape}; the Green's functions are {expect}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, dg)
        self.log(f"  correction {dg.shape} in {time.time() - t0:.0f} s")
        return dg

    def system(self, meshes, obs, corrected=False):
        """`(inversion, NormalSystem)` for the meshes and samples.

        Slip parameters a sheet's `bounds` fix at zero lose their Green's functions, so they
        stay out of the fit and the rank, and the nodes where a sheet meets the one it ends
        on are pinned (`SlipInversion.pinned`). `corrected` adds `correction`'s `dG` to the
        Green's functions (stage X).
        """
        mesh = self._joined(meshes)
        g = self._green(mesh, obs)
        kw, fixed = self.solve_options(mesh)
        if corrected:
            g = g + self._correction(mesh, obs)
        elif fixed is not None and fixed.any():
            g = g.copy()
        if fixed is not None and fixed.any():
            g[:, fixed] = 0.0
        inv = SlipInversion(mesh, obs, engine=self.engine, g=g, basis="node", ramp=self._ramps(obs),
                            velocity_model=self.velocity_model, sees=self.sees, frame_looks=self.frame_looks)
        if any(s.meeting is not None for s in self.sheets):
            inv.pinned = np.concatenate([s.lower_edge(m) for s, m in zip(self.sheets, meshes)])
        if self.constrain is not None:
            inv = self.constrain(inv, meshes)
        return inv, NormalSystem(inv, **kw)

    def solve(self, inv, lam):
        return inv.solve(smoothing=lam, method="active_set", **self.solve_options(inv.mesh)[0])

    def moments(self, model):
        """`{"total": Mw, sheet: Mw, event: Mw}` on the model's rigidity, each sheet's or group's slip alone."""
        counts = [int(c) for c in np.atleast_1d(model.mesh.attrs.get("joined_nodes", [model.basis.n_basis]))]
        n = model.basis.n_basis
        starts = np.r_[0, np.cumsum(counts)]
        groups = {s.name: [k] for k, s in enumerate(self.sheets)}
        groups.update(self.events or {})
        x0 = np.asarray(model.x, float).copy()
        mw = lambda m0: float((2.0 / 3.0) * (np.log10(m0) - 9.1)) if m0 > 0 else float("nan")  # noqa: E731
        out = {"total": mw(model.moment())}
        try:
            for label, members in groups.items():
                keep = np.zeros(n, bool)
                for k in members:
                    keep[starts[k]:starts[k + 1]] = True
                x = x0.copy()
                x[:n][~keep] = 0.0
                x[n:2 * n][~keep] = 0.0
                model.x, model.strike_slip, model.dip_slip = x, x[:n], x[n:2 * n]
                out[label] = mw(model.moment())
        finally:
            model.x, model.strike_slip, model.dip_slip = x0, x0[:n], x0[n:2 * n]
        return out

    def _far_rms(self, model):
        """`{track: residual rms at its samples beyond sigma_far}`, for the log and the summary."""
        trk = np.asarray(model.obs.ds["track"].values).astype(str)
        res = np.asarray(model.residual, float)
        x, y = (np.asarray(model.obs.ds[k].values, float) for k in ("x", "y"))
        dist = np.min([np.asarray(t.distance(x, y, self.frame), float) for t in self.traces], axis=0)
        far = dist > self.recipe.sigma_far
        return {t: float(np.sqrt(np.mean(res[(trk == t) & far] ** 2))) for t in np.unique(trk)
                if np.any((trk == t) & far)}

    def summary(self, model, system, lam, label):
        mw = self.moments(model)
        trk = np.asarray(model.obs.ds["track"].values).astype(str)
        res = np.asarray(model.residual, float)
        w = np.asarray(model.obs.ds["weight"].values, float)
        stats = {t: dict(n=int(np.sum(trk == t)), rms_m=float(np.sqrt(np.mean(res[trk == t] ** 2))),
                         chi=float(np.sqrt(np.mean((w * res)[trk == t] ** 2)))) for t in np.unique(trk)}
        for t, v in self._far_rms(model).items():
            stats[t]["rms_far_m"] = v
        rank = system.rank(lam)
        self.log(f"{label}: {model.mesh.n_elements} elements / {model.mesh.n_nodes} nodes; {model.obs.n} rows; "
                 f"weight {lam:.5g} -> rank {rank:.1f}; Mw {mw['total']:.3f} ("
                 + ", ".join(f"{k} {v:.2f}" for k, v in mw.items() if k != "total")
                 + f"); peak {model.max_slip:.2f} m")
        return dict(label=label, lam=float(lam), rank=rank, Mw=mw, peak=float(model.max_slip), stats=stats,
                    nodes=int(model.mesh.n_nodes), elements=int(model.mesh.n_elements), rows=int(model.obs.n))

    # -- meshes ----------------------------------------------------------------------------------------------------
    def _mesh_map(self, meshes, inv, system, rank, omega):
        r = self.recipe
        lam = system.lam_for_rank(rank)
        D, _, _ = system.normal()
        T = D + system.damping(lam)
        T.flat[:: system.P + 1] += RANK_JITTER * (np.trace(D) or np.trace(T)) / system.P
        report = R.resolution_kernels(inv, cho_solve(cho_factor(T, lower=True), D))
        sizes = np.asarray(R.kernel_element_size(inv, lam, fraction=r.fraction, reduce="gaussian",
                                                 size_min=r.size_min, size_max=r.size_max, report=report), float)
        if omega < 1.0:
            current = np.asarray(R.element_size(inv.mesh), float)
            sizes = np.clip(np.exp(omega * np.log(sizes) + (1.0 - omega) * np.log(current)), r.size_min, r.size_max)
        new, a = [], 0
        for sheet, mesh in zip(self.sheets, meshes):
            b = a + mesh.n_elements
            layout = R.adapt_layout(mesh, sizes[a:b], budget=None, balance=MAX_COUNT_RATIO, structure="unstructured",
                                    length=sheet.length, max_depth=sheet.depth)
            new.append(sheet.mesh(layout))
            a = b
        return new, lam, sizes

    @staticmethod
    def _degenerate(meshes):
        bad = 0
        for m in meshes:
            v = np.asarray(m.nodes, float)[np.asarray(m.triangles)]
            area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
            bad += int(np.sum(area < 1e-6 * np.median(area)))
        return bad

    def _converge(self, meshes, obs, rank, label, max_maps):
        """Re-mesh until the element count changes by less than `count_tol` twice running."""
        rows, streak, omega, last = [], 0, 1.0, 0
        for k in range(1, max_maps + 1):
            t0 = time.time()
            inv, system = self.system(meshes, obs)
            new, lam, sizes = self._mesh_map(meshes, inv, system, rank, omega)
            if self._degenerate(new):
                self.log(f"  {label} map {k}: degenerate triangles in the new meshes; the previous ones are kept")
                return meshes, False, rows
            n_in = sum(m.n_elements for m in meshes)
            n_out = sum(m.n_elements for m in new)
            rel = abs(n_out - n_in) / n_in
            rows.append(dict(stage=label, map=k, n_in=n_in, n_out=n_out, rel_change=rel, lam=lam, omega=omega,
                             target_h_med_km=float(np.median(sizes) / 1e3), seconds=round(time.time() - t0)))
            self.log(f"  {label} map {k}: {n_in} -> {n_out} elements ({100 * rel:.1f}%), weight {lam:.4g}, median "
                     f"target size {np.median(sizes) / 1e3:.2f} km" + (f", relaxed x{omega:g}" if omega < 1 else ""))
            sign = int(np.sign(n_out - n_in))
            if last and sign and sign != last and omega == 1.0:
                omega = 0.5
            last = sign or last
            meshes = new
            streak = streak + 1 if rel < self.recipe.count_tol else 0
            if streak >= 2:
                self.log(f"  {label}: settled after {k} maps ({n_out} elements)")
                return meshes, True, rows
        self.log(f"  {label}: not settled in {max_maps} maps")
        return meshes, False, rows

    # -- sampling --------------------------------------------------------------------------------------------------
    def _halfspace(self):
        """The half-space the surface field is forwarded in: the engine itself if it is one."""
        return self.engine if isinstance(self.engine, HalfSpaceTDE) else HalfSpaceTDE(0.25)

    def _forward(self, mesh, slip, x, y):
        """`_halfspace`'s east, north and up `(n, 3)` of element `slip` at the points, on the workers if any."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        pool = self._workers()
        if pool is None or x.size < 2:
            return _blocked_forward(self._halfspace(), mesh, slip, x, y)
        edges = np.linspace(0, x.size, 4 * self.recipe.workers + 1).astype(int)
        parts = pool.map(_forward_rows, *zip(*[(mesh, slip, x[a:b], y[a:b])
                                               for a, b in zip(edges[:-1], edges[1:]) if b > a]))
        return np.vstack(list(parts))

    def _seen(self, model, name):
        """The faults series `name` sees (`sees`), as a sorted tuple, or None for all of them."""
        sees = getattr(model, "_inversion", None)
        sees = getattr(sees, "sees", None) or self.sees or {}
        faults = sees.get(name)
        return None if faults is None else tuple(sorted(int(f) for f in faults))

    @staticmethod
    def _slip_seen(model, faults):
        """The model's element slip with the faults outside `faults` (None: none) set to zero."""
        slip = np.array(model.element_slip, dtype=float)
        if faults is not None:
            slip[~np.isin(_element_faults(model.mesh), faults)] = 0.0
        return slip

    def field(self, model, need=None, faults=None):
        """The model's east, north and up on the lattice: `(east, north, up)`, NaN outside `need`.

        A half-space forward of the element means (in the engine's Poisson's ratio when the engine is a
        half-space; `sample` adds the difference any other engine makes, `_engine_offset`). It is computed
        exactly at the pixels within two `field_spacing` of the fault's outcrop, where the field steps across
        the fault and bends fastest, and elsewhere interpolated bilinearly from a `field_spacing` grid
        evaluated only at the nodes those pixels use. No interpolated pixel reaches a node across the fault,
        and pixels within a metre of the outcrop, where the solution is singular, are left NaN.

        `need` is a boolean lattice of the pixels wanted (all of them by default), and `faults` keeps only
        those joined faults' slip, for a series that sees only some of them (`sees`).
        """
        lat, step = self._lattice, float(self.recipe.field_spacing)
        X, Y = lat["X"], lat["Y"]
        shape = X.shape
        need = np.ones(shape, bool) if need is None else np.asarray(need, bool)
        out = np.full((3, X.size), np.nan)
        idx = np.flatnonzero(need.ravel())
        if idx.size:
            mesh, slip = model.mesh, self._slip_seen(model, faults)
            outcrop = _surface_segments(mesh)
            px, py = X.ravel()[idx], Y.ravel()[idx]
            near = ~_far_enough(px, py, *outcrop, 2.0 * step)
            exact = idx[near][_far_enough(px[near], py[near], *outcrop, 1.0)]
            if exact.size:
                out[:, exact] = self._forward(mesh, slip, X.ravel()[exact], Y.ravel()[exact]).T
            rest = idx[~near]
            if rest.size:
                fx, fy = X.ravel()[rest], Y.ravel()[rest]
                cx = np.arange(X.min() - 2 * step, X.max() + 3 * step, step)
                cy = np.arange(Y.min() - 2 * step, Y.max() + 3 * step, step)
                ix = np.clip(np.searchsorted(cx, fx, side="right") - 1, 0, cx.size - 2)
                iy = np.clip(np.searchsorted(cy, fy, side="right") - 1, 0, cy.size - 2)
                used = np.zeros((cy.size, cx.size), bool)
                for di in (0, 1):
                    for dj in (0, 1):
                        used[iy + dj, ix + di] = True
                gx, gy = np.meshgrid(cx, cy)
                keep = used.ravel() & _far_enough(gx.ravel(), gy.ravel(), *outcrop, 0.5 * step)
                values = self._forward(mesh, slip, gx.ravel()[keep], gy.ravel()[keep])
                at = np.column_stack([fy, fx])
                for c in range(3):
                    v = np.full(gx.size, np.nan)
                    v[keep] = values[:, c]
                    out[c, rest] = RegularGridInterpolator((cy, cx), _fill_nearest(v.reshape(gx.shape)))(at)
        return [out[c].reshape(shape) for c in range(3)]

    def _engine_offset(self, model, name, valid, faults=None):
        """The engine's line of sight minus `field`'s half-space one, for series `name` on its `valid` pixels.

        Zero for a half-space engine. Otherwise measured at the model's samples of the series, where the
        engine's prediction is known exactly (`SlipModel.fault_prediction`), and interpolated linearly
        between them (from the nearest sample outside their hull). The difference crosses the fault
        continuously, since both fields step there by the slip.
        """
        if isinstance(self.engine, HalfSpaceTDE):
            return 0.0
        ds = model.obs.ds
        rows = np.flatnonzero(np.asarray(ds["track"].values).astype(str) == name)
        if rows.size:
            outcrop = _surface_segments(model.mesh)
            rows = rows[_far_enough(ds["x"].values[rows], ds["y"].values[rows], *outcrop, 1.0)]
        if rows.size < 3:
            self.log(f"    {name}: no samples in the last model; its field is the half-space one")
            return 0.0
        x, y = ds["x"].values[rows], ds["y"].values[rows]
        enu = self._forward(model.mesh, self._slip_seen(model, faults), x, y)
        look = [ds[k].values[rows] for k in ("look_e", "look_n", "look_u")]
        if self.frame_looks:
            look = _rotate_looks(self.frame, x, y, *look)
        hs = enu[:, 0] * look[0] + enu[:, 1] * look[1] + enu[:, 2] * look[2]
        delta = np.asarray(model.fault_prediction, float)[rows] - hs
        lat = self._lattice
        qx, qy = lat["X"][valid], lat["Y"][valid]
        points = np.column_stack([x, y])
        got = LinearNDInterpolator(points, delta)(qx, qy)
        outside = ~np.isfinite(got)
        if outside.any():
            got[outside] = NearestNDInterpolator(points, delta)(qx[outside], qy[outside])
        offset = np.zeros(valid.shape)
        offset[valid] = got
        return offset

    def _sigma(self, s, los, valid, regions, prediction, model):
        """A series' sigma from the model (`Recipe.sigma`)."""
        r, lat = self.recipe, self._lattice
        X, Y = lat["X"], lat["Y"]
        filled = np.where(valid, los, np.nan)
        regions = valid.astype(int) if regions is None else regions
        if r.sigma == "whole":
            if s.kind == "offset":
                trk = np.asarray(model.obs.ds["track"].values).astype(str)
                res = np.asarray(model.residual, float)[trk == s.name]
                if res.size:
                    return float(np.sqrt(np.mean(res ** 2)))
                return _scatter(los, valid)
            m = (regions > 0) & np.isfinite(filled) & np.isfinite(prediction)
            resid = (filled - prediction)[m]
            A = _nuisance_columns(regions[m], X[m], Y[m], 1)
            coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
            return float(np.sqrt(np.mean((resid - A @ coef) ** 2)))
        far = valid & (lat["dist"] > r.sigma_far) & np.isfinite(prediction)
        if far.sum() < 2000:
            self.log(f"    {s.name}: fewer than 2000 pixels beyond {r.sigma_far / 1e3:g} km; sigma from all of them")
            far = valid & np.isfinite(prediction)
        resid = (filled - prediction)[far]
        A = _nuisance_columns(regions[far], X[far], Y[far], RAMP_DEGREE[s.ramp])
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        full = np.full(los.shape, np.nan)
        full[far] = resid - A @ coef
        return _block_rms(full, max(1, int(round(r.sigma_block / lat["spacing"]))))

    def sample(self, model=None):
        """`(observations, {series: sigma})`: every series' quadtree, from the data alone (`model` None) or split on
        the model's field and weighted one over its sigma; the points as they are."""
        r = self.recipe
        usable = {name: self._usable(s) for name, s in self.series.items()}
        fields = {}
        if model is not None:
            # one surface field per set of faults seen, over the pixels the series seeing them use
            need = {}
            for name, (_, valid, _) in usable.items():
                if valid.sum() >= 500:
                    key = self._seen(model, name)
                    need[key] = need[key] | valid if key in need else valid.copy()
            fields = {key: self.field(model, mask, faults=key) for key, mask in need.items()}
        parts, sigmas = [], {}
        for s in self.series.values():
            los, valid, extra = usable[s.name]
            if valid.sum() < 500:
                self.log(f"    {s.name}: only {int(valid.sum())} usable pixels; skipped")
                continue
            ds = s.grid.ds.copy()
            ds["los"] = (("pair", "y", "x"), np.where(valid, los, np.nan)[None].astype(np.float32))
            if model is None:
                first = {**r.round0[s.kind], **(s.first or {})}
                sigma = first.get("sigma")
                if isinstance(sigma, str):
                    sigma = float(np.nanmedian(np.asarray(s.grid.ds[sigma].values, float)[valid]))
                elif sigma is None:
                    sigma = _scatter(los, valid)
                kw = dict(rms_min=first["rms_min"], width_min=first.get("width_min", r.width_min))
            else:
                faults = self._seen(model, s.name)
                enu = fields[faults]
                prediction = sum(c * v for c, v in zip(enu, self._lattice_look(ds)))
                prediction = prediction + self._engine_offset(model, s.name, valid & np.isfinite(prediction), faults)
                sigma = self._sigma(s, los, valid, extra.get("regions"), prediction, model)
                kw = dict(rms_min=r.split[s.kind] * sigma, width_min=r.width_min,
                          field=np.where(valid, prediction, np.nan))
            obs = Observations.from_los(LosGrid(ds), name=s.name, frame=self.frame, trace=self.sheets[0].trace,
                                        width_max=r.width_max, stat="median", min_valid_fraction=r.min_valid_fraction,
                                        **kw, **extra)
            if s.kind == "phase" and r.min_region_samples > 1 and "region" in obs.ds:
                labels, counts = np.unique(obs.ds["region"].values, return_counts=True)
                small = labels[counts < r.min_region_samples]
                if small.size:
                    keep = np.flatnonzero(~np.isin(obs.ds["region"].values, small))
                    self.log(f"    {s.name}: {small.size} of {labels.size} regions with fewer than "
                             f"{r.min_region_samples} samples left out ({obs.n - keep.size} samples)")
                    if keep.size == 0:
                        continue
                    obs = Observations(obs.ds.isel(obs=keep))
            obs.ds["weight"] = ("obs", np.full(obs.n, 1.0 / sigma))
            sigmas[s.name] = sigma
            self.log(f"    {s.name}: {obs.n} samples, sigma {1e3 * sigma:.1f} mm")
            parts.append(obs)
        return Observations.concat(parts + self.points, normalize="none"), sigmas

    # -- cross-validation ----------------------------------------------------------------------------------------
    def _folds(self, obs):
        r = self.recipe
        trk = np.asarray(obs.ds["track"].values).astype(str)
        span = None if r.slab_span == "all" else np.isin(trk, list(self.series))
        folds, width = [], None
        for k in range(max(1, r.shifts)):
            labels, width = slab_folds(obs, r.folds, self.direction, shift=k / max(1, r.shifts), span=span)
            folds.append(labels)
        self.log(f"  CV: {r.folds} slabs of {width / 1e3:.1f} km" + (f", {r.shifts} shifted partitions"
                                                                       if r.shifts > 1 else ""))
        return folds if len(folds) > 1 else folds[0]

    def cross_validate(self, system, obs):
        """`choose_rank` on the run's folds; the rank ladder widened (twice at most) if the minimum is on its edge."""
        ranks = tuple(self.recipe.cv_ranks)
        folds = self._folds(obs)
        for attempt in range(3):
            try:
                return choose_rank(system, folds, ranks=ranks, weighted=True, log=self.log)
            except GridEdge as edge:
                if attempt == 2:
                    raise
                ranks = ((ranks[0], 2.0 * ranks[1]) if edge.rank > np.sqrt(ranks[0] * ranks[1])
                         else (0.5 * ranks[0], ranks[1]))
                self.log(f"  CV minimum on the grid edge (rank {edge.rank:.1f}); ranks widened to {ranks}")

    # -- checkpoints -----------------------------------------------------------------------------------------------
    def _save(self, name, meshes, obs, lam, rank, extra=None):
        extra = dict(extra or {})
        if name in _PARENT:
            extra["parent"] = _file_digest(self.out / _PARENT[name])
        with open(self.out / name, "wb") as fh:
            pickle.dump(dict(meshes=meshes, obs=obs, lam=lam, rank=rank, extra=extra), fh)

    def load(self, name):
        """A stage's checkpoint: `(meshes, observations, weight, rank, extra)`."""
        with open(self.out / name, "rb") as fh:
            z = pickle.load(fh)
        return z["meshes"], z["obs"], z["lam"], z["rank"], z["extra"]

    def _current(self, name):
        """Whether checkpoint `name` exists and was built from its input checkpoint as that is now.

        A checkpoint written before the inputs were recorded counts as current.
        """
        if not (self.out / name).exists():
            return False
        parent = _PARENT.get(name)
        if parent is None:
            return True
        recorded = self.load(name)[4].get("parent")
        return recorded is None or recorded == _file_digest(self.out / parent)

    def _load_current(self, name):
        """`load(name)`, refused when the checkpoint is older than its input (`_current`)."""
        if not self._current(name):
            raise ValueError(f"{name} was built from an earlier {_PARENT[name]}: run its stage again "
                             f"(stages {STAGES}), or delete it")
        return self.load(name)

    # -- the run ---------------------------------------------------------------------------------------------------
    def run(self, stages=STAGES):
        """Run `stages`; return the final model, or None before `D`.

        A stage is skipped while its checkpoint is current: built from the checkpoint before it as that is now
        (E's and X's records from `final.pkl`). Redoing one therefore redoes every later stage that is run, and
        a stage that would read an out-of-date checkpoint raises instead.
        """
        import pandas as pd

        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages {sorted(unknown)}; they are {STAGES}")
        r = self.recipe
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = _Log(self.out / "run.log", echo=self.echo)
        self._setup_lattice()
        record_path = self.out / "stages.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}

        def keep(key, value):
            record[key] = value
            record_path.write_text(json.dumps(record, indent=1, default=float))

        self.log(f"run {self.out.name}: stages {list(stages)}; sheets " + ", ".join(
            f"{s.name} {s.length / 1e3:.0f} km at {s.dip:g}" for s in self.sheets) + f"; series {list(self.series)}; "
            f"points {[t for p in self.points for t in p.tracks]}; rank0 {r.rank0:g}; sigma {r.sigma}; "
            f"CV {r.folds} slabs x {r.shifts}; workers {r.workers}; engine {self._engine_name()}"
            + ("; looks in the frame's axes" if self.frame_looks else ""))
        for s in self.sheets:
            if s.meeting is not None:
                cut = s.meeting[1] < s.depth
                self.log(f"  {s.name} ends on {s.ends_on.name} along {100 * cut.mean():.0f}% of its length"
                         + (f", at {s.meeting[1][cut].min() / 1e3:.1f}-{s.meeting[1][cut].max() / 1e3:.1f} km"
                            if cut.any() else ""))
            if s.bounds:
                self.log(f"  {s.name} bounds: {s.bounds}")
        model = None
        try:
            if "A" in stages and not self._current("A.pkl"):
                meshes = [s.start(r.start_edge) for s in self.sheets]
                if self.samples is not None:
                    obs = self.samples
                    self.log(f"A: start meshes ({sum(m.n_elements for m in meshes)} elements); {obs.n} given samples")
                else:
                    self.log(f"A: start meshes ({sum(m.n_elements for m in meshes)} elements); first samples:")
                    obs, _ = self.sample()
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(r.rank0)
                keep("A", self.summary(self.solve(inv, lam), system, lam, f"A (start meshes, rank {r.rank0:g})"))
                self._save("A.pkl", meshes, obs, lam, r.rank0)

            if "B" in stages and not self._current("B.pkl"):
                meshes, obs, *_ = self._load_current("A.pkl")
                meshes, ok, rows = self._converge(meshes, obs, r.rank0, "B", r.maps_b)
                pd.DataFrame(rows).to_csv(self.out / "converge_B.csv", index=False)
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(r.rank0)
                keep("B", dict(self.summary(self.solve(inv, lam), system, lam, "B"), converged=ok))
                self._save("B.pkl", meshes, obs, lam, r.rank0, {"converged": ok})

            if "C0" in stages and not self._current("C0.pkl"):
                meshes, obs, *_ = self._load_current("B.pkl")
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(r.rank0)
                model = self.solve(inv, lam)
                rows, done = [], False
                for k in range(1, r.rounds + 1):
                    t0 = time.time()
                    new_obs, sigmas = self.sample(model)
                    inv, system = self.system(meshes, new_obs)
                    lam = system.lam_for_rank(r.rank0)
                    new = self.solve(inv, lam)
                    n2 = 2 * inv.basis.n_basis
                    dslip = _slip_change(new.x[:n2], model.x[:n2])
                    dn = abs(new_obs.n - obs.n) / obs.n
                    rows.append(dict(round=k, rows=new_obs.n, rel_rows=dn, rel_slip=dslip, lam=lam,
                                     Mw=self.moments(new)["total"], peak=float(new.max_slip),
                                     **{f"sigma_{t}": v for t, v in sigmas.items()}, seconds=round(time.time() - t0)))
                    self.log(f"  C0 round {k}: {obs.n} -> {new_obs.n} rows ({100 * dn:.1f}%), slip change "
                             f"{100 * dslip:.1f}% rms")
                    pd.DataFrame(rows).to_csv(self.out / "sampling_C0.csv", index=False)
                    obs, model = new_obs, new
                    if dn < r.round_tol and dslip < r.round_tol:
                        done = True
                        break
                keep("C0", dict(self.summary(model, system, lam, "C0"), converged=done))
                self._save("C0.pkl", meshes, obs, lam, r.rank0, {"converged": done})

            if "C" in stages and not self._current("C.pkl"):
                meshes, obs, *_ = self._load_current("C0.pkl")
                inv, system = self.system(meshes, obs)
                self.log("C: blocked cross-validation on mesh B")
                choice = self.cross_validate(system, obs)
                choice.table.to_csv(self.out / "cv_C.csv", index=False)
                (self.out / "cv_C.json").write_text(json.dumps(choice.to_dict(), indent=1, default=float))
                keep("C", self.summary(self.solve(inv, choice.lam), system, choice.lam, f"C (rank {choice.rank:.2f})"))
                self._save("C.pkl", meshes, obs, choice.lam, choice.rank)

            if "D" in stages and not self._current("final.pkl"):
                meshes, obs, _, rank, _ = self._load_current("C.pkl")
                meshes, ok, rows = self._converge(meshes, obs, rank, "D", r.maps_d)
                pd.DataFrame(rows).to_csv(self.out / "converge_D.csv", index=False)
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(rank)
                model = self.solve(inv, lam)
                keep("D", dict(self.summary(model, system, lam, f"final (rank {rank:.2f})"), converged=ok))
                self._save("final.pkl", meshes, obs, lam, rank, {"converged": ok})
                model.save(self.out / "final.slip.zip")

            final = _file_digest(self.out / "final.pkl")

            def current(key):
                """Stage `key`'s record exists and was made from final.pkl as it is now."""
                return key in record and record[key].get("final", final) == final

            if "E" in stages and final is not None and not current("E"):
                meshes, obs, lam, rank, _ = self._load_current("final.pkl")
                inv, system = self.system(meshes, obs)
                self.log("E: blocked cross-validation on the final mesh")
                choice = self.cross_validate(system, obs)
                choice.table.to_csv(self.out / "cv_E.csv", index=False)
                inside = choice.inside(rank)
                self.log(f"E: the final rank {rank:.2f} is {'inside' if inside else 'outside'} the final mesh's 1% "
                         "band" + ("s " if len(choice.bands) > 1 else " ")
                         + ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in choice.bands))
                keep("E", dict(rank=rank, cv_rank=choice.rank, bands=choice.bands, inside=bool(inside), final=final))

            if "X" in stages and self.correction is not None and final is not None and not current("X"):
                meshes, obs, lam, rank, _ = self._load_current("final.pkl")
                self.log(f"X: the final mesh and samples with the correction, at the final rank {rank:.2f}")
                inv, system = self.system(meshes, obs, corrected=True)
                lam_x = system.lam_for_rank(rank)
                model_x = self.solve(inv, lam_x)
                model_x.save(self.out / "final_corrected.slip.zip")
                s = self.summary(model_x, system, lam_x, f"X (corrected, rank {rank:.2f})")
                choice = self.cross_validate(system, obs)
                choice.table.to_csv(self.out / "cv_X.csv", index=False)
                inside = choice.inside(rank)
                self.log(f"X: the final rank {rank:.2f} is {'inside' if inside else 'outside'} the corrected "
                         "system's 1% band" + ("s " if len(choice.bands) > 1 else " ")
                         + ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in choice.bands)
                         + f" (its own rank {choice.rank:.2f})")
                keep("X", dict(lam=lam_x, rank=rank, Mw=s.get("Mw"), cv_rank=choice.rank, bands=choice.bands,
                               inside=bool(inside), final=final))

            if "report" in stages and final is not None:
                meshes, obs, lam, rank, extra = self._load_current("final.pkl")
                for key in [k for k in ("E", "X") if k in record and not current(k)]:
                    if key == "X" and self.correction is not None:
                        raise ValueError("stage X was solved on an earlier final.pkl: run X again")
                    self.log(f"report: stage {key}'s record is from an earlier final.pkl; left out")
                    del record[key]
                corrected = self.correction is not None and "X" in record
                if corrected:
                    # the corrected re-solve is the model to quote; the uncorrected final stays beside it
                    inv0, _ = self.system(meshes, obs)
                    self.solve(inv0, lam).save(self.out / "model_uncorrected.slip.zip")
                    lam = float(record["X"]["lam"])
                inv, system = self.system(meshes, obs, corrected=corrected)
                model = self.solve(inv, lam)
                s = self.summary(model, system, lam, "report (corrected)" if corrected else "report")
                model.save(self.out / "model.slip.zip")
                counts = {t: int(np.sum(np.asarray(obs.ds["track"].values).astype(str) == t)) for t in obs.tracks}
                recipe = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(r).items()}
                (self.out / "summary.json").write_text(json.dumps(dict(
                    recipe=recipe, sheets=[dict(name=sh.name, dip=sh.dip, depth=sh.depth, length=sh.length,
                                                listric=sh.profile is not None,
                                                ends_on=sh.ends_on.name if sh.ends_on is not None else None,
                                                bounds=_jsonable(sh.bounds),
                                                curved=_jsonable(getattr(sh, "curved", None)))
                                           for sh in self.sheets],
                    corrected=corrected, engine=self._engine_name(), frame_looks=self.frame_looks,
                    **s, rows_by_track=counts, converged=extra.get("converged"), stages=record),
                    indent=1, default=float))
            elif (self.out / "final.slip.zip").exists() and model is None:
                model = SlipModel.load(self.out / "final.slip.zip")
        finally:
            self.close()
        self.log("done")
        return model
