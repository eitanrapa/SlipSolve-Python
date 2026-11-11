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
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter1d

from . import resolution as R
from ._geo import LosGrid
from .greens import HalfSpaceTDE
from .inversion import SlipInversion, SlipModel, _far_enough
from .layout import MAX_COUNT_RATIO, lloyd_layout
from .mesh import FaultMesh
from .predict import _fill_nearest
from .sampling import Observations
from .selection import CV_RANKS, RANK_JITTER, GridEdge, NormalSystem, choose_rank, slab_folds

#: The stages `Production.run` knows, in order.
STAGES = ("A", "B", "C0", "C", "D", "E", "report")
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
            interpolated to the lattice, m.
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


class Sheet:
    """One fault sheet: a trace, a dip and a depth, meshed on any `(arc length, depth)` layout.

    Each node at depth `z` is moved `z cot(dip)` along the trace's left-hand normal,
    smoothed along strike over `normal_smoothing` so a wiggly trace does not fold the sheet
    at depth. A dip below 90 degrees dips to the left of the drawing direction. `profile`,
    a function of `(s, z)` arrays returning a dip in degrees, replaces the uniform dip: the
    offset is then the integral of `cot(dip)` over depth, which makes a listric sheet.

    Attributes:
        name: The sheet's name.
        trace: Its `FaultTrace`.
        frame: The `LocalFrame` of every sheet and series.
        dip: The uniform dip, degrees.
        depth: The sheet's depth, m.
        length: The trace's arc length in the mesh's parameterisation, m.
    """

    def __init__(self, name, trace, frame, dip=90.0, depth=20e3, profile=None, normal_smoothing=10e3, spacing=500.0):
        self.name, self.trace, self.frame = str(name), trace, frame
        self.dip, self.depth, self.profile = float(dip), float(depth), profile
        ref = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=[0.0, 2e3], along_spacing=spacing,
                               edge_length=spacing)
        top = np.abs(ref.nodes[:, 2]) < 1.0
        order = np.argsort(ref.params[top, 0])
        self._s = ref.params[top, 0][order]
        self._x, self._y = ref.nodes[top, 0][order], ref.nodes[top, 1][order]
        if abs(self._s[0]) > 1.0:
            raise ValueError(f"sheet {name}: the trace's parameterisation starts at s = {self._s[0]:.1f} m, not 0")
        tx, ty = np.gradient(self._x, self._s), np.gradient(self._y, self._s)
        norm = np.hypot(tx, ty)
        nx, ny = -ty / norm, tx / norm
        step = float(np.median(np.diff(self._s)))
        if normal_smoothing:
            nx = gaussian_filter1d(nx, normal_smoothing / step, mode="nearest")
            ny = gaussian_filter1d(ny, normal_smoothing / step, mode="nearest")
        norm = np.hypot(nx, ny)
        self._nx, self._ny = nx / norm, ny / norm
        self.length = float(self._s[-1])

    def offset(self, s, depth):
        """How far each point `(s, depth)` lies from the trace along the normal, m."""
        if self.profile is None:
            return depth * (0.0 if self.dip == 90.0 else 1.0 / np.tan(np.radians(self.dip)))
        zz = np.linspace(0.0, 1.0, 61)[None, :] * depth[:, None]
        dips = np.asarray(self.profile(np.repeat(s[:, None], zz.shape[1], axis=1), zz), float)
        cot = np.where(np.abs(dips - 90.0) < 1e-9, 0.0, 1.0 / np.tan(np.radians(dips)))
        return np.trapezoid(cot, zz, axis=1)

    def mesh(self, layout):
        """The `FaultMesh` of a layout (`slipsolve.layout.lloyd_layout`, `resolution.adapt_layout`)."""
        s, depth = layout.params[:, 0], -layout.params[:, 1]
        step = self.offset(s, depth)
        nodes = np.column_stack([np.interp(s, self._s, self._x) + step * np.interp(s, self._s, self._nx),
                                 np.interp(s, self._s, self._y) + step * np.interp(s, self._s, self._ny), -depth])
        mesh = FaultMesh(nodes, layout.triangles, layout.params, frame=self.frame,
                         attrs={"kind": "production", "sheet": self.name, "dip_deg": self.dip,
                                "max_depth": self.depth, "top_depth": 0.0, "trace_name": self.name,
                                "trace_length": self.length, "layout": str(layout.attrs.get("layout", ""))})
        mesh._check_winding(self.trace)
        return mesh

    def start(self, edge):
        """A uniform unstructured mesh of element size `edge` (at least four columns along strike)."""
        return self.mesh(lloyd_layout(self.length, self.depth, float(min(edge, self.length / 4.0))))

    @property
    def direction(self):
        """`(dx, dy)` from the trace's first point to its last."""
        return float(self._x[-1] - self._x[0]), float(self._y[-1] - self._y[0])


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
            any of `rms_min`, `sigma` and `width_min`.
    """

    name: str
    grid: object
    kind: str = "phase"
    ramp: str = "linear"
    near: float = 0.0
    reach: float = np.inf
    valid: object = None
    first: dict = None

    def __post_init__(self):
        if self.kind not in ("phase", "offset"):
            raise ValueError(f"series {self.name}: kind={self.kind!r}, one of 'phase', 'offset'")
        if self.ramp not in RAMP_DEGREE or self.ramp == "none":
            raise ValueError(f"series {self.name}: ramp={self.ramp!r}, one of offset, linear, quadratic, cubic")
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


def _start_worker(spec):
    """Build the worker's engines once: the run's own, and a half-space one for the surface field."""
    kind, payload = spec
    if kind == "layered":
        from .layered_tde import LayeredTDE
        from .wavenumber import RemainderTables
        path, options = payload
        _WORKER["engine"] = LayeredTDE(RemainderTables.load(path), **options)
    else:
        _WORKER["engine"] = HalfSpaceTDE(payload)
    _WORKER["halfspace"] = HalfSpaceTDE(0.25)


def _green_rows(mesh, obs):
    return np.asarray(SlipInversion(mesh, obs, engine=_WORKER["engine"], ramp="none", basis="node").g, float)


def _forward_rows(mesh, slip, x, y):
    return np.asarray(_WORKER["halfspace"].forward(mesh, slip, x, y), float)


def _engine_spec(engine):
    from .layered_tde import LayeredTDE
    if isinstance(engine, LayeredTDE):
        path = getattr(engine.tables, "path", None)
        if path is None:
            raise ValueError("workers > 1 with a LayeredTDE needs tables loaded from (or saved to) a file")
        return "layered", (str(path), dict(tolerance=engine.tolerance, max_order=engine.max_order,
                                           far=engine.strain_far, far_rule=engine.far_rule))
    if isinstance(engine, HalfSpaceTDE):
        return "halfspace", float(getattr(engine, "nu", 0.25))
    raise ValueError(f"workers > 1 supports LayeredTDE and HalfSpaceTDE engines, not {type(engine).__name__}")


def _digest(obs):
    h = hashlib.sha1()
    for k in ("x", "y", "look_e", "look_n", "look_u"):
        h.update(np.ascontiguousarray(obs.ds[k].values, dtype=float).tobytes())
    return h.hexdigest()[:16]


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
            `workers > 1`) or a `HalfSpaceTDE` (the default).
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
        echo: Print the log as well as writing it.
    """

    def __init__(self, out, sheets, series, points=(), engine=None, velocity_model=None, recipe=None, sees=None,
                 events=None, traces=None, samples=None, constrain=None, echo=True):
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
        self.echo = echo
        self.frame = self.sheets[0].frame
        self.direction = self.sheets[0].direction
        self._pool = None
        self._lattice = None
        self.log = None

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
            np.save(cache, dist)
        spacing = float(np.median(np.abs(np.diff(x))))
        self._lattice = dict(x=x, y=y, X=X, Y=Y, dist=dist, spacing=spacing)

    def _usable(self, s):
        """`(los, valid, kwargs)`: a series' pixels to sample, and its sampler's extra keywords."""
        lat, r = self._lattice, self.recipe
        los = np.asarray(s.grid.ds["los"].values[0], float)
        valid = np.isfinite(los)
        if s.valid is not None:
            valid &= np.asarray(s.valid, bool)
        extra = {}
        if s.kind == "phase":
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
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            self._pool = ProcessPoolExecutor(int(self.recipe.workers), mp_context=get_context("spawn"),
                                             initializer=_start_worker, initargs=(_engine_spec(self.engine),))
        return self._pool

    def close(self):
        """Shut the worker processes down (`run` does at its end)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _green(self, mesh, obs):
        """The node-basis Green's matrix of `obs` against `mesh`, cached in `out/g_cache`."""
        path = self.out / "g_cache" / f"g_{mesh.digest()[:12]}_{_digest(obs)}.npy"
        if path.exists():
            return np.load(path)
        t0 = time.time()
        pool = self._workers()
        if pool is None:
            g = np.asarray(SlipInversion(mesh, obs, engine=self.engine, ramp="none", basis="node").g, float)
        else:
            edges = np.linspace(0, obs.n, 3 * self.recipe.workers + 1).astype(int)
            jobs = [(mesh, Observations(obs.ds.isel(obs=slice(a, b)))) for a, b in zip(edges[:-1], edges[1:]) if b > a]
            g = np.vstack(list(pool.map(_green_rows, *zip(*jobs))))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, g)
        self.log(f"  G {g.shape} in {time.time() - t0:.0f} s")
        return g

    # -- systems and solves ------------------------------------------------------------------------------------------
    def _joined(self, meshes):
        return FaultMesh.join(meshes) if len(meshes) > 1 else meshes[0]

    def _ramps(self, obs):
        return {t: (self.series[t].ramp if t in self.series else "none") for t in obs.tracks}

    def system(self, meshes, obs):
        """`(inversion, NormalSystem)` for the meshes and samples."""
        mesh = self._joined(meshes)
        g = self._green(mesh, obs)
        inv = SlipInversion(mesh, obs, engine=self.engine, g=g.copy() if self.sees else g, basis="node",
                            ramp=self._ramps(obs), velocity_model=self.velocity_model, sees=self.sees)
        if self.constrain is not None:
            inv = self.constrain(inv, meshes)
        return inv, NormalSystem(inv, **self.recipe.solve)

    def solve(self, inv, lam):
        return inv.solve(smoothing=lam, method="active_set", **self.recipe.solve)

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
        T.flat[:: system.P + 1] += RANK_JITTER * np.trace(T) / system.P
        report = R.resolution_kernels(inv, cho_solve(cho_factor(T, lower=True), D))
        sizes = np.asarray(R.kernel_element_size(inv, lam, fraction=r.fraction, reduce="gaussian",
                                                 size_min=r.size_min, size_max=r.size_max, report=report), float)
        if omega < 1.0:
            current = np.asarray(R.element_size(inv.mesh), float)
            sizes = np.clip(np.exp(omega * np.log(sizes) + (1.0 - omega) * np.log(current)), r.size_min, r.size_max)
        new, a = [], 0
        for sheet, mesh in zip(self.sheets, meshes):
            b = a + mesh.n_elements
            layout = R.adapt_layout(mesh, sizes[a:b], budget=None, balance=MAX_COUNT_RATIO, structure="unstructured")
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
    def field(self, model):
        """The model's east, north and up on the lattice: a half-space forward of the element means on a coarse
        grid, the points within half a step of the outcrop filled from their neighbours, then interpolated."""
        lat, step = self._lattice, self.recipe.field_spacing
        x, y = lat["x"], lat["y"]
        cx = np.arange(x.min() - 2 * step, x.max() + 3 * step, step)
        cy = np.arange(y.min() - 2 * step, y.max() + 3 * step, step)
        gx, gy = np.meshgrid(cx, cy)
        top = model.mesh.nodes[model.mesh.boundary_nodes("top")][:, :2]
        keep = _far_enough(gx.ravel(), gy.ravel(), top, top, 0.5 * step)
        px, py = gx.ravel()[keep], gy.ravel()[keep]
        pool = self._workers()
        if pool is None:
            pts = np.asarray(HalfSpaceTDE(0.25).forward(model.mesh, model.element_slip, px, py), float)
        else:
            edges = np.linspace(0, px.size, 4 * self.recipe.workers + 1).astype(int)
            parts = pool.map(_forward_rows, *zip(*[(model.mesh, model.element_slip, px[a:b], py[a:b])
                                                   for a, b in zip(edges[:-1], edges[1:]) if b > a]))
            pts = np.vstack(list(parts))
        at = np.column_stack([lat["Y"].ravel(), lat["X"].ravel()])
        out = []
        for c in range(3):
            v = np.full(gx.size, np.nan)
            v[keep] = pts[:, c]
            out.append(RegularGridInterpolator((cy, cx), _fill_nearest(v.reshape(gx.shape)), bounds_error=False,
                                               fill_value=np.nan)(at).reshape(lat["X"].shape))
        return out

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
        enu = self.field(model) if model is not None else None
        parts, sigmas = [], {}
        for s in self.series.values():
            los, valid, extra = self._usable(s)
            if valid.sum() < 500:
                self.log(f"    {s.name}: only {int(valid.sum())} usable pixels; skipped")
                continue
            ds = s.grid.ds.copy()
            ds["los"] = (("pair", "y", "x"), np.where(valid, los, np.nan)[None].astype(np.float32))
            if model is None:
                first = {**r.round0[s.kind], **(s.first or {})}
                sigma = first["sigma"] if first.get("sigma") is not None else _scatter(los, valid)
                kw = dict(rms_min=first["rms_min"], width_min=first.get("width_min", r.width_min))
            else:
                prediction = (enu[0] * ds["los_east"].values + enu[1] * ds["los_north"].values
                              + enu[2] * ds["los_up"].values)
                sigma = self._sigma(s, los, valid, extra.get("regions"), prediction, model)
                kw = dict(rms_min=r.split[s.kind] * sigma, width_min=r.width_min,
                          field=np.where(valid, prediction, np.nan))
            obs = Observations.from_los(LosGrid(ds), name=s.name, frame=self.frame, trace=self.sheets[0].trace,
                                        width_max=r.width_max, stat="median", min_valid_fraction=r.min_valid_fraction,
                                        **kw, **extra)
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
        with open(self.out / name, "wb") as fh:
            pickle.dump(dict(meshes=meshes, obs=obs, lam=lam, rank=rank, extra=extra or {}), fh)

    def load(self, name):
        """A stage's checkpoint: `(meshes, observations, weight, rank, extra)`."""
        with open(self.out / name, "rb") as fh:
            z = pickle.load(fh)
        return z["meshes"], z["obs"], z["lam"], z["rank"], z["extra"]

    # -- the run ---------------------------------------------------------------------------------------------------
    def run(self, stages=STAGES):
        """Run `stages` (each skipped when its checkpoint exists); return the final model, or None before `D`."""
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
            f"CV {r.folds} slabs x {r.shifts}; workers {r.workers}")
        model = None
        try:
            if "A" in stages and not (self.out / "A.pkl").exists():
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

            if "B" in stages and not (self.out / "B.pkl").exists():
                meshes, obs, *_ = self.load("A.pkl")
                meshes, ok, rows = self._converge(meshes, obs, r.rank0, "B", r.maps_b)
                pd.DataFrame(rows).to_csv(self.out / "converge_B.csv", index=False)
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(r.rank0)
                keep("B", dict(self.summary(self.solve(inv, lam), system, lam, "B"), converged=ok))
                self._save("B.pkl", meshes, obs, lam, r.rank0, {"converged": ok})

            if "C0" in stages and not (self.out / "C0.pkl").exists():
                meshes, obs, *_ = self.load("B.pkl")
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
                    dslip = float(np.sqrt(np.mean((new.x[:n2] - model.x[:n2]) ** 2))
                                  / np.sqrt(np.mean(new.slip_magnitude ** 2)))
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

            if "C" in stages and not (self.out / "C.pkl").exists():
                meshes, obs, *_ = self.load("C0.pkl")
                inv, system = self.system(meshes, obs)
                self.log("C: blocked cross-validation on mesh B")
                choice = self.cross_validate(system, obs)
                choice.table.to_csv(self.out / "cv_C.csv", index=False)
                (self.out / "cv_C.json").write_text(json.dumps(choice.to_dict(), indent=1, default=float))
                keep("C", self.summary(self.solve(inv, choice.lam), system, choice.lam, f"C (rank {choice.rank:.2f})"))
                self._save("C.pkl", meshes, obs, choice.lam, choice.rank)

            if "D" in stages and not (self.out / "final.pkl").exists():
                meshes, obs, _, rank, _ = self.load("C.pkl")
                meshes, ok, rows = self._converge(meshes, obs, rank, "D", r.maps_d)
                pd.DataFrame(rows).to_csv(self.out / "converge_D.csv", index=False)
                inv, system = self.system(meshes, obs)
                lam = system.lam_for_rank(rank)
                model = self.solve(inv, lam)
                keep("D", dict(self.summary(model, system, lam, f"final (rank {rank:.2f})"), converged=ok))
                self._save("final.pkl", meshes, obs, lam, rank, {"converged": ok})
                model.save(self.out / "final.slip.zip")

            if "E" in stages and (self.out / "final.pkl").exists() and "E" not in record:
                meshes, obs, lam, rank, _ = self.load("final.pkl")
                inv, system = self.system(meshes, obs)
                self.log("E: blocked cross-validation on the final mesh")
                choice = self.cross_validate(system, obs)
                choice.table.to_csv(self.out / "cv_E.csv", index=False)
                inside = choice.inside(rank)
                self.log(f"E: the final rank {rank:.2f} is {'inside' if inside else 'outside'} the final mesh's 1% "
                         "band" + ("s " if len(choice.bands) > 1 else " ")
                         + ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in choice.bands))
                keep("E", dict(rank=rank, cv_rank=choice.rank, bands=choice.bands, inside=bool(inside)))

            if "report" in stages and (self.out / "final.pkl").exists():
                meshes, obs, lam, rank, extra = self.load("final.pkl")
                inv, system = self.system(meshes, obs)
                model = self.solve(inv, lam)
                s = self.summary(model, system, lam, "report")
                model.save(self.out / "model.slip.zip")
                counts = {t: int(np.sum(np.asarray(obs.ds["track"].values).astype(str) == t)) for t in obs.tracks}
                recipe = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(r).items()}
                (self.out / "summary.json").write_text(json.dumps(dict(
                    recipe=recipe, sheets=[dict(name=sh.name, dip=sh.dip, depth=sh.depth, length=sh.length,
                                                listric=sh.profile is not None) for sh in self.sheets],
                    **s, rows_by_track=counts, converged=extra.get("converged"), stages=record),
                    indent=1, default=float))
            elif (self.out / "final.slip.zip").exists() and model is None:
                model = SlipModel.load(self.out / "final.slip.zip")
        finally:
            self.close()
        self.log("done")
        return model
