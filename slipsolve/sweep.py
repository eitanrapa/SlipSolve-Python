"""Segment-by-segment dip sweeps: which dip does each stretch of the fault want?

A geometry sweep lets the data choose the fault's shape. `DipSweep` cuts the
trace into along-strike segments, gives each its own geometry from a grid -- a
planar dip (10 or 5 degree steps), or a dip that changes with depth -- and scores
every combination. Every combination is solved on the same samples with the same
regularization, so the geometry is the only thing that differs between them.

**Two ways to score.** `run(score="evidence")` integrates the slip, the ramp,
the noise scale and the smoothing weight out analytically
(`slipsolve.evidence`), so no rank has to be chosen. `run(rank=...)`
scores the bounded solve's misfit at a fixed effective rank. A fixed rank is
not neutral for geometry: held well below the rank the data support, the rms
can prefer a dip a few degrees off even with an exact forward model, where the
evidence does not.

**Uncertainty comes from** `DipSweep.bootstrap`, not from the evidence.
The iid evidence ranks geometries robustly, but the posterior it implies is
several times too narrow. Its estimate re-made on synthetic data drawn from a
fitted noise model gives intervals with roughly their nominal coverage.
Weighting the evidence by the noise covariance is exact when the forward model
is, and can be degrees wrong when it is not. The bootstrap reports how far the
weighted and unweighted estimates disagree; a large disagreement is the sign of
that failure.

**Why it is affordable.** Every engine builds a node's Green's function as a sum
over the elements around it, each element computed from its own vertices alone
(`slipsolve.layered_tde.LayeredTDE.tent_field` integrates each tent; the half-space
engine projects per-element columns onto the tents). With a node column on every segment
boundary no element straddles one, so the design matrix of any combination is a
sum of per-segment blocks, each depending on one segment's geometry:

    G(dips) = sum_j  P_j G_j(dip_j)

and the sweep costs one full assembly per geometry *value* -- seven for 60-120
degrees in steps of 10 -- instead of one per combination, of which there are
`7**K`. After that a combination costs one solve. `workers` spreads each assembly
over processes, and `cache` keeps the blocks on disk for the next run.

**What the geometry is.** Each segment is what `FaultMesh.curved` builds for
it: every node stepped along the trace's local normal by `depth / tan(dip)`, or,
for a dip that changes with depth, by the integral of `cot(dip)` down to that
depth; with `normal_smoothing`, along the trace's normal smoothed along strike, as
`slipsolve.production.Sheet` steps it. Adjacent segments meet on the surface trace
and step apart below it, by
`depth * |cot(dip_a) - cot(dip_b)|` -- 3.5 km at 20 km depth for 80 against 90.
The slip does not step: boundary nodes are shared parameters, so the slip field
stays continuous along strike and only the geometry kinks. The fault is torn,
not a blended surface, so rebuild the winning combination as one surface before
reporting a model on it: `ProfileFit.sheet` builds a production `Sheet` with the
dips blended across the breaks, and `FaultMesh.curved(surface=...)` takes any
surface.

**What is held fixed.** The samples, the `(s, z)` node layout, the bounds, the
ramp columns, the boundary rows and the smoothing operator. The operator is built
once, on the layout embedded at `reference_dip`, so the regularization is
identical across geometries rather than re-derived from each one's 3-D metric.

**Continuous profiles.** `DipSweep.optimize` maximises the evidence over each
segment's dip profile without a grid: a few dips at evenly spaced depths per
segment, linear in depth between them. It is a trust-region search on the design
linearised in those dips. Because the design is a sum of per-segment blocks, a dip's
derivative costs two assemblies of its own segment, and the evidence's gradient with
respect to the design is analytic (`GeometryEvidence.gradient`). Start it from a sweep
(it is local), add depth knots one at a time, and take intervals from
`DipSweep.bootstrap`, which re-estimates the dips on each replicate by one Newton
step on the linearised design. A dip that changes with depth is resolved far more
coarsely than a planar dip, and extra knots can fit correlated noise; the guide's
dip-sweep page has the numbers.

Read the result as a profile, not a winner: a sweep can reject one dip direction
clearly while its curve is shallow among the angles on the other side.
`SweepResult.profiles` -- each segment's score against its dip, the others held
at the best -- says more than the argmin. And with one-sided coverage a dip
preference can be a coverage artefact: recover a planted geometry through the
same samples (`DipSweep.predict` and `data=`) before believing one.

**A rigidity contrast as a sweep axis.** `contrast=` (a
`slipsolve.bimaterial.RigidityContrast`) and `ratios=` add one coordinate
per interval of the contrast, after the segments' dips. The scaled mode makes a
contrast a row and column scaling of the same blocks, so a ratio costs no
assembly, only a solve or a decomposition. Dip and contrast both make the fault's
two sides move differently, and with one-sided data they can substitute for each
other; sweep them together. The scaled mode is an approximation the data can
see: given exact-mode data, a scaled sweep can misread both the dip and the
ratio, and it smears a step along strike (see `slipsolve.bimaterial`). Read its
answer as where to look, and take the dip, ratio and moment from
`Bimaterial(exact=True)`.
"""

import itertools
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize

from .basis import NodeBasis
from .covariance import CovarianceModel, fit_reml, observation_covariance
from .evidence import GeometryEvidence, smoothing_precision
from .greens import HalfSpaceTDE
from .inversion import DEFAULT_SHEAR_MODULUS, SlipModel, _boundary_rows, box_qp
from .layout import graded_layout
from .mesh import FaultMesh, _arc_length, _depth_levels, _layout_attrs, _snap_vertical_columns
from .regularize import (check_ramp_tracks, laplace_beltrami, ramp_columns, ramp_spec,
                         slip_bounds)
from .resolution import JITTER
from .surface import FaultSurface, centres
from .trace import _smoothed_normals, dip_offset

#: Dips tried on every segment by default, in degrees: 60 to 120 in steps of 10.
#: `FaultMesh.curved(uniform_dip=)`'s convention -- below 90 the fault dips
#: toward the trace's left-hand side, above 90 toward its right. Read
#: `mesh.dip_direction` to see which way that is on a given trace (on a trace
#: drawn from west to east, 80 dips north and 100 dips south).
DEFAULT_DIPS = tuple(range(60, 121, 10))

#: The scores `DipSweep.run` can rank combinations by.
SCORES = ("rms", "weighted_rms", "evidence")

#: A combination has to beat the current one by more than this fraction of its
#: score to replace it in the coordinate search. Without it, two geometries equal
#: to rounding could swap forever.
IMPROVEMENT = 1e-9

#: Resolution, in degrees, at which a segment's continuous dip estimate is read
#: off the spline through its profile.
REFINE_STEP = 0.005

#: `DipSweep.optimize` stops when the linearised design promises less than this gain in
#: log evidence (natural units).
GAIN_TOLERANCE = 1e-4

#: The largest trust radius `DipSweep.optimize` grows to, in degrees.
MAX_RADIUS = 20.0

#: Step, in degrees, of the finite differences that give the Hessian of the linearised design.
HESSIAN_STEP = 1e-3

#: A bootstrap replicate within this many degrees of a bound counts as stopped on it.
BOUND_TOLERANCE = 1e-6


def geometry_key(value):
    """One segment's geometry, as the sweep keys it.

    A number is a planar dip in degrees. A sequence of two or more numbers is a
    dip that changes with depth: the dips at evenly spaced depths from the top of
    the layout to the sweep's `knot_depth` (its bottom by default), linear in
    between and held below (see `dip_profile_offsets`).
    A sequence whose entries are all equal is the planar dip and is returned as
    one, so the two spellings share a mesh and a Green's-function block.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) or np.ndim(value) == 0:
        key = float(value)
        values = (key,)
    else:
        key = tuple(float(v) for v in value)
        values = key
        if len(key) < 2:
            raise ValueError(f"a dip profile needs at least two dips, got {value!r}")
    for v in values:
        if not 0.0 < v < 180.0:
            raise ValueError(f"dips must lie strictly between 0 and 180 degrees, got {v}")
    if isinstance(key, tuple) and all(v == key[0] for v in key):
        return key[0]
    return key


def geometry_label(key):
    """`70` for a planar dip, `70>50` for one that runs from 70 to 50 with depth."""
    return f"{key:g}" if isinstance(key, float) else ">".join(f"{v:g}" for v in key)


def dip_profile_offsets(depths, dips, top=None, bottom=None):
    """Horizontal offset at each depth for a dip that changes with depth.

    `dips` are the dips (degrees) at evenly spaced depths from `top` to
    `bottom` (by default the shallowest and deepest of `depths`), linear in
    depth between them and held below `bottom`. The offset is the integral of
    `cot(dip)` from `top`, taken exactly on each piece (`ln sin` is its
    antiderivative), so a constant profile reproduces `depth / tan(dip)` and the
    near-vertical snapping of `slipsolve.trace.dip_offset` is left to the mesh
    builder.
    """
    depths = np.asarray(depths, dtype=float)
    theta = np.radians(np.asarray(dips, dtype=float))
    top = float(depths.min() if top is None else top)
    bottom = float(depths.max() if bottom is None else bottom)
    knots = np.linspace(top, bottom, theta.size)
    out = np.zeros(depths.shape)
    for i, depth in enumerate(depths.ravel()):
        total = 0.0
        for a, b, ta, tb in zip(knots[:-1], knots[1:], theta[:-1], theta[1:]):
            hi = min(b, depth)
            if hi <= a:
                break
            slope = (tb - ta) / (b - a)
            if abs(slope) * (hi - a) < 1e-12:
                total += (hi - a) * np.cos(ta) / np.sin(ta)
            else:
                total += (np.log(np.sin(ta + slope * (hi - a))) - np.log(np.sin(ta))) / slope
        if depth > bottom:                       # below the last knot the dip holds
            total += (depth - bottom) * np.cos(theta[-1]) / np.sin(theta[-1])
        out.flat[i] = total
    return out


def _project_arc(px, py, s, x, y):
    """The arc length, on the polyline `(x, y)` measured by `s`, of the point nearest each `(px, py)`."""
    s, x, y = (np.asarray(v, dtype=float) for v in (s, x, y))
    dx, dy = np.diff(x), np.diff(y)
    length2 = np.where(dx ** 2 + dy ** 2 > 0.0, dx ** 2 + dy ** 2, 1.0)
    out = []
    for qx, qy in zip(np.atleast_1d(px), np.atleast_1d(py)):
        t = np.clip(((qx - x[:-1]) * dx + (qy - y[:-1]) * dy) / length2, 0.0, 1.0)
        i = int(np.argmin((x[:-1] + t * dx - qx) ** 2 + (y[:-1] + t * dy - qy) ** 2))
        out.append(s[i] + t[i] * (s[i + 1] - s[i]))
    return np.array(out, dtype=float)


def _knot_key(values):
    """Knot dips as a geometry key: a number for one knot, a tuple otherwise."""
    values = [float(v) for v in values]
    return geometry_key(values[0] if len(values) == 1 else tuple(values))


def segment_breaks(length, count=None, size=None, breaks=None):
    """Arc lengths of the segment boundaries, `0 ... length`, in metres.

    Give exactly one of `count` (that many equal segments), `size` (segments
    of about that length) or `breaks` (the interior boundaries; `0` and
    `length` may be included or left out).
    """
    length = float(length)
    if sum(v is not None for v in (count, size, breaks)) != 1:
        raise ValueError("give exactly one of count=, size= or breaks=")
    if breaks is not None:
        values = np.asarray(breaks, dtype=float).ravel()
        inner = values[(values != 0.0) & (values != length)]
        if np.any((inner <= 0.0) | (inner >= length)):
            raise ValueError(f"breaks must lie on the trace, between 0 and {length:.0f} m")
        return np.unique(np.concatenate([[0.0], inner, [length]]))
    if size is not None:
        if size <= 0:
            raise ValueError("size must be positive")
        count = max(1, int(round(length / float(size))))
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    return np.linspace(0.0, length, int(count) + 1)


def segmented_layout(breaks, depths, edge_length):
    """A node lattice with a column on every segment boundary.

    Each segment is divided evenly into steps of at most `edge_length`, so
    every boundary is a lattice column and no triangle straddles one -- the
    property the block assembly in `DipSweep` rests on.
    """
    breaks = np.asarray(breaks, dtype=float)
    parts = [np.linspace(a, b, max(1, int(np.ceil((b - a) / edge_length))) + 1)[:-1]
             for a, b in zip(breaks[:-1], breaks[1:])]
    stations = np.concatenate(parts + [breaks[-1:]])
    return graded_layout(breaks[-1], depths, counts=stations.size, stations=stations)


def _box_qp(N, g, lo, hi, x0, max_iter=100):
    """Minimise `x'Nx/2 - g'x` under `lo <= x <= hi`, exactly; the minimiser.

    `slipsolve.inversion.box_qp`, which `SlipInversion.solve(method="active_set")`
    uses too. Started from a nearby solution, as in a sweep, it settles in a few
    steps.
    """
    return box_qp(N, g, lo, hi, x0, max_iter).x


def _better(a, b):
    """Whether score `a` beats `b` by more than `IMPROVEMENT` of its size."""
    return a < b - IMPROVEMENT * abs(b)


def _ratio_grids(contrast, ratios):
    """One grid of ratios per interval of `contrast`: shared, per interval, or its own."""
    if contrast is None:
        return []
    m = contrast.n_intervals
    if ratios is None:
        grids = [[float(r)] for r in contrast.ratio]
    elif all(np.ndim(v) == 0 for v in ratios):
        grids = [[float(v) for v in ratios] for _ in range(m)]
    else:
        if len(ratios) != m:
            raise ValueError(f"ratios has {len(ratios)} grids for {m} contrast intervals; give "
                             "one grid for all, or one per interval")
        grids = [[float(v) for v in np.atleast_1d(g)] for g in ratios]
    for g in grids:
        if not g or any(not np.isfinite(v) or v <= 0 for v in g):
            raise ValueError(f"ratios must be positive, got {g}")
    return [list(dict.fromkeys(g)) for g in grids]


def _evidence_cache_key(covariance):
    if covariance is None or isinstance(covariance, str):
        return ("plain", covariance)
    if isinstance(covariance, CovarianceModel):
        return ("model", covariance)
    if isinstance(covariance, dict) and all(isinstance(v, CovarianceModel)
                                            for v in covariance.values()):
        return ("models", tuple(sorted(covariance.items())))
    return ("object", id(covariance))


def _combinations(candidates):
    """How many combinations `candidates` (one list per axis) makes, exactly: numpy's int64 wraps at ~9e18,
    and a wrapped count would send `run(method="auto")` to search them all."""
    return math.prod(len(c) for c in candidates)


class DipSweep:
    """Sweep the geometry of each along-strike segment over a grid.

    `obs` is the fixed sample set every geometry is scored on. The segments are
    `segments` equal pieces of the trace (4 if nothing is said),
    `segment_length` metres each, or cut at `breaks` (arc lengths in metres).
    `dips` is the grid every segment draws from: planar dips, and tuples for a
    dip that changes with depth (see `geometry_key`). `max_depth`,
    `edge_length`, `down_dip_levels`, `bias_w` and `depths` place the
    nodes as in `FaultMesh.curved`. `engine`, `ramp`, `velocity_model` and
    `frame_looks` mean what they mean to `SlipInversion`, and the
    remaining keywords are `SlipInversion.solve`'s regularization and
    bounds; the slip is always nodal.

    `reference_dip` is the geometry the smoothing operator is built on, once,
    for every combination. 90 measures it in the `(s, z)` plane itself.

    `knot_depth` (m) is where a profile's deepest dip sits; below it the dip
    holds. By default it is the base of the layout. Set it near the base of the
    slip: a knot below the slip has little to fit, and moves with the noise.

    `contrast` puts the fault between blocks of different rigidity, in the
    scaled mode of `slipsolve.bimaterial`. `ratios` makes its ratio a
    sweep axis: one grid for every interval of the contrast, or one grid per
    interval (a one-value grid holds that interval fixed). Without `ratios`
    the contrast's own ratios hold. A combination is then the segments' dips
    followed by the intervals' ratios; anywhere a combination is taken, the dips
    alone mean the contrast's own ratios.

    `normal_smoothing` (m) embeds the layout as `slipsolve.production.Sheet` does:
    each node is moved along the trace's left-hand normal smoothed along strike by a
    Gaussian of that standard deviation (0 for none), rather than along the trace's own
    local normal as `FaultMesh.curved` moves it (None, the default). On a trace with
    bends a fault that lies far from its trace at depth can fold with the local normal;
    and to sweep the surface a production run will mesh, give its sheets' value
    (10 km by default). Either way the trace is followed as resampled at `edge_length`.

    Green's functions are assembled lazily, one geometry at a time, and kept:
    memory is about one full design matrix per geometry used. `workers` above 1
    assembles each block's rows on that many spawned processes, each with its own
    copy of the engine (a `LayeredTDE` needs its tables loaded from a file); `close`
    shuts them down. `cache`, a directory, keeps every block assembled -- `optimize`'s
    finite-difference geometries too -- in a file named by the segment mesh, the
    samples and the engine, so a repeated run, or another sweep on the same samples,
    reads it back.
    """

    def __init__(self, trace, obs, *, frame=None, segments=None, segment_length=None,
                 breaks=None, dips=DEFAULT_DIPS, max_depth=20e3, edge_length=3e3,
                 down_dip_levels=None, bias_w=1.0, depths=None, engine=None,
                 ramp="none", reference_dip=90.0, velocity_model=None, ss_ratio=1.0,
                 ds_ratio=3.0, boundary_ratio=1.0, sides=("bottom", "left", "right"),
                 polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0),
                 contrast=None, ratios=None, frame_looks=False, knot_depth=None,
                 normal_smoothing=None, workers=1, cache=None):
        if hasattr(engine, "contrast"):
            raise ValueError("give the rigidity contrast to DipSweep(contrast=...), with a plain "
                             "engine, so the sweep can scale its blocks; not a Bimaterial engine")
        if ratios is not None and contrast is None:
            raise ValueError("ratios= sweeps a contrast's ratio; give contrast= too")
        if contrast is not None and contrast.n_bands > 1:
            raise NotImplementedError(
                "DipSweep sweeps one ratio per interval along strike; a contrast with depth_breaks "
                "is not built into the sweep. Use Bimaterial(contrast, exact=True) at fixed ratios")
        self.trace, self.obs = trace, obs
        self.frame = frame or trace.local_frame()
        self.edge_length = float(edge_length)
        # The arc length exactly as `FaultMesh.curved` measures it, or the layout
        # and the meshes built on it would disagree about where the trace ends.
        resampled = trace.resample(self.edge_length, self.frame)
        self._rx, self._ry = resampled.to_local(self.frame)
        self._s = _arc_length(self._rx, self._ry)
        self.length = float(self._s[-1])
        if normal_smoothing is not None and not float(normal_smoothing) >= 0.0:
            raise ValueError(f"normal_smoothing must be a length >= 0 (m) or None, got {normal_smoothing!r}")
        self.normal_smoothing = None if normal_smoothing is None else float(normal_smoothing)
        self._sheet_normals = (None if self.normal_smoothing is None
                         else _smoothed_normals(self._rx, self._ry, self._s, self.normal_smoothing))
        if int(workers) < 1:
            raise ValueError(f"workers must be at least 1, got {workers!r}")
        self.workers = int(workers)
        self.cache = cache
        self._process_pool = None
        if segments is None and segment_length is None and breaks is None:
            segments = 4
        self.breaks = segment_breaks(self.length, count=segments, size=segment_length,
                                     breaks=breaks)
        if depths is None:
            levels = (down_dip_levels if down_dip_levels is not None
                      else max(1, int(round(max_depth / edge_length))) + 1)
            depths = _depth_levels(0.0, float(max_depth), int(levels), float(bias_w))
        self.depths = np.asarray(depths, dtype=float)
        top, base = float(self.depths.min()), float(self.depths.max())
        self.knot_depth = base if knot_depth is None else float(knot_depth)
        if not top < self.knot_depth <= base:
            raise ValueError(f"knot_depth must lie below the top of the layout and no deeper than "
                             f"its base ({base:.0f} m), got {self.knot_depth:.0f}")
        self.layout = segmented_layout(self.breaks, self.depths, self.edge_length)
        self.dips = tuple(dict.fromkeys(geometry_key(v) for v in dips))
        self.engine = HalfSpaceTDE() if engine is None else engine
        self.ramp_kind = ramp_spec(ramp)
        self.velocity_model = velocity_model
        self.frame_looks = bool(frame_looks)
        self.reference_dip = float(reference_dip)
        self.sides = tuple(sides)
        self._meshes, self._submeshes, self._blocks = {}, {}, {}
        self._memo, self._evidence = {}, {}
        self._last = {"lam": None, "x": None}

        self.reference = self._mesh(self.reference_dip)
        self.n = int(self.reference.n_nodes)
        s = np.asarray(self.reference.element_params, dtype=float)[:, 0]
        seg = np.clip(np.searchsorted(self.breaks, s, side="right") - 1, 0, self.n_segments - 1)
        self.segment_elements = [np.flatnonzero(seg == j) for j in range(self.n_segments)]
        self.segment_nodes = [np.unique(np.asarray(self.reference.triangles)[e])
                              for e in self.segment_elements]

        ds = obs.ds
        self.d = np.array(ds["los"].values, dtype=float)
        self.w = np.array(ds["weight"].values, dtype=float)
        self.track = np.asarray(ds["track"].values).astype(str)
        check_ramp_tracks(self.ramp_kind, obs.tracks)
        self.ramp, self.ramp_labels = ramp_columns(obs, self.ramp_kind)

        self.options = dict(ss_ratio=float(ss_ratio), ds_ratio=float(ds_ratio),
                            boundary_ratio=float(boundary_ratio), sides=list(sides),
                            polarity=None if polarity is None else list(polarity),
                            strike=list(strike), dip=list(dip))
        self._smooth = laplace_beltrami(self.reference, ss_ratio, ds_ratio)
        self._rows = self._smooth.shape[0]
        self._SS = (self._smooth.T @ self._smooth).toarray()
        bnd = _boundary_rows(NodeBasis(self.reference), tuple(sides), boundary_ratio)
        self._BB = (bnd.T @ bnd).toarray()
        lo, hi = slip_bounds(self.n, strike, dip, polarity)
        k = self.ramp.shape[1]
        self._lo = np.concatenate([lo, np.full(k, -np.inf)])
        self._hi = np.concatenate([hi, np.full(k, np.inf)])

        self.contrast = contrast
        self.ratio_grids = _ratio_grids(contrast, ratios)
        self._factors = {}
        if contrast is not None:
            contrast.frame.require_match(self.frame, "The rigidity contrast")
            x, y = ds["x"].values, ds["y"].values
            self._obs_left = contrast.trace.side(x, y, contrast.frame) > 0
            self._obs_interval = contrast.interval(
                contrast.trace.to_curvilinear(x, y, contrast.frame)[0])
            # Arc lengths are the layout's, the same at every geometry.
            self._element_interval = contrast.interval(self.reference.element_params[:, 0])
            tent = NodeBasis(self.reference)
            self._tent = (tent.projection(), self.reference.areas, tent.lumped_areas())

    # -- geometry -----------------------------------------------------------
    @property
    def n_segments(self):
        """Number of segments, each with its own dip."""
        return int(self.breaks.size - 1)

    @property
    def n_ratios(self):
        """Coordinates after the dips: one per interval of the contrast, or none."""
        return len(self.ratio_grids)

    @property
    def _default_ratios(self):
        """What a combination given as dips alone means for the contrast's ratios."""
        return tuple(float(g[0]) if len(g) == 1 else float(r)
                     for g, r in zip(self.ratio_grids,
                                     [] if self.contrast is None else self.contrast.ratio))

    def label(self, dips):
        """A combination as text: `80 70` for the dips, then `| r 1.5` for any ratios."""
        combo = self._combination(dips)
        k = self.n_segments
        text = " ".join(geometry_label(v) for v in combo[:k])
        if self.n_ratios:
            text += " | r " + " ".join(f"{v:g}" for v in combo[k:])
        return text

    @property
    def segments(self):
        """One row per segment: its index and arc-length extent in km."""
        return pd.DataFrame({"segment": np.arange(1, self.n_segments + 1),
                             "start_km": self.breaks[:-1] / 1e3,
                             "end_km": self.breaks[1:] / 1e3})

    def _profile_surface(self, key):
        """The `cross(s, z)` field of a dip that changes with depth, on the mesh lattice."""
        z_nodes = -self.depths[::-1]
        top, bottom = float(self.depths.min()), self.knot_depth
        offsets = dip_profile_offsets(self.depths, key, top, bottom)
        cross = np.broadcast_to(offsets[::-1, None], (self.depths.size, self._s.size))
        s_c, z_c = centres(self._s), centres(z_nodes)
        centre = dip_profile_offsets(-z_c, key, top, bottom)
        return FaultSurface(self._s, z_nodes, np.array(cross), s_c, z_c,
                            np.array(np.broadcast_to(centre[:, None], (z_c.size, s_c.size))),
                            attrs={"dips": [float(v) for v in key], "profile": "linear in depth",
                                   "knot_depth": bottom})

    def _mesh(self, key):
        """The whole layout embedded at one geometry."""
        key = geometry_key(key)
        if key not in self._meshes:
            if self._sheet_normals is None:
                geometry = ({"uniform_dip": key} if isinstance(key, float)
                            else {"surface": self._profile_surface(key)})
                self._meshes[key] = FaultMesh.from_layout(
                    self.layout, self.trace, self.frame, edge_length=self.edge_length, **geometry)
            else:
                self._meshes[key] = self._smoothed_mesh(key)
        return self._meshes[key]

    def _smoothed_mesh(self, key):
        """The layout at one geometry, each node moved along the smoothed normal by the geometry's offset.

        As `slipsolve.production.Sheet.mesh` places nodes, with the offsets the
        segments' blocks are keyed by: at the layout's depths, near-vertical steps
        snapped as `FaultMesh.curved` snaps them.
        """
        nx, ny = self._sheet_normals
        params = np.asarray(self.layout.params, dtype=float)
        s, depth = params[:, 0], -params[:, 1]
        levels = _snap_vertical_columns(self._offsets(key)[:, None], self.depths)[:, 0]
        step = np.interp(depth, self.depths, levels)
        nodes = np.column_stack([np.interp(s, self._s, self._rx) + step * np.interp(s, self._s, nx),
                                 np.interp(s, self._s, self._ry) + step * np.interp(s, self._s, ny),
                                 params[:, 1]])
        attrs = {"kind": "sweep", "max_depth": float(self.depths.max()),
                 "top_depth": float(self.depths.min()), "edge_length": self.edge_length,
                 "trace_length": self.length, "trace_name": self.trace.name,
                 "normal_smoothing": self.normal_smoothing, **_layout_attrs(self.layout)}
        if isinstance(key, float):
            attrs["dip_deg"] = key
        else:
            attrs.update(dips=[float(v) for v in key], profile="linear in depth",
                         knot_depth=self.knot_depth)
        mesh = FaultMesh(nodes, self.layout.triangles, params, frame=self.frame, attrs=attrs)
        mesh._check_winding((self._s, nx, ny))
        return mesh

    def segment_mesh(self, j, dip):
        """Segment `j` (0-based) of the layout, embedded at `dip`.

        Its nodes are `segment_nodes[j]` of the global parameter vector, in
        that order; a boundary node appears in both segments it bounds.
        """
        key = (int(j), geometry_key(dip))
        if key not in self._submeshes:
            mesh = self._mesh(key[1])
            nodes = self.segment_nodes[j]
            remap = np.full(mesh.n_nodes, -1)
            remap[nodes] = np.arange(nodes.size)
            triangles = remap[np.asarray(mesh.triangles)[self.segment_elements[j]]]
            self._submeshes[key] = FaultMesh(
                np.asarray(mesh.nodes)[nodes], triangles, np.asarray(mesh.params)[nodes],
                frame=self.frame,
                attrs={**mesh.attrs, "layout": "sweep segment", "segment": int(j)})
        return self._submeshes[key]

    def segment_meshes(self, dips):
        """Every segment at its own geometry, in order -- the stepped fault `dips` describes."""
        return [self.segment_mesh(j, v) for j, v in enumerate(self._combination(dips))]

    def _offsets(self, key):
        """A geometry's horizontal offset from the trace at each of the layout's depths, m."""
        key = geometry_key(key)
        if isinstance(key, tuple):
            return dip_profile_offsets(self.depths, key, float(self.depths.min()), self.knot_depth)
        return np.asarray(dip_offset(self.depths, key), dtype=float)

    def tears(self, dips):
        """How far apart neighbouring segments' surfaces lie where they meet, `(breaks, depths)`, m.

        Row `b` is segment `b`'s offset from the trace less segment `b + 1`'s, at each
        of the layout's `depths`, on the column at `breaks[b + 1]` that both share.
        Each segment is meshed at its own geometry, so the stepped fault tears there
        unless the two geometries agree; the shared column's nodes carry one slip at
        both positions. `optimize(continuity=)` penalises the tear.
        """
        combo = self._combination(dips)[: self.n_segments]
        offsets = [self._offsets(v) for v in combo]
        return np.array([offsets[j] - offsets[j + 1] for j in range(self.n_segments - 1)]
                        ).reshape(self.n_segments - 1, self.depths.size)

    def _combination(self, dips):
        if isinstance(dips, (int, float, np.integer, np.floating)):
            dips = (dips,)
        dips = tuple(dips)
        k, m = self.n_segments, self.n_ratios
        if m and len(dips) == k + m:
            geometry, ratios = dips[:k], tuple(float(v) for v in dips[k:])
        else:
            geometry, ratios = dips, self._default_ratios
        geometry = tuple(geometry_key(v) for v in geometry)
        if len(geometry) == 1 and k > 1:
            geometry = geometry * k
        if len(geometry) != k:
            more = f", then optionally one ratio per contrast interval ({m})" if m else ""
            raise ValueError(
                f"need one dip per segment ({k}){more}, got {len(dips)} entries; a dip that "
                "changes with depth is ONE entry, a tuple -- e.g. [(70, 50)] for one segment")
        if any(not np.isfinite(r) or r <= 0 for r in ratios):
            raise ValueError(f"ratios must be positive, got {ratios}")
        return geometry + ratios

    # -- the design matrix ---------------------------------------------------
    def _assemble(self, j, dip):
        """Segment `j`'s nodal Green's columns at geometry `dip`, `(n_obs, 2 n_j)`."""
        from .production import _cached_green

        g, _ = _cached_green(self.segment_mesh(j, dip), self.obs, self.engine,
                             frame_looks=self.frame_looks, cache=self.cache, pool=self._processes,
                             workers=self.workers)
        return g

    def _processes(self):
        """The process pool, started on first use when `workers` is above 1."""
        if self.workers < 2:
            return None
        if self._process_pool is None:
            from .production import _worker_pool

            self._process_pool = _worker_pool(self.engine, self.workers)
        return self._process_pool

    def close(self):
        """Shut the worker processes down; a later assembly starts them again."""
        if self._process_pool is not None:
            self._process_pool.shutdown()
            self._process_pool = None

    def __getstate__(self):
        # A process pool cannot be pickled; an unpickled sweep starts its own on first use.
        return {**self.__dict__, "_process_pool": None}

    def _segment_block(self, j, dip):
        """`_assemble`, kept: one block per segment and geometry used."""
        key = (int(j), geometry_key(dip))
        if key not in self._blocks:
            self._blocks[key] = self._assemble(*key)
        return self._blocks[key]

    def _transient_block(self, j, dip):
        """`_assemble` without keeping it or the meshes it builds: `optimize` visits each geometry once."""
        key = geometry_key(dip)
        kept = (key in self._meshes, (int(j), key) in self._submeshes)
        try:
            return self._assemble(int(j), key)
        finally:
            if not kept[1]:
                self._submeshes.pop((int(j), key), None)
            if not kept[0]:
                self._meshes.pop(key, None)

    def _compose(self, blocks, ratios):
        """The nodal design from per-segment blocks `{segment: block}`, scaled by any contrast."""
        g = np.zeros((self.d.size, 2 * self.n))
        for j, block in blocks.items():
            nodes = self.segment_nodes[j]
            g[:, np.concatenate([nodes, self.n + nodes])] += block
        if self.n_ratios:
            rows, columns, _ = self._contrast_factors(ratios)
            g *= rows[:, None]
            g *= columns[None, :]
        return g

    def design(self, dips):
        """The nodal Green's matrix of a combination, `(n_obs, 2 * n)`.

        For one geometry on every segment it is exactly the matrix
        `SlipInversion` assembles on that mesh, and with a contrast, the
        matrix a scaled-mode `slipsolve.bimaterial.Bimaterial` would.
        """
        combo = self._combination(dips)
        k = self.n_segments
        return self._compose({j: self._segment_block(j, dip) for j, dip in enumerate(combo[:k])},
                             combo[k:])

    def _contrast_factors(self, ratios):
        """Row factors, nodal column factors and element rigidity factors at `ratios`.

        The scaled mode of `slipsolve.bimaterial`: each sample's row times
        its block's `a`, each element's slip times `1/m`. A node takes its
        tent's area-weighted mean of `1/m`, measured on the reference mesh.
        """
        key = tuple(float(r) for r in ratios)
        if key not in self._factors:
            contrast = self.contrast.with_ratio(key)
            a_left, a_right = contrast._left_right(np.arange(contrast.n_intervals))
            rows = np.where(self._obs_left, a_left[self._obs_interval],
                            a_right[self._obs_interval])
            element = (2.0 / (a_left + a_right))[self._element_interval]
            projection, areas, lumped = self._tent
            node = (projection.T @ (areas * element)) / lumped
            self._factors[key] = (rows, np.concatenate([node, node]), element)
        return self._factors[key]

    def _full_design(self, dips):
        g = self.design(dips)
        return np.hstack([g, self.ramp]) if self.ramp.shape[1] else g

    def predict(self, dips, x):
        """What parameters `x` (slip, then any ramp) predict on geometry `dips`.

        With noise added, the `data=` a planted-geometry recovery test needs.
        """
        return self._full_design(dips) @ np.asarray(x, dtype=float)

    # -- the bounded solve ---------------------------------------------------
    def _damping(self, lam):
        size = 2 * self.n + self.ramp.shape[1]
        out = np.zeros((size, size))
        m = 2 * self.n
        out[:m, :m] = (float(lam) / self._rows) ** 2 * self._SS + self._BB
        return out

    def _rank(self, N, lam):
        """Effective rank: the trace of the slip block of `(N + damping)^-1 N`."""
        total = N + self._damping(lam)
        total.flat[:: total.shape[0] + 1] += JITTER * (np.trace(N) or np.trace(total)) / total.shape[0]
        m = 2 * self.n
        columns = cho_solve(cho_factor(total, lower=True), np.eye(total.shape[0])[:, :m])
        return float(np.sum(columns * N[:, :m]))

    def _lam_for_rank(self, N, rank, guess=None):
        """The smoothing weight giving effective rank `rank`, by Brent in log-weight."""
        def gap(u):
            return self._rank(N, np.exp(u)) - rank

        centre = np.log(guess) if guess else 0.0
        lo, hi = centre - 1.0, centre + 1.0
        f_lo, f_hi = gap(lo), gap(hi)
        for _ in range(40):
            if f_lo >= 0.0 >= f_hi:
                break
            if f_lo < 0.0:          # still too smooth at the light end: go lighter
                hi, f_hi, lo = lo, f_lo, lo - 2.0
                f_lo = gap(lo)
            else:                   # still too rough at the heavy end: go heavier
                lo, f_lo, hi = hi, f_hi, hi + 2.0
                f_hi = gap(hi)
        else:
            raise ValueError(f"effective rank {rank} is out of this problem's reach")
        return float(np.exp(brentq(gap, lo, hi, xtol=1e-6)))

    def _solve(self, N, g, lam):
        total = N + self._damping(lam)
        total.flat[:: total.shape[0] + 1] += JITTER * (np.trace(N) or np.trace(total)) / total.shape[0]
        x0 = self._last["x"]
        if x0 is None or x0.size != total.shape[0]:
            x0 = np.zeros(total.shape[0])
        x = _box_qp(total, g, self._lo, self._hi, x0)
        self._last["x"] = x
        return x

    def _moment_magnitude(self, dips, x):
        """Integrated Mw, summed over the segments' own meshes.

        `SlipModel.moment` integrates whatever exposes a mesh, a basis and the
        nodal slip, so each segment is handed to it as that and the moments
        added: the elements are partitioned, so nothing is counted twice, and the
        rule stays the package's own rather than a copy of it.
        """
        rigidity = (self.velocity_model if self.velocity_model is not None
                    else DEFAULT_SHEAR_MODULUS)
        combo = self._combination(dips)
        k = self.n_segments
        element = self._contrast_factors(combo[k:])[2] if self.n_ratios else None
        total = 0.0
        for j, dip in enumerate(combo[:k]):
            mesh = self.segment_mesh(j, dip)
            nodes = self.segment_nodes[j]
            view = SimpleNamespace(
                basis=NodeBasis(mesh), mesh=mesh, strike_slip=x[nodes], dip_slip=x[self.n + nodes],
                rigidity_factor=None if element is None else element[self.segment_elements[j]])
            total += SlipModel.moment(view, shear_modulus=rigidity)
        return float((2.0 / 3.0) * (np.log10(total) - 9.1)) if total > 0 else float("nan")

    def _summary(self, dips, x, d):
        """The misfits and moment shared by both scores."""
        residual = d - self._full_design(dips) @ x
        m = 2 * self.n
        out = {"rms": float(np.sqrt(np.mean(residual ** 2))),
               "weighted_rms": float(np.sqrt(np.sum((self.w * residual) ** 2)
                                             / np.sum(self.w ** 2))),
               "Mw": self._moment_magnitude(dips, x),
               "peak": float(np.hypot(x[: self.n], x[self.n: m]).max())}
        for t in np.unique(self.track):
            out[f"rms_{t}"] = float(np.sqrt(np.mean(residual[self.track == t] ** 2)))
        return out

    def evaluate(self, dips, *, rank=None, smoothing=None, data=None):
        """Solve one combination of dips; returns a dict of its scores and solution.

        Exactly one of `rank` (an effective rank to hold) or `smoothing` (a
        fixed weight, as `SlipInversion.solve` takes it). `data`
        replaces the observed LOS with another vector on the same samples.
        `x` in the result is the parameter vector: strike-slip for every node,
        then dip-slip, then any ramp coefficients.
        """
        if (rank is None) == (smoothing is None):
            raise ValueError("give exactly one of rank= or smoothing=")
        dips = self._combination(dips)
        key = (dips, rank, smoothing)
        if data is None and key in self._memo:
            return self._memo[key]
        d = self.d if data is None else np.asarray(data, dtype=float)
        if d.shape != self.d.shape:
            raise ValueError(f"data has shape {d.shape}; the samples need {self.d.shape}")

        design = self._full_design(dips)
        weighted = design * self.w[:, None]
        N = weighted.T @ weighted
        g = weighted.T @ (self.w * d)
        lam = (float(smoothing) if smoothing is not None
               else self._lam_for_rank(N, float(rank), guess=self._last["lam"]))
        self._last["lam"] = lam
        x = self._solve(N, g, lam)

        m = 2 * self.n
        bounded = np.isfinite(self._lo[:m]) | np.isfinite(self._hi[:m])
        at_bound = bounded & ((x[:m] <= self._lo[:m] + 1e-6) | (x[:m] >= self._hi[:m] - 1e-6))
        out = {"dips": dips, "lam": lam, "rank": self._rank(N, lam),
               **self._summary(dips, x, d), "at_bound": float(at_bound.sum() / m), "x": x}
        if data is None:
            self._memo[key] = out
        return out

    # -- the evidence ---------------------------------------------------------
    def evidence_model(self, covariance=None):
        """The `slipsolve.evidence.GeometryEvidence` this sweep scores with.

        One per noise model, kept, so every geometry it scores shares one
        `alpha^2` grid. The prior is this sweep's own smoothing operator with
        its `sides` pinned to zero outright.
        """
        key = _evidence_cache_key(covariance)
        if key not in self._evidence:
            q, free = smoothing_precision(self.reference, sides=self.sides,
                                          operator=self._smooth)
            self._evidence[key] = GeometryEvidence(self.obs, q, free, self.ramp,
                                                   covariance=covariance)
        return self._evidence[key]

    def _evidence_solution(self, ev, dec, dh, g, d, alpha2):
        """Full parameter vectors `(2n + k, m)` at each column's `alpha2`."""
        m_free = ev.posterior_mean(dec, dh, alpha2)
        x = np.zeros((2 * self.n + self.ramp.shape[1], m_free.shape[1]))
        x[ev.free] = m_free
        if self.ramp.shape[1]:
            residual = np.asarray(d, dtype=float).reshape(self.d.size, -1) - g[:, ev.free] @ m_free
            x[2 * self.n:] = ev.ramp_coefficients(residual)
        return x

    def evidence(self, dips, *, data=None, covariance=None):
        """Score one combination by its marginal likelihood; a dict like `evaluate`'s.

        `log_evidence` integrates the slip, ramp, noise scale and smoothing
        weight out (see `slipsolve.evidence`); `alpha2`, `rank` and
        `sigma2` are at the integrand's mode, and `x` is the posterior-mean
        solution there -- unbounded, so its `Mw` and `peak` are a guide only.
        `covariance` is the noise: `None` (iid, the robust default),
        `"weights"`, or a `CovarianceModel` per track (whitened).
        """
        dips = self._combination(dips)
        d = self.d if data is None else np.asarray(data, dtype=float)
        if d.shape != self.d.shape:
            raise ValueError(f"data has shape {d.shape}; the samples need {self.d.shape}")
        ev = self.evidence_model(covariance)
        g = self.design(dips)
        dec = ev.decompose(g)
        dh, s0 = ev.prepare(d)
        score = ev.score(dec, dh, s0)
        x = self._evidence_solution(ev, dec, dh, g, d, score["alpha2"])[:, 0]
        return {"dips": dips, "log_evidence": float(score["log_evidence"][0]),
                "alpha2": float(score["alpha2"][0]), "rank": float(score["rank"][0]),
                "sigma2": float(score["sigma2"][0]), "alpha_edge": bool(score["alpha_edge"][0]),
                **self._summary(dips, x, d), "x": x}

    # -- the search ---------------------------------------------------------
    def run(self, *, rank=None, smoothing=None, score="rms", covariance=None, method="auto",
            max_evaluations=1000, passes=4, start=None, fix=None, data=None, log=None):
        """Search the combinations of dips; returns a `SweepResult`.

        `score="evidence"` ranks by the marginal likelihood (no `rank` or
        `smoothing`; `covariance` as `evidence` takes it).
        `"rms"` (unweighted) and `"weighted_rms"` (the solve's own data term)
        rank the bounded solve at a fixed `rank` or `smoothing`.

        `method="exhaustive"` scores every combination; `"coordinate"` starts
        from the best uniform geometry (or `start`) and repeatedly re-chooses one
        segment with the others held, until a whole pass changes nothing.
        `"auto"` is exhaustive when there are at most `max_evaluations`
        combinations. `fix={segment: dip}` (0-based) holds segments at a
        geometry, which need not be on the grid; hold a contrast interval with a
        one-value grid in `ratios=`. A contrast's ratios are coordinates like
        the dips, after them. `data` replaces the observed LOS; `log` is
        called with one line per new evaluation.

        The coordinate search finds a combination no single segment can
        improve, which is not guaranteed to be the global best when segments
        trade off against each other; compare it against an exhaustive run on a
        coarser grid when that matters.
        """
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        if score == "evidence":
            if rank is not None or smoothing is not None:
                raise ValueError("score='evidence' integrates the smoothing weight out; "
                                 "give neither rank= nor smoothing=")
        elif covariance is not None:
            raise ValueError("covariance= applies only to score='evidence'")
        k = self.n_segments
        candidates = [list(self.dips) for _ in range(k)]
        for j, value in (fix or {}).items():
            if not 0 <= int(j) < k:
                raise ValueError(f"fix names segment {j}; segments are 0-{k - 1}")
            candidates[int(j)] = [geometry_key(value)]
        candidates += [list(g) for g in self.ratio_grids]
        count = _combinations(candidates)
        if method == "auto":
            method = "exhaustive" if count <= max_evaluations else "coordinate"
        if method not in ("exhaustive", "coordinate"):
            raise ValueError(f"method must be auto, exhaustive or coordinate, got {method!r}")

        rows = {}

        def value(combo):
            if combo not in rows:
                if score == "evidence":
                    rows[combo] = self.evidence(combo, data=data, covariance=covariance)
                else:
                    rows[combo] = self.evaluate(combo, rank=rank, smoothing=smoothing, data=data)
                if log is not None:
                    r, label = rows[combo], self.label(combo)
                    if score == "evidence":
                        log(f"dips {label}: log evidence {r['log_evidence']:.3f}, "
                            f"rank {r['rank']:.1f}, Mw {r['Mw']:.4f}")
                    else:
                        log(f"dips {label}: {score} {1e3 * r[score]:.3f} mm, "
                            f"rank {r['rank']:.1f}, Mw {r['Mw']:.4f}")
            r = rows[combo]
            return -r["log_evidence"] if score == "evidence" else r[score]

        if method == "exhaustive":
            for combo in itertools.product(*candidates):
                value(tuple(combo))
            best = min(rows, key=value)
        else:
            if start is None:
                tail = tuple(c[0] if len(c) == 1 else r
                             for c, r in zip(candidates[k:], self._default_ratios))
                uniform = [tuple(c[0] if len(c) == 1 else v for c in candidates[:k]) + tail
                           for v in self.dips]
                current = min(uniform, key=value)
            else:
                current = tuple(c[0] if len(c) == 1 else v
                                for c, v in zip(candidates, self._combination(start)))
                value(current)
            for _ in range(int(passes)):
                changed = False
                for j, options in enumerate(candidates):
                    trials = [current[:j] + (v,) + current[j + 1:] for v in options]
                    challenger = min(trials, key=value)
                    if _better(value(challenger), value(current)):
                        current, changed = challenger, True
                if not changed:
                    break
            # Complete the profiles around the best, and keep going if doing so
            # turns up a better combination (it can, when `passes` ran out first):
            # the result's best is then always the best thing evaluated, with its
            # whole one-segment neighbourhood scored.
            best = current
            while True:
                for j, options in enumerate(candidates):
                    for v in options:
                        value(best[:j] + (v,) + best[j + 1:])
                top = min(rows, key=value)
                if not _better(value(top), value(best)):
                    break
                best = top
        return SweepResult(self, rows, best, method=method, score=score, rank=rank,
                           smoothing=smoothing, covariance=covariance, data=data)

    # -- continuous profiles --------------------------------------------------
    def optimize(self, knots=1, *, start=None, fix=None, bounds=None, data=None, covariance=None,
                 continuity=None, step=0.25, radius=5.0, max_iter=30, xtol=0.01, log=None):
        """Maximise the evidence over each segment's dip profile, continuously; returns a `ProfileFit`.

        A segment's geometry is `knots` dips at evenly spaced depths from the top of the
        layout to `knot_depth`, linear in depth between them (`geometry_key`'s tuples): one
        knot is a planar dip, two a dip that changes steadily with depth, three or more a
        curved profile. `knots` is one count for every segment or a list of counts.
        `fix={segment: geometry}` (0-based) holds segments; a contrast's ratios are held
        at the start's.

        `start` is where the search begins: a combination, a `SweepResult` (its best) or
        a `ProfileFit` (re-knotted when the counts differ, so a fit can be refined one
        knot at a time). Without it, `run(score="evidence")` over the grid supplies one.
        The search is local, so start it from a sweep, which sees the whole grid.
        `bounds=(lo, hi)` limits every knot, in degrees, by default to the span of the
        grid's dips; a knot that ends on a bound is flagged and warned about.

        **Tears.** Each segment is meshed at its own geometry, so where two segments
        with different geometries meet, their surfaces lie apart at depth (`tears`,
        reported per break in `ProfileFit.tears`). Nothing in the evidence keeps that
        gap small: a real fault that bends between segments stays continuous.
        `continuity`, a length in metres, adds a prior that each break's rms tear over
        depth is of that order: the search maximises the log evidence less
        `0.5 * sum((rms tear / continuity) ** 2)` over the breaks, and the bootstrap
        re-estimates under the same prior.

        **How.** A trust-region search on a design linearised in the knots. Each knot's
        derivative comes from central differences of `step` degrees, two assemblies of
        its own segment only, because the design is a sum of per-segment blocks. The
        evidence of the linearised design is maximised within `radius` degrees of the
        iterate using its analytic gradient (`GeometryEvidence.gradient`). The step is
        kept if the assembled design confirms the gain, and the radius follows how well
        the two agreed. The search stops when the linearised design finds no gain, or a
        kept step is under `xtol` degrees. An iteration costs about two segment
        assemblies per knot, and the search is then relinearised at the optimum for
        `bootstrap`.

        `data` and `covariance` as `evidence` takes them; `log` is called with one line
        per iteration. Read the uncertainty from `bootstrap(fit)`, not from the evidence's
        curvature, which under iid noise is too narrow (see `slipsolve.evidence`), and
        judge a knot by its interval rather than by the gain in log evidence it brings:
        with correlated noise taken as iid, extra knots gain even on a planar fault.
        """
        k = self.n_segments
        d = self.d if data is None else np.asarray(data, dtype=float)
        if d.shape != self.d.shape:
            raise ValueError(f"data has shape {d.shape}; the samples need {self.d.shape}")
        fixed = {}
        for j, value in (fix or {}).items():
            if not 0 <= int(j) < k:
                raise ValueError(f"fix names segment {j}; segments are 0-{k - 1}")
            fixed[int(j)] = geometry_key(value)
        counts = [int(knots)] * k if np.ndim(knots) == 0 else [int(c) for c in knots]
        if len(counts) != k or min(counts) < 1:
            raise ValueError(f"knots must be a positive count, or one per segment ({k}); got {knots!r}")
        free = [j for j in range(k) if j not in fixed]
        if not free:
            raise ValueError("every segment is fixed: nothing to optimise")
        lo, hi = self._knot_bounds(bounds)
        step, xtol = float(step), float(xtol)
        if not 0.0 < step < 1.0 or xtol <= 0.0:
            raise ValueError("step must lie between 0 and 1 degree, and xtol must be positive")
        if continuity is not None and not float(continuity) > 0.0:
            raise ValueError(f"continuity is a length in metres and must be positive, got {continuity!r}")
        scale_tear = None if continuity is None or k < 2 else float(continuity)

        combo = self._start_combination(start, data=data, covariance=covariance, fix=fixed)
        geometry, ratios = list(combo[:k]), tuple(combo[k:])
        for j, value in fixed.items():
            geometry[j] = value
        params = [(j, m) for j in free for m in range(counts[j])]
        theta = np.clip(np.concatenate([self._knots_of(geometry[j], counts[j]) for j in free]), lo, hi)

        def keys_of(values):
            out, i = {}, 0
            for j in free:
                out[j] = _knot_key(values[i: i + counts[j]])
                i += counts[j]
            return out

        def penalty(values):
            """The continuity prior's cost: half the sum over breaks of (rms tear / continuity)^2."""
            if scale_tear is None:
                return 0.0
            found = keys_of(values)
            tears = self.tears(tuple(found[j] if j in found else fixed[j] for j in range(k)))
            return 0.5 * float(np.sum(np.mean(tears ** 2, axis=1))) / scale_tear ** 2

        def penalty_gradient(values):
            out = np.zeros(values.size)
            if scale_tear is None:
                return out
            for i in range(values.size):
                e = np.zeros(values.size)
                e[i] = HESSIAN_STEP
                out[i] = (penalty(values + e) - penalty(values - e)) / (2.0 * HESSIAN_STEP)
            return out

        def penalty_hessian(values):
            out = np.zeros((values.size, values.size))
            if scale_tear is None:
                return out
            for i in range(values.size):
                e = np.zeros(values.size)
                e[i] = HESSIAN_STEP
                out[:, i] = (penalty_gradient(values + e) - penalty_gradient(values - e)) / (2.0 * HESSIAN_STEP)
            return 0.5 * (out + out.T)

        ev = self.evidence_model(covariance)
        dh, s0 = ev.prepare(d)
        position = np.full(2 * self.n, -1)
        position[ev.free] = np.arange(ev.p)
        columns = {}
        for j in free:
            nodes = self.segment_nodes[j]
            full = np.concatenate([nodes, self.n + nodes])
            found = position[full]
            columns[j] = (full, np.flatnonzero(found >= 0), found[found >= 0])
        scale = self._contrast_factors(ratios)[:2] if self.n_ratios else None

        def white(g):
            return ev.project(ev.whiten(g))

        def derivatives(values, current_blocks):
            """Per knot: (free positions, whitened derivative, raw derivative over the segment's columns)."""
            out = []
            for i, (j, _) in enumerate(params):
                up, down = values.copy(), values.copy()
                up[i] = min(values[i] + step, 180.0 - 1e-6)
                down[i] = max(values[i] - step, 1e-6)
                b_up = (current_blocks[j] if up[i] == values[i]
                        else self._transient_block(j, keys_of(up)[j]))
                b_down = (current_blocks[j] if down[i] == values[i]
                          else self._transient_block(j, keys_of(down)[j]))
                raw = (b_up - b_down) / (up[i] - down[i])
                full, cols, pos = columns[j]
                if scale is not None:
                    raw = raw * scale[0][:, None] * scale[1][full][None, :]
                out.append((pos, white(raw[:, cols]), raw))
            return out

        def linear(gh0, deriv):
            """Log evidence and its gradient in the knots, of the design linearised at `gh0`."""
            def value(delta, with_gradient=True):
                gh = gh0.copy()
                for i, (pos, e, _) in enumerate(deriv):
                    if delta[i]:
                        gh[:, pos] += delta[i] * e
                dec = ev.decompose_whitened(gh)
                logev = float(ev.score(dec, dh, s0)["log_evidence"][0])
                if not with_gradient:
                    return logev
                grad = ev.gradient(dec, dh, s0)
                return logev, np.array([np.sum(grad[:, pos] * e) for pos, e, _ in deriv])
            return value

        def evidence_of(gh):
            return float(ev.score(ev.decompose_whitened(gh), dh, s0)["log_evidence"][0])

        def record(iteration, values, logev, **extra):
            row = {"iteration": iteration, "log_evidence": logev, **extra}
            row.update({f"s{j + 1}k{m + 1}": float(v) for (j, m), v in zip(params, values)})
            history.append(row)

        blocks = {j: self._segment_block(j, value) for j, value in fixed.items()}
        keys = keys_of(theta)
        for j in free:
            blocks[j] = self._transient_block(j, keys[j])
        assemblies = len(free)
        gh_now = white(self._compose(blocks, ratios)[:, ev.free])
        current_ev = evidence_of(gh_now)
        current = current_ev - penalty(theta)            # the objective: the evidence less the prior
        history = []
        extra0 = {"penalty": penalty(theta)} if scale_tear is not None else {}
        record(0, theta, current_ev, predicted=np.nan, ratio=np.nan, radius=float(radius), step=0.0,
               accepted=True, **extra0)
        r = float(radius)
        deriv, at = None, None
        converged, message = False, f"stopped after {int(max_iter)} iterations"
        for iteration in range(1, int(max_iter) + 1):
            if at is None or not np.array_equal(at, theta):
                deriv = derivatives(theta, blocks)
                assemblies += 2 * len(params)
                at = theta.copy()
            model = linear(gh_now, deriv)

            def objective(x, model=model, base=theta):
                value, grad = model(x)
                if scale_tear is None:
                    return -value, -grad
                return -(value - penalty(base + x)), -(grad - penalty_gradient(base + x))

            box = list(zip(np.maximum(-r, lo - theta), np.minimum(r, hi - theta)))
            found = minimize(objective, np.zeros(theta.size), jac=True, method="L-BFGS-B", bounds=box,
                             options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-5})
            predicted = -float(found.fun) - current
            if predicted <= GAIN_TOLERANCE:
                converged, message = True, "the linearised design finds no further gain"
                break
            trial = np.clip(theta + found.x, lo, hi)
            trial_keys = keys_of(trial)
            trial_blocks = dict(blocks)
            for j in free:
                if trial_keys[j] != keys[j]:
                    trial_blocks[j] = self._transient_block(j, trial_keys[j])
                    assemblies += 1
            gh_trial = white(self._compose(trial_blocks, ratios)[:, ev.free])
            actual_ev = evidence_of(gh_trial)
            cost = penalty(trial)
            actual = actual_ev - cost
            ratio = (actual - current) / predicted
            moved = float(np.max(np.abs(trial - theta)))
            accepted = actual > current
            extra = {"penalty": cost} if scale_tear is not None else {}
            record(iteration, trial, actual_ev, predicted=current + predicted + cost, ratio=ratio,
                   radius=r, step=moved, accepted=accepted, **extra)
            if log is not None:
                log(f"optimize {iteration}: knots {' '.join(f'{v:.2f}' for v in trial)}: log evidence "
                    f"{actual_ev:.3f} (predicted {current + predicted + cost:.3f})"
                    + (f", continuity penalty {cost:.3f}" if scale_tear is not None else "")
                    + f", step {moved:.3f} deg, radius {r:.3g}, {'kept' if accepted else 'refused'}")
            if accepted:
                theta, keys, blocks, gh_now = trial, trial_keys, trial_blocks, gh_trial
                current_ev, current = actual_ev, actual
            if ratio < 0.25:
                r = 0.25 * min(r, max(moved, xtol))
            elif ratio > 0.75 and moved > 0.9 * r:
                r = min(2.0 * r, MAX_RADIUS)
            if accepted and moved < xtol:
                converged, message = True, f"a kept step under {xtol:g} degrees"
                break
            if r < xtol:
                converged, message = True, f"the trust radius fell under {xtol:g} degrees"
                break

        if at is None or not np.array_equal(at, theta):
            deriv = derivatives(theta, blocks)
            assemblies += 2 * len(params)
        model = linear(gh_now, deriv)
        _, gradient = model(np.zeros(theta.size))
        hessian = np.empty((theta.size, theta.size))
        for i in range(theta.size):
            e = np.zeros(theta.size)
            e[i] = HESSIAN_STEP
            hessian[:, i] = (model(e)[1] - model(-e)[1]) / (2.0 * HESSIAN_STEP)
        hessian = 0.5 * (hessian + hessian.T)
        prior = dict(gradient=penalty_gradient(theta), hessian=penalty_hessian(theta))

        for j in free:
            self._blocks.setdefault((j, keys[j]), blocks[j])
        dips = tuple(keys[j] if j in keys else fixed[j] for j in range(k)) + ratios
        solution = self.evidence(dips, data=data, covariance=covariance)
        on_bound = (theta <= lo + 1e-9) | (theta >= hi - 1e-9)
        fit = ProfileFit(self, dips=dips, params=params, counts=counts, theta=theta, bounds=(lo, hi),
                         log_evidence=current_ev, gradient=gradient - prior["gradient"],
                         hessian=hessian - prior["hessian"], solution=solution,
                         history=pd.DataFrame(history), assemblies=assemblies,
                         converged=converged, message=message, on_bound=on_bound,
                         data=data, covariance=covariance, continuity=scale_tear,
                         penalty=penalty(theta),
                         linear=dict(design=self._compose(blocks, ratios), deriv=deriv, prior=prior))
        for (j, m), flag, value in zip(params, on_bound, theta):
            if flag:
                warnings.warn(f"segment {j + 1}, knot {m + 1}: the best dip ({value:.2f}) is on a bound "
                              f"({lo:g}-{hi:g}), so it cannot move past it; widen bounds=",
                              stacklevel=2)
        return fit

    def _knot_bounds(self, bounds):
        if bounds is None:
            values = [v for key in self.dips for v in (key if isinstance(key, tuple) else (key,))]
            lo, hi = float(min(values)), float(max(values))
            if hi <= lo:
                raise ValueError("the grid has one dip; give bounds=(lo, hi)")
        else:
            lo, hi = (float(v) for v in bounds)
        if not 0.0 < lo < hi < 180.0:
            raise ValueError(f"bounds must satisfy 0 < lo < hi < 180 degrees, got ({lo}, {hi})")
        return lo, hi

    def _knots_of(self, key, count):
        """Geometry `key` as `count` dips at evenly spaced depths (exact when they include its own knots)."""
        top, bottom = float(self.depths.min()), self.knot_depth
        if isinstance(key, tuple):
            return np.interp(np.linspace(top, bottom, count), np.linspace(top, bottom, len(key)), key)
        return np.full(count, float(key))

    def _start_combination(self, start, *, data, covariance, fix):
        if start is None:
            return self.run(score="evidence", data=data, covariance=covariance, fix=fix).best
        if isinstance(start, SweepResult):
            return start.best
        if isinstance(start, ProfileFit):
            return start.dips
        return self._combination(start)

    # -- the uncertainty ----------------------------------------------------
    def bootstrap(self, result, *, noise="reml", draws=100, seed=0, compare=True, log=None):
        """Parametric bootstrap of an evidence sweep: each segment's dip with an honest spread.

        The best combination's posterior-mean model predicts the data; `draws`
        synthetic data sets add noise drawn from `noise` to that prediction,
        and every combination `result` evaluated is re-scored on each. A
        segment's estimate is read off the spline through its profile, so it is
        continuous and not tied to the grid.

        `noise` is `"reml"` -- fitted to the best model's residual by
        `slipsolve.covariance.fit_reml`, per track -- or a
        `CovarianceModel` (every track) or `{track: model}` mapping.
        With `compare` (the default), each replicate is also scored by the
        other evidence: the noise-weighted one if `result` was iid, the iid
        one if it was weighted. `BootstrapResult.consistency` then says
        whether the two estimates disagree by more than noise explains.

        Replicates are re-scored on the combinations `result` already
        evaluated, so for a coordinate search the spread is conditional on the
        other segments. Each replicate keeps its own best combination where that
        combination's profile was scored; otherwise it falls back to `result`'s.
        Give the grid a step finer than the spread you expect near the best:
        the estimate is splined between grid points, but a profile two points
        wide carries little information. Mesh and other forward-model errors are
        not in these intervals; the weighted-vs-iid comparison is what detects
        them.

        Given a `ProfileFit`, every knot is re-estimated on each data set by one
        bounded Newton step of the evidence, on the design the fit linearised at
        its optimum and with the observed data's Hessian: no assembly, and one
        evidence gradient per data set. The step is exact for a quadratic evidence
        and close while the replicates stay near the optimum. Returns a
        `ProfileBootstrap`.
        """
        if isinstance(result, ProfileFit):
            return self._bootstrap_profile(result, noise=noise, draws=draws, seed=seed,
                                           compare=compare, log=log)
        if result.score != "evidence":
            raise ValueError("bootstrap re-runs the evidence estimator; give it the result of "
                             "run(score='evidence')")
        combos = list(result._rows)
        d_obs = self.d if result.data is None else np.asarray(result.data, dtype=float)
        pred = self.predict(result.best, result.solution(result.best))
        data, models = self._replicate_data(d_obs, pred, noise, draws, seed)

        estimators = {"primary": self.evidence_model(result.covariance)}
        if compare:
            other = models if result.covariance is None or isinstance(result.covariance, str) else None
            estimators["other"] = self.evidence_model(other)
        prepared = {k: ev.prepare(data) for k, ev in estimators.items()}
        logev = {k: np.empty((len(combos), data.shape[1])) for k in estimators}
        for i, combo in enumerate(combos):
            g = self.design(combo)
            for k, ev in estimators.items():
                logev[k][i] = ev.score(ev.decompose(g), *prepared[k])["log_evidence"]
            if log is not None:
                log(f"bootstrap {i + 1}/{len(combos)}: {self.label(combo)}")
        estimates = {k: self._estimates(combos, v, result.best) for k, v in logev.items()}
        return BootstrapResult(self, result, estimates, models, compare=compare)

    def _replicate_data(self, d_obs, pred, noise, draws, seed):
        """The observed data, then `draws` replicates of `pred` plus noise; and the noise models."""
        names = list(dict.fromkeys(self.track))
        if isinstance(noise, str):
            if noise != "reml":
                raise ValueError(f"noise must be 'reml', a CovarianceModel or a mapping; got {noise!r}")
            models = fit_reml(self.obs, d_obs - pred, ramp=self.ramp_kind)
        elif isinstance(noise, CovarianceModel):
            models = {t: noise for t in names}
        else:
            models = dict(noise)
        rng = np.random.default_rng(seed)
        replicates = np.repeat(pred[:, None], int(draws), axis=1)
        for t in names:
            rows = np.flatnonzero(self.track == t)
            factor = sla.cholesky(observation_covariance(self.obs, models[t], track=t), lower=True)
            replicates[rows] += factor @ rng.standard_normal((rows.size, int(draws)))
        return np.column_stack([d_obs, replicates]), models          # column 0 is the observed data

    def _bootstrap_profile(self, fit, *, noise, draws, seed, compare, log):
        d_obs = self.d if fit.data is None else np.asarray(fit.data, dtype=float)
        pred = self.predict(fit.dips, fit.solution["x"])
        data, models = self._replicate_data(d_obs, pred, noise, draws, seed)
        primary = self._newton_estimates(fit, self.evidence_model(fit.covariance), data)
        if log is not None:
            log(f"bootstrap: {data.shape[1] - 1} replicates re-estimated")
        other = None
        if compare:
            plain = fit.covariance is None or isinstance(fit.covariance, str)
            other = self._newton_estimates(fit, self.evidence_model(models if plain else None), data)
        return ProfileBootstrap(fit, primary, other, models)

    def _newton_estimates(self, fit, ev, data):
        """Each data column's knots `(m, knots)`: one bounded Newton step of `ev`'s evidence from the fit."""
        lin = fit._linear
        gh0 = ev.project(ev.whiten(lin["design"][:, ev.free]))
        position = np.full(2 * self.n, -1)
        position[ev.free] = np.arange(ev.p)
        deriv = []
        for (j, _), (_, _, raw) in zip(fit._params, lin["deriv"]):
            nodes = self.segment_nodes[j]
            found = position[np.concatenate([nodes, self.n + nodes])]
            keep = found >= 0
            deriv.append((found[keep], ev.project(ev.whiten(raw[:, keep]))))
        dec0 = ev.decompose_whitened(gh0)
        dh, s0 = ev.prepare(data)

        def gradient(dec, column):
            grad = ev.gradient(dec, dh[:, [column]], s0[column])
            return np.array([np.sum(grad[:, pos] * e) for pos, e in deriv])

        def shifted(delta):
            gh = gh0.copy()
            for i, (pos, e) in enumerate(deriv):
                gh[:, pos] += delta[i] * e
            return ev.decompose_whitened(gh)

        size = len(deriv)
        hessian = np.empty((size, size))
        for i in range(size):
            e = np.zeros(size)
            e[i] = HESSIAN_STEP
            hessian[:, i] = (gradient(shifted(e), 0) - gradient(shifted(-e), 0)) / (2.0 * HESSIAN_STEP)
        # the continuity prior, when the fit had one: the same for every data set
        prior = lin.get("prior") or {"gradient": np.zeros(size), "hessian": np.zeros((size, size))}
        w, v = np.linalg.eigh(-0.5 * (hessian + hessian.T) + prior["hessian"])
        w = np.maximum(w, 1e-9 * max(float(w.max()), np.finfo(float).tiny))
        curvature = (v * w) @ v.T
        lo, hi = fit.bounds
        theta = fit._theta
        out = np.empty((data.shape[1], size))
        for column in range(data.shape[1]):
            step = box_qp(curvature, gradient(dec0, column) - prior["gradient"], lo - theta, hi - theta).x
            out[column] = theta + step
        return out

    def _estimates(self, combos, logev, fallback):
        """Per data column and coordinate: the continuous estimate, and the grid step around it."""
        index = {c: i for i, c in enumerate(combos)}
        k = len(combos[0])
        out = np.empty((logev.shape[1], k), dtype=object)
        steps = np.full((logev.shape[1], k), np.nan)
        best = np.argmax(logev, axis=0)
        for col in range(logev.shape[1]):
            top = combos[best[col]]
            for j in range(k):
                keys, values = None, None
                for anchor in (top, tuple(fallback)):
                    found = [(c[j], logev[i, col]) for c, i in index.items()
                             if all(c[m] == anchor[m] for m in range(k) if m != j)]
                    if keys is None or len(found) > len(keys):
                        keys = [f[0] for f in found]
                        values = np.array([f[1] for f in found])
                out[col, j], steps[col, j] = _refine(keys, values, log=j >= self.n_segments)
        return out, steps


def _refine(keys, values, log=False):
    """A profile's continuous maximiser (planar dips, ratios) or best key, and the local grid step.

    `log` splines a ratio's profile in log ratio, where its grid is usually
    even; the step is still reported in the coordinate's own units.
    """
    if any(isinstance(key, tuple) for key in keys) or len(keys) < 3:
        return keys[int(np.argmax(values))], np.nan
    x = np.asarray(keys, dtype=float)
    order = np.argsort(x)
    x, y = x[order], np.asarray(values, dtype=float)[order]
    i = int(np.argmax(y))
    lo, hi = max(i - 1, 0), min(i + 1, x.size - 1)
    step = float(np.min(np.diff(x[lo: hi + 1]))) if hi > lo else np.nan
    u = np.log(x) if log else x
    h = REFINE_STEP / 10 if log else REFINE_STEP          # 0.05% in ratio, 0.005 degrees in dip
    fine = np.arange(u[lo], u[hi] + 0.5 * h, h)
    curve = (CubicSpline(u, y)(fine) if u.size >= 4
             else np.polyval(np.polyfit(u, y, 2), fine))
    best = float(fine[int(np.argmax(curve))])
    return (float(np.exp(best)) if log else best), step


class SweepResult:
    """Every combination a `DipSweep.run` evaluated, and the best of them.

    `table` has one row per combination -- `dip_1 ... dip_K` in segment order
    (see `segments` for where each lies; a dip that changes with depth is a
    tuple), then `ratio_1 ...` for a contrast's intervals, its score and the
    other misfits, `Mw`, the peak slip, and for the bounded solve the smoothing
    weight, effective rank and fraction of slip parameters on a bound -- sorted
    best first. `score` is the quantity minimised: the misfit, or minus the log
    evidence, whose drop from the best is `delta_log_evidence`.
    `solution(dips)` returns the parameter vector of any evaluated combination.
    """

    def __init__(self, sweep, rows, best, method, score, rank, smoothing,
                 covariance=None, data=None):
        self.segments = sweep.segments
        self.n_dips = sweep.n_segments
        self.best = tuple(best)
        self.method, self.score, self.rank, self.smoothing = method, score, rank, smoothing
        self.covariance, self.data = covariance, data
        self._combination = sweep._combination
        self._label = sweep.label
        self._rows = dict(rows)
        self._solutions = {combo: r["x"] for combo, r in rows.items()}
        records = []
        for combo, r in rows.items():
            record = {(f"dip_{j + 1}" if j < self.n_dips else f"ratio_{j - self.n_dips + 1}"): v
                      for j, v in enumerate(combo)}
            record.update({k: v for k, v in r.items() if k not in ("dips", "x")})
            record["score"] = self._score(r)
            records.append(record)
        self.table = (pd.DataFrame(records).sort_values("score", kind="stable")
                      .reset_index(drop=True))
        if score == "evidence":
            self.table["delta_log_evidence"] = (self.table["log_evidence"]
                                                - self.table["log_evidence"].max())

    def _score(self, row):
        return -row["log_evidence"] if self.score == "evidence" else row[self.score]

    def solution(self, dips):
        """The solution vector for the dip combination `dips`."""
        return self._solutions[self._combination(dips)]

    def profiles(self):
        """Each coordinate's score against its value, the others at the best.

        Long form: `parameter` (`dip` or `ratio`), `segment` (1-based, for
        a dip) or `interval` (1-based, for a contrast's ratio), `dip` or
        `ratio`, `score` and `delta` -- the excess over the best
        combination's score (for the evidence, the drop in log evidence). Only
        evaluated combinations appear, so an exhaustive run and a finished
        coordinate search give complete profiles.
        """
        best_score = self._score(self._rows[self.best])
        rows = []
        for j in range(len(self.best)):
            hits = [(combo[j], self._score(r)) for combo, r in self._rows.items()
                    if all(combo[m] == self.best[m] for m in range(len(combo)) if m != j)]
            hits.sort(key=lambda h: (isinstance(h[0], tuple), h[0]))
            dip = j < self.n_dips
            rows += [{"parameter": "dip" if dip else "ratio",
                      "segment": j + 1 if dip else np.nan,
                      "interval": np.nan if dip else j - self.n_dips + 1,
                      "dip": key if dip else np.nan, "ratio": np.nan if dip else key,
                      "score": s, "delta": s - best_score} for key, s in hits]
        return pd.DataFrame(rows)

    def __repr__(self):
        best = self.table.iloc[0]
        label = self._label(self.best)
        if self.score == "evidence":
            return (f"<SweepResult evidence, {self.method}, {len(self.table)} combinations, best "
                    f"dips {label}: log evidence {best['log_evidence']:.3f} (rank "
                    f"{best['rank']:.1f} at the mode), Mw {best['Mw']:.4f}>")
        return (f"<SweepResult {self.method}, {len(self.table)} combinations, best dips "
                f"{label}: {self.score} {1e3 * best['score']:.3f} mm at rank {best['rank']:.1f}, "
                f"Mw {best['Mw']:.4f}>")


class BootstrapResult:
    """What `DipSweep.bootstrap` found: estimates, their spread, and a consistency check.

    `observed` is each segment's estimate on the data and `replicates` the
    `(draws, segments)` estimates on the synthetic data sets, both continuous
    for planar dips (and the best key for a dip that changes with depth).
    `noise` holds the per-track models the replicates were drawn from.
    """

    def __init__(self, sweep, result, estimates, noise, compare=True):
        self.segments = sweep.segments
        self.n_dips = sweep.n_segments
        self.result = result
        self.noise = noise
        primary, steps = estimates["primary"]
        self.observed = list(primary[0])
        self.replicates = primary[1:]
        self.grid_step = steps[0]
        self._other = estimates["other"][0] if compare else None
        self.weighted = result.covariance is not None and not isinstance(result.covariance, str)
        for j, est in enumerate(self.observed):
            column = self.replicates[:, j]
            if isinstance(est, float) and np.isfinite(self.grid_step[j]):
                spread = float(np.std(column.astype(float), ddof=1))
                if spread < 0.1 * self.grid_step[j]:
                    warnings.warn(
                        f"{self._name(j)}: the bootstrap spread ({spread:.3g}) is under a "
                        f"tenth of the grid's step there ({self.grid_step[j]:.3g}), so the "
                        "estimates rest on the spline between grid points; check with a finer "
                        f"grid around {est:.3f}", stacklevel=3)
        # A coordinate whose best is on its grid's edge cannot move past it, and the
        # others take up what it would have done: with dip and ratio swept together,
        # a dip pinned at the edge leaves the ratio's interval too narrow.
        for j, best in enumerate(result.best):
            values = sorted({c[j] for c in result._rows if not isinstance(c[j], tuple)})
            if len(values) >= 3 and best in (values[0], values[-1]):
                warnings.warn(
                    f"{self._name(j)}: the best value ({best:g}) is on the edge of its grid "
                    f"({values[0]:g}-{values[-1]:g}), so it cannot move past it and the other "
                    "coordinates' intervals come out too narrow; widen the grid", stacklevel=3)

    def _name(self, j):
        return f"segment {j + 1}" if j < self.n_dips else f"ratio {j - self.n_dips + 1}"

    def _keys(self, j):
        """The identifying columns of coordinate `j`'s row."""
        if j < self.n_dips:
            return {"parameter": "dip", "segment": j + 1, "interval": np.nan}
        return {"parameter": "ratio", "segment": np.nan, "interval": j - self.n_dips + 1}

    def intervals(self, level=0.95):
        """Per coordinate: the estimate, its bootstrap sd, and two intervals at `level`.

        `normal` is estimate -/+ z sd; `basic` reflects the replicates'
        quantiles about the estimate. A dip that changes with depth gets the
        share of replicates choosing each candidate instead (`share`). A
        contrast's ratios follow the dips, with `parameter` saying which is which.
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        a = 100 * (1 - level) / 2
        rows = []
        for j, est in enumerate(self.observed):
            column = self.replicates[:, j]
            if not isinstance(est, float):
                keys, counts = np.unique(np.array([geometry_label(v) for v in column]),
                                         return_counts=True)
                rows.append({**self._keys(j), "estimate": geometry_label(est),
                             "share": dict(zip(keys, counts / counts.sum()))})
                continue
            values = column.astype(float)
            sd = float(np.std(values, ddof=1))
            q_lo, q_hi = np.percentile(values - values.mean(), [a, 100 - a])
            rows.append({**self._keys(j), "estimate": est, "sd": sd,
                         "normal_lo": est - z * sd, "normal_hi": est + z * sd,
                         "basic_lo": est - q_hi, "basic_hi": est - q_lo,
                         "grid_step": self.grid_step[j]})
        return pd.DataFrame(rows)

    def consistency(self):
        """Weighted minus iid estimate per coordinate, against its spread under the noise model.

        `z` is the observed difference less its bootstrap mean, over its
        bootstrap sd. With the forward model right, `|z|` is of order one; a
        forward-model error that biases the weighted posterior shows up as `|z|`
        well above that.
        """
        if self._other is None:
            raise ValueError("bootstrap(compare=False) did not score the second estimator")
        rows = []
        for j, est in enumerate(self.observed):
            if not (isinstance(est, float) and isinstance(self._other[0, j], float)):
                rows.append({**self._keys(j), "difference": np.nan, "sd": np.nan, "z": np.nan})
                continue
            diffs = self._other[:, j].astype(float) - self._primary_column(j)
            if self.weighted:                 # report weighted minus iid either way round
                diffs = -diffs
            observed, replicate = diffs[0], diffs[1:]
            sd = float(np.std(replicate, ddof=1))
            rows.append({**self._keys(j), "difference": float(observed),
                         "mean": float(replicate.mean()), "sd": sd,
                         "z": float((observed - replicate.mean()) / sd) if sd > 0 else np.nan})
        return pd.DataFrame(rows)

    def _primary_column(self, j):
        return np.concatenate([[float(self.observed[j])], self.replicates[:, j].astype(float)])

    def __repr__(self):
        parts = []
        for j, est in enumerate(self.observed):
            if isinstance(est, float):
                sd = float(np.std(self.replicates[:, j].astype(float), ddof=1))
                parts.append(f"{self._name(j)} {est:.3g} +- {sd:.2g}")
            else:
                parts.append(f"{self._name(j)} {geometry_label(est)}")
        return f"<BootstrapResult {len(self.replicates)} draws: " + ", ".join(parts) + ">"


class ProfileFit:
    """The dip profiles `DipSweep.optimize` found.

    Attributes
    ----------
    dips : tuple
        The combination: each segment's geometry (a number for one knot, its knot dips
        as a tuple otherwise), then any contrast ratios, as `DipSweep.evidence`,
        `design` and `segment_meshes` take it.
    table : pandas.DataFrame
        One row per optimised knot: `segment` and `knot` (1-based), `depth_km`,
        `dip`, `gradient` (of the objective, per degree, at the optimum) and
        `on_bound`.
    log_evidence : float
        At the optimum.
    penalty : float
        The continuity prior's cost there (0 without `continuity`); the search
        maximised `log_evidence - penalty`.
    continuity : float or None
        The tear scale the search was run with, m.
    tears : pandas.DataFrame
        One row per break between segments: `break` (1-based), `s_km`, and the
        tear's `rms_m` and `max_m` over depth (see `DipSweep.tears`).
    hessian : numpy.ndarray
        Of the objective in the knots, per degree squared, on the design linearised
        at the optimum. Its width is not an uncertainty; see `DipSweep.bootstrap`.
    solution : dict
        `DipSweep.evidence` at the optimum: the posterior-mean parameters `x`,
        `Mw`, `rank`, `alpha2`, ...
    history : pandas.DataFrame
        One row per trust-region step: the knots tried, the log evidence of the
        assembled design and the one the linearised design predicted, their `ratio`,
        the `radius`, the `step` and whether it was `accepted` (and the `penalty`,
        with `continuity`). The ratio and the acceptance are of the objective.
    assemblies : int
        Segment Green's-function assemblies used.
    converged : bool
        Whether the search stopped on its own criteria rather than `max_iter`.
    message : str
        Why it stopped.
    bounds : tuple
        The knots' limits, in degrees.
    """

    def __init__(self, sweep, *, dips, params, counts, theta, bounds, log_evidence, gradient,
                 hessian, solution, history, assemblies, converged, message, on_bound, data,
                 covariance, linear, continuity=None, penalty=0.0):
        self.dips = tuple(dips)
        self.log_evidence = float(log_evidence)
        self.penalty = float(penalty)
        self.continuity = None if continuity is None else float(continuity)
        tears = sweep.tears(self.dips)
        self.tears = pd.DataFrame({"break": np.arange(1, tears.shape[0] + 1),
                                   "s_km": np.asarray(sweep.breaks[1:-1], dtype=float) / 1e3,
                                   "rms_m": np.sqrt(np.mean(tears ** 2, axis=1)),
                                   "max_m": np.abs(tears).max(axis=1) if tears.size else np.zeros(0)})
        self.hessian = np.asarray(hessian, dtype=float)
        self.solution = solution
        self.history = history
        self.assemblies = int(assemblies)
        self.converged = bool(converged)
        self.message = str(message)
        self.bounds = tuple(float(v) for v in bounds)
        self.data, self.covariance = data, covariance
        self._params = list(params)
        self._theta = np.asarray(theta, dtype=float)
        self._linear = linear
        top, bottom = float(sweep.depths.min()), sweep.knot_depth
        self._geometry = SimpleNamespace(
            breaks=np.asarray(sweep.breaks, dtype=float), top=top, knot_depth=float(bottom),
            base=float(sweep.depths.max()), trace=sweep.trace, frame=sweep.frame,
            polyline=(sweep._s, sweep._rx, sweep._ry), normal_smoothing=sweep.normal_smoothing,
            n_segments=sweep.n_segments)
        self.table = pd.DataFrame([
            {"segment": j + 1, "knot": m + 1,
             "depth_km": float(np.linspace(top, bottom, counts[j])[m]) / 1e3, "dip": float(value),
             "gradient": float(g), "on_bound": bool(flag)}
            for (j, m), value, g, flag in zip(params, theta, gradient, on_bound)])

    def profile(self, blend=3e3, breaks=None):
        """The fitted dip as a function of arc length and depth, for `slipsolve.production.Sheet`.

        Returns `profile(s, z)`: the dip in degrees at arc lengths `s` and depths `z`
        (m, positive down), arrays that broadcast. Each segment keeps its own profile
        between its breaks, its knots at evenly spaced depths from the top of the layout
        to `knot_depth`, linear in depth between them and held beyond. Across each break
        the dip changes linearly over `blend` metres either side (0 for a step), so a
        sheet built on it is one surface where the sweep's segments tore. `breaks` are
        the interior breaks in the arc length the function will be called with, by
        default the sweep's; `sheet` carries them onto a sheet's own.
        """
        g = self._geometry
        k = g.n_segments
        inner = np.asarray(g.breaks[1:-1] if breaks is None else breaks, dtype=float).ravel()
        if inner.size != k - 1 or np.any(np.diff(inner) <= 0):
            raise ValueError(f"breaks: {k - 1} increasing interior arc lengths, got {inner}")
        half = float(blend)
        if not half >= 0.0:
            raise ValueError(f"blend must be a length >= 0 (m), got {blend!r}")
        knots = []
        for key in self.dips[:k]:
            values = np.atleast_1d(np.asarray(key, dtype=float))
            knots.append((np.linspace(g.top, g.knot_depth, values.size), values))

        def segment_dip(j, z):
            depths, values = knots[j]
            return np.interp(z, depths, values) if values.size > 1 else np.full(z.shape, values[0])

        def profile(s, z):
            s, z = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(z, dtype=float))
            out = segment_dip(0, z)
            for j, b in enumerate(inner, start=1):
                w = ((s >= b).astype(float) if half == 0.0
                     else np.clip((s - (b - half)) / (2.0 * half), 0.0, 1.0))
                out = (1.0 - w) * out + w * segment_dip(j, z)
            return out

        return profile

    def sheet(self, name, *, blend=3e3, normal_smoothing=None, depth=None, spacing=500.0, **kwargs):
        """A `slipsolve.production.Sheet` on the sweep's trace, with this fit's dips (`profile`).

        The hand-off to a production run. The breaks are carried onto the sheet's own arc
        length by position: its trace is resampled at `spacing`, the sweep's at
        `edge_length`, and the two measure slightly different lengths. `normal_smoothing`
        is the sweep's by default (none if the sweep followed the local normal); `depth`
        the base of the layout. Other keywords (`ends_on`, `bounds`) go to `Sheet`. A
        contrast's ratios in the fit are not carried.
        """
        from .production import Sheet

        g = self._geometry
        if g.top > 0.0:
            raise ValueError(f"the sweep's layout starts {g.top:.0f} m below the surface and a Sheet at it, so "
                             "its surface would not be the sweep's; build one from profile() instead")
        if normal_smoothing is None:
            normal_smoothing = g.normal_smoothing or 0.0
        depth = g.base if depth is None else float(depth)
        own = Sheet(name, g.trace, g.frame, depth=depth, normal_smoothing=normal_smoothing, spacing=spacing)
        s, x, y = g.polyline
        inner = g.breaks[1:-1]
        mapped = _project_arc(np.interp(inner, s, x), np.interp(inner, s, y), own._s, own._x, own._y)
        lengths = np.diff(g.breaks)
        mean = float(np.average([np.mean(np.atleast_1d(v)) for v in self.dips[:g.n_segments]],
                                weights=lengths))
        return Sheet(name, g.trace, g.frame, dip=mean, depth=depth, profile=self.profile(blend, mapped),
                     normal_smoothing=normal_smoothing, spacing=spacing, **kwargs)

    def __repr__(self):
        parts = [f"segment {segment}: " + " / ".join(f"{v:.2f}" for v in group.dip)
                 for segment, group in self.table.groupby("segment")]
        state = "converged" if self.converged else "not converged"
        return (f"<ProfileFit {', '.join(parts)}; log evidence {self.log_evidence:.3f}, "
                f"{self.assemblies} assemblies, {state}>")


class ProfileBootstrap:
    """What `DipSweep.bootstrap` found for a `ProfileFit`: each knot's estimate and spread.

    Attributes
    ----------
    fit : ProfileFit
        The fit that was bootstrapped.
    observed : numpy.ndarray
        Each knot's estimate on the data, `(knots,)`.
    replicates : numpy.ndarray
        Its estimates on the synthetic data sets, `(draws, knots)`.
    noise : dict
        The per-track models the replicates were drawn from.
    weighted : bool
        Whether the fit's evidence was weighted by a noise model.
    on_bound : numpy.ndarray
        Whether each knot's estimate is on a bound, `(knots,)`.
    pinned : numpy.ndarray
        The share of each knot's replicates that stopped on a bound, `(knots,)`.
    """

    def __init__(self, fit, primary, other, noise):
        self.fit = fit
        self.noise = noise
        self.observed = np.asarray(primary[0], dtype=float)
        self.replicates = np.asarray(primary[1:], dtype=float)
        self._other = None if other is None else np.asarray(other, dtype=float)
        self.weighted = fit.covariance is not None and not isinstance(fit.covariance, str)
        self._keys = fit.table[["segment", "knot", "depth_km"]].to_dict("records")
        lo, hi = fit.bounds
        self.on_bound = fit.table["on_bound"].to_numpy(dtype=bool)
        self._upper = self.observed >= 0.5 * (lo + hi)        # which bound an estimate on one is on
        stopped = (self.replicates <= lo + BOUND_TOLERANCE) | (self.replicates >= hi - BOUND_TOLERANCE)
        self.pinned = stopped.mean(axis=0) if len(self.replicates) else np.zeros(self.observed.size)
        for keys, est, flag, share in zip(self._keys, self.observed, self.on_bound, self.pinned):
            name = f"segment {keys['segment']}, knot {keys['knot']}"
            if flag:
                warnings.warn(
                    f"{name}: the estimate ({est:.2f}) is on a bound ({lo:g}-{hi:g}) and "
                    f"{100 * share:.0f}% of the replicates stop there, so it has no sd: its interval "
                    "is one-sided, open on the bound's side, and the other knots' intervals hold "
                    "it on the bound", stacklevel=3)
            elif share > 0.025:
                warnings.warn(
                    f"{name}: {100 * share:.0f}% of the replicates stop on a bound ({lo:g}-{hi:g}), "
                    "which cuts that side of its interval short", stacklevel=3)

    def intervals(self, level=0.95):
        """Per knot: the estimate, its bootstrap sd, and normal and basic intervals at `level`.

        `on_bound` and `pinned` (the share of replicates stopped on a bound) say which
        rows to distrust. A knot whose estimate is on a bound has no sd or normal
        interval (NaN): the data would take it past the bound. Its `basic_lo` or
        `basic_hi` is then a one-sided limit -- the replicates' `level` quantile on
        the inside -- and the other end is NaN.
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        a = 100 * (1 - level) / 2
        rows = []
        for i, keys in enumerate(self._keys):
            est, values = float(self.observed[i]), self.replicates[:, i]
            flags = {"on_bound": bool(self.on_bound[i]), "pinned": float(self.pinned[i])}
            if self.on_bound[i]:
                inside = float(np.percentile(values, 100 * (1 - level) if self._upper[i] else 100 * level))
                rows.append({**keys, "estimate": est, "sd": np.nan, "normal_lo": np.nan,
                             "normal_hi": np.nan,
                             "basic_lo": inside if self._upper[i] else np.nan,
                             "basic_hi": np.nan if self._upper[i] else inside, **flags})
                continue
            sd = float(np.std(values, ddof=1))
            q_lo, q_hi = np.percentile(values - values.mean(), [a, 100 - a])
            rows.append({**keys, "estimate": est, "sd": sd,
                         "normal_lo": est - z * sd, "normal_hi": est + z * sd,
                         "basic_lo": est - q_hi, "basic_hi": est - q_lo, **flags})
        return pd.DataFrame(rows)

    def consistency(self):
        """Weighted minus iid estimate per knot, against its spread under the noise model.

        As `BootstrapResult.consistency`: with the forward model right, `|z|` is of
        order one.
        """
        if self._other is None:
            raise ValueError("bootstrap(compare=False) did not score the second estimator")
        diffs = self._other - np.vstack([self.observed[None, :], self.replicates])
        if self.weighted:                    # weighted minus iid either way round
            diffs = -diffs
        rows = []
        for i, keys in enumerate(self._keys):
            observed, replicate = diffs[0, i], diffs[1:, i]
            sd = float(np.std(replicate, ddof=1))
            rows.append({**keys, "difference": float(observed), "mean": float(replicate.mean()),
                         "sd": sd, "z": float((observed - replicate.mean()) / sd) if sd > 0 else np.nan})
        return pd.DataFrame(rows)

    def __repr__(self):
        sd = np.std(self.replicates, axis=0, ddof=1)
        parts = [f"segment {k['segment']} knot {k['knot']} {est:.2f} "
                 + ("(on a bound)" if flag else f"+- {s:.2g}")
                 for k, est, s, flag in zip(self._keys, self.observed, sd, self.on_bound)]
        return f"<ProfileBootstrap {len(self.replicates)} draws: " + ", ".join(parts) + ">"
