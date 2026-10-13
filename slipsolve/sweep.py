"""Segment-by-segment dip sweeps: which dip does each stretch of the fault want?

A geometry sweep lets the data choose the fault's shape. :class:`DipSweep` cuts the
trace into along-strike segments, gives each its own geometry from a grid -- a
planar dip (10 or 5 degree steps), or a dip that changes with depth -- and scores
every combination. Every combination is solved on the same samples with the same
regularization, so the geometry is the only thing that differs between them.

**Two ways to score.** ``run(score="evidence")`` integrates the slip, the ramp,
the noise scale and the smoothing weight out analytically
(:mod:`slipsolve.evidence`), so no rank has to be chosen. ``run(rank=...)``
scores the bounded solve's misfit at a fixed effective rank. ⚠️ A fixed rank is
not neutral for geometry: on the 2026-10-11 synthetic (Mw 7.7, two-sided data)
rms at rank 60 put a 60 degree fault at 57-58 even with an exact forward model,
because the data supported rank ~190; the evidence put it within 0.3.

**Uncertainty comes from** :meth:`DipSweep.bootstrap`, not from the evidence.
The iid evidence ranks geometries robustly, but the posterior it implies was 4-7x
too narrow on that synthetic. Its estimate re-made on synthetic data drawn from
a fitted noise model gave intervals that covered the true dip 82-97% of the time
at 95%. Weighting the evidence by the noise covariance was exact when the
forward model was, and 2-13 degrees wrong when it was not. The bootstrap reports
how far the weighted and unweighted estimates disagree; a large disagreement is
the sign of that failure.

**Why it is affordable.** Both engines build a node's Green's function as a sum
over the elements around it, each element computed from its own vertices alone
(:meth:`~slipsolve.layered.LayeredPointSource._columns`; the half-space engine
projects per-element columns onto the tents). With a node column on every segment
boundary no element straddles one, so the design matrix of ANY combination is a
sum of per-segment blocks, each depending on one segment's geometry::

    G(dips) = sum_j  P_j G_j(dip_j)

and the sweep costs one full assembly per geometry VALUE -- seven for 60-120
degrees in steps of 10 -- instead of one per combination, of which there are
``7**K``. After that a combination costs one solve.

**What the geometry is.** Each segment is what :meth:`FaultMesh.curved` builds for
it: every node stepped along the trace's local normal by ``depth / tan(dip)``, or,
for a dip that changes with depth, by the integral of ``cot(dip)`` down to that
depth. Adjacent segments meet on the surface trace and STEP apart below it, by
``depth * |cot(dip_a) - cot(dip_b)|`` -- 3.5 km at 20 km depth for 80 against 90.
The slip does not step: boundary nodes are shared parameters, so the slip field
stays continuous along strike and only the geometry kinks. This is the torn mesh
of the Venezuela record (2026-09-21) with slip continuity restored, not a blended
surface, so rebuild the winning combination as one surface
(``FaultMesh.curved(surface=...)``) before reporting a model on it.

**What is held fixed.** The samples, the ``(s, z)`` node layout, the bounds, the
ramp columns, the boundary rows and the smoothing operator. The operator is built
ONCE, on the layout embedded at ``reference_dip``, so the regularization is
identical across geometries rather than re-derived from each one's 3-D metric.

⚠️ Read the result as a profile, not a winner. On Venezuela the dip sweeps' power
was in rejecting the south-dipping side (13.97 against 16.89 mm rms), not in
choosing among the north-dipping angles, whose curves were shallow near their
minimum (2026-09-22). :meth:`SweepResult.profiles` -- each segment's score against
its dip, the others held at the best -- says more than the argmin. And with
one-sided coverage a dip preference can be a coverage artefact: recover a planted
geometry through the same samples (:meth:`DipSweep.predict` and ``data=``) before
believing one.

**A rigidity contrast as a sweep axis.** ``contrast=`` (a
:class:`~slipsolve.bimaterial.RigidityContrast`) and ``ratios=`` add one coordinate
per interval of the contrast, after the segments' dips. The scaled mode makes a
contrast a row and column scaling of the same blocks, so a ratio costs no
assembly, only a solve or a decomposition. Dip and contrast both make the fault's
two sides move differently, and on Venezuela's one-sided data they were
substitutes (2026-10-07); sweep them together. ⚠️ The scaled mode is an
approximation the data can see. On the Panama synthetic, given exact-mode data, the
sweep put the dip 5 degrees off and a ratio of 2 at 2.24, and it smeared a step
along strike (see :mod:`slipsolve.bimaterial`). Read its answer as where to look,
and take the dip, ratio and moment from ``Bimaterial(exact=True)``.
"""

import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from .basis import NodeBasis
from .covariance import CovarianceModel, fit_reml, observation_covariance
from .evidence import GeometryEvidence, smoothing_precision
from .inversion import DEFAULT_SHEAR_MODULUS, SlipInversion, SlipModel, _boundary_rows
from .layout import graded_layout
from .mesh import FaultMesh, _arc_length, _depth_levels
from .regularize import (check_ramp_tracks, laplace_beltrami, ramp_columns, ramp_spec,
                         slip_bounds)
from .resolution import JITTER
from .surface import FaultSurface, centres

#: Dips tried on every segment by default, in degrees: 60 to 120 in steps of 10.
#: ``FaultMesh.curved(uniform_dip=)``'s convention -- below 90 the fault dips
#: toward the trace's left-hand side, above 90 toward its right. Read
#: ``mesh.dip_direction`` to see which way that is on a given trace (on the
#: Venezuela trace, 80 dips north and 100 dips south).
DEFAULT_DIPS = tuple(range(60, 121, 10))

#: The scores :meth:`DipSweep.run` can rank combinations by.
SCORES = ("rms", "weighted_rms", "evidence")

#: A combination has to beat the current one by more than this fraction of its
#: score to replace it in the coordinate search. Without it, two geometries equal
#: to rounding could swap forever.
IMPROVEMENT = 1e-9

#: Resolution, in degrees, at which a segment's continuous dip estimate is read
#: off the spline through its profile.
REFINE_STEP = 0.005


def geometry_key(value):
    """One segment's geometry, as the sweep keys it.

    A number is a planar dip in degrees. A sequence of two or more numbers is a
    dip that changes with depth: the dips at evenly spaced depths from the top of
    the layout to its bottom, linear in between (see :func:`dip_profile_offsets`).
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
    """``70`` for a planar dip, ``70>50`` for one that runs from 70 to 50 with depth."""
    return f"{key:g}" if isinstance(key, float) else ">".join(f"{v:g}" for v in key)


def dip_profile_offsets(depths, dips, top=None, bottom=None):
    """Horizontal offset at each depth for a dip that changes with depth.

    ``dips`` are the dips (degrees) at evenly spaced depths from ``top`` to
    ``bottom`` (by default the shallowest and deepest of ``depths``), linear in
    depth between them. The offset is the integral of ``cot(dip)`` from ``top``,
    taken exactly on each piece (``ln sin`` is its antiderivative), so a
    constant profile reproduces ``depth / tan(dip)`` and the near-vertical
    snapping of :func:`~slipsolve.trace.dip_offset` is left to the mesh builder.
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


def segment_breaks(length, count=None, size=None, breaks=None):
    """Arc lengths of the segment boundaries, ``0 ... length``, in metres.

    Give exactly one of ``count`` (that many equal segments), ``size`` (segments
    of about that length) or ``breaks`` (the interior boundaries; ``0`` and
    ``length`` may be included or left out).
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

    Each segment is divided evenly into steps of at most ``edge_length``, so
    every boundary is a lattice column and no triangle straddles one -- the
    property the block assembly in :class:`DipSweep` rests on.
    """
    breaks = np.asarray(breaks, dtype=float)
    parts = [np.linspace(a, b, max(1, int(np.ceil((b - a) / edge_length))) + 1)[:-1]
             for a, b in zip(breaks[:-1], breaks[1:])]
    stations = np.concatenate(parts + [breaks[-1:]])
    return graded_layout(breaks[-1], depths, counts=stations.size, stations=stations)


def _box_qp(N, g, lo, hi, x0, max_iter=100):
    """Minimise ``x'Nx/2 - g'x`` under ``lo <= x <= hi``, exactly, by primal-dual active set.

    The same problem :meth:`SlipInversion.solve` hands ``lsq_linear``, written in
    its normal equations. Each step solves the free block exactly, so a converged
    answer is the minimiser and not an approximation to it. Started from a nearby
    solution, as in a sweep, it settles in a few steps.
    """
    at_lo = np.isfinite(lo) & (x0 <= lo + 1e-4)
    at_hi = np.isfinite(hi) & (x0 >= hi - 1e-4)
    for _ in range(max_iter):
        free = ~(at_lo | at_hi)
        x = np.where(at_lo, lo, np.where(at_hi, hi, 0.0))
        if free.any():
            rhs = g[free] - N[np.ix_(free, ~free)] @ x[~free]
            x[free] = cho_solve(cho_factor(N[np.ix_(free, free)], lower=True), rhs)
        mu = N @ x - g
        new_lo = np.isfinite(lo) & ((free & (x < lo)) | (at_lo & (mu > 0)))
        new_hi = np.isfinite(hi) & ((free & (x > hi)) | (at_hi & (mu < 0)))
        if np.array_equal(new_lo, at_lo) and np.array_equal(new_hi, at_hi):
            return x
        at_lo, at_hi = new_lo, new_hi
    raise RuntimeError("the bounded solve's active set did not settle")


def _better(a, b):
    """Whether score ``a`` beats ``b`` by more than :data:`IMPROVEMENT` of its size."""
    return a < b - IMPROVEMENT * abs(b)


def _ratio_grids(contrast, ratios):
    """One grid of ratios per interval of ``contrast``: shared, per interval, or its own."""
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


class DipSweep:
    """Sweep the geometry of each along-strike segment over a grid.

    ``obs`` is the fixed sample set every geometry is scored on. The segments are
    ``segments`` equal pieces of the trace (4 if nothing is said),
    ``segment_length`` metres each, or cut at ``breaks`` (arc lengths in metres).
    ``dips`` is the grid every segment draws from: planar dips, and tuples for a
    dip that changes with depth (see :func:`geometry_key`). ``max_depth``,
    ``edge_length``, ``down_dip_levels``, ``bias_w`` and ``depths`` place the
    nodes as in :meth:`FaultMesh.curved`. ``engine``, ``ramp`` and
    ``velocity_model`` mean what they mean to :class:`SlipInversion`, and the
    remaining keywords are :meth:`SlipInversion.solve`'s regularization and
    bounds; the slip is always nodal.

    ``reference_dip`` is the geometry the smoothing operator is built on, once,
    for every combination. 90 measures it in the ``(s, z)`` plane itself.

    ``contrast`` puts the fault between blocks of different rigidity, in the
    scaled mode of :mod:`slipsolve.bimaterial`. ``ratios`` makes its ratio a
    sweep axis: one grid for every interval of the contrast, or one grid per
    interval (a one-value grid holds that interval fixed). Without ``ratios``
    the contrast's own ratios hold. A combination is then the segments' dips
    followed by the intervals' ratios; anywhere a combination is taken, the dips
    alone mean the contrast's own ratios.

    Green's functions are assembled lazily, one geometry at a time, and kept:
    memory is about one full design matrix per geometry used.
    """

    def __init__(self, trace, obs, *, frame=None, segments=None, segment_length=None,
                 breaks=None, dips=DEFAULT_DIPS, max_depth=20e3, edge_length=3e3,
                 down_dip_levels=None, bias_w=1.0, depths=None, engine=None,
                 ramp="none", reference_dip=90.0, velocity_model=None, ss_ratio=1.0,
                 ds_ratio=3.0, boundary_ratio=1.0, sides=("bottom", "left", "right"),
                 polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0),
                 contrast=None, ratios=None):
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
        self._s = _arc_length(*resampled.to_local(self.frame))
        self.length = float(self._s[-1])
        if segments is None and segment_length is None and breaks is None:
            segments = 4
        self.breaks = segment_breaks(self.length, count=segments, size=segment_length,
                                     breaks=breaks)
        if depths is None:
            levels = (down_dip_levels if down_dip_levels is not None
                      else max(1, int(round(max_depth / edge_length))) + 1)
            depths = _depth_levels(0.0, float(max_depth), int(levels), float(bias_w))
        self.depths = np.asarray(depths, dtype=float)
        self.layout = segmented_layout(self.breaks, self.depths, self.edge_length)
        self.dips = tuple(dict.fromkeys(geometry_key(v) for v in dips))
        self.engine = engine
        self.ramp_kind = ramp_spec(ramp)
        self.velocity_model = velocity_model
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
        """A combination as text: ``80 70`` for the dips, then ``| r 1.5`` for any ratios."""
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
        """The ``cross(s, z)`` field of a dip that changes with depth, on the mesh lattice."""
        z_nodes = -self.depths[::-1]
        top, bottom = float(self.depths.min()), float(self.depths.max())
        offsets = dip_profile_offsets(self.depths, key, top, bottom)
        cross = np.broadcast_to(offsets[::-1, None], (self.depths.size, self._s.size))
        s_c, z_c = centres(self._s), centres(z_nodes)
        centre = dip_profile_offsets(-z_c, key, top, bottom)
        return FaultSurface(self._s, z_nodes, np.array(cross), s_c, z_c,
                            np.array(np.broadcast_to(centre[:, None], (z_c.size, s_c.size))),
                            attrs={"dips": [float(v) for v in key], "profile": "linear in depth"})

    def _mesh(self, key):
        """The whole layout embedded at one geometry."""
        key = geometry_key(key)
        if key not in self._meshes:
            geometry = ({"uniform_dip": key} if isinstance(key, float)
                        else {"surface": self._profile_surface(key)})
            self._meshes[key] = FaultMesh.from_layout(
                self.layout, self.trace, self.frame, edge_length=self.edge_length, **geometry)
        return self._meshes[key]

    def segment_mesh(self, j, dip):
        """Segment ``j`` (0-based) of the layout, embedded at ``dip``.

        Its nodes are ``segment_nodes[j]`` of the global parameter vector, in
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
        """Every segment at its own geometry, in order -- the stepped fault ``dips`` describes."""
        return [self.segment_mesh(j, v) for j, v in enumerate(self._combination(dips))]

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
    def _segment_blocks(self, dip):
        """Each segment's Green's columns at ``dip``, one assembly of the whole fault."""
        dip = geometry_key(dip)
        if dip not in self._blocks:
            self._blocks[dip] = [
                np.asarray(SlipInversion(self.segment_mesh(j, dip), self.obs, engine=self.engine,
                                         basis="node").g, dtype=float)
                for j in range(self.n_segments)]
        return self._blocks[dip]

    def design(self, dips):
        """The nodal Green's matrix of a combination, ``(n_obs, 2 * n)``.

        For one geometry on every segment it is exactly the matrix
        :class:`SlipInversion` assembles on that mesh, and with a contrast, the
        matrix a scaled-mode :class:`~slipsolve.bimaterial.Bimaterial` would.
        """
        combo = self._combination(dips)
        k = self.n_segments
        g = np.zeros((self.d.size, 2 * self.n))
        for j, dip in enumerate(combo[:k]):
            nodes = self.segment_nodes[j]
            g[:, np.concatenate([nodes, self.n + nodes])] += self._segment_blocks(dip)[j]
        if self.n_ratios:
            rows, columns, _ = self._contrast_factors(combo[k:])
            g *= rows[:, None]
            g *= columns[None, :]
        return g

    def _contrast_factors(self, ratios):
        """Row factors, nodal column factors and element rigidity factors at ``ratios``.

        The scaled mode of :mod:`slipsolve.bimaterial`: each sample's row times
        its block's ``a``, each element's slip times ``1/m``. A node takes its
        tent's area-weighted mean of ``1/m``, measured on the reference mesh.
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
        """What parameters ``x`` (slip, then any ramp) predict on geometry ``dips``.

        With noise added, the ``data=`` a planted-geometry recovery test needs.
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
        """Effective rank: the trace of the slip block of ``(N + damping)^-1 N``."""
        total = N + self._damping(lam)
        total.flat[:: total.shape[0] + 1] += JITTER * np.trace(total) / total.shape[0]
        m = 2 * self.n
        columns = cho_solve(cho_factor(total, lower=True), np.eye(total.shape[0])[:, :m])
        return float(np.sum(columns * N[:, :m]))

    def _lam_for_rank(self, N, rank, guess=None):
        """The smoothing weight giving effective rank ``rank``, by Brent in log-weight."""
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
        total.flat[:: total.shape[0] + 1] += JITTER * np.trace(total) / total.shape[0]
        x0 = self._last["x"]
        if x0 is None or x0.size != total.shape[0]:
            x0 = np.zeros(total.shape[0])
        x = _box_qp(total, g, self._lo, self._hi, x0)
        self._last["x"] = x
        return x

    def _moment_magnitude(self, dips, x):
        """Integrated Mw, summed over the segments' own meshes.

        ``SlipModel.moment`` integrates whatever exposes a mesh, a basis and the
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

        Exactly one of ``rank`` (an effective rank to hold) or ``smoothing`` (a
        fixed weight, as :meth:`SlipInversion.solve` takes it). ``data``
        replaces the observed LOS with another vector on the same samples.
        ``x`` in the result is the parameter vector: strike-slip for every node,
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
        """The :class:`~slipsolve.evidence.GeometryEvidence` this sweep scores with.

        One per noise model, kept, so every geometry it scores shares one
        ``alpha^2`` grid. The prior is this sweep's own smoothing operator with
        its ``sides`` pinned to zero outright.
        """
        key = _evidence_cache_key(covariance)
        if key not in self._evidence:
            q, free = smoothing_precision(self.reference, sides=self.sides,
                                          operator=self._smooth)
            self._evidence[key] = GeometryEvidence(self.obs, q, free, self.ramp,
                                                   covariance=covariance)
        return self._evidence[key]

    def _evidence_solution(self, ev, dec, dh, g, d, alpha2):
        """Full parameter vectors ``(2n + k, m)`` at each column's ``alpha2``."""
        m_free = ev.posterior_mean(dec, dh, alpha2)
        x = np.zeros((2 * self.n + self.ramp.shape[1], m_free.shape[1]))
        x[ev.free] = m_free
        if self.ramp.shape[1]:
            residual = np.asarray(d, dtype=float).reshape(self.d.size, -1) - g[:, ev.free] @ m_free
            x[2 * self.n:] = ev.ramp_coefficients(residual)
        return x

    def evidence(self, dips, *, data=None, covariance=None):
        """Score one combination by its marginal likelihood; a dict like :meth:`evaluate`'s.

        ``log_evidence`` integrates the slip, ramp, noise scale and smoothing
        weight out (see :mod:`slipsolve.evidence`); ``alpha2``, ``rank`` and
        ``sigma2`` are at the integrand's mode, and ``x`` is the posterior-mean
        solution there -- unbounded, so its ``Mw`` and ``peak`` are a guide only.
        ``covariance`` is the noise: ``None`` (iid, the robust default),
        ``"weights"``, or :class:`CovarianceModel`\\ s per track (whitened).
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
        """Search the combinations of dips; returns a :class:`SweepResult`.

        ``score="evidence"`` ranks by the marginal likelihood (no ``rank`` or
        ``smoothing``; ``covariance`` as :meth:`evidence` takes it).
        ``"rms"`` (unweighted, the figure the Venezuela record quotes) and
        ``"weighted_rms"`` (the solve's own data term) rank the bounded solve at
        a fixed ``rank`` or ``smoothing``.

        ``method="exhaustive"`` scores every combination; ``"coordinate"`` starts
        from the best uniform geometry (or ``start``) and repeatedly re-chooses one
        segment with the others held, until a whole pass changes nothing.
        ``"auto"`` is exhaustive when there are at most ``max_evaluations``
        combinations. ``fix={segment: dip}`` (0-based) holds segments at a
        geometry, which need not be on the grid; hold a contrast interval with a
        one-value grid in ``ratios=``. A contrast's ratios are coordinates like
        the dips, after them. ``data`` replaces the observed LOS; ``log`` is
        called with one line per new evaluation.

        ⚠️ The coordinate search finds a combination no single segment can
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
        count = int(np.prod([len(c) for c in candidates]))
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

    # -- the uncertainty ----------------------------------------------------
    def bootstrap(self, result, *, noise="reml", draws=100, seed=0, compare=True, log=None):
        """Parametric bootstrap of an evidence sweep: each segment's dip with an honest spread.

        The best combination's posterior-mean model predicts the data; ``draws``
        synthetic data sets add noise drawn from ``noise`` to that prediction,
        and every combination ``result`` evaluated is re-scored on each. A
        segment's estimate is read off the spline through its profile, so it is
        continuous and not tied to the grid.

        ``noise`` is ``"reml"`` -- fitted to the best model's residual by
        :func:`~slipsolve.covariance.fit_reml`, per track -- or a
        :class:`CovarianceModel` (every track) or ``{track: model}`` mapping.
        With ``compare`` (the default), each replicate is also scored by the
        other evidence: the noise-weighted one if ``result`` was iid, the iid
        one if it was weighted. :meth:`BootstrapResult.consistency` then says
        whether the two estimates disagree by more than noise explains.

        ⚠️ Replicates are re-scored on the combinations ``result`` already
        evaluated, so for a coordinate search the spread is conditional on the
        other segments. Each replicate keeps its own best combination where that
        combination's profile was scored; otherwise it falls back to ``result``'s.
        Give the grid a step finer than the spread you expect near the best:
        the estimate is splined between grid points, but a profile two points
        wide carries little information. Mesh and other forward-model error are
        NOT in these intervals. The weighted-vs-iid comparison is what detects
        them.
        """
        if result.score != "evidence":
            raise ValueError("bootstrap re-runs the evidence estimator; give it the result of "
                             "run(score='evidence')")
        combos = list(result._rows)
        d_obs = self.d if result.data is None else np.asarray(result.data, dtype=float)
        pred = self.predict(result.best, result.solution(result.best))
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
        data = np.column_stack([d_obs, replicates])          # column 0 is the observed data

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

    ``log`` splines a ratio's profile in log ratio, where its grid is usually
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
    """Every combination a :meth:`DipSweep.run` evaluated, and the best of them.

    ``table`` has one row per combination -- ``dip_1 ... dip_K`` in segment order
    (see ``segments`` for where each lies; a dip that changes with depth is a
    tuple), then ``ratio_1 ...`` for a contrast's intervals, its score and the
    other misfits, ``Mw``, the peak slip, and for the bounded solve the smoothing
    weight, effective rank and fraction of slip parameters on a bound -- sorted
    best first. ``score`` is the quantity minimised: the misfit, or minus the log
    evidence, whose drop from the best is ``delta_log_evidence``.
    ``solution(dips)`` returns the parameter vector of any evaluated combination.
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
        return self._solutions[self._combination(dips)]

    def profiles(self):
        """Each coordinate's score against its value, the others at the best.

        Long form: ``parameter`` (``dip`` or ``ratio``), ``segment`` (1-based, for
        a dip) or ``interval`` (1-based, for a contrast's ratio), ``dip`` or
        ``ratio``, ``score`` and ``delta`` -- the excess over the best
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
    """What :meth:`DipSweep.bootstrap` found: estimates, their spread, and a consistency check.

    ``observed`` is each segment's estimate on the data and ``replicates`` the
    ``(draws, segments)`` estimates on the synthetic data sets, both continuous
    for planar dips (and the best key for a dip that changes with depth).
    ``noise`` holds the per-track models the replicates were drawn from.
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
        # others take up what it would have done. Measured with dip and ratio swept
        # together: a dip pinned at the edge left the ratio's interval 2-3x too narrow,
        # and it missed the truth on 2 of 6 seeds (6 of 6 with the grid widened).
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
        """The identifying columns of coordinate ``j``'s row."""
        if j < self.n_dips:
            return {"parameter": "dip", "segment": j + 1, "interval": np.nan}
        return {"parameter": "ratio", "segment": np.nan, "interval": j - self.n_dips + 1}

    def intervals(self, level=0.95):
        """Per coordinate: the estimate, its bootstrap sd, and two intervals at ``level``.

        ``normal`` is estimate -/+ z sd; ``basic`` reflects the replicates'
        quantiles about the estimate. A dip that changes with depth gets the
        share of replicates choosing each candidate instead (``share``). A
        contrast's ratios follow the dips, with ``parameter`` saying which is which.
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

        ``z`` is the observed difference less its bootstrap mean, over its
        bootstrap sd. With the forward model right, ``|z|`` is of order one; on
        the 2026-10-11 synthetic it was 0.4-0.7 when the mesh was exact and 5.7-47
        when a coarser mesh had biased the weighted posterior by 2-13 degrees.
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
