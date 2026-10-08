"""Segment-by-segment dip sweeps: which dip does each stretch of the fault want?

A geometry sweep lets the data choose the fault's shape. :class:`DipSweep` cuts the
trace into along-strike segments, gives each its own planar dip from a grid (10 or
5 degree steps), and scores a combination of dips by how well the best slip model
on that geometry fits the observations. Every combination is solved on the same
samples, with the same regularization, at the same effective rank, so the geometry
is the only thing that differs between them.

**Why it is affordable.** Both engines build a node's Green's function as a sum
over the elements around it, each element computed from its own vertices alone
(:meth:`~slipsolve.layered.LayeredPointSource._columns`; the half-space engine
projects per-element columns onto the tents). With a node column on every segment
boundary no element straddles one, so the design matrix of ANY combination is a
sum of per-segment blocks, each depending on one segment's dip::

    G(dips) = sum_j  P_j G_j(dip_j)

and the sweep costs one full assembly per dip VALUE -- seven for 60-120 degrees in
steps of 10 -- instead of one per combination, of which there are ``7**K``. After
that a combination costs one normal-equation solve.

**What the geometry is.** Each segment is what :meth:`FaultMesh.curved` builds for
its dip: every node stepped ``depth / tan(dip)`` along the trace's local normal.
Adjacent segments meet on the surface trace and STEP apart below it, by
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
Combinations are compared at a fixed **effective rank** by default, because one
smoothing weight resolves a different number of parameters on a different
geometry.

⚠️ Read the result as a profile, not a winner. On Venezuela the dip sweeps' power
was in rejecting the south-dipping side (13.97 against 16.89 mm rms), not in
choosing among the north-dipping angles, whose curves were shallow near their
minimum (2026-09-22). :meth:`SweepResult.profiles` -- each segment's score against
its dip, the others held at the best -- says more than the argmin. And with
one-sided coverage a dip preference can be a coverage artefact: recover a planted
geometry through the same samples (:meth:`DipSweep.predict` and ``data=``) before
believing one.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from .basis import NodeBasis
from .inversion import DEFAULT_SHEAR_MODULUS, SlipInversion, SlipModel, _boundary_rows
from .layout import graded_layout
from .mesh import FaultMesh, _arc_length, _depth_levels
from .regularize import laplace_beltrami, ramp_columns, slip_bounds
from .resolution import JITTER

#: Dips tried on every segment by default, in degrees: 60 to 120 in steps of 10.
#: ``FaultMesh.curved(uniform_dip=)``'s convention -- below 90 the fault dips
#: toward the trace's left-hand side, above 90 toward its right. Read
#: ``mesh.dip_direction`` to see which way that is on a given trace (on the
#: Venezuela trace, 80 dips north and 100 dips south).
DEFAULT_DIPS = tuple(range(60, 121, 10))

#: The scores :meth:`DipSweep.run` can rank combinations by.
SCORES = ("rms", "weighted_rms")

#: A combination has to beat the current one by more than this fraction of its
#: score to replace it in the coordinate search. Without it, two geometries equal
#: to rounding could swap forever.
IMPROVEMENT = 1e-9


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


class DipSweep:
    """Sweep the dip of each along-strike segment over a grid of dips.

    ``obs`` is the fixed sample set every geometry is scored on. The segments are
    ``segments`` equal pieces of the trace (4 if nothing is said),
    ``segment_length`` metres each, or cut at ``breaks`` (arc lengths in metres).
    ``max_depth``, ``edge_length``, ``down_dip_levels``, ``bias_w`` and
    ``depths`` place the nodes as in :meth:`FaultMesh.curved`. ``engine``,
    ``ramp`` and ``velocity_model`` mean what they mean to
    :class:`SlipInversion`, and the remaining keywords are
    :meth:`SlipInversion.solve`'s regularization and bounds; the slip is always
    nodal.

    ``reference_dip`` is the geometry the smoothing operator is built on, once,
    for every combination. 90 measures it in the ``(s, z)`` plane itself.

    Green's functions are assembled lazily, one dip at a time, and kept: memory
    is about one full design matrix per dip used.
    """

    def __init__(self, trace, obs, *, frame=None, segments=None, segment_length=None,
                 breaks=None, dips=DEFAULT_DIPS, max_depth=20e3, edge_length=3e3,
                 down_dip_levels=None, bias_w=1.0, depths=None, engine=None,
                 ramp="none", reference_dip=90.0, velocity_model=None, ss_ratio=1.0,
                 ds_ratio=3.0, boundary_ratio=1.0, sides=("bottom", "left", "right"),
                 polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0)):
        self.trace, self.obs = trace, obs
        self.frame = frame or trace.local_frame()
        self.edge_length = float(edge_length)
        # The arc length exactly as `FaultMesh.curved` measures it, or the layout
        # and the meshes built on it would disagree about where the trace ends.
        resampled = trace.resample(self.edge_length, self.frame)
        self.length = float(_arc_length(*resampled.to_local(self.frame))[-1])
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
        self.dips = tuple(float(v) for v in dips)
        self.engine = engine
        self.ramp_kind = str(ramp)
        self.velocity_model = velocity_model
        self.reference_dip = float(reference_dip)
        self._meshes, self._submeshes, self._blocks = {}, {}, {}
        self._memo = {}
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
        self.ramp, self.ramp_labels = ramp_columns(obs, ramp)

        self.options = dict(ss_ratio=float(ss_ratio), ds_ratio=float(ds_ratio),
                            boundary_ratio=float(boundary_ratio), sides=list(sides),
                            polarity=None if polarity is None else list(polarity),
                            strike=list(strike), dip=list(dip))
        smooth = laplace_beltrami(self.reference, ss_ratio, ds_ratio)
        self._rows = smooth.shape[0]
        self._SS = (smooth.T @ smooth).toarray()
        bnd = _boundary_rows(NodeBasis(self.reference), tuple(sides), boundary_ratio)
        self._BB = (bnd.T @ bnd).toarray()
        lo, hi = slip_bounds(self.n, strike, dip, polarity)
        k = self.ramp.shape[1]
        self._lo = np.concatenate([lo, np.full(k, -np.inf)])
        self._hi = np.concatenate([hi, np.full(k, np.inf)])

    # -- geometry -----------------------------------------------------------
    @property
    def n_segments(self):
        return int(self.breaks.size - 1)

    @property
    def segments(self):
        """One row per segment: its index and arc-length extent in km."""
        return pd.DataFrame({"segment": np.arange(1, self.n_segments + 1),
                             "start_km": self.breaks[:-1] / 1e3,
                             "end_km": self.breaks[1:] / 1e3})

    def _mesh(self, dip):
        """The whole layout embedded at one uniform dip."""
        dip = float(dip)
        if dip not in self._meshes:
            self._meshes[dip] = FaultMesh.from_layout(
                self.layout, self.trace, self.frame, uniform_dip=dip,
                edge_length=self.edge_length)
        return self._meshes[dip]

    def segment_mesh(self, j, dip):
        """Segment ``j`` (0-based) of the layout, embedded at ``dip``.

        Its nodes are ``segment_nodes[j]`` of the global parameter vector, in
        that order; a boundary node appears in both segments it bounds.
        """
        key = (int(j), float(dip))
        if key not in self._submeshes:
            mesh = self._mesh(dip)
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
        """Every segment at its own dip, in order -- the stepped geometry ``dips`` describes."""
        return [self.segment_mesh(j, v) for j, v in enumerate(self._combination(dips))]

    def _combination(self, dips):
        dips = tuple(float(v) for v in np.atleast_1d(dips))
        if len(dips) == 1 and self.n_segments > 1:
            dips = dips * self.n_segments
        if len(dips) != self.n_segments:
            raise ValueError(f"need one dip per segment ({self.n_segments}), got {len(dips)}")
        return dips

    # -- the design matrix ---------------------------------------------------
    def _segment_blocks(self, dip):
        """Each segment's Green's columns at ``dip``, one assembly of the whole fault."""
        dip = float(dip)
        if dip not in self._blocks:
            self._blocks[dip] = [
                np.asarray(SlipInversion(self.segment_mesh(j, dip), self.obs, engine=self.engine,
                                         basis="node").g, dtype=float)
                for j in range(self.n_segments)]
        return self._blocks[dip]

    def design(self, dips):
        """The nodal Green's matrix of a combination of dips, ``(n_obs, 2 * n)``.

        For one dip on every segment it is exactly the matrix
        :class:`SlipInversion` assembles on the uniform-dip mesh.
        """
        g = np.zeros((self.d.size, 2 * self.n))
        for j, dip in enumerate(self._combination(dips)):
            nodes = self.segment_nodes[j]
            g[:, np.concatenate([nodes, self.n + nodes])] += self._segment_blocks(dip)[j]
        return g

    def _full_design(self, dips):
        g = self.design(dips)
        return np.hstack([g, self.ramp]) if self.ramp.shape[1] else g

    def predict(self, dips, x):
        """What parameters ``x`` (slip, then any ramp) predict on geometry ``dips``.

        With noise added, the ``data=`` a planted-geometry recovery test needs.
        """
        return self._full_design(dips) @ np.asarray(x, dtype=float)

    # -- the solve ----------------------------------------------------------
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
        total = 0.0
        for j, dip in enumerate(dips):
            mesh = self.segment_mesh(j, dip)
            nodes = self.segment_nodes[j]
            view = SimpleNamespace(basis=NodeBasis(mesh), mesh=mesh,
                                   strike_slip=x[nodes], dip_slip=x[self.n + nodes])
            total += SlipModel.moment(view, shear_modulus=rigidity)
        return float((2.0 / 3.0) * (np.log10(total) - 9.1)) if total > 0 else float("nan")

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
        residual = d - design @ x

        m = 2 * self.n
        slip = np.hypot(x[: self.n], x[self.n: m])
        bounded = np.isfinite(self._lo[:m]) | np.isfinite(self._hi[:m])
        at_bound = bounded & ((x[:m] <= self._lo[:m] + 1e-6) | (x[:m] >= self._hi[:m] - 1e-6))
        out = {"dips": dips, "lam": lam, "rank": self._rank(N, lam),
               "rms": float(np.sqrt(np.mean(residual ** 2))),
               "weighted_rms": float(np.sqrt(np.sum((self.w * residual) ** 2)
                                             / np.sum(self.w ** 2))),
               "Mw": self._moment_magnitude(dips, x), "peak": float(slip.max()),
               "at_bound": float(at_bound.sum() / m), "x": x}
        for t in np.unique(self.track):
            out[f"rms_{t}"] = float(np.sqrt(np.mean(residual[self.track == t] ** 2)))
        if data is None:
            self._memo[key] = out
        return out

    # -- the search ---------------------------------------------------------
    def run(self, *, rank=None, smoothing=None, score="rms", method="auto",
            max_evaluations=1000, passes=4, start=None, fix=None, data=None, log=None):
        """Search the combinations of dips; returns a :class:`SweepResult`.

        ``method="exhaustive"`` scores every combination; ``"coordinate"`` starts
        from the best uniform dip (or ``start``) and repeatedly re-chooses one
        segment's dip with the others held, until a whole pass changes nothing.
        ``"auto"`` is exhaustive when there are at most ``max_evaluations``
        combinations. ``fix={segment: dip}`` (0-based) holds segments at a dip,
        which need not be on the grid. ``score`` is ``"rms"`` (unweighted, the
        figure the Venezuela record quotes) or ``"weighted_rms"`` (the solve's own
        data term). ``log`` is called with one line per new evaluation.

        ⚠️ The coordinate search finds a combination no single segment can
        improve, which is not guaranteed to be the global best when segments
        trade off against each other; compare it against an exhaustive run on a
        coarser grid when that matters.
        """
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        candidates = [list(self.dips) for _ in range(self.n_segments)]
        for j, value in (fix or {}).items():
            if not 0 <= int(j) < self.n_segments:
                raise ValueError(f"fix names segment {j}; segments are 0-{self.n_segments - 1}")
            candidates[int(j)] = [float(value)]
        count = int(np.prod([len(c) for c in candidates]))
        if method == "auto":
            method = "exhaustive" if count <= max_evaluations else "coordinate"
        if method not in ("exhaustive", "coordinate"):
            raise ValueError(f"method must be auto, exhaustive or coordinate, got {method!r}")

        rows = {}

        def value(combo):
            if combo not in rows:
                rows[combo] = self.evaluate(combo, rank=rank, smoothing=smoothing, data=data)
                if log is not None:
                    r = rows[combo]
                    log(f"dips {' '.join(f'{v:g}' for v in combo)}: {score} "
                        f"{1e3 * r[score]:.3f} mm, rank {r['rank']:.1f}, Mw {r['Mw']:.4f}")
            return rows[combo][score]

        if method == "exhaustive":
            for combo in itertools.product(*candidates):
                value(tuple(float(v) for v in combo))
            best = min(rows, key=value)
        else:
            if start is None:
                uniform = [tuple(c[0] if len(c) == 1 else v for c in candidates)
                           for v in self.dips]
                current = min(uniform, key=value)
            else:
                current = tuple(c[0] if len(c) == 1 else float(v)
                                for c, v in zip(candidates, self._combination(start)))
                value(current)
            for _ in range(int(passes)):
                changed = False
                for j, options in enumerate(candidates):
                    trials = [current[:j] + (float(v),) + current[j + 1:] for v in options]
                    challenger = min(trials, key=value)
                    if value(challenger) < value(current) * (1.0 - IMPROVEMENT):
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
                        value(best[:j] + (float(v),) + best[j + 1:])
                top = min(rows, key=value)
                if value(top) >= value(best) * (1.0 - IMPROVEMENT):
                    break
                best = top
        return SweepResult(self, rows, best, method=method, score=score, rank=rank,
                           smoothing=smoothing)


class SweepResult:
    """Every combination a :meth:`DipSweep.run` evaluated, and the best of them.

    ``table`` has one row per combination -- ``dip_1 ... dip_K`` in segment order
    (see ``segments`` for where each lies), its score, the other misfits, the
    smoothing weight and effective rank, ``Mw``, the peak slip and the fraction of
    slip parameters on a bound -- sorted best first. ``solution(dips)`` returns
    the parameter vector of any evaluated combination.
    """

    def __init__(self, sweep, rows, best, method, score, rank, smoothing):
        self.segments = sweep.segments
        self.best = tuple(best)
        self.method, self.score, self.rank, self.smoothing = method, score, rank, smoothing
        self._solutions = {combo: r["x"] for combo, r in rows.items()}
        records = []
        for combo, r in rows.items():
            record = {f"dip_{j + 1}": v for j, v in enumerate(combo)}
            record.update({k: v for k, v in r.items() if k not in ("dips", "x")})
            record["score"] = r[score]
            records.append(record)
        self.table = (pd.DataFrame(records).sort_values("score", kind="stable")
                      .reset_index(drop=True))

    def solution(self, dips):
        return self._solutions[tuple(float(v) for v in dips)]

    def profiles(self):
        """Each segment's score against its dip, the other segments at the best.

        Long form: ``segment`` (1-based), ``dip``, ``score`` and ``delta`` -- the
        excess over the best combination's score. Only evaluated combinations
        appear, so an exhaustive run and a finished coordinate search give
        complete profiles.
        """
        columns = [f"dip_{j + 1}" for j in range(len(self.best))]
        best_score = float(self.table["score"].iloc[0])
        rows = []
        values = self.table[columns].to_numpy()
        for j in range(len(self.best)):
            others = np.delete(values, j, axis=1) == np.delete(np.asarray(self.best), j)
            hit = self.table[others.all(axis=1)]
            for _, r in hit.iterrows():
                rows.append({"segment": j + 1, "dip": r[columns[j]], "score": r["score"],
                             "delta": r["score"] - best_score})
        return (pd.DataFrame(rows).sort_values(["segment", "dip"])
                .reset_index(drop=True))

    def __repr__(self):
        best = self.table.iloc[0]
        return (f"<SweepResult {self.method}, {len(self.table)} combinations, best dips "
                f"{' '.join(f'{v:g}' for v in self.best)}: {self.score} "
                f"{1e3 * best['score']:.3f} mm at rank {best['rank']:.1f}, Mw {best['Mw']:.4f}>")
