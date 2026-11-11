"""The smoothing weight by effective rank, chosen by blocked cross-validation.

`NormalSystem` holds the normal equations of an assembled `SlipInversion` in memory:
its bounded solve at any smoothing weight, the effective rank there, and the weight that
gives a chosen rank. It is the system `SlipInversion.solve(method="active_set")`
minimises, built from the same regularization blocks, so a weight means the same thing
in both.

`slab_folds` cuts the data into slabs along the fault. `choose_rank` scores held-out data
over a ladder of weights, each fold solved from the full normal equations minus its own
rows, and picks the minimum. Given several partitions of the same data (the slab edges
shifted), it averages their curves before choosing, which removes most of the dependence
on where the edges fall.

```python
from slipsolve.selection import NormalSystem, choose_rank, slab_folds

system = NormalSystem(inversion, boundary_ratio=1e3, polarity=(-1, 0, 0))
lam = system.lam_for_rank(60.0)
folds = [slab_folds(obs, 12, direction, shift=k / 4)[0] for k in range(4)]
choice = choose_rank(system, folds)
model = inversion.solve(smoothing=choice.lam, boundary_ratio=1e3, polarity=(-1, 0, 0))
```
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .inversion import box_qp, prior_target, regularization_blocks
from .regularize import slip_bounds

#: The default ladder of effective ranks a cross-validation spans.
CV_RANKS = (8.0, 250.0)
#: The ratio between neighbouring smoothing weights on the coarse ladder.
CV_STEP = 2.0 ** 0.5
#: Two local minima of a cross-validation curve are one basin unless the curve rises by
#: more than this fraction between them.
BASIN_RISE = 0.01
#: A bounded solve is exact below this KKT residual (relative to the right-hand side).
KKT_MAX = 1e-6
#: Above this KKT residual a bounded solve is refused. Between the two it is accepted and
#: counted in `NormalSystem.loose_solves`: at heavy damping the free block's Cholesky
#: solve is inexact by about this much.
KKT_ACCEPT = 1e-3
#: Ridge added to the diagonal, relative to the mean diagonal, when the effective rank
#: is computed, so that a nearly singular data block still factorises.
RANK_JITTER = 1e-12


class GridEdge(RuntimeError):
    """The cross-validation minimum lies on the end of its ladder; `rank` says which end."""

    def __init__(self, rank):
        super().__init__(f"the cross-validation minimum is on the grid edge (rank {rank:.1f}); widen `ranks`")
        self.rank = float(rank)


def _projected_newton(N, g, lo, hi, x0, max_iter=200):
    """Bertsekas' projected Newton for `min x'Nx/2 - g'x` on a box.

    Newton on the free set, a scaled gradient on the variables binding within a small
    margin, and an Armijo search along the projection arc. It is monotone, so it finds the
    active set where the primal-dual active set cycles; `box_qp` then finishes exactly.
    """
    x = np.clip(np.zeros_like(g) if x0 is None else np.asarray(x0, float), lo, hi)
    diag = np.maximum(np.diag(N), np.finfo(float).tiny)
    fx = 0.5 * x @ (N @ x) - g @ x
    scale = max(float(np.abs(g).max()), np.finfo(float).tiny)
    for _ in range(max_iter):
        grad = N @ x - g
        step = x - np.clip(x - grad, lo, hi)
        if np.abs(step).max() <= 1e-13 * scale:
            break
        eps = min(1e-3, float(np.abs(step).max()))
        bind = ((x <= lo + eps) & (grad > 0)) | ((x >= hi - eps) & (grad < 0))
        free = ~bind
        d = -grad / diag
        if free.any():
            d[free] = -cho_solve(cho_factor(N[np.ix_(free, free)], lower=True), grad[free])
        alpha, xn, fn = 1.0, x, fx
        for _ in range(60):
            xn = np.clip(x + alpha * d, lo, hi)
            fn = 0.5 * xn @ (N @ xn) - g @ xn
            if fn <= fx + 1e-4 * grad @ (xn - x):
                break
            alpha *= 0.5
        if fn >= fx:
            break
        x, fx = xn, fn
    return x


def _kkt(N, g, lo, hi, x):
    mu = N @ x - g
    free = (x > lo + 1e-10) & (x < hi - 1e-10)
    return float(np.abs(mu[free]).max(initial=0.0)) / max(float(np.abs(g).max()), np.finfo(float).tiny)


def _refine(N, g, lo, hi, x, kkt, steps=3):
    """Iterative refinement on the free block, for a solve left inexact by heavy damping."""
    for _ in range(steps):
        if kkt <= KKT_MAX:
            break
        free = (x > lo + 1e-10) & (x < hi - 1e-10)
        if not free.any():
            break
        x = x.copy()
        x[free] += cho_solve(cho_factor(N[np.ix_(free, free)], lower=True), (g - N @ x)[free])
        x = np.clip(x, lo, hi)
        kkt = _kkt(N, g, lo, hi, x)
    return x, kkt


def solve_box(N, g, lo, hi, x0=None, max_iter=60):
    """`(x, kkt)` minimising `x'Nx/2 - g'x` on `lo <= x <= hi`.

    `slipsolve.inversion.box_qp` first. If its active set does not settle, projected
    Newton from the same start finds one and `box_qp` finishes from there. A solve left
    inexact by heavy damping is refined on its free block.
    """
    try:
        r = box_qp(N, g, lo, hi, x0=x0, max_iter=max_iter)
        x, kkt = np.asarray(r.x, float), float(r.kkt)
    except RuntimeError:
        x1 = _projected_newton(N, g, lo, hi, x0)
        try:
            r = box_qp(N, g, lo, hi, x0=x1, max_iter=max_iter)
            x, kkt = np.asarray(r.x, float), float(r.kkt)
        except RuntimeError:
            x, kkt = x1, _kkt(N, g, lo, hi, x1)
    return _refine(N, g, lo, hi, x, kkt) if kkt > KKT_MAX else (x, kkt)


def renormalised(weights, track, drop):
    """`weights` with the `drop` rows zeroed and each track rescaled to its original influence.

    A track's influence is the sum of its squared weights: what it contributes to the
    normal equations.
    """
    out = np.where(drop, 0.0, weights)
    for name in np.unique(track):
        rows = track == name
        kept = float(np.sum(out[rows] ** 2))
        if kept > 0:
            out[rows] *= np.sqrt(float(np.sum(weights[rows] ** 2)) / kept)
    return out


class NormalSystem:
    """The normal equations of one assembled `SlipInversion`, solved exactly at any smoothing weight.

    The keywords are `SlipInversion.solve`'s, and the system is the one its active-set
    solve minimises: `(WA)'(WA) + (lam / n)^2 S'S + B'B`, with `S` the smoothing operator
    and its `n` rows, `B` the zero-slip edges and any nuisance prior, and the same bounds.
    A dict of solve options can be passed to both.

    Rows may be dropped (`drop=`, a boolean mask): each track is then rescaled to its
    original total influence, as a held-out test needs. A nuisance column left with no row
    (a region whose data are all dropped) is pinned to zero.

    Only per-row weights are supported: an inversion with a full noise covariance is
    refused.

    Attributes:
        inversion: The `SlipInversion` the system was built from.
        n: The number of basis functions; the slip occupies the first `2 n` parameters.
        P: The number of parameters, slip and nuisance.
        A: The design matrix, Green's functions then nuisance columns.
        d: The data.
        rows: The smoothing operator's row count, which divides the weight.
        lo: The lower bound of every parameter.
        hi: The upper bound of every parameter.
        loose_solves: The KKT residuals of solves accepted between `KKT_MAX` and `KKT_ACCEPT`.
    """

    def __init__(self, inversion, ss_ratio=1.0, ds_ratio=3.0, boundary_ratio=1.0,
                 sides=("bottom", "left", "right"), polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0)):
        if getattr(inversion, "_chol", None) is not None:
            raise ValueError("NormalSystem supports per-row weights only; this inversion has a full covariance")
        self.inversion = inversion
        self.n = int(inversion.basis.n_basis)
        self.A = np.hstack([inversion.g, inversion.ramp]) if inversion.n_ramp else np.asarray(inversion.g, float)
        self.d = np.asarray(inversion.d, float)
        self._weights = np.asarray(inversion.w, float)
        self._track = np.asarray(inversion.obs.ds["track"].values).astype(str)
        smooth, boundary = regularization_blocks(inversion, ss_ratio, ds_ratio, boundary_ratio, sides)
        self.rows = max(1, smooth.shape[0])
        self._SS = (smooth.T @ smooth).toarray()
        self._BB = (boundary.T @ boundary).toarray()
        target = prior_target(inversion, boundary)
        self._Bt = np.asarray(boundary.T @ target, float).ravel()
        lo, hi = slip_bounds(self.n, tuple(strike), tuple(dip), polarity)
        extra = self.A.shape[1] - 2 * self.n
        self.lo = np.concatenate([lo, np.full(extra, -np.inf)])
        self.hi = np.concatenate([hi, np.full(extra, np.inf)])
        self.P = self.A.shape[1]
        self._nuisance = self.A[:, 2 * self.n:] != 0
        self._cache = {}
        self.x_last = None
        self.loose_solves = []

    # -- the pieces ------------------------------------------------------------------------------------------------
    def normal(self, drop=None):
        """`(D, g, w)`: `(WA)'(WA)`, `(WA)'(Wd)` and the row weights, with `drop`'s rows removed."""
        key = None if drop is None or not np.any(drop) else np.flatnonzero(drop).tobytes()
        if key not in self._cache:
            w = self._weights if key is None else renormalised(self._weights, self._track, drop)
            aw = self.A * w[:, None]
            self._cache[key] = (aw.T @ aw, aw.T @ (w * self.d), w)
            if len(self._cache) > 4:
                self._cache.pop(next(k for k in self._cache if k is not None and k != key))
        return self._cache[key]

    def damping(self, lam):
        """The regularization block at weight `lam`, `(P, P)`."""
        return (float(lam) / self.rows) ** 2 * self._SS + self._BB

    def starved(self, keep):
        """The nuisance columns with no row in `keep`: singular, so every solve pins them to zero."""
        return 2 * self.n + np.flatnonzero(~self._nuisance[keep].any(axis=0))

    def unpredictable(self, rows, cols):
        """Of `rows`, those loading a column in `cols`: their constant was never fitted."""
        if not len(cols):
            return np.zeros(len(rows), bool)
        return self._nuisance[np.ix_(rows, np.asarray(cols) - 2 * self.n)].any(axis=1)

    @staticmethod
    def pinned(N, cols):
        if not len(cols):
            return N
        N = N.copy()
        N[cols, cols] += 1.0
        return N

    def _checked(self, N, rhs, x0, what):
        x, kkt = solve_box(N, rhs, self.lo, self.hi, x0)
        if kkt > KKT_ACCEPT:
            raise RuntimeError(f"{what}: bounded solve left a KKT residual of {kkt:.2e}")
        if kkt > KKT_MAX:
            self.loose_solves.append(kkt)
        return x

    # -- solving -------------------------------------------------------------------------------------------------
    def solve(self, lam, drop=None, x0=None):
        """The parameters minimising the system at weight `lam` (slip, then nuisance)."""
        D, g, _ = self.normal(drop)
        x0 = self.x_last if x0 is None else x0
        keep = np.ones(self.d.size, bool) if drop is None else ~np.asarray(drop, bool)
        x = self._checked(self.pinned(D + self.damping(lam), self.starved(keep)), g + self._Bt,
                          np.zeros(self.P) if x0 is None else x0, f"solve at lambda {lam:.5g}")
        self.x_last = x
        return x

    def residual(self, x):
        """`d - A x`, unweighted."""
        return self.d - self.A @ x

    def rank(self, lam, drop=None):
        """The effective rank at weight `lam`: the trace of the slip block of the resolution matrix."""
        D, _, _ = self.normal(drop)
        T = D + self.damping(lam)
        T.flat[:: self.P + 1] += RANK_JITTER * np.trace(T) / self.P
        return float(np.trace(cho_solve(cho_factor(T, lower=True), D)[:2 * self.n, :2 * self.n]))

    def lam_for_rank(self, rank, drop=None, lo=1e-8, hi=1e10, tol=1e-4):
        """The weight whose effective rank is `rank`, by bisection in log weight."""
        a, b = np.log(lo), np.log(hi)
        ra, rb = self.rank(np.exp(a), drop), self.rank(np.exp(b), drop)
        if not rb <= rank <= ra:
            raise ValueError(f"rank {rank:g} is outside {rb:.1f}-{ra:.1f} for weights {lo:g}-{hi:g}")
        while b - a > tol:
            m = 0.5 * (a + b)
            if self.rank(np.exp(m), drop) > rank:
                a = m
            else:
                b = m
        return float(np.exp(0.5 * (a + b)))


def slab_folds(obs, n_folds, direction, shift=0.0, span=None):
    """`(labels, width)`: equal-width slabs along `direction`, one label per row.

    `direction` is `(dx, dy)` in the frame, usually the main trace from its first vertex
    to its last. The slabs divide the extent of the rows in `span` (a boolean mask; all
    rows by default), so a sparse network far away does not stretch them; rows outside
    join the end slabs. `shift`, a fraction of a slab, moves every edge. Empty slabs are
    dropped and the rest numbered in order. `width` is in metres.
    """
    x, y = (np.asarray(obs.ds[k].values, float) for k in ("x", "y"))
    axis = np.asarray(direction, float) / np.linalg.norm(direction)
    t = x * axis[0] + y * axis[1]
    inside = np.ones(t.size, bool) if span is None else np.asarray(span, bool)
    edges = np.linspace(t[inside].min(), t[inside].max(), int(n_folds) + 1)
    width = float(edges[1] - edges[0])
    if shift:
        fold = np.clip(np.floor((t - edges[0]) / width - shift).astype(int), 0, int(n_folds) - 1)
    else:
        fold = np.clip(np.digitize(t, edges[1:-1]), 0, int(n_folds) - 1)
    return np.unique(fold, return_inverse=True)[1], width


def _basins(rank, score):
    """`[(rank, score / min)]` of each basin of the curve, the deepest first."""
    order = np.argsort(rank)
    r, s = np.asarray(rank, float)[order], np.asarray(score, float)[order]
    minima = [i for i in range(len(s))
              if (i == 0 or s[i] < s[i - 1]) and (i == len(s) - 1 or s[i] < s[i + 1])]
    merged = []
    for i in minima:
        if merged and s[merged[-1]:i + 1].max() < (1 + BASIN_RISE) * max(s[merged[-1]], s[i]):
            if s[i] < s[merged[-1]]:
                merged[-1] = i
            continue
        merged.append(i)
    best = s.min()
    return sorted(((float(r[i]), float(s[i] / best)) for i in merged), key=lambda b: b[1])


def _bands(rank, score, tol=1.01):
    """Each run of contiguous ranks scoring within `tol` of the minimum, `[lo, hi]`; the minimum's first.

    A run, not the span of every rank within `tol`: on a curve with two basins that span
    would take in the hump between them.
    """
    order = np.argsort(rank)
    r, s = np.asarray(rank, float)[order], np.asarray(score, float)[order]
    edges = np.flatnonzero(np.diff(np.r_[0, (s <= tol * s.min()).astype(int), 0]))
    runs = [[float(r[a]), float(r[b - 1])] for a, b in zip(edges[::2], edges[1::2])]
    best = float(r[np.argmin(s)])
    return sorted(runs, key=lambda ab: not ab[0] <= best <= ab[1])


def _vertex(lam, score, j):
    """The minimum of a parabola in log weight through up to five points around `j`."""
    lo, hi = max(j - 2, 0), min(j + 3, len(lam))
    u, v = np.log(lam[lo:hi]), score[lo:hi]
    c = np.polyfit(u, v, 2)
    if c[0] <= 0:
        return float(lam[j]), "grid point (parabola not convex)"
    star = float(np.exp(-c[1] / (2 * c[0])))
    return float(np.clip(star, lam[max(j - 1, 0)], lam[min(j + 1, len(lam) - 1)])), f"parabola over {hi - lo} points"


@dataclass
class RankChoice:
    """The outcome of `choose_rank`.

    Attributes:
        lam: The chosen smoothing weight.
        rank: Its effective rank.
        fit: How the minimum was located ("parabola over 5 points", or a grid point).
        basins: `[rank, score / minimum]` for each basin of the curve, the deepest first.
        bands: The runs of rank scoring within 1% of the minimum, the minimum's first.
        table: Every weight scored: `lam`, `rank`, `cv_rms`, `cv_prof`, `fit_rms`, and
            `refine` (a point added around the minimum) and `partition` (-1 marks the
            average over partitions).
        n_folds: The folds of the (first) partition.
        fold_sizes: The rows each fold holds out, in the first partition.
        partition_ranks: Each partition's own choice, when several were averaged.
        cv_rms_min_rank: The rank of the lowest held-out rms on the scored grid.
        cv_prof_min_rank: The same for the profiled score (nuisance refitted on the held-out rows).
    """

    lam: float
    rank: float
    fit: str
    basins: list
    bands: list
    table: object = field(repr=False)
    n_folds: int = 0
    fold_sizes: list = field(default_factory=list)
    partition_ranks: list = None
    cv_rms_min_rank: float = float("nan")
    cv_prof_min_rank: float = float("nan")

    def inside(self, rank):
        """Whether `rank` lies inside one of the 1% bands."""
        return any(lo <= rank <= hi for lo, hi in self.bands)

    def to_dict(self):
        """Everything but the table, as plain numbers for a JSON file."""
        return dict(lam=self.lam, rank=self.rank, fit=self.fit, basins=self.basins, bands=self.bands,
                    n_folds=self.n_folds, fold_sizes=self.fold_sizes, partition_ranks=self.partition_ranks,
                    cv_rms_min_rank=self.cv_rms_min_rank, cv_prof_min_rank=self.cv_prof_min_rank)


def _fold_parts(system, folds, drop):
    """Each fold's held-out rows, their share of the normal equations, and which rows can be scored."""
    keep = np.ones(system.d.size, bool) if drop is None else ~np.asarray(drop, bool)
    _, _, w = system.normal(drop)
    parts = []
    for label in np.unique(folds):
        test = np.flatnonzero((folds == label) & keep)
        at = system.A[test] * w[test, None]
        cols = system.starved(keep & (folds != label))
        parts.append((test, at.T @ at, at.T @ (w[test] * system.d[test]), cols,
                      test[~system.unpredictable(test, cols)]))
    return parts, keep


def _score_weight(system, parts, keep, scored, lam, drop, weighted):
    """One weight's row of the table: the full solve, then every fold from it."""
    D, g, w = system.normal(drop)
    damp = system.damping(lam)
    rhs = g + system._Bt
    x = system._checked(system.pinned(D + damp, system.starved(keep)), rhs,
                        system.x_last if system.x_last is not None else np.zeros(system.P), f"full solve at {lam:.5g}")
    system.x_last = x
    n2 = 2 * system.n
    ramp = system.A[:, n2:]
    resid = np.full(system.d.size, np.nan)
    prof = np.full(system.d.size, np.nan)
    for k, (test, Dt, gt, cols, _) in enumerate(parts):
        xf = system._checked(system.pinned(D - Dt + damp, cols), rhs - gt, x, f"fold {k} at {lam:.5g}")
        fault = system.A[test, :n2] @ xf[:n2]
        resid[test] = system.d[test] - fault - ramp[test] @ xf[n2:]
        if ramp.shape[1]:
            coef, *_ = np.linalg.lstsq(ramp[test], system.d[test] - fault, rcond=None)
            prof[test] = system.d[test] - fault - ramp[test] @ coef
        else:
            prof[test] = resid[test]
    u = w if weighted else 1.0
    return dict(lam=float(lam), rank=system.rank(lam, drop),
                cv_rms=float(np.sqrt(np.mean((u * resid)[scored] ** 2))),
                cv_prof=float(np.sqrt(np.mean((u * prof)[scored] ** 2))),
                fit_rms=float(np.sqrt(np.mean((u * system.residual(x))[keep] ** 2))))


def _curve(system, folds, lams, drop, weighted, score, log):
    """The table rows for one partition over the weights `lams`; a weight whose solve fails is skipped."""
    parts, keep = _fold_parts(system, folds, drop)
    scored = np.zeros(system.d.size, bool)
    for part in parts:
        scored[part[4]] = True
    eligible = keep if score is None else keep & np.asarray(score, bool)
    if score is not None:
        scored &= np.asarray(score, bool)
    if (eligible & ~scored).any():
        log(f"    {int((eligible & ~scored).sum())} held-out rows lie in a region with no row left in their fold's "
            "training set; they are not scored")
    rows = []
    for lam in lams:
        try:
            row = _score_weight(system, parts, keep, scored, lam, drop, weighted)
        except RuntimeError as err:
            log(f"    weight {lam:9.4g}: skipped ({err})")
            system.x_last = None
            continue
        rows.append(row)
        log(f"    weight {lam:9.4g} rank {row['rank']:6.1f}: cv_rms {row['cv_rms']:.4g} cv_prof {row['cv_prof']:.4g} "
            f"fit {row['fit_rms']:.4g}")
    return rows, [len(p[0]) for p in parts]


def choose_rank(system, folds, ranks=CV_RANKS, drop=None, weighted=True, score=None, refine=True, log=None):
    """The smoothing weight that predicts held-out data best, as a `RankChoice`.

    `folds` is one label per row (one partition, e.g. from `slab_folds`) or a list of
    such arrays (several partitions of the same rows). The weights run from the one
    giving rank `ranks[1]` (capped below the system's own maximum) to the one giving
    `ranks[0]`, a factor `CV_STEP` apart. Each weight is scored by the rms of the held-out
    residuals: weighted by the row weights (`weighted=True`, residual over sigma when the
    weights are one over sigma) or in the data's units. `score`, a boolean mask, limits
    the scored rows; every row still trains the folds it is not held out of. `drop`
    removes rows from the problem altogether.

    With one partition the minimum is refined with four more weights between its
    neighbours. With several, their coarse curves are averaged and the choice is made on
    the mean; its 1% bands are read on the mean interpolated in log rank.

    Raises `GridEdge` when the minimum lies at an end of the ladder.
    """
    import pandas as pd

    log = log or (lambda *a: None)
    partitions = [np.asarray(f).ravel() for f in (folds if isinstance(folds, (list, tuple)) else [folds])]
    top = min(float(ranks[1]), 0.9 * system.rank(1e-8, drop))
    lam_hi, lam_lo = system.lam_for_rank(ranks[0], drop), system.lam_for_rank(top, drop)
    n_grid = int(np.ceil(np.log(lam_hi / lam_lo) / np.log(CV_STEP))) + 1
    lams = np.geomspace(lam_lo, lam_hi, n_grid)

    if len(partitions) == 1:
        rows, sizes = _curve(system, partitions[0], lams, drop, weighted, score, log)
        grid = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
        j = int(np.argmin(grid.cv_rms))
        if j in (0, len(grid) - 1):
            raise GridEdge(grid["rank"][j])
        extra = []
        if refine:
            extra, _ = _curve(system, partitions[0], np.geomspace(grid.lam[j - 1], grid.lam[j + 1], 6)[1:-1],
                              drop, weighted, score, log)
        table = pd.DataFrame([dict(r, refine=False) for r in rows] + [dict(r, refine=True) for r in extra])
        table = table.sort_values("lam").reset_index(drop=True)
        table = table[np.r_[True, np.diff(np.log(table.lam.to_numpy())) > 1e-9]].reset_index(drop=True)
        table["partition"] = 0
        rel = (table.cv_rms / table.cv_rms.min()).to_numpy()
        lam_star, how = _vertex(table.lam.to_numpy(), rel, int(np.argmin(rel)))
        basins = _basins(grid["rank"].to_numpy(), (grid.cv_rms / grid.cv_rms.min()).to_numpy())
        bands = _bands(table["rank"].to_numpy(), table.cv_rms.to_numpy())
        per = None
    else:
        tables, per, sizes = [], [], None
        for k, f in enumerate(partitions):
            rows, sz = _curve(system, f, lams, drop, weighted, score, log)
            sizes = sizes or sz
            t = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
            jj = int(np.argmin(t.cv_rms))
            per.append(float(t["rank"][jj]))
            tables.append(t.assign(refine=False, partition=k))
        common = tables[0].lam.to_numpy()
        for t in tables[1:]:
            common = common[[np.isclose(t.lam.to_numpy(), v, rtol=1e-9).any() for v in common]]
        aligned = [t[[np.isclose(common, v, rtol=1e-9).any() for v in t.lam.to_numpy()]].reset_index(drop=True)
                   for t in tables]
        mean = pd.DataFrame(dict(lam=common, rank=aligned[0]["rank"].to_numpy(),
                                 cv_rms=np.mean([t.cv_rms.to_numpy() for t in aligned], axis=0),
                                 cv_prof=np.mean([t.cv_prof.to_numpy() for t in aligned], axis=0),
                                 fit_rms=aligned[0].fit_rms.to_numpy(), refine=False, partition=-1))
        rel = (mean.cv_rms / mean.cv_rms.min()).to_numpy()
        j = int(np.argmin(rel))
        if j in (0, len(rel) - 1):
            raise GridEdge(mean["rank"][j])
        lam_star, how = _vertex(common, rel, j)
        r_sorted = np.sort(mean["rank"].to_numpy())
        s_sorted = mean.cv_rms.to_numpy()[np.argsort(mean["rank"].to_numpy())]
        fine = np.geomspace(r_sorted[0], r_sorted[-1], 800)
        bands = _bands(fine, np.interp(np.log(fine), np.log(r_sorted), s_sorted))
        basins = _basins(mean["rank"].to_numpy(), rel)
        table = pd.concat(tables + [mean], ignore_index=True)
        grid = mean
    rank = system.rank(lam_star, drop)
    log(f"    minimum: {how}, weight {lam_star:.5g}, rank {rank:.2f}; basins "
        + ", ".join(f"{r:.1f} {s:.3f}" for r, s in basins) + "; 1% band" + ("s " if len(bands) > 1 else " ")
        + ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in bands))
    return RankChoice(lam=lam_star, rank=rank, fit=how, basins=[[r, s] for r, s in basins], bands=bands, table=table,
                      n_folds=int(np.unique(partitions[0]).size), fold_sizes=list(sizes or []), partition_ranks=per,
                      cv_rms_min_rank=float(grid["rank"][int(np.argmin(grid.cv_rms))]),
                      cv_prof_min_rank=float(grid["rank"][int(np.argmin(grid.cv_prof))]))


def heldout_residuals(system, folds, lam, drop=None):
    """Each row's held-out residual at weight `lam`, times its weight (NaN where no fold can score it).

    The rms over the finite rows is `choose_rank`'s weighted score at `lam`. Summed by
    region it shows where a cross-validation gain comes from.
    """
    folds = np.asarray(folds).ravel()
    parts, keep = _fold_parts(system, folds, drop)
    D, g, w = system.normal(drop)
    damp = system.damping(lam)
    rhs = g + system._Bt
    x = system._checked(system.pinned(D + damp, system.starved(keep)), rhs,
                        system.x_last if system.x_last is not None else np.zeros(system.P), "full solve")
    n2 = 2 * system.n
    out = np.full(system.d.size, np.nan)
    for test, Dt, gt, cols, ok in parts:
        xf = system._checked(system.pinned(D - Dt + damp, cols), rhs - gt, x, "fold solve")
        r = system.d[ok] - system.A[ok, :n2] @ xf[:n2] - system.A[ok, n2:] @ xf[n2:]
        out[ok] = w[ok] * r
    return out
