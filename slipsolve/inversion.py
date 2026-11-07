"""The inversion itself: a bounded, regularized linear least-squares solve.

The system is the reference implementation's, stacked:

    [ G * w        ]        [ d * w ]
    [ lam * L / nL ] * m =  [   0   ]
    [ W_boundary   ]        [   0   ]

with `lb <= m <= ub`. `G` is the Green's matrix, `w` the per-observation
weights, `L` the smoothing operator normalised by its own row count (so
`lam` means the same thing as the mesh is refined), and `W_boundary` the
zero-slip edge rows.

It is *not* assembled as one dense array. `G` is dense and `L`/`W` are
sparse, so the stack is wrapped in a `scipy.sparse.linalg.LinearOperator`
that applies each in its natural form: the smoothing block then costs a sparse
matrix-vector product rather than a dense one, and a dense copy of the whole
stack never exists. `scipy.optimize.lsq_linear` accepts a `LinearOperator` with
`lsq_solver="lsmr"`, which is the combination used here by default.

`solve(method="active_set")` solves the same bounded problem in its normal
equations instead, exactly, by primal-dual active set (`box_qp`); see
`METHODS` for when that is faster and when it is refused. A refused answer
falls back to `"lsq_linear_dense"` (the stacked system as a dense matrix, solved to
within a micron of the minimiser) on problems up to `EXACT_FALLBACK_MAX_PARAMS`
parameters, and to `"lsq_linear"` above.

Solver honesty matters more than usual in this problem. Runtime depends far more
on conditioning than on size, and an iteration-capped solve returns a
plausible-looking model with a meaningless variance reduction -- so
`SlipModel` carries the solver's status and refuses to pretend.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, lapack, solve_triangular
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

from .basis import make_basis
from .greens import HalfSpaceTDE
from .regularize import (
    laplace_beltrami,
    neighbor_smoothing,
    check_ramp_tracks,
    ramp_columns,
    ramp_kind_for,
    ramp_spec,
    scale_smoothing,
    slip_bounds,
    zero_slip_boundary,  # noqa: F401  (public API, and used by callers)
)


def _blocked(projection):
    """A basis projection applied to both the strike and dip halves."""
    zero = sp.csr_matrix(projection.shape)
    return sp.bmat([[projection, zero], [zero, projection]], format="csr")


def _far_enough(x, y, trace_nodes, distance, block=20000):
    """True where a point is further than `distance` from any trace node.

    Blocked over the points, since the full pairwise matrix against a long fault's
    node row is large and only its row minimum is wanted.
    """
    keep = np.empty(x.size, dtype=bool)
    for start in range(0, x.size, block):
        sl = slice(start, start + block)
        d2 = ((x[sl, None] - trace_nodes[None, :, 0]) ** 2
              + (y[sl, None] - trace_nodes[None, :, 1]) ** 2)
        keep[sl] = d2.min(axis=1) > distance ** 2
    return keep


class _MissingEngine:
    """Placeholder for an engine a loaded model cannot rebuild.

    A layered engine is defined by its tables, which are megabytes and are
    deliberately not saved with a model -- as the Green's matrix is not. Silently
    substituting the homogeneous half-space would let `forward` return plausible
    numbers computed with the wrong physics, so it raises instead.
    """

    def __init__(self, name):
        self.name = name
        self.nu = 0.25

    def _refuse(self, *_, **__):
        raise RuntimeError(
            f"This model was solved with the {self.name!r} engine, whose Green's "
            "functions come from EDGRN tables that are not stored in a saved "
            "model. Rebuild the engine (LayeredPointSource with the same tables) "
            "and construct a fresh SlipInversion to forward-model with it."
        )

    forward = displacement = los_matrix = _refuse


def _per_parameter(basis, mesh, factor):
    """A per-element factor carried onto the parameters: as is, or tent-averaged by area."""
    if basis.name == "element":
        return factor
    areas = mesh.areas
    return (basis.projection().T @ (areas * factor)) / basis.lumped_areas()


def _boundary_rows(basis, sides, ratio):
    """Selection rows pulling the flagged boundary parameters toward zero."""
    idx = np.nonzero(basis.boundary(sides))[0]
    n = basis.n_basis
    if idx.size == 0:
        return sp.csr_matrix((0, 2 * n))
    cols = np.concatenate([idx, idx + n])
    return sp.csr_matrix(
        (np.full(cols.size, float(ratio)), (np.arange(cols.size), cols)),
        shape=(cols.size, 2 * n),
    )


def _projected_newton(N, g, lo, hi, x, max_iter):
    """`(at_lo, at_hi, steps)`: the active set of the box QP, by Newton steps projected onto the box (Bertsekas 1982).

    Slower per set than primal-dual active set and duller near an easy answer, but the
    objective falls at every step, so it cannot cycle, and a step can move many variables onto
    their bounds at once. From `x`, clipped into the box. The caller solves on the final set.
    """
    def objective(v):
        return 0.5 * v @ (N @ v) - g @ v

    x = np.clip(x, lo, hi)
    width = np.minimum(hi - lo, 1.0)
    width[~np.isfinite(width)] = 1.0
    near = _SNAP_TO_BOUND * width
    value = objective(x)
    exact_sets = None
    for step in range(1, int(max_iter) + 1):
        mu = N @ x - g
        # A variable a hair off its bound, and pushed onto it, goes there. Otherwise it would take
        # a step of its own and cut every other variable's short.
        snap_lo, snap_hi = (x <= lo + near) & (mu > 0.0), (x >= hi - near) & (mu < 0.0)
        if snap_lo.any() or snap_hi.any():
            x = np.where(snap_lo, lo, np.where(snap_hi, hi, x))
            mu = N @ x - g
            value = objective(x)
        at_lo = (x <= lo) & (mu > 0.0)
        at_hi = (x >= hi) & (mu < 0.0)
        key = (at_lo.tobytes(), at_hi.tobytes())
        if key == exact_sets:
            return at_lo, at_hi, step
        free = ~(at_lo | at_hi)
        direction = np.zeros(g.size)
        if free.any():
            direction[free] = cho_solve(cho_factor(N[np.ix_(free, free)], lower=True), -mu[free])
        alpha = 1.0
        while True:
            trial = np.clip(x + alpha * direction, lo, hi)
            new = objective(trial)
            if new <= value + _ARMIJO * float(mu @ (trial - x)) or alpha < _SMALLEST_STEP:
                break
            alpha *= 0.5
        unclipped = float(np.abs(trial - (x + direction)).max(initial=0.0)) <= _SNAP_TO_BOUND
        exact_sets = key if alpha == 1.0 and unclipped else None
        x, value = trial, new
    raise RuntimeError(f"the active set did not settle in {int(max_iter)} projected Newton steps")


def box_qp(N, g, lo, hi, x0=None, max_iter=100):
    """Minimise `x'Nx/2 - g'x` under `lo <= x <= hi`, exactly, by primal-dual active set.

    The problem `SlipInversion.solve` hands `lsq_linear`, written in its
    normal equations. Each step holds the variables the current guess puts on a
    bound there and solves for the rest by Cholesky. It then moves onto its bound
    every free variable that crossed one, and off its bound every held variable
    whose gradient points inside. When a step changes nothing, every KKT condition
    holds, so the answer is the minimiser and not an approximation to it.

    `x0`, a nearby solution as in a sweep, picks the first guess (a variable
    within 0.1 mm of a bound starts on it), and the set then settles in a few
    steps. Without it every variable starts free.

    Primal-dual active set can cycle when `N` is not an M-matrix, and on a kernel with
    most variables on a bound it does. If the set comes back to a guess already tried, or has not
    settled in `max_iter` steps, the set is found by `_projected_newton` instead, and
    `escaped` says so. The answer depends only on the final set, so the fallback
    changes nothing for a solve that settles on its own.

    Returns a namespace: `x`; `nit`, the steps taken; `at_lo`/`at_hi`, the
    final active set; `kkt`, the largest gradient left on a free variable
    relative to `max|g|`; `rcond`, LAPACK's 1-norm estimate of the
    reciprocal condition number of the free block (1 when nothing is free); and `escaped`.

    Raises `RuntimeError` if neither settles, and `numpy.linalg.LinAlgError` if
    a free block is not positive definite.
    """
    g = np.asarray(g, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    if x0 is None:
        at_lo = np.zeros(g.size, dtype=bool)
        at_hi = np.zeros(g.size, dtype=bool)
    else:
        at_lo = has_lo & (x0 <= lo + 1e-4)
        at_hi = has_hi & (x0 >= hi - 1e-4)

    def solved(at_lo, at_hi):
        """The solution on an active set, and what is needed to read it."""
        held = at_lo | at_hi
        free = ~held
        x = np.where(at_lo, lo, np.where(at_hi, hi, 0.0))
        block = factor = None
        if free.any():
            rhs = g[free] - N[np.ix_(free, held)] @ x[held]
            block = N[np.ix_(free, free)]
            factor = cho_factor(block, lower=True)
            x[free] = cho_solve(factor, rhs)
        return x, free, block, factor, N @ x - g

    tried = set()
    escaped = False
    limit = int(max_iter)
    for step in range(1, limit + 1):
        x, free, block, factor, mu = solved(at_lo, at_hi)
        new_lo = has_lo & ((free & (x < lo)) | (at_lo & (mu > 0)))
        new_hi = has_hi & ((free & (x > hi)) | (at_hi & (mu < 0)))
        if np.array_equal(new_lo, at_lo) and np.array_equal(new_hi, at_hi):
            break
        tried.add(np.packbits(at_lo).tobytes() + np.packbits(at_hi).tobytes())
        if np.packbits(new_lo).tobytes() + np.packbits(new_hi).tobytes() in tried:
            escaped = True
            break
        at_lo, at_hi = new_lo, new_hi
    else:
        escaped = True
    if escaped:
        start = np.clip(x, lo, hi)
        if x0 is not None:
            warm = np.clip(np.asarray(x0, dtype=float), lo, hi)
            if 0.5 * warm @ (N @ warm) - g @ warm < 0.5 * start @ (N @ start) - g @ start:
                start = warm
        at_lo, at_hi, more = _projected_newton(N, g, lo, hi, start, limit)
        step += more
        x, free, block, factor, mu = solved(at_lo, at_hi)

    rcond = 1.0
    if factor is not None:
        rcond, info = lapack.dpocon(factor[0], float(np.abs(block).sum(axis=0).max()), uplo="L")
        rcond = float(rcond) if info == 0 else 0.0
    scale = max(float(np.abs(g).max(initial=0.0)), np.finfo(float).tiny)
    kkt = float(np.abs(mu[free]).max(initial=0.0)) / scale
    return SimpleNamespace(x=x, nit=step, at_lo=at_lo, at_hi=at_hi, kkt=kkt, rcond=rcond, escaped=escaped)


#: `_projected_newton`: a variable this close to a bound (as a share of its range, at most 1) and
#: pushed onto it goes there; the sufficient decrease; and the shortest step along the projection arc.
_SNAP_TO_BOUND = 1e-12
_ARMIJO = 1e-4
_SMALLEST_STEP = 1e-10

#: Crustal shear modulus used for moment when no velocity model is supplied.
DEFAULT_SHEAR_MODULUS = 30e9

#: Sub-triangles per element edge in the integrated moment, `order**2` per element.
#:
#: The rigidity is averaged exactly over each piece; what the order resolves is the
#: slip magnitude, and its covariance with the rigidity inside a piece that
#: straddles a step. At 8 the relative error in moment is of order 1e-5 on
#: kilometre-scale elements and under 1e-3 on coarse ones with a rigidity step
#: inside a row. Do not sample the rigidity at points instead: that is first
#: order and erratic, set by where the points fall against each interface.
MOMENT_ORDER = 8

#: Default cap on trust-region iterations.
#:
#: Generous on purpose. Iteration count tracks *conditioning*, not problem size,
#: and not monotonically in the smoothing weight either: one weight in the middle
#: of a sweep can need several times the iterations of the weights either side.
#: A tight cap therefore silently truncates isolated points of an L-curve sweep
#: rather than the obvious rough end. Since a well-conditioned solve stops long
#: before the cap, raising it costs nothing where it is not needed.
DEFAULT_MAX_ITER = 400

#: Inner-solver tolerance handed to `lsq_linear`'s `lsmr` sub-problem.
#:
#: `"auto"` lets scipy tighten the inner tolerance as the outer trust-region
#: iteration converges, instead of solving every sub-problem to `atol=btol=0`
#: -- which is what scipy's `None` default does, and which makes `lsmr` run
#: to its own iteration cap on every outer step. Most of a solve is spent in
#: those matrix-vector products, so on a single solve this is several times
#: faster for the same answer.
#:
#: The win is not uniform in the smoothing weight: the adaptive tolerance loses
#: at both ends of a sweep and wins by about 2x in the middle, a modest net gain
#: over the whole `SlipInversion.l_curve` sweep, which is why it stays the
#: default. Judge it over a sweep, never at one weight. Outer iteration count is
#: the wrong meter: `"auto"` solves each sub-problem loosely, so it routinely
#: takes *more* outer steps while doing less total work.
#:
#: Do not reach for `lsmr_maxiter` instead. A hard cap is not monotone: a
#: moderate cap can drive the outer loop into `DEFAULT_MAX_ITER` and return a
#: materially worse model while a smaller one is fine. The adaptive tolerance
#: has no such cliff.
DEFAULT_LSMR_TOL = "auto"

#: The solvers `SlipInversion.solve` offers. All minimise the same
#: objective under the same bounds.
#:
#: * `"lsq_linear"`: scipy's trust-region solve of the stacked system, through
#:   `lsmr` on the operator. The default.
#: * `"active_set"`: the normal equations `(A'A + S'S + B'B) m = A'b`,
#:   solved exactly by `box_qp`. `A'A` and `A'b` do not depend on the
#:   regularization, so they are formed once per inversion and kept; a sweep
#:   over smoothing weights then pays one small factorization per weight
#:   instead of a trust-region solve over every observation row.
#:   `slipsolve.sweep.DipSweep` uses the same solve.
#: * `"lsq_linear_dense"`: the stacked system as a dense matrix, `lsq_linear`
#:   with exact inner solves (`lsq_solver="exact"`) to `EXACT_FALLBACK_TOL`,
#:   which ends within a micron of the minimiser (a trust region nears a bound
#:   asymptotically) where `lsmr`'s inexact steps stop short. Its cost grows as
#:   rows x parameters**2 per iteration, so it is for small problems.
#:
#: Normal equations square the condition number, so an active-set answer is
#: refused, with a warning, if its free block is worse conditioned than
#: `ACTIVE_SET_MAX_COND`, if its KKT residual exceeds
#: `ACTIVE_SET_MAX_KKT`, if a free block is not positive definite, or if
#: `box_qp` does not settle even by projected Newton steps, after its active set
#: cycles or runs out. The fallback is `"lsq_linear_dense"` on
#: problems up to `EXACT_FALLBACK_MAX_PARAMS` parameters and
#: `"lsq_linear"` above.
METHODS = ("lsq_linear", "active_set", "lsq_linear_dense")

#: Largest 1-norm condition number an active-set solve's free block may have.
#: The answer's relative error is bounded by about this times machine epsilon.
ACTIVE_SET_MAX_COND = 1e12

#: Largest gradient an active-set answer may leave on a free parameter,
#: relative to the largest gradient at zero slip.
ACTIVE_SET_MAX_KKT = 1e-8

#: Most parameters for which a refused active-set answer falls back to
#: `"lsq_linear_dense"` rather than `"lsq_linear"`. Primal-dual active set has no
#: convergence guarantee when the normal matrix is not an M-matrix, and weak
#: smoothing makes it wander. On such a problem `lsq_linear` can stop well above
#: the minimum, at a place that differs between machines, where the dense solve
#: reaches it. The dense solve grows as rows x parameters**2 an iteration, so
#: larger problems keep `lsq_linear`, which at moderate smoothing stops only
#: negligibly short.
EXACT_FALLBACK_MAX_PARAMS = 1000

#: `"lsq_linear_dense"`'s tolerance on the relative change in cost. Looser
#: values stop measurably above the minimum.
EXACT_FALLBACK_TOL = 1e-13


class SlipInversion:
    """A mesh, a set of observations, and the Green's matrix relating them.

    The Green's matrix is built once, on construction, and reused across every
    smoothing weight -- which is what makes an L-curve sweep cheap.
    """

    def __init__(self, mesh, obs, engine=None, ramp="none", basis="element",
                 velocity_model=None, covariance=None, sees=None):
        """`engine` and `basis` choose the physics and the parameterization.

        `ramp` adds unbounded per-track nuisance columns -- `"none"`,
        `"offset"`, `"linear"`, `"quadratic"` or `"cubic"`, or a
        `{track: kind}` dict with `"*"` for the rest, e.g. a cubic for the
        interferograms and an offset for pixel-offset tracks (see
        `slipsolve.regularize.ramp_columns`). Solving the ramp here
        is preferable to pre-removing it from the interferogram: a pre-removal
        fits the polynomial to the data alone and has to extrapolate under a
        masked signal region, while these columns are constrained jointly with
        the slip. Run `slipsolve.diagnostics.ramp_content` first to see what
        each degree would absorb.

        `velocity_model` is **not** used to compute Green's functions -- for a
        layered medium that information is already inside the engine's tables,
        and for a homogeneous half-space the displacements do not depend
        on rigidity at all (it cancels; only Poisson's ratio survives). What it is
        for is **moment**: `sum(mu * area * slip)` needs a rigidity, and without
        one every reported `Mw` silently assumes
        `DEFAULT_SHEAR_MODULUS` = 30 GPa. Pass the same
        `slipsolve.edgrn.VelocityModel` you built the tables from
        and every statistic -- `moment`, `moment_magnitude`, `to_text`,
        `repr` -- uses the rigidity of the layered crust instead.

        It matters more than it sounds: a crust whose shallowest layer is 9 GPa
        has a third of the default's rigidity there, which is exactly where a
        strike-slip fault's shallow slip sits.

        `covariance` replaces the per-observation weights with a full spatially
        correlated noise model: one
        `slipsolve.covariance.CovarianceModel` for every track, or
        a `{track: model}` mapping. The system is then solved whitened,
        `L^-1 G m = L^-1 d` with `Cd = L L'`, which is what
        `whiten` applies. `None` keeps the diagonal weights.

        `sees` restricts which faults of a mesh built by
        `slipsolve.mesh.FaultMesh.join` a track measures: `{track: [fault, ...]}`,
        faults by their position in the join. A track not named sees every fault.
        Made for an earthquake sequence whose radar pairs span every event while
        GNSS resolves each one: the GNSS for event 1 sees only the faults event 1
        broke, so its rows get zero Green's functions on the others, and the radar
        rows see the sum. That is exact when no fault slipped in more than one
        event; slip a fault took between two events (afterslip in the hours
        between) is assigned to whichever track sees it.
        """
        self.mesh = mesh
        self.obs = obs
        self.engine = engine or HalfSpaceTDE()
        self.basis = make_basis(mesh, basis)
        self.velocity_model = velocity_model

        if mesh.frame is not None and obs.ds.attrs.get("frame") is not None:
            mesh.frame.require_match(obs.ds.attrs["frame"], "Observations")

        ds = obs.ds
        self.d = np.asarray(ds["los"].values, dtype=float)
        self.w = np.asarray(ds["weight"].values, dtype=float)
        # An engine that can integrate a basis function directly is given the
        # basis, so a nodal column is the integral of the tent itself. One that
        # cannot -- the closed-form triangular dislocation, which is defined on a
        # patch of *constant* slip -- assembles per element and has the basis
        # projected on afterwards. Both preserve the moment exactly; only the
        # first gets the near-field shape of a linearly varying slip patch right.
        points = (ds["x"].values, ds["y"].values,
                  ds["look_e"].values, ds["look_n"].values, ds["look_u"].values)
        self.exact_basis = bool(getattr(self.engine, "supports_basis", False))
        if self.exact_basis:
            self.g = self.engine.los_matrix(mesh, *points, basis=self.basis)
        else:
            element_g = self.engine.los_matrix(mesh, *points)
            self.g = (element_g if self.basis.name == "element"
                      else element_g @ _blocked(self.basis.projection()))

        self.sees = None
        if sees:
            self._restrict_views(sees)

        self.ramp_kind = ramp_spec(ramp)
        check_ramp_tracks(self.ramp_kind, obs.tracks)
        self.ramp, self.ramp_labels = ramp_columns(obs, self.ramp_kind)
        self.n_slip = self.basis.n_param
        self.n_ramp = self.ramp.shape[1]

        self.covariance = covariance
        self._chol = self._track_rows = None
        if covariance is not None:
            self._build_whitening(covariance)

    def fault_labels(self):
        """The joined fault each basis function belongs to, `(n_basis,)`.

        Read from the counts `slipsolve.mesh.FaultMesh.join` records, so it
        follows the join's order. A mesh that was not joined is one fault, label 0.
        """
        key = "joined_nodes" if self.basis.name == "node" else "joined_elements"
        counts = self.mesh.attrs.get(key)
        if counts is None:
            return np.zeros(self.basis.n_basis, dtype=int)
        labels = np.repeat(np.arange(len(counts)), np.asarray(counts, dtype=int))
        if labels.size != self.basis.n_basis:
            raise ValueError(
                f"mesh.attrs[{key!r}] sums to {labels.size} but the {self.basis.name} basis "
                f"has {self.basis.n_basis} functions; was the mesh rebuilt after the join?")
        return labels

    def _restrict_views(self, sees):
        """Zero the Green's functions of the faults each named track does not see."""
        labels = self.fault_labels()
        n_faults = int(labels.max()) + 1
        track = np.asarray(self.obs.ds["track"].values, dtype=object).astype(str)
        unknown = sorted(set(map(str, sees)) - set(track))
        if unknown:
            raise ValueError(f"sees names track(s) {unknown} not in the observations")
        recorded = {}
        for name, faults in sees.items():
            faults = sorted({int(f) for f in faults})
            if not faults or min(faults) < 0 or max(faults) >= n_faults:
                raise ValueError(f"sees[{name!r}] = {faults}: faults are 0..{n_faults - 1}")
            hidden = ~np.isin(labels, faults)
            cols = np.concatenate([np.flatnonzero(hidden),
                                   np.flatnonzero(hidden) + self.basis.n_basis])
            rows = np.flatnonzero(track == str(name))
            self.g[np.ix_(rows, cols)] = 0.0
            recorded[str(name)] = faults
        self.sees = recorded

    def _build_whitening(self, covariance):
        """Cholesky factors of `Cd`, one block per track."""
        from .covariance import cell_ratio_warning, whitening_factors

        normalize = self.obs.ds.attrs.get("normalize")
        if normalize not in (None, "none"):
            raise ValueError(
                f"Observations were concatenated with normalize={normalize!r}, "
                "which scales each track by its own sample count -- the same job "
                "a covariance model does, and does properly. Passing both states "
                "the relative track weighting twice. Re-run "
                'Observations.concat(..., normalize="none") and let the '
                "covariance set the weights."
            )

        track = np.asarray(self.obs.ds["track"].values, dtype=object)
        self._chol = whitening_factors(self.obs, covariance)
        self._track_rows = {name: np.flatnonzero(track == name)
                            for name in self._chol}
        for name in self._chol:
            model = (covariance if not isinstance(covariance, dict)
                     else covariance[name])
            message = cell_ratio_warning(self.obs, model, track=name)
            if message is not None:
                warnings.warn(f"track {name!r}: {message}", RuntimeWarning,
                              stacklevel=3)

    def whiten(self, values):
        """Left-multiply by `Cd^-1/2` -- or by the diagonal weights if no `Cd`.

        Every place the design matrix or the data meets the solver goes through
        here, so that the resolution matrix, the data importance and the ramp
        diagnostics all describe the system that was actually solved. Missing one
        of them is caught by `trace(R) == trace(N)`, which holds only when both
        halves whiten the same `A`.

        Takes a vector or a matrix; whitening is applied down the observation
        axis in both cases.
        """
        values = np.asarray(values, dtype=float)
        if self._chol is None:
            return values * (self.w[:, None] if values.ndim == 2 else self.w)

        out = np.empty_like(values)
        for name, factor in self._chol.items():
            rows = self._track_rows[name]
            out[rows] = solve_triangular(factor, values[rows], lower=True)
        return out

    @property
    def n_param(self):
        return self.n_slip + self.n_ramp

    def with_ramp(self, kind):
        """The same inversion with a different nuisance ramp, **sharing** `G`.

        Only the ramp block depends on `kind`, and it costs milliseconds while
        the Green's matrix costs tens of seconds -- so comparing two ramp degrees
        is affordable, which is what makes
        `slipsolve.diagnostics.ramp_sweep` a routine check rather
        than an hour of re-assembly. The copy shares `g` by reference; nothing
        writes to it.
        """
        import copy

        other = copy.copy(self)
        other.ramp_kind = ramp_spec(kind)
        check_ramp_tracks(other.ramp_kind, self.obs.tracks)
        other.ramp, other.ramp_labels = ramp_columns(self.obs, other.ramp_kind)
        other.n_ramp = other.ramp.shape[1]
        return other

    # -- solving -----------------------------------------------------------
    def solve(self, smoothing=0.3, ss_ratio=1.0, ds_ratio=3.0,
              boundary_ratio=1.0, sides=("bottom", "left", "right"),
              polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0),
              max_iter=DEFAULT_MAX_ITER, tol=1e-10,
              lsmr_tol=DEFAULT_LSMR_TOL, smooth_weighting="uniform", verbose=0,
              method="lsq_linear"):
        """Solve for slip at one smoothing weight; returns a `SlipModel`.

        `method` is `"lsq_linear"` (the default), `"active_set"` or
        `"lsq_linear_dense"`; see `METHODS`. All solve the same bounded
        problem. `max_iter` caps the trust-region iterations of the first and the
        active-set steps of the second; the dense solve takes at least
        `50 x parameters`. `tol` and `lsmr_tol` are `lsq_linear`'s alone. The
        model's `options["method"]` records the solver that produced it: when an
        active-set answer was refused, `"lsq_linear_dense"` up to
        `EXACT_FALLBACK_MAX_PARAMS` parameters and `"lsq_linear"` above.

        `lsmr_tol` is the inner least-squares tolerance; see
        `DEFAULT_LSMR_TOL` for why the default is `"auto"` and not
        scipy's `None`.

        `smooth_weighting` applies only to the *element* basis: `"length"`
        turns the neighbour smoother from a difference into a gradient, which is
        what a mesh with widely varying element size needs. The node basis's
        Laplace-Beltrami operator is already area- and shape-weighted and
        ignores it.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if self.basis.name == "element":
            smooth = neighbor_smoothing(self.mesh.neighbors, ss_ratio, ds_ratio,
                                        weighting=smooth_weighting, mesh=self.mesh)
        else:
            # A continuous field lives on the nodes, so neighbouring *elements* is
            # the wrong thing to difference; the surface Laplacian is the right one.
            smooth = laplace_beltrami(self.mesh, ss_ratio, ds_ratio)
        boundary = _boundary_rows(self.basis, sides, boundary_ratio)
        lo, hi = slip_bounds(self.basis.n_basis, strike, dip, polarity)

        if self.n_ramp:
            # Nuisance columns are unbounded and unregularized: their whole job is
            # to soak up whatever the slip model should not be explaining.
            pad = sp.csr_matrix((smooth.shape[0], self.n_ramp))
            smooth = sp.hstack([smooth, pad], format="csr")
            boundary = sp.hstack(
                [boundary, sp.csr_matrix((boundary.shape[0], self.n_ramp))], format="csr"
            )
            lo = np.concatenate([lo, np.full(self.n_ramp, -np.inf)])
            hi = np.concatenate([hi, np.full(self.n_ramp, np.inf)])

        scaled_smooth = scale_smoothing(smooth, smoothing)

        result = None
        if method == "active_set":
            result, problem = self._solve_active_set(scaled_smooth, boundary, lo, hi,
                                                     max_iter, verbose)
            if result is None:
                method = ("lsq_linear_dense" if self.n_param <= EXACT_FALLBACK_MAX_PARAMS
                          else "lsq_linear")
                warnings.warn(f"the active-set solve was refused ({problem}); solved "
                              f"with {method} instead", RuntimeWarning, stacklevel=2)
        if method == "lsq_linear_dense":
            result = self._solve_lsq_dense(scaled_smooth, boundary, lo, hi, max_iter, verbose)
        elif result is None:
            method = "lsq_linear"
            result = self._solve_lsq(scaled_smooth, boundary, lo, hi, max_iter,
                                     tol, lsmr_tol, verbose)

        return SlipModel(
            self, result.x, result,
            smoothing=(float(smoothing) if np.ndim(smoothing) == 0
                       else np.asarray(smoothing, dtype=float)),
            smooth_matrix=smooth,
            options={
                "ss_ratio": float(ss_ratio), "ds_ratio": float(ds_ratio),
                "boundary_ratio": float(boundary_ratio), "sides": list(sides),
                "polarity": None if polarity is None else list(polarity),
                "strike": list(strike), "dip": list(dip), "max_iter": int(max_iter),
                "lsmr_tol": lsmr_tol, "method": method,
                "smooth_weighting": str(smooth_weighting),
                # Carried so a reloaded model can rebuild the ramp columns and
                # separate the nuisance surface from the fault's own prediction.
                "ramp": getattr(self, "ramp_kind", "none"),
                # Carried so a saved model keeps its own rigidity; without it a
                # reloaded layered model would report a 30 GPa moment.
                "velocity_model": (None if self.velocity_model is None
                                   else self.velocity_model.to_dict()),
                # Carried so a saved model keeps its rigidity contrast: the moment
                # uses it, and a reloaded half-space model forwards through it.
                "bimaterial": (self.engine.describe() if hasattr(self.engine, "describe")
                               else None),
                # Which faults each track was allowed to see; None = all of them.
                "sees": getattr(self, "sees", None),
            },
        )

    def _solve_lsq(self, smooth, boundary, lo, hi, max_iter, tol, lsmr_tol, verbose):
        """`method="lsq_linear"`: the stacked system through scipy's trust region."""
        design = np.hstack([self.g, self.ramp]) if self.n_ramp else self.g
        weighted = self.whiten(design)
        rhs = np.concatenate([
            self.whiten(self.d),
            np.zeros(smooth.shape[0]),
            np.zeros(boundary.shape[0]),
        ])

        operator = _StackedOperator(weighted, smooth, boundary)
        return lsq_linear(
            operator, rhs, bounds=(lo, hi),
            lsq_solver="lsmr", max_iter=max_iter, tol=tol,
            lsmr_tol=lsmr_tol, verbose=verbose,
        )

    def _solve_lsq_dense(self, smooth, boundary, lo, hi, max_iter, verbose):
        """`method="lsq_linear_dense"`: the stacked system as a dense matrix, solved to within a micron of the minimiser.

        `lsq_linear`'s trust region with exact inner solves (`lsq_solver="exact"`),
        at `EXACT_FALLBACK_TOL`; it works on the stacked system, whose condition
        number is the square root of the normal equations'.
        """
        design = np.hstack([self.g, self.ramp]) if self.n_ramp else self.g
        stacked = np.vstack([self.whiten(design), smooth.toarray(), boundary.toarray()])
        rhs = np.concatenate([
            self.whiten(self.d),
            np.zeros(smooth.shape[0]),
            np.zeros(boundary.shape[0]),
        ])
        return lsq_linear(stacked, rhs, bounds=(lo, hi), method="trf", lsq_solver="exact",
                          tol=EXACT_FALLBACK_TOL, max_iter=max(int(max_iter), 50 * stacked.shape[1]),
                          verbose=verbose)

    def _data_block(self):
        """`(WA)'(WA)` and `(WA)'(Wd)`, the half of the normal equations the data set.

        Neither depends on the regularization, so they are kept between solves
        and a sweep forms them once. The cache holds while `G`, the ramp and
        the whitening are the same objects (nothing writes to `G` in place; see
        `with_ramp`) and the data and weights the same *values*: `d` can
        be the observation dataset's own buffer, so an in-place edit must not
        find a stale block. The last answer rides along as the next warm start.
        """
        chol = getattr(self, "_chol", None)
        cached = getattr(self, "_normal_cache", None)
        if (cached is not None and cached.g is self.g and cached.ramp is self.ramp
                and cached.chol is chol and np.array_equal(cached.d, self.d)
                and np.array_equal(cached.w, self.w)):
            return cached
        design = np.hstack([self.g, self.ramp]) if self.n_ramp else self.g
        weighted = self.whiten(design)
        block = SimpleNamespace(
            g=self.g, ramp=self.ramp, chol=chol, d=self.d.copy(), w=self.w.copy(),
            normal=weighted.T @ weighted, rhs=weighted.T @ self.whiten(self.d),
            last=None,
        )
        self._normal_cache = block
        return block

    def _solve_active_set(self, smooth, boundary, lo, hi, max_iter, verbose):
        """`method="active_set"`: `(result, None)`, or `(None, why)` if refused.

        Refused under the conditions `METHODS` lists; the caller warns and
        falls back (`EXACT_FALLBACK_MAX_PARAMS`).
        """
        block = self._data_block()
        total = block.normal + (smooth.T @ smooth + boundary.T @ boundary).toarray()
        problem = None
        try:
            qp = box_qp(total, block.rhs, lo, hi, x0=block.last, max_iter=max_iter)
        except (RuntimeError, np.linalg.LinAlgError) as err:
            problem = str(err)
        else:
            cond = 1.0 / qp.rcond if qp.rcond > 0 else np.inf
            if cond > ACTIVE_SET_MAX_COND:
                problem = f"its free block's condition number is ~{cond:.1e}"
            elif qp.kkt > ACTIVE_SET_MAX_KKT:
                problem = f"its KKT residual is {qp.kkt:.1e}"
        if problem is not None:
            return None, problem

        x = qp.x
        block.last = x
        n = self.n_slip
        prediction = self.g @ x[:n] + (self.ramp @ x[n:] if self.n_ramp else 0.0)
        misfit = self.whiten(self.d - prediction)
        rough, edge = smooth @ x, boundary @ x
        free = ~(qp.at_lo | qp.at_hi)
        if verbose:
            print(f"active set: {qp.nit} steps, {int(free.sum())} of {x.size} "
                  f"parameters free, condition ~{1.0 / qp.rcond:.1e}, KKT {qp.kkt:.1e}")
        return SimpleNamespace(
            x=x, status=1, nit=qp.nit, success=True, method="active_set",
            cost=0.5 * float(misfit @ misfit + rough @ rough + edge @ edge),
            optimality=float(np.abs((total @ x - block.rhs)[free]).max(initial=0.0)),
            kkt=qp.kkt, condition=1.0 / qp.rcond,
            active_mask=np.where(qp.at_lo, -1, np.where(qp.at_hi, 1, 0)),
            message=f"active set settled in {qp.nit} steps; KKT residual {qp.kkt:.1e}",
        ), None

    def l_curve(self, smoothing_values, **kwargs):
        """Solve at each smoothing weight and tabulate misfit against roughness.

        The corner of the resulting curve is the conventional choice of weight --
        past it the model stops improving the fit and only gets rougher. Returned
        as an `xarray.Dataset` so it can be persisted and plotted; the
        models themselves come back in `models`.

        Swept from **large to small** smoothing deliberately: a smoother problem
        is better conditioned and converges in fewer iterations, and the sweep's
        total cost is dominated by its roughest end.

        `method="active_set"` (see `METHODS`) forms the data half of the
        normal equations once for the whole sweep, and each weight starts from
        the previous weight's active set.

        The sweep is serial on purpose; a worker pool gains almost nothing, for
        two reasons:

        * **Load imbalance dominates.** The weights do not cost the same, and the
          most expensive is usually one that runs to `DEFAULT_MAX_ITER` -- whose
          result is meaningless (`converged` is False) and should be dropped,
          not parallelised.
        * **Threads cannot overlap the solver.** scipy's `lsmr` is pure Python,
          so its iteration loop holds the GIL between matrix-vector products,
          and the dense products already use multi-threaded BLAS.

        What makes a sweep faster is `DEFAULT_LSMR_TOL` and dropping the weights
        that never converge. Green's assembly is serial too; see
        `slipsolve.greens`.
        """
        import xarray as xr

        values = np.sort(np.asarray(smoothing_values, dtype=float))[::-1]
        models = [self.solve(smoothing=v, **kwargs) for v in values]

        rows = []
        for model in models:
            rows.append((model.rms_misfit, model.roughness, model.variance_reduction,
                         model.max_slip, model.moment_magnitude,
                         int(model.result.nit), int(model.result.status),
                         bool(model.converged)))

        rows = np.array(rows, dtype=float)
        ds = xr.Dataset(
            {
                "rms_misfit": ("smoothing", rows[:, 0]),
                "roughness": ("smoothing", rows[:, 1]),
                "variance_reduction": ("smoothing", rows[:, 2]),
                "max_slip": ("smoothing", rows[:, 3]),
                "moment_magnitude": ("smoothing", rows[:, 4]),
                "iterations": ("smoothing", rows[:, 5].astype(int)),
                "status": ("smoothing", rows[:, 6].astype(int)),
                "converged": ("smoothing", rows[:, 7].astype(bool)),
            },
            coords={"smoothing": values},
        )
        ds.attrs["n_observations"] = self.obs.n
        ds.attrs["n_parameters"] = self.n_param
        return ds, models

    def __repr__(self):
        return (f"<SlipInversion obs={self.obs.n} elements={self.mesh.n_elements} "
                f"params={self.n_param}>")


class _StackedOperator(LinearOperator):
    """`[dense; sparse; sparse]` applied without materialising the stack.

    Keeping the smoothing and boundary blocks sparse is not a micro-optimisation:
    at realistic sizes the smoothing block has more rows than the data does, and
    densifying it would both dominate memory and make every matrix-vector product
    several times more expensive than the physics it is regularising.
    """

    def __init__(self, dense, *sparse_blocks):
        self.dense = dense
        self.blocks = [b for b in sparse_blocks if b.shape[0] > 0]
        rows = dense.shape[0] + sum(b.shape[0] for b in self.blocks)
        super().__init__(dtype=np.float64, shape=(rows, dense.shape[1]))

    def _matvec(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return np.concatenate([self.dense @ x] + [b @ x for b in self.blocks])

    def _rmatvec(self, y):
        y = np.asarray(y, dtype=float).ravel()
        n = self.dense.shape[0]
        out = self.dense.T @ y[:n]
        for block in self.blocks:
            out = out + block.T @ y[n:n + block.shape[0]]
            n += block.shape[0]
        return out


class SlipModel:
    """A solved slip distribution, with its fit statistics and provenance."""

    STAGE = "slip_model"

    def __init__(self, inversion, x, result, smoothing, smooth_matrix, options):
        self.mesh = inversion.mesh
        self.obs = inversion.obs
        self.x = np.asarray(x, dtype=float)
        self.result = result
        self.smoothing = smoothing
        self.options = options
        self._smooth = smooth_matrix
        self._inversion = inversion

        self.basis = getattr(inversion, "basis", None) or make_basis(self.mesh)
        n = self.basis.n_basis
        self.strike_slip = self.x[:n]
        self.dip_slip = self.x[n:2 * n]
        self.ramp = self.x[2 * n:]
        self.ramp_labels = inversion.ramp_labels

        design = (np.hstack([inversion.g, inversion.ramp])
                  if inversion.n_ramp else inversion.g)
        self.prediction = design @ self.x
        self.data = inversion.d
        #: Observed minus modelled, the geodetic convention -- so a positive
        #: residual is displacement the model failed to account for, and the
        #: residual map reads in the same sense as the data map beside it.
        self.residual = self.data - self.prediction
        self._ramp_design = inversion.ramp

    # -- fit ---------------------------------------------------------------
    @property
    def inversion(self):
        """The `SlipInversion` this model came out of.

        Worth having in public because the Green's matrix is the expensive part
        and it is already built: re-solving at another smoothing weight, or
        sweeping `SlipInversion.l_curve`, costs nothing extra through this
        handle and a full re-assembly without it. That matters most after
        `slipsolve.predict.iterate_sampling`, which returns a model
        built on observations the caller never assembled a matrix for.

        A model from `load` has no matrix -- `G` is deliberately not
        saved -- so the object comes back with `g is None` and can be read but
        not re-solved.
        """
        return self._inversion

    @property
    def converged(self):
        """True if the solver stopped on a tolerance rather than the iteration cap.

        Every statistic below is meaningless otherwise, which is why it is a
        property of the model and not something buried in a log line.
        """
        return bool(self.result.status > 0 and self.result.nit < self.options["max_iter"])

    @property
    def variance_reduction(self):
        """Percentage of the data's variance explained by the model."""
        total = float(self.data @ self.data)
        if total == 0:
            return 0.0
        return 100.0 * (total - float(self.residual @ self.residual)) / total

    @property
    def rms_misfit(self):
        return float(np.sqrt(np.mean(self.residual ** 2)))

    @property
    def chi_square(self):
        """`r' Cd^-1 r` -- the misfit in the metric that was actually minimised.

        `variance_reduction` and `rms_misfit` are computed on the
        **unweighted** residual, so with a covariance model they report a
        different quantity from the one the solver optimised. Both are still
        worth having -- rms in millimetres is what a reader understands -- but
        this is the one that says whether the fit is consistent with the stated
        noise.

        **`None` unless a covariance model was supplied**, and that is the
        point rather than a limitation. Without one the whitening is a
        multiplication by `1/sqrt(n)`, so the sum of squares comes out around
        `1e-4` times the residual norm -- a number in arbitrary units that looks
        like a chi-square, reads as a spectacularly good fit, and means nothing.
        A statistic that is only interpretable against a stated noise model should
        be absent when no noise model was stated.

        Also `None` for a model whose inversion is gone (a loaded one), since
        whitening needs the covariance factors and those are not saved.
        """
        inversion = self._inversion
        if inversion is None or not hasattr(inversion, "whiten"):
            return None
        if getattr(inversion, "covariance", None) is None:
            return None
        whitened = inversion.whiten(self.residual)
        return float(whitened @ whitened)

    @property
    def reduced_chi_square(self):
        """`chi_square` per observation. **1.0 means the fit matches the noise.**

        Above 1 the model cannot explain the data to within its stated errors --
        either the noise model is optimistic or the fault geometry is wrong. Below
        1 the data is being over-fitted, or the noise was over-estimated.

        Divided by the observation count rather than by a degree-of-freedom count,
        deliberately. The nominal `n_obs - n_param` is meaningless here: this
        problem routinely has more parameters than observations and the number
        would go negative. The honest denominator is `n_obs - trace(R)`, and
        since a heavily damped inversion resolves only tens of parameters against
        thousands of observations, `n_obs` is usually within a few percent of it.
        Take `trace(R)` from `slipsolve.resolution.resolution_report` when that
        matters.
        """
        chi2 = self.chi_square
        if chi2 is None:
            return None
        return chi2 / max(self.data.size, 1)

    @property
    def roughness(self):
        """RMS of the smoothing operator applied to the model."""
        v = self._smooth @ self.x
        return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0

    @property
    def slip_magnitude(self):
        return np.hypot(self.strike_slip, self.dip_slip)

    @property
    def max_slip(self):
        return float(self.slip_magnitude.max())

    def moment(self, shear_modulus=None, rule="integrated", order=MOMENT_ORDER):
        """Scalar seismic moment, `integral(mu * |slip|) dA`, in newton-metres.

        `shear_modulus` may be a scalar, one value per parameter, or a
        `slipsolve.edgrn.VelocityModel`; `None` takes the inversion's own
        velocity model, else 30 GPa. That is the point of a layered inversion
        reaching this far: a uniform 30 GPa overstates the rigidity of the shallow
        crust, so it overstates the moment of shallow slip.

        `rule` says how the integral is taken:

        * `"integrated"` (default) -- over the slip field the model holds. A
          velocity model's rigidity is averaged **exactly** over each piece
          (`slipsolve.edgrn.VelocityModel.triangle_mean`, steps included);
          the slip magnitude, which a tent field makes nonlinear inside an element,
          is taken at the centroids of `order**2` sub-triangles per element. For
          element-constant slip that is exact and `order` is ignored. One value per
          parameter is interpolated with the basis.
        * `"lumped"` -- `sum(mu_i * A_i * |s_i|)`, rigidity at each parameter's
          own depth over its lumped area (a third of a node's 1-ring). Kept
          because the export node tables partition it exactly (`to_vertex_text`).

        The lumped rule is biased on a layered crust, by an amount that is not a
        constant: a node's third of its 1-ring reaches an element height above and
        below it, all at the rigidity of the node's own depth, so it misplaces any
        rigidity step it straddles. The bias can have either sign, depending on
        where the nodes sit against the interfaces, and grows as a rougher field
        makes the node values less representative of the ground around them.

        A fault between blocks of different rigidity (a
        `slipsolve.bimaterial.Bimaterial` engine) takes the harmonic mean
        of the two blocks' rigidities, `rigidity_factor` times the reference
        block's; `shear_modulus`, if given, is the reference block's.
        """
        factor = getattr(self, "rigidity_factor", None)
        if rule == "lumped":
            areas = self.basis.lumped_areas()
            if shear_modulus is None:
                shear_modulus = self._default_shear_modulus()
            elif hasattr(shear_modulus, "at"):
                shear_modulus = shear_modulus.at(self._parameter_depths(), "mu")
            mu = np.broadcast_to(np.asarray(shear_modulus, dtype=float), areas.shape)
            if factor is not None:
                mu = mu * _per_parameter(self.basis, self.mesh, factor)
            return float(np.sum(mu * areas * self.slip_magnitude))
        if rule != "integrated":
            raise ValueError(f"rule must be 'integrated' or 'lumped', got {rule!r}")

        rigidity = self._rigidity() if shear_modulus is None else shear_modulus
        layered = hasattr(rigidity, "triangle_mean")
        mu = None if layered else np.asarray(rigidity, dtype=float)
        if mu is not None and mu.ndim and mu.shape != (self.basis.n_basis,):
            raise ValueError(f"shear_modulus has {mu.size} values for "
                             f"{self.basis.n_basis} parameters")
        triangles, areas = self.mesh.triangles, self.mesh.areas
        corner_z = self.mesh.nodes[:, 2][triangles]

        if self.basis.name == "element":
            if layered:
                mu = rigidity.triangle_mean(corner_z)
            if factor is not None:
                mu = mu * factor
            return float(np.sum(mu * areas * np.hypot(*self.element_slip.T)))

        from .layered import _sub_triangles

        pieces = _sub_triangles(order)
        centroid = pieces.mean(axis=1)
        magnitude = np.hypot(self.strike_slip[triangles] @ centroid.T,
                             self.dip_slip[triangles] @ centroid.T)
        if layered:
            mu = rigidity.triangle_mean(np.einsum("ej,kvj->ekv", corner_z, pieces))
        elif mu.ndim:
            # Linear across each element, so a piece's mean is its centroid value.
            mu = mu[triangles] @ centroid.T
        if factor is not None:
            mu = mu * np.asarray(factor, dtype=float)[:, None]
        return float(np.sum(mu * magnitude * (areas[:, None] / pieces.shape[0])))

    @property
    def rigidity_factor(self):
        """Per element, the moment's rigidity over the rigidity model's; `None` for one block.

        Set by a `slipsolve.bimaterial.Bimaterial` engine: the harmonic
        mean of the two blocks' rigidities over the reference block's. A loaded
        model reads it from its options, so it survives `save`.
        """
        engine = getattr(getattr(self, "_inversion", None), "engine", None)
        if hasattr(engine, "rigidity_factor"):
            return engine.rigidity_factor(self.mesh)
        described = (getattr(self, "options", None) or {}).get("bimaterial")
        if described and "contrast" in described:
            from .bimaterial import RigidityContrast

            return RigidityContrast.from_dict(described["contrast"]).element_factor(self.mesh)
        return None

    def _rigidity(self):
        """The inversion's own velocity model, else `DEFAULT_SHEAR_MODULUS`."""
        model = getattr(self._inversion, "velocity_model", None) or \
            self.options.get("velocity_model")
        if model is None:
            return DEFAULT_SHEAR_MODULUS
        if hasattr(model, "at"):
            return model
        from .edgrn import VelocityModel

        return VelocityModel.from_dict(model)

    def _default_shear_modulus(self):
        """The rigidity at each parameter's own depth, or the flat default."""
        rigidity = self._rigidity()
        if hasattr(rigidity, "at"):
            return rigidity.at(self._parameter_depths(), "mu")
        return rigidity

    @property
    def shear_modulus(self):
        """The rigidity at each parameter's own depth.

        What `moment(rule="lumped")` weights each parameter with, and what the
        export tables print. The default integrated moment averages the rigidity
        over each element's depth range instead, which differs wherever an element
        straddles a layer interface. With a rigidity contrast it is the harmonic
        mean the moment uses.
        """
        mu = np.broadcast_to(np.asarray(self._default_shear_modulus(), dtype=float),
                             self.basis.lumped_areas().shape)
        factor = self.rigidity_factor
        return mu if factor is None else mu * _per_parameter(self.basis, self.mesh, factor)

    def _parameter_depths(self):
        """Depth of each parameter: an element centroid, or a node."""
        if self.basis.name == "element":
            return self.mesh.centroids[:, 2]
        return self.mesh.nodes[:, 2]

    @property
    def moment_magnitude(self):
        m0 = self.moment()
        return float((2.0 / 3.0) * (np.log10(m0) - 9.1)) if m0 > 0 else float("nan")

    def track_residual(self, name):
        """One track's residual, observed minus modelled."""
        return self.residual[self.obs.track_mask(name)]

    @property
    def ramp_design(self):
        """The nuisance ramp's design columns, `(n_obs, n_ramp)`.

        Rebuilt from the observations when the model came from `load` --
        `ramp_columns` is a pure function of `obs` and the ramp kind, the
        same reasoning that lets `roughness` recover its smoothing operator
        without storing a sparse matrix.
        """
        if self._ramp_design is None:
            kind = self.options.get("ramp")
            if kind is None:
                # Saved without the kind in its options; the column count per
                # track identifies it exactly.
                kind = ramp_kind_for(self.ramp.size, len(self.obs.tracks))
            self._ramp_design = ramp_columns(self.obs, kind)[0]
        return self._ramp_design

    @property
    def ramp_prediction(self):
        """The nuisance surface alone, per observation -- zeros without a ramp.

        What the unbounded per-track columns absorbed: an unwrapped
        interferogram's arbitrary constant plus whatever orbital or ionospheric
        trend sits on top. It is *not* part of the fault's prediction, and
        looking at the two together is how a ramp that has drifted into eating
        the earthquake gets caught -- variance reduction cannot see it (see
        `slipsolve.regularize.ramp_columns`).
        """
        if self.ramp.size == 0:
            return np.zeros_like(self.data)
        return self.ramp_design @ self.ramp

    @property
    def fault_prediction(self):
        """The model with its ramp removed: displacement the *slip* explains.

        This is the panel to compare against `data - ramp_prediction`; the
        full `prediction` flatters the fit by however much the nuisance
        columns are carrying.
        """
        return self.prediction - self.ramp_prediction

    # -- forward -----------------------------------------------------------
    @property
    def element_slip(self):
        """`(n_elements, 2)` slip per element, whatever the basis.

        The engines are defined on elements, so this is the form anything that
        evaluates the model needs. For element-constant slip it is the parameters
        themselves; for a tent field it is each element's mean of its three nodes.
        """
        return np.column_stack([self.basis.element_values(self.strike_slip),
                                self.basis.element_values(self.dip_slip)])

    def forward(self, x, y, look=None):
        """Predict displacement at arbitrary points from this slip model."""
        engine = self._inversion.engine
        if getattr(self._inversion, "exact_basis", False):
            # The engine assembled in the basis, so hand it the coefficients.
            return engine.forward(self.mesh, self.x[:2 * self.basis.n_basis],
                                  x, y, look=look, basis=self.basis)
        return engine.forward(self.mesh, self.element_slip, x, y, look=look)

    def surface_displacement(self, spacing=1000.0, pad=50e3, bounds=None,
                             exclude_within=None, block=4096):
        """The full three-component surface displacement field this model predicts.

        Returns an `xarray.Dataset` on the mesh's own local frame with
        `ux`, `uy` and `uz` -- east, north and up in metres, positive in
        those directions. This is the model's *prediction of the ground*, not of
        any satellite: line of sight collapses three components into one, and what
        an inversion recovers is the full vector, so it is worth being able to
        look at it. Ascending and descending line-of-sight fields can be checked
        against it, and the horizontal field is what a GPS network would measure.

        `bounds` is `(x_min, x_max, y_min, y_max)` in local-frame metres;
        without it the mesh's footprint plus `pad` is used.

        Evaluated in blocks of `block` points, because the engines build a
        `(points, 3, 2 * n_elements)` array and a 1 km grid over a real
        footprint would ask for gigabytes of it. The block never materialises
        more than a few tens of megabytes.

        It forwards `element_slip`, each element's mean slip, even where
        `forward` would use the nodal tents.

        Grid points on the fault trace are a genuine singularity, so points within
        `exclude_within` of the surface trace come back NaN rather than
        non-finite or, worse, enormous. It defaults to the mesh's own element
        size, which is the scale below which a discretised fault does not mean
        anything anyway.
        """
        import xarray as xr

        if self.mesh.frame is None:
            raise ValueError(
                "surface_displacement needs the mesh's LocalFrame to georeference "
                "the grid; this mesh has none."
            )
        nodes = self.mesh.nodes
        if bounds is None:
            bounds = (nodes[:, 0].min() - pad, nodes[:, 0].max() + pad,
                      nodes[:, 1].min() - pad, nodes[:, 1].max() + pad)
        x_min, x_max, y_min, y_max = (float(v) for v in bounds)
        x = np.arange(x_min, x_max + spacing, spacing)
        # Descending, so the raster is north-up like every other grid here.
        y = np.arange(y_max, y_min - spacing, -spacing)
        xx, yy = np.meshgrid(x, y)

        if exclude_within is None:
            # The 90th percentile, not the mean: on a graded mesh the element
            # area spans two orders of magnitude and a mean is then smaller than
            # most of the elements it is meant to stand for.
            exclude_within = float(
                np.sqrt(2.0 * np.percentile(self.mesh.areas, 90)))
        # The fault outcrops along its shallowest node row; that is where a
        # dislocation solution is singular at the free surface.
        top = nodes[self.mesh.boundary_nodes("top")]
        keep = _far_enough(xx.ravel(), yy.ravel(), top, exclude_within)

        flat = np.full((xx.size, 3), np.nan)
        index = np.nonzero(keep)[0]
        for start in range(0, index.size, int(block)):
            at = index[start:start + int(block)]
            flat[at] = self._inversion.engine.forward(
                self.mesh, self.element_slip, xx.ravel()[at], yy.ravel()[at])

        ds = xr.Dataset(
            {name: (("y", "x"), flat[:, i].reshape(xx.shape).astype(np.float32))
             for i, name in enumerate(("ux", "uy", "uz"))},
            coords={"y": y, "x": x},
        )
        for name, direction in (("ux", "east"), ("uy", "north"), ("uz", "up")):
            ds[name].attrs.update(units="m", long_name=f"surface displacement, {direction}")
        ds.attrs.update(engine=self._inversion.engine.name, basis=self.basis.name,
                        smoothing=self.smoothing, spacing=float(spacing),
                        exclude_within=float(exclude_within))
        # `local_crs`, not `crs`: this grid's x/y are local metres, and the frame's
        # bare projection is offset from them by the origin's projected position
        # (its false easting, and its distance from the equator) -- an error
        # `to_grd` would then bake into a lon/lat file.
        return ds.rio.write_crs(self.mesh.frame.local_crs)

    def to_grd(self, outdir, fields=("ux", "uy", "uz"), **kwargs):
        """Write the surface displacement field as GMT `.grd` files.

        One single-variable grid per component, reprojected to lon/lat by
        `slipsolve._geo.write_grd`, so the files drop straight into an existing
        GMT workflow. Extra keyword arguments go to `surface_displacement`.
        """
        from pathlib import Path

        from ._geo import write_grd

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid = self.surface_displacement(**kwargs)

        unknown = set(fields) - set(grid.data_vars)
        if unknown:
            raise KeyError(
                f"Unknown field(s) {sorted(unknown)}; available: {sorted(grid.data_vars)}")
        # `local_crs`, as `surface_displacement` tagged the grid. Re-tagging with
        # the frame's bare projection would write a plausible-looking file far
        # from the fault.
        return [write_grd(grid[name].rio.write_crs(self.mesh.frame.local_crs),
                          outdir / f"{name}.grd") for name in fields]

    # -- output ------------------------------------------------------------
    def to_dataset(self):
        """The model as an `xarray.Dataset` on `element` and `obs`.

        Slip is reported **per element** in either basis, so the file format does
        not change with the parameterization; the parameters themselves are kept
        alongside under `coefficient`.
        """
        import xarray as xr

        lon, lat = (self.mesh.frame.to_lonlat(*self.mesh.centroids[:, :2].T)
                    if self.mesh.frame is not None else (None, None))
        element = self.element_slip
        data = {
            "strike_slip": ("element", element[:, 0]),
            "dip_slip": ("element", element[:, 1]),
            "slip": ("element", np.hypot(*element.T)),
            "coefficient_strike_slip": ("basis", self.strike_slip),
            "coefficient_dip_slip": ("basis", self.dip_slip),
            "area": ("element", self.mesh.areas),
            "depth": ("element", self.mesh.centroids[:, 2]),
            "along_strike": ("element", self.mesh.element_params[:, 0]),
            "strike": ("element", self.mesh.strike),
            "dip": ("element", self.mesh.dip),
            "data": ("obs", self.data),
            "prediction": ("obs", self.prediction),
            "ramp_prediction": ("obs", self.ramp_prediction),
            "residual": ("obs", self.residual),
        }
        if lon is not None:
            data["lon"] = ("element", np.asarray(lon))
            data["lat"] = ("element", np.asarray(lat))

        ds = xr.Dataset(data)
        ds.attrs.update(
            smoothing=self.smoothing,
            variance_reduction=self.variance_reduction,
            rms_misfit=self.rms_misfit,
            roughness=self.roughness,
            max_slip=self.max_slip,
            moment=self.moment(),
            moment_magnitude=self.moment_magnitude,
            converged=int(self.converged),
            solver_status=int(self.result.status),
            solver_iterations=int(self.result.nit),
            mesh_digest=self.mesh.digest(),
            engine=self._inversion.engine.name,
            basis=self.basis.name,
            options=self.options,
        )
        if self.ramp_labels:
            ds.attrs["ramp"] = dict(zip(self.ramp_labels, self.ramp.tolist()))
        return ds

    # -- shared by the text writers ----------------------------------------
    def _refuse_if_unconverged(self):
        """A capped solve's slip is not an answer, so it does not get written."""
        if not self.converged:
            raise ValueError(
                "Refusing to write an unconverged model "
                f"(solver status {self.result.status}, {self.result.nit} iterations). "
                "Raise max_iter or increase the smoothing weight."
            )

    def _lonlat(self, points):
        """Geographic coordinates of local-frame `(n, 2+)` points."""
        if self.mesh.frame is None:
            raise ValueError(
                "Writing a geographic table needs the mesh's LocalFrame; this mesh "
                "has none. Rebuild it with frame= so its nodes can be projected."
            )
        return self.mesh.frame.to_lonlat(*points[:, :2].T)

    def _mu_at(self, shear_modulus, depths):
        """Rigidity at each depth: what was passed, else the run's own, else 30 GPa.

        Sampling at the caller's own depths is why this is shared rather than
        computed once -- an element table wants it at centroids and a node table at
        nodes, and a layered model gives different answers at the two.
        """
        if shear_modulus is None:
            model = getattr(self._inversion, "velocity_model", None) or \
                self.options.get("velocity_model")
            if model is None:
                shear_modulus = DEFAULT_SHEAR_MODULUS
            else:
                from .edgrn import VelocityModel

                shear_modulus = (model if hasattr(model, "at")
                                 else VelocityModel.from_dict(model))
        if hasattr(shear_modulus, "at"):
            shear_modulus = shear_modulus.at(depths, "mu")
        return np.broadcast_to(np.asarray(shear_modulus, dtype=float), np.shape(depths))

    def nodal_slip(self):
        """`(n_nodes, 2)` slip at the mesh's nodes, whatever the basis.

        For a nodal model these *are* the parameters -- `strike_slip` and
        `dip_slip` are already indexed like `mesh.nodes`, so nothing is
        transformed. For an element model the element values are scattered back
        onto the nodes area-weighted.

        That scatter conserves the slip integral **componentwise and exactly**:
        `P`'s rows sum to one and `lumped_areas == P.T @ areas`, so
        `sum(lumped_areas * nodal) == sum(areas * element)` for strike and dip
        slip separately. It does **not** conserve the sum of slip *magnitudes*,
        because a magnitude is not linear -- an element's value is the mean of
        three nodes, and the mean of three vectors is shorter than the mean of
        their lengths whenever they disagree. The two magnitude sums can differ
        by about a percent, so do not use one as a check on the other.
        """
        if self.basis.name == "node":
            return np.column_stack([self.strike_slip, self.dip_slip])

        from .basis import NodeBasis

        nodal = NodeBasis(self.mesh)
        scatter, lumped = nodal.projection().T, nodal.lumped_areas()
        element = self.element_slip
        return np.column_stack([
            (scatter @ (self.mesh.areas * element[:, i])) / lumped for i in (0, 1)
        ])

    def to_text(self, path, shear_modulus=None):
        """Write the reference implementation's ten-column element table.

        `element_id lon lat depth_m strike_deg dip_deg strike_slip_m dip_slip_m
        area_m2 shear_modulus_pa` -- the format of SlipSolve's
        `simple_triangular_model.txt`, so downstream GMT scripts work unchanged.

        Column 10 is the rigidity the inversion's own velocity model gives at each
        element's centroid depth, or 30 GPa if it had none -- the rigidity actually
        used rather than a constant that may not be. It is the value
        `moment(rule="lumped")` weights an element with; the default integrated
        `moment` averages the rigidity over the element's depth range, which
        differs wherever an element straddles a layer interface. With a rigidity
        contrast it is the two blocks' harmonic mean, as the moment uses.
        """
        self._refuse_if_unconverged()
        lon, lat = self._lonlat(self.mesh.centroids)
        mu = self._mu_at(shear_modulus, self.mesh.centroids[:, 2])
        factor = self.rigidity_factor
        if factor is not None:
            mu = mu * factor
        element = self.element_slip
        table = np.column_stack([
            np.arange(1, self.mesh.n_elements + 1), lon, lat,
            self.mesh.centroids[:, 2], self.mesh.strike, self.mesh.dip,
            element[:, 0], element[:, 1], self.mesh.areas, mu,
        ])
        header = ("element_id\tlongitude_deg\tlatitude_deg\tdepth_m\tstrike_deg\t"
                  "dip_deg\tstrike_slip_m\tdip_slip_m\tarea_m2\tshear_modulus_pa")
        np.savetxt(path, table, delimiter="\t", header=header, comments="")
        return path

    def to_vertex_text(self, outdir, shear_modulus=None):
        """Write the mesh's vertices and its connectivity as two tables.

        `to_text` describes elements by their *centroids*, which is enough to map
        slip but not to rebuild the fault: it says nothing about where a triangle's
        corners are or which triangles share one. This pair does, and it reports
        slip **at the nodes** -- which for `basis="node"` is what was actually
        solved for, rather than the per-element mean `to_text` prints.

        `vert_nodes.txt`, one row per node:

            node_id longitude_deg latitude_deg depth_m along_strike_m
            strike_slip_m dip_slip_m area_m2 shear_modulus_pa

        `vert_elements.txt`, one row per element:

            element_id node_1 node_2 node_3

        Conventions follow `to_text` throughout: tab-delimited, one header
        line, depth in metres negative-down, slip in metres with positive
        strike-slip left-lateral, and **1-based** ids. `element_id` is the same
        numbering `to_text` uses, so row *i* of `vert_elements.txt` is row *i* of
        the `to_text` table -- the two join without a key.

        `area_m2` is the node's **lumped** area -- a third of its 1-ring -- not an
        element area, so the rows are a partition of the fault and
        `sum(shear_modulus * area * hypot(strike_slip, dip_slip))` over this file
        reproduces `moment(rule="lumped")` exactly for a nodal model. Not the
        default `moment`, which integrates over the tent field and can differ
        from it by a percent or more on a layered crust (see its docstring). Nor
        does it agree with the same sum over the `to_text` table to better than
        about a percent; see `nodal_slip` for why that is expected rather than an
        inconsistency.

        Returns both paths.
        """
        from pathlib import Path

        from .basis import NodeBasis

        self._refuse_if_unconverged()
        nodes = self.mesh.nodes
        lon, lat = self._lonlat(nodes)
        slip = self.nodal_slip()

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        nodal = NodeBasis(self.mesh)
        mu = self._mu_at(shear_modulus, nodes[:, 2])
        factor = self.rigidity_factor
        if factor is not None:
            mu = mu * _per_parameter(nodal, self.mesh, factor)
        node_table = np.column_stack([
            np.arange(1, self.mesh.n_nodes + 1), lon, lat, nodes[:, 2],
            self.mesh.params[:, 0], slip[:, 0], slip[:, 1],
            nodal.lumped_areas(), mu,
        ])
        node_path = outdir / "vert_nodes.txt"
        # `node_id` as an integer, the rest at numpy's default precision: these two
        # files are joined on their id columns, so an id that reads back as
        # 1.000000000000000000e+00 in one and 1 in the other is a trap for anything
        # matching them as text.
        np.savetxt(node_path, node_table, delimiter="\t", comments="",
                   fmt=["%d"] + ["%.18e"] * 8,
                   header=("node_id\tlongitude_deg\tlatitude_deg\tdepth_m\t"
                           "along_strike_m\tstrike_slip_m\tdip_slip_m\tarea_m2\t"
                           "shear_modulus_pa"))

        # Integer format: a connectivity index written as 1.000000000000000000e+00
        # is read back as a float by every consumer and silently indexes nothing.
        element_path = outdir / "vert_elements.txt"
        np.savetxt(element_path,
                   np.column_stack([np.arange(1, self.mesh.n_elements + 1),
                                    self.mesh.triangles + 1]),
                   delimiter="\t", fmt="%d", comments="",
                   header="element_id\tnode_1\tnode_2\tnode_3")
        return [node_path, element_path]

    def persist(self, workspace, name=None, overwrite=False, **params):
        name = name or self.STAGE
        full = {
            "stage": name,
            "mesh": self.mesh.digest(),
            "smoothing": self.smoothing,
            "options": self.options,
            "engine": self._inversion.engine.name,
            **params,
        }
        return workspace.store(name, self.to_dataset(), full, overwrite=overwrite)

    # -- save / load -------------------------------------------------------
    def save(self, path):
        """Write the whole model to one self-contained file, a zipped Zarr store.

        Everything needed to work with the result later travels with it -- the
        slip vector, the fit, the mesh, and the observations -- in three groups
        (`model`, `mesh`, `observations`). `load` gives back a
        `SlipModel` that reports every statistic, re-exports with
        `to_text`, plots, and can `forward`-model new points.

        This is the counterpart to a long run: solve under `nohup`, save, and
        come back to it in a notebook. Unlike `persist` it needs no workspace
        and produces a single file that can just be copied off the machine that
        ran it.

        Zarr rather than netCDF: netCDF groups need a backend beyond scipy's
        group-less one, while `zarr` is already a dependency -- and its
        attributes are JSON, so the provenance dicts (`options`, `frame`,
        `quadtree`) round-trip natively instead of needing an encoding layer.

        The **Green's matrix is deliberately not saved** -- it is the largest
        object in the problem (hundreds of megabytes) and nothing downstream of a
        solved model needs it. A loaded model therefore cannot be re-solved at a
        new smoothing weight; rebuild the `SlipInversion` for that.
        """
        import zarr

        path = str(path)
        model = self.to_dataset()
        # to_dataset() splits the parameter vector into its physical parts; keep
        # the raw vector too so a load is exact rather than reassembled.
        model["x"] = ("param", self.x)
        model.attrs["ramp_labels"] = list(self.ramp_labels)
        model.attrs["engine_nu"] = float(getattr(self._inversion.engine, "nu", 0.25))

        obs = self.obs.ds.copy()
        # Zarr has no object dtype; track names ride as fixed-width text.
        obs["track"] = obs["track"].astype(str)

        store = zarr.ZipStore(path, mode="w")
        try:
            model.to_zarr(store, group="model", mode="w", consolidated=False)
            self.mesh.to_dataset().to_zarr(
                store, group="mesh", mode="a", consolidated=False)
            obs.to_zarr(store, group="observations", mode="a", consolidated=False)
        finally:
            store.close()
        return path

    @classmethod
    def load(cls, path):
        """Read back a model written by `save`."""
        import xarray as xr
        import zarr

        from .mesh import FaultMesh
        from .sampling import Observations

        path = str(path)
        store = zarr.ZipStore(path, mode="r")
        try:
            model = xr.open_zarr(store, group="model", consolidated=False).load()
            mesh = FaultMesh.from_dataset(
                xr.open_zarr(store, group="mesh", consolidated=False).load())
            obs = Observations(
                xr.open_zarr(store, group="observations", consolidated=False).load())
        finally:
            store.close()

        options = dict(model.attrs["options"])
        self = object.__new__(cls)
        self.mesh = mesh
        self.obs = obs
        self.x = np.asarray(model["x"].values, dtype=float)
        self.result = SimpleNamespace(
            status=int(model.attrs["solver_status"]),
            nit=int(model.attrs["solver_iterations"]),
            x=self.x,
        )
        # A scalar for most runs, one weight per smoothing row when the solve was given an array
        # (see `scale_smoothing`); `float` cannot read the second, so a per-row model would not reopen.
        smoothing = model.attrs["smoothing"]
        self.smoothing = (float(smoothing) if np.ndim(smoothing) == 0
                          else np.asarray(smoothing, dtype=float))
        self.options = options
        self.ramp_labels = [str(s) for s in model.attrs.get("ramp_labels", [])]

        self.basis = make_basis(mesh, str(model.attrs.get("basis", "element")))
        n = self.basis.n_basis
        self.strike_slip = self.x[:n]
        self.dip_slip = self.x[n:2 * n]
        self.ramp = self.x[2 * n:]
        self.data = np.asarray(model["data"].values, dtype=float)
        self.prediction = np.asarray(model["prediction"].values, dtype=float)
        # Recomputed rather than read back: it is a pure function of the two
        # fields above, and recomputing gives every file the observed-minus-
        # modelled sign, whatever convention it was written with.
        self.residual = self.data - self.prediction

        # The smoothing operator is a pure function of the mesh and the options,
        # so `roughness` is recoverable exactly without storing a sparse matrix.
        smooth = (neighbor_smoothing(mesh.neighbors, options["ss_ratio"],
                                     options["ds_ratio"],
                                     weighting=options.get("smooth_weighting", "uniform"),
                                     mesh=mesh)
                  if self.basis.name == "element"
                  else laplace_beltrami(mesh, options["ss_ratio"], options["ds_ratio"]))
        n_ramp = self.ramp.size
        if n_ramp:
            smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], n_ramp))],
                               format="csr")
        self._smooth = smooth
        # A stand-in for the inversion: `forward` needs only the engine, and the
        # Green's matrix is deliberately absent (see `save`).
        #
        # A layered engine is *not* reconstructed: its EDGRN tables are megabytes
        # and are not saved, so `forward` on a loaded layered model would silently
        # use the homogeneous half-space instead of saying so. Refuse, and name
        # what to do about it.
        engine_name = str(model.attrs.get("engine", HalfSpaceTDE.name))
        nu = float(model.attrs.get("engine_nu", 0.25))
        described = options.get("bimaterial")
        if described and described.get("engine") == HalfSpaceTDE.name:
            from .bimaterial import Bimaterial

            # A contrast round the closed-form engine is pure arithmetic: rebuild it.
            engine = Bimaterial.from_description(described, HalfSpaceTDE(nu))
        elif engine_name == HalfSpaceTDE.name:
            engine = HalfSpaceTDE(nu)
        else:
            engine = _MissingEngine(engine_name)
        self._inversion = SimpleNamespace(
            engine=engine, basis=self.basis,
            g=None, ramp=None, n_ramp=n_ramp, ramp_labels=self.ramp_labels,
            # As SlipInversion decides it, so `forward` takes the same path as
            # the model it was saved from.
            exact_basis=bool(getattr(engine, "supports_basis", False)),
        )
        # Rebuilt on demand from obs + the ramp kind, like the smoothing
        # operator above -- G is deliberately not saved, but the ramp columns
        # cost nothing to regenerate and the nuisance surface is worth seeing.
        self._ramp_design = None
        return self

    def __repr__(self):
        flag = "" if self.converged else " UNCONVERGED"
        return (f"<SlipModel VR={self.variance_reduction:.1f}% "
                f"max_slip={self.max_slip:.2f}m Mw={self.moment_magnitude:.2f} "
                f"roughness={self.roughness:.3g}{flag}>")
