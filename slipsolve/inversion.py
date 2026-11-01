"""The inversion itself: a bounded, regularized linear least-squares solve.

The system is the reference implementation's, stacked:

.. code-block:: text

    [ G * w        ]        [ d * w ]
    [ lam * L / nL ] * m =  [   0   ]
    [ W_boundary   ]        [   0   ]

with ``lb <= m <= ub``. ``G`` is the Green's matrix, ``w`` the per-observation
weights, ``L`` the smoothing operator normalised by its own row count (so
``lam`` means the same thing as the mesh is refined), and ``W_boundary`` the
zero-slip edge rows.

It is *not* assembled as one dense array. ``G`` is dense and ``L``/``W`` are
sparse, so the stack is wrapped in a :class:`~scipy.sparse.linalg.LinearOperator`
that applies each in its natural form: the smoothing block then costs a sparse
matrix-vector product rather than a dense one, and the 0.5-1 GB materialised copy
never exists. ``scipy.optimize.lsq_linear`` accepts a ``LinearOperator`` with
``lsq_solver="lsmr"``, which is the combination used here by default.

``solve(method="active_set")`` solves the same bounded problem in its normal
equations instead, exactly, by primal-dual active set (:func:`box_qp`); see
:data:`METHODS` for when that is faster and when it is refused.

Solver honesty matters more than usual in this problem. Runtime depends far more
on conditioning than on size, and an iteration-capped solve returns a
plausible-looking model with a meaningless variance reduction -- so
:class:`SlipModel` carries the solver's status and refuses to pretend.
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
    """True where a point is further than ``distance`` from any trace node.

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

    A layered engine is defined by its EDGRN tables, which are megabytes and are
    deliberately not saved with a model -- as the Green's matrix is not. Silently
    substituting the homogeneous half-space would let ``forward`` return plausible
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


def box_qp(N, g, lo, hi, x0=None, max_iter=100):
    """Minimise ``x'Nx/2 - g'x`` under ``lo <= x <= hi``, exactly, by primal-dual active set.

    The problem :meth:`SlipInversion.solve` hands ``lsq_linear``, written in its
    normal equations. Each step holds the variables the current guess puts on a
    bound there and solves for the rest by Cholesky. It then moves onto its bound
    every free variable that crossed one, and off its bound every held variable
    whose gradient points inside. When a step changes nothing, every KKT condition
    holds, so the answer is the minimiser and not an approximation to it.

    ``x0``, a nearby solution as in a sweep, picks the first guess (a variable
    within 0.1 mm of a bound starts on it), and the set then settles in a few
    steps. Without it every variable starts free.

    Returns a namespace: ``x``; ``nit``, the steps taken; ``at_lo``/``at_hi``, the
    final active set; ``kkt``, the largest gradient left on a free variable
    relative to ``max|g|``; and ``rcond``, LAPACK's 1-norm estimate of the
    reciprocal condition number of the free block (1 when nothing is free).

    Raises ``RuntimeError`` if the active set comes back to a guess it has
    already tried (primal-dual active set can cycle when ``N`` is not an
    M-matrix) or has not settled in ``max_iter`` steps, and
    :class:`numpy.linalg.LinAlgError` if a free block is not positive definite.
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
    tried = set()
    for step in range(1, int(max_iter) + 1):
        held = at_lo | at_hi
        free = ~held
        x = np.where(at_lo, lo, np.where(at_hi, hi, 0.0))
        block = factor = None
        if free.any():
            rhs = g[free] - N[np.ix_(free, held)] @ x[held]
            block = N[np.ix_(free, free)]
            factor = cho_factor(block, lower=True)
            x[free] = cho_solve(factor, rhs)
        mu = N @ x - g
        new_lo = has_lo & ((free & (x < lo)) | (at_lo & (mu > 0)))
        new_hi = has_hi & ((free & (x > hi)) | (at_hi & (mu < 0)))
        if np.array_equal(new_lo, at_lo) and np.array_equal(new_hi, at_hi):
            break
        tried.add(np.packbits(at_lo).tobytes() + np.packbits(at_hi).tobytes())
        if np.packbits(new_lo).tobytes() + np.packbits(new_hi).tobytes() in tried:
            raise RuntimeError(f"the active set cycled after {step} steps")
        at_lo, at_hi = new_lo, new_hi
    else:
        raise RuntimeError(f"the active set did not settle in {int(max_iter)} steps")

    rcond = 1.0
    if factor is not None:
        rcond, info = lapack.dpocon(factor[0], float(np.abs(block).sum(axis=0).max()), uplo="L")
        rcond = float(rcond) if info == 0 else 0.0
    scale = max(float(np.abs(g).max(initial=0.0)), np.finfo(float).tiny)
    kkt = float(np.abs(mu[free]).max(initial=0.0)) / scale
    return SimpleNamespace(x=x, nit=step, at_lo=at_lo, at_hi=at_hi, kkt=kkt, rcond=rcond)

#: Crustal shear modulus used for moment when no velocity model is supplied.
DEFAULT_SHEAR_MODULUS = 30e9

#: Sub-triangles per element edge in the integrated moment, ``order**2`` per element.
#:
#: The rigidity is averaged exactly over each piece; what the order resolves is the
#: slip magnitude, and its covariance with the rigidity inside a piece that
#: straddles a step. Measured against order 48-128, relative error in moment:
#:
#: * Venezuela r84 (2-4 km elements): 2.2e-3 at 1, 4.1e-4 at 2, 1.4e-4 at 4,
#:   **9.7e-6 at 8** (Mw 3e-6), 3.0e-6 at 16.
#: * the test fixture (6.7 km rows with the 2 km lid step inside the top one, dip
#:   slip changing sign in 88 of 240 elements): 4.6e-2 at 1, 1.8e-3 at 4,
#:   **6.3e-4 at 8** (Mw 2e-4), 8.2e-5 at 16, 2.8e-6 at 64.
#:
#: Sampling the rigidity at the centroids instead gave 5.4e-4 at 8 and 2.6e-3 at
#: 13 on r84 -- first order and erratic, set by where the lattice falls against
#: each interface.
MOMENT_ORDER = 8

#: Default cap on trust-region iterations.
#:
#: Generous on purpose. Iteration count tracks *conditioning*, not problem size,
#: and not monotonically in the smoothing weight either: on the reference
#: Venezuela problem, weights of 2.0, 0.3 and 0.01 each converge in about 30
#: iterations while 0.05 needs 236. A tight cap therefore silently truncates
#: isolated points of an L-curve sweep rather than the obvious rough end. Since a
#: well-conditioned solve stops long before the cap, raising it costs nothing
#: where it is not needed.
DEFAULT_MAX_ITER = 400

#: Inner-solver tolerance handed to ``lsq_linear``'s ``lsmr`` sub-problem.
#:
#: ``"auto"`` lets scipy tighten the inner tolerance as the outer trust-region
#: iteration converges, instead of solving every sub-problem to ``atol=btol=0``
#: -- which is what scipy's ``None`` default does, and which makes ``lsmr`` run
#: to its own iteration cap on every single outer step.
#:
#: Measured on the reference problem (590 elements, 860 observations, 1186
#: parameters): ``None`` takes 162 outer iterations and **174 809** matrix-vector
#: products in 22.3 s; ``"auto"`` takes 28 outer iterations and **26 359** in
#: 3.9 s -- a **5.8x** speed-up for the same variance reduction (90.06%), the two
#: slip models agreeing to 6.6 mm. Since roughly 89% of a solve is spent inside
#: those products, cutting their number is by far the largest available win.
#:
#: ⚠️ **That win is not uniform in the smoothing weight.** Re-measured 2026-07-28
#: on the test fixture (240 elements, 761 observations, 480 parameters), counting
#: matrix-vector products, ``None`` / ``"auto"``::
#:
#:     lam    2.0    1.0    0.5    0.3    0.1   0.05
#:          0.70x  2.18x  1.93x  1.91x  1.19x  0.78x
#:
#: so the adaptive tolerance *loses* at both ends of the sweep and wins by ~2x in
#: the middle, netting **1.17x** over the whole sweep -- which is how
#: :meth:`SlipInversion.l_curve` uses it, and why it stays the default. Judge it
#: over a sweep, never at one weight. Note also that outer iteration count is the
#: wrong meter: ``"auto"`` solves each sub-problem loosely, so it routinely takes
#: *more* outer steps while doing less total work.
#:
#: Do **not** reach for ``lsmr_maxiter`` instead. A hard cap is not monotone:
#: caps of 100 and 200 both drove the outer loop into ``DEFAULT_MAX_ITER`` and
#: returned materially different, *worse* models (4.15 m and 2.47 m from the
#: converged answer) while a cap of 25 was fine. The adaptive tolerance has no
#: such cliff.
DEFAULT_LSMR_TOL = "auto"

#: The solvers :meth:`SlipInversion.solve` offers. Both minimise the same
#: objective under the same bounds.
#:
#: * ``"lsq_linear"``: scipy's trust-region solve of the stacked system, through
#:   ``lsmr`` on the operator. The default, and what every model on record used.
#: * ``"active_set"``: the normal equations ``(A'A + S'S + B'B) m = A'b``,
#:   solved exactly by :func:`box_qp`. ``A'A`` and ``A'b`` do not depend on the
#:   regularization, so they are formed once per inversion and kept; a sweep
#:   over smoothing weights then pays one small factorization per weight
#:   instead of a trust-region solve over every observation row. This is the
#:   solve :class:`~slipsolve.sweep.DipSweep` and the production blocked CV
#:   always used (that CV's scores matched ``lsq_linear`` re-solves' to
#:   0.0002 mm).
#:
#: Normal equations square the condition number, so an active-set answer is
#: refused, with a warning and a fallback to ``lsq_linear``, if its free block
#: is worse conditioned than :data:`ACTIVE_SET_MAX_COND`, if its KKT residual
#: exceeds :data:`ACTIVE_SET_MAX_KKT`, if a free block is not positive
#: definite, or if the active set cycles.
METHODS = ("lsq_linear", "active_set")

#: Largest 1-norm condition number an active-set solve's free block may have.
#: The answer's relative error is bounded by about this times machine epsilon.
ACTIVE_SET_MAX_COND = 1e12

#: Largest gradient an active-set answer may leave on a free parameter,
#: relative to the largest gradient at zero slip.
ACTIVE_SET_MAX_KKT = 1e-8


class SlipInversion:
    """A mesh, a set of observations, and the Green's matrix relating them.

    The Green's matrix is built once, on construction, and reused across every
    smoothing weight -- which is what makes an L-curve sweep cheap.
    """

    def __init__(self, mesh, obs, engine=None, ramp="none", basis="element",
                 velocity_model=None, covariance=None, sees=None):
        """``engine`` and ``basis`` choose the physics and the parameterization.

        ``ramp`` adds unbounded per-track nuisance columns -- ``"none"``,
        ``"offset"``, ``"linear"``, ``"quadratic"`` or ``"cubic"``, or a
        ``{track: kind}`` dict with ``"*"`` for the rest, e.g. a cubic for the
        interferograms and an offset for pixel-offset tracks (see
        :func:`~slipsolve.regularize.ramp_columns`). Solving the ramp here
        is preferable to pre-removing it with
        :meth:`~nisar_tools.unwrap.UnwrappedStack.deramp`: a pre-removal fits the
        polynomial to the data alone and has to extrapolate under a masked signal
        region, while these columns are constrained jointly with the slip. Run
        :func:`~slipsolve.diagnostics.ramp_content` first to see what each
        degree would absorb.

        ``velocity_model`` is **not** used to compute Green's functions -- for a
        layered medium that information is already inside the engine's EDGRN
        tables, and for a homogeneous half-space the displacements do not depend
        on rigidity at all (it cancels; only Poisson's ratio survives). What it is
        for is **moment**: ``sum(mu * area * slip)`` needs a rigidity, and without
        one every reported ``Mw`` silently assumes
        :data:`DEFAULT_SHEAR_MODULUS` = 30 GPa. Pass the same
        :class:`~slipsolve.edgrn.VelocityModel` you built the tables from
        and every statistic -- ``moment``, ``moment_magnitude``, ``to_text``,
        ``repr`` -- uses the rigidity at each parameter's own depth instead.

        It matters more than it sounds. A model whose shallowest layer is 9 GPa,
        as the Venezuela CRUST2.0 profile's is, has a third the rigidity of the
        default over the top 2 km -- which is exactly where a vertical
        strike-slip fault's shallow slip sits.

        ``covariance`` replaces the per-observation weights with a full spatially
        correlated noise model: one
        :class:`~slipsolve.covariance.CovarianceModel` for every track, or
        a ``{track: model}`` mapping. The system is then solved whitened,
        ``L^-1 G m = L^-1 d`` with ``Cd = L L'``, which is what
        :meth:`whiten` applies. ``None`` keeps the diagonal weights and is
        bit-for-bit what every run before this did.

        ``sees`` restricts which faults of a mesh built by
        :meth:`~slipsolve.mesh.FaultMesh.join` a track measures: ``{track: [fault, ...]}``,
        faults by their position in the join. A track not named sees every fault. Made for an earthquake sequence
        whose radar pairs span every event while GNSS resolves each one: the GNSS for
        event 1 sees only the faults event 1 broke, so its rows get zero Green's
        functions on the others, and the radar rows see the sum. That is exact when no
        fault slipped in more than one event; slip a fault took between two events
        (afterslip in the hours between) is assigned to whichever track sees it.
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
        """The joined fault each basis function belongs to, ``(n_basis,)``.

        Read from the counts :meth:`~slipsolve.mesh.FaultMesh.join` records, so it
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
        """Cholesky factors of ``Cd``, one block per track."""
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
        """Left-multiply by ``Cd^-1/2`` -- or by the diagonal weights if no ``Cd``.

        Every place the design matrix or the data meets the solver goes through
        here, so that the resolution matrix, the data importance and the ramp
        diagnostics all describe the system that was actually solved. Missing one
        of them is caught by ``trace(R) == trace(N)``, which holds only when both
        halves whiten the same ``A``.

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
        """The same inversion with a different nuisance ramp, **sharing** ``G``.

        Only the ramp block depends on ``kind``, and it costs milliseconds while
        the Green's matrix costs tens of seconds -- so comparing two ramp degrees
        is affordable, which is what makes
        :func:`~slipsolve.diagnostics.ramp_sweep` a routine check rather
        than an hour of re-assembly. The copy shares ``g`` by reference; nothing
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
        """Solve for slip at one smoothing weight; returns a :class:`SlipModel`.

        ``method`` is ``"lsq_linear"`` (the default) or ``"active_set"``; see
        :data:`METHODS`. Both solve the same bounded problem. ``max_iter`` caps
        the trust-region iterations of the first and the active-set steps of the
        second. ``tol`` and ``lsmr_tol`` are ``lsq_linear``'s alone. The model's
        ``options["method"]`` records the solver that produced it, which is
        ``"lsq_linear"`` when an active-set answer was refused.

        ``lsmr_tol`` is the inner least-squares tolerance; see
        :data:`DEFAULT_LSMR_TOL` for why the default is ``"auto"`` and not
        scipy's ``None``. Pass ``lsmr_tol=None`` to reproduce a pre-2026-07-27
        run exactly.

        ``smooth_weighting`` applies only to the *element* basis: ``"length"``
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
            result = self._solve_active_set(scaled_smooth, boundary, lo, hi,
                                            max_iter, verbose)
        if result is None:
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
        """``method="lsq_linear"``: the stacked system through scipy's trust region."""
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

    def _data_block(self):
        """``(WA)'(WA)`` and ``(WA)'(Wd)``, the half of the normal equations the data set.

        Neither depends on the regularization, so they are kept between solves
        and a sweep forms them once. The cache holds while ``G``, the ramp and
        the whitening are the same objects (nothing writes to ``G`` in place; see
        :meth:`with_ramp`) and the data and weights the same VALUES: ``d`` can
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
        """``method="active_set"``: an ``lsq_linear``-like result, or None if refused.

        Refused, with a warning, under the conditions :data:`METHODS` lists; the
        caller then solves with ``lsq_linear``.
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
            warnings.warn(f"the active-set solve was refused ({problem}); solved "
                          "with lsq_linear instead", RuntimeWarning, stacklevel=3)
            return None

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
        )

    def l_curve(self, smoothing_values, **kwargs):
        """Solve at each smoothing weight and tabulate misfit against roughness.

        The corner of the resulting curve is the conventional choice of weight --
        past it the model stops improving the fit and only gets rougher. Returned
        as an :class:`xarray.Dataset` so it can be persisted and plotted; the
        models themselves come back in ``models``.

        Swept from **large to small** smoothing deliberately: a smoother problem
        is better conditioned and converges in fewer iterations, and the sweep's
        total cost is dominated by its roughest end.

        ``method="active_set"`` (see :data:`METHODS`) forms the data half of the
        normal equations once for the whole sweep, and each weight starts from
        the previous weight's active set.

        **The sweep is serial, deliberately.** Every weight is an independent
        solve over the same Green's matrix, which makes a worker pool the obvious
        move; measured on an 8-weight sweep (240 elements, 578 observations, 10
        cores), threads gave **1.02x** at 2 workers and got *worse* above that
        (0.96x at 4, 0.90x at 8), and a process pool -- pickling ``G`` once per
        worker -- managed only **1.10x**. Two independent reasons:

        * **Load imbalance dominates.** The sweep is not eight equal solves. On
          that run the weights cost 0.78/0.75/0.54/0.84/0.61/0.91/**12.15**/1.59
          seconds: ``lam=0.02`` ran to :data:`DEFAULT_MAX_ITER` and was **67% of
          the entire sweep** on its own, so no scheduler beats **1.50x**. The
          expensive weight is precisely the one whose result is meaningless
          (``converged`` is False) -- it should be dropped, not parallelised.
        * **Threads cannot overlap the solver.** scipy's ``lsmr`` is pure Python,
          so its iteration loop holds the GIL between matrix-vector products.
          Capping ``OMP_NUM_THREADS=1`` to rule out BLAS oversubscription changed
          nothing (1.03x at 2 workers), which is what identifies the GIL rather
          than core contention as the limit. (The dense matrix-vector product on
          its own already runs at ~34 GFLOP/s, i.e. multi-threaded BLAS.)

        A ``workers=`` argument was written, verified to give identical results,
        and then removed: it is API surface that can only make a sweep slower.
        What does make one faster is :data:`DEFAULT_LSMR_TOL` (a measured
        **5.8x** on a single solve) and dropping the weights that never converge.

        Green's assembly is not parallelisable either -- see
        :mod:`slipsolve.greens`, where threading over elements measured
        uniformly worse for a different reason again (numpy dispatch overhead).
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
    """``[dense; sparse; sparse]`` applied without materialising the stack.

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
        """The :class:`SlipInversion` this model came out of.

        Worth having in public because the Green's matrix is the expensive part
        and it is already built: re-solving at another smoothing weight, or
        sweeping :meth:`SlipInversion.l_curve`, costs nothing extra through this
        handle and a full re-assembly without it. That matters most after
        :func:`~slipsolve.predict.iterate_sampling`, which returns a model
        built on observations the caller never assembled a matrix for.

        ⚠️ A model from :meth:`load` has no matrix -- ``G`` is deliberately not
        saved -- so the object comes back with ``g is None`` and can be read but
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
        """``r' Cd^-1 r`` -- the misfit in the metric that was actually minimised.

        ⚠️ :attr:`variance_reduction` and :attr:`rms_misfit` are computed on the
        **unweighted** residual, so with a covariance model they report a
        different quantity from the one the solver optimised. Both are still
        worth having -- rms in millimetres is what a reader understands, and VR is
        what every earlier run recorded -- but this is the one that says whether
        the fit is consistent with the stated noise.

        ⚠️ **``None`` unless a covariance model was supplied**, and that is the
        point rather than a limitation. Without one the whitening is a
        multiplication by ``1/sqrt(n)``, so the sum of squares comes out around
        ``1e-4`` times the residual norm -- a number in arbitrary units that looks
        like a chi-square, reads as a spectacularly good fit, and means nothing.
        A statistic that is only interpretable against a stated noise model should
        be absent when no noise model was stated.

        Also ``None`` for a model whose inversion is gone (a loaded one), since
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
        """:attr:`chi_square` per observation. **1.0 means the fit matches the noise.**

        Above 1 the model cannot explain the data to within its stated errors --
        either the noise model is optimistic or the fault geometry is wrong. Below
        1 the data is being over-fitted, or the noise was over-estimated.

        Divided by the observation count rather than by a degree-of-freedom count,
        deliberately. The nominal ``n_obs - n_param`` is meaningless here: this
        problem routinely has more parameters than observations and the number
        would go negative. The honest denominator is ``n_obs - trace(R)``, and
        since a heavily damped inversion resolves only tens of parameters against
        thousands of observations (38.7 of 1836 on the Venezuela run), ``n_obs``
        is within a percent of it. Take ``trace(R)`` from
        :func:`~slipsolve.resolution.resolution_report` when the percent
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
        """Scalar seismic moment, ``integral(mu * |slip|) dA``, in newton-metres.

        ``shear_modulus`` may be a scalar, one value per parameter, or a
        :class:`~slipsolve.edgrn.VelocityModel`; ``None`` takes the inversion's own
        velocity model, else 30 GPa. That is the point of a layered inversion
        reaching this far: a uniform 30 GPa overstates the rigidity of the shallow
        crust, so it overstates the moment of shallow slip.

        ``rule`` says how the integral is taken:

        * ``"integrated"`` (default) -- over the slip field the model holds. A
          velocity model's rigidity is averaged **exactly** over each piece
          (:meth:`~slipsolve.edgrn.VelocityModel.triangle_mean`, steps included);
          the slip magnitude, which a tent field makes nonlinear inside an element,
          is taken at the centroids of ``order**2`` sub-triangles per element. For
          element-constant slip that is exact and ``order`` is ignored. One value per
          parameter is interpolated with the basis.
        * ``"lumped"`` -- ``sum(mu_i * A_i * |s_i|)``, rigidity at each parameter's
          own depth over its lumped area (a third of a node's 1-ring). What this
          method returned before 2026-09-25, kept because the export node tables
          partition it exactly (:meth:`to_vertex_text`).

        ⚠️ **The lumped rule reads HIGH on a layered crust, by an amount that is not
        a constant.** A node's third of its 1-ring reaches an element height above
        and below it, all at the rigidity of the node's own depth -- across the
        9 -> 34 GPa step at 2 km, for one. Measured on the Venezuela r84 model:
        +1.9% in moment, Mw +0.0054; across rank-84 arms +0.0048...+0.0064, and
        +0.0085 / +0.0111 at rank 240 / 340, where a rougher field makes the node
        values less representative of the ground they stand for.

        A fault between blocks of different rigidity (a
        :class:`~slipsolve.bimaterial.Bimaterial` engine) takes the harmonic mean
        of the two blocks' rigidities, :attr:`rigidity_factor` times the reference
        block's; ``shear_modulus``, if given, is the reference block's.
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
        """Per element, the moment's rigidity over the rigidity model's; ``None`` for one block.

        Set by a :class:`~slipsolve.bimaterial.Bimaterial` engine: the harmonic
        mean of the two blocks' rigidities over the reference block's. A loaded
        model reads it from its options, so it survives :meth:`save`.
        """
        engine = getattr(getattr(self, "_inversion", None), "engine", None)
        if hasattr(engine, "rigidity_factor"):
            return engine.rigidity_factor(self.mesh)
        described = (getattr(self, "options", None) or {}).get("bimaterial")
        if described:
            from .bimaterial import RigidityContrast

            return RigidityContrast.from_dict(described["contrast"]).element_factor(self.mesh)
        return None

    def _rigidity(self):
        """The inversion's own velocity model, else ``DEFAULT_SHEAR_MODULUS``."""
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

        What ``moment(rule="lumped")`` weights each parameter with, and what the
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
        """The nuisance ramp's design columns, ``(n_obs, n_ramp)``.

        Rebuilt from the observations when the model came from :meth:`load` --
        ``ramp_columns`` is a pure function of ``obs`` and the ramp kind, the
        same reasoning that lets ``roughness`` recover its smoothing operator
        without storing a sparse matrix.
        """
        if self._ramp_design is None:
            kind = self.options.get("ramp")
            if kind is None:
                # Written before the options carried the kind; the column count
                # per track identifies it exactly.
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
        :func:`~slipsolve.regularize.ramp_columns`).
        """
        if self.ramp.size == 0:
            return np.zeros_like(self.data)
        return self.ramp_design @ self.ramp

    @property
    def fault_prediction(self):
        """The model with its ramp removed: displacement the *slip* explains.

        This is the panel to compare against ``data - ramp_prediction``; the
        full :attr:`prediction` flatters the fit by however much the nuisance
        columns are carrying.
        """
        return self.prediction - self.ramp_prediction

    # -- forward -----------------------------------------------------------
    @property
    def element_slip(self):
        """``(n_elements, 2)`` slip per element, whatever the basis.

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

        Returns an :class:`xarray.Dataset` on the mesh's own local frame with
        ``ux``, ``uy`` and ``uz`` -- east, north and up in metres, positive in
        those directions. This is the model's *prediction of the ground*, not of
        any satellite: line of sight collapses three components into one, and what
        an inversion recovers is the full vector, so it is worth being able to
        look at it. Ascending and descending line-of-sight fields can be checked
        against it, and the horizontal field is what a GPS network would measure.

        ``bounds`` is ``(x_min, x_max, y_min, y_max)`` in local-frame metres;
        without it the mesh's footprint plus ``pad`` is used.

        Evaluated in blocks of ``block`` points, because the engines build a
        ``(points, 3, 2 * n_elements)`` array and a 1 km grid over a real
        footprint would ask for gigabytes of it -- 60 000 points against 1148
        elements is 3.3 GB. The block never materialises more than a few tens of
        megabytes.

        Grid points on the fault trace are a genuine singularity, so points within
        ``exclude_within`` of the surface trace come back NaN rather than
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
        # bare projection would place them 500 km east and 1167 km south of where
        # they are -- which `to_grd` would then bake into a lon/lat file.
        return ds.rio.write_crs(self.mesh.frame.local_crs)

    def to_grd(self, outdir, fields=("ux", "uy", "uz"), **kwargs):
        """Write the surface displacement field as GMT ``.grd`` files.

        One single-variable grid per component, reprojected to lon/lat by the
        same :func:`slipsolve._geo.write_grd`, vendored from the raster package,
        so the files drop straight into an existing GMT workflow. Extra keyword
        arguments go to :meth:`surface_displacement`.
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
        # `local_crs`, matching what `surface_displacement` already tagged the grid
        # with. Re-tagging with the frame's bare projection here is the exact
        # crs/local_crs mix-up that method's comment warns about: the grid is in
        # local metres, so the bare projection writes the file 500 km east and
        # 1167 km south of the fault -- plausible-looking, and wrong.
        return [write_grd(grid[name].rio.write_crs(self.mesh.frame.local_crs),
                          outdir / f"{name}.grd") for name in fields]

    # -- output ------------------------------------------------------------
    def to_dataset(self):
        """The model as an :class:`xarray.Dataset` on ``element`` and ``obs``.

        Slip is reported **per element** in either basis, so the file format does
        not change with the parameterization; the parameters themselves are kept
        alongside under ``coefficient``.
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
        """Geographic coordinates of local-frame ``(n, 2+)`` points."""
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
        """``(n_nodes, 2)`` slip at the mesh's nodes, whatever the basis.

        For a nodal model these *are* the parameters -- ``strike_slip`` and
        ``dip_slip`` are already indexed like ``mesh.nodes``, so nothing is
        transformed. For an element model the element values are scattered back
        onto the nodes area-weighted.

        That scatter conserves the slip integral **componentwise and exactly**:
        ``P``'s rows sum to one and ``lumped_areas == P.T @ areas``, so
        ``sum(lumped_areas * nodal) == sum(areas * element)`` for strike and dip
        slip separately (measured: agreement to 3e-16). It does **not** conserve
        the sum of slip *magnitudes*, because a magnitude is not linear -- an
        element's value is the mean of three nodes, and the mean of three vectors
        is shorter than the mean of their lengths whenever they disagree. Measured
        on the recovery fixture the two magnitude sums differ by 0.9%, so do not
        use one as a check on the other.
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

        ``element_id lon lat depth_m strike_deg dip_deg strike_slip_m dip_slip_m
        area_m2 shear_modulus_pa`` -- the format of SlipSolve's
        ``simple_triangular_model.txt``, so downstream GMT scripts work unchanged.

        Column 10 is the rigidity the inversion's own velocity model gives at each
        element's centroid depth, or 30 GPa if it had none -- the rigidity actually
        used rather than a constant that may not be. ⚠️ It is the value
        ``moment(rule="lumped")`` weights an element with; the default integrated
        :meth:`moment` averages the rigidity over the element's depth range, which
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

        ``to_text`` describes elements by their *centroids*, which is enough to map
        slip but not to rebuild the fault: it says nothing about where a triangle's
        corners are or which triangles share one. This pair does, and it reports
        slip **at the nodes** -- which for ``basis="node"`` is what was actually
        solved for, rather than the per-element mean ``to_text`` prints.

        ``vert_nodes.txt``, one row per node::

            node_id longitude_deg latitude_deg depth_m along_strike_m
            strike_slip_m dip_slip_m area_m2 shear_modulus_pa

        ``vert_elements.txt``, one row per element::

            element_id node_1 node_2 node_3

        Conventions follow :meth:`to_text` throughout: tab-delimited, one header
        line, depth in metres negative-down, slip in metres with positive
        strike-slip left-lateral, and **1-based** ids. ``element_id`` is the same
        numbering ``to_text`` uses, so row *i* of ``vert_elements.txt`` is row *i* of
        ``slip_model.txt`` -- the two join without a key.

        ``area_m2`` is the node's **lumped** area -- a third of its 1-ring -- not an
        element area, so the rows are a partition of the fault and
        ``sum(shear_modulus * area * hypot(strike_slip, dip_slip))`` over this file
        reproduces ``moment(rule="lumped")`` exactly for a nodal model. ⚠️ Not the
        default :meth:`moment`, which integrates over the tent field and reads
        0.5-2% lower on a layered crust (see its docstring). Nor does it agree with
        the same sum over ``slip_model.txt`` to better than about a percent; see
        :meth:`nodal_slip` for why that is expected rather than an inconsistency.

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
        """Write the whole model to one self-contained netCDF file.

        Everything needed to work with the result later travels with it -- the
        slip vector, the fit, the mesh, and the observations -- in three netCDF
        groups (``model``, ``mesh``, ``observations``). :meth:`load` gives back a
        :class:`SlipModel` that reports every statistic, re-exports with
        :meth:`to_text`, plots, and can :meth:`forward`-model new points.

        This is the counterpart to a long run: solve under ``nohup``, save, and
        come back to it in a notebook. Unlike :meth:`persist` it needs no
        :class:`~nisar_tools.workspace.Workspace` and produces a single file that
        can just be copied off the machine that ran it.

        The container is a **zipped Zarr store** rather than netCDF: netCDF
        groups need a backend the environment does not have (only scipy's
        group-less one is installed), while ``zarr`` is already a hard dependency
        -- and its attributes are JSON, so the provenance dicts (``options``,
        ``frame``, ``quadtree``) round-trip natively instead of needing an
        encoding layer.

        The **Green's matrix is deliberately not saved** -- it is the largest
        object in the problem (hundreds of megabytes) and nothing downstream of a
        solved model needs it. A loaded model therefore cannot be re-solved at a
        new smoothing weight; rebuild the :class:`SlipInversion` for that.
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
        """Read back a model written by :meth:`save`."""
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
        # Recomputed rather than read back. It is a pure function of the two
        # fields above, so deriving it costs nothing and means a file written
        # before the residual convention changed to observed-minus-modelled loads
        # with the current sign instead of a silently mirrored one.
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
