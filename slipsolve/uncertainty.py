"""How well a solved model is known: bounded Monte Carlo and planted recovery.

`posterior_covariance` propagates noise through the *unbounded*
solve. That stops describing the estimator once parameters sit on their limits, and a
coupling map is mostly on its limits: a node at 0 or 1 has a one-sided uncertainty, and how
many sit there is itself a result. These two functions re-solve instead.

`monte_carlo` asks "if the data had come out differently, how far would the answer
move?": it adds noise to the data `n` times and re-solves, under the same smoothing and
limits. (To the data, not to the model's prediction: re-solving a prediction smooths it a
second time, and a noiseless copy would not even be the model.) It measures the spread of the
estimator, not its error, because smoothing's bias is the same in every copy. `recovery` measures that: it plants a known
slip, makes data from it, adds the same noise and asks what comes back. Quote both.

Each solve costs one bounded solve of normal equations that are formed once, warm-started
from the model, so at a thousand parameters a copy costs a fraction of a second. Both
return an `Ensemble`: an array of solutions, with `Ensemble.apply` to read any
statistic of the model class from each one (`lambda m: m.moment()`, `lambda m: m.coupling`,
`lambda m: m.slip_between(1, 300).moment_magnitude`).

**The noise.** A residual is what the data say the noise is, model error included, and it
is usually not white. So the default is `noise="blocks"`: the model's own
residual with its sign flipped at random, one sign per `block` metres square per track (a wild
cluster bootstrap). This keeps its size and its correlation out to the block's width, and
needs no covariance model. It suits data with many rows per block: with one row per block it
is only a random sign on each row. `"white"` is independent noise at each track's residual
rms, `"weights"` independent noise at the stated `1 / weight` (trust it only if the fit
reached it), a `CovarianceModel` or `{track: model}` draws from
it, and a callable `f(rng)` gives one array per copy.

A block must be wider than the noise's correlation, or the draw is too white and the
spread too small; and a copy's noise has the residual's size, which already lost what the fit
absorbed. These are lower bounds on the error, as any resampling of a residual is.
"""

import copy
import warnings

import numpy as np

from .calibrate import recovery_score
from .covariance import whitening_factors
from .inversion import ACTIVE_SET_MAX_KKT, box_qp, prior_target, regularization_blocks
from .regularize import scale_smoothing, slip_bounds
from .resolution import weighted_design

__all__ = ["Ensemble", "checkerboard", "monte_carlo", "recovery", "recovery_score", "with_solution"]

#: Default width of a residual block, metres.
DEFAULT_BLOCK = 50e3

#: Share of copies that may fail to settle before a warning says so.
_FAILED_WARNING = 0.02


class _Repeat:
    """One inversion's normal equations, formed once, for repeated bounded solves on new data.

    Exactly the problem `SlipInversion.solve`
    posed for `model`: its smoothing, edges, priors and limits, read back from its options.
    """

    def __init__(self, model, smoothing=None):
        inversion = model.inversion
        if inversion is None or getattr(inversion, "g", None) is None:
            raise ValueError("this needs the model's inversion and Green's matrix, which a loaded model does not have")
        options = model.options
        self.model = model
        self.inversion = inversion
        smooth, boundary = regularization_blocks(inversion, options["ss_ratio"], options["ds_ratio"],
                                                 options["boundary_ratio"], tuple(options["sides"]),
                                                 options.get("smooth_weighting", "uniform"))
        scaled = scale_smoothing(smooth, model.smoothing if smoothing is None else smoothing)
        block = inversion._data_block()
        self.total = block.normal + (scaled.T @ scaled + boundary.T @ boundary).toarray()
        self.prior_rhs = boundary.T @ prior_target(inversion, boundary)     # a prior's mean, as `solve` has it
        self.weighted = weighted_design(inversion)
        lo, hi = slip_bounds(inversion.basis.n_basis, options["strike"], options["dip"], options["polarity"])
        self.lo = np.concatenate([lo, np.full(inversion.n_ramp, -np.inf)])
        self.hi = np.concatenate([hi, np.full(inversion.n_ramp, np.inf)])
        self.max_iter = int(options["max_iter"])
        self.limits_for = getattr(inversion, "limits_for", None)
        self.coupling = options.get("coupling")
        self.resettles = (self.limits_for is not None and self.coupling is not None
                          and not options.get("rate_fixed", True))

    def _bounded(self, rhs, lo, hi, x0):
        qp = box_qp(self.total, rhs, lo, hi, x0=x0, max_iter=self.max_iter)
        if qp.kkt > ACTIVE_SET_MAX_KKT:
            raise RuntimeError(f"KKT residual {qp.kkt:.1e}")
        return qp.x

    def solve(self, d, x0, resettle=True, rounds=4, tolerance=1e-5):
        """`(x, extra)`: the bounded solution for data `d`; `extra` is what the model class needs beside it."""
        rhs = self.weighted.T @ self.inversion.whiten(d) + self.prior_rhs
        if not (resettle and self.resettles):
            return self._bounded(rhs, self.lo, self.hi, x0), None
        lo, hi, long_term = self.limits_for(x0, self.coupling)
        x = x0
        for _ in range(int(rounds)):
            x = self._bounded(rhs, lo, hi, x)
            lo, hi, updated = self.limits_for(x, self.coupling)
            change = max(float(np.abs(updated[i] - long_term[i]).max()) for i in (0, 1))
            if change <= tolerance:
                break
            long_term = updated
        return x, long_term


def _sign_blocks(ds, block, rng):
    """One sign per `block`-square cell per track, `(n_obs,)`."""
    x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
    cell = np.column_stack([np.floor(x / block), np.floor(y / block)]).astype(np.int64)
    track = np.unique(np.asarray(ds["track"].values, dtype=str), return_inverse=True)[1].ravel()
    _, index = np.unique(np.column_stack([track, cell]), axis=0, return_inverse=True)
    index = index.ravel()
    return rng.choice([-1.0, 1.0], size=int(index.max()) + 1)[index]


def _noise_drawer(model, noise, block):
    """`f(rng)`: one draw of the observation noise, `(n_obs,)`."""
    inversion = model.inversion
    obs = inversion.obs
    residual = model.residual
    weights = np.asarray(inversion.w, dtype=float)
    if callable(noise):
        return noise
    if isinstance(noise, str):
        if noise == "blocks":
            return lambda rng: residual * _sign_blocks(obs.ds, float(block), rng)
        if noise == "weights":
            return lambda rng: rng.normal(size=residual.size) / weights
        if noise == "white":
            track = np.asarray(obs.ds["track"].values, dtype=str)
            scale = np.empty_like(residual)
            for name in np.unique(track):
                rows = track == name
                scale[rows] = np.sqrt(np.mean((residual[rows] * weights[rows]) ** 2)) / weights[rows]
            return lambda rng: rng.normal(size=residual.size) * scale
        raise ValueError(f"noise must be 'blocks', 'white', 'weights', a covariance model or a callable, got {noise!r}")
    factors = whitening_factors(obs, noise)
    track = np.asarray(obs.ds["track"].values, dtype=str)
    rows = {name: np.flatnonzero(track == str(name)) for name in factors}

    def draw(rng):
        out = np.zeros(residual.size)
        for name, factor in factors.items():
            out[rows[name]] = factor @ rng.normal(size=factor.shape[0])
        return out

    return draw


def with_solution(model, x, long_term=None):
    """`model` with its parameters replaced by `x`, sharing its mesh and inversion.

    Reads any statistic of a parameter vector through the model class: the moment of a slip you
    mean to plant, one member of an ensemble. `long_term` replaces an interseismic model's rate.
    For reading the slip only: `prediction` and `residual` stay the original's.
    """
    out = copy.copy(model)
    x = np.asarray(x, dtype=float)
    n = model.basis.n_basis
    out.x = x
    out.strike_slip, out.dip_slip, out.ramp = x[:n], x[n:2 * n], x[2 * n:]
    if long_term is not None:
        out.long_term_slip = long_term
    return out


class Ensemble:
    """Many solutions of one model's problem: from `monte_carlo` or `recovery`.

    `x` is `(n, n_param)`; `centre` is what the copies were made from: the model's own
    solution for `monte_carlo`, the planted truth for `recovery`. `failed` counts the
    copies whose active set did not settle (they are left out).
    """

    def __init__(self, model, x, centre, extras, failed, noise):
        self.model = model
        self.x = np.asarray(x, dtype=float)
        self.centre = np.asarray(centre, dtype=float)
        self.extras = extras
        self.failed = int(failed)
        self.noise = noise

    def __len__(self):
        return self.x.shape[0]

    @property
    def mean(self):
        """Mean of the solutions, per parameter."""
        return self.x.mean(axis=0)

    @property
    def std(self):
        """Sample standard deviation of the solutions (`ddof=1`), per parameter."""
        return self.x.std(axis=0, ddof=1)

    @property
    def bias(self):
        """Mean over the copies minus `centre`: under a planted truth, the estimator's bias."""
        return self.mean - self.centre

    def member(self, i):
        """Copy `i` as a model of the original's own class: `with_solution`."""
        return with_solution(self.model, self.x[i], self.extras[i])

    def apply(self, statistic):
        """`statistic(model)` of every copy, stacked: `(n, ...)`."""
        return np.array([statistic(self.member(i)) for i in range(len(self))])

    def __repr__(self):
        return f"<Ensemble {len(self)} copies, noise {self.noise}, {self.failed} failed>"


def _run(model, clean, centre, n, noise, block, seed, resettle, smoothing):
    repeat = _Repeat(model, smoothing=smoothing)
    draw = _noise_drawer(model, noise, block)
    rng = np.random.default_rng(seed)
    solutions, extras, failed = [], [], 0
    for _ in range(int(n)):
        try:
            x, extra = repeat.solve(clean + draw(rng), model.x, resettle=resettle)
        except (RuntimeError, np.linalg.LinAlgError):
            failed += 1
            continue
        solutions.append(x)
        extras.append(extra)
    if not solutions:
        raise RuntimeError(f"none of the {n} copies settled: the smoothing is too weak for the active set")
    if failed > _FAILED_WARNING * n:
        warnings.warn(f"{failed} of {n} copies did not settle and are left out; the smoothing may be too weak",
                      RuntimeWarning, stacklevel=3)
    label = noise if isinstance(noise, str) else getattr(noise, "__name__", type(noise).__name__)
    return Ensemble(model, np.array(solutions), centre, extras, failed, label)


def monte_carlo(model, n=200, noise="blocks", block=DEFAULT_BLOCK, seed=0, resettle=True):
    """The spread of `model`'s estimator: `n` re-solves of its data plus noise; an `Ensemble`.

    Same smoothing, same limits, the same rotations-and-bounds iteration where the model has
    one (`resettle=False` holds an interseismic model's long-term rate). The copies centre on
    the model, so the spread says how far noise moves the answer and not how far the answer
    is from the truth: `recovery` is that. See the module docstring for `noise`.
    """
    return _run(model, model.inversion.d, model.x, n, noise, block, seed, resettle, None)


def recovery(model, truth, n=20, noise="blocks", block=DEFAULT_BLOCK, seed=0, resettle=True, smoothing=None):
    """What comes back of a planted `truth`: data made from it, noise added, re-solved; an `Ensemble`.

    `truth` is the whole parameter vector, as `model.x`. To plant slip under a model's own
    nuisance terms: `np.r_[strike, dip, model.ramp]`. It goes through the model's own design
    and is solved at its smoothing (or at `smoothing`), on its rows and under its limits, with
    the noise of `monte_carlo`, so the test is of this data set. Read the result with
    `slipsolve.calibrate.recovery_score` on the part you plant, whose mean it removes.

    Plant something the limits allow, or this measures the limits.
    """
    inversion = model.inversion
    truth = np.asarray(truth, dtype=float)
    if truth.size != inversion.n_param:
        raise ValueError(f"truth has {truth.size} values; this problem has {inversion.n_param} parameters")
    clean = inversion.g @ truth[:inversion.n_slip] + (inversion.ramp @ truth[inversion.n_slip:] if inversion.n_ramp else 0.0)
    return _run(model, clean, truth, n, noise, block, seed, resettle, smoothing)


def checkerboard(mesh, along, down, low=0.0, high=1.0, nodes=True):
    """A checkerboard on the fault, `along` x `down` metres a square: per node (or element).

    In `(s, z)`, so it follows the fault. A field for `recovery`, with values
    `low` and `high` chosen inside the limits.
    """
    params = mesh.params if nodes else mesh.element_params
    phase = np.floor(params[:, 0] / float(along)) + np.floor(-params[:, 1] / float(down))
    return np.where(phase % 2 == 0, float(high), float(low))
