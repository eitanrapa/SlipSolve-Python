"""What the data can actually resolve, and the mesh and sampling that follow.

Variance reduction measures *fit*, and a slip model can fit beautifully on a
discretisation the data never constrained -- the extra freedom is simply spent
on whatever the noise asks for. Model resolution measures the other thing: how
much of each parameter's estimate comes from the data rather than from the
regulariser. Two long-standing recipes fall straight out of it, and this module
is both, sharing one matrix:

* **Mesh from resolution** (Barnhart & Lohman 2010): make the elements large
  where resolution is poor and small where it is good, so no parameter is
  either starved or free. :func:`target_element_size` and :func:`adapt_layout`.
* **Sampling from resolution** (Lohman & Simons 2005): put the observations
  where they most improve resolution. :func:`resolution_gain` returns a field
  that :meth:`~slipsolve.sampling.Observations.from_los` already accepts
  as its ``field=`` argument, so no new sampler is needed -- only a different
  thing to quadtree.

Using one quantity for both is the point. Alternating a resolution-sized mesh
against a Wang & Fialko predicted-field sampling is two heuristics in different
currencies; driving both from ``R`` is one objective.

⚠️ **Two caveats, both real, both in the callers' way if ignored.**

1. :meth:`~slipsolve.inversion.SlipInversion.solve` is **bound
   constrained**, so ``R`` describes the *unbounded* damped problem, which is a
   different problem. Treat these numbers as a discretisation heuristic, never
   as a statement about uncertainty in the model that was actually solved. How
   far apart the two problems are is a property of the run, and it has moved a
   long way -- see :data:`MAX_BOUND_FRACTION` for the measured range.
2. ``R`` depends on the smoothing weight, so it must be evaluated at the weight
   the run uses; a resolution computed at the wrong lambda sizes the mesh for a
   different inversion.

⚠️ **There is deliberately no cheap column-norm stand-in for ``R``, and that is
measured.** ``||G[:, k]||`` -- how strongly the data sees element ``k`` -- is the
obvious way to avoid a factorisation, and it does not work: on the two-scene
synthetic fixture its correlation with depth is **-0.036** against the true
resolution's **-0.742**, and it stays flat at ~0.003 from the surface to 20 km
while resolution falls from 0.34 to 2e-6. A deeper element radiates a broader but
weaker pattern whose norm is roughly conserved; resolution decays because deep
elements *trade off with each other*, and a column norm structurally cannot see a
trade-off. Nothing is lost by removing it either -- the premise that round zero
has no smoothing weight to evaluate ``R`` at was wrong, since the loop's own
``solve_kwargs`` always carries one.

A third assumption is worth naming rather than leaving implicit: the data
covariance here is ``sigma**2 I`` modulated by the per-track weights
:class:`~slipsolve.sampling.Observations` already carries. Lohman & Simons'
other contribution is a full ``Cd`` estimated from the residual variogram, which
would enter as ``G' Cd^-1 G`` throughout. That is not built.
"""

import warnings

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.linalg import cho_factor, cho_solve

from .layout import (
    MAX_COUNT_RATIO,
    MeshLayout,
    lloyd_layout,
    require_boundaries,
    triangulate,
)

#: Ridge added to the normal equations before factorising, relative to the mean
#: diagonal. Only to keep the Cholesky well posed when a parameter is completely
#: unconstrained -- far below the smoothing term for anything the data sees.
JITTER = 1e-10

#: Grid the resolution-gain field is evaluated on, in metres. Deliberately much
#: coarser than :data:`~slipsolve.predict.DEFAULT_SPACING`: the gain is a
#: smooth sum of squared Green's functions, and it is only ever used to *rank*
#: locations, while the cost is a full design matrix at every probe point. On the
#: real Venezuela crop at 2 km that is a 30 580 x 3392 matrix -- 830 MB and two
#: minutes, per scene per round; at 5 km it is 134 MB and twenty seconds.
GAIN_SPACING = 5000.0


def _smoothing_label(smoothing):
    """``smoothing`` for an attrs dict: a float when it is one, else the array.

    Per-row weights (:func:`epic_prior`) are as much "the smoothing this was
    evaluated at" as a scalar is, and coercing them with ``float`` raises rather
    than losing information -- which is how this was found.
    """
    return (float(smoothing) if np.ndim(smoothing) == 0
            else np.asarray(smoothing, dtype=float))

def weighted_design(inversion):
    """``Cd^-1/2 [G | ramp]`` -- the design matrix the solver actually sees.

    Extracted because three functions here need it verbatim
    (:func:`_normal_equations`, :func:`data_importance`,
    :func:`resolution_report`) and because ``trace(R) == trace(N)`` holds **only
    if all three whiten the same ``A``**. When the whitening changed and one site
    was missed, that identity is what caught it -- so it is worth having exactly
    one place the change can be made.
    """
    design = (np.hstack([inversion.g, inversion.ramp]) if inversion.n_ramp
              else inversion.g)
    return inversion.whiten(design)


def _normal_equations(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """``(data, damping)`` normal-equation blocks for the solve's own operator.

    Assembled from exactly what :meth:`SlipInversion.solve` stacks -- weighted
    design, scaled smoothing, boundary rows -- so the resolution describes the
    inversion being run rather than a tidier relative of it.
    """
    from .regularize import (laplace_beltrami, neighbor_smoothing,
                             scale_smoothing)

    if inversion.basis.name == "element":
        smooth = neighbor_smoothing(inversion.mesh.neighbors, ss_ratio, ds_ratio,
                                    weighting=smooth_weighting, mesh=inversion.mesh)
    else:
        smooth = laplace_beltrami(inversion.mesh, ss_ratio, ds_ratio)
    boundary = _boundary_rows(inversion.basis, sides, boundary_ratio)

    n_ramp = inversion.n_ramp
    if n_ramp:
        smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], n_ramp))],
                           format="csr")
        boundary = sp.hstack([boundary, sp.csr_matrix((boundary.shape[0], n_ramp))],
                             format="csr")
    smooth = scale_smoothing(smooth, smoothing)

    weighted = weighted_design(inversion)

    data = weighted.T @ weighted
    damping = (smooth.T @ smooth + boundary.T @ boundary).toarray()
    return data, damping


def _boundary_rows(basis, sides, ratio):
    from .inversion import _boundary_rows as rows
    return rows(basis, sides, ratio)


def _factorized(inversion, smoothing, ss_ratio, ds_ratio, boundary_ratio, sides,
                smooth_weighting):
    """``(data, cholesky)`` of the damped normal equations, ready to solve against.

    Factored out because the model resolution matrix, its diagonal and the data
    importance are three readings of the *same* factorization, and at production
    size that factorization is the whole cost.
    """
    data, damping = _normal_equations(inversion, smoothing, ss_ratio, ds_ratio,
                                      boundary_ratio, sides, smooth_weighting)
    total = data + damping
    total.flat[:: total.shape[0] + 1] += JITTER * np.trace(total) / total.shape[0]
    return data, cho_factor(total, lower=True)


def resolution_matrix(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """The **full** model resolution matrix ``R``, not just its diagonal.

    Row ``k`` is the averaging kernel of parameter ``k``: the weights with which
    the *true* model contributes to its estimate. The diagonal says how much of
    the estimate is the parameter's own value;
    :func:`resolution_kernels` reads the off-diagonal structure, which is where
    the actual spatial resolution lives -- a parameter can have a respectable
    diagonal and still be an average over 40 km of fault.

    ``(n_param, n_param)`` float64: 35 MB at the production 1035-node mesh, and
    :func:`resolution_diagonal` was already forming it internally and discarding
    everything off the diagonal, so this costs nothing extra.
    """
    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    return cho_solve(chol, data)


def resolution_diagonal(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                        boundary_ratio=1.0, sides=("bottom", "left", "right"),
                        smooth_weighting="uniform"):
    """Per-parameter model resolution, the diagonal of ``R``.

    ``R = (A'A + S'S + B'B)^-1 A'A`` with ``A`` the weighted design matrix and
    ``S``/``B`` the smoothing and boundary blocks -- i.e. the resolution of the
    damped problem :meth:`SlipInversion.solve` sets up, minus its bounds (see
    this module's caveats). ``1`` means the estimate came entirely from the data,
    ``0`` entirely from the regulariser.

    Returns an :class:`xarray.Dataset` on the ``basis`` dimension with
    ``strike_slip``, ``dip_slip`` and their pointwise maximum ``resolution``,
    plus the ramp columns' diagonal in ``attrs`` where there are any.

    One dense Cholesky of a ``(2 n_basis + n_ramp)`` square matrix: seconds and a
    few hundred megabytes at the production 918-node mesh, which is why it is
    affordable once per round of an adaptive loop but not per candidate cell.
    """
    resolved = resolution_matrix(inversion, smoothing, ss_ratio, ds_ratio,
                                 boundary_ratio, sides, smooth_weighting)
    diagonal = np.diag(resolved).copy()

    n = inversion.basis.n_basis
    strike, dip = diagonal[:n], diagonal[n:2 * n]
    out = xr.Dataset(
        {
            "strike_slip": ("basis", strike),
            "dip_slip": ("basis", dip),
            "resolution": ("basis", np.maximum(strike, dip)),
        },
        coords={"basis": np.arange(n)},
    )
    out.attrs.update(
        smoothing=_smoothing_label(smoothing), basis=inversion.basis.name,
        n_ramp=int(inversion.n_ramp),
        # The effective number of parameters the data pays for. Falls with the
        # smoothing weight, which is the sanity check that this is a resolution
        # and not just a scaled column norm.
        effective_rank=float(np.sum(diagonal[:2 * n])),
        ramp_resolution=[float(v) for v in diagonal[2 * n:]],
    )
    return out


def parameter_params(inversion):
    """Each parameter's ``(s, z)`` position on the fault, in metres.

    The parameter plane, not 3-D space, because that is where "along strike" and
    "down dip" mean something -- and they are the two directions whose resolution
    differs most.
    """
    mesh = inversion.mesh
    return (mesh.params if inversion.basis.name != "element"
            else mesh.element_params)


def parameter_positions(inversion):
    """Each parameter's ``(x, y, z)`` position in the local frame, metres.

    The 3-D counterpart of :func:`parameter_params`, and the distance Barnhart &
    Lohman's Gaussian width is fitted against. On a steeply dipping fault it is
    close to the parameter-plane distance, but only because the surface happens
    to be nearly developable there -- the two separate on a shallow or strongly
    curved one, and it is the 3-D chord the published metric uses.
    """
    mesh = inversion.mesh
    return (mesh.nodes if inversion.basis.name != "element" else mesh.centroids)


#: Rows whose weight falls below this fraction of the row's own peak are dropped
#: from the Gaussian fit. Not a tuning knob so much as a floor: entries at or
#: below rounding noise carry no information about a width and, in the log-space
#: seed, contribute arbitrarily large negative values.
GAUSSIAN_FIT_FLOOR = 1e-4

#: Damped Gauss-Newton steps for :func:`gaussian_kernel_width`. Each is a handful
#: of array reductions over the whole block at once, so this is cheap; 40 is well
#: past convergence on every kernel measured and the step is rejected once it
#: stops improving.
GAUSSIAN_FIT_STEPS = 40


def gaussian_kernel_width(weights, distance, floor=GAUSSIAN_FIT_FLOOR,
                          steps=GAUSSIAN_FIT_STEPS):
    """One isotropic 1-sigma width per row, by least-squares Gaussian fit.

    **Barnhart & Lohman's own metric**, which :func:`resolution_kernels`'
    second moments are not: model row ``k`` of ``R`` as
    ``|R[k, j]| ~ A exp(-d_kj^2 / 2 sigma^2)`` against 3-D distance and report
    ``sigma``. One scalar per parameter, isotropic by construction -- the
    anisotropy the second moments report is genuinely there, but it is not what
    the published criterion measures.

    **Why it differs from the second moment, and by how much.** Second moments
    weight by ``d^2`` and so are dominated by the far field. Measured on the real
    six-scene model, 19.4% of a row's ``|R|`` mass sits beyond 3 sigma against
    the 0.27% a Gaussian would put there -- a 70x excess -- and the moment comes
    out **3.09x** the fitted sigma along strike (39.3 km against 11.5). Trimming
    the weakest 10% of each row brings the ratio to 1.05, which identifies the
    tails as the whole of the difference. A least-squares fit weights by residual
    in linear space instead, so it follows the core.

    ⚠️ **Fitted properly, not linearised.** Taking logs makes the problem linear
    in ``d^2``, and that is used as a starting point, but a log-space fit is
    biased by exactly the tails this metric exists to look past. Measured against
    ``scipy.optimize.curve_fit`` on a Gaussian of sigma 15 km: with a 2% flat
    floor added, weighted log fits return 28.2 km (weights ``|R|^2``) and 35.4 km
    (weights ``|R|``) where the true least-squares answer is 15.4 km. The damped
    Gauss-Newton below returns 15.43 km, and matches ``curve_fit`` to the
    reported precision on every kernel shape tried -- clean, noisy, heavy-tailed
    and floored -- while staying vectorised over every row at once (0.49 s for a
    900-parameter block).

    Returns ``nan`` for a row no Gaussian describes: fewer than three usable
    entries, or a fit that does not decay.
    """
    weights = np.abs(np.asarray(weights, dtype=float))
    distance = np.asarray(distance, dtype=float)
    if weights.ndim != 2 or distance.shape != weights.shape:
        raise ValueError(
            f"weights {weights.shape} and distance {distance.shape} must be the "
            "same 2-D shape -- one row of R against the distances from that "
            "parameter to every other")
    x = distance ** 2
    peak = np.maximum(weights.max(axis=1, keepdims=True), 1e-300)
    keep = weights >= peak * float(floor)
    kept = np.where(keep, weights, 0.0)

    amplitude, decay = _log_space_seed(weights, x, keep)
    # `decay` is 1 / (2 sigma^2), so a positive value is a kernel that falls off.
    amplitude = np.where(np.isfinite(amplitude), amplitude,
                         np.log(peak[:, 0]))
    decay = np.where(np.isfinite(decay) & (decay > 0), decay,
                     1.0 / np.maximum(x.max(axis=1), 1e-9))

    def residual_cost(a, b):
        return (np.where(keep, kept - _gaussian(a, b, x), 0.0) ** 2).sum(axis=1)

    damping = np.full(amplitude.shape, 1e-3)
    cost = residual_cost(amplitude, decay)
    for _ in range(int(steps)):
        model = np.where(keep, _gaussian(amplitude, decay, x), 0.0)
        residual = kept - model
        squared = model * model
        # Normal equations for the Jacobian of exp(a - b x) in (a, b), damped
        # Levenberg-style so a bad step is rejected rather than diverging.
        h11 = squared.sum(axis=1)
        h12 = -(x * squared).sum(axis=1)
        h22 = (x * x * squared).sum(axis=1)
        g1 = (model * residual).sum(axis=1)
        g2 = -(x * model * residual).sum(axis=1)
        det = h11 * (1 + damping) * h22 * (1 + damping) - h12 ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            step_a = (h22 * (1 + damping) * g1 - h12 * g2) / det
            step_b = (h11 * (1 + damping) * g2 - h12 * g1) / det
        trial_a = amplitude + np.nan_to_num(step_a)
        trial_b = decay + np.nan_to_num(step_b)
        trial = residual_cost(trial_a, trial_b)

        better = np.isfinite(trial) & (trial < cost) & (trial_b > 0)
        amplitude = np.where(better, trial_a, amplitude)
        decay = np.where(better, trial_b, decay)
        cost = np.where(better, trial, cost)
        damping = np.where(better, np.maximum(damping * 0.3, 1e-9),
                           np.minimum(damping * 3.0, 1e6))

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 1.0 / np.sqrt(2.0 * decay)
    # ⚠️ A row that does not decay is a *fit*, not a failure: the least-squares
    # answer for a flat row is a Gaussian so wide it is locally constant, and it
    # came back as 2.4e13 m before this guard. Nothing beyond the largest
    # distance in the row is evidence, so a width past it is extrapolation and is
    # reported as no width at all -- a hole a caller can see, rather than a
    # number that silently swamps a size field.
    reach = distance.max(axis=1)
    return np.where((keep.sum(axis=1) >= 3) & (decay > 0) & np.isfinite(sigma)
                    & (sigma <= reach), sigma, np.nan)


def _gaussian(amplitude, decay, x):
    """``exp(a - b x)`` per row, clipped where it would overflow rather than warn."""
    return np.exp(np.clip(amplitude[:, None] - decay[:, None] * x, -700.0, 700.0))


def _log_space_seed(weights, x, keep):
    """``(a, b)`` from a weighted straight-line fit of ``log|R|`` against ``d^2``.

    Only a starting point. Weights are ``|R|^2``, which is the first-order
    correction for the log's variance distortion and the least-bad linearisation
    -- but it is still biased by heavy tails, which is why it is followed by the
    real fit rather than returned.
    """
    weight = np.where(keep, weights, 0.0) ** 2
    y = np.log(np.where(keep, weights, 1.0))
    sw = weight.sum(axis=1)
    sx = (weight * x).sum(axis=1)
    sy = (weight * y).sum(axis=1)
    sxx = (weight * x * x).sum(axis=1)
    sxy = (weight * x * y).sum(axis=1)
    det = sw * sxx - sx ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(det > 0, (sw * sxy - sx * sy) / det, np.nan)
        intercept = (sy - slope * sx) / np.where(sw > 0, sw, 1.0)
    return intercept, -slope


def resolution_kernels(inversion, matrix):
    """Averaging-kernel width and offset per parameter, from the full ``R``.

    **This is the number the diagonal cannot give you.** Row ``k`` of ``R`` is the
    weighted average of the true model that the estimate of parameter ``k``
    actually represents; its *width* is the shortest wavelength that estimate can
    distinguish. A parameter can sit at resolution 0.05 and still be an average
    over tens of kilometres, in which case the elements around it are several
    names for one number and the mesh is finer than the data can support.

    Widths are second moments of ``|R[k, :]|`` about parameter ``k``'s own
    position, taken **within each slip component** (the strike-strike and dip-dip
    diagonal blocks): a cross-component weight is not a spatial spread. Returned
    per component and combined as the pointwise maximum, matching how
    :func:`resolution_diagonal` reports its diagonal.

    ``offset_along`` / ``offset_down`` are the signed centroids of the same
    weights. A large offset says the estimate is not even centred on its own
    parameter -- classically the case for deep parameters, whose kernels are
    pulled toward the surface where the data is.

    ``self_weight`` is ``|R[k, k]|`` over the row's total absolute weight: the
    fraction of the kernel that is the parameter itself.

    ``width_gaussian`` is the other convention: one isotropic 1-sigma from a
    least-squares Gaussian fit against 3-D distance, which is what Barnhart &
    Lohman report. It is not a smoothed version of the moments and does not
    bracket them -- on a kernel with heavy tails it is several times smaller. See
    :func:`gaussian_kernel_width` for which to believe when they disagree.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = inversion.basis.n_basis
    expected = 2 * n + inversion.n_ramp
    if matrix.shape != (expected, expected):
        raise ValueError(
            f"the matrix is {matrix.shape} but this inversion has {expected} "
            f"parameters ({n} basis functions x 2 components + "
            f"{inversion.n_ramp} ramp). Kernel positions come from the basis, so "
            "the matrix has to be the one this inversion produced."
        )
    params = np.asarray(parameter_params(inversion), dtype=float)
    if params.shape[0] != n:
        raise ValueError(
            f"the basis has {n} parameters but the mesh offers {params.shape[0]} "
            "positions; resolution_kernels needs the basis the matrix was built in"
        )
    s, z = params[:, 0], params[:, 1]
    # Barnhart & Lohman fit against 3-D distance, so the Gaussian width needs
    # positions in space rather than in the parameter plane.
    xyz = np.asarray(parameter_positions(inversion), dtype=float)
    distance = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=-1)

    blocks = {}
    for name, start in (("strike_slip", 0), ("dip_slip", n)):
        block = np.abs(matrix[start:start + n, start:start + n])
        mass = block.sum(axis=1)
        safe = np.maximum(mass, 1e-300)
        ds = s[None, :] - s[:, None]
        dz = z[None, :] - z[:, None]
        blocks[name] = dict(
            width_along=np.sqrt((block * ds ** 2).sum(axis=1) / safe),
            width_down=np.sqrt((block * dz ** 2).sum(axis=1) / safe),
            offset_along=(block * ds).sum(axis=1) / safe,
            offset_down=(block * dz).sum(axis=1) / safe,
            self_weight=np.abs(np.diag(block)) / safe,
            width_gaussian=gaussian_kernel_width(block, distance),
        )

    data = {}
    for field in ("width_along", "width_down", "width_gaussian",
                  "offset_along", "offset_down", "self_weight"):
        for name in blocks:
            data[f"{name}_{field}"] = ("basis", blocks[name][field])
    # The worse-resolved component is the one that bounds what the mesh may
    # claim, so the summary fields take the wider kernel and the smaller share.
    data["width_along"] = ("basis", np.maximum(
        blocks["strike_slip"]["width_along"], blocks["dip_slip"]["width_along"]))
    data["width_down"] = ("basis", np.maximum(
        blocks["strike_slip"]["width_down"], blocks["dip_slip"]["width_down"]))
    # `fmax`, not `maximum`: a row that does not decay fits no width at all, and
    # one component failing must not erase the component that succeeded.
    data["width_gaussian"] = ("basis", np.fmax(
        blocks["strike_slip"]["width_gaussian"],
        blocks["dip_slip"]["width_gaussian"]))
    data["self_weight"] = ("basis", np.minimum(
        blocks["strike_slip"]["self_weight"], blocks["dip_slip"]["self_weight"]))
    # Offsets are signed, so "worse" is the larger magnitude, not the larger value.
    for field in ("offset_along", "offset_down"):
        a = blocks["strike_slip"][field]
        b = blocks["dip_slip"][field]
        data[field] = ("basis", np.where(np.abs(a) >= np.abs(b), a, b))
    data["s"] = ("basis", s)
    data["depth"] = ("basis", -z)

    out = xr.Dataset(data, coords={"basis": np.arange(n)})
    out.attrs.update(basis=inversion.basis.name, n_ramp=int(inversion.n_ramp))
    return out


def data_importance(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                    boundary_ratio=1.0, sides=("bottom", "left", "right"),
                    smooth_weighting="uniform"):
    """The data resolution matrix's diagonal, per observation.

    ``N = A (A'A + S'S + B'B)^-1 A'`` -- how much each observation's own value
    determines its own prediction. Near 1 means the datum is doing work no other
    datum can do; near 0 means the model would predict it just as well if it were
    deleted, because its neighbours already say the same thing. The other half of
    the pair :func:`resolution_matrix` gives: **model** resolution asks what the
    parameters mean, **data** resolution asks which observations paid for them.

    ``trace(N) == trace(R)``, both being the effective number of parameters the
    data supports, which is the arithmetic check that the two are consistent --
    :func:`resolution_report` asserts it.

    Only the diagonal: the full ``N`` is ``n_obs`` square (780 MB at 9900
    observations) and nothing here needs its off-diagonal structure.
    """
    from scipy.linalg import solve_triangular

    _, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                          boundary_ratio, sides, smooth_weighting)
    weighted = weighted_design(inversion)
    # diag(N)_i = ||L^-1 a_i||^2, so one triangular solve against A' rather than
    # forming N. Costs one (n_param, n_obs) array -- 165 MB at production size.
    factor, lower = chol
    projected = solve_triangular(factor, weighted.T, lower=lower)
    return np.einsum("ij,ij->j", projected, projected)


#: Fraction by which achieved posterior standard deviations may spread around
#: the target before :func:`epic_prior` warns. EPIC's whole claim is uniformity,
#: so a prior that did not reach it is not the thing the name promises -- and the
#: failure is silent otherwise, since the weights come back looking fine.
EPIC_UNIFORMITY_TOLERANCE = 0.05

#: How far, in natural logs, a per-parameter prior weight may travel from the
#: uniform starting point. Wide enough for the 1e5-fold spread the real geometry
#: asks for and narrow enough to keep the search out of the region where every
#: posterior variance has hit its floor and the Jacobian is uninformative.
_EPIC_LOG_RANGE = 40.0


def epic_prior(inversion, sigma_target, ss_ratio=1.0, ds_ratio=3.0,
               boundary_ratio=1.0, sides=("bottom", "left", "right"),
               max_iter=80, verbose=False):
    """Ortega-Culaciati et al. (2021)'s EPIC prior, as per-row smoothing weights.

    Equal Posterior Information Condition: choose a **diagonal** prior covariance
    ``Ch`` on the regularized quantity ``h = H m`` so that every slip parameter
    comes out with the *same* posterior variance,

        ``diag[(G' Cd^-1 G + H' Ch^-1 H)^-1]_ii = sigma_target^2``

    -- a square nonlinear system in ``diag(Ch)``, solved here by trust-region
    least squares (their Coleman & Li choice). The result is returned as weights
    for :meth:`SlipInversion.solve`'s ``smoothing``, so

        ``model = inversion.solve(smoothing=epic_prior(inversion, 0.3), ...)``

    is the whole usage. With ``H`` the Laplace-Beltrami operator this is their
    **ET2**; ``sigma_target`` is in **metres of slip** -- which is only true
    when the inversion carries a real ``Cd``, so this refuses without one.
    Whitening by ``1/sqrt(n)`` per track has no units, and the "variance" it
    equalised would be a number in arbitrary units dressed as metres squared.

    **Why this rather than a scalar lambda.** Tikhonov imposes one prior variance
    on every patch regardless of how well the data sees it, so the posterior
    variance -- and with it the bias -- is whatever the observation geometry
    leaves. Where coverage is one-sided, as it permanently is here (only ~19% of
    samples north of the trace, and none along the eastern two thirds), that is
    exactly where the model is least constrained and most free to absorb noise.
    EPIC counterbalances it by construction.

    **And it is the response to a problem measured here, not an import.** Four
    criteria for choosing lambda span effective rank 21 to 345 on this data --
    L-curve 21, checkerboard recovery 79-129, chi2/n 173, GCV 257-345 -- while
    lambda itself sets the answer (measured five independent ways). No criterion
    picks the level and the level decides the model. EPIC does not fix the first
    half; it attacks the second, which is the only one still open.

    ⚠️ **It does NOT remove the choice, it renames it.** ``sigma_target`` is a
    regularization parameter like lambda, and the paper picks it with GCV. What
    changes is that it is a slip standard deviation in metres -- a quantity with
    a physical scale, unlike lambda, whose meaning here has already moved twice
    (the smoothing-units fix made it ~2400x stronger on 13.7% of edges, and
    whitening rescales it by ~3000x). The claim to test is that the *answer*
    moves less across sigma_target than across lambda.

    ⚠️ **Formally invalid under bound constraints, and knowingly so.** The
    posterior covariance above is the linear one, and :meth:`solve` is
    bound-constrained; the paper (section 5.2) says the same of its own use and
    proceeds "for expediency", justified when slip is far from its bounds.
    Measured across 20 saved production models here that is 5.8-18.6% of
    parameters on a bound (:data:`MAX_BOUND_FRACTION`), 8.6% at the production
    operating point -- defensible, and worth re-checking on an EPIC solution,
    since equalising variance into the low-sensitivity regions is exactly what
    might drive more parameters onto their bounds.

    ⚠️ **``sigma_target`` is bounded from BOTH sides, and both bounds are
    structural.** Above, by the best-resolved parameter's data-only standard
    deviation: ``H' diag(a) H`` is positive semidefinite, so a prior can only
    tighten a posterior variance and never loosen one. Below, by the smoothing
    operator's **null space**: a curvature operator does not see a slip field
    that varies linearly across the fault, so that component is constrained by
    the data alone however overwhelming the prior, and the achievable variance
    stalls. Both raise with the reachable number rather than returning weights
    that quietly missed. Measured on a two-track fixture the window was roughly
    0.3-1 m of slip standard deviation; it is a property of the geometry and the
    coverage, so measure it rather than assuming a range transfers.

    ⚠️ **Node basis only.** ``H`` must be square for the condition to be a square
    system with one prior variance per slip parameter, which is what makes
    ``diag(Ch)`` a field on the fault. ``neighbor_smoothing`` emits one row per
    undirected *edge*, so on the element basis this would solve for something
    else and mean something else; it raises rather than guess.
    """
    from scipy.optimize import least_squares

    if inversion.basis.name == "element":
        raise ValueError(
            "epic_prior needs basis='node': the EPIC condition is one prior "
            "variance per regularized quantity, and neighbor_smoothing emits one "
            "row per undirected edge rather than one per parameter, so diag(Ch) "
            "would be a field on the edges and not on the fault."
        )
    if getattr(inversion, "covariance", None) is None:
        raise ValueError(
            "epic_prior needs a noise model: sigma_target is a posterior "
            "standard deviation in METRES OF SLIP, and without a covariance "
            "`whiten` multiplies by 1/sqrt(n) per track -- which is not "
            "Cd^-1/2 and carries no units, so the variance it equalises would "
            "be a number in arbitrary units. Build the inversion with "
            "SlipInversion(..., covariance={track: CovarianceModel}). Same rule "
            "as posterior_covariance and SlipModel.chi_square."
        )
    target = float(sigma_target)
    if not target > 0:
        raise ValueError(f"sigma_target must be positive, got {sigma_target!r}")

    smooth, fixed = _epic_blocks(inversion, ss_ratio, ds_ratio, boundary_ratio,
                                 sides)
    operator = smooth.toarray() if sp.issparse(smooth) else np.asarray(smooth)
    n_rows = operator.shape[0]
    n_slip = 2 * inversion.basis.n_basis
    wanted = np.log(target ** 2)

    # ⚠️ There is a ceiling on what can be asked for, and it is not obvious.
    # `H' diag(a) H` is positive semidefinite for `a >= 0`, so adding it can only
    # *shrink* the diagonal of the inverse: the prior can tighten a parameter and
    # never loosen one. The largest posterior variance reachable is therefore the
    # data's own, and the binding one is the BEST-resolved parameter's.
    ceiling = float(np.sqrt(np.diag(np.linalg.inv(fixed))[:n_slip].min()))
    if target > ceiling:
        raise ValueError(
            f"sigma_target={target:.4g} m is above what a positive prior can "
            f"reach: the best-resolved parameter already has a posterior "
            f"standard deviation of {ceiling:.4g} m from the data alone, and "
            "regularization only tightens. Ask for less than that. (There is a "
            "floor as well as this ceiling -- see the null-space message.)"
        )

    # Variances more than ten orders below the target carry no information about
    # the residual and their reciprocal overflows; the first attempt divided by
    # 1e-300 and handed `least_squares` an array of infinities.
    floor = 1e-10 * target ** 2

    def variances(scale):
        """Posterior variance of each slip parameter at prior weights ``scale``."""
        total = fixed + operator.T @ (scale[:, None] * operator)
        inverse = np.linalg.inv(total)
        return np.maximum(np.diag(inverse)[:n_slip], floor), inverse

    def residual(log_scale):
        return np.log(variances(np.exp(log_scale))[0]) - wanted

    def jacobian(log_scale):
        scale = np.exp(log_scale)
        v, inverse = variances(scale)
        # d[M^-1]_ii / d scale_j = -([M^-1 H']_ij)^2 -- the whole Jacobian from
        # the inverse the residual already formed, verified against finite
        # differences to 8e-8. Without it this needs one factorization per
        # unknown and EPIC would not be affordable.
        projected = inverse @ operator.T
        derivative = -(projected[:n_slip] ** 2)
        return derivative * (scale / v[:, None])

    start = np.log(_epic_start(operator, fixed, n_slip, target))
    # Bounded around the scalar start, which is already the right decade. Left
    # unbounded the search walks into a region where every variance has hit the
    # floor and the Jacobian says nothing.
    fit = least_squares(residual, start, jac=jacobian, method="trf",
                        bounds=(start - _EPIC_LOG_RANGE, start + _EPIC_LOG_RANGE),
                        max_nfev=int(max_iter), verbose=2 if verbose else 0)

    scale = np.exp(fit.x)
    achieved = np.sqrt(np.maximum(variances(scale)[0], 0.0))
    spread = float(np.abs(achieved - target).max() / target)
    if spread > EPIC_UNIFORMITY_TOLERANCE:
        warnings.warn(
            f"EPIC did not reach a uniform posterior: achieved standard "
            f"deviations span {achieved.min():.4g} to {achieved.max():.4g} m "
            f"against a target of {target:.4g} m ({spread:.1%} worst deviation, "
            f"{fit.nfev} evaluations). The weights are still usable but they are "
            "not the equal-information prior the name promises -- raise "
            "max_iter, or ask for a sigma_target the data can support.",
            RuntimeWarning, stacklevel=2,
        )
    # `solve` divides the operator by its own row count, so undo that here and
    # the caller passes these straight through as `smoothing=`.
    return n_rows * np.sqrt(scale)


def _epic_blocks(inversion, ss_ratio, ds_ratio, boundary_ratio, sides):
    """``(H, P + B'B)`` -- the operator EPIC weights, and everything it does not."""
    from .regularize import laplace_beltrami

    smooth = laplace_beltrami(inversion.mesh, ss_ratio, ds_ratio)
    boundary = _boundary_rows(inversion.basis, sides, boundary_ratio)
    if inversion.n_ramp:
        # Nuisance columns are unregularized, so EPIC has no handle on them and
        # the condition is imposed on the slip parameters alone.
        smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0],
                                                   inversion.n_ramp))], format="csr")
        boundary = sp.hstack([boundary, sp.csr_matrix((boundary.shape[0],
                                                       inversion.n_ramp))], format="csr")
    design = weighted_design(inversion)
    fixed = design.T @ design + (boundary.T @ boundary).toarray()
    fixed.flat[:: fixed.shape[0] + 1] += JITTER * np.trace(fixed) / fixed.shape[0]
    return smooth, fixed


def _epic_reachable(operator, fixed, n_slip, scale):
    """Median slip posterior variance at a uniform prior weight ``scale``."""
    total = fixed + operator.T @ (scale * operator)
    try:
        return float(np.median(np.diag(np.linalg.inv(total))[:n_slip]))
    except np.linalg.LinAlgError:
        return np.nan


def _epic_start(operator, fixed, n_slip, target):
    """A uniform prior weight whose *median* posterior variance is already right.

    Starting from an arbitrary scale costs iterations and can leave the solver
    where variances have underflowed; bisecting one scalar first is a few dense
    inverses and lands the whole vector in the right decade.

    Bracketed **relative to the problem's own scale** rather than at absolute
    1e-16..1e16. The absolute bracket put ``a`` at 1e16 on a real fixture, where
    the normal matrix is singular to working precision and the variances came
    back ``nan``.

    Raises when the target is not bracketed, which is not a solver failure but a
    statement about the operator -- see :func:`epic_prior`.
    """
    base = np.trace(fixed) / max(float((operator ** 2).sum()), 1e-300)
    lo, hi = 1e-12 * base, 1e12 * base
    reachable = _epic_reachable(operator, fixed, n_slip, hi)
    if not np.isfinite(reachable) or reachable > target ** 2:
        raise ValueError(
            f"sigma_target={target:.4g} m is below what this smoothing operator "
            f"can reach: even at overwhelming prior weight the median posterior "
            f"standard deviation stalls at {np.sqrt(reachable):.4g} m. A "
            "curvature operator has a null space -- a slip field varying "
            "linearly across the fault has zero second derivative -- so the "
            "component of the model lying in it is constrained by the data "
            "alone, however strong the prior. Ask for more than that."
        )
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        # More prior weight means less posterior variance, so the bracket flips.
        if _epic_reachable(operator, fixed, n_slip, mid) > target ** 2:
            lo = mid
        else:
            hi = mid
    return np.full(operator.shape[0], np.sqrt(lo * hi))



def generalized_cross_validation(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                                 boundary_ratio=1.0,
                                 sides=("bottom", "left", "right"),
                                 smooth_weighting="uniform"):
    """Craven & Wahba's GCV score at one smoothing weight, and what it is made of.

    ``GCV = n * ||(I - N) d||^2 / trace(I - N)^2`` in the whitened space, with
    ``N`` the data resolution matrix :func:`data_importance` takes the diagonal
    of. It estimates the prediction error at a datum that was left out, without
    leaving any out -- the numerator is the residual and the denominator charges
    for every effective parameter spent reaching it, so the score turns up at
    both ends and its minimum is a choice of ``smoothing``.

    Returned as a dict: ``gcv``, ``smoothing``, ``effective_rank``,
    ``effective_parameters``, ``n_ramp``, ``rss``, ``whitened_rms`` and
    ``n_obs``. ``whitened_rms`` is **not** metres of line of sight: it is the
    residual in the space the solver works in, which is per-track rescaled by
    ``Observations.concat(normalize=)`` when there is no covariance model and
    divided by sigma when there is. Use :attr:`SlipModel.rms_misfit` for a
    misfit in metres.
    Minimise ``gcv`` over a sweep; :func:`gcv_curve` does that and is usually
    what you want.

    ⚠️ **``effective_rank`` and ``effective_parameters`` are not the same
    number.** GCV's own denominator uses ``effective_parameters`` -- the whole
    ``trace(N)``, ramp columns included, since an unregularised nuisance column
    costs very nearly a full degree of freedom. ``effective_rank`` is the
    slip-only trace, which is what every other rank in this package means, and
    so the one to compare against an L-curve corner or a calibration rung.

    **Why have it when there is already an L-curve.** Because the criteria do not
    agree, and the size of the disagreement is the point. Measured on the real
    seven-scene data, the L-curve corner sits at effective rank ~21, recovery
    against a planted checkerboard minimises at 79-129, and ``chi2/n == 1``
    arrives at ~173 -- an eight-fold spread in effective rank, and rank 21 scores
    ``rel_error`` 0.4616 against 0.3012 at the recovery optimum. GCV is a fourth
    reading, and the one most of the literature standardises on (Ortega-Culaciati
    et al. 2021 use it for every scheme they compare), so it is worth knowing
    where it lands rather than inferring it.

    ⚠️ **This describes the UNBOUNDED damped problem**, like everything
    else built on ``R``: ``N`` is a linear operator and
    :meth:`SlipInversion.solve` is bound-constrained. The residual here is
    therefore the *linear* damped solution's, not the solve's -- deliberately,
    since mixing a bounded residual with a linear trace would score neither
    problem. Real runs sit at 5.8-18.6% of parameters on a bound (see
    :data:`MAX_BOUND_FRACTION`), so the two problems are close but not the same.

    ⚠️ **GCV is not an oracle, and its own users say so.**
    Ortega-Culaciati et al. (2021, section 4) report that GCV "or any other model
    class selection method" tends to return solutions that are slightly over- or
    under-regularised, and their response is not a better criterion but an
    estimator less sensitive to the one they have. Read a GCV minimum as a fourth
    opinion, not a verdict -- and if it disagrees with the recovery optimum, the
    recovery test is the one that scored against a known answer.
    """
    from scipy.linalg import cho_solve

    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    design = weighted_design(inversion)
    # ⚠️ `whiten` ALREADY applies the diagonal weights when there is no
    # covariance model, so `whiten(d * w)` would square them -- and leave the
    # data in a different space from `weighted_design`, which whitens once. This
    # is exactly what `solve` passes.
    observed = inversion.whiten(inversion.d)

    # The linear damped solution and its residual, from the factorization that
    # is already paid for.
    estimate = cho_solve(chol, design.T @ observed)
    residual = observed - design @ estimate
    rss = float(residual @ residual)

    # trace(N) == trace(R), asserted elsewhere to 1e-14, and R is the cheaper of
    # the two here: one (n_param, n_param) solve against an (n_param, n_obs) one.
    resolved = np.diag(cho_solve(chol, data))
    n_slip = 2 * inversion.basis.n_basis
    # ⚠️ TWO different traces, and confusing them shifts every rank comparison
    # in the package. GCV's denominator must charge for *every* fitted
    # parameter, ramp columns included -- they are unregularised, so each costs
    # very nearly a whole degree of freedom. But `effective_rank` everywhere else
    # here (`resolution_diagonal`, the L-curve, the calibration ladder) counts
    # slip parameters ONLY. On a two-track offset ramp the gap is 2; on seven
    # tracks with a quadratic ramp it is 42, which is larger than the whole
    # answer at the production operating point.
    parameters = float(resolved.sum())
    rank = float(resolved[:n_slip].sum())
    n = int(observed.size)
    slack = n - parameters
    if slack <= 0:
        raise ValueError(
            f"trace(N) = {parameters:.1f} against {n} observations, so GCV's "
            "denominator is not positive -- the model has at least as many "
            "effective parameters as there are data and the score is undefined. "
            "Raise smoothing, or use fewer parameters."
        )
    return {"smoothing": _smoothing_label(smoothing),
            "gcv": n * rss / slack ** 2,
            "effective_rank": rank, "effective_parameters": parameters,
            "n_ramp": int(inversion.n_ramp), "rss": rss,
            "whitened_rms": float(np.sqrt(rss / n)), "n_obs": n}


def gcv_curve(inversion, weights, **kwargs):
    """:func:`generalized_cross_validation` over a sweep, plus where it minimises.

    Returns an :class:`xarray.Dataset` on ``smoothing`` carrying ``gcv``,
    ``effective_rank``, ``effective_parameters``, ``rss`` and
    ``whitened_rms``, with the
    minimising weight and rank in ``attrs`` as ``best_smoothing`` /
    ``best_rank``. ``best_rank`` is the **slip-only** rank, comparable with every
    other rank here; see :func:`generalized_cross_validation` on why that is not
    the number GCV divides by.

    ⚠️ **Check that the minimum is interior before believing it.** A GCV
    curve that falls monotonically to the end of the sweep has not chosen
    anything, it has run out of weights -- ``attrs["interior"]`` says which
    happened. GCV is also known to be flat near its minimum, which is the
    mechanism behind the over- and under-regularisation its users report, so read
    the neighbouring rungs rather than the argmin alone.

    One factorization per weight, so this costs a resolution sweep -- seconds per
    rung at a few hundred parameters, minutes at a couple of thousand.
    """
    weights = [float(w) for w in np.atleast_1d(weights)]
    if len(weights) < 2:
        raise ValueError("a GCV curve needs at least two weights to compare")
    rows = [generalized_cross_validation(inversion, w, **kwargs) for w in weights]

    order = np.argsort(weights)
    weights = np.asarray(weights)[order]
    fields = {name: ("smoothing", np.array([rows[i][name] for i in order]))
              for name in ("gcv", "effective_rank", "effective_parameters",
                           "rss", "whitened_rms")}
    out = xr.Dataset(fields, coords={"smoothing": weights})
    best = int(np.argmin(out["gcv"].values))
    out.attrs.update(
        best_smoothing=float(weights[best]),
        best_rank=float(out["effective_rank"].values[best]),
        best_gcv=float(out["gcv"].values[best]),
        interior=bool(0 < best < len(weights) - 1),
    )
    return out

def resolution_report(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """Model and data resolution together, with the discretisation verdict.

    One factorization, three readings: the diagonal of ``R`` (how much of each
    estimate is data), the averaging kernels (what spatial scale each estimate
    actually represents) and ``diag(N)`` (which observations paid for it).

    ``over_discretisation`` is the median kernel width divided by the median
    element size.

    ⚠️ **Do NOT read it as "the mesh is this many times too fine". That reading
    was tried on the real data and it is wrong.** The kernel width is not a
    property of the data: it falls with the effective rank on *every* mesh
    (47.6 -> 26.1 km measured across four mesh resolutions on one fixed
    dataset), so the ratio moves when you change lambda without changing the
    mesh at all. Measured properly -- four meshes from 2.5 to 20 km compared at
    **matched effective rank** -- rms agreed to 0.3% and Mw to 0.005, and where
    they differed the *coarser* mesh was marginally better. The mesh sets only
    the ceiling on reachable rank (140 parameters cannot reach rank 100); within
    that ceiling it does not change the answer.

    So use this number as a description, not a verdict: a large value says the
    estimates overlap heavily, which is expected and not by itself a fault.

    ⚠️ **The diagonal and the kernel width answer different questions, and only
    the kernel width is a length.** Measured on a two-track synthetic across a
    100x sweep of lambda, the median diagonal fell 0.327 -> 0.034 (10x) while the
    along-strike kernel width moved 24.6 -> 21.7 km (12%) -- the diagonal is
    largely a statement about how hard the problem is damped, the kernel width
    about what the data geometry can separate. Note this does *not* mean a
    diagonal-driven size field drifts with lambda:
    :func:`target_element_size` normalises by the median resolution, which
    divides most of that out. Its actual failure is saturation -- see
    :func:`kernel_element_size`.

    Returns an :class:`xarray.Dataset` merging the diagonal and kernel fields on
    ``basis``, with ``importance`` on ``obs``.
    """
    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    matrix = cho_solve(chol, data)
    diagonal = np.diag(matrix).copy()
    kernels = resolution_kernels(inversion, matrix)

    from scipy.linalg import solve_triangular
    weighted = weighted_design(inversion)
    factor, lower = chol
    projected = solve_triangular(factor, weighted.T, lower=lower)
    importance = np.einsum("ij,ij->j", projected, projected)

    n = inversion.basis.n_basis
    out = kernels.copy()
    out["strike_slip"] = ("basis", diagonal[:n])
    out["dip_slip"] = ("basis", diagonal[n:2 * n])
    out["resolution"] = ("basis", np.maximum(diagonal[:n], diagonal[n:2 * n]))
    out["importance"] = ("obs", importance)

    element = element_size(inversion.mesh)
    width = np.maximum(kernels["width_along"].values, kernels["width_down"].values)
    trace_model = float(np.trace(matrix))
    trace_data = float(importance.sum())
    out.attrs.update(
        smoothing=_smoothing_label(smoothing), basis=inversion.basis.name,
        n_ramp=int(inversion.n_ramp),
        effective_rank=float(np.sum(diagonal[:2 * n])),
        trace_model_resolution=trace_model,
        trace_data_resolution=trace_data,
        # trace(A M^-1 A') == trace(M^-1 A'A) identically, so a mismatch means
        # the two halves were not built from the same factorization.
        trace_identity_error=float(abs(trace_model - trace_data)),
        median_kernel_width_m=float(np.median(width)),
        median_element_size_m=float(np.median(element)),
        over_discretisation=float(np.median(width) / max(np.median(element), 1e-9)),
        median_self_weight=float(np.median(kernels["self_weight"].values)),
        median_importance=float(np.median(importance)),
    )
    return out


#: Fraction of parameters allowed on a bound before
#: :func:`posterior_covariance` refuses. The linear covariance describes the
#: *unbounded* damped problem; a solution pressed against its bounds is a
#: different problem and the number would be a plausible-looking lie.
#:
#: ⚠️ This is a property of the **model**, not of the code, and it has moved a
#: long way. Measured 2026-09-05 over all 20 saved seven-scene models -- four
#: mesh configurations x five rank rungs -- it runs **5.8% to 18.6%**, rising
#: monotonically with rank (a looser model has more parameters free to reach a
#: bound) and with coarseness:
#:
#: ===========================  ======  ======  =======
#: configuration                rank 21 rank 79 rank 130
#: ===========================  ======  ======  =======
#: production, not deramped       5.8%    8.6%    10.4%
#: deramped, budget 330           5.8%    9.1%    10.7%
#: fixed rows, size_max 10 km     8.4%   11.2%    14.5%
#: grid unstructured, budget 200 10.2%   13.8%    18.1%
#: ===========================  ======  ======  =======
#:
#: ⚠️ **So this threshold sits inside the operating range, not above it.** The
#: production choice (rank 79, budget 330) is 8.6% and passes; the same mesh at
#: rank 130 is 10.4% and does **not**. That is the gate working as intended, but
#: it means a rank sweep will start refusing covariances part way up and the
#: refusal is about the model, not a bug.
#:
#: ⚠️ **The 86% this used to quote should not be carried forward.** It dates from
#: 2026-07 on the 1696-element mesh, before the budget was cut to ~330 -- and the
#: same number and month appear in :data:`slipsolve._geo.PHASE_RANGE_SIGN` as the
#: signature of the LOS **sign bug**, where a dextral fault recovered sinistral
#: with everything pinned on the polarity bound. Whether or not those were the
#: same run, 86% describes a pathological configuration and not this estimator.
MAX_BOUND_FRACTION = 0.10

#: Which covariance :func:`posterior_covariance` returns. Named apart from
#: :data:`slipsolve.covariance.COVARIANCE_KINDS`, which is the *noise*
#: model's shape and a different thing entirely.
POSTERIOR_KINDS = ("propagated", "posterior")


def bound_fraction(model, tolerance=1e-3):
    """Fraction of slip parameters sitting on one of their bounds.

    The gate on every linear uncertainty statement here -- see
    :data:`MAX_BOUND_FRACTION`.
    """
    options = getattr(model, "options", None) or {}
    strike = options.get("strike", (-10.0, 10.0))
    dip = options.get("dip", (-10.0, 10.0))
    polarity = options.get("polarity")
    if polarity:
        lo, hi = strike
        if polarity[0] < 0:
            hi = min(hi, 0.0)
        elif polarity[0] > 0:
            lo = max(lo, 0.0)
        strike = (lo, hi)

    on = 0
    for values, (lo, hi) in ((model.strike_slip, strike), (model.dip_slip, dip)):
        v = np.asarray(values, dtype=float)
        for edge in (lo, hi):
            if np.isfinite(edge):
                on += int(np.isclose(v, edge, atol=tolerance).sum())
    total = np.size(model.strike_slip) + np.size(model.dip_slip)
    return float(on) / max(total, 1)


def posterior_covariance(inversion, smoothing, kind="propagated", full=False,
                         model=None, noise=None, ss_ratio=1.0, ds_ratio=3.0,
                         boundary_ratio=1.0, sides=("bottom", "left", "right"),
                         smooth_weighting="uniform"):
    """Parameter covariance of the damped solve. Returns sigma, or the full matrix.

    Two readings, and **they differ by an order of magnitude here**, so the choice
    is not a detail:

    * ``"propagated"`` (default) -- ``M^-1 (A' Cd^-1 A) M^-1``: the estimator's
      variance under data noise. **This is the one that matches reality.**
      Validated against 60 repeated bounded solves on noise drawn from the same
      ``Cd``: empirical spread / analytic sigma = **1.01**, per-parameter
      correlation **0.950**.
    * ``"posterior"`` -- ``M^-1`` with ``M = A' Cd^-1 A + lam^2 L'L + W'W``. The
      Bayesian posterior *if the smoothing is taken as a prior distribution you
      believe*. On the same test it is **11x too wide** (ratio 0.09).

    They differ by exactly ``M^-1 (lam^2 L'L + W'W) M^-1``, the prior's own
    contribution, so ``posterior = propagated + prior term`` identically. The
    factor of eleven is not an error in either: repeated observation varies only
    the data, which is what ``"propagated"`` predicts, while ``M^-1`` also carries
    a spread that no amount of re-observing samples. Ask for ``"posterior"`` only
    if you mean the Bayesian question.

    ⚠️ **Summarise with the mean or the max, never the median.** At light damping
    the per-parameter sigma spans four orders of magnitude -- the unresolved
    parameters carry enormous variance while the bulk stays small -- so the median
    describes the bulk and hides the tail. Measured over lambda 0.01 to 1e4 on the
    synthetic: mean sigma falls monotonically **128 -> 0.03 m** as it should,
    while the median *rises* 0.059 -> 0.207 before falling, purely because the
    damping pulls the tail in faster than it lifts the bulk. Reading the median as
    "the uncertainty" would say more smoothing makes the model less certain.

    ⚠️ **That the prior term dominates by 11x is itself the result.** It is the
    same fact as an effective rank of 21 out of 748 parameters -- a typical
    parameter is **97% regulariser** -- seen from the covariance side. Quote either
    number alongside :func:`resolution_diagonal`, which says how much of each
    estimate the data actually paid for.

    ``noise`` gives the **sandwich for an estimator whose weighting is not
    ``Cd^-1/2``** -- which is the one this package ships. The production model is
    solved with the diagonal weights, because whitening measurably worsens the
    point estimate, so the formula above does not describe it: the solver's
    weighting ``w`` and the true noise ``Cd`` disagree. With ``noise`` set,

    .. code-block:: text

        C = M^-1 (A' w Cd w A) M^-1,    M = (wA)'(wA) + lam^2 L'L + B'B

    i.e. the solver's own normal matrix outside and the *true* noise in the
    middle. Pass ``noise={track: CovarianceModel}`` on a plain inversion and this
    is a seconds-long stand-in for an hours-long Monte Carlo. When the two
    weightings agree it reduces to ``"propagated"`` exactly.

    ⚠️ **One or the other is required.** Without ``noise``, and without a
    covariance on the inversion,
    :meth:`~slipsolve.inversion.SlipInversion.whiten` multiplies by
    ``1/sqrt(n)`` per track, which is not ``Cd^-1/2`` and carries no units -- the
    resulting "covariance" would be a number in arbitrary units that looks like
    a variance. Same rule as
    :attr:`~slipsolve.inversion.SlipModel.chi_square`.

    ⚠️ **Describes the UNBOUNDED problem.** Pass ``model=`` and it checks the
    fraction of parameters on a bound against :data:`MAX_BOUND_FRACTION` and
    raises rather than returning a number for a problem that was not solved.
    """
    if kind not in POSTERIOR_KINDS:
        raise ValueError(f"kind must be one of {POSTERIOR_KINDS}, got {kind!r}")
    if noise is None and getattr(inversion, "covariance", None) is None:
        raise ValueError(
            "posterior_covariance needs a noise model: without one the weights "
            "are 1/sqrt(n) per track, which is not Cd^-1/2, and the result would "
            "be in arbitrary units rather than metres squared. Either build the "
            "inversion with SlipInversion(..., covariance=...), or -- for a "
            "plainly-weighted solve, which is what this package ships -- pass "
            "noise={track: CovarianceModel} for the sandwich form."
        )
    if noise is not None and kind != "propagated":
        raise ValueError(
            "noise= gives the propagated sandwich; there is no Bayesian posterior "
            "for an estimator whose weighting is not the noise model's own"
        )
    if model is not None:
        fraction = bound_fraction(model)
        if fraction > MAX_BOUND_FRACTION:
            raise ValueError(
                f"{fraction:.1%} of parameters sit on a bound, above "
                f"MAX_BOUND_FRACTION={MAX_BOUND_FRACTION:.0%}. This covariance "
                "describes the unbounded damped problem, so on a solution pressed "
                "against its bounds it would be a plausible-looking lie. Loosen "
                "the bounds, or use a Monte Carlo through the bounded solve."
            )

    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    inverse = cho_solve(chol, np.eye(data.shape[0]))
    if noise is not None:
        middle = _noise_normal_matrix(inversion, noise)
    elif kind == "posterior":
        middle = None
    else:
        middle = data
    covariance = inverse if middle is None else inverse @ middle @ inverse
    if full:
        return covariance
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _noise_normal_matrix(inversion, noise):
    """``A' w^2 Cd w^2 A`` -- the sandwich's middle, with ``Cd = L L'`` per track.

    ⚠️ **``w`` appears TWICE on each side, not once.** The solver's normal
    equations are ``(wA)'(wA) m = (wA)'(w d)``, so the right-hand side carries
    ``w^2`` and its variance is ``A' w^2 Cd w^2 A``. Applying ``w`` once gives a
    covariance too large by ``1/w^2`` -- with ``w = 1/sqrt(n)`` and tracks of
    ~1000 samples that is a factor of a thousand in variance, **29x in sigma**,
    which is exactly what the first version of this returned against the Monte
    Carlo. The correlation was 0.978 throughout, which is what identified it as a
    scale error rather than a wrong shape.

    Formed as ``sum_t (L_t' w_t Aw_t)' (L_t' w_t Aw_t)`` so the dense ``Cd`` is
    never built: at production size one track's block is a few MB against ~900 MB
    for the whole thing.
    """
    from .covariance import whitening_factors

    design = weighted_design(inversion)
    weights = np.asarray(inversion.w, dtype=float)
    track = np.asarray(inversion.obs.ds["track"].values, dtype=object)
    middle = np.zeros((design.shape[1], design.shape[1]))
    for name, factor in whitening_factors(inversion.obs, noise).items():
        at = np.flatnonzero(track == name)
        projected = factor.T @ (weights[at, None] * design[at])
        middle += projected.T @ projected
    return middle


def resolution_bias(matrix, model):
    """``(R - I) @ m`` -- the smoothing's pull on *this* model, in metres.

    The other half of the error. A regularised estimate carries variance from the
    data noise (:func:`posterior_covariance`) **and** bias from the regulariser,
    and they trade off along the L-curve: more damping buys variance down and
    bias up. This is the bias term evaluated at the recovered model, which is the
    best stand-in available for the true one.

    Returns the pull per parameter **and** ``relative``, its norm over the model's
    own norm.

    ⚠️ **Across a lambda sweep, pass the SAME model every time.** Two traps sit
    here and both were hit before this warning existed:

    1. The **absolute** pull is taken at the recovered model, which shrinks as the
       damping rises -- so it shrinks too, and appears to say that more smoothing
       biases the answer *less*. Measured: mean pull 0.317 -> 0.098 m over lambda
       1 -> 100, exactly backwards. Use ``relative``.
    2. ``relative`` is not monotone either when the model moves with lambda,
       because ``m`` **rotates** toward the smooth directions ``R`` preserves
       best: measured 0.203 -> 0.431 -> 0.290 over the same sweep. The recovered
       model is already the smoothed one, so ``(R - I) m`` understates the pull
       away from the *truth*.

    So a lambda sweep must hold the reference model fixed -- the model at the
    corner, or the truth on a synthetic -- and vary only ``matrix``. Then
    ``relative`` rises monotonically toward 1, which is the statement that at
    heavy damping the estimate is entirely the regulariser's.

    ⚠️ **Do not add this in quadrature with the posterior sigma and call the
    result an error bar.** Both move when ``lambda`` moves, so they are not
    independent contributions to one total; the honest presentation is the
    trade-off across a ``lambda`` ladder, not a single number.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = np.size(model.strike_slip)
    slip = np.concatenate([np.asarray(model.strike_slip, dtype=float),
                           np.asarray(model.dip_slip, dtype=float)])
    if matrix.shape[0] < 2 * n:
        raise ValueError(
            f"the matrix is {matrix.shape} but the model has {2 * n} slip "
            "parameters; pass the resolution matrix this inversion produced")
    block = matrix[:2 * n, :2 * n]
    pulled = block @ slip - slip
    norm = float(np.linalg.norm(slip))
    return {
        "strike_slip": pulled[:n],
        "dip_slip": pulled[n:2 * n],
        "relative": float(np.linalg.norm(pulled) / norm) if norm > 0 else 0.0,
    }


def element_resolution(inversion, resolution):
    """Resolution per element, whatever basis it was computed in."""
    values = np.asarray(resolution["resolution"].values, dtype=float)
    if inversion.basis.name == "element":
        return values
    return inversion.basis.element_values(values)


def element_size(mesh):
    """The equivalent square edge of each element, ``sqrt(2 * area)``.

    The same length ``layered._orders`` uses to price a triangle against its
    distance to an observation, so "element size" means one thing in the package.
    """
    return np.sqrt(2.0 * np.asarray(mesh.areas, dtype=float))


def smooth_over_mesh(mesh, values, rounds=3):
    """Average a per-element field with its neighbours, ``rounds`` times.

    A target size field has to be Lipschitz or it asks for a jump no
    triangulation can honour without slivers; a few Jacobi sweeps over the
    element adjacency is the cheapest way to get there and needs no geometry
    beyond ``mesh.neighbors``.
    """
    values = np.asarray(values, dtype=float).copy()
    neighbors = mesh.neighbors
    for _ in range(int(rounds)):
        total = values.copy()
        count = np.ones(values.size)
        for k in range(neighbors.shape[1]):
            other = neighbors[:, k]
            has = other >= 0
            total[has] += values[other[has]]
            count[has] += 1.0
        values = total / count
    return values


def target_element_size(mesh, resolution, target=None, gain=0.5, max_ratio=3.0,
                        size_min=None, size_max=None, rounds=3):
    """Element sizes that would even out ``resolution`` across the mesh.

    ``h_new = h * (target / r) ** gain``. An element resolved better than
    ``target`` can afford to be smaller; one resolved worse is carrying
    parameters the data does not pay for and is grown. ``target`` defaults to the
    median, so the node budget is roughly conserved and the *distribution* is
    what moves -- pass a higher value to buy a coarser mesh outright.

    ``gain`` below 1 under-relaxes deliberately. Resolution is itself a function
    of the discretisation, so a full step chases its own tail; half a step
    converges in a few rounds instead of oscillating.

    ⚠️ **``max_ratio`` is not a nicety.** Below a few kilometres of slip depth,
    resolution does not merely get small, it underflows to *exactly* zero -- on
    the 330-element synthetic fixture at lambda 0.3 the mean below 14 km is 0.0000
    against 0.3304 above 6 km. ``target / r`` is then whatever the floor makes it,
    ``gain`` cannot temper an infinity, and one step asks for elements four times
    the mesh's own extent: measured, an unbounded first pass collapsed a 20 km
    fault to **three** depth levels with a 7.5 degree minimum angle. Bounding the
    *ratio* per step, and iterating, is what keeps a converging loop from taking
    one meaningless leap. ``size_min``/``size_max`` remain as absolute clamps on
    top.

    ⚠️ **This sets the size field's *shape*, not the mesh's size.** Targeting the
    median looks self-normalising and is not: the resolution distribution is
    heavily skewed -- most elements near zero, a few near one -- so most of them
    hit the growth cap while few can shrink, and the mean element size climbs
    every round. Measured on the two-scene synthetic fixture, a loop with no
    element budget coarsened 330 elements to 186, then 46, then 44, with variance
    reduction collapsing 94.3% -> 91.6% -> **0.74%** and peak slip falling to a
    millimetre: it ran end to end, converged, and returned nothing. The count is
    held by :func:`adapt_layout`'s ``budget``, which is the right division of
    labour anyway -- the caller says how many parameters they can afford, the
    resolution says where to put them.
    """
    resolution = np.asarray(resolution, dtype=float)
    if resolution.shape != (mesh.n_elements,):
        raise ValueError(
            f"resolution has {resolution.shape} entries for {mesh.n_elements} elements; "
            "pass element_resolution(inversion, ...) rather than the basis diagonal"
        )
    current = element_size(mesh)
    target = float(np.median(resolution)) if target is None else float(target)
    if target <= 0:
        raise ValueError("target resolution must be positive")
    max_ratio = float(max_ratio)
    if max_ratio < 1.0:
        raise ValueError("max_ratio is a growth factor and must be at least 1")

    floor = np.maximum(resolution, 1e-12)
    ratio = np.clip((target / floor) ** float(gain), 1.0 / max_ratio, max_ratio)
    # The *correction* is smoothed, not the absolute size, so a resolution field
    # asking for no change gets none: smoothing `current * ratio` would blur the
    # mesh's own slightly uneven element sizes into each other and drift.
    size = current * smooth_over_mesh(mesh, ratio, rounds=rounds)

    lo = current.min() / max_ratio if size_min is None else float(size_min)
    hi = current.max() * max_ratio if size_max is None else float(size_max)
    if lo > hi:
        raise ValueError(f"size_min={lo} is above size_max={hi}")
    return np.clip(size, lo, hi)


def size_field(mesh, size):
    """A per-element target size, read back as a continuous ``h(s, depth)``.

    The seam between "what resolution says each element should be" and "what a
    mesh generator needs", which is a function of position rather than a value
    per element. Scattered from element centroids in ``(s, z)`` and read back by
    **nearest neighbour**: this is a target for a mesh generator, not a function
    being integrated accurately, so a smooth interpolant would be false
    precision. Distances are in metres on both axes, so nearest is well posed
    without a scaling choice -- the parameter plane is already metric.

    Both :func:`adapt_layout` structures consume this: the row builder samples it
    along one vertical and then along each level, the unstructured builder hands
    it straight to :func:`~slipsolve.layout.lloyd_layout` as its density.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (mesh.n_elements,):
        raise ValueError(f"size has {size.shape} entries for {mesh.n_elements} elements")
    centre = mesh.element_params

    def at(s, depth):
        # Shape-agnostic: the row builder asks along a line, the unstructured one
        # asks on a 2-D probe grid, and both get their own shape back.
        s = np.atleast_1d(np.asarray(s, dtype=float))
        depth = np.broadcast_to(np.asarray(depth, dtype=float), s.shape)
        flat_s, flat_d = s.ravel(), depth.ravel()
        d2 = ((flat_s[:, None] - centre[None, :, 0]) ** 2
              + (-flat_d[:, None] - centre[None, :, 1]) ** 2)
        return size[np.argmin(d2, axis=1)].reshape(s.shape)

    return at


#: How element size is read off an anisotropic averaging kernel. The kernels here
#: are far wider along strike than down dip (26-41 km against 10-16 km on the
#: Venezuela data), and an element carries **one** size, so the two directions
#: have to be reconciled. ``"min"`` keeps the better-resolved direction, which
#: costs redundancy along strike but never discards structure the data can see;
#: ``"max"`` does the opposite and is the one that loses information.
#: How :func:`kernel_element_size` turns an anisotropic kernel into one number.
#: ``"gaussian"`` is the odd one out and the only one that is Barnhart & Lohman's
#: own metric: it is already a single isotropic width, so nothing is reduced.
KERNEL_REDUCTIONS = ("min", "geometric", "max", "gaussian")


def kernel_element_size(inversion, smoothing, fraction=0.5, reduce="min",
                        size_min=None, size_max=None, rounds=3, report=None,
                        **kwargs):
    """Target element size from the averaging-kernel **width**, not the diagonal.

    ``h = fraction * kernel_width``. Two elements per resolvable feature at the
    default ``fraction=0.5``, which is the sampling analogue of Nyquist: finer
    than that and neighbouring elements are several names for one number.

    **Why this rather than** :func:`target_element_size`. That one forms
    ``(target / r) ** gain`` on the *diagonal* of ``R``, and on real data ``r``
    has underflowed over most of the fault -- so it is a ratio of two nearly-zero
    numbers and simply runs to its ``max_ratio`` bound. Measured on a
    132-element synthetic across a 1000x sweep of lambda, **42-75% of elements sit
    pinned at the clamp** (median ``r`` 0.0000, min 2e-8): more than half the size
    field is the clamp rather than the data. The kernel width over the same sweep
    stays a real length everywhere, 4-11 km, varying smoothly.

    ⚠️ It is **not** that the diagonal criterion chases lambda -- an earlier note
    here said so and it was wrong. ``target`` defaults to the *median* resolution,
    which divides out most of the lambda dependence: over that same sweep the
    diagonal-driven size field varied 13% and the kernel-driven one 18%, so if
    anything the diagonal is the steadier of the two. The defect is degeneracy,
    not drift.

    ``reduce`` picks how the anisotropic kernel becomes one number; see
    :data:`KERNEL_REDUCTIONS`. ``report`` reuses a
    :func:`resolution_report` you already have, which is the expensive part.

    ⚠️ **The moment reductions are NOT Barnhart & Lohman's metric, despite being
    the kernel-width option.** They fit a *Gaussian* to row ``k`` of ``R`` against
    3-D distance and take its 1-sigma; :func:`resolution_kernels`' widths are
    *second moments*, which weight by distance squared. The averaging kernel here
    is strongly non-Gaussian -- measured on the real six-scene model, **19.4% of a
    row's ``|R|`` mass sits beyond 3 sigma of the Gaussian fit against the 0.27% a
    Gaussian would put there**, a 70x excess -- so the second moment reads the
    tails and comes out **3.09x** their sigma along strike (39.3 km against 11.5).
    Trimming the weakest 10% of each row brings the ratio to 1.05, which
    identifies the tails as the whole of the difference. Down dip the medians
    nearly agree (1.21x) but the per-parameter correlation is only **0.50**.

    ⚠️ And ``reduce="min"`` selects the down-dip width on **100%** of parameters
    on that model, so the along-strike number never reaches the size field at all.
    The anisotropy is real (2.67x median) but it is not being *reconciled* -- one
    direction is simply always chosen.

    Consequence, measured: a four-round loop sized from ``reduce="min"`` lost to
    one sized from the resolution diagonal at every effective rank, and to a
    *fixed* mesh at low rank. That was evidence against **that estimator**, not
    against Barnhart & Lohman's method.

    ``reduce="gaussian"`` is now their method --
    :func:`gaussian_kernel_width`, fitted rather than linearised -- and it is the
    arm that has not been run against the others on real data. It falls back to
    ``"min"`` on any parameter whose row fits no width.
    """
    if reduce not in KERNEL_REDUCTIONS:
        raise ValueError(
            f"reduce must be one of {KERNEL_REDUCTIONS}, got {reduce!r}")
    if report is None:
        report = resolution_report(inversion, smoothing, **kwargs)

    along = np.asarray(report["width_along"].values, dtype=float)
    down = np.asarray(report["width_down"].values, dtype=float)
    if reduce == "gaussian":
        # Already isotropic -- B&L fit one sigma against 3-D distance -- so there
        # is nothing to reconcile. A row that fitted no width falls back to the
        # moment reduction the other options use, rather than to a hole in the
        # size field.
        width = np.asarray(report["width_gaussian"].values, dtype=float)
        width = np.where(np.isfinite(width), width, np.minimum(along, down))
    elif reduce == "min":
        width = np.minimum(along, down)
    elif reduce == "max":
        width = np.maximum(along, down)
    else:
        width = np.sqrt(along * down)

    if inversion.basis.name != "element":
        width = inversion.basis.element_values(width)
    size = float(fraction) * width
    size = smooth_over_mesh(inversion.mesh, size, rounds=rounds)

    current = element_size(inversion.mesh)
    lo = current.min() / 3.0 if size_min is None else float(size_min)
    hi = current.max() * 3.0 if size_max is None else float(size_max)
    if lo > hi:
        raise ValueError(f"size_min={lo} is above size_max={hi}")
    return np.clip(size, lo, hi)


def _stations_for_density(extent, density, samples=1024):
    """Positions across ``[0, extent]`` at a varying target spacing.

    Integrating ``1/h`` and then splitting the total into a whole number of
    equal steps is what lands on both ends *exactly*, which is what
    :func:`~slipsolve.layout.boundary_node_mask` needs; marching by the
    local spacing would leave the far end a ragged fraction short.
    """
    grid = np.linspace(0.0, float(extent), int(samples))
    inverse = 1.0 / np.maximum(density(grid), 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (inverse[1:] + inverse[:-1]) * np.diff(grid))])
    n = max(1, int(round(cumulative[-1])))
    out = np.interp(np.linspace(0.0, cumulative[-1], n + 1), cumulative, grid)
    out[0], out[-1] = 0.0, float(extent)
    return out


def adapt_layout(mesh, size, budget="conserve", top_depth=None, max_depth=None,
                 length=None, balance=MAX_COUNT_RATIO, samples=1024, tolerance=0.05,
                 structure="rows", iterations=25, seed=0):
    """A new layout honouring a per-element target size.

    ``structure`` decides how much freedom the node set has:

    * ``"rows"`` (default) -- depth levels, each with its own node count and its
      own along-strike positions. What every run before 2026-08-27 produced.
    * ``"unstructured"`` -- no levels at all: the size field is handed to
      :func:`~slipsolve.layout.lloyd_layout` as a density and the nodes
      relax to it in two dimensions.

    ⚠️ **The row builder reads the depth levels off ONE vertical, at the fault's
    midpoint** (``s = length/2``), because a level is a single depth for the
    whole fault and so cannot be in two places at once. Along-strike node
    *density* does vary per level, but the layering does not: a resolution field
    wanting fine levels at one end and coarse ones at the other has no way to say
    so. ``"unstructured"`` is that constraint removed, and it is the reason it
    exists. The *outline* is a rectangle in ``(s, z)`` either way -- freeform
    means the discretisation, not the fault's extent.

    ``iterations`` and ``seed`` reach the relaxation and matter only for
    ``"unstructured"``. Both are fixed by default, so a layout is reproducible
    and ``FaultMesh.digest()`` is stable across runs.

    ``budget`` is the element count to land on: ``"conserve"`` keeps the mesh's
    own, an integer asks for that many, ``None`` takes whatever ``size`` implies.
    It is found by scaling the whole size field and rebuilding -- a handful of
    bisection steps -- rather than from a tiling formula, because neither
    builder's count follows one: the row builder integrates ``1/h`` along each
    axis *separately*, so a closed form is wrong by a factor that depends on how
    anisotropic the size field is, and the unstructured builder's count comes out
    of a relaxation. Each rebuild is cheap either way (0.02 s at 240 elements).

    ⚠️ **Without a budget an adaptive loop does not converge, it drifts**; see
    :func:`target_element_size` for the measurement.

    Under ``"rows"``, depth levels come from integrating ``1/h`` down dip and
    each level's nodes from integrating ``1/h`` along strike *at that depth* --
    so both the count **and** the positions vary per level, which is what
    "vary the number of nodes at the depth levels" costs once the size field is
    not separable.

    The size field is read back through :func:`size_field`. Note ``h`` is
    honoured as a node *spacing* while :func:`element_size` measures
    ``sqrt(2 * area)``, about 7% smaller for a well-shaped triangle; the budget
    absorbs that constant, and it is the same on both paths, so nothing between
    them is compared unfairly.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (mesh.n_elements,):
        raise ValueError(f"size has {size.shape} entries for {mesh.n_elements} elements")
    if structure not in ("rows", "unstructured"):
        raise ValueError(
            f"structure must be 'rows' or 'unstructured', got {structure!r}"
        )
    if budget is not None:
        wanted = mesh.n_elements if budget == "conserve" else int(budget)
        if wanted < 1:
            raise ValueError("budget must be at least one element")
        return _fit_budget(
            lambda scale: adapt_layout(mesh, size * scale, budget=None,
                                       top_depth=top_depth, max_depth=max_depth,
                                       length=length, balance=balance, samples=samples,
                                       structure=structure, iterations=iterations,
                                       seed=seed),
            wanted, float(tolerance))

    params = mesh.params
    length = float(params[:, 0].max()) if length is None else float(length)
    top_depth = float(-params[:, 1].max()) if top_depth is None else float(top_depth)
    max_depth = float(-params[:, 1].min()) if max_depth is None else float(max_depth)
    at = size_field(mesh, size)

    if structure == "unstructured":
        layout = lloyd_layout(length, max_depth, at, top_depth=top_depth,
                              iterations=iterations, seed=seed)
        layout.attrs["layout"] = "adapted_unstructured"
        layout.attrs["quality"] = layout.quality()
        _warn_if_slivered(layout)
        return layout

    middle = 0.5 * length
    depths = top_depth + _stations_for_density(
        max_depth - top_depth,
        lambda d: at(np.full_like(d, middle), top_depth + d),
        samples=samples,
    )
    depths = _balance_levels(depths, balance)

    rows, counts = [], []
    for depth in depths:
        stations = _stations_for_density(
            length, lambda s, d=depth: at(s, np.full_like(s, d)), samples=samples)
        rows.append(stations)
        counts.append(stations.size)

    counts = _rebalance(np.asarray(counts, dtype=int), balance)
    params = np.vstack([
        np.column_stack([_resample_stations(row, int(n)),
                         np.full(int(n), -float(depth))])
        for row, n, depth in zip(rows, counts, depths)
    ])

    layout = MeshLayout(params, triangulate(params),
                        levels=depths, counts=counts,
                        attrs={"layout": "adapted"})
    require_boundaries(layout.params)
    return layout


#: Minimum triangle angle, in degrees, below which an unstructured layout is
#: flagged. Chosen from measurement, not taste: the density-weighted relaxation
#: produces 21-27 degrees on every size field tried, and the *uniform* production
#: mesh already sits at 11.3 degrees (5 km along strike against ~1 km top
#: levels), so this is below both and fires only on something genuinely
#: degenerate. Relaxation is what buys the quality -- the unrelaxed cloud
#: measures 0.4 degrees.
MIN_ANGLE_WARN_DEG = 10.0


def _warn_if_slivered(layout):
    """Flag an unstructured layout whose triangulation has gone degenerate.

    Worth a warning rather than silence because the adaptive loop runs
    unattended for several rounds: a sliver degrades the TDE Green's functions
    and ``basis._barycentric_weight`` divides by a quantity that vanishes on one.
    A warning rather than a raise because the mesh is still usable and the caller
    may have asked for something extreme on purpose.
    """
    angle = float(layout.attrs["quality"]["min_angle_deg"])
    if angle < MIN_ANGLE_WARN_DEG:
        warnings.warn(
            f"The adapted unstructured layout has a minimum triangle angle of "
            f"{angle:.1f} degrees, below {MIN_ANGLE_WARN_DEG:.0f}. Raise "
            "iterations=, or bound the size field harder with max_ratio= / "
            "size_min= / size_max= on target_element_size.",
            RuntimeWarning,
            stacklevel=3,
        )


def _fit_budget(build, wanted, tolerance):
    """Scale the size field until the built layout holds about ``wanted`` elements.

    Monotone in the scale -- bigger elements, fewer of them -- so a geometric
    bracket followed by bisection in log-scale converges in a handful of builds,
    and each build is cheap enough to do this properly rather than estimate it.
    """
    layout = build(1.0)
    if abs(layout.n_elements - wanted) <= tolerance * wanted:
        return layout

    lo = hi = 1.0
    if layout.n_elements > wanted:
        for _ in range(12):                     # too many elements -> coarsen
            lo, hi = hi, hi * 1.5
            layout = build(hi)
            if layout.n_elements <= wanted:
                break
    else:
        for _ in range(12):                     # too few -> refine
            lo, hi = lo / 1.5, lo
            layout = build(lo)
            if layout.n_elements >= wanted:
                break

    best = layout
    for _ in range(20):
        mid = float(np.sqrt(lo * hi))
        layout = build(mid)
        if abs(layout.n_elements - wanted) < abs(best.n_elements - wanted):
            best = layout
        if abs(layout.n_elements - wanted) <= tolerance * wanted:
            return layout
        if layout.n_elements > wanted:
            lo = mid
        else:
            hi = mid
    return best


def _balance_levels(depths, balance):
    """Limit how fast the level thickness may change from one level to the next.

    The counts are balanced across strike by
    :func:`~slipsolve.layout._balance_counts`; this is the same rule down
    dip, and it is needed for the same reason. A size field that saturates at
    depth otherwise puts one 17 km-tall element under a 2.5 km one, which is a
    7-to-1 aspect ratio however many nodes the row has.
    """
    thickness = np.diff(np.asarray(depths, dtype=float))
    if thickness.size < 2:
        return np.asarray(depths, dtype=float)
    for _ in range(thickness.size):
        before = thickness.copy()
        thickness[1:] = np.minimum(thickness[1:], thickness[:-1] * balance)
        thickness[:-1] = np.minimum(thickness[:-1], thickness[1:] * balance)
        if np.allclose(thickness, before):
            break
    # Renormalise so the levels still span exactly the same depth range; the
    # bottom must stay at max_depth or the mesh quietly changes extent.
    thickness *= (depths[-1] - depths[0]) / thickness.sum()
    return depths[0] + np.concatenate([[0.0], np.cumsum(thickness)])


def _resample_stations(stations, n):
    """Change a row's node count while keeping the shape of its spacing.

    Resampled uniformly *in index*, so a row that was dense at one end stays
    dense there, and both ends land on their exact original values -- which is
    what keeps the boundary test exact after ``_balance_counts`` has raised a
    count.
    """
    if stations.size == n:
        return stations
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, stations.size), stations)


def _rebalance(counts, balance):
    from .layout import _balance_counts

    # Quiet: these counts came from integrating a size field, not from a caller
    # asking for them, so refining one is the algorithm rather than an override
    # -- and the budget search rebuilds a dozen times.
    return _balance_counts(counts, balance, warn=False)


def resolution_gain(model, los_stack, resolution, frame=None, spacing=GAIN_SPACING,
                    exponent=1.0):
    """Where another observation would most improve model resolution.

    Lohman & Simons 2005's criterion, evaluated as a field rather than by
    recursing on candidate cells: a sample at ``(x, y)`` helps in proportion to
    how strongly it sees the parameters that are *currently unresolved*,

    ``w(x, y) = sum_k G(x, y; k)**2 * (1 - r_k)``.

    Returns an ``(ny, nx)`` array on ``los_stack``'s own grid, normalised to unit
    standard deviation over its finite samples, ready to hand to
    :meth:`~slipsolve.sampling.Observations.from_los` as ``field=``.

    Evaluated on a coarse grid (:data:`GAIN_SPACING`, coarser than the predicted
    field's) and interpolated up, because the cost is a full Green's-function
    assembly at every probe point. Recursing instead -- testing
    each candidate cell against ``R`` as the paper describes -- needs a
    Green's-function row per candidate, thousands of them, against a quadtree
    that currently costs 0.05 s on 0.32 M pixels.
    """
    from scipy.interpolate import RegularGridInterpolator

    from .sampling import _to_frame, stack_epsg

    spacing = GAIN_SPACING if spacing is None else float(spacing)
    frame = frame or model.mesh.frame
    if frame is None:
        raise ValueError("resolution_gain needs a LocalFrame; the mesh carries none")

    mesh = model.mesh
    unresolved = 1.0 - np.clip(
        np.asarray(resolution["resolution"].values, dtype=float), 0.0, 1.0)

    ds = los_stack.ds
    epsg = stack_epsg(ds)
    x_native = np.asarray(ds["x"].values, dtype=float)
    y_native = np.asarray(ds["y"].values, dtype=float)
    gx, gy = np.meshgrid(x_native, y_native)
    fx, fy = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)

    cx = np.arange(fx.min(), fx.max() + spacing, spacing)
    cy = np.arange(fy.min(), fy.max() + spacing, spacing)
    px, py = np.meshgrid(cx, cy)

    # The scene's own mean look vector, renormalised. One direction for the whole
    # coarse grid is an approximation -- a wide swath runs 25 to 49 degrees of
    # incidence -- but the gain only has to *rank* candidate locations, and using
    # the mean beats a nominal vertical, which would misweight the east-west
    # sensitivity that a strike-slip fault's signal mostly lives in.
    look = np.array([np.nanmean(np.asarray(ds[f"los_{c}"].values, dtype=float))
                     for c in ("east", "north", "up")])
    norm = np.linalg.norm(look)
    look = look / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    ones = np.ones(px.size)
    engine = model.inversion.engine if model.inversion is not None else None
    if engine is None:
        raise ValueError(
            "resolution_gain needs the model's engine, which SlipModel.load "
            "deliberately does not restore; rebuild the SlipInversion first"
        )
    green = engine.los_matrix(mesh, px.ravel(), py.ravel(),
                              look[0] * ones, look[1] * ones, look[2] * ones)

    n = model.basis.n_basis
    if green.shape[1] != 2 * n:
        projection = model.basis.projection()
        green = np.hstack([green[:, :mesh.n_elements] @ projection,
                           green[:, mesh.n_elements:] @ projection])
    weight = np.concatenate([unresolved, unresolved]) ** float(exponent)
    gain = (green ** 2) @ weight

    interp = RegularGridInterpolator(
        (cy, cx), gain.reshape(px.shape), bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([fy, fx])).reshape(gx.shape)

    finite = np.isfinite(out)
    if not finite.any():
        raise ValueError("the resolution-gain field is empty over this scene")
    scale = out[finite].std()
    if scale == 0:
        warnings.warn(
            "the resolution-gain field is constant, so it cannot steer a "
            "quadtree; every parameter is equally (un)resolved from this scene.",
            RuntimeWarning, stacklevel=2,
        )
        return out
    return (out - out[finite].mean()) / scale
