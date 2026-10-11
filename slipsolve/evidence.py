"""How strongly the data support a fault geometry, with slip and ramp integrated out.

A geometry search compares fault surfaces that the same data can fit with
different slip. Misfit at a fixed effective rank does that only once a rank has
been chosen, and the rank is not neutral. On the 2026-10-11 synthetic (Mw 7.7,
data on both sides, 4 km mesh) rms at rank 60 placed a 60 degree fault at 57-58
even with an exact forward model, because the data supported rank ~190.

This module scores a geometry by its marginal likelihood instead (Yabuki &
Matsu'ura 1992; Fukahata & Wright 2008). For a design matrix ``G`` (the
geometry), ramp columns ``R`` and data covariance ``C``::

    d = G m + R b + e,               e ~ N(0, sigma^2 C)
    m ~ N(0, sigma^2 (alpha^2 Q)^-1)     smoothing prior, pinned edges removed
    b flat,  sigma^2 Jeffreys,  alpha^2 log-uniform on a fixed grid

and everything except the geometry is integrated out. Let ``P`` remove the ramp
in the ``C^-1`` metric, ``K = G'PG`` and ``mu`` the generalised eigenvalues of
``(K, Q)``. Then::

    log p(d | G, alpha^2) = (p/2) log alpha^2 - 1/2 sum log(mu + alpha^2)
                            - ((n - k)/2) log S(alpha^2)  + const
    S(alpha^2) = d'Pd - sum z^2 / (mu + alpha^2)

One eigendecomposition per geometry makes every ``alpha^2`` O(p). Because ``K``
does not depend on the data, every further data vector costs O(np), which is
what makes :meth:`~slipsolve.sweep.DipSweep.bootstrap` affordable. The formula is
pinned against the dense marginal likelihood in the tests.

⚠️ **Rank geometries with the evidence; do not read it as a posterior.**
Measured over 100 noise draws on the synthetic (``geometry_calibration/``):

- **Noise taken as iid** (``covariance=None``): the best dip was within 0.34
  degrees of the truth on every truth and mesh tried. But the posterior it
  implies was 4-7x too narrow; 95% intervals covered the truth 18-41% of the
  time.
- **Whitened by the true covariance:** honest widths, but only with an exact
  forward model (95% coverage). A 4 km mesh against a 2 km truth moved the dip
  +2.2 degrees (6.5 posterior sd, 0% coverage); an 8 km mesh, +12.6.
  Whitening up-weights the short wavelengths, which is where discretisation
  error lives.

So take the estimate from the iid evidence and its uncertainty from a parametric
bootstrap (:meth:`DipSweep.bootstrap`), which covered 82-97% on every case. A
disagreement between the whitened and the iid estimates means the forward model
is not good enough to weight by ``C``.

The evidence knows nothing of slip bounds or polarity: the prior is Gaussian on
every free parameter. The zero-slip edges are removed outright rather than
penalised, so ``boundary_ratio`` plays no part here.
"""

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq
from scipy.special import logsumexp

from .covariance import CovarianceModel, observation_covariance
from .regularize import laplace_beltrami

#: The ``alpha^2`` grid, in decades relative to the largest generalised
#: eigenvalue of the first geometry scored, and its step in decades. The grid is
#: fixed on first use, so every geometry an evidence object scores is integrated
#: over the same one. The integrand decays at both ends well inside these limits
#: on every problem measured; ``alpha_edge`` in the scores says when it does not.
ALPHA_DECADES = (-14.0, 2.0)
ALPHA_STEP = 0.05


def pinned_nodes(mesh, sides):
    """Boolean mask of the nodes on the zero-slip ``sides``."""
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    for side in sides:
        mask |= np.asarray(mesh.boundary_nodes(side), dtype=bool)
    return mask


def smoothing_precision(mesh, ss_ratio=1.0, ds_ratio=1.0,
                        sides=("bottom", "left", "right"), operator=None):
    """The prior precision ``Q`` on the free nodal parameters, and their indices.

    ``Q = L'L`` for the Laplace-Beltrami operator on ``mesh`` (or ``operator``,
    one already built), restricted to the nodes off the zero-slip ``sides``.
    Those nodes are held at zero, so their rows still penalise their neighbours
    and ``Q`` is positive definite. The indices point into the ``2n`` nodal
    vector, strike-slip first.
    """
    op = laplace_beltrami(mesh, ss_ratio, ds_ratio) if operator is None else operator
    keep = np.flatnonzero(~pinned_nodes(mesh, sides))
    free = np.concatenate([keep, mesh.n_nodes + keep])
    sub = op.tocsc()[:, free]
    return np.asarray((sub.T @ sub).todense()), free


class TrackWhitener:
    """``L^-1`` for a block-diagonal ``C = L L'`` with one block per track.

    A block is ``None`` (identity), a 1-D array of variances, or a covariance
    matrix. Tracks are separate acquisitions, so their noise is independent and
    ``C`` is block diagonal by construction.
    """

    def __init__(self, rows, blocks):
        self.rows = [np.asarray(r) for r in rows]
        self._ops = []
        for block in blocks:
            if block is None:
                self._ops.append(None)
                continue
            block = np.asarray(block, dtype=float)
            if block.ndim == 1:
                self._ops.append(("scale", 1.0 / np.sqrt(block)))
            else:
                self._ops.append(("chol", sla.cholesky(0.5 * (block + block.T), lower=True)))

    def __call__(self, a):
        a = np.asarray(a, dtype=float)
        out = np.array(a, dtype=float, copy=True)
        for rows, op in zip(self.rows, self._ops):
            if op is None:
                continue
            if op[0] == "scale":
                out[rows] = a[rows] * (op[1] if a.ndim == 1 else op[1][:, None])
            else:
                out[rows] = sla.solve_triangular(op[1], a[rows], lower=True)
        return out


def track_blocks(obs, covariance):
    """Per-track noise blocks for :class:`TrackWhitener`, in track order.

    ``covariance`` is ``None`` (iid), ``"weights"`` (variance ``1/w^2`` from the
    observations' own weights), one :class:`CovarianceModel` for every track,
    or a ``{track: CovarianceModel or ndarray}`` mapping.
    """
    ds = obs.ds
    track = np.asarray(ds["track"].values).astype(str)
    names = list(dict.fromkeys(track))
    rows = [np.flatnonzero(track == t) for t in names]
    if covariance is None:
        return names, rows, [None] * len(names)
    if isinstance(covariance, str):
        if covariance != "weights":
            raise ValueError(f"covariance must be None, 'weights', a CovarianceModel or a "
                             f"mapping of them; got {covariance!r}")
        w = np.asarray(ds["weight"].values, dtype=float)
        return names, rows, [1.0 / w[r] ** 2 for r in rows]
    if isinstance(covariance, CovarianceModel):
        covariance = {t: covariance for t in names}
    missing = [t for t in names if t not in covariance]
    if missing:
        raise ValueError(f"no covariance for track(s) {missing}; tracks are {names}")
    blocks = []
    for t in names:
        c = covariance[t]
        blocks.append(observation_covariance(obs, c, track=t)
                      if isinstance(c, CovarianceModel) else np.asarray(c, dtype=float))
    return names, rows, blocks


class GeometryEvidence:
    """The marginal likelihood of fault geometries against one set of samples.

    ``q`` and ``free`` are the prior (see :func:`smoothing_precision`); ``ramp``
    the nuisance columns (``ramp_columns``'s matrix); ``covariance`` the noise,
    as :func:`track_blocks` takes it. Geometries enter as nodal design matrices
    over the same samples, so the ``2n`` columns must be in the same order for
    every one.

    Use: ``dh, s0 = ev.prepare(data)`` once per data set; then for each geometry
    ``dec = ev.decompose(g)`` and ``ev.score(dec, dh, s0)``. ``decompose`` does
    not look at the data, so it can be reused across any number of data vectors.
    """

    def __init__(self, obs, q, free, ramp, covariance=None,
                 decades=ALPHA_DECADES, step=ALPHA_STEP):
        self.tracks, rows, blocks = track_blocks(obs, covariance)
        self.n = int(obs.ds.sizes["obs"])
        self.whiten = TrackWhitener(rows, blocks)
        ramp = np.asarray(ramp, dtype=float).reshape(self.n, -1)
        if ramp.shape[1]:
            self.ur, self.tr = np.linalg.qr(self.whiten(ramp))
        else:
            self.ur, self.tr = np.zeros((self.n, 0)), np.zeros((0, 0))
        self.k = self.ur.shape[1]
        self.free = np.asarray(free)
        self.p = int(self.free.size)
        if self.n - self.k <= 0:
            raise ValueError("fewer samples than ramp columns")
        q = np.asarray(q, dtype=float)
        uq = sla.cholesky(0.5 * (q + q.T), lower=False)
        self.uq_inv = sla.solve_triangular(uq, np.eye(q.shape[0]), lower=False)
        self.decades, self.step = tuple(decades), float(step)
        self.alpha2 = None
        self.dlog = self.step * np.log(10.0)

    def project(self, a):
        """Remove the ramp's column space, in the whitened metric."""
        return a - self.ur @ (self.ur.T @ a) if self.k else a

    def prepare(self, data):
        """Whitened, ramp-free data ``(n, m)`` and their squared norms ``(m,)``."""
        d = np.asarray(data, dtype=float).reshape(self.n, -1)
        dh = self.project(self.whiten(d))
        return dh, np.einsum("ij,ij->j", dh, dh)

    def decompose(self, g):
        """The data-independent part for one nodal design matrix ``g`` ``(n, 2N)``."""
        gh = self.project(self.whiten(np.asarray(g, dtype=float)[:, self.free]))
        b = self.uq_inv.T @ (gh.T @ gh) @ self.uq_inv
        mu, v = np.linalg.eigh(0.5 * (b + b.T))
        mu = np.clip(mu, 0.0, None)
        if self.alpha2 is None:
            exps = np.arange(self.decades[0], self.decades[1] + 1e-9, self.step)
            self.alpha2 = max(float(mu.max()), np.finfo(float).tiny) * 10.0 ** exps
        return {"mu": mu, "v": v, "gh": gh}

    def _z(self, dec, dh):
        return dec["v"].T @ (self.uq_inv.T @ (dec["gh"].T @ dh))

    def score(self, dec, dh, s0, rank=None):
        """Per data column: the log evidence and what it rests on.

        ``log_evidence`` integrates ``alpha^2`` with ``sigma^2`` integrated too;
        ``log_evidence_fixed`` takes ``sigma^2 = 1``, i.e. trusts the covariance's
        scale. ``alpha2``, ``rank`` and ``sigma2`` are at the integrand's mode;
        ``alpha_edge`` flags a mode on the end of the grid. With ``rank`` given,
        ``rms_at_rank`` is the whitened rms of the solution at that effective
        rank.
        """
        mu, n = dec["mu"], dh.shape[0]
        z2 = self._z(dec, dh) ** 2
        a2 = self.alpha2[:, None]
        inv = 1.0 / (mu[None, :] + a2)
        s = np.maximum(s0[None, :] - inv @ z2, np.finfo(float).tiny)
        common = 0.5 * self.p * np.log(self.alpha2) - 0.5 * np.sum(np.log(mu[None, :] + a2), axis=1)
        l_int = common[:, None] - 0.5 * (n - self.k) * np.log(s)
        l_fix = common[:, None] - 0.5 * s
        mode = np.argmax(l_int, axis=0)
        cols = np.arange(s.shape[1])
        out = {"log_evidence": logsumexp(l_int, axis=0) + np.log(self.dlog),
               "log_evidence_fixed": logsumexp(l_fix, axis=0) + np.log(self.dlog),
               "alpha2": self.alpha2[mode],
               "rank": np.sum(mu[None, :] / (mu[None, :] + self.alpha2[mode][:, None]), axis=1),
               "sigma2": s[mode, cols] / (n - self.k),
               "alpha_edge": (mode == 0) | (mode == self.alpha2.size - 1)}
        if rank is not None:
            top = np.log(max(float(mu.max()), np.finfo(float).tiny))
            u = brentq(lambda t: np.sum(mu / (mu + np.exp(t))) - float(rank), top - 60.0, top + 10.0)
            a = np.exp(u)
            resid = s0 - np.sum(z2 * ((mu + 2.0 * a) / (mu + a) ** 2)[:, None], axis=0)
            out["rms_at_rank"] = np.sqrt(np.maximum(resid, 0.0) / n)
        return out

    def posterior_mean(self, dec, dh, alpha2):
        """Free-parameter posterior mean ``(p, m)``, each column at its own ``alpha2``."""
        y = self._z(dec, dh) / (dec["mu"][:, None] + np.asarray(alpha2, dtype=float)[None, :])
        return self.uq_inv @ (dec["v"] @ y)

    def ramp_coefficients(self, residual):
        """The ramp that best fits ``residual`` ``(n, m)`` in the same metric."""
        if not self.k:
            return np.zeros((0, np.asarray(residual).reshape(self.n, -1).shape[1]))
        rt = self.whiten(np.asarray(residual, dtype=float).reshape(self.n, -1))
        return sla.solve_triangular(self.tr, self.ur.T @ rt, lower=False)
