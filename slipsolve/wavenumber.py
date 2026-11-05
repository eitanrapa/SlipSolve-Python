"""Static Green's functions of a layered half-space, by wavenumber integration.

:mod:`slipsolve.edgrn` reads what Rongjiang Wang's EDGRN computes. This module
computes the same kind of table itself, because the exact bimaterial solve
(:mod:`slipsolve.bimaterial`) needs three things EDGRN does not give:

* **Receivers at depth.** An EDGRN run tabulates one receiver depth, and ours are at
  the surface. The interface solve needs the field on the fault itself, at every
  collocation depth.
* **An explosion source.** EDGRN's three sources (strike-slip, dip-slip, CLVD) span
  the shear dislocations only. An opening has an isotropic moment tensor.
* **Accuracy next to the source.** Every element needs its own field at its own
  centroid. The fine EDGRN tables already read 11% low 300 m from the trace and
  75% low at 60 m.

**What is tabulated is a remainder.** ``R = layered - half-space``, where the
half-space has the elastic constants of the source's own layer and the same free
surface. The two share the singularity at the source, so ``R`` is smooth there,
and the half-space part is added back exactly by :mod:`slipsolve._tde` (see
:mod:`slipsolve.layered_tde`). The subtraction happens in the wavenumber domain,
before the Hankel transform, so the remainder never passes through a difference of
two large numbers in space.

⚠️ **Smooth except next to a layer interface.** A source near an interface has an
image across it, so ``R`` grows like ``1 / (r**2 + (z_s + z_r - 2 z_I)**2)`` as source
and receiver approach the interface together. The depth grid is graded toward
every interface for that reason (:func:`depth_grid`), and lookups never
interpolate across one.

The formulation. Depth ``z`` is positive down and the horizontal axes are EDGRN's
(``x`` north, ``y`` east, azimuth from north toward east). For azimuthal order
``m`` and wavenumber ``k``, with ``Y = J_m(kr) e^{i m phi}``::

    u_z = U(z) Y,    u_h = V(z) grad_h Y / k  -  W(z) e_z x grad_h Y / k,

and the tractions on a horizontal plane are ``R = sigma_zz`` and the poloidal and
toroidal parts ``S = mu (V' + k U)``, ``T = mu W'``. In a homogeneous layer::

    U' = (R + lam k V) / (lam + 2 mu)      R' = k S
    V' = S / mu - k U                      S' = -k lam R / (lam + 2 mu)
                                                + 4 k^2 mu (lam + mu) V / (lam + 2 mu)
    W' = T / mu                            T' = k^2 mu W

whose solutions are ``e^{-kz}``, ``kz e^{-kz}``, ``e^{kz}`` and ``kz e^{kz}`` (from
the Papkovich-Neuber potentials). Each layer writes its decaying family from its
top and its growing family from its bottom, so nothing exceeds 1 and the global
matrix stays well conditioned at any ``k h`` -- the global-matrix method of Schmidt
and Tango (1986). A point source is a jump in ``(U, V, R, S)`` and ``(W, T)`` at its
depth, derived from the equivalent body force ``-M grad delta``.

Checked three ways (``tests/test_slip_wavenumber.py``; 2026-10-12):

* **Homogeneous half-space against the analytic point source** (tiny triangular
  dislocations through :func:`slipsolve._tde.tde_disp_hs`, the way
  :meth:`~slipsolve.edgrn.EdgrnTables.homogeneous` builds its tables): all four
  sources, all three components, receivers at the surface and at depth, within
  1.7e-5 of each component's peak over 100 m - 200 km.
* **Reciprocity of point forces across the Venezuela crust's layers**:
  ``G_zz``, ``G_xx``, ``G_yy`` and ``G_xz = G_zx`` between depths in different layers
  agree to 1e-14 - 1e-17. A wrong layer matrix or interface condition breaks it.
* **Against the fine EDGRN tables** (1601 x 401, the production crust) at the
  surface: 1e-6 - 1e-3 of each component's peak beyond 10 km. Inside it, EDGRN is
  the one that is off: a 550 m deep CLVD reads -2.0e-9 at 1 km against our
  +2.3e-9 and EDGRN's own +2.9e-9 at 2 km, and a 25 km deep dip-slip source's
  transverse component flips sign between EDGRN's nodes at 0, 1 and 2 km, where
  the field cannot vary on that scale.
"""

from pathlib import Path

import numpy as np
from scipy import fft as _fft

#: Source types: EDGRN's three, then the explosion.
SOURCE_TYPES = ("ss", "ds", "cl", "ep")

#: The (source, component) pairs that are not identically zero, in table order.
#: ``cl`` and ``ep`` are axisymmetric and drive no transverse motion.
TABLE_KEYS = (("ss", "uz"), ("ss", "ur"), ("ss", "ut"),
              ("ds", "uz"), ("ds", "ur"), ("ds", "ut"),
              ("cl", "uz"), ("cl", "ur"),
              ("ep", "uz"), ("ep", "ur"))

#: The combinations of :data:`TABLE_KEYS` that the strain divides by ``r``
#: (:func:`~slipsolve.layered_tde.combine_strain`), each tabulated over ``r``.
OVER_R_KEYS = (("ss", "ur-2ut"), ("ss", "2ur-ut"), ("ss", "uz"), ("ds", "ur+ut"), ("ds", "uz"),
               ("cl", "ur"), ("ep", "ur"))

#: The strain tables' columns: :data:`TABLE_KEYS`' derivatives in distance (``"dr"``), then
#: in receiver depth (``"dz"``, depth positive down), then :data:`OVER_R_KEYS` over ``r``
#: (``"/r"``). The azimuthal derivatives need no table: the azimuthal patterns are known.
STRAIN_KEYS = (tuple(("dr",) + key for key in TABLE_KEYS)
               + tuple(("dz",) + key for key in TABLE_KEYS)
               + tuple(("/r",) + key for key in OVER_R_KEYS))

#: Wavenumber range (1/m) of the Hankel transforms, and the number of samples.
#: Fourteen decades put the transform's periodic ends far outside any distance
#: used: ``1/k`` spans 1 cm to 1e12 m against tables from 0.5 m to 2000 km.
K_RANGE = (1e-12, 1e2)
N_WAVENUMBERS = 4096

#: Traction rows of the global matrix are divided by ``k * MU_SCALE`` so that
#: every entry is of order one.
MU_SCALE = 3e10

_TWO_PI = 2.0 * np.pi


# -- the layer stack ---------------------------------------------------------------
def layer_stack(model):
    """``(tops, lam, mu)`` of a piecewise-constant :class:`~slipsolve.edgrn.VelocityModel`.

    The last layer is the half-space. Rows at one depth are a step; two rows at
    different depths with the same properties are one layer, and consecutive
    layers with equal properties are merged.

    ⚠️ A gradient (two rows at different depths with different values) raises:
    this solver needs homogeneous layers, and cutting a gradient into steps would
    put an interface -- with its graded depth grid -- at every step.
    """
    depth = np.asarray(model.depth, dtype=float)
    lam = np.asarray(model.lam, dtype=float)
    mu = np.asarray(model.mu, dtype=float)
    if depth[0] > 0:
        raise ValueError("the velocity model must start at the surface")
    tops, lams, mus = [], [], []
    for i in range(depth.size - 1):
        if depth[i + 1] == depth[i]:
            continue
        if not (np.isclose(lam[i], lam[i + 1], rtol=1e-9)
                and np.isclose(mu[i], mu[i + 1], rtol=1e-9)):
            raise NotImplementedError(
                f"the model has a gradient between {depth[i]:.0f} and {depth[i + 1]:.0f} m; "
                "the wavenumber solver needs homogeneous layers")
        tops.append(depth[i])
        lams.append(lam[i])
        mus.append(mu[i])
    tops.append(depth[-1])
    lams.append(lam[-1])
    mus.append(mu[-1])
    keep_t, keep_l, keep_m = [tops[0]], [lams[0]], [mus[0]]
    for t, l, m in zip(tops[1:], lams[1:], mus[1:]):
        if np.isclose(l, keep_l[-1], rtol=1e-9) and np.isclose(m, keep_m[-1], rtol=1e-9):
            continue
        keep_t.append(t)
        keep_l.append(l)
        keep_m.append(m)
    return np.array(keep_t), np.array(keep_l), np.array(keep_m)


def depth_grid(tops, z_max=200e3, first=5.0, growth=1.5, spacing=500.0, deep_spacing=5000.0):
    """Depth nodes for sources and receivers, graded toward every interface.

    Inside each finite layer the nodes start ``first`` from each interface and
    grow by ``growth`` until they are ``spacing`` apart, with a uniform middle. The
    top layer starts at the surface itself, where the remainder is smooth (both
    fields have the free surface). Below the last interface the spacing grows to
    ``deep_spacing``, since the remainder decays away from it. No node sits on an
    interface.
    """
    tops = np.asarray(tops, dtype=float)

    def graded(limit, cap):
        out, d, step = [], first, first
        while d < limit:
            out.append(d)
            step = min(step * growth, cap)
            d += step
        return np.array(out)

    nodes = [0.0]
    for i, a in enumerate(tops):
        if i + 1 == tops.size:
            nodes += list(a + graded(z_max - a, deep_spacing)) + [z_max]
            continue
        b = tops[i + 1]
        offsets = graded(0.5 * (b - a), spacing)
        lower = b - offsets
        upper = a + offsets if i > 0 else np.array([0.0])
        lo, hi = upper.max(), lower.min()
        middle = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / spacing)) + 1))[1:-1]
        nodes += list(upper) + list(middle) + list(lower)
    nodes = np.unique(np.array(nodes))
    return nodes[nodes <= z_max]


def distance_grid(r_min=0.5, r_max=2000e3, per_decade=25):
    """Log-spaced horizontal distances; a lookup below ``r_min`` takes its value."""
    n = int(np.ceil(np.log10(r_max / r_min) * per_decade)) + 1
    return np.geomspace(r_min, r_max, n)


# -- the wavenumber-domain solution ------------------------------------------------
def _psv_columns(zt, zb, lam, mu, half):
    """P-SV solutions as columns, rows ``(U, V, R / (k MU_SCALE), S / (k MU_SCALE))``.

    ``zt = k (z - top) >= 0`` and ``zb = k (z - bottom) <= 0``, so every exponential
    is at most 1. ``half`` keeps only the family that decays downward.
    """
    c = (lam + 3.0 * mu) / (lam + mu)          # 3 - 4 nu
    g = (lam + 2.0 * mu) / (lam + mu)
    h = 2.0 * mu / (lam + mu)                  # c - 1
    s = mu / MU_SCALE
    ea = np.exp(-zt)
    cols = [np.stack([ea, -ea, -2.0 * s * ea, 2.0 * s * ea], -1),
            np.stack([(c + zt) * ea, -zt * ea, -2.0 * s * (g + zt) * ea,
                      s * (h + 2.0 * zt) * ea], -1)]
    if not half:
        eb = np.exp(zb)
        cols += [np.stack([eb, eb, 2.0 * s * eb, 2.0 * s * eb], -1),
                 np.stack([(c - zb) * eb, -zb * eb, 2.0 * s * (g - zb) * eb,
                           s * (h - 2.0 * zb) * eb], -1)]
    return np.stack(cols, -1)


def _sh_columns(zt, zb, mu, half):
    """SH solutions as columns, rows ``(W, T / (k MU_SCALE))``."""
    s = mu / MU_SCALE
    ea = np.exp(-zt)
    cols = [np.stack([ea, -s * ea], -1)]
    if not half:
        eb = np.exp(zb)
        cols.append(np.stack([eb, s * eb], -1))
    return np.stack(cols, -1)


def source_jumps(lam, mu):
    """Jumps in ``(U, V, R', S')`` and ``(W, T')`` across a unit-potency point source.

    ``(4, 4)`` and ``(4, 2)``, one row per :data:`SOURCE_TYPES`, with ``R'``, ``S'``,
    ``T'`` scaled as in the global matrix. The moment of a unit potency is ``mu``
    times the unit tensor for ``ss`` (``m12``), ``ds`` (``m13``) and ``cl``
    (``m33 = 1``, ``m11 = m22 = -1/2``), and ``K = lam + 2 mu / 3`` times the
    identity for ``ep`` -- so like EDGRN's, a table carries the source layer's
    moduli and a value steps where the source crosses an interface.

    The pairing of P-SV and SH follows the azimuthal pattern: ``ss`` is the
    ``sin 2 phi`` P-SV system with the ``cos 2 phi`` SH one, ``ds`` the ``cos phi``
    P-SV with the ``sin phi`` SH.
    """
    lam, mu = float(lam), float(mu)
    p = lam + 2.0 * mu
    bulk = lam + 2.0 * mu / 3.0
    psv = np.zeros((4, 4))
    sh = np.zeros((4, 2))
    # ss: a horizontal double couple, a jump in horizontal traction only.
    psv[0, 3] = -mu / (_TWO_PI * MU_SCALE)
    sh[0, 1] = mu / (_TWO_PI * MU_SCALE)
    # ds: m13 is also a horizontal plane slipping north -- a jump in displacement.
    psv[1, 1] = 1.0 / _TWO_PI
    sh[1, 0] = 1.0 / _TWO_PI
    # cl and ep: axisymmetric.
    psv[2, 0] = mu / (_TWO_PI * p)
    psv[2, 3] = -mu * (lam / p + 0.5) / (_TWO_PI * MU_SCALE)
    psv[3, 0] = bulk / (_TWO_PI * p)
    psv[3, 3] = bulk * (2.0 * mu / p) / (_TWO_PI * MU_SCALE)
    return psv, sh


def _layer_index(tops, z):
    """The layer each depth is in; a depth on an interface belongs to the layer below."""
    return np.searchsorted(tops, z, side="right") - 1


class _Solution:
    """The layer coefficients for one source depth, at every wavenumber."""

    def __init__(self, k, tops, lam, mu, z_source, psv_jump, sh_jump):
        k = np.asarray(k, dtype=float)
        j = int(_layer_index(tops, z_source))
        if z_source > 0 and np.any(np.isclose(tops, z_source, rtol=0, atol=1e-9)):
            raise ValueError(f"a source at {z_source} m sits on a layer interface")
        # Split the source's layer at the source: the jump goes on that interface.
        self.tops = np.insert(tops, j + 1, z_source)
        self.lam = np.insert(lam, j + 1, lam[j])
        self.mu = np.insert(mu, j + 1, mu[j])
        self.k = k
        n_layers = self.tops.size
        thick = np.diff(self.tops)
        nk, nsrc = k.size, psv_jump.shape[0]

        n_psv = 4 * (n_layers - 1) + 2
        self._col = ([slice(4 * i, 4 * i + 4) for i in range(n_layers - 1)]
                     + [slice(4 * (n_layers - 1), n_psv)])
        n_sh = 2 * (n_layers - 1) + 1
        self._cols = ([slice(2 * i, 2 * i + 2) for i in range(n_layers - 1)]
                      + [slice(2 * (n_layers - 1), n_sh)])
        a = np.zeros((nk, n_psv, n_psv))
        b = np.zeros((nk, n_sh, n_sh))
        rhs = np.zeros((nk, n_psv, nsrc))
        rhs_sh = np.zeros((nk, n_sh, nsrc))
        zero = np.zeros(nk)

        # The free surface: no traction.
        half = n_layers == 1
        below = zero if half else -k * thick[0]
        a[:, 0:2, self._col[0]] = _psv_columns(zero, below, self.lam[0], self.mu[0], half)[:, 2:4]
        b[:, 0:1, self._cols[0]] = _sh_columns(zero, below, self.mu[0], half)[:, 1:2]
        # Every interface: continuity, or the source's jump.
        for i in range(n_layers - 1):
            half = i + 1 == n_layers - 1
            below = zero if half else -k * thick[i + 1]
            rows = slice(2 + 4 * i, 6 + 4 * i)
            a[:, rows, self._col[i]] = -_psv_columns(k * thick[i], zero, self.lam[i],
                                                     self.mu[i], False)
            a[:, rows, self._col[i + 1]] = _psv_columns(zero, below, self.lam[i + 1],
                                                        self.mu[i + 1], half)
            rows_sh = slice(1 + 2 * i, 3 + 2 * i)
            b[:, rows_sh, self._cols[i]] = -_sh_columns(k * thick[i], zero, self.mu[i], False)
            b[:, rows_sh, self._cols[i + 1]] = _sh_columns(zero, below, self.mu[i + 1], half)
            if i == j:
                rhs[:, rows, :] = psv_jump.T[None]
                rhs_sh[:, rows_sh, :] = sh_jump.T[None]
        self.c = np.linalg.solve(a, rhs)
        self.c_sh = np.linalg.solve(b, rhs_sh)

    def at(self, z_receiver, derivatives=False):
        """``U``, ``V``, ``W`` at the receiver depths: each ``(n_z, n_k, n_sources)``.

        With ``derivatives``, also their derivatives in depth (``z`` down), ``U'``,
        ``V'``, ``W'``: six arrays. They come from the layer's equations and the
        tractions the solution already carries, ``U' = (R + lam k V) / (lam + 2 mu)``,
        ``V' = S / mu - k U`` and ``W' = T / mu``, with the receiver layer's constants.
        A receiver at the source depth takes the side below it.
        """
        z_receiver = np.atleast_1d(np.asarray(z_receiver, dtype=float))
        k = self.k
        n_layers = self.tops.size
        out = np.empty((6 if derivatives else 3, z_receiver.size, k.size, self.c.shape[-1]))
        layer = _layer_index(self.tops, z_receiver)
        for which in np.unique(layer):
            pick = np.flatnonzero(layer == which)
            half = which == n_layers - 1
            z = z_receiver[pick][:, None]
            zt = k[None, :] * (z - self.tops[which])
            zb = (np.zeros_like(zt) if half
                  else k[None, :] * (z - self.tops[which + 1]))
            e = _psv_columns(zt, zb, self.lam[which], self.mu[which], half)
            uv = np.einsum("zkri,kis->rzks", e[:, :, 0:2, :], self.c[:, self._col[which], :])
            out[0, pick], out[1, pick] = uv[0], uv[1]
            e_sh = _sh_columns(zt, zb, self.mu[which], half)
            out[2, pick] = np.einsum("zki,kis->zks", e_sh[:, :, 0, :],
                                     self.c_sh[:, self._cols[which], :])
            if derivatives:
                lam, mu = self.lam[which], self.mu[which]
                # The traction rows, scaled by 1 / (k MU_SCALE) as in the global matrix.
                rs = np.einsum("zkri,kis->rzks", e[:, :, 2:4, :], self.c[:, self._col[which], :])
                t = np.einsum("zki,kis->zks", e_sh[:, :, 1, :], self.c_sh[:, self._cols[which], :])
                kk = k[None, :, None]
                out[3, pick] = kk * (MU_SCALE * rs[0] + lam * uv[1]) / (lam + 2.0 * mu)
                out[4, pick] = kk * (MU_SCALE * rs[1] / mu - uv[0])
                out[5, pick] = kk * MU_SCALE * t / mu
        return tuple(out)


def wavenumbers(k_range=K_RANGE, n=N_WAVENUMBERS):
    """The log-spaced wavenumbers and their log spacing."""
    k = np.geomspace(k_range[0], k_range[1], int(n))
    return k, float(np.log(k_range[1] / k_range[0]) / (int(n) - 1))


def kernels(tops, lam, mu, z_source, z_receiver, k, remainder=True, derivatives=False):
    """``U``, ``V``, ``W`` per unit potency, ``(n_z, n_k, 4)`` each.

    ``remainder=True`` subtracts the half-space with the source layer's ``lam`` and
    ``mu``, which is what the tables hold. ``False`` keeps the whole layered field,
    which is singular at the source: transform it only for receivers off the
    source's depth. ``derivatives`` adds their depth derivatives ``U'``, ``V'``,
    ``W'`` (see :meth:`_Solution.at`), six arrays in all; the remainder's are
    continuous at the source, where both fields jump alike.
    """
    j = int(_layer_index(tops, z_source))
    psv, sh = source_jumps(lam[j], mu[j])
    layered = _Solution(k, tops, lam, mu, z_source, psv, sh).at(z_receiver, derivatives)
    if not remainder:
        return layered
    half = _Solution(k, tops[:1], lam[j:j + 1], mu[j:j + 1], z_source, psv, sh).at(z_receiver,
                                                                                derivatives)
    return tuple(a - b for a, b in zip(layered, half))


def hankel(f, k, dln, order):
    """``int f(k) J_order(k r) k dk`` by FFTLog; returns ``(r, values)`` on its own grid.

    ``f`` is ``(..., n_k)`` on :func:`wavenumbers`' grid. ``scipy.fft.fht`` computes
    ``int a(k) J(k r) r dk``, so it is handed ``a = k f`` and its output divided by
    ``r``. The offset is FFTLog's low-ringing one.
    """
    offset = _fft.fhtoffset(dln, mu=order, initial=0.0)
    values = _fft.fht(k * f, dln, mu=order, offset=offset)
    n = k.size
    centre = (n - 1) / 2.0
    k_centre = k[0] * np.exp(centre * dln)
    r = np.exp(offset) / k_centre * np.exp((np.arange(n) - centre) * dln)
    return r, values / r


def _log_distance(r):
    """``log r``, with ``r = 0`` at the smallest positive float: interpolation clamps it to the first node."""
    return np.log(np.maximum(np.asarray(r, dtype=float), np.finfo(float).tiny))


def _hankel_at(f, k, dln, order, log_r):
    """:func:`hankel` at ``r = exp(log_r)``, linear in ``log r`` between its own nodes."""
    grid, values = hankel(f, k, dln, order)
    flat = values.reshape(-1, values.shape[-1])
    out = np.stack([np.interp(log_r, np.log(grid), row) for row in flat])
    return out.reshape(values.shape[:-1] + (log_r.size,))


def _hankel_dr_at(f, k, dln, order, log_r):
    """``d/dr`` of :func:`_hankel_at`'s integral: ``(H_{n-1}[k f] - H_{n+1}[k f]) / 2``.

    From ``J_n'(x) = (J_{n-1}(x) - J_{n+1}(x)) / 2`` under the integral, and
    ``J_{-1} = -J_1`` for order 0 (FFTLog's kernel has a pole at order -1).
    """
    g = k * f
    lower = (-_hankel_at(g, k, dln, 1, log_r) if order == 0
             else _hankel_at(g, k, dln, order - 1, log_r))
    return 0.5 * (lower - _hankel_at(g, k, dln, order + 1, log_r))


def _hankel_over_r_at(f, k, dln, order, log_r):
    """:func:`_hankel_at`'s integral over ``r``: ``(H_{n-1}[k f] + H_{n+1}[k f]) / (2 n)``.

    From ``J_n(x) / x = (J_{n-1}(x) + J_{n+1}(x)) / (2 n)``, ``n >= 1``. No division by
    ``r``: dividing a transform that vanishes on the axis by ``r`` amplifies FFTLog's
    absolute error floor there (a strike-slip source's strain was 1000x off on the axis,
    2026-11-04), where these are finite and accurate.
    """
    if order < 1:
        raise ValueError("over r needs order >= 1; an order-0 transform does not vanish on the axis")
    g = k * f
    return (_hankel_at(g, k, dln, order - 1, log_r)
            + _hankel_at(g, k, dln, order + 1, log_r)) / (2.0 * order)


def transform(U, V, W, k, dln, r, radial=False):
    """The ten table components from ``U``, ``V``, ``W``, at distances ``r``.

    Returns ``(..., n_r, 10)`` in :data:`TABLE_KEYS` order, with EDGRN's azimuthal
    factors stripped: ``ss`` is ``sin 2 phi`` in ``uz``/``ur`` and ``cos 2 phi`` in
    ``ut``; ``ds`` is ``cos phi`` and ``sin phi``; ``cl`` and ``ep`` are axisymmetric.
    The horizontal components use ``J_m' = (J_{m-1} - J_{m+1}) / 2`` and
    ``m J_m / x = (J_{m-1} + J_{m+1}) / 2``.

    ``radial=True`` returns their derivatives in ``r`` instead: every column is a sum
    of Hankel transforms, and each transform's derivative is two more
    (:func:`_hankel_dr_at`), so the recipe is the same. Their derivatives in depth are
    this transform of ``U'``, ``V'``, ``W'`` (:func:`kernels` with ``derivatives``).
    A distance below FFTLog's first node (about 1 cm), ``r = 0`` included, takes that
    node's value.
    """
    log_r = _log_distance(r)
    one = _hankel_dr_at if radial else _hankel_at

    def h(f, order):
        return one(f, k, dln, order, log_r)

    u = {name: U[..., i] for i, name in enumerate(SOURCE_TYPES)}
    v = {name: V[..., i] for i, name in enumerate(SOURCE_TYPES)}
    w = {name: W[..., i] for i, name in enumerate(SOURCE_TYPES)}
    ss_minus = h(v["ss"] - w["ss"], 1)
    ss_plus = h(v["ss"] + w["ss"], 3)
    ds_plus = h(v["ds"] + w["ds"], 0)
    ds_minus = h(v["ds"] - w["ds"], 2)
    columns = [
        h(u["ss"], 2), 0.5 * ss_minus - 0.5 * ss_plus, 0.5 * ss_minus + 0.5 * ss_plus,
        h(u["ds"], 1), 0.5 * ds_plus - 0.5 * ds_minus, -0.5 * ds_plus - 0.5 * ds_minus,
        h(u["cl"], 0), -h(v["cl"], 1),
        h(u["ep"], 0), -h(v["ep"], 1),
    ]
    return np.stack(columns, axis=-1)


def transform_over_r(U, V, W, k, dln, r):
    """The :data:`OVER_R_KEYS` combinations divided by ``r``, at distances ``r``: ``(..., n_r, 7)``.

    With :func:`transform`'s recipe, ``ss_ur = (S1 - S3) / 2`` and ``ss_ut = (S1 + S3) / 2``
    for ``S1 = H_1[V - W]``, ``S3 = H_3[V + W]``; ``ds_ur + ds_ut = -H_2[V - W]`` (the order-0
    parts cancel exactly); ``cl_ur = -H_1[V]``. Every term has order >= 1, so each goes
    over ``r`` by :func:`_hankel_over_r_at`, finite and smooth on the axis.
    """
    log_r = _log_distance(r)

    def over(f, order):
        return _hankel_over_r_at(f, k, dln, order, log_r)

    u = {name: U[..., i] for i, name in enumerate(SOURCE_TYPES)}
    v = {name: V[..., i] for i, name in enumerate(SOURCE_TYPES)}
    w = {name: W[..., i] for i, name in enumerate(SOURCE_TYPES)}
    s1 = over(v["ss"] - w["ss"], 1)
    s3 = over(v["ss"] + w["ss"], 3)
    columns = [-0.5 * s1 - 1.5 * s3, 0.5 * s1 - 1.5 * s3, over(u["ss"], 2),
               -over(v["ds"] - w["ds"], 2), over(u["ds"], 1),
               -over(v["cl"], 1), -over(v["ep"], 1)]
    return np.stack(columns, axis=-1)


def strain_columns(U, V, W, dU, dV, dW, k, dln, r):
    """The :data:`STRAIN_KEYS` columns from the kernels and their depth derivatives: ``(..., n_r, 27)``."""
    return np.concatenate([transform(U, V, W, k, dln, r, radial=True),
                           transform(dU, dV, dW, k, dln, r),
                           transform_over_r(U, V, W, k, dln, r)], axis=-1)


# -- the tables ---------------------------------------------------------------------
#: A source at the surface itself has an integrand that does not decay for a receiver
#: there; the surface tables take their first node from this depth instead, metres.
SURFACE_SOURCE_FLOOR = 0.5


def surface_depth_grid(tops, fine_to=60e3, z_max=200e3, first=5.0, growth=1.5, spacing=100.0):
    """Source depths for the surface tables: ``spacing`` to ``fine_to``, then growing to 2 km.

    A point source's field at the surface varies over a length near its depth, so
    linear interpolation in depth needs about 100 m to hold 1e-3 at a few km
    (EDGRN's fine tables use the same). The surface tables are 2-D, so this costs
    little; the 3-D remainder (:func:`depth_grid`) cannot afford it.
    """
    fine = depth_grid(tops, z_max=fine_to, first=first, growth=growth, spacing=spacing,
                      deep_spacing=spacing)
    deep, step, d = [], spacing, fine_to
    while d < z_max:
        step = min(step * 1.1, 2000.0)
        d = min(d + step, z_max)
        deep.append(d)
    return np.unique(np.concatenate([fine, deep]))


def _one_source_depth(task):
    """One source depth's remainder slice, ``(n_r, n_z, 10)``, and with ``strain`` its
    strain columns, ``(n_r, n_z, 27)`` (:data:`STRAIN_KEYS`), else ``None``."""
    tops, lam, mu, z_source, z, r, k_range, n_k, strain = task
    k, dln = wavenumbers(k_range, n_k)
    if not strain:
        U, V, W = kernels(tops, lam, mu, z_source, z, k)
        return np.moveaxis(transform(U, V, W, k, dln, r), 1, 0), None
    U, V, W, dU, dV, dW = kernels(tops, lam, mu, z_source, z, k, derivatives=True)
    values = transform(U, V, W, k, dln, r)
    derivatives = strain_columns(U, V, W, dU, dV, dW, k, dln, r)
    return np.moveaxis(values, 1, 0), np.moveaxis(derivatives, 1, 0)


def _one_surface_depth(task):
    """One source depth's whole and remainder fields at the surface, two ``(n_r, 10)`` rows."""
    tops, lam, mu, z_source, r, k_range, n_k = task
    k, dln = wavenumbers(k_range, n_k)
    z_s = max(float(z_source), SURFACE_SOURCE_FLOOR)
    j = int(_layer_index(tops, z_s))
    psv, sh = source_jumps(lam[j], mu[j])
    layered = _Solution(k, tops, lam, mu, z_s, psv, sh).at(np.array([0.0]))
    half = _Solution(k, tops[:1], lam[j:j + 1], mu[j:j + 1], z_s, psv, sh).at(np.array([0.0]))
    whole = transform(*layered, k, dln, r)[0]
    remainder = transform(*(a - b for a, b in zip(layered, half)), k, dln, r)[0]
    return whole, remainder


class _LogAxis:
    """Distances on a log-spaced grid: a lower node and a weight; below the first, its value."""

    def __init__(self, r):
        self.r = np.asarray(r, dtype=float)
        self._log = np.log(self.r)

    def stencil(self, distance):
        lr = np.log(np.maximum(np.asarray(distance, dtype=float), self.r[0]))
        fi = (lr - self._log[0]) / (self._log[1] - self._log[0])
        i0 = np.clip(np.floor(fi).astype(int), 0, self.r.size - 2)
        return i0, np.clip(fi - i0, 0.0, 1.0)


class _DepthAxis:
    """Depths on a grid with layer interfaces: two nodes in the depth's own layer and a weight."""

    def __init__(self, z, tops):
        self.z = np.asarray(z, dtype=float)
        self.tops = np.asarray(tops, dtype=float)
        layer = _layer_index(self.tops, self.z)
        #: Each layer's own nodes, which are all a lookup may interpolate between.
        self.nodes = [np.flatnonzero(layer == i) for i in range(self.tops.size)]
        if any(n.size < 2 for n in self.nodes):
            raise ValueError("every layer needs at least two depth nodes")

    def stencil(self, depth):
        depth = np.abs(np.asarray(depth, dtype=float))
        lo = np.empty(depth.shape, dtype=int)
        hi = np.empty(depth.shape, dtype=int)
        t = np.empty(depth.shape)
        layer = _layer_index(self.tops, depth)
        for which in np.unique(layer):
            m = layer == which
            nodes = self.nodes[which]
            zn = self.z[nodes]
            pos = np.clip(np.searchsorted(zn, depth[m]), 1, nodes.size - 1)
            a, b = nodes[pos - 1], nodes[pos]
            lo[m], hi[m] = a, b
            # Beyond the layer's first or last node: extrapolate linearly within it,
            # but never past the table's own depth range.
            w = (depth[m] - self.z[a]) / (self.z[b] - self.z[a])
            if which == self.tops.size - 1:
                w = np.minimum(w, 1.0)
            t[m] = np.maximum(w, 0.0) if which == 0 else w
        return lo, hi, t


class RemainderTables:
    """``R = layered - half-space`` per unit potency, on a ``(r, z_source, z_receiver)`` grid.

    ``values`` is ``(n_r, n_z, n_z, 10)`` in :data:`TABLE_KEYS` order, in EDGRN's
    north/east/down with the azimuthal factors stripped (see :func:`transform`).
    Sources and receivers share the depth grid ``z``. ``tops``, ``lam`` and ``mu``
    are the layer stack the half-space part takes its constants from: a source at
    depth ``z_s`` subtracts the half-space with its own layer's ``lam`` and ``mu``.

    ``surface`` (optional) holds receivers at the surface on their own, finer 2-D
    grid -- a dict of ``r``, ``z`` and two ``(n_r, n_z, 10)`` arrays: ``remainder``,
    and ``whole``, the whole layered field (EDGRN's kind of table) for point
    sources seen from far away (:class:`~slipsolve.layered_tde.LayeredTDE`'s
    ``far``). ⚠️ The 3-D grid is too coarse in depth for a point source's field at
    the surface (500 m mid-layer against the ~100 m that 1e-3 needs at a few km),
    so surface receivers should always come from ``surface``.

    ``strain`` (optional) is ``(n_r, n_z, n_z, 27)`` on the 3-D grid: the remainder's
    derivatives in distance and receiver depth and its ``1/r`` combinations
    (:data:`STRAIN_KEYS`), for the strain of a point source at depth (:meth:`gather_strain`).
    Receivers at the surface need none.
    """

    def __init__(self, r, z, values, tops, lam, mu, attrs=None, surface=None, strain=None):
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.values = np.asarray(values)
        self.tops = np.asarray(tops, dtype=float)
        self.lam = np.asarray(lam, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.attrs = dict(attrs or {})
        expected = (self.r.size, self.z.size, self.z.size, len(TABLE_KEYS))
        if self.values.shape != expected:
            raise ValueError(f"values have shape {self.values.shape}; expected {expected}")
        self.strain = None if strain is None else np.asarray(strain)
        if self.strain is not None and self.strain.shape != expected[:-1] + (len(STRAIN_KEYS),):
            raise ValueError(f"strain has shape {self.strain.shape}; expected "
                             f"{expected[:-1] + (len(STRAIN_KEYS),)}")
        self._r_axis = _LogAxis(self.r)
        self._z_axis = _DepthAxis(self.z, self.tops)
        self._nodes = self._z_axis.nodes
        self.surface = None
        if surface is not None:
            surface = {k: np.asarray(v) for k, v in surface.items()}
            shape = (surface["r"].size, surface["z"].size, len(TABLE_KEYS))
            for key in ("whole", "remainder"):
                if surface[key].shape != shape:
                    raise ValueError(f"surface {key} has shape {surface[key].shape}; expected {shape}")
            self.surface = surface
            self._surface_r = _LogAxis(surface["r"])
            self._surface_z = _DepthAxis(surface["z"], self.tops)

    @property
    def has_surface(self):
        return self.surface is not None

    @property
    def has_strain(self):
        return self.strain is not None

    @classmethod
    def compute(cls, model, z=None, r=None, k_range=K_RANGE, n_k=N_WAVENUMBERS,
                z_max=200e3, surface_z=None, surface_r=None, workers=None, strain=False):
        """Tabulate the remainder for ``model`` (a :class:`~slipsolve.edgrn.VelocityModel`).

        ``z`` defaults to :func:`depth_grid` to ``z_max`` and ``r`` to
        :func:`distance_grid`; the surface tables to :func:`surface_depth_grid` and
        50 distances a decade. Each source depth is independent; ``workers`` runs
        them in that many processes (the Venezuela crust's 172 depths took 377 s in
        one). ``strain`` also tabulates the 3-D grid's strain columns (:attr:`strain`):
        about five times the transforms and 2.7 times the memory of the 3-D part.
        """
        tops, lam, mu = layer_stack(model)
        z = depth_grid(tops, z_max) if z is None else np.asarray(z, dtype=float)
        r = distance_grid() if r is None else np.asarray(r, dtype=float)
        sz = (surface_depth_grid(tops, z_max=z_max) if surface_z is None
              else np.asarray(surface_z, dtype=float))
        sr = distance_grid(per_decade=50) if surface_r is None else np.asarray(surface_r, dtype=float)
        values = np.empty((r.size, z.size, z.size, len(TABLE_KEYS)))
        derivatives = (np.empty((r.size, z.size, z.size, len(STRAIN_KEYS))) if strain
                       else None)
        whole = np.empty((sr.size, sz.size, len(TABLE_KEYS)))
        remainder = np.empty_like(whole)
        tasks = [(tops, lam, mu, z_source, z, r, k_range, n_k, bool(strain)) for z_source in z]
        surface_tasks = [(tops, lam, mu, z_source, sr, k_range, n_k) for z_source in sz]

        def keep(j, block):
            values[:, j] = block[0]
            if strain:
                derivatives[:, j] = block[1]

        if workers and workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(int(workers)) as pool:
                for j, block in enumerate(pool.map(_one_source_depth, tasks)):
                    keep(j, block)
                for j, (w, rem) in enumerate(pool.map(_one_surface_depth, surface_tasks)):
                    whole[:, j], remainder[:, j] = w, rem
        else:
            for j, task in enumerate(tasks):
                keep(j, _one_source_depth(task))
            for j, task in enumerate(surface_tasks):
                whole[:, j], remainder[:, j] = _one_surface_depth(task)
        attrs = {"k_range": list(k_range), "n_k": int(n_k),
                 "model": model.to_dict() if hasattr(model, "to_dict") else None}
        return cls(r, z, values, tops, lam, mu, attrs=attrs,
                   surface={"r": sr, "z": sz, "whole": whole, "remainder": remainder},
                   strain=derivatives)

    # -- lookups ---------------------------------------------------------------------
    def layer(self, depth):
        """The layer index of each depth (an interface belongs to the layer below)."""
        return _layer_index(self.tops, np.abs(np.asarray(depth, dtype=float)))

    def poisson(self, depth):
        """Poisson's ratio of the layer each depth is in."""
        i = self.layer(depth)
        return self.lam[i] / (2.0 * (self.lam[i] + self.mu[i]))

    def _depth_stencil(self, depth):
        """Two nodes and a weight per depth, both nodes in the depth's own layer."""
        return self._z_axis.stencil(depth)

    def interpolate(self, distance, z_source, z_receiver):
        """Trilinear lookup in ``(log r, z_source, z_receiver)``; ``(..., 10)``.

        Below the first distance the value is the first node's (the remainder is
        flat there); depths interpolate within their own layer only.
        """
        distance = np.asarray(distance, dtype=float)
        shape = np.broadcast_shapes(distance.shape, np.shape(z_source), np.shape(z_receiver))
        distance = np.broadcast_to(distance, shape).ravel()
        z_source = np.broadcast_to(np.abs(np.asarray(z_source, dtype=float)), shape).ravel()
        z_receiver = np.broadcast_to(np.abs(np.asarray(z_receiver, dtype=float)), shape).ravel()
        out = self.gather(self.distance_stencil(distance), self._depth_stencil(z_source),
                          self._depth_stencil(z_receiver))
        return out.reshape(shape + (self.values.shape[-1],))

    def distance_stencil(self, distance):
        """Lower node and weight of each distance on the 3-D table's log grid."""
        return self._r_axis.stencil(distance)

    def depth_stencil(self, depth):
        """Two nodes in the depth's layer and a weight, on the 3-D table's depth grid."""
        return self._z_axis.stencil(depth)

    def gather(self, distance, source, receiver, table=None):
        """The trilinear sum over three stencils, one gather of all eight corners.

        ``table`` is the 3-D array to read, :attr:`values` by default.
        """
        table = self.values if table is None else table
        i0, ti = distance
        s0, s1, ts = source
        q0, q1, tq = receiver
        nz = self.z.size
        flat = table.reshape(-1, table.shape[-1])
        index, weight = [], []
        for ii, wi in ((i0, 1.0 - ti), (i0 + 1, ti)):
            for jj, wj in ((s0, 1.0 - ts), (s1, ts)):
                for ll, wl in ((q0, 1.0 - tq), (q1, tq)):
                    index.append((ii * nz + jj) * nz + ll)
                    weight.append(wi * wj * wl)
        corners = flat[np.stack(index)]                   # (8, n, 10)
        return np.einsum("cn,cnk->nk", np.stack(weight), corners)

    def gather_strain(self, distance, source, receiver):
        """:meth:`gather` on the strain tables: ``(n, 27)`` in :data:`STRAIN_KEYS` order."""
        if self.strain is None:
            raise ValueError("these tables carry no strain; compute them with strain=True")
        return self.gather(distance, source, receiver, table=self.strain)

    def surface_lookup(self, distance, source_depth, which="remainder"):
        """Bilinear lookup in ``(log r, z_source)`` for receivers at the surface; ``(n, 10)``.

        ``which`` is ``"remainder"`` or ``"whole"``.
        """
        i0, ti = self._surface_r.stencil(distance)
        s0, s1, ts = self._surface_z.stencil(source_depth)
        table = self.surface[which]
        nz = table.shape[1]
        flat = table.reshape(-1, table.shape[-1])
        index, weight = [], []
        for ii, wi in ((i0, 1.0 - ti), (i0 + 1, ti)):
            for jj, wj in ((s0, 1.0 - ts), (s1, ts)):
                index.append(ii * nz + jj)
                weight.append(wi * wj)
        return np.einsum("cn,cnk->nk", np.stack(weight), flat[np.stack(index)])

    # -- files ------------------------------------------------------------------------
    def save(self, path):
        import json
        path = Path(path).expanduser()
        extra = ({} if self.surface is None else
                 {f"surface_{k}": v for k, v in self.surface.items()})
        if self.strain is not None:
            extra["strain"] = self.strain
        np.savez_compressed(path, r=self.r, z=self.z, values=self.values, tops=self.tops,
                            lam=self.lam, mu=self.mu, attrs=json.dumps(self.attrs), **extra)
        if not path.suffix:
            path = path.with_suffix(".npz")
        #: The file the tables were saved to or loaded from: where pool workers load their own copy
        #: (:func:`~slipsolve.layered_tde.engine_pool`).
        self.path = path
        return path

    @classmethod
    def load(cls, path):
        import json
        with np.load(Path(path).expanduser(), allow_pickle=False) as f:
            surface = ({k: f[f"surface_{k}"] for k in ("r", "z", "whole", "remainder")}
                       if "surface_whole" in f.files else None)
            strain = f["strain"] if "strain" in f.files else None
            out = cls(f["r"], f["z"], f["values"], f["tops"], f["lam"], f["mu"],
                      attrs=json.loads(str(f["attrs"])), surface=surface, strain=strain)
        out.path = Path(path).expanduser().resolve()
        return out

    def __repr__(self):
        return (f"<RemainderTables {len(self.tops)} layers, nr={self.r.size} "
                f"r={self.r[0]:.1f}m-{self.r[-1] / 1e3:.0f}km, nz={self.z.size} "
                f"z=0-{self.z[-1] / 1e3:.0f}km{', strain' if self.has_strain else ''}>")
