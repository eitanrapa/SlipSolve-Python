"""Okada (1992): a point source in an elastic half-space, displacement and its gradient at depth.

A vectorised port of ``DC3D0`` with ``DCCON0``, ``DCCON1``, ``UA0``, ``UB0`` and ``UC0`` from
Y. Okada's ``DC3D.f`` (coded 1991, revised 2002), read on 2026-11-04 from the copy in
okada_wrapper (github.com/tbenthompson/okada_wrapper). Strike-slip, dip-slip and tensile
potencies; the inflation source is not ported. Float64 throughout (that copy's driver
declares its arguments ``REAL*4``).

Okada's frame: ``x`` along strike, ``y`` horizontal and to the left of strike, ``z`` up;
the source at depth ``depth`` under the origin, on a plane through it that dips at ``dip``
toward ``-y``; receivers at ``z <= 0``. Potency = moment / ``mu`` (slip x area) for the
double couples, opening x area for the tensile source.

Used by :meth:`slipsolve.layered_tde.LayeredTDE.strain`'s far path. Okada, Y. (1992),
Internal deformation due to shear and tensile faults in a half-space, BSSA 82, 1018-1040.
"""
import numpy as np

_PI2 = 2.0 * np.pi
_EPS = 1e-6


def point_source_unit(alpha, x, y, z, depth, sin_dip, cos_dip):
    """``DC3D0`` for unit strike-slip, dip-slip and tensile potency: ``(3, n, 12)``.

    All arguments broadcast to ``(n,)``: ``alpha = (lambda + mu) / (lambda + 2 mu)``, the
    receiver ``x, y, z`` in Okada's frame (``z <= 0``), the source ``depth > 0`` and the
    plane's ``sin(dip)``, ``cos(dip)``. The last axis is ``DC3D0``'s output order:
    ``ux, uy, uz``, their ``x`` derivatives, ``y`` derivatives, ``z`` derivatives. A
    receiver on the source (``R = 0``) gets zeros, as ``DC3D0`` returns.
    """
    alpha, x, y, z, depth, sd, cd = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in
                                                         (alpha, x, y, z, depth, sin_dip, cos_dip)))
    if np.any(z > 0.0):
        raise ValueError("receivers must be at or below the surface (z <= 0)")
    # DCCON0: the medium and dip constants
    alp = ((1.0 - alpha) / 2.0, alpha / 2.0, (1.0 - alpha) / alpha, 1.0 - alpha, alpha)
    vertical = np.abs(cd) < _EPS
    cd = np.where(vertical, 0.0, cd)
    sd = np.where(vertical, np.sign(sd), sd)
    dip = dict(sd=sd, cd=cd, sdsd=sd * sd, cdcd=cd * cd, sdcd=sd * cd, s2d=2.0 * sd * cd,
               c2d=cd * cd - sd * sd)
    x = np.where(np.abs(x) < _EPS, 0.0, x)
    y = np.where(np.abs(y) < _EPS, 0.0, y)
    out = np.zeros((3,) + x.shape + (12,))

    # the real source: -UA0 for the displacement and its x, y derivatives, +UA0 for z
    geo = _dccon1(x, y, depth + z, dip)
    singular = geo["r"] == 0.0
    ua = _ua0(geo, alp, dip)
    out[..., :9] -= ua[..., :9]
    out[..., 9:] += ua[..., 9:]
    # the image source: UA0 + UB0 + z UC0, and UC0 itself added to the z derivatives
    geo = _dccon1(x, y, depth - z, dip)
    uc = _uc0(geo, alp, dip, z)
    out += _ua0(geo, alp, dip) + _ub0(geo, alp, dip, z) + z[..., None] * uc
    out[..., 9:] += uc[..., :3]
    if np.any(singular):
        out[:, singular] = 0.0
    return out


def _dccon1(x, y, d, dip):
    """``DCCON1``: the station-geometry constants for a point source."""
    sd, cd = dip["sd"], dip["cd"]
    d = np.where(np.abs(d) < _EPS, 0.0, d)
    p = y * cd + d * sd
    q = y * sd - d * cd
    s = p * sd + q * cd
    t = p * cd - q * sd
    x2, y2, d2 = x * x, y * y, d * d
    r2 = x2 + y2 + d2
    r = np.sqrt(r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r3 = r * r2
        r5 = r3 * r2
        qr = 3.0 * q / r5
        g = dict(x=x, y=y, d=d, p=p, q=q, s=s, t=t, xy=x * y, x2=x2, y2=y2, d2=d2, r=r, r2=r2,
                 r3=r3, r5=r5, a3=1.0 - 3.0 * x2 / r2, a5=1.0 - 5.0 * x2 / r2,
                 b3=1.0 - 3.0 * y2 / r2, c3=1.0 - 3.0 * d2 / r2, qr=qr, qrx=5.0 * qr * x / r2,
                 uy=sd - 5.0 * y * q / r2, uz=cd + 5.0 * d * q / r2,
                 vy=s - 5.0 * y * p * q / r2, vz=t + 5.0 * d * p * q / r2)
    g["wy"] = g["uy"] + sd
    g["wz"] = g["uz"] + cd
    return g


def _ua0(g, alp, dip):
    """``UA0``: part A, ``(3, n, 12)`` for unit strike-, dip- and tensile potency."""
    alp1, alp2 = alp[0], alp[1]
    sd, cd, s2d, c2d = dip["sd"], dip["cd"], dip["s2d"], dip["c2d"]
    x, y, d, p, q, s, t = g["x"], g["y"], g["d"], g["p"], g["q"], g["s"], g["t"]
    xy, x2, r3, r5, qr, qrx = g["xy"], g["x2"], g["r3"], g["r5"], g["qr"], g["qrx"]
    a3, a5, uy, uz, vy, vz, wy, wz = (g[k] for k in ("a3", "a5", "uy", "uz", "vy", "vz", "wy", "wz"))
    out = np.empty((3,) + x.shape + (12,))
    with np.errstate(divide="ignore", invalid="ignore"):
        # strike-slip
        out[0] = np.stack([
            alp1 * q / r3 + alp2 * x2 * qr,
            alp1 * x / r3 * sd + alp2 * xy * qr,
            -alp1 * x / r3 * cd + alp2 * x * d * qr,
            x * qr * (-alp1 + alp2 * (1.0 + a5)),
            alp1 * a3 / r3 * sd + alp2 * y * qr * a5,
            -alp1 * a3 / r3 * cd + alp2 * d * qr * a5,
            alp1 * (sd / r3 - y * qr) + alp2 * 3.0 * x2 / r5 * uy,
            3.0 * x / r5 * (-alp1 * y * sd + alp2 * (y * uy + q)),
            3.0 * x / r5 * (alp1 * y * cd + alp2 * d * uy),
            alp1 * (cd / r3 + d * qr) + alp2 * 3.0 * x2 / r5 * uz,
            3.0 * x / r5 * (alp1 * d * sd + alp2 * y * uz),
            3.0 * x / r5 * (-alp1 * d * cd + alp2 * (d * uz - q)),
        ], axis=-1)
        # dip-slip
        out[1] = np.stack([
            alp2 * x * p * qr,
            alp1 * s / r3 + alp2 * y * p * qr,
            -alp1 * t / r3 + alp2 * d * p * qr,
            alp2 * p * qr * a5,
            -alp1 * 3.0 * x * s / r5 - alp2 * y * p * qrx,
            alp1 * 3.0 * x * t / r5 - alp2 * d * p * qrx,
            alp2 * 3.0 * x / r5 * vy,
            alp1 * (s2d / r3 - 3.0 * y * s / r5) + alp2 * (3.0 * y / r5 * vy + p * qr),
            -alp1 * (c2d / r3 - 3.0 * y * t / r5) + alp2 * 3.0 * d / r5 * vy,
            alp2 * 3.0 * x / r5 * vz,
            alp1 * (c2d / r3 + 3.0 * d * s / r5) + alp2 * 3.0 * y / r5 * vz,
            alp1 * (s2d / r3 - 3.0 * d * t / r5) + alp2 * (3.0 * d / r5 * vz - p * qr),
        ], axis=-1)
        # tensile
        out[2] = np.stack([
            alp1 * x / r3 - alp2 * x * q * qr,
            alp1 * t / r3 - alp2 * y * q * qr,
            alp1 * s / r3 - alp2 * d * q * qr,
            alp1 * a3 / r3 - alp2 * q * qr * a5,
            -alp1 * 3.0 * x * t / r5 + alp2 * y * q * qrx,
            -alp1 * 3.0 * x * s / r5 + alp2 * d * q * qrx,
            -alp1 * 3.0 * xy / r5 - alp2 * x * qr * wy,
            alp1 * (c2d / r3 - 3.0 * y * t / r5) - alp2 * (y * wy + q) * qr,
            alp1 * (s2d / r3 - 3.0 * y * s / r5) - alp2 * d * qr * wy,
            alp1 * 3.0 * x * d / r5 - alp2 * x * qr * wz,
            -alp1 * (s2d / r3 - 3.0 * d * t / r5) - alp2 * y * qr * wz,
            alp1 * (c2d / r3 + 3.0 * d * s / r5) - alp2 * (d * wz - q) * qr,
        ], axis=-1)
    return out / _PI2


def _ub0(g, alp, dip, z):
    """``UB0``: part B, ``(3, n, 12)``."""
    alp3 = alp[2]
    sd, sdsd, sdcd = dip["sd"], dip["sdsd"], dip["sdcd"]
    x, y, d, p, q = g["x"], g["y"], g["d"], g["p"], g["q"]
    xy, x2, y2, d2, r, r2, r3, r5 = (g[k] for k in ("xy", "x2", "y2", "d2", "r", "r2", "r3", "r5"))
    qr, qrx, a3, a5, uy, uz, vy, vz, wy, wz = (g[k] for k in
                                               ("qr", "qrx", "a3", "a5", "uy", "uz", "vy", "vz", "wy", "wz"))
    c = d + z
    out = np.empty((3,) + x.shape + (12,))
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = r + d
        d12 = 1.0 / (r * rd * rd)
        d32 = d12 * (2.0 * r + d) / r2
        d33 = d12 * (3.0 * r + d) / (r2 * rd)
        d53 = d12 * (8.0 * r2 + 9.0 * r * d + 3.0 * d2) / (r2 * r2 * rd)
        d54 = d12 * (5.0 * r2 + 4.0 * r * d + d2) / r3 * d12
        fi1 = y * (d12 - x2 * d33)
        fi2 = x * (d12 - y2 * d33)
        fi3 = x / r3 - fi2
        fi4 = -xy * d32
        fi5 = 1.0 / (r * rd) - x2 * d32
        fj1 = -3.0 * xy * (d33 - x2 * d54)
        fj2 = 1.0 / r3 - 3.0 * d12 + 3.0 * x2 * y2 * d54
        fj3 = a3 / r3 - fj2
        fj4 = -3.0 * xy / r5 - fj1
        fk1 = -y * (d32 - x2 * d53)
        fk2 = -x * (d32 - y2 * d53)
        fk3 = -3.0 * x * d / r5 - fk2
        # strike-slip
        out[0] = np.stack([
            -x2 * qr - alp3 * fi1 * sd,
            -xy * qr - alp3 * fi2 * sd,
            -c * x * qr - alp3 * fi4 * sd,
            -x * qr * (1.0 + a5) - alp3 * fj1 * sd,
            -y * qr * a5 - alp3 * fj2 * sd,
            -c * qr * a5 - alp3 * fk1 * sd,
            -3.0 * x2 / r5 * uy - alp3 * fj2 * sd,
            -3.0 * xy / r5 * uy - x * qr - alp3 * fj4 * sd,
            -3.0 * c * x / r5 * uy - alp3 * fk2 * sd,
            -3.0 * x2 / r5 * uz + alp3 * fk1 * sd,
            -3.0 * xy / r5 * uz + alp3 * fk2 * sd,
            3.0 * x / r5 * (-c * uz + alp3 * y * sd),
        ], axis=-1)
        # dip-slip
        out[1] = np.stack([
            -x * p * qr + alp3 * fi3 * sdcd,
            -y * p * qr + alp3 * fi1 * sdcd,
            -c * p * qr + alp3 * fi5 * sdcd,
            -p * qr * a5 + alp3 * fj3 * sdcd,
            y * p * qrx + alp3 * fj1 * sdcd,
            c * p * qrx + alp3 * fk3 * sdcd,
            -3.0 * x / r5 * vy + alp3 * fj1 * sdcd,
            -3.0 * y / r5 * vy - p * qr + alp3 * fj2 * sdcd,
            -3.0 * c / r5 * vy + alp3 * fk1 * sdcd,
            -3.0 * x / r5 * vz - alp3 * fk3 * sdcd,
            -3.0 * y / r5 * vz - alp3 * fk1 * sdcd,
            -3.0 * c / r5 * vz + alp3 * a3 / r3 * sdcd,
        ], axis=-1)
        # tensile
        out[2] = np.stack([
            x * q * qr - alp3 * fi3 * sdsd,
            y * q * qr - alp3 * fi1 * sdsd,
            c * q * qr - alp3 * fi5 * sdsd,
            q * qr * a5 - alp3 * fj3 * sdsd,
            -y * q * qrx - alp3 * fj1 * sdsd,
            -c * q * qrx - alp3 * fk3 * sdsd,
            x * qr * wy - alp3 * fj1 * sdsd,
            qr * (y * wy + q) - alp3 * fj2 * sdsd,
            c * qr * wy - alp3 * fk1 * sdsd,
            x * qr * wz + alp3 * fk3 * sdsd,
            y * qr * wz + alp3 * fk1 * sdsd,
            c * qr * wz - alp3 * a3 / r3 * sdsd,
        ], axis=-1)
    return out / _PI2


def _uc0(g, alp, dip, z):
    """``UC0``: part C, ``(3, n, 12)``."""
    alp4, alp5 = alp[3], alp[4]
    sd, cd, sdsd, sdcd, s2d, c2d = (dip[k] for k in ("sd", "cd", "sdsd", "sdcd", "s2d", "c2d"))
    x, y, d, p, q, s, t = g["x"], g["y"], g["d"], g["p"], g["q"], g["s"], g["t"]
    xy, x2, y2, d2, r2, r3, r5 = (g[k] for k in ("xy", "x2", "y2", "d2", "r2", "r3", "r5"))
    qr, qrx, a3, a5 = g["qr"], g["qrx"], g["a3"], g["a5"]
    c = d + z
    out = np.empty((3,) + x.shape + (12,))
    with np.errstate(divide="ignore", invalid="ignore"):
        q2 = q * q
        r7 = r5 * r2
        a7 = 1.0 - 7.0 * x2 / r2
        b5 = 1.0 - 5.0 * y2 / r2
        b7 = 1.0 - 7.0 * y2 / r2
        c5 = 1.0 - 5.0 * d2 / r2
        c7 = 1.0 - 7.0 * d2 / r2
        d7 = 2.0 - 7.0 * q2 / r2
        qr5 = 5.0 * q / r2
        qr7 = 7.0 * q / r2
        dr5 = 5.0 * d / r2
        # strike-slip
        du5 = 3.0 / r5 * (alp4 * y * a5 * cd + alp5 * c * (a5 * sd - y * qr5 * a7))
        out[0] = np.stack([
            -alp4 * a3 / r3 * cd + alp5 * c * qr * a5,
            3.0 * x / r5 * (alp4 * y * cd + alp5 * c * (sd - y * qr5)),
            3.0 * x / r5 * (-alp4 * y * sd + alp5 * c * (cd + d * qr5)),
            alp4 * 3.0 * x / r5 * (2.0 + a5) * cd - alp5 * c * qrx * (2.0 + a7),
            du5,
            3.0 / r5 * (-alp4 * y * a5 * sd + alp5 * c * (a5 * cd + d * qr5 * a7)),
            du5,
            3.0 * x / r5 * (alp4 * b5 * cd - alp5 * 5.0 * c / r2 * (2.0 * y * sd + q * b7)),
            3.0 * x / r5 * (-alp4 * b5 * sd + alp5 * 5.0 * c / r2 * (d * b7 * sd - y * c7 * cd)),
            3.0 / r5 * (-alp4 * d * a5 * cd + alp5 * c * (a5 * cd + d * qr5 * a7)),
            15.0 * x / r7 * (alp4 * y * d * cd + alp5 * c * (d * b7 * sd - y * c7 * cd)),
            15.0 * x / r7 * (-alp4 * y * d * sd + alp5 * c * (2.0 * d * cd - q * c7)),
        ], axis=-1)
        # dip-slip
        du5 = 3.0 * x / r5 * (alp4 * (c2d - 5.0 * y * t / r2) - alp5 * 5.0 * c / r2 * (s - y * p * qr7))
        out[1] = np.stack([
            alp4 * 3.0 * x * t / r5 - alp5 * c * p * qrx,
            -alp4 / r3 * (c2d - 3.0 * y * t / r2) + alp5 * 3.0 * c / r5 * (s - y * p * qr5),
            -alp4 * a3 / r3 * sdcd + alp5 * 3.0 * c / r5 * (t + d * p * qr5),
            alp4 * 3.0 * t / r5 * a5 - alp5 * 5.0 * c * p * qr / r2 * a7,
            du5,
            3.0 * x / r5 * (alp4 * (2.0 + a5) * sdcd - alp5 * 5.0 * c / r2 * (t + d * p * qr7)),
            du5,
            3.0 / r5 * (alp4 * (2.0 * y * c2d + t * b5) + alp5 * c * (s2d - 10.0 * y * s / r2 - p * qr5 * b7)),
            3.0 / r5 * (alp4 * y * a5 * sdcd - alp5 * c * ((3.0 + a5) * c2d + y * p * dr5 * qr7)),
            3.0 * x / r5 * (-alp4 * (s2d - t * dr5) - alp5 * 5.0 * c / r2 * (t + d * p * qr7)),
            3.0 / r5 * (-alp4 * (d * b5 * c2d + y * c5 * s2d) - alp5 * c * ((3.0 + a5) * c2d + y * p * dr5 * qr7)),
            3.0 / r5 * (-alp4 * d * a5 * sdcd - alp5 * c * (s2d - 10.0 * d * t / r2 + p * qr5 * c7)),
        ], axis=-1)
        # tensile
        du5 = 3.0 * x / r5 * (-alp4 * (s2d - 5.0 * y * s / r2)
                              - alp5 * 5.0 / r2 * (c * (t - y + y * q * qr7) - y * z))
        out[2] = np.stack([
            3.0 * x / r5 * (-alp4 * s + alp5 * (c * q * qr5 - z)),
            alp4 / r3 * (s2d - 3.0 * y * s / r2) + alp5 * 3.0 / r5 * (c * (t - y + y * q * qr5) - y * z),
            -alp4 / r3 * (1.0 - a3 * sdsd) - alp5 * 3.0 / r5 * (c * (s - d + d * q * qr5) - d * z),
            -alp4 * 3.0 * s / r5 * a5 + alp5 * (c * qr * qr5 * a7 - 3.0 * z / r5 * a5),
            du5,
            3.0 * x / r5 * (alp4 * (1.0 - (2.0 + a5) * sdsd) + alp5 * 5.0 / r2 * (c * (s - d + d * q * qr7) - d * z)),
            du5,
            3.0 / r5 * (-alp4 * (2.0 * y * s2d + s * b5)
                        - alp5 * (c * (2.0 * sdsd + 10.0 * y * (t - y) / r2 - q * qr5 * b7) + z * b5)),
            3.0 / r5 * (alp4 * y * (1.0 - a5 * sdsd) + alp5 * (c * (3.0 + a5) * s2d - y * dr5 * (c * d7 + z))),
            3.0 * x / r5 * (-alp4 * (c2d + s * dr5) + alp5 * (5.0 * c / r2 * (s - d + d * q * qr7) - 1.0 - z * dr5)),
            3.0 / r5 * (alp4 * (d * b5 * s2d - y * c5 * c2d)
                        + alp5 * (c * ((3.0 + a5) * s2d - y * dr5 * d7) - y * (1.0 + z * dr5))),
            3.0 / r5 * (-alp4 * d * (1.0 - a5 * sdsd)
                        - alp5 * (c * (c2d + 10.0 * d * (s - d) / r2 - q * qr5 * c7) + z * (1.0 + c5))),
        ], axis=-1)
    return out / _PI2


def point_source_enu(nu, normal, slips, px, py, pz, x, y, z):
    """Displacement ``(n, k, 3)`` and strain ``(n, k, 6)`` of point sources, east/north/up.

    Unit potency of each of the ``k`` slip vectors ``slips`` (``(k, 3)``) on a plane with
    unit ``normal``, at sources ``(px, py, pz)`` (``pz < 0``), seen at receivers
    ``(x, y, z)`` (``z <= 0``); all ``(n,)``, one source per receiver. A slip vector's part
    in the plane is a double couple and its part along the normal an opening; the side
    the normal points into moves by ``+slip`` against the other, as for
    :func:`~slipsolve._tde.tde_disp_hs_unit`. Strain components as
    :data:`~slipsolve._tde.STRAIN_COMPONENTS` (tensor shears); per unit potency.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    slips = np.atleast_2d(np.asarray(slips, dtype=float))
    up = np.array([0.0, 0.0, 1.0])
    sign = 1.0 if normal[2] >= 0.0 else -1.0
    n_up = sign * normal                                   # Okada's hanging-wall normal
    h = float(np.hypot(n_up[0], n_up[1]))
    y_ok = np.array([0.0, 1.0, 0.0]) if h < 1e-12 else -np.array([n_up[0], n_up[1], 0.0]) / h
    x_ok = np.cross(y_ok, up)
    frame = np.stack([x_ok, y_ok, up])                     # rows: Okada's axes in east/north/up
    sin_dip, cos_dip = h, float(n_up[2])
    up_dip = cos_dip * y_ok + sin_dip * up
    along = slips @ normal                                 # opening
    plane = slips - along[:, None] * normal[None]          # double couple
    coeff = np.stack([sign * (plane @ x_ok), sign * (plane @ up_dip), along], axis=1)   # (k, 3 sources)

    px, py, pz, x, y, z = np.broadcast_arrays(*(np.asarray(a, dtype=float).ravel() for a in (px, py, pz, x, y, z)))
    de, dn = x - px, y - py
    alpha = 1.0 / (2.0 * (1.0 - float(nu)))
    unit = point_source_unit(alpha, de * x_ok[0] + dn * x_ok[1], de * y_ok[0] + dn * y_ok[1], z, -pz,
                             sin_dip, cos_dip)                        # (3, n, 12)
    u = np.moveaxis(np.tensordot(unit, coeff, axes=([0], [1])), -1, 1)   # (n, k, 12)
    disp = u[..., :3] @ frame                                         # u_enu = R^T u_ok
    grad_ok = np.stack([u[..., 3:6], u[..., 6:9], u[..., 9:12]], axis=-1)   # [i, j] = d u_i / d x_j
    grad = frame.T @ grad_ok @ frame        # R^T G R; a three-operand einsum took 3x as long (2026-11-05)
    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    i, j = (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)
    return disp, strain[..., i, j]
