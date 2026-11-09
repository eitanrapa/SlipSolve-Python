"""Earth's curvature and gravity, as a correction to a flat layered model's Green's functions.

Every engine puts the crust on a flat, non-gravitating half-space. Near a fault the real Earth differs from that by
little; at a few hundred kilometres, by several percent. `CurvatureTables` tabulates the difference for point sources at
the surface, per unit potency, as distance and source depth go: the field of the layers on a sphere (the deepest layer
filling the ball beneath) less the field of the same layers flat, plus, by default, the flat layers' buoyancy, which
the elastic engines leave out. `Curvature` wraps any engine and adds it.

The two are of a size and do not share a sign. Relative to the local field at 100-500 km, for a source at 10 km in a
layered crust: a vertical strike-slip source's horizontal field changes by 2-10% with the sphere and under 0.4% with
gravity, its vertical by 4-12% and 4-9% of opposite signs, which nearly cancel; a 45 degree thrust's horizontal by
0.3-1.7% and 2-11%, its vertical by 5-15% and 6-26%, of the same sign.

**The sphere.** Static, non-gravitating, solved degree by degree. In each homogeneous shell the solutions are powers of
the radius (Love's): `r^p` for `p = n - 1, n + 1, -n, -n - 2` (spheroidal) and `r^q` for `q = n, -n - 1` (toroidal), each
written from the shell's top (growing) or bottom (decaying) so that nothing exceeds 1, the global-matrix method of the
flat propagator (`slipsolve.wavenumber`). A point source at the pole splits into a displacement jump, from its radial
components, and a surface force, from its in-plane part: the surface divergence of that part, including its trace over
the radius pushing radially, which is what a flat layer does not have. The surface fields are Legendre sums. Degree 1
leaves out the rigid motions: the Earth's centre stays put.

**Gravity.** The flat layers hydrostatically prestressed, as the relaxation tables have them
(`slipsolve.relaxation`: Wang, Lorenzo-Martin & Roth 2006, eq. 10, buoyancy with no self-gravitation). The product of
the two corrections is left out, about 1% of the field at 500 km.

**The differences.** All three fields are computed with one Gaussian taper in wavenumber, `exp(-(k / k_taper)^2)` with
`k = (n + 1/2) / R` on the sphere, so the source's own near field, which they share, cancels instead of being summed to
high degree. A difference is then good wherever it is smooth on the taper's scale, `1 / k_taper`, which is everywhere it
is not negligible.

**Checked.** As the radius grows the sphere becomes the flat layers, its difference falling as `1 / R` (a check of every
sign and normalisation); degree by degree it satisfies Betti's reciprocal theorem against the same sphere loaded at its
surface, which does not use the source's jumps at all (1e-8, every source type, layered, degrees 2 and up). The flat
field of a uniform half-space agrees with an exact dislocation to 1e-4 within 500 km.

**Approximations.** Distances and azimuths are the local frame's, so the transverse Mercator's own distortion (a few
tenths of a percent at 500 km) is not corrected; each element's correction is integrated with a few point sources,
enough for a field that varies on the scale of the distance; the correction is computed on the layered reference, so
its product with a rigidity contrast or voxels (`slipsolve.bimaterial`, `slipsolve.heterogeneity`) is left out.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import wavenumber as wn
from ._quadrature import _moment_components, _slip_directions, _triangle_quadrature
from .relaxation import GRAVITY, _layers

__all__ = ["CurvatureTables", "Curvature"]

#: The components tabulated, as `slipsolve.relaxation`'s tables have them.
COMPONENTS = {"ss": ("uz", "ur", "ut"), "ds": ("uz", "ur", "ut"), "cl": ("uz", "ur")}
#: The default taper wavenumber (1/m): the sphere is summed to about four times it.
K_TAPER = 1.0e-3
#: Degrees solved together.
_DEGREE_CHUNK = 2000
#: From degree `n` on, layers more than `_DECAY R / n` below the source are left out: the field reaches them at
#: `exp(-_DECAY)` of itself.
_DECAY = 28.0


# -- the sphere, degree by degree ------------------------------------------------------------------------------------

def _sph_solutions(n, lam, mu):
    """Love's spheroidal solutions of degree `n` in a homogeneous shell: exponents and `(U, V)` directions, `(n, 4)`."""
    p = np.stack([n - 1.0, n + 1.0, -n, -n - 2.0], axis=-1)
    u = np.stack([n, (n + 1.0) * (lam * n + mu * (n - 2.0)), n * (lam * (n + 1.0) + mu * (n + 3.0)), n + 1.0], axis=-1)
    v = np.stack([np.ones_like(n), lam * (n + 3.0) + mu * (n + 5.0), lam * (2.0 - n) + mu * (4.0 - n), -np.ones_like(n)],
                 axis=-1)
    scale = np.maximum(np.abs(u), np.abs(v))
    return p, u / scale, v / scale


def _sph_columns(n, r, refs, lam, mu, which, scale):
    """Spheroidal solution columns at radius `r`, rows `(U, V, P / scale, S / scale)`: `(n, 4, len(which))`.

    `P` is the radial normal stress and `S` the shear stress on a sphere, the coefficients of `Y` and of its surface
    gradient; each solution is `(r / ref)^p` for its reference radius.
    """
    p, u, v = _sph_solutions(n, lam, mu)
    L = n * (n + 1.0)
    cols = []
    for j, ref in zip(which, refs):
        pj, uj, vj = p[:, j], u[:, j], v[:, j]
        f = (r / ref) ** pj
        P = ((lam + 2.0 * mu) * pj * uj + lam * (2.0 * uj - L * vj)) * f / r
        S = mu * ((pj - 1.0) * vj + uj) * f / r
        cols.append(np.stack([uj * f, vj * f, P / scale, S / scale], axis=-1))
    return np.stack(cols, axis=-1)


def _tor_columns(n, r, refs, mu, which, scale):
    """Toroidal solution columns, rows `(W, T / scale)`: `(n, 2, len(which))`."""
    q = np.stack([n, -n - 1.0], axis=-1)
    cols = []
    for j, ref in zip(which, refs):
        qj = q[:, j]
        f = (r / ref) ** qj
        cols.append(np.stack([f, mu * (qj - 1.0) * f / r / scale], axis=-1))
    return np.stack(cols, axis=-1)


def _shells(radius, tops, lam, mu, r_s, cut=None):
    """Shell radii with the source's radius inserted, their constants, and the source's interface index.

    Layers wholly deeper than `cut` (m below the surface) are dropped, the layer at `cut` filling the ball.
    """
    tops = np.asarray(tops, dtype=float)
    keep = tops < cut if cut is not None else np.ones(tops.size, dtype=bool)
    keep[0] = True
    tops, lam, mu = tops[keep], np.asarray(lam)[keep], np.asarray(mu)[keep]
    j = int(np.searchsorted(tops, radius - r_s, side="right") - 1)
    bounds = np.insert(radius - tops, j + 1, r_s)
    return bounds, np.insert(lam, j + 1, lam[j]), np.insert(mu, j + 1, mu[j]), j + 1


def _global(n, radius, bounds, lams, mus, src, jumps, scale, kind):
    """The global matrix, right-hand side and the surface shell's columns for one family (`"s"` or `"t"`)."""
    nd = n.size
    n_shell = bounds.size - 1
    per, in_ball = (4, 2) if kind == "s" else (2, 1)
    size = per * n_shell + in_ball
    A = np.zeros((nd, size, size))
    rhs = np.zeros((nd, size, jumps.shape[-1]))

    def shell(i, r):
        top, bottom = bounds[i], bounds[i + 1]
        if kind == "s":
            return _sph_columns(n, r, [top, top, bottom, bottom], lams[i], mus[i], [0, 1, 2, 3], scale)
        return _tor_columns(n, r, [top, bottom], mus[i], [0, 1], scale)

    def ball(r):
        if kind == "s":
            return _sph_columns(n, r, [bounds[-1], bounds[-1]], lams[-1], mus[-1], [0, 1], scale)
        return _tor_columns(n, r, [bounds[-1]], mus[-1], [0], scale)

    half = per // 2
    A[:, 0:half, 0:per] = shell(0, radius)[:, half:, :]                     # a free surface: no traction
    for i in range(n_shell):
        rows = slice(half + per * i, half + per * (i + 1))
        r_i = bounds[i + 1]
        A[:, rows, per * i:per * (i + 1)] = shell(i, r_i)
        A[:, rows, per * (i + 1):per * (i + 2) if i + 1 < n_shell else size] = (
            -shell(i + 1, r_i) if i + 1 < n_shell else -ball(r_i))
        if i + 1 == src:
            jump = jumps.copy()
            jump[:, half:] /= scale[:, None, None]
            rhs[:, rows, :] = jump
    return A, rhs, shell(0, radius), per * n_shell


def _solve(A, rhs, n, first_ball):
    coef = np.zeros(rhs.shape)
    regular = n >= 2
    if regular.any():
        coef[regular] = np.linalg.solve(A[regular], rhs[regular])
    for d in np.flatnonzero(~regular):
        keep = [c for c in range(A.shape[1]) if c != first_ball]           # degree 1: the ball's rigid motion
        coef[d][keep] = np.linalg.lstsq(A[d][:, keep], rhs[d], rcond=None)[0]
    return coef


def _jumps(n, lam, mu, r_s):
    """Spheroidal `(n, 4, 3)` (ss, ds, cl) and toroidal `(n, 2, 2)` (ss, ds) jumps per unit potency, above less below.

    Patterns: ss `P_n^2 sin 2 psi` (spheroidal), `P_n^2 cos 2 psi` (toroidal); ds `P_n^1 cos psi`, `P_n^1 sin psi`; cl
    `P_n`, with `psi` the longitude about the source (right-handed about up). The local axes at the source are x
    (psi 0, the tables' north) and y (psi 90 degrees, the tables' west). The source is a displacement jump
    `b = (M_rr / (lam + 2 mu), M_rx / mu, M_ry / mu)` times the point's delta, and a surface force from the in-plane
    moment `N = M_tt - lam b_r I`: its surface divergence, whose normal part is `trace(N) / r_s`.
    """
    L = n * (n + 1.0)
    safe = np.where(L > 0, L, 1.0)
    c = (2.0 * n + 1.0) / (4.0 * np.pi)
    sph = np.zeros((n.size, 4, 3))
    tor = np.zeros((n.size, 2, 2))
    # ss: north-east M = mu, so M_xy = -mu locally.
    n_xy = -mu
    sph[:, 3, 0] = np.where(n >= 2, -n_xy * c / (safe * r_s ** 3), 0.0)
    tor[:, 1, 0] = np.where(n >= 2, n_xy * c / (safe * r_s ** 3), 0.0)
    # ds: north-down M = mu, so M_xr = -mu: a jump in tangential displacement b_x = -1.
    b_x = -1.0
    sph[:, 1, 1] = np.where(n >= 1, -b_x * c / (safe * r_s ** 2), 0.0)
    tor[:, 0, 1] = np.where(n >= 1, -b_x * c / (safe * r_s ** 2), 0.0)
    # cl: M_rr = mu, M_xx = M_yy = -mu / 2.
    b_r = mu / (lam + 2.0 * mu)
    trace = 2.0 * (-0.5 * mu - lam * b_r)
    sph[:, 0, 2] = b_r * c / r_s ** 2
    sph[:, 2, 2] = -trace * c / r_s ** 3
    sph[:, 3, 2] = 0.5 * trace * c / r_s ** 3
    return sph, tor


def _radial(radius, tops, lam, mu, r_s, jumps, cut=None):
    """Degree 0: the surface `U` per column of `jumps` (`(4, k)`). Each shell `U = a r + b / r^2`, the ball `a r`."""
    bounds, lams, mus, src = _shells(radius, tops, lam, mu, r_s, cut)
    n_shell = bounds.size - 1
    size = 2 * n_shell + 1
    A = np.zeros((size, size))
    rhs = np.zeros((size, jumps.shape[-1]))
    scale = wn.MU_SCALE / radius

    def col(r, ref, lam_, mu_, p):
        f = (r / ref) ** p
        return np.array([f, ((3.0 * lam_ + 2.0 * mu_) if p == 1 else -4.0 * mu_) * f / r / scale])

    def shell(i, r):
        return np.column_stack([col(r, bounds[i], lams[i], mus[i], 1), col(r, bounds[i + 1], lams[i], mus[i], -2)])

    A[0:1, 0:2] = shell(0, radius)[1:2]
    for i in range(n_shell):
        rows = slice(1 + 2 * i, 3 + 2 * i)
        r_i = bounds[i + 1]
        A[rows, 2 * i:2 * i + 2] = shell(i, r_i)
        if i + 1 < n_shell:
            A[rows, 2 * (i + 1):2 * (i + 1) + 2] = -shell(i + 1, r_i)
        else:
            A[rows, 2 * n_shell:] = -col(r_i, bounds[-1], lams[-1], mus[-1], 1)[:, None]
        if i + 1 == src:
            rhs[rows, :] = np.array([jumps[0], jumps[2] / scale])
    return shell(0, radius)[0] @ np.linalg.solve(A, rhs)[0:2]


def _degrees(radius, tops, lam, mu, depth, n_max):
    """Surface `U, V` `(n_max + 1, 3)` and `W` `(n_max + 1, 2)` for a source at `depth`, every degree."""
    r_s = radius - depth
    j = int(np.searchsorted(np.asarray(tops), depth, side="right") - 1)
    U, V, W = np.zeros((n_max + 1, 3)), np.zeros((n_max + 1, 3)), np.zeros((n_max + 1, 2))
    sph0, _ = _jumps(np.array([0.0]), lam[j], mu[j], r_s)
    U[0] = _radial(radius, tops, lam, mu, r_s, sph0[0])
    for start in range(1, n_max + 1, _DEGREE_CHUNK):
        n = np.arange(start, min(start + _DEGREE_CHUNK, n_max + 1), dtype=float)
        cut = depth + _DECAY * radius / n[0]                                    # deeper layers do not reach the surface
        bounds, lams, mus, src = _shells(radius, tops, lam, mu, r_s, cut)
        scale = wn.MU_SCALE * n / radius
        sph, tor = _jumps(n, lam[j], mu[j], r_s)
        rows = slice(start, start + n.size)
        for kind, jumps, per in (("s", sph, 4), ("t", tor, 2)):
            A, rhs, top, first_ball = _global(n, radius, bounds, lams, mus, src, jumps, scale, kind)
            y = np.einsum("dri,dik->drk", top, _solve(A, rhs, n, first_ball)[:, 0:per, :])
            if kind == "s":
                U[rows], V[rows] = y[:, 0], y[:, 1]
            else:
                W[rows] = y[:, 0]
    return U, V, W


def _legendre_blocks(theta, n_max, block=2048):
    """`(start, P, dP)` blocks of `P_n^m(cos theta)` (Condon-Shortley) and `d/dtheta`, `m = 0, 1, 2`: arrays
    `(block, 3, n_theta)` for degrees `start .. start + block - 1`, by the upward recurrence in degree."""
    x, s = np.cos(theta), np.sin(theta)
    m = np.array([0.0, 1.0, 2.0])[:, None]
    lowest = np.stack([np.ones_like(x), -s, 3.0 * s * s])                   # P_m^m
    prev = np.zeros((3, x.size))                                            # P_{n-1}^m
    here = np.zeros((3, x.size))                                            # P_n^m
    for start in range(0, n_max + 1, block):
        degrees = np.arange(start, min(start + block, n_max + 1))
        P = np.zeros((degrees.size, 3, x.size))
        before = np.zeros((degrees.size, 3, x.size))
        for i, n in enumerate(degrees):
            if n <= 2:
                new = here.copy()
                for mm in range(3):
                    if n == mm:
                        new[mm] = lowest[mm]
                    elif n == mm + 1:
                        new[mm] = (2 * mm + 1) * x * here[mm]
                    elif n > mm + 1:
                        new[mm] = ((2 * n - 1) * x * here[mm] - (n + mm - 1) * prev[mm]) / (n - mm)
            else:
                new = ((2 * n - 1) * x * here - (n + m - 1) * prev) / (n - m)
                if n == 3:                                                  # m = 2 starts its recurrence here
                    new[2] = 5.0 * x * here[2]
            prev, here = here, new
            P[i], before[i] = here, prev
        n_col = degrees[:, None, None].astype(float)
        dP = (n_col * x * P - (n_col + m[None]) * before) / s
        yield start, P, dP


def _sphere_field(radius, tops, lam, mu, depth, theta, n_max, k_taper):
    """The sphere's surface field per unit potency, `(n_theta, 8)` in `COMPONENTS` order, the flat tables' convention:
    `uz` down, `ur` away from the source, `ut` clockwise in map view, the azimuthal factors stripped."""
    U, V, W = _degrees(radius, tops, lam, mu, depth, n_max)
    taper = np.exp(-(((np.arange(n_max + 1) + 0.5) / radius) / k_taper) ** 2)
    U, V, W = U * taper[:, None], V * taper[:, None], W * taper[:, None]
    theta = np.asarray(theta, dtype=float)
    out = np.zeros((theta.size, 8))
    axis = theta == 0.0
    if (~axis).any():
        t = theta[~axis]
        inv_s = 1.0 / np.sin(t)
        acc = np.zeros((t.size, 8))
        for start, P, dP in _legendre_blocks(t, n_max):
            rows = slice(start, start + P.shape[0])
            u, v, w = U[rows], V[rows], W[rows]
            p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
            d0, d1, d2 = dP[:, 0], dP[:, 1], dP[:, 2]
            acc[:, 0] += u[:, 0] @ p2
            acc[:, 1] -= v[:, 0] @ d2 - 2.0 * (w[:, 0] @ p2) * inv_s
            acc[:, 2] -= 2.0 * (v[:, 0] @ p2) * inv_s - w[:, 0] @ d2
            acc[:, 3] -= u[:, 1] @ p1
            acc[:, 4] += v[:, 1] @ d1 + (w[:, 1] @ p1) * inv_s
            acc[:, 5] -= (v[:, 1] @ p1) * inv_s + w[:, 1] @ d1
            acc[:, 6] -= u[:, 2] @ p0
            acc[:, 7] += v[:, 2] @ d0
        out[~axis] = acc
    if axis.any():                                                         # the limits at the source's own vertical
        n = np.arange(n_max + 1, dtype=float)
        half_l = 0.5 * n * (n + 1.0)
        out[axis, 4] = -np.sum(half_l * (V[:, 1] + W[:, 1]))
        out[axis, 5] = np.sum(half_l * (V[:, 1] + W[:, 1]))
        out[axis, 6] = -np.sum(U[:, 2])
    return out


def _flat_field(tops, lam, mu, depth, r, k_taper, buoyancy=None, k_range=(1e-10, 1.0), n_k=4096):
    """The flat layers' surface field with the same taper, `(n_r, 8)`; `buoyancy` (`rho g` per layer) adds gravity."""
    k, dln = wn.wavenumbers(k_range, n_k)
    taper = np.exp(-(k / k_taper) ** 2)
    j = int(wn._layer_index(np.asarray(tops), depth))
    psv, sh = wn.source_jumps(lam[j], mu[j])
    solution = wn._Solution(k, np.asarray(tops), lam, mu, depth, psv, sh, buoyancy=buoyancy)
    U, V, W = (a[0] * taper[:, None] for a in solution.at(np.array([0.0])))
    out = wn.transform(U, V, W, k, dln, r)[:, :8]
    axis = r == 0.0
    if axis.any():
        out[axis] = wn.on_axis(U[None], V[None], W[None], k, dln)[0, :8]
    return out


def _one_depth(task):
    """One source depth's correction `(nr, 8)`: the sphere less the flat layers, plus the flat layers' buoyancy."""
    tops, lam, mu, rho, gravity, depth, r, radius, n_max, k_taper = task
    flat = _flat_field(tops, lam, mu, depth, r, k_taper)
    out = _sphere_field(radius, tops, lam, mu, depth, r / radius, n_max, k_taper) - flat
    if gravity:
        out += _flat_field(tops, lam, mu, depth, r, k_taper, buoyancy=rho * gravity) - flat
    return out


class CurvatureTables:
    """The field of a layered sphere less that of the same layers flat, plus their buoyancy, at the surface, per unit
    potency.

    `values[kind][component]` is `(nr, nz)` on uniform distances `r` (arc length, m) and source depths `z` (m), in the
    layout of `slipsolve.relaxation`'s tables: kinds `ss`, `ds`, `cl`; components `uz` (down), `ur`, `ut`; the azimuthal
    factors stripped. `interfaces` are the depths no interpolation crosses.
    """

    def __init__(self, r, z, values, interfaces=(), attrs=None):
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.values = values
        self.interfaces = np.asarray(interfaces, dtype=float)
        self.attrs = dict(attrs or {})

    @classmethod
    def compute(cls, model, *, r_max=1000e3, nr=501, z_min=500.0, z_max=39500.0, nz=40, radius=None,
                gravity=GRAVITY, k_taper=K_TAPER, workers=None):
        """The correction for a `slipsolve.crust.VelocityModel`.

        - Distances are uniform from 0 to `r_max` (`nr`), source depths from `z_min` to `z_max` (`nz`), off every
          interface. The sphere has `radius` (default `slipsolve.interseismic.EARTH_RADIUS`); the model's deepest
          layer fills it beneath its last interface.
        - `gravity` (m/s^2) is the buoyancy's acceleration; 0 leaves gravity out, for the curvature alone.
        - `k_taper` (1/m) is the taper's wavenumber: the sphere is summed to degree `4 k_taper R`. The difference
          converges where the field is smooth on `1 / k_taper`; the default (1 km) leaves the tables good beyond a few
          kilometres, where the correction is still under 0.1% of the field.
        - `workers` runs the depths in that many processes.
        """
        from .interseismic import EARTH_RADIUS

        radius = float(EARTH_RADIUS if radius is None else radius)
        tops, lam, mu, rho, _ = _layers(model)
        r = np.linspace(0.0, float(r_max), int(nr))
        z = np.linspace(float(z_min), float(z_max), int(nz))
        if np.any(np.isclose(tops[1:, None], z[None, :], rtol=0.0, atol=1e-6)):
            raise ValueError("a source depth sits on a layer interface")
        n_max = int(np.ceil(4.0 * float(k_taper) * radius))
        tasks = [(tops, lam, mu, rho, float(gravity), depth, r, radius, n_max, float(k_taper)) for depth in z]
        if workers and int(workers) > 1:
            with ProcessPoolExecutor(int(workers)) as pool:
                rows = list(pool.map(_one_depth, tasks))
        else:
            rows = [_one_depth(task) for task in tasks]
        stack = np.stack(rows, axis=1)                                       # (nr, nz, 8)
        values, c = {}, 0
        for kind, comps in COMPONENTS.items():
            values[kind] = {}
            for comp in comps:
                values[kind][comp] = stack[:, :, c]
                c += 1
        attrs = {"source": "slipsolve.curvature", "radius": radius, "gravity": float(gravity), "k_taper": float(k_taper),
                 "n_max": n_max}
        return cls(r, z, values, interfaces=tops[1:], attrs=attrs)

    def save(self, path):
        """Write the tables to an `.npz`."""
        import json

        flat = {f"{kind}.{comp}": v for kind, comps in self.values.items() for comp, v in comps.items()}
        np.savez(path, r=self.r, z=self.z, interfaces=self.interfaces, attrs=json.dumps(self.attrs), **flat)

    @classmethod
    def load(cls, path):
        """Read tables `save` wrote."""
        import json

        d = np.load(path)
        values = {kind: {comp: d[f"{kind}.{comp}"] for comp in comps} for kind, comps in COMPONENTS.items()}
        return cls(d["r"], d["z"], values, interfaces=d["interfaces"], attrs=json.loads(str(d["attrs"])))

    def __repr__(self):
        return (f"<CurvatureTables r 0-{self.r[-1] / 1e3:.0f} km ({self.r.size}), z {self.z[0] / 1e3:.1f}-"
                f"{self.z[-1] / 1e3:.1f} km ({self.z.size}), R {self.attrs.get('radius', 0) / 1e3:.0f} km, "
                f"g {self.attrs.get('gravity', 0):g}>")


# -- the engine --------------------------------------------------------------------------------------------------------

class Curvature:
    """Another engine's Green's functions plus Earth's curvature and gravity, from `CurvatureTables`.

    `base` is any engine on the layered model the tables were computed for: `slipsolve.layered_tde.LayeredTDE`, or a
    `slipsolve.heterogeneity.HeterogeneousEngine` on it (the correction goes on the layered reference: the voxels'
    own fields are local, where curvature does not reach). Each element's correction is `order**2` point sources
    (`slipsolve._quadrature._triangle_quadrature`), weighted by its nodal tents on a nodal basis, at the frame's
    distances and azimuths. Receivers are on the surface; topography lowers the fault as for the base engine.
    """

    #: Assembles in any basis itself (a nodal column integrates its tent).
    supports_basis = True

    def __init__(self, base, tables, order=2):
        self.base = base
        self.tables = tables
        self.order = int(order)
        self.supports_topography = bool(getattr(base, "supports_topography", True))
        self.native_topography = bool(getattr(base, "native_topography", False))
        self.name = f"curved:{getattr(base, 'name', type(base).__name__)}"
        self.nu = getattr(base, "nu", None)
        self._grid = None

    def __getattr__(self, name):
        if name in ("rigidity_factor",) and "base" in self.__dict__:      # the moment's rigidity, when the base has one
            return getattr(self.base, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def describe(self):
        """A JSON-able record for a saved model's options: the base engine's, and the curvature tables'."""
        record = dict(self.base.describe()) if hasattr(self.base, "describe") else {}
        record["curvature"] = {k: self.tables.attrs[k] for k in ("radius", "gravity", "k_taper", "n_max")
                               if k in self.tables.attrs}
        return record

    # -- the correction ----------------------------------------------------------------------------------------------

    def _sources(self, mesh, basis):
        """Point sources: positions `(q, 3)`, packed moments per unit strike and dip slip `(2, 5, q)`, and the sparse
        `(q, n_parameters)` map of their potency weights onto the basis."""
        from scipy import sparse

        from ._quadrature import _tent

        nodal = basis is not None and getattr(basis, "name", "element") == "node"
        points, moments, rows, cols, vals = [], [], [], [], []
        q = 0
        for e in range(mesh.n_elements):
            vertices = mesh.vertices[e]
            pts, share = _triangle_quadrature(*vertices, self.order)
            along, down = _slip_directions(mesh.normals[e])
            m = np.stack([_moment_components(mesh.normals[e], along), _moment_components(mesh.normals[e], down)])
            weight = share * mesh.areas[e] * np.ones(len(pts))
            idx = np.arange(q, q + len(pts))
            if nodal:
                triangle = mesh.triangles[e]
                for node in triangle:
                    rows.append(idx)
                    cols.append(np.full(len(pts), node))
                    vals.append(weight * _tent(vertices, triangle, node, pts))
            else:
                rows.append(idx)
                cols.append(np.full(len(pts), e))
                vals.append(weight)
            points.append(pts)
            moments.append(np.repeat(m[:, :, None], len(pts), axis=2))
            q += len(pts)
        n_params = mesh.n_nodes if nodal else mesh.n_elements
        weights = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(q, n_params))
        return np.vstack(points), np.concatenate(moments, axis=2), weights

    def correction(self, mesh, x, y, look=None, basis=None, budget=1e6):
        """The curvature's part of the Green's functions: `(npts, 2 n_parameters)` along `look`
        (`(look_e, look_n, look_u)`), or east/north/up `(npts, 3, 2 n_parameters)` without it.

        `basis` is the element basis by default; a nodal basis integrates each node's tent. Projections onto any
        other basis go through its element means.
        """
        from .relaxation import _Grid, _stencil

        x, y = (np.asarray(v, dtype=float).ravel() for v in (x, y))
        projected = basis is not None and getattr(basis, "name", "element") not in ("element", "node")
        points, moments, weights = self._sources(mesh, None if projected else basis)
        if self._grid is None:
            self._grid = _Grid(self.tables.r, self.tables.z, self.tables.interfaces)
        grid = self._grid
        depth = -points[:, 2]
        spacing = grid.z[1] - grid.z[0] if grid.z.size > 1 else 0.0
        if depth.min() < grid.z[0] - spacing or depth.max() > grid.z[-1] + spacing:
            raise ValueError(f"the sources lie at {depth.min() / 1e3:.2f}-{depth.max() / 1e3:.2f} km, the tables at "
                             f"{grid.z[0] / 1e3:.2f}-{grid.z[-1] / 1e3:.2f} km")
        if look is not None:
            look = np.stack([np.asarray(a, dtype=float).ravel() for a in look], axis=1)
            if look.shape[0] != x.size:
                raise ValueError("Look-vector components must match the observation count")
        n_params = weights.shape[1]
        out = np.zeros((x.size, 2 * n_params) if look is not None else (x.size, 3, 2 * n_params))
        values = self.tables.values
        chunk = max(1, int(budget // max(x.size, 1)))
        for start in range(0, points.shape[0], chunk):
            sl = slice(start, start + chunk)
            de, dn = x[:, None] - points[None, sl, 0], y[:, None] - points[None, sl, 1]
            distance = np.hypot(de, dn)
            if distance.max() > grid.r[-1]:
                raise ValueError(f"a receiver lies {distance.max() / 1e3:.0f} km from a source; the tables stop at "
                                 f"{grid.r[-1] / 1e3:.0f} km")
            azimuth = np.arctan2(de, dn)
            i0, i1, j0, j1, ti, tj = _stencil(grid, distance, np.broadcast_to(depth[None, sl], distance.shape))
            w00, w10, w01, w11 = (1 - ti) * (1 - tj), ti * (1 - tj), (1 - ti) * tj, ti * tj
            tab = {(kind, comp): w00 * values[kind][comp][i0, j0] + w10 * values[kind][comp][i1, j0]
                   + w01 * values[kind][comp][i0, j1] + w11 * values[kind][comp][i1, j1]
                   for kind, comps in COMPONENTS.items() for comp in comps}
            c1, s1, c2, s2 = np.cos(azimuth), np.sin(azimuth), np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)
            block = weights[sl]
            for half in (0, 1):
                m1, m2, m3, m4, m5 = (moments[half, i, sl][None, :] for i in range(5))
                ps_ss, sh_ss = m1 * s2 + m4 * c2, m1 * c2 - m4 * s2
                ps_ds, sh_ds = m2 * c1 + m5 * s1, m2 * s1 - m5 * c1
                uz = ps_ss * tab["ss", "uz"] + ps_ds * tab["ds", "uz"] + m3 * tab["cl", "uz"]
                ur = ps_ss * tab["ss", "ur"] + ps_ds * tab["ds", "ur"] + m3 * tab["cl", "ur"]
                ut = sh_ss * tab["ss", "ut"] + sh_ds * tab["ds", "ut"]
                east, north, up = ur * s1 + ut * c1, ur * c1 - ut * s1, -uz
                cols = slice(half * n_params, (half + 1) * n_params)
                if look is not None:
                    los = east * look[:, 0:1] + north * look[:, 1:2] + up * look[:, 2:3]
                    out[:, cols] += (block.T @ los.T).T
                else:
                    for c, comp in enumerate((east, north, up)):
                        out[:, c, cols] += (block.T @ comp.T).T
        if projected:
            from .inversion import _blocked

            p = _blocked(basis.projection())
            out = out @ p if look is not None else np.einsum("pce,ek->pck", out, p.toarray())
        return out

    # -- the engine protocol ----------------------------------------------------------------------------------------

    def _base_kwargs(self, basis):
        return {"basis": basis} if getattr(self.base, "supports_basis", False) else {}

    def _base_projected(self, g, basis, look):
        """The base engine's element columns in `basis`, when it cannot assemble in it."""
        if getattr(self.base, "supports_basis", False) or basis is None or basis.name == "element":
            return g
        from .inversion import _blocked

        p = _blocked(basis.projection())
        return g @ p if look else np.einsum("pce,ek->pck", g, p.toarray())

    @staticmethod
    def _surface_only(z):
        if z is not None and np.any(np.asarray(z, dtype=float) != 0.0):
            raise ValueError("the curvature tables hold surface receivers only")

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, basis=None, elevation=None, step=None):
        """Line-of-sight design matrix: the base engine's plus the curvature's, `(npts, 2 n_parameters)`.

        `elevation` and `step` are for a base engine with its own topography (`native_topography`); with any other
        the inversion lowers the fault itself and calls this once per level.
        """
        self._surface_only(z)
        look = (look_e, look_n, look_u)
        if elevation is not None and np.any(np.asarray(elevation, dtype=float)):
            if not self.native_topography:
                raise ValueError("pass elevations to `slipsolve.inversion.assemble_design`, which lowers the fault")
            from .inversion import TOPOGRAPHY_STEP, _by_levels

            g = self.base.los_matrix(mesh, x, y, *look, z=z, elevation=elevation, step=step, **self._base_kwargs(basis))
            x_, y_ = (np.asarray(v, dtype=float).ravel() for v in (x, y))
            look_ = [np.asarray(v, dtype=float).ravel() for v in look]
            return self._base_projected(g, basis, True) + _by_levels(elevation, TOPOGRAPHY_STEP if step is None else step, x_.size,
                                  lambda depth, rows: self.correction(mesh.lowered(depth), x_[rows], y_[rows],
                                                                      [v[rows] for v in look_], basis=basis))
        g = self.base.los_matrix(mesh, x, y, *look, z=z, **self._base_kwargs(basis))
        return self._base_projected(g, basis, True) + self.correction(mesh, x, y, look, basis=basis)

    def displacement(self, mesh, x, y, z=None, basis=None):
        """East/north/up per unit slip, `(npts, 3, 2 n_parameters)`: the base engine's plus the curvature's."""
        self._surface_only(z)
        u = self.base.displacement(mesh, x, y, z=z, **self._base_kwargs(basis))
        return self._base_projected(u, basis, False) + self.correction(mesh, x, y, basis=basis)

    def forward(self, mesh, slip, x, y, look=None, z=None, basis=None):
        """Displacement of `slip` (`2 n_parameters`, or `(n, 2)`): east/north/up `(npts, 3)`, or line of sight with
        `look`."""
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        if look is None:
            return self.displacement(mesh, x, y, z=z, basis=basis) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z, basis=basis) @ slip

    def __repr__(self):
        return f"<Curvature {self.tables!r} on {self.base!r}>"
