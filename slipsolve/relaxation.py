"""Viscoelastic relaxation of a layered half-space after a step dislocation.

The tables a postseismic flow needs (`RelaxationTables`) are computed here by the correspondence principle. In the
Laplace domain a linear viscoelastic body answers a step dislocation as the elastic body would, with each shear modulus
replaced by its transform `mu(s)`, the whole divided by `s` (Lee 1955; for the layered Earth, Peltier 1974 and Pollitz
1992). `slipsolve.wavenumber`'s propagator gives the elastic answer at complex moduli as it does at real ones, and
Talbot's method (Talbot 1979, on the fixed contour of Abate & Valkó 2004) brings it back to time.

- Each layer is elastic, Maxwell, a Standard Linear Solid or Burgers in shear (`Rheology`); the bulk modulus stays
  elastic. Sources lie in elastic layers.
- What is inverted is the postseismic part alone, the response less the coseismic, for each output time on its own
  contour. So the times need not be uniform, and the half-space part of the source's field, which does not change with
  time, never enters.
- **Gravity** (on by default, `gravity=` in m/s^2): the layers are hydrostatically prestressed, so buoyancy resists
  vertical motion where density steps, at the surface and between layers. This is the reduced viscoelastic-gravitational
  problem of Wang, Lorenzo-Martín & Roth (2006, their eq. 10, after Rundle 1982): buoyancy with an adiabatic density
  gradient `rho^2 g / kappa` inside each layer, which keeps a compressible layer neutrally stable, and no
  self-gravitation (a correction of order `2 pi G rho / (g k)`: 0.2% of the buoyancy at 100 km wavelength). It matters
  for dip-slip relaxation over long times; the strike-slip field barely moves.

The rest does not depend on where the tables came from:

- `RelaxationTables.modes` splits their time dependence into a few functions with an SVD;
- `relaxation_history` and `relaxation_pattern` integrate them over a solved slip model's own mesh and basis, so the
  relaxation goes into the postseismic mode as a `slipsolve.patterns.Pattern` with one amplitude.

**Accuracy.** Against Savage & Prescott's (1978) solution for a long strike-slip fault in an elastic layer over a Maxwell
half-space, and against the fully relaxed state of a Standard Linear Solid computed elastically, as the tests check;
gravity's layer solutions satisfy the layer equations to rounding, and vanish into the elastic ones with `g`.
Talbot's method with the default 16 terms is good to about 1e-9 of each table; the tables' distance and depth spacing
limit the integration over a mesh, as for any point-source tables.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import wavenumber as wn
from .edgrn import EdgrnTables
from .layered import _moment_components, _slip_directions, _tent, _triangle_quadrature
from .patterns import Pattern
from .temporal import TimeBasis

__all__ = ["Rheology", "RelaxationTables", "shear_modulus", "relaxation_history", "relaxation_pattern"]

#: The tables a shear dislocation needs, as `slipsolve.edgrn` names them: no explosion, and no tangential CLVD.
COMPONENTS = {"ss": ("uz", "ur", "ut"), "ds": ("uz", "ur", "ut"), "cl": ("uz", "ur")}
SECONDS_PER_DAY = 86400.0
#: Terms of Talbot's method (Abate & Valkó's fixed contour); about 0.6 significant digits each, until rounding.
TALBOT_TERMS = 16
#: Wavenumbers (1/m) and their number for the relaxation tables: the flow is far from the receivers, so a shorter
#: range than the elastic tables' serves, ten decades about every distance used.
RELAXATION_K_RANGE = (1e-10, 1.0)
RELAXATION_N_K = 2048
#: How many Laplace variables one propagator solve carries at once (each a stretch of `n_k` wavenumbers).
_LAPLACE_CHUNK = 4
#: The acceleration of gravity in the buoyancy terms (m/s^2), as Wang et al. (2006) take it.
GRAVITY = 9.82


class Rheology:
    """The rheology of one depth range, in metres from `top` to `bottom` (`None`: to the bottom of the half-space).

    A Burgers body: a Maxwell element (the layer's own shear modulus with `eta_maxwell`) in series with a Kelvin
    element (`eta_kelvin`, and a modulus set by `alpha`). `None` is an infinite viscosity, so:

    - both `None` is elastic;
    - `eta_kelvin=None` (or `alpha=1`) is Maxwell;
    - `eta_maxwell=None` is a Standard Linear Solid.

    `alpha` is the fully relaxed over the unrelaxed modulus of the Kelvin part, `mu_K / (mu_K + mu)` (as in Wang et al.
    2006).
    A Standard Linear Solid relaxes stress with the time `eta_kelvin (1 - alpha) / mu` and creeps with
    `eta_kelvin (1 - alpha) / (alpha mu)`. A Maxwell body relaxes stress with `eta_maxwell / mu`. The bulk modulus
    stays elastic.
    """

    def __init__(self, top, bottom=None, eta_maxwell=None, eta_kelvin=None, alpha=1.0):
        self.top = float(top)
        self.bottom = None if bottom is None else float(bottom)
        if self.bottom is not None and not self.bottom > self.top:
            raise ValueError(f"a rheology's bottom ({self.bottom}) must lie below its top ({self.top})")
        for label, value in (("eta_maxwell", eta_maxwell), ("eta_kelvin", eta_kelvin)):
            if value is not None and not float(value) > 0:
                raise ValueError(f"{label} must be positive, or None for infinite")
        self.eta_maxwell = None if eta_maxwell is None else float(eta_maxwell)
        self.eta_kelvin = None if eta_kelvin is None else float(eta_kelvin)
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def contains(self, depth):
        """Whether `depth` lies in `[top, bottom)`."""
        return depth >= self.top and (self.bottom is None or depth < self.bottom)

    def _key(self):
        """`(eta_kelvin, eta_maxwell, alpha)`, with 0 for infinite: equal keys, equal rheologies."""
        return (0.0 if self.eta_kelvin is None else self.eta_kelvin,
                0.0 if self.eta_maxwell is None else self.eta_maxwell,
                self.alpha)

    def scaled(self, factor):
        """The same with every viscosity times `factor`."""
        return Rheology(self.top, self.bottom,
                        None if self.eta_maxwell is None else self.eta_maxwell * factor,
                        None if self.eta_kelvin is None else self.eta_kelvin * factor, self.alpha)

    def __repr__(self):
        bottom = "" if self.bottom is None else f"-{self.bottom / 1e3:g}"
        return (f"<Rheology {self.top / 1e3:g}{bottom} km maxwell={self.eta_maxwell} kelvin={self.eta_kelvin} "
                f"alpha={self.alpha:g}>")



def _constant_layers(model):
    """`[(top, vp, vs, rho)]` of a piecewise-constant velocity model, the last row the half-space's top."""
    depth, vp, vs, rho = (np.asarray(a, dtype=float) for a in (model.depth, model.vp, model.vs, model.rho))
    if depth[0] > 0:
        raise ValueError("the velocity model must start at the surface")
    layers = []
    for i in range(depth.size - 1):
        if depth[i + 1] == depth[i]:
            continue
        if not np.allclose([vp[i], vs[i], rho[i]], [vp[i + 1], vs[i + 1], rho[i + 1]], rtol=1e-9):
            raise NotImplementedError(f"the model has a gradient between {depth[i]:.0f} and {depth[i + 1]:.0f} m; "
                                      "give it as constant layers")
        layers.append((depth[i], vp[i], vs[i], rho[i]))
    layers.append((depth[-1], vp[-1], vs[-1], rho[-1]))
    merged = [layers[0]]
    for row in layers[1:]:
        if np.allclose(row[1:], merged[-1][1:], rtol=1e-9):
            continue
        merged.append(row)
    return merged




def shear_modulus(rheology, mu, s):
    """A layer's shear modulus in the Laplace domain, `mu(s)`, for its `Rheology` (`None`: elastic).

    A Burgers body is a Maxwell element (`mu` with `eta_maxwell`) in series with a Kelvin element (`mu_K` with
    `eta_kelvin`), so its compliance is `1 / mu + 1 / (eta_maxwell s) + 1 / (mu_K + eta_kelvin s)`, with
    `mu_K = alpha mu / (1 - alpha)`. An element drops out at an infinite viscosity (`None`), and the Kelvin one at
    `alpha = 1`: Maxwell is `mu s / (s + mu / eta_maxwell)`, the Standard Linear Solid
    `mu (mu_K + eta_kelvin s) / (mu + mu_K + eta_kelvin s)`, relaxed to `alpha mu`.
    """
    s = np.asarray(s, dtype=complex)
    compliance = np.full(s.shape, 1.0 / float(mu), dtype=complex)
    if rheology is not None:
        if rheology.eta_maxwell is not None:
            compliance = compliance + 1.0 / (rheology.eta_maxwell * s)
        if rheology.eta_kelvin is not None and rheology.alpha < 1.0:
            mu_k = rheology.alpha * float(mu) / (1.0 - rheology.alpha)
            compliance = compliance + 1.0 / (mu_k + rheology.eta_kelvin * s)
    return 1.0 / compliance


def _talbot(t, terms=TALBOT_TERMS):
    """Nodes and weights of Talbot's method for one time `t > 0` (s): `f(t) = Re(sum(weights * F(nodes)))`.

    Abate & Valkó's (2004) fixed contour, `s(theta) = r theta (cot theta + i)` with `r = 2 M / (5 t)`.
    """
    m = int(terms)
    r = 2.0 * m / (5.0 * float(t))
    theta = np.arange(1, m) * np.pi / m
    cot = np.cos(theta) / np.sin(theta)
    nodes = np.concatenate([[r], r * theta * (cot + 1j)])
    sigma = theta + (theta * cot - 1.0) * cot
    weights = np.concatenate([[0.5 * np.exp(r * t)], np.exp(t * nodes[1:]) * (1.0 + 1j * sigma)]) * (r / m)
    return nodes, weights


def _elastic(rheology):
    return rheology is None or (rheology.eta_maxwell is None and (rheology.eta_kelvin is None or rheology.alpha >= 1.0))


def _layers(model, rheology=()):
    """`(tops, lam, mu, rho, rheologies)`: constant layers, cut wherever an elastic property or the rheology changes.

    The last is the half-space. A rheology is `None` where the layer is elastic.
    """
    rheology = sorted(rheology, key=lambda r: r.top)
    for a, b in zip(rheology, rheology[1:]):
        if a.bottom is None or a.bottom > b.top:
            raise ValueError(f"rheologies overlap: {a} and {b}")
    layers = _constant_layers(model)
    cuts = {top for top, *_ in layers}
    for r in rheology:
        cuts.add(r.top)
        if r.bottom is not None:
            cuts.add(r.bottom)
    cuts = sorted(c for c in cuts if c >= 0.0)
    if cuts[0] != 0.0:
        cuts = [0.0] + cuts
    tops, lams, mus, rhos, flows = [], [], [], [], []
    for top in cuts:
        _, vp, vs, rho = [layer for layer in layers if layer[0] <= top][-1]
        mu = rho * vs ** 2
        lam = rho * vp ** 2 - 2.0 * mu
        flow = next((r for r in rheology if r.contains(top)), None)
        flow = None if _elastic(flow) else flow
        same = (tops and np.isclose(lams[-1], lam, rtol=1e-12) and np.isclose(mus[-1], mu, rtol=1e-12)
                and np.isclose(rhos[-1], rho, rtol=1e-12)
                and (flows[-1] is flow or (flows[-1] is not None and flow is not None
                                           and flows[-1]._key() == flow._key())))
        if same:
            continue
        tops.append(top)
        lams.append(lam)
        mus.append(mu)
        rhos.append(rho)
        flows.append(flow)
    return np.array(tops), np.array(lams), np.array(mus), np.array(rhos), flows


def _one_depth(task):
    """One source depth's coseismic `(nr, 8)` and postseismic `(nt, nr, 8)` rows, in `COMPONENTS` order."""
    tops, lam, mu, rho, flows, gravity, z_source, seconds, r, k_range, n_k, terms = task
    k, dln = wn.wavenumbers(k_range, n_k)
    j = int(wn._layer_index(tops, z_source))
    psv, sh = wn.source_jumps(lam[j], mu[j])
    surface = np.array([0.0])
    buoyancy = rho * float(gravity) if gravity else None
    elastic = wn._Solution(k, tops, lam, mu, z_source, psv, sh, buoyancy=buoyancy).at(surface)
    half = wn._Solution(k, tops[:1], lam[j:j + 1], mu[j:j + 1], z_source, psv, sh).at(surface)
    axis = r == 0.0

    def to_space(U, V, W):
        out = wn.transform(U, V, W, k, dln, r)[:, :8]
        out[axis] = wn.on_axis(U[None], V[None], W[None], k, dln)[0, :8]
        return out

    coseismic = to_space(*(a[0] - b[0] for a, b in zip(elastic, half)))
    bulk = lam + 2.0 * mu / 3.0
    post = np.zeros((seconds.size, r.size, 8))
    for i, t in enumerate(seconds):
        if not t > 0.0:
            continue
        nodes, weights = _talbot(t, terms)
        total = [np.zeros((k.size, len(wn.SOURCE_TYPES))) for _ in range(3)]
        for start in range(0, nodes.size, _LAPLACE_CHUNK):
            s = nodes[start:start + _LAPLACE_CHUNK]
            w = weights[start:start + _LAPLACE_CHUNK] / s
            mu_s = np.array([shear_modulus(flow, m, s) for flow, m in zip(flows, mu)])        # (n_layers, n_s)
            lam_s = bulk[:, None] - 2.0 * mu_s / 3.0
            fields = wn._Solution(np.tile(k, s.size), tops, np.repeat(lam_s, k.size, axis=1),
                                  np.repeat(mu_s, k.size, axis=1), z_source, psv, sh, buoyancy=buoyancy).at(surface)
            for acc, field, still in zip(total, fields, elastic):
                change = field[0].reshape(s.size, k.size, -1) - still[0][None]
                acc += np.real(np.einsum("j,jkc->kc", w, change))
        post[i] = to_space(*total)
    return coseismic, post


class RelaxationTables:
    """Displacement tables of a layered viscoelastic half-space, on a uniform `(r, z)` grid at `times` (days, the first 0).

    `coseismic[kind][component]` is `(nr, nz)`: the elastic response less the source layer's half-space; the relaxation
    uses only the postseismic part.
    `postseismic[kind][component]` is `(nt, nr, nz)`: the response at each time less the coseismic, 0 at `times[0]`.
    Kinds are `ss`, `ds`, `cl`. Components are `uz` (positive down), `ur` and `ut`, as `slipsolve.edgrn.EdgrnTables`
    has them: per unit potency, the azimuthal factor stripped. `interfaces` are the depths no interpolation crosses.
    """

    def __init__(self, r, z, times, coseismic, postseismic, interfaces=(), attrs=None):
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.times = np.asarray(times, dtype=float)
        if self.r.size < 2 or self.z.size < 2:
            raise ValueError("the tables need at least two distances and two source depths")
        self.coseismic = coseismic
        self.postseismic = postseismic
        self.interfaces = np.unique(np.asarray(interfaces, dtype=float))
        self.attrs = dict(attrs or {})
        self._grid = EdgrnTables(self.r, self.z, self._edgrn_dict(lambda kind, comp: coseismic[kind][comp]),
                                 interfaces=self.interfaces)

    @classmethod
    def compute(cls, model, rheology=(), *, r_max=300e3, nr=301, z_min=500.0, z_max=29500.0, nz=30, times=None,
                window_days=365.25, nt=48, first_day=None, terms=TALBOT_TERMS, k_range=RELAXATION_K_RANGE,
                n_k=RELAXATION_N_K, gravity=GRAVITY, workers=None):
        """Our own tables for a `slipsolve.edgrn.VelocityModel` with a `Rheology` per depth range.

        - Distances are uniform from 0 to `r_max` (`nr`) and source depths from `z_min` to `z_max` (`nz`), all in
          elastic layers and off every interface, receivers at the surface.
        - `times` (days) are the output times. By default 0, then `nt - 1` log-spaced from `first_day` (default
          `window_days / 1000`) to `window_days`; any times work, each is inverted alone.
        - `gravity` (m/s^2) is the acceleration of the buoyancy terms; 0 leaves them out.
        - `terms` is the number of Talbot terms; `workers` runs the source depths in that many processes.

        Depths outside every `Rheology` are elastic. The tables are laid out as Wang et al.'s (2006).
        """
        tops, lam, mu, rho, flows = _layers(model, rheology)
        r = np.linspace(0.0, float(r_max), int(nr))
        z = np.linspace(float(z_min), float(z_max), int(nz))
        for depth in z:
            j = int(wn._layer_index(tops, depth))
            if flows[j] is not None:
                raise ValueError(f"a source at {depth:.0f} m lies in a viscous layer; sources must be in elastic ones")
            if np.any(np.isclose(tops[1:], depth, rtol=0.0, atol=1e-6)):
                raise ValueError(f"a source at {depth:.0f} m sits on a layer interface")
        if times is None:
            first = float(window_days) / 1e3 if first_day is None else float(first_day)
            times = np.concatenate([[0.0], np.geomspace(first, float(window_days), int(nt) - 1)])
        times = np.asarray(times, dtype=float)
        if times[0] != 0.0:
            times = np.concatenate([[0.0], times])
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must increase")
        tasks = [(tops, lam, mu, rho, flows, float(gravity), depth, times * SECONDS_PER_DAY, r, tuple(k_range), int(n_k),
                  int(terms)) for depth in z]
        if workers and int(workers) > 1:
            with ProcessPoolExecutor(int(workers)) as pool:
                rows = list(pool.map(_one_depth, tasks))
        else:
            rows = [_one_depth(task) for task in tasks]
        keys = [(kind, comp) for kind, comps in COMPONENTS.items() for comp in comps]
        coseismic = {kind: {} for kind in COMPONENTS}
        postseismic = {kind: {} for kind in COMPONENTS}
        for c, (kind, comp) in enumerate(keys):
            coseismic[kind][comp] = np.stack([row[0][:, c] for row in rows], axis=-1)
            postseismic[kind][comp] = np.stack([row[1][:, :, c] for row in rows], axis=-1)
        attrs = {"source": "slipsolve.relaxation", "rheology": [repr(f) for f in rheology], "terms": int(terms),
                 "k_range": list(k_range), "n_k": int(n_k), "gravity": float(gravity)}
        return cls(r, z, times, coseismic, postseismic, interfaces=tops[1:], attrs=attrs)

    def _edgrn_dict(self, values):
        out = {}
        for kind, comps in COMPONENTS.items():
            out[kind] = {comp: values(kind, comp) for comp in comps}
            if "ut" not in out[kind]:
                out[kind]["ut"] = np.zeros((self.r.size, self.z.size))
        return out

    def rescaled(self, factor):
        """The tables of the same structure with every viscosity times `factor`: the times scale with it."""
        return type(self)(self.r, self.z, self.times * float(factor), self.coseismic, self.postseismic,
                           self.interfaces, dict(self.attrs, viscosity_factor=self.attrs.get("viscosity_factor", 1.0) * factor))

    def at(self, day):
        """The postseismic part at `day` as `slipsolve.edgrn.EdgrnTables`, linear between the stored times."""
        k = int(np.clip(np.searchsorted(self.times, day), 1, self.times.size - 1))
        t0, t1 = self.times[k - 1], self.times[k]
        w = 0.0 if not np.isfinite(t1) else float(np.clip((day - t0) / (t1 - t0), 0.0, 1.0))
        return EdgrnTables(self.r, self.z,
                           self._edgrn_dict(lambda kind, comp: (1 - w) * self.postseismic[kind][comp][k - 1]
                                            + w * self.postseismic[kind][comp][k]),
                           interfaces=self.interfaces)

    def modes(self, tolerance=1e-4, max_modes=16, until=None):
        """`(times, functions, stacks)`: the postseismic tables as a few functions of time times a few tables.

        An SVD over time of every table, each first scaled by its own largest value. It keeps the fewest modes
        that bring every table back to `tolerance` of its largest value, up to `max_modes`.

        - `functions`: `(n_times, M)`.
        - `stacks[kind][component]`: `(nr, nz, M)`.
        - `until` (days): only times up to the first sample at or past it.
        """
        times = self.times
        keep = times.size if until is None else int(min(np.searchsorted(times, float(until)) + 1, times.size))
        times = times[:keep]
        if times.size < 2 or not np.isfinite(times).all():
            raise ValueError("modes need a time series (nt > 2)")
        keys = [(kind, comp) for kind, comps in COMPONENTS.items() for comp in comps]
        scales = {key: max(float(np.abs(self.postseismic[key[0]][key[1]][:keep]).max()), 1e-300) for key in keys}
        blocks = [self.postseismic[k][c][:keep].reshape(keep, -1) / scales[(k, c)] for k, c in keys]
        history = np.hstack(blocks)
        values, vectors = np.linalg.eigh(history @ history.T)
        order = np.argsort(values)[::-1]
        values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
        sizes = [b.shape[1] for b in blocks]
        chosen = None
        for m in range(1, min(int(max_modes), keep) + 1):
            u = vectors[:, :m]
            residual = history - u @ (u.T @ history)
            worst = max(float(np.abs(part).max()) for part in np.split(residual, np.cumsum(sizes)[:-1], axis=1))
            if worst <= tolerance:
                chosen = m
                break
        if chosen is None:
            chosen = min(int(max_modes), keep)
        u = vectors[:, :chosen]
        spatial = u.T @ history                        # (M, N): the functions are u itself, unit-normed
        stacks, start = {kind: {} for kind in COMPONENTS}, 0
        for (kind, comp), size in zip(keys, sizes):
            part = spatial[:, start:start + size] * scales[(kind, comp)]
            stacks[kind][comp] = np.moveaxis(part.reshape(chosen, self.r.size, self.z.size), 0, -1)
            start += size
        return times, u, stacks

    def __repr__(self):
        return (f"<{type(self).__name__} nr={self.r.size} r=0-{self.r[-1] / 1e3:.0f}km nz={self.z.size} "
                f"z={self.z[0] / 1e3:.2f}-{self.z[-1] / 1e3:.2f}km nt={self.times.size} "
                f"to {self.times[-1]:.4g} d interfaces={self.interfaces.size}>")


class _Sources:
    """A slip model's point sources: positions (ENU), area weights, and five packed moment components with slip."""

    def __init__(self, mesh, basis, strike_slip, dip_slip, order):
        points, weights, moments = [], [], []
        strike_slip = np.asarray(strike_slip, dtype=float)
        dip_slip = np.asarray(dip_slip, dtype=float)
        nodal = basis is not None and basis.name == "node"
        for e in range(mesh.n_elements):
            p1, p2, p3 = mesh.vertices[e]
            normal = mesh.normals[e]
            pts, share = _triangle_quadrature(p1, p2, p3, int(order))
            if nodal:
                triangle = mesh.triangles[e]
                tents = [_tent(mesh.vertices[e], triangle, node, pts) for node in triangle]
                ss = sum(t * strike_slip[node] for t, node in zip(tents, triangle))
                ds = sum(t * dip_slip[node] for t, node in zip(tents, triangle))
            else:
                ss = np.full(len(pts), strike_slip[e])
                ds = np.full(len(pts), dip_slip[e])
            along, down = _slip_directions(normal)
            m_ss = np.asarray(_moment_components(normal, along))
            m_ds = np.asarray(_moment_components(normal, down))
            points.append(pts)
            weights.append(share * mesh.areas[e] * np.ones(len(pts)))
            moments.append(np.outer(m_ss, ss) + np.outer(m_ds, ds))       # (5, nq)
        self.points = np.vstack(points)
        self.weights = np.concatenate(weights)
        self.moments = np.hstack(moments)


def _stencil(grid, distance, depth):
    """`EdgrnTables.interpolate`'s bilinear stencil, never across an interface: `(i0, i1, j0, j1, ti, tj)`."""
    r, z = grid.r, grid.z
    nr, nz = r.size, z.size
    dr = (r[-1] - r[0]) / (nr - 1)
    dz = (z[-1] - z[0]) / (nz - 1)
    fi = np.minimum(np.maximum((distance - r[0]) / dr, 0.0), nr - 1)
    fj = np.minimum(np.maximum((depth - z[0]) / dz, 0.0), nz - 1)
    i0 = np.minimum(np.floor(fi).astype(int), nr - 2)
    j0 = np.minimum(np.floor(fj).astype(int), nz - 2)
    ti, tj = fi - i0, fj - j0
    i1, j1 = i0 + 1, j0 + 1
    if grid._layers is not None:
        j0, j1, tj = grid._within_layers(fj, dz, j0, j1, tj)
    return i0, i1, j0, j1, ti, tj


def _evaluate(sources, stacks, grid, x, y, budget=4e6):
    """East, north and up at `(x, y)` from `stacks[kind][component]`, `(nr, nz, K)`: `(n, 3, K)`."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    k = next(iter(stacks["ss"].values())).shape[-1]
    out = np.zeros((x.size, 3, k))
    r_max = grid.r[-1]
    z_lo, z_hi = grid.z[0], grid.z[-1]
    depth_all = -sources.points[:, 2]
    spacing = grid.z[1] - grid.z[0]
    # Within one spacing of the table's first or last depth a source takes that row, as EdgrnTables clamps.
    if depth_all.min() < z_lo - spacing or depth_all.max() > z_hi + spacing:
        raise ValueError(f"the sources lie at {depth_all.min() / 1e3:.2f}-{depth_all.max() / 1e3:.2f} km, the tables "
                         f"at {z_lo / 1e3:.2f}-{z_hi / 1e3:.2f} km")
    chunk = max(1, int(budget // max(1, x.size * k)))
    for start in range(0, sources.weights.size, chunk):
        sl = slice(start, start + chunk)
        pts, w, m = sources.points[sl], sources.weights[sl], sources.moments[:, sl]
        dn = y[:, None] - pts[None, :, 1]
        de = x[:, None] - pts[None, :, 0]
        distance = np.hypot(dn, de)
        if distance.max() > r_max:
            raise ValueError(f"a receiver lies {distance.max() / 1e3:.1f} km from a source; the tables stop at "
                             f"{r_max / 1e3:.1f} km")
        depth = np.broadcast_to(-pts[None, :, 2], distance.shape)
        azimuth = np.arctan2(de, dn)
        cos1, sin1 = np.cos(azimuth), np.sin(azimuth)
        cos2, sin2 = np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)
        m1, m2, m3, m4, m5 = (row[None, :] for row in m)
        ps_ss = m1 * sin2 + m4 * cos2
        sh_ss = m1 * cos2 - m4 * sin2
        ps_ds = m2 * cos1 + m5 * sin1
        sh_ds = m2 * sin1 - m5 * cos1
        i0, i1, j0, j1, ti, tj = _stencil(grid, distance, depth)
        w00, w10 = ((1 - ti) * (1 - tj))[..., None], (ti * (1 - tj))[..., None]
        w01, w11 = ((1 - ti) * tj)[..., None], (ti * tj)[..., None]

        def look(kind, comp):
            a = stacks[kind][comp]
            return w00 * a[i0, j0] + w10 * a[i1, j0] + w01 * a[i0, j1] + w11 * a[i1, j1]

        uz = ps_ss[..., None] * look("ss", "uz") + ps_ds[..., None] * look("ds", "uz") + m3[..., None] * look("cl", "uz")
        ur = ps_ss[..., None] * look("ss", "ur") + ps_ds[..., None] * look("ds", "ur") + m3[..., None] * look("cl", "ur")
        ut = sh_ss[..., None] * look("ss", "ut") + sh_ds[..., None] * look("ds", "ut")
        north = ur * cos1[..., None] - ut * sin1[..., None]
        east = ur * sin1[..., None] + ut * cos1[..., None]
        weight = w[None, :, None]
        out[:, 0] += (east * weight).sum(axis=1)
        out[:, 1] += (north * weight).sum(axis=1)
        out[:, 2] += (-uz * weight).sum(axis=1)
    return out


def _model_parts(model, frame):
    mesh = model.mesh if frame is None else model.mesh.in_frame(frame)
    return mesh, model.basis, model.strike_slip, model.dip_slip


def relaxation_history(model, tables, x, y, order=4, frame=None):
    """`(times, enu)`: a slip model's postseismic displacement at `(x, y)` at every time of the tables.

    `enu` is `(n_times, n, 3)` (east, north, up), zero at `times[0]`. `model` is a solved
    `slipsolve.inversion.SlipModel`, its mesh moved into `frame` if given. Each triangle is cut into `order**2`
    point sources weighted by the slip there (its nodal tents on a nodal basis).
    """
    mesh, basis, ss, ds = _model_parts(model, frame)
    sources = _Sources(mesh, basis, ss, ds, order)
    stacks = {kind: {comp: np.moveaxis(tables.postseismic[kind][comp], 0, -1) for comp in comps}
              for kind, comps in COMPONENTS.items()}
    out = _evaluate(sources, stacks, tables._grid, x, y)
    return tables.times, np.moveaxis(out, -1, 0)


def relaxation_pattern(model, tables, name="viscoelastic", until=None, tolerance=1e-4, max_modes=16, order=4,
                       frame=None):
    """A `slipsolve.patterns.Pattern` of a slip model's relaxation: an amplitude of 1 is these tables' rheology.

    - The tables' time dependence is split into a few functions (`RelaxationTables.modes`, to `tolerance`); their shapes,
      integrated over the model's mesh, are the pattern's terms.
    - `until` (days) is the last date it will be asked about; past the tables' last time the pattern holds its last
      value.
    - `frame`: the inversion's local frame, if the model was solved in another.
    """
    mesh, basis, ss, ds = _model_parts(model, frame)
    if until is not None and tables.times[-1] < float(until):
        raise ValueError(f"the tables end at {tables.times[-1]:.4g} d, before {float(until):.4g} d")
    times, functions, stacks = tables.modes(tolerance=tolerance, max_modes=max_modes, until=until)
    sources = _Sources(mesh, basis, ss, ds, order)
    grid = tables._grid

    def enu(x, y):
        out = _evaluate(sources, stacks, grid, x, y)
        return out[:, 0], out[:, 1], out[:, 2]

    pattern = Pattern.from_enu(name, enu, time=TimeBasis.tabulated(times, functions))
    pattern.frame = mesh.frame
    return pattern
