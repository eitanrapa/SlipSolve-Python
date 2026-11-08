"""Viscoelastic relaxation of a layered half-space, from PSGRN.

PSGRN (Wang, Lorenzo-Martin & Roth 2006, Computers & Geosciences 32, 527-541) computes the response of point sources
in a layered viscoelastic-gravitational half-space at a set of times after a step dislocation. Each layer can be
elastic, Maxwell, a Standard Linear Solid or Burgers, in shear only. It runs here as a separate program, through
files. This module:

- writes its input from a `slipsolve.edgrn.VelocityModel` and a `Rheology` per depth range (`write_psgrn_input`,
  `run_psgrn`);
- reads its displacement tables (`PsgrnTables`);
- splits their time dependence into a few functions with an SVD (`PsgrnTables.modes`);
- integrates each part over a solved slip model's own mesh and basis (`relaxation_pattern`), so the relaxation goes
  into the postseismic mode as a `slipsolve.patterns.Pattern` with one amplitude.

**What is used.** For t > 0, PSGRN's tables hold the postseismic part, coseismic subtracted. These samples are the
complete field: the half-space solution PSGRN subtracts from its coseismic tables, and PSCMP adds back, is elastic
and does not change with time. So the relaxation needs no near-field add-back, and point-source quadrature over a
triangle is accurate wherever the flow is below the receivers' layer.

**Use a corrected PSGRN.** The 2020 release (github.com/RongjiangWang/PSGRN-PSCMP_2020) has three problems in its
time domain:

1. a Standard Linear Solid layer (Maxwell viscosity "infinite", written as 0) sets the shortest relaxation time to
   zero, and with it the number of frequencies to the minimum, whatever the number of output times;
2. the step response's trapezoid sum undamps both samples at the later time;
3. the response switches on at t = 0, where the inverse FFT returns the midpoint of the jump, so the first panel
   takes half the initial rate.

Together they put a Standard Linear Solid's series 2-11% off the correspondence-principle value. Corrected, the series
of all three rheologies agree with it to 0.2-1.5%, and the elastic tables agree with an independent layered code to
3-7e-4 of the peak. Pyrocko's older copy (version 2008a) also blurs every table over 10% of the source-receiver
distance.

**Choosing the grids.**

- Distances are uniform from 0 to `r_max`. Source depths are uniform; `write_psgrn_input` asks for them as PSGRN2020
  wants (cell edges), and `PsgrnTables` reads them back from the tables. PSGRN shifts them if they would reach the
  receiver depth.
- Keep source depths off layer interfaces: `PsgrnTables` never interpolates across one.
- Times are uniform, `nt` over a window in days, as PSGRN's FFT needs. Take the step at about a twentieth of the
  earliest time that matters, and run early postseismic in its own short window: the first few samples of any run
  are the least accurate.

PSGRN's time scales with a uniform viscosity: multiply every viscosity by `c` and every time by `c`. So a scan over a
uniform level needs one run per shape, read at rescaled times (`PsgrnTables.rescaled`).
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .edgrn import EdgrnTables
from .layered import _moment_components, _slip_directions, _tent, _triangle_quadrature
from .patterns import Pattern
from .temporal import TimeBasis

__all__ = ["Rheology", "PsgrnTables", "psgrn_layers", "write_psgrn_input", "run_psgrn", "find_psgrn",
           "relaxation_history", "relaxation_pattern"]

#: The tables a shear dislocation needs, as `slipsolve.edgrn` names them: no explosion, and no tangential CLVD.
COMPONENTS = {"ss": ("uz", "ur", "ut"), "ds": ("uz", "ur", "ut"), "cl": ("uz", "ur")}
#: PSGRN's output names for the three displacement, six stress and five other fields; only the first three are read.
OUTPUT_NAMES = (("uz", "ur", "ut"), ("szz", "srr", "stt", "szr", "srt", "stz"), ("tr", "tt", "rot", "gd", "gr"))
#: Environment variable naming a PSGRN executable, checked after `executable=`.
ENVIRONMENT = "SLIPSOLVE_PSGRN"
SECONDS_PER_DAY = 86400.0


class Rheology:
    """The rheology of one depth range, in metres from `top` to `bottom` (`None`: to the bottom of the half-space).

    A Burgers body: a Maxwell element (the layer's own shear modulus with `eta_maxwell`) in series with a Kelvin
    element (`eta_kelvin`, and a modulus set by `alpha`). `None` is an infinite viscosity, so:

    - both `None` is elastic;
    - `eta_kelvin=None` (or `alpha=1`) is Maxwell;
    - `eta_maxwell=None` is a Standard Linear Solid.

    `alpha` is the fully relaxed over the unrelaxed modulus of the Kelvin part, `mu_K / (mu_K + mu)` in PSGRN's terms.
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

    def psgrn(self):
        """PSGRN's `eta1 eta2 alpha` (Kelvin, Maxwell, alpha), with 0 for infinite."""
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


_ELASTIC = (0.0, 0.0, 1.0)


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


def psgrn_layers(model, rheology=()):
    """PSGRN's model lines, `(depth km, vp km/s, vs km/s, rho, eta1, eta2, alpha)`, from a velocity model and rheologies.

    Every depth where an elastic property or the rheology changes becomes an interface, written as two lines (upper and
    lower side), and the half-space's top as one. Depths outside every `Rheology` are elastic. Rheologies must not
    overlap.
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

    def properties(z):
        row = [layer for layer in layers if layer[0] <= z][-1]
        return tuple(row[1:])

    def flow(z):
        for r in rheology:
            if r.contains(z):
                return r.psgrn()
        return _ELASTIC

    intervals = []
    for top, bottom in zip(cuts, cuts[1:] + [None]):
        key = properties(top) + flow(top)
        if intervals and intervals[-1][2] == key:
            intervals[-1][1] = bottom
        else:
            intervals.append([top, bottom, key])
    rows = []
    for top, bottom, (vp, vs, rho, eta1, eta2, alpha) in intervals:
        line = (vp / 1e3, vs / 1e3, rho, eta1, eta2, alpha)
        rows.append((top / 1e3,) + line)
        if bottom is not None:
            rows.append((bottom / 1e3,) + line)
    return rows


def write_psgrn_input(path, model, rheology=(), *, r_max=300e3, nr=301, z_min=500.0, z_max=29500.0, nz=30,
                      nt=512, window_days=365.25, accuracy=0.025, gravity=1.0, receiver_depth=0.0):
    """Write a PSGRN input file for `model` with `rheology`, receivers at `receiver_depth` (m).

    - Distances: `nr` uniformly from 0 to `r_max`.
    - Source depths: `nz` uniformly from `z_min` to `z_max`. PSGRN2020 reads its two depths as the edges of `nz` equal
      cells and puts the sources at their centres, so the file gets `z_min - dz/2` and `z_max + dz/2`.
    - Times: `nt` over `window_days`. `nt = 1` gives the coseismic tables alone; `nt = 2` gives the coseismic and the
      fully relaxed state.
    - `gravity` scales gravity's effect, from 0 (none) to 1 (full).

    Lengths are metres here and kilometres in the file. Returns the path.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = psgrn_layers(model, rheology)
    nz = int(nz)
    dz = (float(z_max) - float(z_min)) / (nz - 1) if nz > 1 else 0.0
    if nz > 1 and z_min - 0.5 * dz < 0.0:
        raise ValueError(f"the first source depth must be at least half a spacing ({0.5 * dz:.1f} m) below the surface")
    lines = ["# PSGRN input written by slipsolve.psgrn",
             f" {receiver_depth / 1e3:.6f} 1",
             f" {int(nr)} 0.0 {r_max / 1e3:.6f} 1.0",
             f" {nz} {(z_min - 0.5 * dz) / 1e3:.6f} {(z_max + 0.5 * dz) / 1e3:.6f}",
             f" {int(nt)} {float(window_days):.6f}",
             f" {float(accuracy):.6f}",
             f" {float(gravity):.6f}",
             " './'"]
    for group in OUTPUT_NAMES:
        lines.append(" " + " ".join(f"'{name}'" for name in group))
    lines.append(f" {len(rows)}")
    for i, (d, vp, vs, rho, eta1, eta2, alpha) in enumerate(rows, 1):
        lines.append(f" {i:4d} {d:12.6f} {vp:10.6f} {vs:10.6f} {rho:10.3f} {eta1:13.6E} {eta2:13.6E} {alpha:8.5f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def find_psgrn(executable=None):
    """A PSGRN executable: `executable`, else `$SLIPSOLVE_PSGRN`, else `psgrn2020` or `psgrn` on `PATH`."""
    for candidate in (executable, os.environ.get(ENVIRONMENT), shutil.which("psgrn2020"), shutil.which("psgrn")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    raise RuntimeError(f"No PSGRN executable: pass executable=, set ${ENVIRONMENT}, or put psgrn on PATH. Build "
                       "github.com/RongjiangWang/PSGRN-PSCMP_2020 with the time-domain corrections described in "
                       "slipsolve.psgrn.")


def run_psgrn(model, rheology, workdir, executable=None, **grid):
    """Run PSGRN in `workdir` and read its tables back; `grid` as `write_psgrn_input`.

    If `workdir` already holds the tables of the same input, nothing is run. The executable and its hash go into
    the tables' `attrs`.
    """
    workdir = Path(workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "psgrn.inp"
    previous = target.read_text() if target.exists() else None
    scratch = write_psgrn_input(workdir / "psgrn.inp.new", model, rheology, **grid)
    text = scratch.read_text()
    scratch.unlink()
    binary = find_psgrn(executable)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()[:16]
    done = previous == text and all((workdir / f"{c}.{k}").exists() for k, comps in COMPONENTS.items() for c in comps)
    if not done:
        target.write_text(text)
        out = subprocess.run([str(binary)], input="psgrn.inp\n", text=True, cwd=workdir, capture_output=True)
        (workdir / "psgrn.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
        if out.returncode != 0:
            raise RuntimeError(f"PSGRN failed in {workdir} (exit {out.returncode}): {out.stderr[-2000:]}")
    tables = PsgrnTables.from_directory(workdir)
    tables.attrs.update(executable=str(binary), executable_sha256=digest)
    return tables


def _numbers(line):
    return [float(v) for v in line.replace("D", "E").replace("d", "e").split()]


def _read_table(path):
    """One PSGRN table file: `(r, zs, t, coseismic (nr, nz), postseismic (nt, nr, nz))`, t in days with t[0] = 0.

    The layout of `psgprocess.f`: four header lines (distances, receiver, depths, times), then per source depth its
    layer's parameters, the coseismic line, one decimal exponent per distance and `nt - 1` lines of integers.
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    nr, r1, r2, ratio = _numbers(lines[0])
    nr = int(nr)
    if abs(ratio - 1.0) > 1e-9:
        raise ValueError(f"{path}: distances must be uniform (sampling ratio 1), not {ratio}")
    nz = int(_numbers(lines[2])[0])
    nt_line = _numbers(lines[3])
    nt, window = int(nt_line[0]), nt_line[1] / SECONDS_PER_DAY
    r = np.linspace(r1, r2, nr) if nr > 1 else np.array([r1])
    t = np.arange(nt) * (window / (nt - 1)) if nt > 2 else (np.array([0.0, np.inf]) if nt == 2 else np.array([0.0]))
    block = 3 + (nt - 1)
    body = lines[4:]
    if len(body) != nz * block:
        raise ValueError(f"{path}: {len(body)} data lines, expected {nz} x {block}")
    zs = np.empty(nz)
    co = np.empty((nr, nz))
    post = np.zeros((nt, nr, nz))
    for j in range(nz):
        b = body[j * block:(j + 1) * block]
        zs[j] = _numbers(b[0])[0]
        co[:, j] = _numbers(b[1])
        scale = 10.0 ** np.array(_numbers(b[2]))
        for k in range(nt - 1):
            post[k + 1, :, j] = np.array(_numbers(b[3 + k])) * scale
    return r, zs, t, co, post


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")


def _leading_numbers(line):
    """The numbers on an input line, before any trailing remark such as `|int: no_model_lines;`."""
    return [float(v.replace("D", "E").replace("d", "e")) for v in _NUMBER.findall(line.split("|")[0])]


def _interfaces(text):
    """The depths, m, where PSGRN's model lines change anything, read from its input text.

    The model block follows ten parameter lines (receiver, distances, depths, times, accuracy, gravity, directory and
    three lines of names): a count, then the lines `i depth vp vs rho eta1 eta2 alpha`.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    count_at = 10
    n = int(_leading_numbers(lines[count_at])[0])
    rows = [_leading_numbers(ln)[1:8] for ln in lines[count_at + 1:count_at + 1 + n]]
    out = []
    for a, b in zip(rows, rows[1:]):
        if a[0] == b[0] and not np.allclose(a[1:], b[1:]):
            out.append(1e3 * a[0])
    return np.unique(out)


class PsgrnTables:
    """PSGRN's displacement tables, on a uniform `(r, z)` grid at `times` (days, the first 0).

    `coseismic[kind][component]` is `(nr, nz)`: the elastic response less the receiver layer's half-space.
    `postseismic[kind][component]` is `(nt, nr, nz)`: the response at each time less the coseismic, 0 at `times[0]`.
    Kinds are `ss`, `ds`, `cl`. Components are `uz` (positive down), `ur` and `ut`, as `slipsolve.edgrn.EdgrnTables`
    has them: per unit potency, the azimuthal factor stripped. `interfaces` are the depths no interpolation crosses.
    """

    def __init__(self, r, z, times, coseismic, postseismic, interfaces=(), attrs=None):
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.times = np.asarray(times, dtype=float)
        if self.r.size < 2 or self.z.size < 2:
            raise ValueError("PSGRN tables need at least two distances and two source depths")
        self.coseismic = coseismic
        self.postseismic = postseismic
        self.interfaces = np.unique(np.asarray(interfaces, dtype=float))
        self.attrs = dict(attrs or {})
        self._grid = EdgrnTables(self.r, self.z, self._edgrn_dict(lambda kind, comp: coseismic[kind][comp]),
                                 interfaces=self.interfaces)

    @classmethod
    def from_directory(cls, path):
        """Read the tables of a PSGRN run whose input file is `path/psgrn.inp` and whose output names are this module's."""
        path = Path(path).expanduser()
        text = (path / "psgrn.inp").read_text()
        coseismic = {kind: {} for kind in COMPONENTS}
        postseismic = {kind: {} for kind in COMPONENTS}
        grid = None
        for kind, comps in COMPONENTS.items():
            for comp in comps:
                r, z, t, co, post = _read_table(path / f"{comp}.{kind}")
                if grid is None:
                    grid = (r, z, t)
                elif not (np.allclose(r, grid[0]) and np.allclose(z, grid[1]) and np.array_equal(t, grid[2])):
                    raise ValueError(f"{path}: {comp}.{kind} is on a different grid")
                coseismic[kind][comp], postseismic[kind][comp] = co, post
        r, z, t = grid
        return cls(r, z, t, coseismic, postseismic, interfaces=_interfaces(text),
                   attrs={"source": "psgrn", "directory": str(path), "input": text})

    def _edgrn_dict(self, values):
        out = {}
        for kind, comps in COMPONENTS.items():
            out[kind] = {comp: values(kind, comp) for comp in comps}
            if "ut" not in out[kind]:
                out[kind]["ut"] = np.zeros((self.r.size, self.z.size))
        return out

    def rescaled(self, factor):
        """The tables of the same structure with every viscosity times `factor`: the times scale with it."""
        return PsgrnTables(self.r, self.z, self.times * float(factor), self.coseismic, self.postseismic,
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
        return (f"<PsgrnTables nr={self.r.size} r=0-{self.r[-1] / 1e3:.0f}km nz={self.z.size} "
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

    - The tables' time dependence is split into a few functions (`PsgrnTables.modes`, to `tolerance`); their shapes,
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
