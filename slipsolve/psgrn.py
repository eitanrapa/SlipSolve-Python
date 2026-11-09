"""Viscoelastic relaxation of a layered half-space, from PSGRN.

PSGRN (Wang, Lorenzo-Martin & Roth 2006, Computers & Geosciences 32, 527-541) computes the response of point sources
in a layered viscoelastic-gravitational half-space at a set of times after a step dislocation. Each layer can be
elastic, Maxwell, a Standard Linear Solid or Burgers, in shear only. It runs here as a separate program, through
files. This module:

- writes its input from a `slipsolve.edgrn.VelocityModel` and a `Rheology` per depth range (`write_psgrn_input`,
  `run_psgrn`);
- reads its displacement tables (`PsgrnTables`), into the class `slipsolve.relaxation` computes its own in, so the
  modes, `relaxation_history` and `relaxation_pattern` work on either.

`slipsolve.relaxation` computes the same tables without PSGRN and without gravity. PSGRN is the way to have gravity,
which dip-slip relaxation over long times needs.

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

from .relaxation import (COMPONENTS, SECONDS_PER_DAY, RelaxationTables, Rheology, _constant_layers,  # noqa: F401
                         relaxation_history, relaxation_pattern)

__all__ = ["Rheology", "PsgrnTables", "psgrn_layers", "write_psgrn_input", "run_psgrn", "find_psgrn",
           "relaxation_history", "relaxation_pattern"]

#: PSGRN's output names for the three displacement, six stress and five other fields; only the first three are read.
OUTPUT_NAMES = (("uz", "ur", "ut"), ("szz", "srr", "stt", "szr", "srt", "stz"), ("tr", "tt", "rot", "gd", "gr"))
#: Environment variable naming a PSGRN executable, checked after `executable=`.
ENVIRONMENT = "SLIPSOLVE_PSGRN"


_ELASTIC = (0.0, 0.0, 1.0)


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


class PsgrnTables(RelaxationTables):
    """PSGRN's displacement tables: a `slipsolve.relaxation.RelaxationTables` read from a PSGRN run.

    `coseismic` is PSGRN's: the elastic response less the receiver layer's half-space.
    """

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
