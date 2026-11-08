"""Signals of known shape beside the slip: patterns with a free amplitude.

A postseismic series holds more than afterslip. Poroelastic rebound and viscoelastic flow have
shapes that can be computed from the coseismic slip, and left out of the model they are mapped
into slip. A `Pattern` puts such a signal beside the slip with one unknown, its amplitude:
1 means as computed, 0 means none of it, and how far from those it lands is the test.

```text
d(cell, pair) = G(cell) sum_k [f_k(t1) - f_k(t0)] m_k  +  alpha sum_j [g_j(t1) - g_j(t0)] p_j(cell)
```

The pattern's time functions `g_j` join the slip's in the reduction of each series to virtual
pairs (`TrackSeries.reduce`), so it stays exact: the test file checks it
against the whole system. One shape and one time function is the usual case. A signal whose
shape changes with time, as viscoelastic flow's does, is several `(g_j, p_j)` sharing one
amplitude (`separate_history`).

An amplitude is an unbounded nuisance term, in `SlipInversion.ramp_labels`
as `"pattern:{name}"`, so `nuisance_prior={"pattern:*": 1.0}` limits one, and blocked
cross-validation, resolution and saving treat it as they treat a ramp.

A pattern trades off with slip wherever the two shapes and the two time functions look alike, and
its amplitude can be pulled by anything else with that shape, so read it beside the fit it buys
on held-out rows: an amplitude alone is not a detection.
"""

import numpy as np

from .basis import make_basis
from .greens import HalfSpaceTDE
from .inversion import assemble_design

__all__ = ["Pattern", "poroelastic_pattern", "separate_history", "amplitudes"]

#: The label of a pattern's amplitude among a model's nuisance terms.
PREFIX = "pattern:"

#: A drained crust's Poisson's ratio, and an undrained one's: Jonsson et al. (2003, Nature 424), who
#: read 0.27 to 0.31 off Iceland's rebound; Peltzer et al. (1996, Science 273) used 0.28 and 0.35.
NU_DRAINED, NU_UNDRAINED = 0.25, 0.31


class Pattern:
    """A known signal with a free amplitude.

    `los(x, y, look_e, look_n, look_u)` gives the signal along each look per unit amplitude: frame
    metres, and look vectors in the frame's axes, as an engine is handed them. It returns `(n,)`
    or, with several time functions, `(n, M)`. `time` is a
    `TimeBasis` of `M` functions; `None` where there is no time, as in
    a velocity fit. `from_enu` builds one from a field in east, north and up.

    `frame` is the `LocalFrame` the signal lives in, when it comes from something
    built in one: an inversion then refuses a pattern from another frame, as it does observations.
    """

    def __init__(self, name, los, time=None, frame=None):
        self.name = str(name)
        self.los = los
        self.time = time
        self.frame = frame

    @classmethod
    def from_enu(cls, name, enu, time=None):
        """From `enu(x, y) -> (east, north, up)`, each `(n,)` or `(n, M)`, in the frame's axes."""
        def los(x, y, look_e, look_n, look_u):
            east, north, up = (np.asarray(v, dtype=float) for v in enu(x, y))
            looks = [np.asarray(v, dtype=float) for v in (look_e, look_n, look_u)]
            if east.ndim == 2:
                looks = [v[:, None] for v in looks]
            return east * looks[0] + north * looks[1] + up * looks[2]

        return cls(name, los, time)

    @property
    def n_modes(self):
        """How many `(g_j, p_j)` the pattern has: one where there is no time."""
        return 1 if self.time is None else len(self.time)

    def values(self, x, y, look_e, look_n, look_u):
        """Its signal per unit amplitude at points: `(n, n_modes)`."""
        x = np.asarray(x, dtype=float)
        out = np.asarray(self.los(x, y, look_e, look_n, look_u), dtype=float)
        out = out.reshape(x.size, -1)
        if out.shape[1] != self.n_modes:
            raise ValueError(f"pattern {self.name!r} gave {out.shape[1]} columns for its {self.n_modes} time functions")
        if not np.isfinite(out).all():
            raise ValueError(f"pattern {self.name!r} is not finite at {int(np.sum(~np.isfinite(out).all(axis=1)))} points")
        return out

    def __repr__(self):
        return f"<Pattern {self.name!r}, {self.time}>"


def amplitudes(labels, values):
    """`{name: amplitude}` out of a model's nuisance labels and coefficients."""
    return {str(label)[len(PREFIX):]: float(value) for label, value in zip(labels, values) if str(label).startswith(PREFIX)}


def poroelastic_pattern(model, time=None, nu_drained=NU_DRAINED, nu_undrained=NU_UNDRAINED, drained=None, undrained=None,
                        name="poroelastic", frame=None):
    """Poroelastic rebound of a solved coseismic `model`: the drained field minus the undrained one.

    Just after an earthquake pore fluid has not had time to move, and rock responds with its undrained
    Poisson's ratio; as the pressure diffuses away it reaches its drained one (Peltzer et al. 1996). What
    the surface does in between is the difference of two elastic solutions of the same slip, so an
    amplitude of 1 means exactly those two ratios. It is not a scaled copy of the coseismic field:
    between 0.25 and 0.31 its ratio to the coseismic vertical runs from -0.8 to +0.8 from place to place, with a median
    of +0.43 over a surface-breaking strike-slip fault and -0.32 over one buried at 5 to 15 km (Okada 1985),
    and its horizontal part is as large as its vertical.

    `time` defaults to `TimeBasis.exp_decay(100.0)`: months is the usual scale, set by a
    hydraulic diffusivity this does not know. `drained` and `undrained` are engines, for a layered
    crust whose `Vp` carries each ratio; by default half-spaces with these ratios. `frame` is the inversion's
    `LocalFrame`, if it is not the coseismic model's: the model's mesh moves into it.

    It assumes the whole crust is saturated and that the two ratios are uniform, and with rock
    this shallow is where the signal is largest and the assumption worst: read the amplitude as a
    scale on the two ratios' difference, not as their measurement.
    """
    from .temporal import TimeBasis

    time = TimeBasis.exp_decay(100.0) if time is None else time
    if len(time) != 1:
        raise ValueError("a poroelastic pattern has one shape, so one time function")
    drained = HalfSpaceTDE(float(nu_drained)) if drained is None else drained
    undrained = HalfSpaceTDE(float(nu_undrained)) if undrained is None else undrained
    mesh, basis = model.mesh, model.basis
    slip = np.asarray(model.x, dtype=float)[:2 * basis.n_basis]
    if frame is not None and not mesh.frame.matches(frame):
        mesh = mesh.in_frame(frame)
        basis = make_basis(mesh, basis.name)

    def los(x, y, look_e, look_n, look_u):
        points = (x, y, look_e, look_n, look_u)
        return (assemble_design(drained, mesh, basis, points) - assemble_design(undrained, mesh, basis, points)) @ slip

    return Pattern(name, los, time, frame=mesh.frame)


def separate_history(times, history, tolerance=1e-3, max_modes=8):
    """`(modes, functions)`: a signal whose shape changes with time as a few shapes times a few functions.

    `history` is the signal at `times` (days), `(n_times, n_points)`, from any forward model, with its
    first time the reference (all zero). An SVD splits it into `history = functions @ modes`, keeping
    what carries more than `tolerance` of its largest part, `functions` being `(n_times, r)` and
    `modes` `(r, n_points)`. So the signal goes in as a pattern of `r` terms and one amplitude
    (by interpolating `functions` in time, a `TimeBasis` of `TimeBasis.tabulated` terms).
    """
    history = np.asarray(history, dtype=float)
    times = np.asarray(times, dtype=float)
    if history.ndim != 2 or history.shape[0] != times.size:
        raise ValueError(f"history has shape {history.shape}, expected ({times.size}, n_points)")
    u, s, vt = np.linalg.svd(history, full_matrices=False)
    keep = min(int(np.sum(s > float(tolerance) * s[0])), int(max_modes))
    return s[:keep, None] * vt[:keep], u[:, :keep]
