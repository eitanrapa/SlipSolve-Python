"""Slip that changes with time: the postseismic mode of the inversion.

A pair spanning `t0` to `t1` measures the slip accumulated between them. Writing the
cumulative slip as known time functions times unknown slip fields,
`s(x, t) = sum_k f_k(t) m_k(x)`, makes each row of the problem linear:

```text
d(cell, pair) = G(cell) sum_k [f_k(t1) - f_k(t0)] m_k + ramp(pair) + noise
```

`TimeBasis` holds the `f_k`: decay shapes, or windows of time whose amplitudes
are the slip increments of those windows (so pairs need not start or end on a window's
edge, and each track keeps its own dates). `TimeDependentInversion` solves for the
`m_k`.

**Nothing about the elastic problem grows with time.** A track's cells are the same on
every date, so there is one Green's matrix per set of cells however many dates there are,
and the `m_k` are held as `K` copies of the fault joined into one mesh
(`FaultMesh.join`). Smoothing, the zero-slip edges, bounds, blocked
cross-validation, resolution, posterior covariance and saving all work on that mesh
unchanged. `SlipInversion.g` is built by
multiplying the one matrix by the time weights rather than by asking the engine again.

**A series reduces to K virtual pairs with no loss** (`TrackSeries.reduce`), which is
what makes hundreds of dates affordable. With `D` the track's `(pairs, cells)` data,
`T` its `(pairs, K)` time weights, `Ct = L L'` the noise covariance between pairs and
`L^-1 T = U S V'`, the misfit splits exactly:

```text
| L^-1 (D - T M' G') - L^-1 C R' |^2  =  | U' L^-1 D - S V' M' G' - C_u R' |^2 + rest
```

where `C` are per-pair free ramp coefficients and `rest` holds neither the slip nor
`C_u`. So each row of `U' L^-1 D` is a virtual pair with its own free ramp and time
weights `S V'`. With one decay shape this is the familiar weighted stack
`sum f_i d_i / sum f_i^2`. `Ct` comes from the dates (`TrackSeries.time_covariance`):
pairs sharing a date share its noise, and a redundant network is handled by its
pseudo-inverse, which is the same as inverting the network first. The test file checks the
identity against the brute-force system to 1e-7 m.

**The noise model is separable.** Every date's noise has the same spatial pattern of
variances (`cell_weight`) on one track, and the free ramps are the only long-wavelength
term. Cells must be valid on every date; `TrackSeries.from_los` masks to the common set.

**The data must already be free of what the basis cannot represent**: the secular rate
(from pre-event data, or add `TimeBasis.linear` to the basis), seasonal terms, and any
viscoelastic or poroelastic signal you do not want mapped into slip, unless it goes in beside
the slip with a free amplitude (`slipsolve.patterns`).

**Free ramps remove what they can copy, on every pair.** Deep afterslip is mostly in
their span, and stacking does not bring it back: the ramp is free on every pair. Only GNSS,
or a smaller ramp (a lower degree or `nuisance_prior`), can.
Read the deep slip with the ramp degree in mind.
"""

import copy
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from .basis import make_basis
from .greens import HalfSpaceTDE
from .inversion import SlipInversion, SlipModel, assemble_design, pinned_edges
from .mesh import FaultMesh
from .patterns import PREFIX as PATTERN_PREFIX
from .patterns import amplitudes
from .regularize import laplace_beltrami, neighbor_smoothing, ramp_columns, ramp_spec
from .sampling import Observations, frame_looks

#: Days in the year `TimeBasis.linear` measures rates per.
DAYS_PER_YEAR = 365.25

#: A singular value below this fraction of the largest carries nothing (a redundant
#: network's null space, or a time function a track's dates do not sample).
RANK_TOLERANCE = 1e-10


class TimeBasis:
    """The cumulative time functions `f_k(t)`, with `t` in days from a chosen origin.

    Build with `log_decay`, `exp_decay`, `epochs`, `linear` or
    `step`, and add bases together: `TimeBasis.step() + TimeBasis.log_decay(30.0)`
    is a coseismic step plus one decay shape, the event at `t = 0`.
    """

    def __init__(self, terms):
        self.terms = [dict(term) for term in terms]
        if not self.terms:
            raise ValueError("a time basis needs at least one function")
        for term in self.terms:
            kind = term.get("kind")
            if kind in ("log", "exp"):
                if not float(term["tau"]) > 0:
                    raise ValueError(f"tau must be positive: {term}")
            elif kind == "window":
                if not float(term["end"]) > float(term["start"]):
                    raise ValueError(f"a window must end after it starts: {term}")
            elif kind == "linear":
                if not float(term["per"]) > 0:
                    raise ValueError(f"per must be positive: {term}")
            elif kind == "table":
                if len(term["t"]) < 2 or len(term["t"]) != len(term["f"]) or np.any(np.diff(term["t"]) <= 0):
                    raise ValueError("a tabulated function needs matching values at two or more increasing times")
            elif kind != "step":
                raise ValueError(f"unknown time function {term}")

    @classmethod
    def log_decay(cls, *tau):
        """`log(1 + t / tau)` after `t = 0`: rate-strengthening afterslip's shape."""
        return cls([{"kind": "log", "tau": float(t)} for t in tau])

    @classmethod
    def exp_decay(cls, *tau):
        """`1 - exp(-t / tau)` after `t = 0`."""
        return cls([{"kind": "exp", "tau": float(t)} for t in tau])

    @classmethod
    def epochs(cls, edges):
        """One function per window between `edges`, rising linearly from 0 to 1 across it.

        Its amplitude is the slip accumulated in the window, at a steady rate inside it.
        A pair covering part of a window sees that part, so tracks with different dates
        share the windows. Edges spaced evenly in log time suit a decay: the increments
        are then about equal, which is what `temporal` smoothing asks for.
        """
        edges = np.asarray(edges, dtype=float)
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("epochs needs at least two increasing edges")
        return cls([{"kind": "window", "start": float(a), "end": float(b)}
                    for a, b in zip(edges[:-1], edges[1:])])

    @classmethod
    def linear(cls, per=DAYS_PER_YEAR):
        """`t / per`: a steady rate, its amplitude the slip per `per` days (a year)."""
        return cls([{"kind": "linear", "per": float(per)}])

    @classmethod
    def step(cls, at=0.0):
        """0 before `at` and 1 from it on: a coseismic step."""
        return cls([{"kind": "step", "at": float(at)}])

    @classmethod
    def tabulated(cls, times, functions):
        """Functions read off a table: `functions` is `(n_times, M)`, linear between `times`, constant past their ends."""
        times = np.asarray(times, dtype=float)
        functions = np.asarray(functions, dtype=float)
        functions = functions[:, None] if functions.ndim == 1 else functions
        if functions.shape[0] != times.size:
            raise ValueError(f"functions has {functions.shape[0]} rows for {times.size} times")
        return cls([{"kind": "table", "t": times.tolist(), "f": column.tolist()} for column in functions.T])

    def __add__(self, other):
        return TimeBasis(self.terms + other.terms)

    def __len__(self):
        return len(self.terms)

    @property
    def names(self):
        """A short label per function."""
        out = []
        for term in self.terms:
            kind = term["kind"]
            if kind in ("log", "exp"):
                out.append(f"{kind}({term['tau']:g} d)")
            elif kind == "window":
                out.append(f"{term['start']:g}-{term['end']:g} d")
            elif kind == "linear":
                out.append(f"per {term['per']:g} d")
            elif kind == "table":
                out.append(f"table of {len(term['t'])}")
            else:
                out.append(f"step at {term['at']:g} d")
        return out

    def cumulative(self, t):
        """`f_k(t)`, `(n_times, K)`."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((t.size, len(self)))
        for k, term in enumerate(self.terms):
            kind = term["kind"]
            if kind == "log":
                out[:, k] = np.log1p(np.maximum(t, 0.0) / term["tau"])
            elif kind == "exp":
                out[:, k] = -np.expm1(-np.maximum(t, 0.0) / term["tau"])
            elif kind == "window":
                out[:, k] = np.clip((t - term["start"]) / (term["end"] - term["start"]), 0.0, 1.0)
            elif kind == "linear":
                out[:, k] = t / term["per"]
            elif kind == "table":
                out[:, k] = np.interp(t, term["t"], term["f"])
            else:
                out[:, k] = (t >= term["at"]).astype(float)
        return out

    def weights(self, t0, t1):
        """`f_k(t1) - f_k(t0)`, `(n_pairs, K)`: what a pair from `t0` to `t1` sees."""
        return self.cumulative(t1) - self.cumulative(t0)

    def to_dict(self):
        """JSON-able, for a saved model's options."""
        return {"terms": self.terms}

    @classmethod
    def from_dict(cls, d):
        """Rebuild a basis from `to_dict`'s output."""
        return cls(d["terms"])

    def __repr__(self):
        return f"<TimeBasis {self.names}>"


class TrackSeries:
    """One set of cells over a set of pairs: an InSAR track, or a GNSS station's component.

    `x`, `y` are local-frame metres, `look` target-to-sensor unit vectors in
    geographic east/north/up, `(n_cells, 3)` or one 3-vector, and `values` metres,
    `(n_pairs, n_cells)`, positive along the look. A pair runs from `t0[i]` to
    `t1[i]`, in days from the origin of the `TimeBasis`.

    `sigma` is the noise of **one date**, a scalar or one per date (sorted): a pair of two
    dates has sqrt(2) times it, and pairs sharing a date share its noise (`noise="dates"`,
    `time_covariance`). `noise="pairs"` makes `sigma` each pair's own, independent
    noise, a scalar or one per pair. `cell_weight` scales it per cell (noise is
    `sigma / cell_weight`; default 1). `random_walk` (m/sqrt(yr)) adds noise that wanders with time,
    as a GNSS monument's does: two pairs share it over the time they overlap.

    `ramp` is the nuisance on each pair, a `slipsolve.regularize.ramp_columns`
    kind: free on every pair. Use one name per track; `"none"` series may share a name,
    and so GNSS stations do. `elevation` (m) is each cell's height above `datum`, as
    `Observations.from_los` records it.
    """

    def __init__(self, x, y, look, values, t0, t1, sigma, name="track", ramp="none", frame=None,
                 noise="dates", cell_weight=None, region=None, cell_size=None, count=None, random_walk=0.0,
                 elevation=None, datum=0.0):
        if frame is None:
            raise ValueError("TrackSeries needs the frame the mesh was built in")
        self.x = np.asarray(x, dtype=float).ravel()
        self.y = np.asarray(y, dtype=float).ravel()
        n = self.x.size
        look = np.asarray(look, dtype=float)
        self.look = np.broadcast_to(look, (n, 3)).copy() if look.ndim == 1 else look.reshape(n, 3)
        self.values = np.atleast_2d(np.asarray(values, dtype=float))
        self.t0 = np.atleast_1d(np.asarray(t0, dtype=float))
        self.t1 = np.atleast_1d(np.asarray(t1, dtype=float))
        if self.values.shape != (self.t0.size, n) or self.t1.size != self.t0.size:
            raise ValueError(f"values has shape {self.values.shape}; expected (pairs, cells) = "
                             f"({self.t0.size}, {n}), with t0 and t1 one per pair")
        if not np.isfinite(self.values).all():
            raise ValueError(f"series {name!r} has non-finite values: every cell must be valid "
                             "on every pair (drop the cell or the pair)")
        if np.any(self.t1 == self.t0):
            raise ValueError(f"series {name!r} has a pair whose two dates are the same")
        if noise not in ("dates", "pairs"):
            raise ValueError(f"noise must be 'dates' or 'pairs', got {noise!r}")
        self.sigma = np.asarray(sigma, dtype=float)
        if not np.all(self.sigma > 0):
            raise ValueError("sigma must be positive")
        self.noise = noise
        self.random_walk = float(random_walk)
        if self.random_walk < 0.0:
            raise ValueError("random_walk cannot be negative")
        self.name = str(name)
        self.ramp = ramp_spec(ramp)
        if not isinstance(self.ramp, str):
            raise ValueError("a series has one ramp kind")
        self.frame = frame
        self.cell_weight = (np.ones(n) if cell_weight is None
                            else np.broadcast_to(np.asarray(cell_weight, dtype=float), (n,)).copy())
        if not np.all(self.cell_weight > 0):
            raise ValueError("cell_weight must be positive")
        self.region = None if region is None else np.asarray(region, dtype=np.int64).ravel()
        self.cell_size = np.zeros(n) if cell_size is None else np.asarray(cell_size, dtype=float).ravel()
        self.count = np.ones(n) if count is None else np.asarray(count, dtype=float).ravel()
        self.elevation = None
        if elevation is not None:
            self.elevation = np.broadcast_to(np.asarray(elevation, dtype=float), (n,)).copy()
            if not np.all(np.isfinite(self.elevation)):
                raise ValueError(f"series {name!r} has non-finite elevations")
        self.datum = float(datum)

    def _with_heights(self, obs):
        """`obs` (rows over this series' cells) with their heights."""
        if self.elevation is not None:
            obs.ds["elevation"] = ("obs", self.elevation)
            obs.ds.attrs["datum"] = self.datum
        return obs

    @classmethod
    def from_los(cls, stack, t0, t1, sigma, name="track", frame=None, trace=None, ramp="none",
                 pairs=None, field=None, noise="dates", **sample_kwargs):
        """Sample a `(pair, y, x)` LOS stack on one set of cells for every pair.

        The quadtree splits on `field` (default the stack's last pair, usually the
        largest signal; see `Observations.from_los` for what
        `rms_min` means then) over the pixels valid on **every** pair, so each pair's
        cells are the same rectangles. `t0`, `t1` are one per pair sampled.
        `sample_kwargs` go to `Observations.from_los`.
        """
        ds = stack.ds
        pairs = list(range(ds.sizes["pair"])) if pairs is None else [int(p) for p in pairs]
        los = ds["los"].isel(pair=pairs)
        common = ~los.isnull().any(dim="pair")
        grid = type(stack)(ds.assign(los=ds["los"].where(common)))
        if field is None:
            field = grid.ds["los"].isel(pair=pairs[-1]).values
        rows = [Observations.from_los(grid, pair=p, name=name, frame=frame, trace=trace, field=field,
                                      **sample_kwargs).ds for p in pairs]
        first = rows[0]
        for p, row in zip(pairs[1:], rows[1:]):
            if not (np.array_equal(row["x"].values, first["x"].values)
                    and np.array_equal(row["y"].values, first["y"].values)):
                raise RuntimeError(f"pair {p} sampled different cells from pair {pairs[0]}")
        return cls(first["x"].values, first["y"].values,
                   np.column_stack([first["look_e"].values, first["look_n"].values, first["look_u"].values]),
                   np.stack([row["los"].values for row in rows]), t0, t1, sigma, name=name, ramp=ramp,
                   frame=Observations(first).frame, noise=noise,
                   region=first["region"].values if "region" in first else None,
                   cell_size=first["cell_size"].values, count=first["count"].values,
                   elevation=first["elevation"].values if "elevation" in first else None,
                   datum=first.attrs.get("datum", 0.0))

    @property
    def n_cells(self):
        """Number of cells."""
        return self.x.size

    @property
    def n_pairs(self):
        """Number of pairs."""
        return self.t0.size

    def time_covariance(self):
        """`Ct`, `(pairs, pairs)`: the noise covariance between this series' pairs.

        `noise="dates"`: `B diag(sigma^2) B'` with `B` the pair-date incidence, so a
        series referred to one date has `sigma^2 (I + 11')`, and consecutive pairs
        share their common date's noise with opposite signs. `random_walk` adds
        `walk^2` times the years two pairs overlap, signed by their directions.
        """
        if self.noise == "pairs":
            covariance = np.diag(np.broadcast_to(self.sigma, (self.n_pairs,)) ** 2)
        else:
            dates, index = np.unique(np.concatenate([self.t0, self.t1]), return_inverse=True)
            sigma = np.broadcast_to(self.sigma, dates.shape)
            B = np.zeros((self.n_pairs, dates.size))
            rows = np.arange(self.n_pairs)
            B[rows, index[:self.n_pairs]] -= 1.0
            B[rows, index[self.n_pairs:]] += 1.0
            covariance = (B * sigma ** 2) @ B.T
        if self.random_walk > 0.0:
            early, late = np.minimum(self.t0, self.t1), np.maximum(self.t0, self.t1)
            overlap = np.minimum.outer(late, late) - np.maximum.outer(early, early)
            sense = np.sign(self.t1 - self.t0)
            covariance = covariance + (self.random_walk ** 2 / DAYS_PER_YEAR) * np.maximum(overlap, 0.0) * np.outer(sense, sense)
        return covariance

    def reduce(self, basis):
        """`(values, time_weights, scale)`: the series as at most K virtual pairs, no loss.

        `values` is `(r, n_cells)`, `time_weights` `(r, K)`, and a virtual row's
        noise is `scale / cell_weight`, independent between virtual pairs. See the
        module docstring for the identity.
        """
        T = basis.weights(self.t0, self.t1)
        C = self.time_covariance()
        eigenvalues, vectors = np.linalg.eigh(C)
        keep = eigenvalues > RANK_TOLERANCE * eigenvalues.max()
        W = (vectors[:, keep] / np.sqrt(eigenvalues[keep])).T
        U, s, Vt = np.linalg.svd(W @ T, full_matrices=False)
        if not s.size or s[0] == 0.0:
            raise ValueError(f"series {self.name!r}: none of the time functions {basis.names} "
                             f"changes across its pairs ({self.t0.min():g} to {self.t1.max():g} d)")
        good = s > RANK_TOLERANCE * s[0]
        U, s, Vt = U[:, good], s[good], Vt[good]
        # A virtual pair's sign is arbitrary: make its largest time weight positive, so that
        # with one shape the virtual pair reads as the stack it is.
        flip = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(axis=1)])
        U, Vt = U * flip, Vt * flip[:, None]
        scale = float(np.sqrt(np.mean(np.diag(C))))       # keeps the rows in metres
        return scale * (U.T @ (W @ self.values)), scale * (s[:, None] * Vt), scale

    def velocity(self, seasonal=True):
        """`(rate, sigma)`, m/yr per cell: each cell's rate by least squares over the pairs.

        Weighted by `time_covariance`, with an annual and a semi-annual term beside
        the rate unless `seasonal=False`. A series years long reduces to what
        `InterseismicInversion` takes, through
        `velocity_observations`. Nothing here removes a per-date ramp: a per-date
        plane averages down in the rate but what is left is a plane on the velocity, so give
        the track a `ramp` there.
        """
        def terms(t):
            year = np.asarray(t, dtype=float) / DAYS_PER_YEAR
            columns = [year]
            if seasonal:
                for h in (1.0, 2.0):
                    columns += [np.sin(2.0 * np.pi * h * year), np.cos(2.0 * np.pi * h * year)]
            return np.column_stack(columns)

        T = terms(self.t1) - terms(self.t0)
        eigenvalues, vectors = np.linalg.eigh(self.time_covariance())
        keep = eigenvalues > RANK_TOLERANCE * eigenvalues.max()
        W = (vectors[:, keep] / np.sqrt(eigenvalues[keep])).T
        Tw = W @ T
        if np.linalg.matrix_rank(Tw) < T.shape[1]:
            raise ValueError(f"series {self.name!r}: {self.n_pairs} pairs over "
                             f"{np.ptp(np.r_[self.t0, self.t1]) / DAYS_PER_YEAR:.2f} yr cannot separate a rate"
                             f"{' from seasonal terms' if seasonal else ''}")
        rate = np.linalg.lstsq(Tw, W @ self.values, rcond=None)[0][0]
        sigma = float(np.sqrt(np.linalg.inv(Tw.T @ Tw)[0, 0]))
        return rate, sigma / self.cell_weight

    def velocity_observations(self, seasonal=True, sigma_floor=0.0):
        """`velocity` as `Observations` rows, `sigma` raised to `sigma_floor`."""
        rate, sigma = self.velocity(seasonal=seasonal)
        out = Observations.from_points(self.x, self.y, rate, self.look, np.maximum(sigma, float(sigma_floor)),
                                       name=self.name, frame=self.frame, epsg=None)
        out.ds["cell_size"] = ("obs", self.cell_size)
        out.ds["count"] = ("obs", self.count)
        if self.region is not None:
            out.ds["region"] = ("obs", self.region)
        return self._with_heights(out)

    def __repr__(self):
        return (f"<TrackSeries {self.name!r} {self.n_cells} cells x {self.n_pairs} pairs, "
                f"{min(self.t0.min(), self.t1.min()):g} to {max(self.t0.max(), self.t1.max()):g} d, "
                f"ramp {self.ramp}>")


class TimeDependentInversion(SlipInversion):
    """Slip fields `m_k` of a `TimeBasis`, from `TrackSeries` of any length.

    `mesh` is one fault, or several joined (`FaultMesh.join`). The
    returned `TimeDependentModel` reads the slip of any window. `smoothing` means
    what it means in `SlipInversion`, over all K fields, so read
    the rank, not the weight, when K changes.

    `temporal` adds rows `temporal / rows * (m_(k+1) - m_k)`, tying consecutive
    fields; use it with `TimeBasis.epochs` spaced so a smooth history has equal
    increments (log-spaced for a decay). Sweep it with `with_temporal`, which keeps `G`.

    `polarity` applies to every field, so with epochs it makes the slip monotonic.
    `solve`'s limits may be one value per parameter of one field, or of all K.

    `frame_looks` rotates the look vectors into the frame's axes
    (`slipsolve.sampling.frame_looks`). `method` defaults to `"active_set"`.

    `g` is `K * cells x K * parameters` at most, in memory; K times the rows and K
    times the columns of one solve. `K` above ~10 on a production mesh wants thought.

    Blocked cross-validation holds out cells on all their virtual pairs, and scores in
    units of each row's noise (`cv_rms` of 1 is the noise level). A virtual pair's rms is
    not a displacement: read `series_fit` for that.

    `patterns` puts signals of known shape beside the slip, each with a free amplitude
    (`slipsolve.patterns`).
    """

    cv_in_noise_units = True

    def __init__(self, mesh, series, time_basis, engine=None, basis="node", velocity_model=None,
                 covariance=None, nuisance_prior=None, temporal=0.0, frame_looks=True, patterns=(), _greens=None,
                 _pattern_cells=None):
        series = [series] if isinstance(series, TrackSeries) else list(series)
        if not series:
            raise ValueError("needs at least one TrackSeries")
        named = [s.name for s in series if s.ramp != "none"]
        repeated = sorted({n for n in named if named.count(n) > 1})
        if repeated:
            raise ValueError(f"series {repeated} would share one ramp: give each pair set its "
                             "own name (only ramp='none' series may share a name)")
        self.single_mesh = mesh
        self.time_basis = time_basis
        self.series = series
        self.temporal = float(temporal)
        self.use_frame_looks = bool(frame_looks)
        engine = engine or HalfSpaceTDE()
        K = len(time_basis)
        self.patterns = list(patterns)
        names = [p.name for p in self.patterns]
        if len(set(names)) != len(names):
            raise ValueError(f"patterns need different names, got {names}")
        extended = time_basis
        for pattern in self.patterns:
            if pattern.time is None:
                raise ValueError(f"pattern {pattern.name!r} has no time function, which a series needs")
            extended = extended + pattern.time

        parts, weights, row_cell, ramp, offset = [], [], [], {}, 0
        for s in series:
            # The patterns' time functions are in the reduction too, or it would not be exact.
            values, T, scale = s.reduce(extended)
            cells = np.arange(s.n_cells)
            for j in range(values.shape[0]):
                name = s.name if s.ramp == "none" or values.shape[0] == 1 else f"{s.name}#{j}"
                row = Observations.from_points(s.x, s.y, values[j], s.look, scale / s.cell_weight,
                                               name=name, frame=s.frame, epsg=None)
                row.ds["cell_size"] = ("obs", s.cell_size)
                row.ds["count"] = ("obs", s.count)
                if s.region is not None:
                    row.ds["region"] = ("obs", s.region)
                parts.append(s._with_heights(row))
                weights.append(np.tile(T[j], (s.n_cells, 1)))
                row_cell.append(offset + cells)
                ramp[name] = s.ramp
            offset += s.n_cells
        obs = Observations.concat(parts, normalize="none")
        weights = np.concatenate(weights)
        self.time_weights = weights[:, :K]
        self.row_cell = np.concatenate(row_cell)
        self.pattern_cells = (self._pattern_values(series, self.patterns, frame_looks) if _pattern_cells is None
                              else _pattern_cells)
        extra = None
        if self.patterns:
            columns, start = [], 0
            rows = self.pattern_cells[self.row_cell]
            for pattern in self.patterns:
                stop = start + pattern.n_modes
                columns.append(np.sum(weights[:, K + start:K + stop] * rows[:, start:stop], axis=1))
                start = stop
            extra = (np.column_stack(columns), [f"{PATTERN_PREFIX}{name}" for name in names])

        single = make_basis(mesh, basis)
        self.single_basis = single
        if _greens is None:
            _greens = self._cell_greens(series, engine, mesh, single, frame_looks)
        self.cell_greens = _greens

        n = single.n_basis
        g = np.zeros((obs.n, 2 * K * n))
        rows = self.cell_greens[self.row_cell]
        for k in range(K):
            weight = self.time_weights[:, k, None]
            g[:, k * n:(k + 1) * n] = weight * rows[:, :n]
            g[:, (K + k) * n:(K + k + 1) * n] = weight * rows[:, n:]
        del rows

        joined = FaultMesh.join([mesh] * K)
        super().__init__(joined, obs, engine=engine, ramp=ramp, basis=basis, velocity_model=velocity_model,
                         covariance=covariance, nuisance_prior=nuisance_prior, g=g, extra_columns=extra)
        self.boundary_scale = pinned_edges(self.w)

    @staticmethod
    def _pattern_values(series, patterns, use_frame_looks):
        """Every pattern's signal at every series' cells, `(sum cells, sum modes)`."""
        count = sum(s.n_cells for s in series)
        if not patterns:
            return np.zeros((count, 0))
        for pattern in patterns:
            if pattern.frame is not None:
                series[0].frame.require_match(pattern.frame, f"The pattern {pattern.name!r}")
        points = np.concatenate([np.column_stack([s.x, s.y, s.look]) for s in series])
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        x, y, le, ln, lu = unique.T
        if use_frame_looks:
            le, ln, lu = frame_looks(series[0].frame, x, y, le, ln, lu)
        return np.column_stack([pattern.values(x, y, le, ln, lu) for pattern in patterns])[inverse.ravel()]

    @staticmethod
    def _cell_greens(series, engine, mesh, basis, use_frame_looks):
        """The Green's matrix at every series' cells, `(sum cells, 2 n)`, one engine call."""
        with_heights = [s.elevation is not None for s in series]
        if any(with_heights) and not all(with_heights):
            raise ValueError("some series carry elevations and some do not: give every one, or none")
        heights = [s.elevation if s.elevation is not None else np.zeros(s.n_cells) for s in series]
        points = np.concatenate([np.column_stack([s.x, s.y, s.look, h]) for s, h in zip(series, heights)])
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        x, y, le, ln, lu, elevation = unique.T
        if use_frame_looks:
            le, ln, lu = frame_looks(series[0].frame, x, y, le, ln, lu)
        return assemble_design(engine, mesh, basis, (x, y, le, ln, lu),
                               elevation=elevation if all(with_heights) else None)[inverse.ravel()]

    @property
    def n_fields(self):
        """Number of slip fields: one per term of `time_basis`."""
        return len(self.time_basis)

    def with_temporal(self, temporal):
        """The same inversion with another `temporal` weight, sharing `G`."""
        other = copy.copy(self)
        other.temporal = float(temporal)
        return other

    def with_time_basis(self, time_basis):
        """The same data and mesh under another `TimeBasis`, sharing the engine's work.

        What a search over `tau` wants: only the time weights are rebuilt.
        """
        return TimeDependentInversion(self.single_mesh, self.series, time_basis, engine=self.engine,
                                      basis=self.basis.name, velocity_model=self.velocity_model,
                                      covariance=self.covariance, nuisance_prior=self.nuisance_prior,
                                      temporal=self.temporal, frame_looks=self.use_frame_looks, patterns=self.patterns,
                                      _greens=self.cell_greens, _pattern_cells=self.pattern_cells)

    def prior_rows(self):
        """The nuisance priors, plus `temporal` rows between consecutive fields."""
        prior = super().prior_rows()
        K, n = self.n_fields, self.single_basis.n_basis
        if self.temporal <= 0.0 or K < 2:
            return prior
        step = sp.kron(sp.diags([-np.ones(K - 1), np.ones(K - 1)], [0, 1], shape=(K - 1, K)),
                       sp.identity(n), format="csr")
        rows = sp.bmat([[step, None], [None, step]], format="csr")
        rows = rows * (self.temporal / rows.shape[0])
        rows = sp.hstack([rows, sp.csr_matrix((rows.shape[0], self.n_ramp))], format="csr")
        return rows if prior is None else sp.vstack([rows, prior], format="csr")

    def solve(self, smoothing=0.3, method="active_set", strike=(-10.0, 10.0), dip=(-10.0, 10.0), **kwargs):
        """As `SlipInversion.solve`; a limit may be one value per parameter of one field."""
        def per_field(value):
            value = np.asarray(value, dtype=float)
            return np.tile(value, self.n_fields) if value.size == self.single_basis.n_basis else value

        return super().solve(smoothing=smoothing, method=method,
                             strike=tuple(per_field(v) for v in strike),
                             dip=tuple(per_field(v) for v in dip), **kwargs)

    def _model(self, x, result, smoothing, smooth_matrix, options):
        options = dict(options)
        options.update(time_basis=self.time_basis.to_dict(), temporal=self.temporal,
                       frame_looks=self.use_frame_looks, boundary_scale=self.boundary_scale)
        return TimeDependentModel(self, x, result, smoothing, smooth_matrix, options)

    def series_fit(self, model):
        """Each series' fit on its real pairs, `{name: {...}}`, ramps re-fitted per pair.

        `SlipModel.rms_misfit` and friends are measured on the
        virtual pairs, which are not data anyone would recognise. This returns, in metres,
        `rms_data` and `rms_residual` over all pairs and `rms_residual_pairs` for each
        pair, both after that pair's own ramp. Series sharing a name are pooled.
        """
        coefficients = model.coefficients                 # (K, 2, n)
        fields = coefficients.reshape(self.n_fields, -1)
        amplitude = model.pattern_amplitudes
        out, offset = {}, 0
        for s in self.series:
            greens = self.cell_greens[offset:offset + s.n_cells]
            cells = self.pattern_cells[offset:offset + s.n_cells]
            offset += s.n_cells
            history = self.time_basis.weights(s.t0, s.t1) @ fields      # (pairs, 2 n)
            predicted = history @ greens.T                               # (pairs, cells)
            start = 0
            for pattern in self.patterns:
                stop = start + pattern.n_modes
                predicted = predicted + amplitude[pattern.name] * (pattern.time.weights(s.t0, s.t1) @ cells[:, start:stop].T)
                start = stop
            design = self._series_ramp(s)
            data, residual = s.values, s.values - predicted
            if design.shape[1]:
                q, _ = np.linalg.qr(design * s.cell_weight[:, None])
                def remove(v):
                    v = v * s.cell_weight
                    return (v - (v @ q) @ q.T) / s.cell_weight
                data, residual = remove(data), remove(residual)
            entry = out.setdefault(s.name, {"sum_data": 0.0, "sum_residual": 0.0, "rows": 0,
                                            "rms_residual_pairs": []})
            entry["sum_data"] += float(np.sum(data ** 2))
            entry["sum_residual"] += float(np.sum(residual ** 2))
            entry["rows"] += data.size
            entry["rms_residual_pairs"].extend(np.sqrt(np.mean(residual ** 2, axis=1)).tolist())
        for entry in out.values():
            rows = max(entry.pop("rows"), 1)
            entry["rms_data"] = float(np.sqrt(entry.pop("sum_data") / rows))
            entry["rms_residual"] = float(np.sqrt(entry.pop("sum_residual") / rows))
            entry["rms_residual_pairs"] = np.asarray(entry["rms_residual_pairs"])
        return out

    @staticmethod
    def _series_ramp(s):
        """The nuisance columns of one pair of a series."""
        if s.ramp == "none":
            return np.zeros((s.n_cells, 0))
        obs = Observations.from_points(s.x, s.y, np.zeros(s.n_cells), s.look, 1.0, name=s.name,
                                       frame=s.frame, epsg=None)
        if s.region is not None:
            obs.ds["region"] = ("obs", s.region)
        return ramp_columns(obs, s.ramp)[0]

    def __repr__(self):
        return (f"<TimeDependentInversion {len(self.series)} series, {self.obs.n} rows, "
                f"{self.n_fields} fields {self.time_basis.names}, params={self.n_param}>")


class SlipSnapshot(SlipModel):
    """The slip of one window of a `TimeDependentModel`, on the original mesh.

    An ordinary `SlipModel` for everything about the slip:
    moment, the text writers, the plots of a slip distribution, `forward`. It has no
    fit of its own (the data span many windows), so `data` and `residual` are empty.
    """

    @property
    def rms_misfit(self):
        """NaN: a snapshot has no fit of its own, since the data span many windows."""
        return float("nan")

    @property
    def variance_reduction(self):
        """NaN: a snapshot has no fit of its own, since the data span many windows."""
        return float("nan")

    def __repr__(self):
        return f"<SlipSnapshot max_slip={self.max_slip:.3f}m Mw={self.moment_magnitude:.2f}>"


class TimeDependentModel(SlipModel):
    """A solved `TimeDependentInversion`.

    `coefficients` are the `m_k`. `slip_between` is the model to use for
    anything physical; the inherited `moment` and `strike_slip` run over all K fields
    joined, and a sum over decay shapes' amplitudes means little.
    """

    def __init__(self, inversion, x, result, smoothing, smooth_matrix, options):
        super().__init__(inversion, x, result, smoothing, smooth_matrix, options)
        self.time_basis = inversion.time_basis
        self._single = (inversion.single_mesh, inversion.single_basis)

    @classmethod
    def load(cls, path):
        """Read back a model written by `save`. The first field's mesh is rebuilt from the joined one."""
        self = super().load(path)
        self.time_basis = TimeBasis.from_dict(self.options["time_basis"])
        K, mesh = len(self.time_basis), self.mesh
        nodes, elements = mesh.n_nodes // K, mesh.n_elements // K
        single = FaultMesh(mesh.nodes[:nodes], mesh.triangles[:elements], mesh.params[:nodes],
                           frame=mesh.frame, attrs=mesh.attrs)
        self._single = (single, make_basis(single, self.basis.name))
        return self

    @property
    def n_fields(self):
        """Number of slip fields: one per term of `time_basis`."""
        return len(self.time_basis)

    @property
    def pattern_amplitudes(self):
        """`{name: amplitude}` of the patterns solved for beside the slip."""
        return amplitudes(self.ramp_labels, self.ramp)

    @property
    def coefficients(self):
        """`(K, 2, n)`: each field's strike- and dip-slip per basis function."""
        n = self._single[1].n_basis
        K = self.n_fields
        return np.stack([self.x[:K * n].reshape(K, n), self.x[K * n:2 * K * n].reshape(K, n)], axis=1)

    def _snapshot(self, weights):
        mesh, basis = self._single
        c = self.coefficients
        strike_slip, dip_slip = weights @ c[:, 0], weights @ c[:, 1]
        out = object.__new__(SlipSnapshot)
        out.mesh, out.basis, out.obs = mesh, basis, self.obs
        out.x = np.concatenate([strike_slip, dip_slip])
        out.strike_slip, out.dip_slip = strike_slip, dip_slip
        out.ramp = np.zeros(0)
        out.ramp_labels = []
        out.result = self.result
        out.smoothing = self.smoothing
        out.options = {**self.options, "ramp": "none"}
        out.data = out.prediction = out.residual = np.zeros(0)
        out._ramp_design = np.zeros((0, 0))
        out._extra_design = None
        out._smooth = (neighbor_smoothing(mesh.neighbors, self.options["ss_ratio"], self.options["ds_ratio"],
                                          weighting=self.options.get("smooth_weighting", "uniform"), mesh=mesh)
                       if basis.name == "element"
                       else laplace_beltrami(mesh, self.options["ss_ratio"], self.options["ds_ratio"]))
        inversion = self._inversion
        out._inversion = SimpleNamespace(engine=inversion.engine, basis=basis, g=None, ramp=None, n_ramp=0,
                                         ramp_labels=[], exact_basis=inversion.exact_basis,
                                         velocity_model=getattr(inversion, "velocity_model", None))
        return out

    def slip_between(self, t0, t1):
        """The slip accumulated from `t0` to `t1` days, as a `SlipSnapshot`."""
        return self._snapshot(self.time_basis.weights([t0], [t1])[0])

    def field(self, k):
        """Field `k` itself (an epoch's increment, a decay shape's amplitude), as a `SlipSnapshot`."""
        weights = np.zeros(self.n_fields)
        weights[k] = 1.0
        return self._snapshot(weights)

    def moment_history(self, times, **moment_kwargs):
        """Moment accumulated from `times[0]` to each of `times`, N m."""
        return np.array([self.slip_between(times[0], t).moment(**moment_kwargs) if t != times[0] else 0.0
                         for t in times])

    def __repr__(self):
        flag = "" if self.converged else " UNCONVERGED"
        return (f"<TimeDependentModel {self.n_fields} fields {self.time_basis.names}, "
                f"rms {1e3 * self.rms_misfit:.2f} mm on virtual pairs{flag}>")
