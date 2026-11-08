# Postseismic and interseismic modes

A slip inversion is linear in slip whatever the time scale, so both modes are
`SlipInversion` with what their data need added. Both are kinematic: they map slip
fields and coupling. Signals with a physical shape, such as poroelastic rebound or
viscoelastic relaxation from a forward code, go in beside the slip as patterns with a
free amplitude ([Signals beside the slip](#signals-beside-the-slip),
[Flow from PSGRN](#flow-from-psgrn)).

## Postseismic: slip that changes with time

`TimeDependentInversion` writes the cumulative slip as known time functions times
unknown slip fields, \(s(\mathbf{x}, t) = \sum_k f_k(t)\, m_k(\mathbf{x})\), and solves
for the \(m_k\):

```python
from slipsolve import TimeBasis, TimeDependentInversion, TrackSeries, gnss_series

t0, t1 = ...                                 # one per pair, in days after the event
series = [TrackSeries.from_los(stack, t0, t1, sigma=0.005, name="asc", frame=frame, trace=trace,
                               ramp="linear", rms_min=0.01, width_min=2e3, width_max=20e3,
                               exclude_within=2e3)]
series += gnss_series(lon, lat, days, east, north, up, 1e-3, 1e-3, 4e-3, frame)   # per station

windows = TimeBasis.epochs([1, 7, 30, 100, 300])     # or TimeBasis.log_decay(30.0); bases add
inversion = TimeDependentInversion(mesh, series, windows, engine=engine, velocity_model=model)
result = inversion.solve(smoothing=lam, polarity=(-1, 0, 0))       # monotonic: every window right-lateral

result.slip_between(30, 100)                 # an ordinary model on `mesh`: moment, text, plots
result.moment_history([1, 7, 30, 100, 300])
inversion.series_fit(result)                 # each series' fit on its real pairs, in metres
inversion.with_time_basis(TimeBasis.log_decay(100.0))   # a search over tau: no new Green's functions
```

- **Time functions.** `log_decay(tau)`, `exp_decay(tau)`, `epochs(edges)` (slip accumulated
  in each window, so tracks need not share dates), `linear()` (a rate), `step(at)` (a
  coseismic step, to separate it from early afterslip on pairs spanning the event).
  Add them: `TimeBasis.step() + TimeBasis.log_decay(30.0)`.
- **Series of any length cost the same.** A track's cells are the same on every date, so
  there is one Green's matrix per set of cells, and its pairs reduce exactly to at most K
  virtual pairs. With one decay shape this is the weighted stack
  \(\sum_i f_i d_i / \sum_i f_i^2\). The reduction equals the full system, redundant
  networks and free per-pair ramps included. Pairs sharing a date share its noise; give
  `sigma` as one date's noise (`noise="pairs"` for independent pairs).
- **Everything else is unchanged.** The K fields are K copies of the fault joined into one
  mesh, so smoothing, edges, bounds, resolution, posterior covariance, blocked CV and saving
  all apply. `TimeDependentModel.load` reads a saved model back. `temporal=` ties consecutive
  fields; `with_temporal` sweeps it.

**Read with care:**

- **Remove what the basis cannot represent first**: the secular rate (or add
  `TimeBasis.linear()`), seasonal terms, and any poroelastic or viscoelastic signal you do
  not want mapped into slip. Slip on vertical strike-slip faults makes little vertical
  motion, so a vertical signal they cannot fit stays in the residual, as it should.
- **A free ramp on every pair removes what it can copy, and stacking does not bring it
  back.** Deep slip's surface field is broad, and a per-track cubic copies most of it.
  Deep afterslip needs GNSS, or a smaller ramp (`nuisance_prior`).
- **Read `series_fit`, not `rms_misfit`.** The inherited statistics are measured on virtual
  pairs, which are not displacements. The inherited `moment` sums all K fields: use
  `slip_between`.
- **Blocked CV scores in units of each row's noise** here, so `cv_rms` of 1 is the noise, and
  it only means something with a noise model that holds (next section).
- **GNSS alone may not place afterslip in depth.** Slip planted at different depths can come
  back at nearly the same depth, with its moment traded against the depth. Run a planted
  recovery (`recovery`) on your own stations before reading a depth or a moment.
- `G` holds up to K times the rows and K times the columns of one solve in memory.

## Preparing GNSS position series

A daily position is not white: monuments wander, the network moves together from day to day,
and equipment leaves steps. `prepare_stations` takes the Nevada Geodetic Laboratory's files to
what the mode needs:

```python
from slipsolve import prepare_stations, read_steps, read_tenv3

tables = {path.stem: read_tenv3(path) for path in folder.glob("*.tenv3")}
steps = read_steps("steps.txt", stations=tables)                  # geodesy.unr.edu/NGLStationPages/steps.txt
stations, report = prepare_stations(tables, event_mjd, window=(1.0, 300.0), steps=steps,
                                    reference=far_away)           # names: the common mode's stations
series = [c for name in near for c in stations[name].track_series(frame)]
```

- **Each station:** a robust trajectory over 2-5 years before the event (offset, rate, annual and
  semi-annual terms, a step at each antenna change and earthquake of `min_magnitude`), which is
  then subtracted through the window; antenna steps inside the window by the medians either side;
  each day's median over `reference` stations. `report` says what became of every station.
- **Noise:** `noise_levels` gives white noise from a day's lag and a random walk from lags of 60 days
  to a year, and `TrackSeries(random_walk=)` carries the walk into the weights (`time_covariance`:
  two pairs share it over the time they overlap). A stack then trusts its late dates less.
- **Test your noise with a placebo.** Run all of it at a fake event before the real one, where
  there is no signal, and read each stack against its stated noise: a robust scatter near 1 means
  the noise model holds. White noise alone leaves several times that. Measuring the walk over all
  lags instead tends to come out low, since long stacks feel the long lags.
- **With that noise model the misfit means something.** Held-out misfit is in units of the noise,
  so what is left above 1 is signal the model does not explain.

**Read with care:**

- The common mode lowers day-to-day scatter, but not the amplitude of a long stack, so it helps
  short windows. Its stations must hold none of the signal, and it removes whatever
  long-wavelength signal they share.
- An earthquake inside `window` is left in, as deformation: stop the window before a large one.
  The Nevada Geodetic Laboratory lists a station in `steps.txt` if it was inside a distance
  threshold, not because a step was seen.

## Signals beside the slip

Poroelastic rebound and viscoelastic flow have shapes that can be computed from the coseismic
slip, and left out they are mapped into slip. A `Pattern` puts one beside the slip with a single
unknown, its amplitude: 1 means as computed, 0 none of it.

```python
from slipsolve import TimeBasis, TimeDependentInversion, monte_carlo, poroelastic_pattern

rebound = poroelastic_pattern(coseismic, TimeBasis.exp_decay(100.0), nu_drained=0.25, nu_undrained=0.31, frame=frame)
inversion = TimeDependentInversion(mesh, series, TimeBasis.log_decay(1.0), engine=engine, patterns=[rebound])
result = inversion.solve(smoothing=lam)
result.pattern_amplitudes                       # {"poroelastic": ...}
monte_carlo(result, n=60).apply(lambda m: m.pattern_amplitudes["poroelastic"])
```

- **Exact, at no cost.** The pattern's time functions join the slip's in the reduction of each
  series to virtual pairs. An amplitude is a nuisance term like a ramp, so
  `nuisance_prior={"pattern:*": 1.0}`, cross-validation, resolution and saving treat it as one.
- **Any field:** `Pattern(name, los, time)`, `Pattern.from_enu`, or a forward model's history
  over time through `separate_history`, which splits it into a few shapes and a few functions
  sharing one amplitude (`TimeBasis.tabulated`). Velocities take one too:
  `InterseismicInversion(patterns=)`.
- **A pattern says which frame it lives in**, and an inversion refuses one from another. A
  coseismic model solved in another frame is moved with `frame=` (`FaultMesh.in_frame`).
- **Compare on the same rows.** A pattern adds a virtual pair per cell, so its held-out misfit is
  not comparable with the plain inversion's. Solve the reference with the amplitude held at zero
  (`nuisance_prior={"pattern:*": 1e-4}`, not so strong that it breaks the numerics) on the same
  pattern.

**Read with care:**

- **An amplitude alone is not a detection.** It takes up anything with that shape and that time
  function, so read it beside the fit it buys on held-out rows.
- **Fit an amplitude freely before holding it at 1.** A pattern held at 1 (`(1.0, sigma)`) is only
  as good as the model it came from. If the data want much less of it, it carries something they
  reject; split it into parts and fit each, since one part may be sound and another not.
- **Only amplitude x (nu_u - nu_d) is read.** Different pairs of Poisson's ratios give nearly the
  same held-out misfit, since the shape barely depends on which pair it is. The pattern is not a
  scaled copy of the coseismic field: its ratio to the coseismic vertical changes sign from place
  to place (Okada 1985).
- It assumes the whole crust is saturated and that the two ratios are uniform, and rebound is
  largest in the shallowest rock, where that is least true. The fields are for a half-space
  unless you hand it engines.

## Flow from PSGRN

Viscoelastic relaxation of a layered crust comes from PSGRN (Wang, Lorenzo-Martín & Roth 2006,
Computers & Geosciences 32, 527-541), run as a separate program. `slipsolve.psgrn` writes its
input from the crust's `VelocityModel` and a `Rheology` per depth range, reads its tables, and
integrates them over the coseismic model's own mesh. The result is a `Pattern` with one
amplitude:

```python
from slipsolve import Rheology, TimeBasis, TimeDependentInversion, relaxation_pattern, run_psgrn

mantle = [Rheology(32e3, None, eta_maxwell=3e18)]                         # Maxwell below the Moho
tables = run_psgrn(crust, mantle, "psgrn_mantle", r_max=300e3, nr=301,
                   z_min=500.0, z_max=24500.0, nz=25, nt=512, window_days=320.0)
flow = relaxation_pattern(coseismic, tables, name="flow", until=300.0, frame=frame)
inversion = TimeDependentInversion(mesh, series, TimeBasis.log_decay(1.0), engine=engine, patterns=[flow])
```

- **Rheologies:** Maxwell (`eta_maxwell`), a Standard Linear Solid (`eta_kelvin` and `alpha`) or
  Burgers (both), in shear; the bulk modulus stays elastic. A Standard Linear Solid relaxes
  stress with the time `eta_kelvin (1 - alpha) / mu`.
- **The executable:** pass `executable=`, set `SLIPSOLVE_PSGRN`, or put `psgrn` on `PATH`. Use a
  build of the 2020 release (github.com/RongjiangWang/PSGRN-PSCMP_2020) with the three
  time-domain corrections the module's notes describe; as released it puts a Standard Linear
  Solid's series 2-11% off.
- **Viscosity scales time:** multiply every viscosity by `c` and every time by `c`
  (`PsgrnTables.rescaled`), so one run per profile shape serves a whole scan of uniform levels.
- **Reading a viscosity:** with the amplitude free, the viscosity at which it comes out at 1 is
  the one the data ask for. A free amplitude can also take up a residual of the same shape at
  any viscosity, so a held-out gain that does not change with the viscosity does not choose one.
- **Grids:**
  - the tables must cover the mesh's depths and every source-receiver distance (it raises
    otherwise);
  - keep source depths off layer interfaces;
  - take the time step at about a twentieth of the earliest time fitted, and run early
    postseismic in its own window, since the first samples of a run are the least accurate;
  - gravity is on by default (`gravity=1`).
- **Accuracy:**
  - corrected, the relaxed state of a Standard Linear Solid matches an independent layered code
    to 5e-3 of each source type's largest component;
  - the time series of Maxwell, Standard Linear Solid and Burgers bodies match the
    correspondence principle to 0.2-1.5%;
  - the split into modes (`PsgrnTables.modes`) rebuilds the history to its `tolerance`.

## Interseismic: rigid blocks plus back-slip

`InterseismicInversion` fits velocities, in m/yr, with rigid block rotations and the
elastic effect of the slip the faults are not taking (Savage 1983):

```python
from slipsolve import (InterseismicInversion, Observations, read_midas, side_labels,
                       velocity_observations)

gnss = velocity_observations(read_midas("midas.IGS14.txt", bbox=box, min_years=5.0), frame,
                             components="en", floor=1e-3)
insar = Observations.from_los(velocity_grid, name="asc", frame=frame, trace=trace, rms_min=5e-4,
                              width_min=2e3, width_max=20e3)      # a grid of m/yr, as any LOS grid
# or, from a dated series: TrackSeries.from_los(...).velocity_observations()
obs = Observations.concat([gnss, insar], normalize="none")

blocks = side_labels(obs, trace, "east", "west")     # or block_labels(obs, polygons)
inversion = InterseismicInversion(mesh, obs, blocks=blocks, faults=("east", "west"), engine=engine,
                                  ramp={"gnss": "none", "*": "linear"})
result = inversion.solve(smoothing=lam)              # coupling in [0, 1]

result.coupling, result.deficit_rate, result.poles   # per node; per node; {block: EulerPole}
result.moment()                                      # the moment deficit rate, N m/yr
```

- **`faults`** says which blocks a fault separates, `(left, right)` looking along its trace,
  one pair per fault of a joined mesh. A swapped pair warns. Each node's long-term slip is
  the blocks' relative motion there, in the package's sign: negative strike-slip is
  right-lateral, and the back-slip has the opposite sign.
- **The bound depends on the rotations**, so `solve` alternates between them until the
  long-term rate settles, in a few rounds. Pass `rate=(V_s, V_d)`, scalars or per node, to
  fix it instead, which is also what keeps a CV sweep cheap:
  `blocked_cv_curve(inversion, weights, rate=rate)`. `coupling=None` leaves the back-slip free.
- **Rotations are three columns per block**: east and north velocity at the block's centre
  and a spin, which `EulerPole` converts exactly. They travel with a saved model
  (`InterseismicModel.load`).
- **`nuisance_prior={"asc:d*": 0.002}`** puts a Gaussian prior on matching nuisance terms,
  which ties a track's plane to GNSS instead of leaving it free. It means what it says only
  with rows weighted by one over their noise (`normalize="none"`). A pair `(mean, sigma)` gives
  the prior a mean, which ties a coefficient to a value known from elsewhere:
  - a block's rotation from a regional block model, through its three coefficients
    `"block:{name}:ve"`, `":vn"` and `":spin"` (m/yr: the velocity at the block's centre, and
    the spin's velocity 1000 km from it);
  - or a pattern's amplitude held at 1 (`{"pattern:far": (1.0, 1e-4)}`), for a signal computed
    elsewhere, such as the velocity of faults the model leaves out.

**Read with care:**

- **A free plane removes most of the signal.** A locked fault's velocity field across a scene
  is mostly a plane, and a cubic copies more of it. The rate rests on GNSS and the near field,
  so a ramp's degree is a modelling choice here, and C-band's usual linear is not
  automatically safe.
- **Two blocks and one fault are a demonstration.** The motion of every fault left out falls on
  the blocks and on the fault's long-term rate. Use the fault network the region has, and the
  far field.
- **Screen the velocities first** (`screen_velocities`). Subsiding basins and landslides move
  stations by mm/yr to cm/yr, horizontals included, and they bias the locking depth.
- **Vertical faults get no dip-slip deficit** (fault-normal rates would need opening). A forward
  model's field, such as a past earthquake's relaxation, goes in as a pattern (`patterns=`).
- **Everything is in m/yr**, including `rms_min` and `strike=`/`dip=`. `moment_magnitude` is
  one year's worth.

## Fault networks

A fault may separate different blocks along its length, and a block pair that changes at a
junction changes the rate there. Rates close at junctions by construction, being differences of
the same rotations.

```python
from slipsolve import FaultMesh, InterseismicInversion, TSurf, block_labels, cell_folds, read_tsurf

# GOCAD t-surfs, as SCEC's Community Fault Model ships them: several patches of one fault, remeshed as one
surface = TSurf.merged([read_tsurf(path, crs="EPSG:26711")[0] for path in paths], name="san_andreas")
san_andreas = FaultMesh.from_tsurf(surface, surface.trace(), frame, edge_length=5e3, max_depth=40e3)
mesh = FaultMesh.join([san_andreas, hayward])

faults = [[(None, junction, "bay", "pacific"), (junction, None, "east", "salinia")],   # (arc_start, arc_end, left, right), metres
          ("eastbay", "bay")]                                                           # or one pair all along
inversion = InterseismicInversion(mesh, obs, blocks=block_labels(obs, polygons), faults=faults, strain=("east",))
result = inversion.solve(smoothing=lam, sides=("bottom",))       # only the bottom pinned: a sheet's end may be a junction

result.fault_table()    # per segment: its blocks, long-term rate, mean coupling, moment deficit
result.block_table()    # per block: rms of the data, less the rotation, and of the residual
result.strain           # {block: (exx, exy, eyy)}, m/yr per 1000 km
folds = cell_folds(obs, cell=50e3, n_folds=8, groups=blocks)     # blocked_cv_curve(..., folds=folds)
```

- **`from_tsurf`** reads the surface as `cross = f(s, z)` off its trace and fits it on the mesh's
  own lattice, so the mesh is regular whatever the file's triangles; `mesh.attrs` holds how far
  the fit lies from the vertices. Below the surface's bottom edge, which a community model puts
  where seismicity stops, it goes straight down (`below="vertical"`): carrying a surface's last
  dip on can put it tens of kilometres off at depth.
- **`strain=`** adds three coefficients per block, a uniform strain rate. A fault's long-term
  rate is then the difference of the whole field on its two sides, since the rotation alone
  depends on where the strain is referred to.
- **Hold out with `cell_folds`, not slabs.** A slab through a small block holds out all of it,
  and its rotation is then predicted from nothing, which can dominate the held-out misfit. Cells
  dealt within each block (`groups=`) ask the question that was meant, and a fold that still
  leaves a block too few rows warns.

**Read with care:**

- **Rates are determined; locking depth and moment deficit may not be.** Held-out misfit is
  often flat over a wide range of smoothing, and over that range the long-term rates barely move
  while the locking depth and the moment deficit change a great deal.
- **A deep reading is not necessarily a depth.** A locking depth beyond what planted depths
  recover is broad shear the model has nowhere else to put: a fault left out, a viscoelastic
  cycle, or strain.
- **Strain and fault rates trade.** With block strain in, a fault's rate can fall by several
  mm/yr while held-out misfit improves, so what the strain stands for is the question.
- **A fault left out is paid for elsewhere**: in a block's pole, in its stations' residual, and
  in the sense of slip on its neighbours.
- **Coupling on a slow fault has no meaning**: a ratio of two small numbers. Read its deficit
  rate.

## How well it is known

`posterior_covariance` carries noise through the *unbounded* solve, which stops describing the
estimator once parameters sit on their limits, and a coupling map is mostly on its limits. These
re-solve instead. They work on any solved model (coseismic, time-dependent, interseismic) and
cost one bounded solve per copy, on normal equations formed once:

```python
import numpy as np
from slipsolve import checkerboard, monte_carlo, recovery, recovery_score

copies = monte_carlo(model, n=200, noise="blocks", block=50e3)   # same smoothing, limits and block iteration
copies.apply(lambda m: m.long_term_slip[0].mean())               # any statistic of the model class, per copy
np.nanmean(copies.apply(lambda m: m.coupling) > 0.5, axis=0)     # the chance each node is more locked than not

planted = np.r_[-field * v_s, -field * v_d, model.ramp]          # the whole parameter vector, as model.x
back = recovery(model, planted, n=30)                            # data made from it, the same noise, solved again
recovery_score(back.mean[:n], planted[:n])                       # back.bias; recovery(..., smoothing=) for another weight
checkerboard(mesh, 40e3, 10e3, low=0.1, high=0.9)                # a field to plant, in (s, z)
```

- **`noise`**: `"blocks"` is the model's own residual with one random sign per `block` metres
  square per track, which keeps its size and its correlation out to the block's width and needs
  no covariance model. `"white"` is independent at each track's residual rms, `"weights"`
  independent at the stated `1 / weight`; a `CovarianceModel`, `{track: model}` or `f(rng)` also
  work.
- **Read `monte_carlo` and `recovery` together.** The first is the estimator's spread, centred on
  the model: smoothing's bias is the same in every copy, so it says nothing about the error. The
  second measures that bias. Spatially correlated noise (`"blocks"`) gives a much wider spread
  than white noise of the same size.
- A block must be wider than the noise's correlation, or the spread comes out too small, and a
  residual has already lost what the fit absorbed, so both are lower bounds. On one row per block
  `"blocks"` is only a random sign on each row.
- **A good recovery does not make a reading right.** A tidy ladder of planted locking depths shows
  that the data can see depth in this model class. A missing fault or a bad station biases the
  real reading and leaves the ladder intact.

## Production stages: samples and meshes from the model

The coseismic loop, samples from the model's field and meshes from its averaging kernels, works
on the interseismic mode through `iterate_sampling`:

```python
from slipsolve import TraceSet, block_labels, iterate_sampling

def make(mesh, obs):                              # each round's inversion: its labels follow the round's samples
    return InterseismicInversion(mesh, obs, blocks=block_labels(obs, polygons), faults=faults, ramp={"gnss": "none", "*": "linear"})

obs, model, history = iterate_sampling(
    velocity_grids, mesh, TraceSet(traces), frame, sample, coarse_kwargs=coarse, make_inversion=make, fixed=[gnss], normalize="none",
    solve_kwargs=dict(smoothing=lam, sides=sides), tol=1e-4, spacing=9e3, rms_fraction=0.2, refine_within=0.0,
    mesh_adapt=dict(sheets=[dict(trace=trace, geometry=dict(surface=sheet.surface, edge_length=edge)) for trace, sheet in zip(traces, mesh.sheets())],
                    size_from="kernel", fraction=0.5, structure="rows", rounds=1, size_min=3e3, size_max=30e3))
```

- **`tol` is in the model's own unit**, and its default means metres: in m/yr, give `tol=1e-4`
  or the loop stops at once.
- **The quadtree's threshold on the model's field is a fraction of that field's own scatter**,
  and `refine_within` keeps every cell near a fault fine, by default within four
  `exclude_within`. The defaults (`rms_fraction=0.02`, 12 km) can draw ten times the rows of the
  data's own quadtree; choose them by the number of rows you want.
- `fixed` keeps observations as they are every round, such as GNSS. Back-slip's field is what
  gets split, so samples go to where coupling changes and not to where blocks rotate.
- `TraceSet(traces)` measures `exclude_within` and `refine_within` from the nearest of several
  faults, and `mesh_adapt["sheets"]` remeshes a joined mesh sheet by sheet.
- **Give each sheet its `surface` and the mesh's `edge_length`** (`geometry=`): the new layout
  then stays on the surface already fitted (`FaultSurface.on_lattice`).
- The slow part is the forward model on the field's grid, per scene and round (`spacing`). A
  series is sampled from the last round's fields, `history[-1]["fields"]`, by
  `TrackSeries.from_los(..., field=)`.
- **Whiten first.** In synthetic tests with a correlated atmosphere, samples weighted as if
  independent fit the atmosphere, and the more so the closer to a fault they go. Whitening each
  track by its covariance (`scene_covariance`, `fit_reml`) mattered more than where the samples
  went or a new mesh, and the smoothing weight moved the result more than either stage.

## Both modes

- `frame_looks` (on by default) rotates look vectors into the local frame's axes for the
  Green's functions. A transverse Mercator's grid north turns away from true north with
  distance from its central meridian: 3 degrees at 500 km at 36 N, about 1 mm/yr on 20 mm/yr of
  elastic signal. The plain `SlipInversion` keeps geographic looks, as every coseismic model has.
- The zero-slip rows are scaled by the median row weight (`pinned_edges`), so an edge is pinned
  in m/yr as it is in metres. A plain `SlipInversion` sets `boundary_ratio` by hand.
- `solve` defaults to `method="active_set"`. Limits whose two ends coincide fix a parameter. An
  active set that cycles, as it can at weak smoothing with most parameters on their limits, goes
  on by projected Newton steps and stays exact; only if that fails too does it fall back to
  `lsq_linear`.
