"""GNSS velocities and position series, in the forms the slip modes take.

`read_midas`, `read_tenv3` and `read_steps` read the Nevada Geodetic
Laboratory's files (columns checked against `midas.IGS14.txt`, an IGS20 `.tenv3` and
`steps.txt`). `gnss_observations` turns one displacement or velocity per
station into the rows of `Observations`, and `gnss_series`
turns a station's positions over time into the `TrackSeries` of each
component.

A station is one row per component, with a unit look vector along geographic east, north
or up, and with no ramp: give `ramp={"gnss": "none", ...}`.

**Preparing a postseismic series** (`prepare_stations`). A daily position is not white:
monuments wander, the whole network moves together from day to day, and equipment changes
leave steps. Stacking hundreds of days as if they were independent trusts them far too much. So:

1. a robust **trajectory** on the pre-event window: offset, rate, annual and semi-annual terms,
   and a step at each of its antenna changes and earthquakes (`fit_trajectory`);
2. antenna-change steps inside the post-event window, each by the difference of the medians
   either side. What remains in that window is the signal;
3. **common-mode** removal: each day's median over reference stations outside the signal
   (`common_mode`);
4. each component's **noise** from what is left of its pre-event window: white plus a random
   walk (`noise_levels`), which `TrackSeries` carries into the
   weights.

The test of all this is a **placebo**: run it at a fake event before the real one, where there
is no signal at all, and read the stack against its stated noise (scatter 1 means right; see
`noise_levels`). The common mode lowers day-to-day scatter but not a long
stack's amplitude: it helps short windows.
"""

import numpy as np
import pandas as pd

from .sampling import Observations

#: Days in the year rates and random walks are measured per.
DAYS_PER_YEAR = 365.25

#: The lags, days, a station's random walk is measured at.
NOISE_LONG_LAGS = (60, 90, 120, 180, 240, 300, 365)

_MIN_NOISE_DAYS = 30
_MIN_NOISE_PAIRS = 20

#: The least white noise a station may be given, m: a position's rounding, so never zero.
_NOISE_FLOOR = 1e-5

_COMPONENTS = {"e": (1.0, 0.0, 0.0), "n": (0.0, 1.0, 0.0), "u": (0.0, 0.0, 1.0)}

_MIDAS_COLUMNS = ["station", "version", "first", "last", "years", "epochs", "epochs_good", "pairs",
                  "ve", "vn", "vu", "se", "sn", "su", "e0", "n0", "u0",
                  "outliers_e", "outliers_n", "outliers_u", "scatter_e", "scatter_n", "scatter_u",
                  "steps", "lat", "lon", "height"]

_TENV3_COLUMNS = ["station", "date", "year", "mjd", "week", "day", "reflon", "e0", "east", "n0", "north",
                  "u0", "up", "antenna", "sigma_e", "sigma_n", "sigma_u", "corr_en", "corr_eu", "corr_nu",
                  "lat", "lon", "height"]


def read_midas(path, bbox=None, min_years=0.0, exclude=()):
    """A MIDAS velocity table as a `pandas.DataFrame`, velocities in m/yr.

    `bbox` is `(lon_min, lon_max, lat_min, lat_max)`, `min_years` the shortest record
    kept (MIDAS needs at least a year; its authors trust 2.5 and up), `exclude` station
    names dropped. Longitudes are wrapped into [-180, 180]; the file's run past -180.
    """
    table = pd.read_csv(path, sep=r"\s+", header=None, names=_MIDAS_COLUMNS, comment="#")
    table["lon"] = (table["lon"] + 180.0) % 360.0 - 180.0
    keep = (table["years"] >= float(min_years)) & ~table["station"].isin(list(exclude))
    if bbox is not None:
        keep &= table["lon"].between(bbox[0], bbox[1]) & table["lat"].between(bbox[2], bbox[3])
    return table[keep].reset_index(drop=True)


def read_tenv3(path):
    """One NGL `.tenv3` as a `pandas.DataFrame`.

    `east`, `north`, `up` are whole positions (the file's integer and fractional
    parts added), metres, sorted by `mjd`. Steps are not removed: antenna changes and
    earthquakes are for the caller.
    """
    table = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=_TENV3_COLUMNS)
    table["east"] = table["e0"] + table["east"]
    table["north"] = table["n0"] + table["north"]
    table["up"] = table["u0"] + table["up"]
    table["lon"] = (table["lon"] + 180.0) % 360.0 - 180.0
    return table.drop(columns=["e0", "n0", "u0"]).sort_values("mjd").reset_index(drop=True)


def gnss_observations(lon, lat, east, north, up, sigma_e, sigma_n, sigma_u, frame, name="gnss",
                      components="enu", scale=1.0, floor=0.0):
    """Stations' displacements or velocities as observation rows, one per component.

    `sigma_*` are multiplied by `scale` and raised to `floor`, in the data's units. A
    typical choice for displacements is `scale=3`, `floor=0.005`; MIDAS's sigmas for a
    long record reach 0.2 mm/yr, which is not the model error, so floor those too.
    `components` is any of `"enu"`: horizontal only for `"en"`.
    """
    lon, lat = np.atleast_1d(np.asarray(lon, dtype=float)), np.atleast_1d(np.asarray(lat, dtype=float))
    values = {"e": east, "n": north, "u": up}
    sigmas = {"e": sigma_e, "n": sigma_n, "u": sigma_u}
    rows = []
    for c in components:
        if c not in _COMPONENTS:
            raise ValueError(f"components are any of 'enu', got {components!r}")
        rows.append((lon, lat, np.broadcast_to(np.asarray(values[c], dtype=float), lon.shape),
                     np.tile(_COMPONENTS[c], (lon.size, 1)),
                     np.maximum(np.broadcast_to(np.asarray(sigmas[c], dtype=float), lon.shape) * float(scale),
                                float(floor))))
    stacked = [np.concatenate([r[i] for r in rows]) for i in range(5)]
    return Observations.from_points(*stacked, name=name, frame=frame)


def velocity_observations(table, frame, name="gnss", components="enu", scale=1.0, floor=0.0):
    """Observation rows from a `read_midas` table, in m/yr."""
    return gnss_observations(table["lon"].values, table["lat"].values, table["ve"].values, table["vn"].values,
                             table["vu"].values, table["se"].values, table["sn"].values, table["su"].values,
                             frame, name=name, components=components, scale=scale, floor=floor)


def screen_velocities(table, max_vertical=0.01, max_from_neighbours=0.01, neighbours=6, apart=2e3, groups=None):
    """`(keep, reason)`: which stations of a `read_midas` table an elastic block model should fit.

    Two kinds of station move for reasons that have nothing to do with faults, and they are not
    rare. Over a pumped aquifer the ground subsides, and those stations carry horizontal residuals;
    a station with `|vu|` above `max_vertical` (m/yr) goes. And a station can sit on something
    sliding, such as a slow landslide.
    A station whose horizontal velocity is more than `max_from_neighbours` (m/yr) from the median
    of its `neighbours` nearest goes too.

    Its neighbours are at least `apart` metres away, since those on one landslide or monument
    agree with each other, and in the same `groups` (one label per station, such as a block),
    since the step across a fault is not an outlier. `reason` says why, one string per station.
    """
    lon, lat = table["lon"].values.astype(float), table["lat"].values.astype(float)
    ve, vn, vu = table["ve"].values.astype(float), table["vn"].values.astype(float), table["vu"].values.astype(float)
    reason = np.full(lon.size, "", dtype=object)
    reason[np.abs(vu) > float(max_vertical)] = "vertical"
    groups = np.zeros(lon.size, dtype=int) if groups is None else np.asarray(groups)
    if groups.size != lon.size:
        raise ValueError(f"groups has {groups.size} labels for {lon.size} stations")
    degree = np.radians(1.0) * 6371e3
    x, y = lon * degree * np.cos(np.radians(lat.mean())), lat * degree
    good = reason == ""
    for k in np.flatnonzero(good):
        distance = np.hypot(x - x[k], y - y[k])
        others = np.flatnonzero(good & (groups == groups[k]) & (distance >= float(apart)))
        if others.size < 3:
            continue
        nearest = others[np.argsort(distance[others])[:int(neighbours)]]
        if np.hypot(ve[k] - np.median(ve[nearest]), vn[k] - np.median(vn[nearest])) > float(max_from_neighbours):
            reason[k] = "neighbours"
    return reason == "", reason.astype(str)


def gnss_series(lon, lat, t, east, north, up, sigma_e, sigma_n, sigma_u, frame, name="gnss",
                components="enu", reference=0, random_walk=(0.0, 0.0, 0.0)):
    """One station's positions as a `TrackSeries` per component.

    `t` is days from the time basis's origin; `east`, `north`, `up` metres, each
    one value per day, with the secular motion, seasonal terms and any steps already
    removed. Each pair runs from date `reference` (an index, default the first) to
    another date, and shares the reference date's noise. `sigma_*` is the noise of one
    date, and `random_walk` its east, north and up random walk, m/sqrt(yr). A missing day is
    simply absent; a NaN in one component drops that day there.

    The series are named `{name}_e`, `_n` and `_u`, so
    `TimeDependentInversion.series_fit` pools each component over the stations.
    """
    from .temporal import TrackSeries

    t = np.asarray(t, dtype=float)
    x, y = frame.to_local(float(lon), float(lat))
    out = []
    for c, values, sigma, walk in zip("enu", (east, north, up), (sigma_e, sigma_n, sigma_u), random_walk):
        if c not in components:
            continue
        values = np.asarray(values, dtype=float)
        days = np.flatnonzero(np.isfinite(values))
        if days.size < 2:
            continue
        times, positions = t[days], values[days]
        anchor = min(int(np.searchsorted(days, reference)), days.size - 1)   # the first valid day from it on
        others = np.arange(days.size) != anchor
        out.append(TrackSeries([x], [y], _COMPONENTS[c], (positions[others] - positions[anchor])[:, None],
                               np.full(int(others.sum()), times[anchor]), times[others], float(sigma),
                               name=f"{name}_{c}", ramp="none", frame=frame, random_walk=float(walk)))
    return out


# -- preparing position series ------------------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def _date_to_mjd(date):
    """NGL's `YYMMMDD`, such as `19JUL06`, as a Modified Julian Date."""
    year, month, day = int(date[:2]), _MONTHS[date[2:5]], int(date[5:7])
    year += 1900 if year >= 69 else 2000
    return (np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") - np.datetime64("1858-11-17")).astype(int)


def read_steps(path, stations=None):
    """NGL's `steps.txt` as a `pandas.DataFrame`, one row per step.

    Columns `station`, `mjd`, `kind` (`"equipment"` or `"earthquake"`), `detail` (for
    equipment, what changed), `distance_km`, `magnitude` and `event` (for earthquakes).
    `stations` keeps only those. An earthquake row means the station was inside NGL's distance
    threshold for that event, not that a step was seen; a receiver change usually moves nothing.
    """
    wanted = None if stations is None else set(stations)
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3 or (wanted is not None and parts[0] not in wanted):
                continue
            if parts[2] == "1":
                rows.append((parts[0], _date_to_mjd(parts[1]), "equipment", " ".join(parts[3:]), np.nan, np.nan, ""))
            elif parts[2] == "2" and len(parts) >= 7:
                rows.append((parts[0], _date_to_mjd(parts[1]), "earthquake", "", float(parts[4]), float(parts[5]), parts[6]))
    return pd.DataFrame(rows, columns=["station", "mjd", "kind", "detail", "distance_km", "magnitude", "event"])


def trajectory_design(t, step_times=(), seasonal=True):
    """Offset, rate per year, annual and semi-annual terms and a step per `step_times`: `(n, m)`. `t` in days."""
    t = np.asarray(t, dtype=float)
    columns = [np.ones_like(t), t / DAYS_PER_YEAR]
    if seasonal:
        for harmonic in (1.0, 2.0):
            phase = 2.0 * np.pi * harmonic * t / DAYS_PER_YEAR
            columns += [np.sin(phase), np.cos(phase)]
    columns += [(t >= time).astype(float) for time in step_times]
    return np.column_stack(columns)


def fit_trajectory(t, values, step_times=(), seasonal=True, clip=4.0, rounds=4):
    """`(coefficients, scatter)` of a robust trajectory: rows beyond `clip` scatters are dropped and it is refitted.

    Steps with no data on one side are left out of the fit (their columns are zero or constant),
    which leaves their coefficients zero. `scatter` is 1.4826 x the median absolute residual.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    usable = [time for time in step_times if np.any(t < time) and np.any(t >= time)]
    design = trajectory_design(t, usable, seasonal)
    keep = np.isfinite(values)
    if keep.sum() <= design.shape[1]:
        raise ValueError(f"{int(keep.sum())} valid days cannot fit a trajectory of {design.shape[1]} terms")
    scatter = np.inf
    for _ in range(int(rounds)):
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values - design @ coefficients
        scatter = 1.4826 * float(np.nanmedian(np.abs(residual[keep] - np.nanmedian(residual[keep]))))
        new = np.isfinite(values) & (np.abs(residual) <= clip * max(scatter, 1e-9))
        if np.array_equal(new, keep):
            break
        keep = new
    full = np.zeros(trajectory_design(t[:1], step_times, seasonal).shape[1])
    positions = [k for k, time in enumerate(step_times) if time in usable]
    base = design.shape[1] - len(usable)
    full[:base] = coefficients[:base]
    full[[base + k for k in positions]] = coefficients[base:]
    return full, scatter


def noise_levels(t, residual, long_lags=NOISE_LONG_LAGS):
    """`(sigma, random_walk)`: white noise (m) and random walk (m/sqrt(yr)) from a residual's structure function.

    The mean squared difference at lag `tau` is `2 sigma^2 + walk^2 tau`. `sigma` comes from
    a day's lag and `walk` from `long_lags` (days), the median of what they say, each by a
    robust structure function (the median of the squares, scaled) that matches days exactly, so
    gaps need no filling.

    Long lags on purpose. Real noise is not white plus a walk: flicker noise lies between. A fit
    over all lags is steered by the many short ones, and a postseismic stack feels the long
    ones. `t` is days, one sample a day.
    """
    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual, dtype=float)
    keep = np.isfinite(residual)
    t, residual = t[keep], residual[keep]
    if t.size < _MIN_NOISE_DAYS:
        raise ValueError(f"{t.size} days are too few to measure a noise level")
    # Counted from the first day: rounding t itself would collapse days falling on half-integers.
    day = np.round(t - t.min()).astype(np.int64)
    series = np.full(int(day.max()) + 1, np.nan)
    series[day] = residual

    def structure(lag):
        difference = series[lag:] - series[:-lag]
        difference = difference[np.isfinite(difference)]
        # the median of a chi-square with one degree of freedom is 0.4549
        return float(np.median(difference ** 2)) / 0.4549 if difference.size >= _MIN_NOISE_PAIRS else np.nan

    lags = [int(lag) for lag in long_lags if lag <= series.size // 2] or [max(2, series.size // 2)]
    measured = [(lag, structure(lag)) for lag in lags]
    measured = [(lag, value) for lag, value in measured if np.isfinite(value)]
    short = structure(1)
    if not measured or not np.isfinite(short):
        raise ValueError(f"too few overlapping days in {t.size} to measure a noise level")
    walk_squared = 0.0
    for _ in range(3):                  # the day's lag holds a day of the walk: a small correction, soon settled
        sigma_squared = max(0.5 * (short - walk_squared / DAYS_PER_YEAR), _NOISE_FLOOR ** 2)
        walk_squared = float(np.median([max(value - 2.0 * sigma_squared, 0.0) * DAYS_PER_YEAR / lag for lag, value in measured]))
    return float(np.sqrt(sigma_squared)), float(np.sqrt(walk_squared))


def common_mode(residuals):
    """Each day's median over `residuals`, `{station: (mjd, east, north, up)}`: `(mjd, east, north, up)`.

    Pass only stations outside the signal. A day with fewer than three of them gets NaN, and is
    not corrected.
    """
    frames = []
    for mjd, east, north, up in residuals.values():
        frames.append(pd.DataFrame({"mjd": np.asarray(mjd, dtype=np.int64), "east": east, "north": north, "up": up}))
    if not frames:
        raise ValueError("common_mode needs at least one station")
    table = pd.concat(frames, ignore_index=True)
    counts = table.groupby("mjd")["east"].count()
    median = table.groupby("mjd")[["east", "north", "up"]].median()
    median.loc[counts < 3] = np.nan
    return (median.index.values, median["east"].values, median["north"].values, median["up"].values)


class StationSeries:
    """One station's post-event positions, prepared by `prepare_stations`.

    `t` is days after the event; `east`, `north`, `up` metres, with the pre-event motion,
    its steps and the common mode removed; `sigma` and `random_walk` the noise per component.
    """

    def __init__(self, name, lon, lat, t, east, north, up, sigma, random_walk):
        self.name, self.lon, self.lat = str(name), float(lon), float(lat)
        self.t = np.asarray(t, dtype=float)
        self.east, self.north, self.up = (np.asarray(v, dtype=float) for v in (east, north, up))
        self.sigma = tuple(float(v) for v in sigma)
        self.random_walk = tuple(float(v) for v in random_walk)

    def track_series(self, frame, name="gnss", components="enu"):
        """The `TrackSeries` of each component, referred to the first day."""
        return gnss_series(self.lon, self.lat, self.t, self.east, self.north, self.up, *self.sigma, frame, name=name,
                           components=components, random_walk=self.random_walk)

    def __repr__(self):
        return (f"<StationSeries {self.name} {self.t.size} days, sigma {tuple(round(1e3 * v, 2) for v in self.sigma)} mm, "
                f"walk {tuple(round(1e3 * v, 2) for v in self.random_walk)} mm/sqrt(yr)>")


def _antenna_days(steps, station):
    if steps is None:
        return np.zeros(0)
    rows = steps[(steps["station"] == station) & (steps["kind"] == "equipment")
                 & steps["detail"].str.contains("Antenna", case=False, na=False)]
    return np.unique(rows["mjd"].values.astype(float))


def _earthquake_days(steps, station, min_magnitude):
    if steps is None:
        return np.zeros(0)
    rows = steps[(steps["station"] == station) & (steps["kind"] == "earthquake") & (steps["magnitude"] >= min_magnitude)]
    return np.unique(rows["mjd"].values.astype(float))


def prepare_stations(tables, event_mjd, window=(1.0, 365.0), pre_years=(2.0, 5.0), steps=None, reference=(),
                     pre_gap=2.0, step_days=15, min_magnitude=5.0, max_scatter=(4e-3, 4e-3, 12e-3), min_days=100):
    """`(stations, report)`: post-event series ready for the postseismic mode, and what became of each.

    `tables` is `{station: read_tenv3 table}`. `window` is days after `event_mjd` to keep,
    `pre_years` the least and most of the record before it to fit, ending `pre_gap` days
    before the event (so a foreshock's day is not in it: pass the earliest shock). `steps` is
    `read_steps`' table; earthquakes of `min_magnitude` and up get a step in the
    pre-event fit. Antenna changes get one too, and those inside `window` are taken out by
    medians over `step_days` either side. Earthquakes inside `window` are left in: they
    are deformation, so stop the window before a large one nearby.

    `reference` names the stations whose daily median is the common mode: far enough to hold
    none of the signal, near enough to share the network's noise. Whatever they share of a
    very long-wavelength signal is removed with it. They get no series of their own unless they
    are in `tables` for that too: every table is prepared, and `stations` holds them all, so
    leave them out of the inversion yourself.

    `max_scatter` drops a component-noisy station (pre-event scatter, m, east, north, up). The
    `report` has one row per station with its status, days and noise.
    """
    t_days = {}
    rows, prepared = [], {}
    columns = ("east", "north", "up")
    for name, table in tables.items():
        t = table["mjd"].values + 0.5 - float(event_mjd)
        pre = (t < -float(pre_gap)) & (t > -float(pre_years[1]) * DAYS_PER_YEAR)
        post = (t >= float(window[0])) & (t <= float(window[1]))
        status = "ok"
        if pre.sum() < float(pre_years[0]) * DAYS_PER_YEAR * 0.8:
            status = f"only {int(pre.sum())} pre-event days"
        elif post.sum() < int(min_days):
            status = f"only {int(post.sum())} post-event days"
        if status != "ok":
            rows.append(dict(station=name, status=status, days=int(post.sum())))
            continue
        antennas = _antenna_days(steps, name) + 0.5 - float(event_mjd)
        quakes = _earthquake_days(steps, name, min_magnitude) + 0.5 - float(event_mjd)
        pre_steps = np.unique(np.r_[antennas[antennas < -float(pre_gap)], quakes[quakes < -float(pre_gap)]])
        residuals, failed = {}, None
        for column in columns:
            try:
                coefficients, _ = fit_trajectory(t[pre], table[column].values[pre], pre_steps)
            except ValueError as err:
                failed = str(err)
                break
            residuals[column] = table[column].values - trajectory_design(t, pre_steps) @ coefficients
        if failed:
            rows.append(dict(station=name, status=failed, days=int(post.sum())))
            continue
        for time in antennas[(antennas > float(window[0])) & (antennas <= float(window[1]))]:
            for column in columns:
                before = residuals[column][(t < time) & (t >= time - step_days)]
                after = residuals[column][(t >= time) & (t < time + step_days)]
                if before.size >= 3 and after.size >= 3:
                    residuals[column] = residuals[column] - (np.median(after) - np.median(before)) * (t >= time)
        t_days[name] = (t, pre, post, table["mjd"].values, table["lon"].values[0], table["lat"].values[0], residuals)

    mode = None
    usable_reference = [r for r in reference if r in t_days]
    if usable_reference:
        mode = common_mode({r: (t_days[r][3], *[t_days[r][6][c] for c in columns]) for r in usable_reference})
    for name, (t, pre, post, mjd, lon, lat, residuals) in t_days.items():
        corrected = {}
        for k, column in enumerate(columns):
            value = residuals[column]
            if mode is not None:
                shift = pd.Series(mode[k + 1], index=mode[0]).reindex(mjd).fillna(0.0).values
                value = value - shift
            corrected[column] = value
        sigmas, walks, reason = [], [], None
        for column, limit in zip(columns, max_scatter):
            sigma, walk = noise_levels(t[pre], corrected[column][pre])
            sigmas.append(sigma)
            walks.append(walk)
            if sigma > limit:
                reason = f"{column} noise {1e3 * sigma:.1f} mm"
        row = dict(station=name, status=reason or "ok", days=int(post.sum()), sigma_e=sigmas[0], sigma_n=sigmas[1], sigma_u=sigmas[2],
                   walk_e=walks[0], walk_n=walks[1], walk_u=walks[2], lon=lon, lat=lat)
        rows.append(row)
        if reason is None:
            prepared[name] = StationSeries(name, lon, lat, t[post], *[corrected[c][post] for c in columns], sigmas, walks)
    return prepared, pd.DataFrame(rows)
