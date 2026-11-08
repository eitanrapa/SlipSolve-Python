"""Slip deficit from interseismic velocities: rigid blocks plus back-slip.

Between earthquakes the surface moves as rigid blocks, with the elastic effect of the
slip that the locked parts of faults are *not* taking subtracted (Savage 1983):

```text
v(x) = rotation(block of x) + G(x) back_slip,    back_slip = -coupling * long_term_slip
```

A velocity is an ordinary row, so the solve is `SlipInversion`'s
with three additions: rotation columns per block (`rotation_columns`, unbounded, as a
ramp is), a bound on each node from its long-term slip rate (`coupling_bounds`), and
the back-slip sign kept in one place. Every length is metres and every time a year: `los`
in m/yr, the slip fields in m/yr, `moment` in N m/yr.

The identity holds in a layered crust too: uniform slip on a fault running through the
whole medium is a rigid offset with no strain, whatever the rigidities. That is checked
against the arctangent of Savage & Burford (1973) in the tests.

**Sign.** Positive strike-slip is left-lateral, and a slip vector is what the block on
the trace's left does relative to the block on its right. So on a right-lateral fault the
long-term strike-slip is negative and the back-slip positive. `coupling` is
`-(s V_s + d V_d) / |V|^2`: 1 where fully locked, 0 where creeping at the long-term rate.

**Rotations** are parameterised at each block's centre, where the three coefficients are
east velocity, north velocity (m/yr at the centre) and spin (m/yr per 1000 km,
anticlockwise from above), rather than as an Euler vector's Cartesian components: over a
small block those three columns are nearly parallel, and these are not. `EulerPole`
converts, exactly.

**Networks.** A fault may separate different blocks along its length: `faults` takes a list
of `(arc_start, arc_end, left, right)` per sheet of a joined mesh, and a block pair that
changes at a junction changes the rate there. Rates close at a junction by construction,
since they are all differences of the same rotations. Solve with `sides=("bottom",)` so that
coupling stays free at the sheet ends a junction joins.

**Intra-block strain** (`strain=`): three more coefficients per block, a uniform strain rate
about its centre (`exx`, `exy`, `eyy`, m/yr per 1000 km, so 1 is 1e-6 per year). A fault's
long-term rate is the difference of the *whole* field on its two sides, strain included,
because the rotation alone is not defined: moving the strain's reference point moves motion
between it and the rotation, and the total at the fault does not change. (Blocks of Meade &
Loveless 2009 are the other choice, rotation only; there the rate depends on that point.)
Strain and fault slip trade off: the more of a block's gradient is strain, the less the
faults need. Compare with and without, and limit it with `nuisance_prior` if it runs away.

**Free InSAR planes and block rotations trade off**, since a rotation over one track is
nearly a plane. GNSS is what separates them: solve them jointly, and read
`ramp="linear"` as "tied to GNSS by being in the same system", or tighten it with
`nuisance_prior`. A free plane per track can absorb most of a locked fault's signal as well,
so an interseismic inversion on a plane per track rests on the GNSS and the near field.

A signal of known shape, such as a forward model's prediction of a past earthquake's relaxation,
goes in beside the back-slip as a pattern with a free amplitude (`patterns=`, `slipsolve.patterns`).

**Not here:** tensile slip (a fault-normal relative rate on a vertical fault gets no back-slip).
"""

import warnings

import numpy as np

from .basis import make_basis
from .greens import HalfSpaceTDE
from .inversion import SlipInversion, SlipModel, _far_enough, assemble_design, pinned_edges
from .patterns import PREFIX as PATTERN_PREFIX
from .patterns import amplitudes
from .regularize import slip_bounds
from .sampling import frame_looks

#: WGS84, for the positions a rotation acts on.
_A = 6378137.0
_E2 = 6.69437999014e-3

#: The radius the east and north velocity coefficients are referred to.
EARTH_RADIUS = 6371e3

#: The length a block's spin coefficient is per: m/yr of velocity per this many metres.
SPIN_LENGTH = 1e6

#: rad/yr per deg/Myr.
_DEG_PER_MYR = np.radians(1.0) / 1e6

#: A block's strain coefficients.
STRAIN_COMPONENTS = ("exx", "exy", "eyy")

#: Slack, metres, when segments of a fault must meet and cover it.
_ARC_TOLERANCE = 1.0

#: How many of a block's nearest rows say which side of a segment it lies on.
_SIDE_ROWS = 10

#: Faults steeper than this have no dip-slip long-term rate: horizontal fault-normal motion
#: across a near-vertical plane would be a dip-slip of `rate / cos(dip)`, unbounded.
MAX_DIP_FOR_DIP_SLIP = 85.0


def _axes(lon, lat):
    """Geodetic east, north and up unit vectors in Earth-centred axes, `(n, 3)` each."""
    lon, lat = np.radians(lon), np.radians(lat)
    slon, clon, slat, clat = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    return (np.column_stack([-slon, clon, np.zeros_like(lon)]),
            np.column_stack([-slat * clon, -slat * slon, clat]),
            np.column_stack([clat * clon, clat * slon, slat]))


def _position(lon, lat):
    """Earth-centred position on the WGS84 ellipsoid, metres, `(n, 3)`."""
    lon, lat = np.radians(lon), np.radians(lat)
    prime = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return np.column_stack([prime * np.cos(lat) * np.cos(lon), prime * np.cos(lat) * np.sin(lon),
                            prime * (1.0 - _E2) * np.sin(lat)])


class EulerPole:
    """A rigid rotation of the Earth's surface: `omega` in rad/yr, Earth-centred axes."""

    def __init__(self, omega):
        self.omega = np.asarray(omega, dtype=float).reshape(3)

    @classmethod
    def from_pole(cls, lon, lat, rate):
        """From a pole's position (degrees) and rate (deg/Myr, positive anticlockwise)."""
        lon, lat = np.radians(float(lon)), np.radians(float(lat))
        return cls(float(rate) * _DEG_PER_MYR * np.array([np.cos(lat) * np.cos(lon),
                                                          np.cos(lat) * np.sin(lon), np.sin(lat)]))

    @property
    def rate(self):
        """Deg/Myr."""
        return float(np.linalg.norm(self.omega) / _DEG_PER_MYR)

    @property
    def lon(self):
        """Longitude of the pole, degrees."""
        return float(np.degrees(np.arctan2(self.omega[1], self.omega[0])))

    @property
    def lat(self):
        """Latitude of the pole, degrees."""
        return float(np.degrees(np.arcsin(self.omega[2] / np.linalg.norm(self.omega))))

    def velocity(self, lon, lat):
        """East, north and up velocity, m/yr, at geodetic positions."""
        lon, lat = np.atleast_1d(np.asarray(lon, dtype=float)), np.atleast_1d(np.asarray(lat, dtype=float))
        v = np.cross(self.omega, _position(lon, lat))
        return tuple(np.einsum("ij,ij->i", v, axis) for axis in _axes(lon, lat))

    def __sub__(self, other):
        return EulerPole(self.omega - other.omega)

    def __repr__(self):
        return f"<EulerPole {self.lon:.3f} E {self.lat:.3f} N, {self.rate:.4f} deg/Myr>"


def block_labels(obs, polygons, default=None):
    """Each row's block, from `{name: (lons, lats)}` polygons; `(n_obs,)` of str.

    A row in no polygon takes `default`, or raises naming how many there are. Later
    polygons win where they overlap.
    """
    ds = obs.ds
    lon, lat = obs.frame.to_lonlat(ds["x"].values.astype(float), ds["y"].values.astype(float))
    out = np.full(obs.n, "" if default is None else str(default), dtype=object)
    for name, (lons, lats) in polygons.items():
        px, py = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
        inside = np.zeros(obs.n, dtype=bool)
        for x0, y0, x1, y1 in zip(px, py, np.roll(px, -1), np.roll(py, -1)):
            if y0 == y1:
                continue
            crosses = (y0 > lat) != (y1 > lat)
            inside ^= crosses & (lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0))
        out[inside] = str(name)
    missing = int(np.sum(out == ""))
    if missing:
        raise ValueError(f"{missing} rows fall in no block polygon; give default= or extend them")
    return out.astype(str)


def side_labels(obs, trace, left, right):
    """Each row's block by the side of `trace` it lies on (`FaultTrace.side`)."""
    side = trace.side(obs.ds["x"].values.astype(float), obs.ds["y"].values.astype(float), obs.frame)
    return np.where(side >= 0, str(left), str(right))


def rotation_columns(obs, blocks, strain=()):
    """`(matrix, labels, centres)`: three rigid-rotation columns per block, and strain where asked.

    `blocks` is one label per row. Each row's value is the block's rigid velocity
    projected on the row's look, in geographic axes; the labels are `"block:{name}:ve"`,
    `":vn"` and `":spin"`, and `centres` is `{name: [lon, lat]}`, where the
    coefficients are referred to. `strain` names the blocks that also get `":exx"`,
    `":exy"` and `":eyy"` (`True`: all), which follow all the rotation columns.
    """
    blocks = np.asarray(blocks, dtype=str)
    if blocks.size != obs.n:
        raise ValueError(f"blocks has {blocks.size} labels for {obs.n} rows")
    ds = obs.ds
    lon, lat = obs.frame.to_lonlat(ds["x"].values.astype(float), ds["y"].values.astype(float))
    position = _position(lon, lat)
    east, north, up = _axes(lon, lat)
    look = (ds["look_e"].values[:, None] * east + ds["look_n"].values[:, None] * north
            + ds["look_u"].values[:, None] * up)
    names = sorted(set(blocks.tolist()))
    matrix = np.zeros((obs.n, 3 * len(names)))
    labels, centres = [], {}
    for b, name in enumerate(names):
        rows = blocks == name
        if rows.sum() < 3:
            raise ValueError(f"block {name!r} has {int(rows.sum())} rows; three rotation "
                             "parameters need at least three")
        axes = _rotation_axes(position[rows])
        centres[name] = _centre(position[rows])
        for j, omega in enumerate(axes):
            matrix[rows, 3 * b + j] = np.einsum("ij,ij->i", np.cross(omega, position[rows]), look[rows])
        labels += [f"block:{name}:{c}" for c in ("ve", "vn", "spin")]
    strained = names if strain is True else list(strain)
    unknown = sorted(set(strained) - set(names))
    if unknown:
        raise ValueError(f"strain names blocks with no rows: {unknown}; the blocks are {names}")
    columns = []
    for name in strained:
        rows = blocks == name
        if rows.sum() < 6:
            raise ValueError(f"block {name!r} has {int(rows.sum())} rows; rotation and strain need at least six")
        for column in _strain_vectors(centres[name], position):
            values = np.zeros(obs.n)
            values[rows] = np.einsum("ij,ij->i", column[rows], look[rows])
            columns.append(values)
        labels += [f"block:{name}:{c}" for c in STRAIN_COMPONENTS]
    if columns:
        matrix = np.column_stack([matrix, *columns])
    return matrix, labels, centres


def _strain_vectors(centre, position):
    """The velocity, per unit coefficient, of each strain component at `position`: three `(n, 3)`.

    In the tangent plane at the centre, whose east and north axes a position's offset is read
    along. `exy` is the symmetric shear: it holds no rotation.
    """
    lon, lat = np.radians(centre)
    r = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(r, east)
    dx, dy = (position @ east)[:, None] / SPIN_LENGTH, (position @ north)[:, None] / SPIN_LENGTH
    return dx * east, dy * east + dx * north, dy * north


def _centre(position):
    """A set of positions' mean direction, as `[lon, lat]` (geocentric latitude)."""
    r = position.mean(axis=0)
    r = r / np.linalg.norm(r)
    return [float(np.degrees(np.arctan2(r[1], r[0]))), float(np.degrees(np.arcsin(r[2])))]


def _axes_at(centre):
    """The rotation vectors, rad/yr per unit coefficient, of a block centred on `[lon, lat]`."""
    lon, lat = np.radians(centre)
    r = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(r, east)
    # omega x (R r) = v at the centre needs omega = (r x v) / R, and r x east = north.
    return north / EARTH_RADIUS, -east / EARTH_RADIUS, r / SPIN_LENGTH


def _rotation_axes(position):
    return _axes_at(_centre(position))


def poles_from_coefficients(labels, values, centres):
    """`{block: EulerPole}` from the solved rotation coefficients."""
    values = dict(zip(labels, np.asarray(values, dtype=float)))
    poles = {}
    for name, centre in centres.items():
        axes = _axes_at(centre)
        poles[name] = EulerPole(sum(values[f"block:{name}:{c}"] * axis
                                    for c, axis in zip(("ve", "vn", "spin"), axes)))
    return poles


def strain_from_coefficients(labels, values):
    """`{block: (exx, exy, eyy)}` for the blocks that have strain, m/yr per 1000 km."""
    values = dict(zip(labels, np.asarray(values, dtype=float)))
    names = sorted({str(label).split(":")[1] for label in values if str(label).endswith(":exx")})
    return {name: tuple(float(values[f"block:{name}:{c}"]) for c in STRAIN_COMPONENTS) for name in names}


def _fault_segments(faults, labels, s):
    """`faults` as one list of `[arc_start, arc_end, left, right]` per sheet, each covering its sheet."""
    entries = [faults] if isinstance(faults[0], str) else list(faults)
    n_faults = int(labels.max()) + 1
    if len(entries) != n_faults:
        raise ValueError(f"faults has {len(entries)} entries for {n_faults} joined faults")
    out = []
    for f, entry in enumerate(entries):
        arc = s[labels == f]
        first, last = float(arc.min()), float(arc.max())
        if isinstance(entry[0], str):
            if len(entry) != 2:
                raise ValueError(f"fault {f}: expected (left, right), got {entry!r}")
            out.append([[first, last, str(entry[0]), str(entry[1])]])
            continue
        segments = []
        for start, end, left, right in entry:
            segments.append([first if start is None else max(float(start), first),
                             last if end is None else min(float(end), last), str(left), str(right)])
        segments.sort(key=lambda segment: segment[0])
        if segments[0][0] > first + _ARC_TOLERANCE or segments[-1][1] < last - _ARC_TOLERANCE:
            raise ValueError(f"fault {f}: its segments cover arc {segments[0][0]:.0f} to {segments[-1][1]:.0f} m "
                             f"of {first:.0f} to {last:.0f}; give (left, left) to a stretch with no relative motion")
        for one, other in zip(segments, segments[1:]):
            if abs(one[1] - other[0]) > _ARC_TOLERANCE:
                raise ValueError(f"fault {f}: segments end at {one[1]:.0f} m and start at {other[0]:.0f} m; they must meet")
        out.append(segments)
    return out


def coupling_bounds(long_term, coupling=(0.0, 1.0)):
    """`{"strike": (lo, hi), "dip": (lo, hi)}` limits on the back-slip, per parameter.

    `long_term` is `(V_s, V_d)`; `coupling` the range allowed. The box is the corners'
    back-slip, so a node's rake is not pinned to its long-term rake: the components are
    bounded separately. A component with no long-term rate has coinciding limits, so it
    is fixed at zero, not solved for: on a vertical fault there is no dip-slip deficit.
    """
    out = {}
    for key, rate in zip(("strike", "dip"), long_term):
        rate = np.asarray(rate, dtype=float)
        a, b = -float(coupling[0]) * rate, -float(coupling[1]) * rate
        out[key] = (np.minimum(a, b), np.maximum(a, b))
    return out


class InterseismicInversion(SlipInversion):
    """Rigid blocks plus back-slip on the faults of `mesh`, from velocities in m/yr.

    `blocks` labels each row of `obs` with its block (`block_labels`,
    `side_labels`), and adds `rotation_columns`. `faults` says which blocks a
    fault separates: `(left, right)` looking along its trace, or one entry per fault of a
    joined mesh (`FaultMesh.join`), each `(left, right)` or, where the
    pair changes along the fault, a list of `(arc_start, arc_end, left, right)` covering it
    (`None` for the fault's own end). With both, `solve` takes each node's long-term
    rate from the blocks' solved motion. Without them, pass `rate=`, having removed the rigid
    motion from the data yourself. `strain` adds a uniform strain rate to the blocks it names
    (`True`: all). `patterns` adds velocity signals of known shape with free amplitudes
    (`Pattern`, whose `time` is ignored here).

    `ramp` is per track, as ever: `{"gnss": "none", "*": "linear"}`. `nuisance_prior`
    limits a ramp or a rotation; see `SlipInversion`, and note
    that it needs rows weighted by one over their noise (`normalize="none"`).

    `frame_looks` rotates look vectors into the frame's axes for the Green's functions
    (`slipsolve.sampling.frame_looks`), which matters on a block-model scale.
    `basis` defaults to `"node"`.

    Blocked cross-validation scores in units of each row's noise (`cv_rms` of 1 is the
    noise level): GNSS and InSAR velocities have noises a factor of a few apart. Pass
    `rate=` through it (`blocked_cv_curve(inv, weights, rate=...)`) to keep the sweep
    cheap; without it every fold re-settles its own blocks and bounds.
    """

    cv_in_noise_units = True

    def __init__(self, mesh, obs, blocks=None, faults=None, engine=None, ramp="none", basis="node",
                 velocity_model=None, covariance=None, nuisance_prior=None, frame_looks=True, strain=(), patterns=()):
        self.blocks = None if blocks is None else np.asarray(blocks, dtype=str)
        extra, self.block_centres = None, {}
        if self.blocks is not None:
            matrix, labels, self.block_centres = rotation_columns(obs, self.blocks, strain=strain)
            extra = (matrix, labels)
        elif strain:
            raise ValueError("strain names blocks, so it needs blocks=")
        # A velocity pattern with a free amplitude: one column.
        self.patterns = list(patterns)
        for pattern in self.patterns:
            if pattern.frame is not None:
                mesh.frame.require_match(pattern.frame, f"The pattern {pattern.name!r}")
        if self.patterns:
            looks = obs.frame_looks() if frame_looks else tuple(obs.ds[k].values for k in ("look_e", "look_n", "look_u"))
            x, y = obs.ds["x"].values.astype(float), obs.ds["y"].values.astype(float)
            columns = [pattern.values(x, y, *looks).sum(axis=1) for pattern in self.patterns]
            names = [f"{PATTERN_PREFIX}{pattern.name}" for pattern in self.patterns]
            extra = ((np.column_stack(columns), names) if extra is None
                     else (np.column_stack([extra[0], *columns]), list(extra[1]) + names))
        engine = engine or HalfSpaceTDE()
        g = None
        if frame_looks:
            ds = obs.ds
            g = assemble_design(engine, mesh, make_basis(mesh, basis),
                                (ds["x"].values, ds["y"].values, *obs.frame_looks()))
        self.use_frame_looks = bool(frame_looks)
        super().__init__(mesh, obs, engine=engine, ramp=ramp, basis=basis, velocity_model=velocity_model,
                         covariance=covariance, extra_columns=extra, nuisance_prior=nuisance_prior, g=g)
        self.boundary_scale = pinned_edges(self.w)
        self.faults = None
        if faults is not None:
            if blocks is None:
                raise ValueError("faults names blocks, so it needs blocks=")
            self.faults = _fault_segments(faults, self.fault_labels(), self._basis_arc())
            unknown = sorted({name for segments in self.faults for segment in segments for name in segment[2:]}
                             - set(self.block_centres))
            if unknown:
                raise ValueError(f"faults names blocks with no rows: {unknown}; "
                                 f"the blocks are {sorted(self.block_centres)}")
            self._check_sides()

    def _basis_arc(self):
        """Each basis function's arc length along its fault."""
        params = self.mesh.params if self.basis.name == "node" else self.mesh.element_params
        return params[:, 0]

    def _check_sides(self):
        """Warn if a segment's `(left, right)` looks swapped: the sign of every rate rides on it.

        Read from the block's nearest rows, since a long block can bend around a fault so that
        its mean position lies past it.
        """
        ds = self.obs.ds
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
        element_faults = np.repeat(np.arange(len(self.faults)),
                                   np.asarray(self.mesh.attrs.get("joined_elements", [self.mesh.n_elements])))
        arc = self.mesh.element_params[:, 0]
        for f, segments in enumerate(self.faults):
            for start, end, left, right in segments:
                on = (element_faults == f) & (arc >= start) & (arc <= end)
                if not on.any() or left == right:
                    continue
                normal = self.mesh.normals[on][:, :2].mean(axis=0)
                centre = self.mesh.centroids[on][:, :2].mean(axis=0)
                for name, sign in ((left, 1.0), (right, -1.0)):
                    rows = np.flatnonzero(self.blocks == name)
                    nearest = rows[np.argsort(np.hypot(x[rows] - centre[0], y[rows] - centre[1]))[:_SIDE_ROWS]]
                    if sign * np.median((x[nearest] - centre[0]) * normal[0] + (y[nearest] - centre[1]) * normal[1]) < 0:
                        warnings.warn(f"fault {f}, arc {start / 1e3:.0f}-{end / 1e3:.0f} km: block {name!r} was given as "
                                      f"the {'left' if sign > 0 else 'right'} of the trace, but its nearest rows lie "
                                      "on the other side; check (left, right)", RuntimeWarning, stacklevel=3)

    def _fault_axes(self):
        """Each basis function's map position and unit strike and dip vectors."""
        mesh = self.mesh
        normals = mesh.normals
        strike = np.cross([0.0, 0.0, 1.0], normals)
        strike = strike / np.linalg.norm(strike, axis=1, keepdims=True)
        dip = np.cross(normals, strike)
        if self.basis.name == "element":
            return mesh.centroids, strike, dip
        scatter = self.basis.projection().T
        weighted = [np.asarray(scatter @ (mesh.areas[:, None] * v)) for v in (strike, dip)]
        strike, dip = (v / np.linalg.norm(v, axis=1, keepdims=True) for v in weighted)
        return mesh.nodes, strike, dip

    def _segment_masks(self):
        """`(fault, start, end, left, right, on)` per segment: `on` picks its basis functions, each in one segment."""
        labels = self.fault_labels()
        arc = self._basis_arc()
        for f, segments in enumerate(self.faults):
            for k, (start, end, left, right) in enumerate(segments):
                on = (labels == f) & (arc >= start) & ((arc < end) if k < len(segments) - 1 else np.ones_like(arc, dtype=bool))
                yield f, start, end, left, right, on

    def _block_velocity(self, name, poles, strain, lon, lat):
        """A block's whole long-term field at geodetic positions: east and north, m/yr."""
        ve, vn, _ = poles[name].velocity(lon, lat)
        if strain and name in strain:
            position = _position(lon, lat)
            east, north, _ = _axes(lon, lat)
            vectors = _strain_vectors(self.block_centres[name], position)
            velocity = sum(value * vector for value, vector in zip(strain[name], vectors))
            ve = ve + np.einsum("ij,ij->i", velocity, east)
            vn = vn + np.einsum("ij,ij->i", velocity, north)
        return ve, vn

    def long_term_slip(self, poles, strain=None):
        """`(V_s, V_d)`, m/yr per basis function: the slip the blocks' relative motion asks of each.

        The left block's velocity minus the right's, at the basis function's map position, in
        the frame's axes, `strain` (`{block: (exx, exy, eyy)}`) included where given. `V_s`
        is its component along the element's own strike vector, so negative on a right-lateral
        fault. `V_d` is the dip-slip whose horizontal part matches the across-strike component,
        and zero on faults steeper than `MAX_DIP_FOR_DIP_SLIP`.
        """
        if self.faults is None:
            raise ValueError("long_term_slip needs faults= (or pass rate= to solve)")
        points, strike, dip = self._fault_axes()
        frame = self.mesh.frame
        lon, lat = frame.to_lonlat(points[:, 0], points[:, 1])
        relative = np.zeros((points.shape[0], 2))
        for _, _, _, left, right, on in self._segment_masks():
            if left == right or not on.any():
                continue
            ve_left, vn_left = self._block_velocity(left, poles, strain, lon[on], lat[on])
            ve_right, vn_right = self._block_velocity(right, poles, strain, lon[on], lat[on])
            ve, vn = ve_left - ve_right, vn_left - vn_right
            vx, vy, _ = frame_looks(frame, points[on, 0], points[on, 1], ve, vn, np.zeros_like(ve))
            relative[on, 0], relative[on, 1] = vx, vy
        v_s = np.einsum("ij,ij->i", relative, strike[:, :2])
        horizontal = np.sum(dip[:, :2] ** 2, axis=1)
        flat_enough = horizontal > np.cos(np.radians(MAX_DIP_FOR_DIP_SLIP)) ** 2
        v_d = np.zeros_like(v_s)
        v_d[flat_enough] = (np.einsum("ij,ij->i", relative[flat_enough], dip[flat_enough, :2])
                            / horizontal[flat_enough])
        return v_s, v_d

    def rigid_poles(self, far=None):
        """`{block: EulerPole}` fitted with no fault at all: the first half of `rigid_motion`."""
        return self.rigid_motion(far)[0]

    def rigid_motion(self, far=None):
        """`(poles, strain)` fitted with no fault at all, to rows far from the faults.

        The starting point of `solve`, and the two arguments of `long_term_slip`.
        `far` is the distance from the fault's shallowest nodes, default twice its depth; all
        rows if that leaves a block too few.
        """
        if self.blocks is None:
            raise ValueError("rigid_motion needs blocks=")
        ds = self.obs.ds
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
        nodes = self.mesh.nodes
        top = nodes[nodes[:, 2] >= nodes[:, 2].max() - 1.0]
        distance = 2.0 * float(-nodes[:, 2].min()) if far is None else float(far)
        keep = _far_enough(x, y, top[:, :2], top[:, :2], distance)      # zero-length segments: the nodes
        strained = strain_from_coefficients(self.ramp_labels, np.zeros(len(self.ramp_labels)))
        if any(np.sum(keep & (self.blocks == name)) < (6 if name in strained else 3) for name in self.block_centres):
            keep = np.ones_like(keep)
        design = self.whiten(self.ramp)[keep]
        data = self.whiten(self.d)[keep]
        coefficients = np.linalg.lstsq(design, data, rcond=None)[0]
        return (poles_from_coefficients(self.ramp_labels, coefficients, self.block_centres),
                strain_from_coefficients(self.ramp_labels, coefficients))

    def limits_for(self, x, coupling):
        """`(lo, hi, long_term)`: the limits on the whole parameter vector that a solution's rotations imply.

        What one round of `solve` applies, for anything re-solving it on new data
        (`monte_carlo`).
        """
        poles = poles_from_coefficients(self.ramp_labels, x[self.n_slip:], self.block_centres)
        long_term = self.long_term_slip(poles, strain_from_coefficients(self.ramp_labels, x[self.n_slip:]))
        limits = coupling_bounds(long_term, coupling)
        lo, hi = slip_bounds(self.basis.n_basis, limits["strike"], limits["dip"], None)
        return (np.concatenate([lo, np.full(self.n_ramp, -np.inf)]),
                np.concatenate([hi, np.full(self.n_ramp, np.inf)]), long_term)

    def solve(self, smoothing=0.3, coupling=(0.0, 1.0), rate=None, iterations=6, tolerance=1e-5,
              method="active_set", strike=(-10.0, 10.0), dip=(-10.0, 10.0), **kwargs):
        """Solve for the back-slip within a coupling range; returns an `InterseismicModel`.

        `coupling` is the range allowed, default fully creeping to fully locked; `None`
        leaves the back-slip to `strike` and `dip`. `rate` is `(V_s, V_d)`, m/yr,
        scalars or one per basis function: the long-term slip, fixed. `sides=("bottom",)`
        frees the fault's ends, for ends a junction joins.

        Without `rate`, the long-term slip comes from the blocks, and the bound that
        uses it depends on the rotations being solved for. So this alternates: rotations
        give bounds, a solve gives rotations, starting from `rigid_poles`, until no
        node's long-term rate moves by more than `tolerance` (m/yr) or `iterations` run out.
        The rate reported is the one the final solve was bounded by.
        """
        def attempt(long_term):
            limits = (dict(strike=strike, dip=dip) if coupling is None or long_term is None
                      else coupling_bounds(long_term, coupling))
            return super(InterseismicInversion, self).solve(smoothing=smoothing, method=method, **limits, **kwargs)

        if rate is not None:
            n = self.basis.n_basis
            long_term = (np.broadcast_to(np.asarray(rate[0], dtype=float), (n,)).copy(),
                         np.broadcast_to(np.asarray(rate[1], dtype=float), (n,)).copy())
            model = attempt(long_term)
            rounds = 1
        elif self.faults is None:
            if coupling is not None:
                raise ValueError("a coupling bound needs a long-term rate: pass rate=, or blocks= and faults=")
            long_term, model, rounds = None, attempt(None), 1
        elif coupling is None:
            model = attempt(None)
            long_term, rounds = self.long_term_slip(model.poles, model.strain), 1
        else:
            long_term = self.long_term_slip(*self.rigid_motion())
            for rounds in range(1, int(iterations) + 1):
                model = attempt(long_term)
                updated = self.long_term_slip(model.poles, model.strain)
                change = max(float(np.abs(updated[i] - long_term[i]).max()) for i in (0, 1))
                if change <= tolerance:
                    break
                if rounds < int(iterations):
                    long_term = updated
            else:
                warnings.warn(f"the block rotations and coupling bounds had not settled after "
                              f"{rounds} rounds: the long-term rate last moved {1e3 * change:.3f} mm/yr",
                              RuntimeWarning, stacklevel=2)
        model.long_term_slip = long_term
        model.options.update(
            coupling=None if coupling is None else [float(c) for c in coupling], rounds=int(rounds),
            rate_fixed=rate is not None,
            long_term_slip=None if long_term is None else [v.tolist() for v in long_term])
        return model

    def _model(self, x, result, smoothing, smooth_matrix, options):
        options = dict(options)
        options.update(block_centres=self.block_centres, frame_looks=self.use_frame_looks,
                       faults=self.faults, boundary_scale=self.boundary_scale)
        return InterseismicModel(self, x, result, smoothing, smooth_matrix, options)

    def __repr__(self):
        pairs = None if self.faults is None else [[tuple(segment[2:]) for segment in segments] for segments in self.faults]
        return (f"<InterseismicInversion obs={self.obs.n} blocks={sorted(self.block_centres)} "
                f"faults={pairs} params={self.n_param}>")


class InterseismicModel(SlipModel):
    """A solved `InterseismicInversion`.

    `strike_slip` and `dip_slip` are the **back-slip**
    (m/yr); `moment()` is the moment deficit rate, N m/yr. `moment_magnitude` is one
    year's worth, so rarely worth reading.
    """

    long_term_slip = None

    @classmethod
    def load(cls, path):
        """Read back a model written by `save`."""
        self = super().load(path)
        saved = self.options.get("long_term_slip")
        self.long_term_slip = None if saved is None else tuple(np.asarray(v, dtype=float) for v in saved)
        return self

    @property
    def coupling(self):
        """Back-slip over long-term slip per basis function; 1 locked, 0 creeping, NaN with no rate."""
        if self.long_term_slip is None:
            raise ValueError("this model has no long-term rate: solve with rate=, or blocks= and faults=")
        v_s, v_d = self.long_term_slip
        squared = v_s ** 2 + v_d ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(squared > 0, -(self.strike_slip * v_s + self.dip_slip * v_d) / squared, np.nan)

    @property
    def deficit_rate(self):
        """The back-slip's magnitude per basis function, m/yr."""
        return self.slip_magnitude

    def _rotation_slice(self):
        names = [i for i, label in enumerate(self.ramp_labels) if str(label).startswith("block:")]
        return np.asarray(names, dtype=int)

    @property
    def poles(self):
        """`{block: EulerPole}`: the solved rotations."""
        rows = self._rotation_slice()
        return poles_from_coefficients([self.ramp_labels[i] for i in rows], self.ramp[rows],
                                       self.options.get("block_centres") or {})

    @property
    def pattern_amplitudes(self):
        """`{name: amplitude}` of the patterns solved for beside the blocks and the back-slip."""
        return amplitudes(self.ramp_labels, self.ramp)

    @property
    def strain(self):
        """`{block: (exx, exy, eyy)}`: the solved strain rates, m/yr per 1000 km (1 is 1e-6 per year)."""
        rows = self._rotation_slice()
        return strain_from_coefficients([self.ramp_labels[i] for i in rows], self.ramp[rows])

    @property
    def rotation_prediction(self):
        """The blocks' long-term motion alone, per observation: rotation, and strain where there is any."""
        rows = self._rotation_slice()
        return self.ramp_design[:, rows] @ self.ramp[rows]

    def fault_table(self):
        """One row per fault segment: its arc range, blocks, long-term rates, coupling and moment deficit.

        Rates in mm/yr, in the package's sign (negative strike-slip is right-lateral);
        `coupling` is the mean over all of the segment's nodes, whatever their depth, and
        `moment` the deficit rate of the back-slip on them, N m/yr. A saved model reads this
        only with its inversion.
        """
        import pandas as pd

        from .uncertainty import with_solution

        inversion = self.inversion
        if inversion is None or inversion.faults is None:
            raise ValueError("fault_table needs the model's inversion, with faults=")
        v_s, v_d = self.long_term_slip
        coupling = self.coupling
        n = self.basis.n_basis
        rows = []
        for fault, start, end, left, right, on in inversion._segment_masks():
            mask = np.r_[on, on].astype(float)
            x = np.r_[self.x[:2 * n] * mask, self.x[2 * n:]]
            rows.append(dict(fault=fault, arc_start_km=start / 1e3, arc_end_km=end / 1e3, left=left, right=right,
                             strike_rate=1e3 * float(v_s[on].mean()) if on.any() else np.nan,
                             strike_rate_min=1e3 * float(v_s[on].min()) if on.any() else np.nan,
                             strike_rate_max=1e3 * float(v_s[on].max()) if on.any() else np.nan,
                             dip_rate=1e3 * float(v_d[on].mean()) if on.any() else np.nan,
                             coupling=float(np.nanmean(coupling[on])) if on.any() and np.isfinite(coupling[on]).any() else np.nan,
                             moment=with_solution(self, x).moment()))
        return pd.DataFrame(rows)

    def block_table(self):
        """One row per block: its rows, and the rms of its data, of the data less the rotation, and of the residual, mm/yr."""
        import pandas as pd

        inversion = self.inversion
        if inversion is None or inversion.blocks is None:
            raise ValueError("block_table needs the model's inversion, with blocks=")
        relative = self.data - self.rotation_prediction
        out = []
        poles = self.poles
        strain = self.strain
        for name in sorted(self.options.get("block_centres") or {}):
            rows = inversion.blocks == name
            pole = poles[name]
            out.append(dict(block=name, rows=int(rows.sum()), rms_data=1e3 * float(np.sqrt(np.mean(self.data[rows] ** 2))),
                            rms_less_rotation=1e3 * float(np.sqrt(np.mean(relative[rows] ** 2))),
                            rms_residual=1e3 * float(np.sqrt(np.mean(self.residual[rows] ** 2))),
                            pole_lon=pole.lon, pole_lat=pole.lat, pole_rate=pole.rate,
                            **{c: strain[name][k] if name in strain else np.nan for k, c in enumerate(STRAIN_COMPONENTS)}))
        return pd.DataFrame(out)

    @property
    def elastic_prediction(self):
        """The elastic effect of the back-slip alone: `fault_prediction`, by its other name."""
        return self.fault_prediction

    def __repr__(self):
        flag = "" if self.converged else " UNCONVERGED"
        return (f"<InterseismicModel rms {1e3 * self.rms_misfit:.2f} mm/yr, max back-slip "
                f"{1e3 * self.max_slip:.1f} mm/yr, moment deficit {self.moment():.3g} N m/yr{flag}>")
