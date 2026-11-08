"""EDGRN Green's-function tables: what a layered crust does to a point source.

A homogeneous half-space has one elastic modulus everywhere, so its response to a
buried source is a closed form and `slipsolve._tde` evaluates it
directly. A layered crust has no closed form: the response has to be built by
wavenumber integration through the layer stack, which is what Rongjiang Wang's
**EDGRN** (Wang, Lorenzo-Martin & Roth 2003, *Computers & Geosciences* **29**,
195-207) does, once, into a table indexed by source depth and epicentral
distance. **EDCMP** then reads that table and superposes point sources over a
fault. This module is the table half; `slipsolve.layered` is the
superposition half.

Three tables are needed because a general moment tensor decomposes into three
sources with distinct azimuthal symmetry, and the tables store each with its
azimuthal dependence stripped out:

| file | source                        | azimuthal order |
|------|-------------------------------|-----------------|
| ss   | `m12 = m21 = M0`              | 2               |
| ds   | `m13 = m31 = M0`              | 1               |
| cl   | `m33 = M0, m11 = m22 = -M0/2` | 0               |

so a table row is a function of `(r, z_source)` alone and the azimuth is put
back at evaluation time (`edgmoment.f` sets `ms = 2, 1, 0` for exactly this
reason). `slipsolve.layered` does the putting back.

**Coordinates here are EDGRN's, not the package's.** `x` is north, `y` is
east and `z` is down; `uz` in a table is positive downward. The
conversion happens at the boundary of `slipsolve.layered`, which is
also where the reference (SlipSolve-Curve) does it -- the apparent axis swap at the
top of its `calc_green_gps_3d_edcmp_xyz.m` is that conversion and not, as it looks,
a bug.

Running EDGRN itself is optional and external, exactly as in the reference: the
user owns the Earth model. `run_edgrn` will drive it through `pygrnwang`
if that is installed, and `EdgrnTables.homogeneous` synthesises a
uniform-medium table analytically so the whole layered pipeline can be tested
without any Fortran at all.
"""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

#: Source types, in EDGRN's own order (`istype` 1, 2, 3).
SOURCE_TYPES = ("ss", "ds", "cl")

#: Displacement columns of a table row, in file order.
_DISPLACEMENT_COLUMNS = ("uz", "ur", "ut")

#: Two vertex depths closer than this, in metres, take the divided difference's
#: confluent limit in `VelocityModel.triangle_mean` -- a rows mesh has a
#: horizontal edge on every element. The limit is off by `p' (y - x)**2 / 24`.
_CONFLUENT_DEPTH = 1e-3

#: A triangle whose depths span less than this, in metres, is flat: its mean is the
#: property at its mean depth. The second divided difference would otherwise divide
#: a rounding-limited numerator by a vanishing span.
_FLAT_DEPTH = 1.0


class VelocityModel:
    """A layered elastic Earth: depth, P and S velocity, density.

    The input EDGRN needs, and the source of the depth-dependent rigidity
    `mu = rho * vs**2` that a moment calculation wants. A discontinuity is
    written as two rows at the same depth; a gradient as two rows at different
    depths with different values -- EDGRN's own convention, preserved so a model
    can be round-tripped through an `edgrn.inp` unchanged.
    """

    def __init__(self, depth, vp, vs, rho, name=None):
        self.depth = np.asarray(depth, dtype=float).ravel()
        self.vp = np.asarray(vp, dtype=float).ravel()
        self.vs = np.asarray(vs, dtype=float).ravel()
        self.rho = np.asarray(rho, dtype=float).ravel()
        self.name = name
        sizes = {self.depth.size, self.vp.size, self.vs.size, self.rho.size}
        if len(sizes) != 1:
            raise ValueError("depth, vp, vs and rho must have the same length")
        if self.depth.size == 0:
            raise ValueError("A velocity model needs at least one layer")
        if np.any(np.diff(self.depth) < 0):
            raise ValueError("Velocity-model depths must be non-decreasing")
        if np.any(self.vs <= 0) or np.any(self.rho <= 0):
            raise ValueError("vs and rho must be positive")

    @classmethod
    def uniform(cls, vp=6000.0, vs=3464.0, rho=2670.0, max_depth=100e3):
        """One layer -- the model whose answer must match the half-space engine.

        The defaults give `mu = rho * vs**2` = 32.0 GPa and a Poisson ratio of
        0.25, i.e. a Poisson solid close to the 30 GPa this package assumes when
        no model is supplied.
        """
        return cls([0.0, max_depth], [vp, vp], [vs, vs], [rho, rho], name="uniform")

    @classmethod
    def from_file(cls, path):
        """Read `depth vp vs rho` rows (metres, m/s, m/s, kg/m^3)."""
        path = Path(path).expanduser()
        rows = []
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = [float(v) for v in re.split(r"[\s,]+", line)]
            # Tolerate EDGRN's own five-column form, whose first column is an index.
            rows.append(fields[-4:] if len(fields) >= 5 else fields[:4])
        table = np.asarray(rows, dtype=float)
        if table.ndim != 2 or table.shape[1] != 4:
            raise ValueError(f"{path} must have four columns: depth vp vs rho")
        return cls(*table.T, name=path.stem)

    @property
    def interfaces(self):
        """Depths where a property steps, in metres: two rows at one depth that differ.

        A gradient -- two rows at different depths with different values -- is
        continuous and is not an interface. These are the depths across which
        `EdgrnTables.interpolate` must not interpolate.
        """
        return _step_depths(self.depth, self.vp, self.vs, self.rho)

    @property
    def mu(self):
        """Shear modulus per row, `rho * vs**2`, in pascals."""
        return self.rho * self.vs ** 2

    @property
    def lam(self):
        """First Lame parameter per row, `rho * (vp**2 - 2 vs**2)`."""
        return self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)

    def poisson(self, depth=0.0):
        """Poisson's ratio at `depth`."""
        lam, mu = self.at(depth, "lam"), self.at(depth, "mu")
        return float(lam / (2.0 * (lam + mu)))

    def at(self, depth, field="mu"):
        """A property sampled at one or more depths, piecewise-linearly.

        Depths below the deepest row take the deepest row's value, which is what
        a half-space bottom layer means.
        """
        values = getattr(self, field) if field in ("vp", "vs", "rho") else getattr(self, field)
        depth = np.abs(np.asarray(depth, dtype=float))
        # np.interp already clamps outside the range at both ends.
        return np.interp(depth, self.depth, values)

    def triangle_mean(self, depths, field="mu"):
        """The exact mean of a property over planar triangles, from their vertex depths.

        Depth is affine across a planar triangle and area is uniform in its
        barycentric coordinates, so by the Hermite-Genocchi formula the mean of
        `p(z)` over the triangle is `2 P[z1, z2, z3]`: twice the second divided
        difference of `P`, the second antiderivative of `p`. `at` is
        piecewise linear in depth, with a step wherever two rows share a depth, so
        `P` is piecewise cubic and the mean is exact -- a triangle cut by an
        interface gets the area-weighted mean of the two layers.

        Sampling `at` at points inside the triangle cannot do that: which side
        of a step a point falls on decides its whole cell, so the error is first
        order with an erratic constant, and a finer sampling can be worse; this is
        exact.

        `depths` is `(..., 3)`, sign ignored like `at`; returns `(...)`.
        """
        z = np.sort(np.abs(np.asarray(depths, dtype=float)), axis=-1)
        if z.shape[-1] != 3:
            raise ValueError(
                f"depths must end in an axis of 3 vertex depths, got {z.shape}")
        knots, values, slope, first, second = self._antiderivatives(field)

        def integral(x, order):
            k = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 1)
            h = x - knots[k]
            if order == 1:
                return first[k] + values[k] * h + 0.5 * slope[k] * h ** 2
            return (second[k] + first[k] * h + 0.5 * values[k] * h ** 2
                    + slope[k] * h ** 3 / 6.0)

        def divided(x, y):
            """`P[x, y]` for `x <= y`; the confluent limit `P'` where they meet."""
            close = (y - x) <= _CONFLUENT_DEPTH
            return np.where(close, integral(0.5 * (x + y), 1),
                            (integral(y, 2) - integral(x, 2))
                            / np.where(close, 1.0, y - x))

        a, b, c = z[..., 0], z[..., 1], z[..., 2]
        flat = (c - a) <= _FLAT_DEPTH
        mean = 2.0 * (divided(b, c) - divided(a, b)) / np.where(flat, 1.0, c - a)
        return np.where(flat, self.at((a + b + c) / 3.0, field), mean)

    def _antiderivatives(self, field):
        """Knots from depth 0, the property and its slope there, and `P'` and `P`.

        The knots are the rows at or below the surface -- duplicates kept, so a step
        is a zero-width interval that adds nothing -- with depth 0 prepended at
        `at`'s own value when no row sits there. `slope[k]` is the gradient
        from knot `k` to the next, 0 past the last (the half-space).
        """
        values = np.asarray(getattr(self, field), dtype=float)
        keep = self.depth >= 0.0
        knots, values = self.depth[keep], values[keep]
        if knots.size == 0 or knots[0] > 0.0:
            knots = np.concatenate([[0.0], knots])
            values = np.concatenate([[float(self.at(0.0, field))], values])
        width = np.diff(knots)
        gradient = np.divide(np.diff(values), width, out=np.zeros_like(width),
                             where=width > 0)
        first = np.concatenate([[0.0], np.cumsum(
            values[:-1] * width + 0.5 * gradient * width ** 2)])
        second = np.concatenate([[0.0], np.cumsum(
            first[:-1] * width + 0.5 * values[:-1] * width ** 2
            + gradient * width ** 3 / 6.0)])
        return knots, values, np.append(gradient, 0.0), first, second

    def to_dict(self):
        """A JSON-serialisable form, so a model can carry its own rigidity."""
        return {"depth": self.depth.tolist(), "vp": self.vp.tolist(),
                "vs": self.vs.tolist(), "rho": self.rho.tolist(),
                "name": self.name}

    @classmethod
    def from_dict(cls, d):
        """Rebuild a model from `to_dict`'s output."""
        return cls(d["depth"], d["vp"], d["vs"], d["rho"], name=d.get("name"))

    def __len__(self):
        return self.depth.size

    def __repr__(self):
        return (f"<VelocityModel {self.name or ''} {len(self)} rows "
                f"0-{self.depth.max() / 1e3:.0f}km mu={self.mu.min() / 1e9:.1f}"
                f"-{self.mu.max() / 1e9:.1f}GPa>")


class EdgrnTables:
    """The three EDGRN databases on their shared `(r, z_source)` grid.

    `ss`, `ds` and `cl` are each a dict of `(nr, nz)` arrays keyed by
    `uz`, `ur`, `ut` -- the displacement at the receiver depth `zrec` per
    unit source area times slip, with the azimuthal factor removed. `cl`'s
    `ut` is identically zero: an axisymmetric source drives no transverse
    motion.

    `interfaces` are the depths where the crust's properties step (read from the
    input file's layer block by `from_input_file`). Per unit area times slip a
    table value jumps there by about the rigidity ratio, because the moment is
    `mu` times the potency and `mu` is the source layer's. `interpolate` therefore
    never interpolates across one; see there.
    """

    def __init__(self, r, z, tables, zrec=0.0, lam=None, mu=None, attrs=None,
                 interfaces=None):
        self.r = np.asarray(r, dtype=float).ravel()
        self.z = np.asarray(z, dtype=float).ravel()
        self.tables = {k: {c: np.asarray(v[c], dtype=float) for c in _DISPLACEMENT_COLUMNS}
                       for k, v in tables.items()}
        missing = set(SOURCE_TYPES) - set(self.tables)
        if missing:
            raise ValueError(f"Missing EDGRN table(s): {sorted(missing)}")
        shape = (self.r.size, self.z.size)
        for kind, table in self.tables.items():
            for component, values in table.items():
                if values.shape != shape:
                    raise ValueError(
                        f"{kind}.{component} has shape {values.shape}; expected {shape} "
                        "= (nr, nz)"
                    )
        self.zrec = float(zrec)
        self.lam = lam
        self.mu = mu
        self.attrs = dict(attrs or {})
        self.interfaces = np.unique(np.asarray(
            [] if interfaces is None else interfaces, dtype=float).ravel())
        if self.interfaces.size:
            self.attrs.setdefault("interfaces", self.interfaces.tolist())
        self._layers = self._layer_stencils()

    def _layer_stencils(self):
        """The depth intervals that straddle an interface, and each layer's nodes.

        `None` when no interface falls strictly inside the depth grid, which keeps
        `interpolate` on its plain path. A node lying on an interface is
        usable by neither layer: EDGRN assigns it to one side, and which side is not
        something to rely on.
        """
        z = self.z
        if z.size < 2:
            return None
        cuts = self.interfaces[(self.interfaces > z[0]) & (self.interfaces < z[-1])]
        if cuts.size == 0:
            return None
        dz = (z[-1] - z[0]) / (z.size - 1)
        on = (np.abs(z[:, None] - cuts[None, :]) <= 1e-6 * dz).any(axis=1)
        layer = np.searchsorted(cuts, z, side="right")
        straddles = on[:-1] | on[1:] | (layer[:-1] != layer[1:])
        nodes = [np.nonzero((layer == k) & ~on)[0] for k in range(cuts.size + 1)]
        return cuts, straddles, nodes

    # -- reading -----------------------------------------------------------
    @classmethod
    def from_input_file(cls, path):
        """Read the three databases an `edgrn.inp` names.

        The directory and the three filenames sit on the fifth line containing
        no `#`, single-quote delimited -- `getedgrn.m`'s rule, kept because
        it is the only thing that locates the tables. Relative directories are
        resolved against the input file rather than the working directory, which
        is why the reference has to `cd` before calling its own reader.
        """
        path = Path(path).expanduser()
        lines = [ln for ln in path.read_text().splitlines() if "#" not in ln and ln.strip()]
        if len(lines) < 5:
            raise ValueError(
                f"{path} has {len(lines)} comment-free lines; an EDGRN input file "
                "names its Green's-function files on the fifth"
            )
        quoted = re.findall(r"'([^']*)'", lines[4])
        if len(quoted) < 4:
            raise ValueError(
                f"Line 5 of {path} should quote a directory and three filenames "
                f"(ss, ds, cl); found {quoted}"
            )
        directory = (path.parent / quoted[0]).resolve()
        return cls.from_files({k: directory / name
                               for k, name in zip(SOURCE_TYPES, quoted[1:4])},
                              interfaces=_input_file_interfaces(lines))

    @classmethod
    def from_files(cls, paths, interfaces=None):
        """Read three already-located database files, keyed `ss`/`ds`/`cl`.

        The database files do not carry the layer model, so `interfaces` has to be
        given here to keep `interpolate` inside layers; `from_input_file`
        reads them from the input file's layer block.
        """
        header, tables = None, {}
        for kind in SOURCE_TYPES:
            head, columns = _read_table(Path(paths[kind]).expanduser(), kind)
            if header is None:
                header = head
            elif not np.allclose(head[:7], header[:7]):
                raise ValueError(
                    f"The {kind} database is on a different grid than the ss one "
                    f"({head[:7]} vs {header[:7]}); they must share (nr, nz)."
                )
            tables[kind] = columns

        nr, r1, r2, nz, z1, z2, zrec, lam, mu = header
        return cls(np.linspace(r1, r2, int(nr)), np.linspace(z1, z2, int(nz)),
                   tables, zrec=zrec, lam=lam, mu=mu,
                   attrs={"source": "edgrn"}, interfaces=interfaces)

    # -- synthesis ---------------------------------------------------------
    @classmethod
    def homogeneous(cls, nu=0.25, r=None, z=None, n_azimuth=16, source_size=1.0):
        """Synthesise the tables a *uniform* medium would produce.

        The point of this is testing. A layered code handed a one-layer model has
        to reproduce the homogeneous half-space answer, and EDCMP is built around
        that idea -- its own input file carries a "layered (1) or homogeneous (0)"
        switch and it ships Okada's routines for the second branch. Synthesising
        the tables here gives the same check without any Fortran, and it exercises
        every part of `slipsolve.layered`: the discretisation, the
        bilinear interpolation, the azimuthal recombination, the north/east/down
        conventions and the quadrature.

        What it does not test is the physics of layering, since the field it
        tabulates comes from `slipsolve._tde.tde_disp_hs` -- the same
        solution the engine is compared against. That is a test of the machinery,
        deliberately, and it is the part with all the transcription risk.

        Each source is built as a dislocation small enough to be a point at the
        table's own resolution, and the azimuthal orders are projected out
        numerically rather than derived by hand -- the exact inverse of the
        recombination in `slipsolve.layered`. The CLVD, which is not a
        dislocation at all, comes from three tensile sources:

            CLVD = (1/2) * [ T_z - (T_x + T_y) / 2 ]

        because a unit tensile source with normal `w` has moment tensor
        `lam*I + 2*mu*w(x)w`, so the isotropic parts cancel and what is left is
        `mu*(z(x)z - x(x)x/2 - y(x)y/2)` -- the CLVD, exactly.
        """
        from ._tde import tde_disp_hs

        r = np.linspace(0.0, 400e3, 201) if r is None else np.asarray(r, dtype=float)
        z = np.linspace(500.0, 50e3, 100) if z is None else np.asarray(z, dtype=float)
        phi = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)

        # Receivers on the free surface. EDGRN's azimuth is measured from north
        # towards east, so (north, east) = r * (cos phi, sin phi).
        rr, pp = np.meshgrid(r, phi, indexing="ij")
        north, east = rr * np.cos(pp), rr * np.sin(pp)
        obs_x, obs_y = east.ravel(), north.ravel()          # ENU east, north
        obs_z = np.zeros(obs_x.size)

        area = 0.5 * source_size ** 2
        tables = {kind: {c: np.empty((r.size, z.size)) for c in _DISPLACEMENT_COLUMNS}
                  for kind in SOURCE_TYPES}

        for j, depth in enumerate(z):
            fields = {}
            for label, normal, slip in _SOURCE_GEOMETRY:
                p1, p2, p3, components = _point_dislocation(
                    normal, slip, depth, source_size)
                ue, un, uu = tde_disp_hs(obs_x, obs_y, obs_z, p1, p2, p3,
                                         *components, nu)
                fields[label] = np.stack([ue, un, uu]) / area

            combined = {
                "ss": fields["ss"],
                "ds": fields["ds"],
                "cl": 0.5 * (fields["tz"] - 0.5 * (fields["tx"] + fields["ty"])),
            }
            for kind, enu in combined.items():
                ue, un, uu = (c.reshape(r.size, phi.size) for c in enu)
                # ENU -> EDGRN's north/east/down, then to cylindrical.
                ux, uy, uz = un, ue, -uu
                ur = ux * np.cos(phi) + uy * np.sin(phi)
                ut = -ux * np.sin(phi) + uy * np.cos(phi)
                for component, values in (("uz", uz), ("ur", ur), ("ut", ut)):
                    tables[kind][component][:, j] = _project(values, phi, kind, component)

        return cls(r, z, tables, zrec=0.0,
                   mu=None, lam=None,
                   attrs={"source": "synthetic_homogeneous", "nu": float(nu)})

    # -- use ---------------------------------------------------------------
    def interpolate(self, kind, component, distance, depth):
        """Bilinear lookup in `(r, z_source)`, EDCMP's own stencil, within layers.

        Outside the tabulated range the value is taken from the edge cell rather
        than extrapolated, at every edge: below `r1` (as EDCMP does), beyond
        `r2` (where EDCMP pads with zeros instead) and above or below the
        tabulated source depths. A source outside the table therefore takes the
        edge row's value, not zero, so the table must cover the mesh.

        **Never across an interface.** A table value steps by about the rigidity
        ratio where the source crosses one (see the class docstring), so the plain
        stencil would hand every source within one depth spacing of an interface a
        blend of both layers, which on a coarse depth grid moves the slip by several
        percent of its peak. So a query whose depth interval straddles an interface
        -- or ends on one -- takes its two depth nodes from its own layer instead,
        extrapolating linearly from the nearest two when it sits between the last
        node and the interface; a query exactly at an interface depth belongs to the
        layer below it. Every other query takes the plain stencil, bit for bit, and a
        table with no interfaces takes the plain path.
        """
        table = self.tables[kind][component]
        nr, nz = table.shape
        dr = (self.r[-1] - self.r[0]) / (nr - 1) if nr > 1 else 1.0
        dz = (self.z[-1] - self.z[0]) / (nz - 1) if nz > 1 else 1.0

        distance = np.asarray(distance, dtype=float)
        depth = np.asarray(depth, dtype=float)
        # `np.minimum(np.maximum(...))` rather than `np.clip`, for speed: this is the
        # layered engine's hot path, and `np.clip`'s per-call overhead (it builds a
        # `finfo` for its NaN handling) was a fifth of assembly time. The two are
        # bit-identical here (pinned by `test_interpolation_is_unchanged_by_the_clip_rewrite`)
        # because none of these values is ever NaN: `r` and `z` are finite by
        # construction and the caller's distances come from finite geometry.
        fi = np.minimum(np.maximum((distance - self.r[0]) / dr, 0.0), nr - 1)
        fj = np.minimum(np.maximum((depth - self.z[0]) / dz, 0.0), nz - 1)
        i0 = np.minimum(np.floor(fi).astype(int), nr - 2 if nr > 1 else 0)
        j0 = np.minimum(np.floor(fj).astype(int), nz - 2 if nz > 1 else 0)
        ti = fi - i0
        tj = fj - j0
        i1 = np.minimum(i0 + 1, nr - 1)
        j1 = np.minimum(j0 + 1, nz - 1)
        if self._layers is not None:
            j0, j1, tj = self._within_layers(fj, dz, j0, j1, tj)

        return ((1 - ti) * (1 - tj) * table[i0, j0]
                + ti * (1 - tj) * table[i1, j0]
                + (1 - ti) * tj * table[i0, j1]
                + ti * tj * table[i1, j1])

    def _within_layers(self, fj, dz, j0, j1, tj):
        """Replace the depth stencil of the queries that straddle an interface.

        Works on the few such queries only: one gather over `j0` finds them, and
        the rest of the arrays are returned as they came.
        """
        cuts, straddles, nodes = self._layers
        k = np.flatnonzero(straddles[j0])
        if k.size == 0:
            return j0, j1, tj
        shape = np.shape(tj)
        j0, j1, tj = (np.array(a, ndmin=1).reshape(-1) for a in (j0, j1, tj))
        z = self.z
        d = z[0] + np.ravel(fj)[k] * dz                  # the clamped query depth
        layer = np.searchsorted(cuts, d, side="right")
        for which in np.unique(layer):
            usable = nodes[which]
            if usable.size == 0:
                continue
            m = layer == which
            if usable.size == 1:
                j0[k[m]] = j1[k[m]] = usable[0]
                tj[k[m]] = 0.0
                continue
            pos = np.clip(np.searchsorted(z[usable], d[m]), 1, usable.size - 1)
            lo, hi = usable[pos - 1], usable[pos]
            j0[k[m]], j1[k[m]] = lo, hi
            tj[k[m]] = (d[m] - z[lo]) / (z[hi] - z[lo])
        return j0.reshape(shape), j1.reshape(shape), tj.reshape(shape)

    def __repr__(self):
        layers = f" interfaces={self.interfaces.size}" if self.interfaces.size else ""
        return (f"<EdgrnTables nr={self.r.size} r={self.r[0] / 1e3:.0f}-"
                f"{self.r[-1] / 1e3:.0f}km nz={self.z.size} "
                f"z={self.z[0] / 1e3:.1f}-{self.z[-1] / 1e3:.0f}km{layers} "
                f"{self.attrs.get('source', '?')}>")


def _step_depths(depth, vp, vs, rho):
    """Depths of the steps in a layer block: consecutive rows at one depth that differ."""
    depth = np.asarray(depth, dtype=float).ravel()
    if depth.size < 2:
        return []
    props = np.column_stack([np.ravel(vp), np.ravel(vs), np.ravel(rho)]).astype(float)
    same_depth = np.isclose(depth[1:], depth[:-1], rtol=0.0, atol=1e-6)
    differ = ~np.all(np.isclose(props[1:], props[:-1], rtol=1e-9, atol=0.0), axis=1)
    return [float(d) for d in depth[1:][same_depth & differ & (depth[1:] > 0)]]


def _input_file_interfaces(lines):
    """The interfaces of an `edgrn.inp`'s layer block; `[]` if it has none.

    `lines` are the input's comment-free lines: the sixth holds the row count and
    each row after it is `index depth vp vs rho`.
    """
    def numbers(line):
        return [float(v) for v in _fortran_floats(line).split()]

    try:
        n = int(numbers(lines[5])[0])
        rows = [numbers(line) for line in lines[6:6 + n]]
    except (IndexError, ValueError):
        return []
    rows = [row for row in rows if len(row) >= 5]
    if len(rows) < 2:
        return []
    table = np.asarray(rows, dtype=float)[:, 1:5]
    return _step_depths(*table.T)


# Dislocation geometry for each canonical source, in ENU. `normal` is the fault
# plane's unit normal and `slip` the unit slip direction of the hanging wall.
#   ss  m12 = m21: plane normal north, slip east   (a vertical east-striking fault)
#   ds  m13 = m31: plane normal north, slip down
#   tx/ty/tz     : unit tensile sources, combined into the CLVD
_SOURCE_GEOMETRY = (
    ("ss", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    ("ds", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ("tx", (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),      # normal north -> x(x)x in NED
    ("ty", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),      # normal east  -> y(x)y in NED
    ("tz", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),      # normal up    -> z(x)z in NED
)


def _point_dislocation(normal, slip, depth, size):
    """A tiny triangle with the requested normal, and its `(ss, ds, ts)`.

    Solving for the slip components rather than naming them avoids the sign trap
    in the triangular-dislocation basis: `Vstrike = cross(eZ, Vnorm)` can come
    out anti-parallel to the direction you meant, and the resulting moment tensor
    is then the negative of the one intended -- which produces a perfectly smooth,
    entirely wrong table. Projecting the desired slip vector onto the element's
    own orthonormal basis is correct whatever that basis turns out to be.
    """
    from ._tde import _element_basis

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    # Any orthonormal pair (u, v) with u x v = n makes cross(P2-P1, P3-P1) = n.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centre = np.array([0.0, 0.0, -abs(depth)])
    p1 = centre - size * (u + v) / 3.0
    p2 = p1 + size * u
    p3 = p1 + size * v

    vnorm, vstrike, vdip = _element_basis(p1, p2, p3)
    d = np.asarray(slip, dtype=float)
    d = d / np.linalg.norm(d)
    return p1, p2, p3, (float(d @ vstrike), float(d @ vdip), float(d @ vnorm))


def _project(values, phi, kind, component):
    """Strip the azimuthal factor, inverting the recombination exactly.

    `values` is `(nr, nphi)`. The factors are the ones
    `slipsolve.layered` puts back: order 2 for `ss`, 1 for `ds`,
    0 for `cl`, with `ut` carrying the quadrature partner of the `uz`/`ur`
    factor.
    """
    if kind == "cl":
        return np.zeros(values.shape[0]) if component == "ut" else values.mean(axis=1)
    order = 2 if kind == "ss" else 1
    basis = (np.cos(order * phi) if (kind == "ss") == (component == "ut")
             else np.sin(order * phi))
    return 2.0 * (values * basis).mean(axis=1)


def _read_table(path, kind):
    """One EDGRN database: its metadata line and its displacement columns.

    The column count is read from the file rather than assumed. Stock EDGRN 2.0
    writes ten columns for `ss` and `ds` but only seven for `cl`
    (`edgmain.f` drops `ut`, `ert` and `etz`, which vanish for an
    axisymmetric source), while `getedgrn.m` and SlipSolve-Curve's own converter both
    read ten and assert the row count -- so a stock `cl` file would fail there.
    Only the leading displacement columns are used either way; the six strain
    columns and `duz/dr` are read and discarded.
    """
    text = Path(path).read_text().splitlines()
    body = [ln for ln in text if ln.strip() and "#" not in ln]
    if not body:
        raise ValueError(f"{path} has no data lines")

    header = [float(v) for v in re.split(r"[\s,]+", _fortran_floats(body[0]).strip())]
    if len(header) < 9:
        raise ValueError(
            f"{path}: the metadata line should hold nr r1 r2 nz z1 z2 zrec lambda mu; "
            f"found {len(header)} values"
        )
    nr, nz = int(header[0]), int(header[3])

    rows = np.array([[float(v) for v in re.split(r"[\s,]+", _fortran_floats(ln).strip())]
                     for ln in body[1:]], dtype=float)
    if rows.shape[0] != nr * nz:
        raise ValueError(
            f"{path} has {rows.shape[0]} data rows; the header declares "
            f"nr*nz = {nr}*{nz} = {nr * nz}"
        )

    # Row order is z outer, r inner (`do izs ... do j=1,nr` in edgmain.f), which
    # is Fortran order for an (nr, nz) array.
    n_disp = 3 if rows.shape[1] >= 10 else 2
    columns = {}
    for index, name in enumerate(_DISPLACEMENT_COLUMNS):
        columns[name] = (rows[:, index].reshape(nr, nz, order="F") if index < n_disp
                         else np.zeros((nr, nz)))
    if kind == "cl":
        columns["ut"] = np.zeros((nr, nz))
    return header[:9], columns


def _fortran_floats(line):
    """Turn Fortran's `1.0D+03` exponent marker into one Python can parse."""
    return re.sub(r"([0-9])[dD]([+-]?[0-9])", r"\1e\2", line)


def write_edgrn_input(path, model, obs_depth=0.0, nr=1601, r_max=400e3,
                      nz=401, z_min=50.0, z_max=50050.0, srate=12.0,
                      directory="edgrnfcts/", names=("edgrn.ss", "edgrn.ds", "edgrn.cl")):
    """Write an `edgrn.inp` for `model` on the given `(r, z)` grid.

    The grid is the table's resolution, and the interpolation in
    `EdgrnTables.interpolate` is bilinear, so it wants to be fine enough
    that the response is nearly linear across a cell near the fault. `z_min`
    is not zero because a source exactly at the free surface radiates nothing.

    **The grid sets EDGRN's own accuracy, not just the interpolation's.**
    EDGRN resolves the field only down to its grid spacing, and its error follows
    the *coarser* of the radial and depth spacings. Against the exact point source
    on a homogeneous crust, worst over source types and components for receivers
    2-5 km out, at source depths 0.5 / 2 / 5 / 10 km (EDGRN's run time on one
    desktop CPU):

    | grid (`nr x nz`, spacing, depths)   | node error              | EDGRN |
    |-------------------------------------|-------------------------|-------|
    | 201 x 100, 2 km / 500 m, 0.5-50 km  | 61 / 78 / 44 / 16 %     | 4 s   |
    | 1601 x 401, 250 m / 100 m, to 40 km | 1.3 / 5.0 / 1.4 / 1.0 % | 163 s |
    | 1601 x 401, 250 m / 125 m, to 50 km | 1.8 / 4.9 / 1.4 / 1.0 % | 146 s |

    A grid as coarse as the first row can move an inversion's slip by more than
    10% of its peak. The default is the last row, kept to 50 km depth so that
    sources between 40 and 50 km stay inside the table. `EdgrnTables.interpolate`
    clamps at the table edge rather than failing, so make sure the table covers the
    mesh. `nz` cannot exceed 401 (EDGRN's compiled `nzsmax`), so a deeper table
    means coarser depth spacing.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"  {i + 1:3d}  {d:14.4e}  {vp:14.4e}  {vs:14.4e}  {rho:14.4e}"
        for i, (d, vp, vs, rho) in enumerate(
            zip(model.depth, model.vp, model.vs, model.rho))
    )
    path.write_text(
        "# EDGRN input written by slipsolve\n"
        f"  {obs_depth:.6e}\n"
        f" {int(nr)}  {0.0:.6e}  {r_max:.6e}\n"
        f" {int(nz)}  {z_min:.6e}  {z_max:.6e}\n"
        f" {srate:.4f}\n"
        f" '{directory}'  '{names[0]}'  '{names[1]}'  '{names[2]}'\n"
        f"  {len(model)}\n{rows}\n"
    )
    (path.parent / directory).mkdir(parents=True, exist_ok=True)
    return path


def run_edgrn(model, workdir, executable=None, **grid):
    """Generate tables by actually running EDGRN, and read them back.

    Optional and lazily resolved, in the same style as the package's other
    external tools. The reference implementation (SlipSolve-Curve) does not do this
    at all -- it only converts EDGRN's text output, on the reasoning that the user
    owns the Earth model -- so this is a convenience, not a port.

    Resolution order: an explicit `executable`, then `edgrn` on `PATH`, then
    the copy `pygrnwang` bundles and builds. Raises with all three named if
    none is available.
    """
    workdir = Path(workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    inp = write_edgrn_input(workdir / "edgrn.inp", model, **grid)

    binary = executable or shutil.which("edgrn") or _pygrnwang_edgrn()
    if binary is None:
        raise RuntimeError(
            "No EDGRN executable found. Pass executable=, put `edgrn` on PATH, or "
            "`pip install pygrnwang` (which bundles Rongjiang Wang's Fortran and "
            "builds it with gfortran). Alternatively supply tables directly with "
            "EdgrnTables.from_input_file, or use EdgrnTables.homogeneous for a "
            "uniform medium."
        )

    # EDGRN reads the input file's name on stdin and writes relative to its own
    # working directory, so it has to be run from there.
    subprocess.run([str(binary)], input=f"{inp.name}\n", text=True, check=True,
                   cwd=workdir, capture_output=True)
    return EdgrnTables.from_input_file(inp)


def _pygrnwang_edgrn():
    """The EDGRN binary `pygrnwang` ships, if it is installed and built."""
    try:
        import pygrnwang
    except ImportError:
        return None
    root = Path(pygrnwang.__file__).parent
    for candidate in sorted(root.rglob("edgrn*")):
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None
