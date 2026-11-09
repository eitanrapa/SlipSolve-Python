"""A layered elastic crust: depth, P and S velocity, density.

`VelocityModel` is what a layered engine (`slipsolve.wavenumber.RemainderTables`,
`slipsolve.layered_tde.LayeredTDE`) is built from, and the source of the
depth-dependent rigidity a moment calculation wants.
"""

import re
from pathlib import Path

import numpy as np

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

    The source of the depth-dependent rigidity `mu = rho * vs**2` that a moment
    calculation wants, and of a layered engine's layers. A discontinuity is written
    as two rows at the same depth; a gradient as two rows at different depths with
    different values.
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
        """Read `depth vp vs rho` rows (metres, m/s, m/s, kg/m^3).

        Rows of five columns, whose first is an index (the layout of an EDGRN or
        PSGRN input file's layer block), are read as their last four.
        """
        path = Path(path).expanduser()
        rows = []
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = [float(v) for v in re.split(r"[\s,]+", line)]
            rows.append(fields[-4:] if len(fields) >= 5 else fields[:4])
        table = np.asarray(rows, dtype=float)
        if table.ndim != 2 or table.shape[1] != 4:
            raise ValueError(f"{path} must have four columns: depth vp vs rho")
        return cls(*table.T, name=path.stem)

    @property
    def interfaces(self):
        """Depths where a property steps, in metres: two rows at one depth that differ.

        A gradient -- two rows at different depths with different values -- is
        continuous and is not an interface. These are the depths across which a
        table of point sources must not be interpolated.
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


def _step_depths(depth, vp, vs, rho):
    """Depths of the steps in a layer block: consecutive rows at one depth that differ."""
    depth = np.asarray(depth, dtype=float).ravel()
    if depth.size < 2:
        return []
    props = np.column_stack([np.ravel(vp), np.ravel(vs), np.ravel(rho)]).astype(float)
    same_depth = np.isclose(depth[1:], depth[:-1], rtol=0.0, atol=1e-6)
    differ = ~np.all(np.isclose(props[1:], props[:-1], rtol=1e-9, atol=0.0), axis=1)
    return [float(d) for d in depth[1:][same_depth & differ & (depth[1:] > 0)]]
