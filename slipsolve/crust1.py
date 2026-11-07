"""CRUST1.0 (Laske, Masters, Ma & Pasyanos 2013) as moduli for `slipsolve.heterogeneity.VoxelModel`.

CRUST1.0 gives, on 1-degree cells, the tops of nine layers (water, ice, three sediments, three crystalline crust, and
the mantle below the Moho) with Vp, Vs and density in each: four files of 64,800 rows (`crust1.bnds`, `.vp`,
`.vs`, `.rho`; km below sea level as negative elevations, km/s, g/cm^3), cells from 89.5 N 179.5 W eastward, then
southward. A cell is an average over its square, so a point takes the cell holding it.

Depths here are below the rock surface: water and ice are dropped and each column is measured from its own top, as
the layered reference (`slipsolve.wavenumber.RemainderTables`) flattens topography and bathymetry. The static
load of a water column is not modelled; its rock is. Within a column the moduli are piecewise constant in depth, so
`Crust1.average` integrates them exactly over any depth range (the arithmetic, Voigt, mean of `lambda` and `mu`).
"""
from pathlib import Path

import numpy as np

#: Layer indices of a CRUST1.0 column: 0 water, 1 ice, 2-4 sediments, 5-7 crystalline crust, 8 mantle.
WATER, ICE, MANTLE = 0, 1, 8


class Crust1:
    """CRUST1.0's cells: `bounds` (layer tops, km, elevation), `vp`, `vs` (km/s), `rho` (g/cm^3), each `(64800, 9)`."""

    def __init__(self, bounds, vp, vs, rho):
        arrays = [np.asarray(a, dtype=float) for a in (bounds, vp, vs, rho)]
        for a in arrays:
            if a.shape != (64800, 9):
                raise ValueError(f"CRUST1.0 arrays are (64800, 9), got {a.shape}")
        self.bounds, self.vp, self.vs, self.rho = arrays
        self._columns = {}

    @classmethod
    def load(cls, directory):
        """The four files from the CRUST1.0 distribution's directory."""
        d = Path(directory).expanduser()
        return cls(*(np.loadtxt(d / f"crust1.{k}") for k in ("bnds", "vp", "vs", "rho")))

    @staticmethod
    def cell(lon, lat):
        """The index of the cell holding each point (row from 90 N, column from 180 W)."""
        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        row = np.clip(np.floor(90.0 - lat), 0, 179).astype(int)
        col = np.mod(np.floor(lon + 180.0), 360).astype(int)
        return row * 360 + col

    def column(self, k):
        """Cell `k`'s solid layers below its rock surface: `(top, bottom, lam, mu)`, metres and Pa.

        Water and ice are dropped, and layers of zero thickness; the mantle runs to infinity.
        """
        k = int(k)
        if k not in self._columns:
            tops = self.bounds[k]
            thick = tops[:-1] - tops[1:]
            solid = [i for i in range(ICE + 1, MANTLE) if thick[i] > 0.0] + [MANTLE]
            surface = tops[solid[0]]
            top = np.array([(surface - tops[i]) * 1e3 for i in solid])
            bottom = np.append(top[1:], np.inf)
            rho = self.rho[k, solid] * 1e3
            vs, vp = self.vs[k, solid] * 1e3, self.vp[k, solid] * 1e3
            mu = rho * vs ** 2
            lam = rho * vp ** 2 - 2.0 * mu
            if np.any(mu <= 0.0) or np.any(lam + 2.0 * mu / 3.0 <= 0.0):
                raise ValueError(f"CRUST1.0 cell {k} has a solid layer without positive moduli")
            self._columns[k] = (top, bottom, lam, mu)
        return self._columns[k]

    def average(self, lon, lat, top, bottom):
        """`(lam, mu)`: thickness-weighted means over `top`-`bottom` (m below the rock surface) at each point.

        All arguments broadcast together; the cell is the one holding `(lon, lat)`.
        """
        lon, lat, top, bottom = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (lon, lat, top, bottom)))
        if np.any(bottom <= top) or np.any(top < 0.0):
            raise ValueError("each depth range needs 0 <= top < bottom")
        cells = self.cell(lon, lat)
        lam = np.empty(cells.shape)
        mu = np.empty(cells.shape)
        for k in np.unique(cells):
            at = cells == k
            t, b, layer_lam, layer_mu = self.column(k)
            overlap = np.clip(np.minimum(bottom[at][:, None], b[None]) - np.maximum(top[at][:, None], t[None]), 0.0, None)
            span = (bottom[at] - top[at])[:, None]
            lam[at] = (overlap * layer_lam[None]).sum(axis=1) / span[:, 0]
            mu[at] = (overlap * layer_mu[None]).sum(axis=1) / span[:, 0]
        return lam, mu


__all__ = ["Crust1"]
