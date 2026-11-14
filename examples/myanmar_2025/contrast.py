"""The rigidity contrast across the fault, solved exactly, as a correction to the Green's functions (stage X).

The west block (the Central Myanmar Basin) is taken 1.3 times softer than the 1-D crust above 10 km (Xu et al. 2025).
`slipsolve.bimaterial`'s exact mode solves it on the layered crust and returns uniform-slip element columns projected
onto the nodal tents, while the run's own Green's functions are exact tents. So the contrast enters as
dG = G_contrast - G_none, where G_none is the same element projection with no contrast (the exact mode at ratio 1).
The element discretisation cancels in the difference, and G + dG is the run's G with the contrast's effect added.
"""
import time

import numpy as np

from slipsolve import LayeredTDE, SlipInversion
from slipsolve.bimaterial import Bimaterial, RigidityContrast

EXTEND = {"cross": 9.0, "strike": 8.0}       # the interface's reach beyond the fault, in units of its depth extent


class ElementOnly:
    """The layered engine's uniform-slip element columns, which `SlipInversion` projects onto the tents."""

    supports_basis = False

    def __init__(self, engine):
        self.engine, self.name, self.nu = engine, "element-only " + engine.name, getattr(engine, "nu", 0.25)

    def los_matrix(self, mesh, *points, basis=None, **kw):
        return self.engine.los_matrix(mesh, *points, **kw)


def softer_side(trace, side="west", ratio=1.3, depth=10e3):
    """The block on `side` of `trace` `ratio` times softer above `depth` (m); below it and across, the 1-D crust."""
    return RigidityContrast(trace, side=side, ratio=[[ratio, 1.0]], depth_breaks=(depth,))


def correction(contrast, tables, log=print):
    """A `Production(correction=...)` function: dG on a mesh and its samples."""

    def dG(mesh, obs):
        t0 = time.time()
        g_none = SlipInversion(mesh, obs, engine=ElementOnly(LayeredTDE(tables)), basis="node", ramp="none").g
        log(f"  element columns without the contrast {g_none.shape} in {time.time() - t0:.0f} s")
        t0 = time.time()
        engine = Bimaterial(contrast, engine=LayeredTDE(tables), exact=True, extend=EXTEND)
        g_contrast = SlipInversion(mesh, obs, engine=engine, basis="node", ramp="none").g
        log(f"  element columns with the contrast, solved exactly, in {time.time() - t0:.0f} s; the interface "
            f"{engine.interface(mesh).n_elements} elements")
        return np.asarray(g_contrast, float) - np.asarray(g_none, float)

    return dG
