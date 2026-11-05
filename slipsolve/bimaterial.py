"""Faults between blocks of different rigidity: bimaterial faults.

A fault often separates two different rocks, and then the block on one side is
softer than the block on the other. A softer block moves more for the same slip,
so the contrast shows up in the data as an asymmetry across the trace -- the same
asymmetry a dip makes. On Venezuela a north block about twice as soft stood in
for the west's 80 degree north dip (2026-10-07).

**The model.** The material boundary is the fault surface itself, continued down
dip and past the trace's ends. The two blocks' moduli are one reference model --
the velocity model, or a homogeneous half-space -- scaled by a constant on each
side, so the layering and Poisson's ratio are the same on both sides and only the
level differs. :class:`RigidityContrast` says which block is softer, by how much,
and where along strike that changes.

**The exact solution.** Give each block ``a = mu_ref / mu_block``. The
displacement in a block is then exactly ``a F(D)``, where ``F(D)`` is the
reference half-space's field of a displacement discontinuity ``D`` on the
boundaries, and ``D`` solves::

    (a+ + a-)/2 D + (a+ - a-) Fbar(D) = s     on every boundary,

with ``s`` the slip on the fault (zero elsewhere) and ``Fbar`` the reference
field averaged across the boundary. Each scaled field is in equilibrium in its
block and traction-free at the surface. It is traction-continuous across the
boundary, because ``mu a`` is the same on both sides and a dislocation's traction
is continuous; the equation sets its jump. Nothing here assumes the boundary is
flat or vertical, or the reference homogeneous: with a layered reference the same
holds layer by layer, since both blocks scale it by a constant. :class:`Bimaterial`
with ``exact=True`` solves it with the half-space engine, or with
:class:`~slipsolve.layered_tde.LayeredTDE` for a layered crust. Both give the
field at depth with opening as well as slip; EDGRN's tables give neither.

**The scaled mode is its first term.** Dropping ``Fbar`` leaves::

    G = diag(a_obs) G_ref diag(1/m),        m = (a+ + a-)/2:

each sample's row times its own block's ``a``, and each slip column times
``1/m``, which is the harmonic mean of the two rigidities over the reference one.
With one ratio ``r = mu_other / mu_side`` that is ``2r/(1+r)`` on the named side
and ``2/(1+r)`` on the other, which is what the Venezuela scripts used.

``Fbar`` vanishes only for strike-slip on a vertical fault in 2-D. Elsewhere the
scaled mode is still a model of the exact form, but its slip is the effective
``D``, not the slip. Measured in 2-D (strike-slip, slip free, two-sided data):

- it recovered the true ratio (2.00-2.01 at dips 80-50 degrees);
- its potency was x0.95 at 80 degrees and x0.83 at 50 (ratio 2, hanging wall
  softer), and x1.06 / x1.23 with the footwall softer;
- correcting the slip afterwards recovered only about two thirds of that.

In 3-D, ``Fbar`` also opens the boundary near the rupture ends and slip
gradients, and an opening lifts the ground, which LOS sees. On the Panama
synthetic (Mw 7.7, two-sided, ratio 2) the scaled mode's error was 11% of the
contrast's effect on a vertical fault and 44% at 60 degrees. A scaled-mode sweep
fed exact-mode data put both dips 5 degrees off and the ratio at 2.24 (2026-10-12).
So the scaled mode is for exploring; take a dip, ratio or moment from ``exact=True``.

**Changing along strike.** ``breaks`` cut the fault into intervals, each with its
own ratio. The reference block stays uniform. The other block steps at each
break, across a vertical boundary normal to the trace, which the exact mode
meshes as part of the interface. ⚠️ The scaled mode cannot represent a step. On
the Panama synthetic, with the ratio stepping from 2 to 1 at mid-length, its error
was 112% of the contrast's effect, and a sweep read the two ratios as 1.48 and
1.29. ⚠️ Where only one block has data, the ratio there cannot be measured. It
rescales the slip by up to ``(1+r)/2`` and is an assumption, not a result.

**The moment** uses the harmonic-mean rigidity ``2 mu+ mu- / (mu+ + mu-)``. In
2-D the far field on either side is a homogeneous medium's at that rigidity.
:meth:`Bimaterial.rigidity_factor` gives it per element over the reference's.
:meth:`SlipModel.moment <slipsolve.inversion.SlipModel.moment>` applies it.

**Changing with depth.** ``depth_breaks`` step the contrast in depth
(:class:`RigidityContrast`); the exact mode meshes a horizontal boundary at each.
**Not built yet:** several blocks across strike. Stepped, it fits the exact form
too, since each region is still a scaled copy of the reference, and its boundaries
would join the interface as the breaks do. **Outside it:** different Poisson's
ratios on the two sides, whose moduli are not proportional, and a contrast
varying smoothly with depth except as steps.
"""

import copy
import warnings

import numpy as np
import scipy.linalg as sla

from ._tde import tde_disp_hs, tde_disp_hs_unit
from .greens import HalfSpaceTDE, _as_points
from .trace import FaultTrace, _arc_length

#: Compass words ``side=`` accepts, as unit (east, north) vectors.
COMPASS = {name: (np.sin(np.radians(az)), np.cos(np.radians(az))) for name, az in (
    ("north", 0.0), ("northeast", 45.0), ("east", 90.0), ("southeast", 135.0),
    ("south", 180.0), ("southwest", 225.0), ("west", 270.0), ("northwest", 315.0))}

#: A compass ``side=`` must make at most 60 degrees with the trace's mean normal
#: (one way or the other), or it does not say which block it means.
COMPASS_MIN_COSINE = 0.5

#: How far the exact mode's interface reaches beyond the fault, in units of the
#: fault's depth extent: below its base (``depth``), past each end of the trace
#: (``strike``), and out from the fault along an across-strike boundary
#: (``cross``). The discontinuity it carries decays only slowly with depth.
#: Measured on a 400 km fault dipping 50 degrees, at mid-length, against the 2-D
#: exact solution: the contrast's effect came out 91% of the 2-D one with the
#: interface cut 1 depth below the base, 94% at 3 and 95% at 6. Past the end of a
#: boundary nothing cancels the blocks' different scales, so samples there see the
#: contrast's whole jump. Keep ``strike`` and ``cross`` reaching past the data.
EXTEND = {"depth": 3.0, "strike": 4.0, "cross": 4.0}

#: Element growth from one row of the exact mode's extension to the next.
GROWTH = 1.35

#: The first row and column of the patches that meet the surface beside the
#: fault (past its ends, and at a break), as a fraction of the fault's element
#: size.
FIRST = 0.5

#: The two points an element's field is averaged between, either side of its
#: centroid, as a fraction of the element's size.
AVERAGE_OFFSET = 1e-6

#: Cross offsets smaller than this (metres) are taken as exactly zero. A
#: vertical fault rebuilt from its own nodes otherwise leans by rounding, and
#: an element edge a hair off vertical is inside the triangular dislocation's
#: near-vertical resonance (see `trace.VERTICAL_TOL_DEG`).
CROSS_SNAP = 1e-6


class RigidityContrast:
    """Which block is softer, by how much, and where along strike that changes.

    ``side`` names the block whose rigidity differs from the reference's:
    ``"left"`` or ``"right"`` of the trace as it runs from its first vertex to
    its last (the side :meth:`FaultTrace.side` calls +1 is the left), or a
    compass word (``"north"``, ``"southwest"``, ...) resolved against the
    trace's mean left-hand normal. A compass word more than 60 degrees from
    that normal is refused as ambiguous.

    ``ratio`` is ``mu_other / mu_side``: above 1 the named block is softer. One
    value holds along the whole fault. With ``breaks`` (arc lengths in metres
    along the trace, increasing) there is one value per interval, so
    ``ratio=(1.45, 1.0), breaks=(55e3,)`` puts a contrast west of 55 km and none
    east of it.

    ``reference`` names the block the rigidity model describes, the velocity
    model or the flat default: ``"other"`` (the default) or ``"side"``. That
    block is uniform along strike, and the other block's rigidity is the
    reference's scaled by each interval's ratio. So with one ratio it changes only
    the moment; with several it also decides which block steps at the breaks.

    Arc lengths are measured along ``trace``. A mesh built from it measures its
    own (``mesh.params[:, 0]``) on the trace resampled at its edge length, which
    differs from the trace's by metres. Keep breaks clear of that, or put a
    node column on each.

    ``depth_breaks`` (metres, increasing) cut the stepping block into bands from
    the surface down, and then ``ratio`` is ``(n_intervals, n_bands)``: a contrast
    that changes with depth, ``depth_breaks=(5e3,), ratio=[[2.0, 1.0]]`` putting it
    in the top 5 km only. Each band is still a scaled copy of the reference, so the
    exact form holds; the exact mode meshes a horizontal boundary at each break.
    ⚠️ A break exactly on a layer interface puts that boundary on a material
    interface, where the layered remainder is singular -- keep it off one.
    """

    def __init__(self, trace, side="left", ratio=1.0, breaks=(), reference="other",
                 frame=None, depth_breaks=()):
        self.trace = trace
        self.frame = frame or trace.local_frame()
        self.length = trace.length(self.frame)
        self.side = self._resolve_side(side)
        self.side_sign = 1 if self.side == "left" else -1
        self.requested_side = str(side)

        self.breaks = np.asarray(breaks, dtype=float).ravel()
        if np.any(np.diff(self.breaks) <= 0):
            raise ValueError(f"breaks must increase, got {self.breaks.tolist()}")
        if self.breaks.size and (self.breaks[0] <= 0.0 or self.breaks[-1] >= self.length):
            raise ValueError(f"breaks must lie inside the trace, between 0 and "
                             f"{self.length:.0f} m, got {self.breaks.tolist()}")
        self.depth_breaks = np.asarray(depth_breaks, dtype=float).ravel()
        if np.any(np.diff(self.depth_breaks) <= 0) or np.any(self.depth_breaks <= 0):
            raise ValueError(f"depth_breaks must be positive and increase, got "
                             f"{self.depth_breaks.tolist()}")
        n_int, n_band = self.breaks.size + 1, self.depth_breaks.size + 1
        ratio = np.asarray(ratio, dtype=float)
        if ratio.ndim == 0 or ratio.size == 1:
            ratios = np.full((n_int, n_band), float(ratio.ravel()[0]))
        elif ratio.ndim == 1 and n_band == 1 and ratio.size == n_int:
            ratios = ratio[:, None]
        elif ratio.ndim == 1 and n_int == 1 and ratio.size == n_band:
            ratios = ratio[None, :]
        elif ratio.shape == (n_int, n_band):
            ratios = ratio.copy()
        else:
            raise ValueError(f"{self.breaks.size} breaks and {self.depth_breaks.size} depth breaks "
                             f"make {n_int} intervals x {n_band} bands; got ratios of shape "
                             f"{ratio.shape}")
        if np.any(~np.isfinite(ratios) | (ratios <= 0)):
            raise ValueError(f"ratios must be positive, got {ratios.tolist()}")
        #: ``(n_intervals, n_bands)``, always.
        self.ratios = ratios
        #: One per interval without depth breaks (as before); the 2-D table with them.
        self.ratio = ratios[:, 0] if n_band == 1 else ratios
        if reference not in ("other", "side"):
            raise ValueError(f"reference must be 'other' or 'side', got {reference!r}")
        self.reference = reference

    def _resolve_side(self, side):
        side = str(side).lower()
        if side in ("left", "right"):
            return side
        if side not in COMPASS:
            raise ValueError(f"side must be 'left', 'right' or a compass word {tuple(COMPASS)}, "
                             f"got {side!r}")
        nx, ny = self.trace.normals(self.frame)
        mean = np.array([nx.mean(), ny.mean()])
        cosine = float(np.dot(mean / np.linalg.norm(mean), COMPASS[side]))
        if abs(cosine) < COMPASS_MIN_COSINE:
            raise ValueError(
                f"side={side!r} is {np.degrees(np.arccos(abs(cosine))):.0f} degrees from the "
                "trace's normal, so it does not name one block; say 'left' or 'right' (of the "
                "trace as it runs from its first vertex to its last)")
        return "left" if cosine > 0 else "right"

    # -- intervals and blocks -------------------------------------------------
    @property
    def n_intervals(self):
        return int(self.ratios.shape[0])

    @property
    def n_bands(self):
        return int(self.ratios.shape[1])

    def with_ratio(self, ratio):
        """The same contrast with other ratios (one, one per interval, or per interval and band)."""
        return RigidityContrast(self.trace, side=self.side, ratio=ratio, breaks=self.breaks,
                                reference=self.reference, frame=self.frame,
                                depth_breaks=self.depth_breaks)

    def interval(self, s):
        """The interval each arc length lies in, 0-based; beyond the ends, the end one."""
        return np.searchsorted(self.breaks, np.asarray(s, dtype=float), side="right")

    def band(self, depth):
        """The depth band each depth (metres, sign ignored) lies in, 0 at the surface."""
        return np.searchsorted(self.depth_breaks, np.abs(np.asarray(depth, dtype=float)),
                               side="right")

    def stepping_block(self):
        """``"side"`` or ``"other"``: the block whose rigidity changes at the breaks."""
        return "side" if self.reference == "other" else "other"

    def a(self, block, interval, band=0):
        """``mu_ref / mu`` of a block (``"side"`` or ``"other"``) in each interval and band."""
        r = self.ratios[np.asarray(interval), np.asarray(band)]
        if self.reference == "other":
            return r if block == "side" else np.ones_like(r)
        return np.ones_like(r) if block == "side" else 1.0 / r

    def _left_right(self, interval, band=0):
        """``a`` of the left and right blocks in each interval and band."""
        side, other = self.a("side", interval, band), self.a("other", interval, band)
        return (side, other) if self.side == "left" else (other, side)

    def observation_factor(self, x, y):
        """Each surface point's ``a``: its block's ``mu_ref / mu``, by the trace's side.

        A point is in the block on its side of the trace -- where the fault meets
        the surface -- and in the interval of the closest point on the trace.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        left = self.trace.side(x, y, self.frame) > 0
        s, _ = self.trace.to_curvilinear(x, y, self.frame)
        a_left, a_right = self._left_right(self.interval(s))
        return np.where(left, a_left, a_right)

    def element_factor(self, mesh):
        """``1/m`` per element: the two blocks' harmonic-mean rigidity over the reference's.

        An element takes the band of its centroid's depth.
        """
        a_left, a_right = self._left_right(self.interval(mesh.element_params[:, 0]),
                                           self.band(np.asarray(mesh.centroids)[:, 2]))
        return 2.0 / (a_left + a_right)

    def node_factor(self, mesh):
        """:meth:`element_factor` averaged over each node's tent, area-weighted."""
        from .basis import NodeBasis

        projection = NodeBasis(mesh).projection()
        areas = mesh.areas
        return (projection.T @ (areas * self.element_factor(mesh))) / (projection.T @ areas)

    def column_factor(self, mesh, basis=None):
        """One factor per Green's column: strike-slip parameters, then dip-slip."""
        nodal = basis is not None and getattr(basis, "name", "element") == "node"
        f = self.node_factor(mesh) if nodal else self.element_factor(mesh)
        return np.concatenate([f, f])

    # -- persistence ------------------------------------------------------------
    def to_dict(self):
        return {"lon": self.trace.lon.tolist(), "lat": self.trace.lat.tolist(),
                "trace_name": self.trace.name, "frame": self.frame.to_dict(),
                "side": self.side, "ratio": self.ratio.tolist(),
                "breaks": self.breaks.tolist(), "reference": self.reference,
                "depth_breaks": self.depth_breaks.tolist()}

    @classmethod
    def from_dict(cls, d):
        from .frame import LocalFrame

        trace = FaultTrace.from_lonlat(np.asarray(d["lon"]), np.asarray(d["lat"]),
                                       name=d.get("trace_name"))
        return cls(trace, side=d["side"], ratio=d["ratio"], breaks=d["breaks"],
                   reference=d["reference"], frame=LocalFrame.from_dict(d["frame"]),
                   depth_breaks=d.get("depth_breaks", ()))

    def __repr__(self):
        edges = np.concatenate([[0.0], self.breaks, [self.length]]) / 1e3
        depths = np.concatenate([[0.0], self.depth_breaks / 1e3, [np.inf]])
        def show(row):
            if self.n_bands == 1:
                return f"r {row[0]:g}"
            return " / ".join(f"r {r:g} to {b:g} km" if np.isfinite(b) else f"r {r:g} below"
                              for r, b in zip(row, depths[1:]))
        parts = ", ".join(f"{a:.0f}-{b:.0f} km {show(row)}"
                          for a, b, row in zip(edges[:-1], edges[1:], self.ratios))
        named = (self.side if self.requested_side in ("left", "right")
                 else f"{self.requested_side} = {self.side}")
        return (f"<RigidityContrast mu_other/mu_side on the {named} block: {parts}; "
                f"reference block {self.reference}>")


class Bimaterial:
    """An engine whose fault separates blocks of different rigidity.

    Wraps ``engine`` (the closed-form half-space by default) and answers to the
    same calls, so :class:`~slipsolve.inversion.SlipInversion`, forward fields,
    :class:`~slipsolve.inversion.SlipModel` and resolution maps take it as they
    take any engine. ``contrast`` is a :class:`RigidityContrast`.

    ``exact=False`` is the scaled mode, the first term of the exact solution
    (see the module docstring). It costs nothing beyond the wrapped engine. It is
    exact only for strike-slip on a vertical fault in 2-D. On a dipping fault its
    slip is an effective one, and near a change of ratio along strike it is a
    poor approximation.

    ``exact=True`` solves for the discontinuity on the whole interface. That is
    the fault plus its continuation (``extend``, in units of the fault's depth
    extent; see :data:`EXTEND`) and the across-strike boundaries at the breaks.
    It needs the field of each element at depth, with opening as well as slip:
    the half-space engine, or :class:`~slipsolve.layered_tde.LayeredTDE` for a
    layered crust. It returns element columns, which :class:`SlipInversion`
    projects onto a nodal basis as it does for the half-space engine. Cost: an
    assembly ``(3 N)^2`` on the interface's ``N`` elements and a dense solve. Then
    each Green's matrix costs one engine matrix over the interface. With
    :class:`LayeredTDE` both cost about five times the half-space's: elements that
    cross an interface are cut into pieces, and the remainder is integrated on
    top.
    """

    def __init__(self, contrast, engine=None, exact=False, extend=None, growth=GROWTH,
                 first=FIRST):
        self.contrast = contrast
        self.engine = engine if engine is not None else HalfSpaceTDE()
        self.exact = bool(exact)
        if self.exact and not (isinstance(self.engine, HalfSpaceTDE)
                               or hasattr(self.engine, "element_field")):
            raise NotImplementedError(
                "exact=True needs each element's field at depth, with opening as well as "
                "slip: the half-space engine, or LayeredTDE for a layered crust. "
                "EDGRN's tables (LayeredPointSource) have neither.")
        self.extend = {**EXTEND, **(extend or {})}
        self.growth = float(growth)
        self.first = float(first)
        self.nu = float(getattr(self.engine, "nu", 0.25))
        self.name = f"bimaterial:{self.engine.name}"
        # The scaled mode always takes the basis, so a nodal model's columns are
        # scaled per node, as DipSweep scales them, whatever the wrapped engine.
        self.supports_basis = not self.exact
        self._interfaces = {}

    def _nodal(self, basis):
        return basis is not None and getattr(basis, "name", "element") != "element"

    def _inner(self, method, mesh, *args, basis=None, **kwargs):
        """The wrapped engine's matrix in ``basis``, projecting element columns if it must."""
        nodal = self._nodal(basis)
        if nodal and getattr(self.engine, "supports_basis", False):
            return getattr(self.engine, method)(mesh, *args, basis=basis, **kwargs)
        g = getattr(self.engine, method)(mesh, *args, **kwargs)
        if not nodal:
            return g
        from .inversion import _blocked

        projection = _blocked(basis.projection())
        return (g.reshape(-1, g.shape[-1]) @ projection).reshape(g.shape[:-1] + (-1,))

    # -- the engine interface ---------------------------------------------------
    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, basis=None):
        """Line-of-sight design matrix, ``(npts, 2 * n_params)``."""
        rows = self.contrast.observation_factor(x, y)
        if self.exact:
            self._refuse_basis(basis)
            return rows[:, None] * self.interface(mesh).los(x, y, z, (look_e, look_n, look_u))
        g = self._inner("los_matrix", mesh, x, y, look_e, look_n, look_u, z=z, basis=basis)
        return rows[:, None] * g * self.contrast.column_factor(mesh, basis)[None, :]

    def displacement(self, mesh, x, y, z=None, basis=None):
        """ENU displacement per unit slip, ``(npts, 3, 2 * n_params)``."""
        rows = self.contrast.observation_factor(x, y)
        if self.exact:
            self._refuse_basis(basis)
            return rows[:, None, None] * self.interface(mesh).enu(x, y, z)
        g = self._inner("displacement", mesh, x, y, z=z, basis=basis)
        return rows[:, None, None] * g * self.contrast.column_factor(mesh, basis)[None, None, :]

    def forward(self, mesh, slip, x, y, look=None, z=None, basis=None):
        """Displacement from a slip model: scalar LOS with ``look``, else ``(npts, 3)`` ENU."""
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        rows = self.contrast.observation_factor(x, y)
        if self.exact:
            self._refuse_basis(basis)
            solve = self.interface(mesh)
            out = solve.field(solve.discontinuity(slip), x, y, z, look)
        else:
            scaled = slip * self.contrast.column_factor(mesh, basis)
            if self._nodal(basis) and not getattr(self.engine, "supports_basis", False):
                n = basis.n_basis
                element = np.column_stack([basis.element_values(scaled[:n]),
                                           basis.element_values(scaled[n:2 * n])])
                out = self.engine.forward(mesh, element, x, y, look=look, z=z)
            else:
                kwargs = {} if basis is None else {"basis": basis}
                out = self.engine.forward(mesh, scaled, x, y, look=look, z=z, **kwargs)
        return out * (rows if np.ndim(out) == 1 else rows[:, None])

    def rigidity_factor(self, mesh):
        """Per element, the harmonic-mean rigidity of the two blocks over the reference's."""
        return self.contrast.element_factor(mesh)

    def interface(self, mesh):
        """The exact mode's :class:`InterfaceSolve` for ``mesh``, built once and kept."""
        key = mesh.digest()
        if key not in self._interfaces:
            kernel = (self.engine if hasattr(self.engine, "element_field")
                      else _HalfSpaceKernel(self.nu))
            self._interfaces[key] = InterfaceSolve(mesh, self.contrast, nu=self.nu,
                                                   extend=self.extend, growth=self.growth,
                                                   first=self.first, kernel=kernel)
        return self._interfaces[key]

    def _refuse_basis(self, basis):
        if basis is not None and getattr(basis, "name", "element") != "element":
            raise ValueError("the exact mode returns element columns; project them onto the "
                             "basis afterwards, as SlipInversion does for the half-space engine")

    # -- provenance ---------------------------------------------------------------
    def describe(self):
        """A JSON-able record of the model, for a saved :class:`SlipModel`'s options."""
        return {"contrast": self.contrast.to_dict(), "exact": self.exact,
                "engine": self.engine.name, "nu": self.nu, "extend": dict(self.extend),
                "growth": self.growth, "first": self.first}

    @classmethod
    def from_description(cls, d, engine=None):
        return cls(RigidityContrast.from_dict(d["contrast"]),
                   engine=engine if engine is not None else HalfSpaceTDE(d.get("nu", 0.25)),
                   exact=d.get("exact", False), extend=d.get("extend"),
                   growth=d.get("growth", GROWTH), first=d.get("first", FIRST))

    def __repr__(self):
        mode = "exact" if self.exact else "scaled"
        return f"<Bimaterial {mode}, {self.engine.name}: {self.contrast!r}>"


class _Curvilinear:
    """The trace's ``(arc length, cross) -> (x, y)`` map, continued straight past its ends."""

    def __init__(self, trace, frame):
        self.trace, self.frame = trace, frame
        px, py = trace.to_local(frame)
        self.length = float(_arc_length(px, py)[-1])
        ends = np.array([0.0, self.length])
        x0, y0 = trace.from_curvilinear(ends, 0.0, frame)
        x1, y1 = trace.from_curvilinear(ends, 1.0, frame)
        self.normal_ends = np.column_stack([x1 - x0, y1 - y0])
        # The tangent is the normal turned clockwise: the normal is the left-hand one.
        self.tangent_ends = np.column_stack([self.normal_ends[:, 1], -self.normal_ends[:, 0]])

    def point(self, s, cross):
        s = np.asarray(s, dtype=float)
        cross = np.broadcast_to(np.asarray(cross, dtype=float), s.shape)
        x, y = self.trace.from_curvilinear(np.clip(s, 0.0, self.length).ravel(),
                                           cross.ravel(), self.frame)
        x, y = x.reshape(s.shape), y.reshape(s.shape)
        before, after = s < 0.0, s > self.length
        x = np.where(before, x + s * self.tangent_ends[0, 0], x)
        y = np.where(before, y + s * self.tangent_ends[0, 1], y)
        x = np.where(after, x + (s - self.length) * self.tangent_ends[1, 0], x)
        y = np.where(after, y + (s - self.length) * self.tangent_ends[1, 1], y)
        return x, y

    def normal(self, s):
        """The unit normal ``point`` offsets along at each ``s``."""
        x0, y0 = self.point(s, 0.0)
        x1, y1 = self.point(s, 1.0)
        return np.stack([x1 - x0, y1 - y0], axis=-1)


def _graded(start, extent, first, growth):
    """Stations from ``start`` outward by ``extent`` (either sign), steps growing by ``growth``."""
    out, step, total = [0.0], float(first), 0.0
    while total + 1e-9 * abs(extent) < abs(extent):
        total = min(total + step, abs(extent))
        out.append(total)
        step *= growth
    return start + np.sign(extent) * np.asarray(out)


def _depth_stations(bottom, total, first, cap, growth):
    """Depths from the surface down to ``total``: fine at the surface, at most ``cap`` to ``bottom``.

    Steps start at ``first`` and grow by ``growth``, held at ``cap`` while
    still above ``bottom`` (the fault's base), then grow freely. The surface is
    where the data are and where the field changes fastest.
    """
    out, depth, step = [0.0], 0.0, float(first)
    while depth < total * (1.0 - 1e-12):
        depth = min(depth + step, total)
        out.append(depth)
        step *= growth
        if depth < bottom:
            step = min(step, cap)
    return np.asarray(out)


def _insert(stations, levels):
    """Stations with every level inside their range added, unless already there."""
    out = np.sort(np.asarray(stations, dtype=float))
    inside = [v for v in np.asarray(levels, dtype=float)
              if out[0] < v < out[-1] and not np.any(np.isclose(out, v, rtol=0, atol=1e-6))]
    return np.sort(np.concatenate([out, inside])) if inside else out


def _align(stations, levels, snap=0.3):
    """Stations moved onto any level within ``snap`` of their gap from it.

    Only moved, never added: a level between two distant stations still cuts the
    elements there into pieces, which cost a dislocation each but no unknowns,
    while an added row of elements costs both. Measured on a 284-element interface
    with the Venezuela crust, adding rows took it to 450 elements and 20% longer.
    The first and last stations never move.
    """
    out = np.sort(np.asarray(stations, dtype=float))
    for level in np.asarray(levels, dtype=float):
        if not out[0] < level < out[-1] or np.any(out == level):
            continue
        k = int(np.searchsorted(out, level))
        below, above = out[k - 1], out[k]
        gap = above - below
        if level - below < snap * gap and k - 1 > 0:
            out[k - 1] = level
        elif above - level < snap * gap and k < out.size - 1:
            out[k] = level
    return out


def _grid(us, vs):
    """Every quad ``(u0, u1, v0, v1)`` of the tensor grid on stations ``us`` x ``vs``."""
    u0, v0 = np.meshgrid(us[:-1], vs[:-1], indexing="ij")
    u1, v1 = np.meshgrid(us[1:], vs[1:], indexing="ij")
    return np.column_stack([u0.ravel(), u1.ravel(), v0.ravel(), v1.ravel()])


def _strips(edges, lo, hi, first, aspect=1.0):
    """Quads in strips: between consecutive ``edges``, the span ``lo..hi`` cut evenly.

    Each strip is cut at about its own width (never under ``first``), so the
    elements stay roughly square while they grow away from the fault. Strips do
    not share their cross cuts; the quads still tile the patch without gaps.
    """
    quads = []
    for a, b in zip(edges[:-1], edges[1:]):
        step = max(float(first), aspect * abs(b - a))
        n = max(1, int(np.ceil((hi - lo) / step - 1e-9)))
        cuts = np.linspace(lo, hi, n + 1)
        quads += [(a, b, c, d) for c, d in zip(cuts[:-1], cuts[1:])]
    return np.asarray(quads, dtype=float).reshape(-1, 4)


class _HalfSpaceKernel:
    """One triangle's field in the closed-form half-space, in :class:`LayeredTDE`'s calls.

    The exact mode asks its kernel for :meth:`element_field` and :meth:`slip_field`;
    this answers them with :func:`~slipsolve._tde.tde_disp_hs` alone, and
    :class:`~slipsolve.layered_tde.LayeredTDE` with the layered crust.
    """

    def __init__(self, nu=0.25):
        self.nu = float(nu)
        self.name = HalfSpaceTDE.name

    def element_field(self, vertices, x, y, z=None, offset=None, components=(0, 1, 2)):
        x, y, z = _as_points(x, y, z)
        n = x.size
        if offset is not None:      # both sides in one call
            x, y, z = (np.concatenate([a + offset[:, i], a - offset[:, i]])
                       for i, a in enumerate((x, y, z)))
        u = np.moveaxis(tde_disp_hs_unit(x, y, z, *vertices, self.nu)[list(components)], -1, 0)
        return u if offset is None else 0.5 * (u[:n] + u[n:])

    def slip_field(self, vertices, slip, x, y, z=None):
        x, y, z = _as_points(x, y, z)
        return np.column_stack(tde_disp_hs(x, y, z, *vertices, *slip, self.nu))


def _fbar_columns(kernel, vertices, cx, cy, cz, offset, frames):
    """``fbar``'s columns for the interface elements ``vertices`` ``(k, 3, 3)``: ``(3 N, 3 k)``."""
    out = np.empty((3 * cx.size, 3 * len(vertices)))
    for i, v in enumerate(vertices):
        mean = kernel.element_field(v, cx, cy, cz, offset=offset)
        for comp in range(3):
            out[:, 3 * i + comp] = np.einsum("eki,ei->ek", frames, mean[:, comp]).ravel()
    return out


def _fbar_task(vertices, cx, cy, cz, offset, frames):
    from .layered_tde import worker_engine

    return _fbar_columns(worker_engine(), vertices, cx, cy, cz, offset, frames)


def _response_task(vertices, x, y, z, look):
    """A worker's block of :meth:`InterfaceSolve.response_matrix` for the elements ``vertices``."""
    from .layered_tde import worker_engine

    kernel = worker_engine()
    out = np.empty((x.size, 3, 3 * len(vertices)) if look is None else (x.size, 3 * len(vertices)))
    for k, v in enumerate(vertices):
        f = kernel.element_field(v, x, y, z)
        for comp in range(3):
            if look is None:
                out[:, :, 3 * k + comp] = f[:, comp]
            else:
                out[:, 3 * k + comp] = f[:, comp, 0] * look[0] + f[:, comp, 1] * look[1] + f[:, comp, 2] * look[2]
    return out


class InterfaceSolve:
    """The exact bimaterial field of one fault mesh: its interface, and the discontinuity on it.

    ``vertices`` holds the interface's triangles. First come the fault's own, in
    the mesh's order. Then the fault surface continued below its base, past its
    ends and up to the surface over a buried top, and the across-strike
    boundaries where the ratio changes. ``a_plus`` / ``a_minus`` are the ``a``
    on each element's two sides, its normal pointing to ``a_plus``.
    ``kind`` labels each element ``fault``, ``extension`` or ``break``.

    The unknown is element-constant: three components per element, in each
    triangle's own strike, dip and normal directions. The equation is collocated
    at the centroids, where the reference field is averaged between two points
    :data:`AVERAGE_OFFSET` of an element's size either side.

    ``kernel`` gives one triangle's reference field: the closed-form half-space
    with ``nu`` by default, or a :class:`~slipsolve.layered_tde.LayeredTDE` for a
    layered crust.
    """

    def __init__(self, mesh, contrast, nu=0.25, extend=None, growth=GROWTH, first=FIRST,
                 kernel=None, fbar=None, workers=None, factor=True):
        self.mesh, self.contrast, self.nu = mesh, contrast, float(nu)
        self.kernel = kernel if kernel is not None else _HalfSpaceKernel(nu)
        #: Processes for the assembly and :meth:`response_matrix` with a layered kernel
        #: (:func:`~slipsolve.layered_tde.engine_pool`); None or 1 runs here.
        self.workers = workers
        self.extend = {**EXTEND, **(extend or {})}
        self.growth = float(growth)
        self.first = float(first)
        self.n_fault = int(mesh.n_elements)
        self._build()
        if fbar is None:
            self._assemble(factor=factor)
        else:
            # A saved ``fbar`` from the same mesh, kernel and reach: the geometry is
            # rebuilt (cheap and deterministic) and only its size is checked.
            fbar = np.asarray(fbar, dtype=float)
            if fbar.shape != (3 * self.n_elements,) * 2:
                raise ValueError(f"fbar is {fbar.shape}; this interface needs {(3 * self.n_elements,) * 2}")
            self.fbar = fbar
            if factor:
                self._factor()
        self._x = None

    # -- geometry -------------------------------------------------------------------
    def _surface(self):
        """The fault's ``cross(s, depth)``, and its continuation outside the mesh."""
        from matplotlib.tri import LinearTriInterpolator, Triangulation

        mesh, contrast = self.mesh, self.contrast
        edge = mesh.attrs.get("edge_length")
        trace = (contrast.trace.resample(float(edge), contrast.frame) if edge
                 else contrast.trace)
        self.curv = _Curvilinear(trace, contrast.frame)
        nodes = np.asarray(mesh.nodes, dtype=float)
        s = np.asarray(mesh.params[:, 0], dtype=float)
        depth = -np.asarray(mesh.params[:, 1], dtype=float)
        x0, y0 = self.curv.point(s, 0.0)
        normal = self.curv.normal(s)
        cross = (nodes[:, 0] - x0) * normal[:, 0] + (nodes[:, 1] - y0) * normal[:, 1]
        cross[np.abs(cross) < CROSS_SNAP] = 0.0
        miss = np.hypot(nodes[:, 0] - (x0 + cross * normal[:, 0]),
                        nodes[:, 1] - (y0 + cross * normal[:, 1]))
        self.h0 = float(np.median(np.sqrt(2.0 * mesh.areas)))
        if miss.max() > 1e-3 * self.h0:
            warnings.warn(f"the mesh's nodes sit up to {miss.max():.3g} m off the trace's "
                          "normals; the interface continues a surface they do not quite lie on",
                          stacklevel=4)
        self.s_range = (float(s.min()), float(s.max()))
        self.d_range = (float(depth.min()), float(depth.max()))
        interp = LinearTriInterpolator(Triangulation(s, depth, mesh.triangles), cross)
        span_s = self.s_range[1] - self.s_range[0]
        span_d = self.d_range[1] - self.d_range[0]
        inset_s, inset_d = 1e-9 * span_s, 1e-9 * span_d

        def inside(s_, d_):
            s_, d_ = np.broadcast_arrays(np.asarray(s_, dtype=float), np.asarray(d_, dtype=float))
            s_ = np.clip(s_, self.s_range[0] + inset_s, self.s_range[1] - inset_s)
            d_ = np.clip(d_, self.d_range[0] + inset_d, self.d_range[1] - inset_d)
            value = np.ma.filled(interp(s_, d_), np.nan)
            if np.isnan(value).any():
                raise RuntimeError("could not read the fault's cross offset inside its own mesh")
            return value

        h = min(self.h0, 0.5 * span_d)
        top, bottom = self.d_range

        def cross_at(s_, d_):
            s_ = np.asarray(s_, dtype=float)
            d_ = np.asarray(d_, dtype=float)
            value = inside(s_, d_)
            below, above = d_ > bottom, d_ < top
            if below.any():
                slope = (inside(s_, bottom) - inside(s_, bottom - h)) / h
                value = np.where(below, inside(s_, bottom) + (d_ - bottom) * slope, value)
            if above.any():
                slope = (inside(s_, top + h) - inside(s_, top)) / h
                value = np.where(above, inside(s_, top) - (top - d_) * slope, value)
            value[np.abs(value) < CROSS_SNAP] = 0.0
            return value

        self.cross_at = cross_at

    def _place(self, s, cross, depth):
        x, y = self.curv.point(s, cross)
        return np.stack([x, y, -np.asarray(depth, dtype=float)], axis=-1)

    def _build(self):
        self._surface()
        contrast = self.contrast
        s_lo, s_hi = self.s_range
        top, bottom = self.d_range
        reach = bottom - top
        h0, growth = self.h0, self.growth
        e_depth = self.extend["depth"] * reach
        e_strike = self.extend["strike"] * reach
        e_cross = self.extend["cross"] * reach
        # A layered kernel cuts every element that crosses an interface into pieces,
        # each a triangular dislocation of its own, so the rows the solve lays out
        # itself stop at the interfaces instead of crossing them. A depth break must
        # be a row edge outright: the blocks' scale steps there.
        levels = getattr(self.kernel, "interfaces", ())
        cuts = contrast.depth_breaks

        def stations(values):
            return _insert(_align(values, levels), cuts)

        rows = stations(_graded(bottom, e_depth, h0, growth))      # depths below the base
        # Patches that reach the surface beside the fault (its ends, the breaks) are
        # tensor grids: fine at the surface and next to the fault, where the
        # discontinuity they carry changes fastest. Their columns start at half
        # the fault's element size.
        depths = stations(_depth_stations(bottom, bottom + e_depth, self.first * h0, h0, growth))

        patches = []                                           # (quads in (s, depth), kind)
        patches.append((_strips(rows, s_lo, s_hi, h0)[:, [2, 3, 0, 1]], "base"))
        if top > 0:
            up = stations(np.linspace(0.0, top, max(1, int(np.ceil(top / h0))) + 1))
            patches.append((_strips(up, s_lo, s_hi, h0)[:, [2, 3, 0, 1]], "top"))
        for sign, edge in ((-1.0, s_lo), (1.0, s_hi)):
            columns = np.sort(_graded(edge, sign * e_strike, self.first * h0, growth))
            patches.append((_grid(columns, depths), "end"))

        # Which two blocks each element separates, so a new ratio needs no new geometry:
        # an interval and a band for the fault and its continuation (left block on the
        # normal's side), plus a break index for an across-strike boundary, or a depth
        # break index for a horizontal one.
        vertices, kinds, intervals, breaks, bands, levels_ = [], [], [], [], [], []
        mesh_vertices = np.asarray(self.mesh.vertices, dtype=float)
        vertices.append(mesh_vertices)
        kinds += ["fault"] * self.n_fault
        intervals.append(contrast.interval(self.mesh.element_params[:, 0]))
        breaks.append(np.full(self.n_fault, -1))
        bands.append(contrast.band(mesh_vertices[:, :, 2].mean(axis=1)))
        levels_.append(np.full(self.n_fault, -1))
        if cuts.size:
            depth = -mesh_vertices[:, :, 2]
            straddle = (depth.min(axis=1)[:, None] < cuts - 1e-6) & (depth.max(axis=1)[:, None] > cuts + 1e-6)
            if straddle.any():
                warnings.warn(f"{int(straddle.any(axis=1).sum())} fault elements cross a depth break; "
                              "each takes its centroid's band. Lay the mesh's rows out on the breaks "
                              "to make that exact", stacklevel=4)

        for quads, kind in patches:
            s0, s1, d0, d1 = quads.T
            corners_s = np.stack([s0, s1, s1, s0], axis=1)
            corners_d = np.stack([d0, d0, d1, d1], axis=1)
            points = self._place(corners_s, self.cross_at(corners_s, corners_d), corners_d)
            tri = np.concatenate([points[:, [0, 1, 2]], points[:, [0, 2, 3]]])
            mid_s = np.concatenate([0.5 * (s0 + s1)] * 2)
            want = np.column_stack([self.curv.normal(mid_s), np.zeros(mid_s.size)])
            vertices.append(_orient(tri, want))
            kinds += ["extension"] * tri.shape[0]
            intervals.append(contrast.interval(mid_s))
            breaks.append(np.full(tri.shape[0], -1))
            bands.append(contrast.band(np.concatenate([0.5 * (d0 + d1)] * 2)))
            levels_.append(np.full(tri.shape[0], -1))

        stepping = contrast.stepping_block()
        outward = contrast.side_sign if stepping == "side" else -contrast.side_sign
        for b, s_b in enumerate(contrast.breaks):
            if np.array_equal(contrast.ratios[b], contrast.ratios[b + 1]):
                continue
            w0, w1, d0, d1 = _grid(_graded(0.0, e_cross, self.first * h0, growth), depths).T
            corners_w = np.stack([w0, w1, w1, w0], axis=1)
            corners_d = np.stack([d0, d0, d1, d1], axis=1)
            corners_s = np.full(corners_w.shape, s_b)
            cross = self.cross_at(corners_s, corners_d) + outward * corners_w
            points = self._place(corners_s, cross, corners_d)
            tri = np.concatenate([points[:, [0, 1, 2]], points[:, [0, 2, 3]]])
            normal = self.curv.normal(np.array([s_b]))[0]
            tangent = np.array([normal[1], -normal[0], 0.0])
            vertices.append(_orient(tri, np.broadcast_to(tangent, (tri.shape[0], 3))))
            kinds += ["break"] * tri.shape[0]
            intervals.append(np.full(tri.shape[0], -1))
            breaks.append(np.full(tri.shape[0], b))
            bands.append(contrast.band(np.concatenate([0.5 * (d0 + d1)] * 2)))
            levels_.append(np.full(tri.shape[0], -1))

        # A depth break: a horizontal boundary in the stepping block, from the fault
        # out to the cross reach and along the whole interface's length, its strips
        # widening away from the fault. The normal points up, to the band above.
        for k, d_k in enumerate(cuts):
            if np.array_equal(contrast.ratios[:, k], contrast.ratios[:, k + 1]):
                continue
            w_edges = _graded(0.0, e_cross, self.first * h0, growth)
            quads = _strips(w_edges, s_lo - e_strike, s_hi + e_strike, h0)  # (w0, w1, s0, s1)
            w0, w1, s0, s1 = quads.T
            corners_w = np.stack([w0, w0, w1, w1], axis=1)
            corners_s = np.stack([s0, s1, s1, s0], axis=1)
            corners_d = np.full(corners_w.shape, d_k)
            cross = self.cross_at(corners_s, corners_d) + outward * corners_w
            points = self._place(corners_s, cross, corners_d)
            tri = np.concatenate([points[:, [0, 1, 2]], points[:, [0, 2, 3]]])
            vertices.append(_orient(tri, np.broadcast_to([0.0, 0.0, 1.0], (tri.shape[0], 3))))
            kinds += ["depth break"] * tri.shape[0]
            intervals.append(contrast.interval(np.concatenate([0.5 * (s0 + s1)] * 2)))
            breaks.append(np.full(tri.shape[0], -1))
            bands.append(np.full(tri.shape[0], -1))
            levels_.append(np.full(tri.shape[0], k))

        self.vertices = np.concatenate(vertices)
        self.kind = np.asarray(kinds)
        self._interval = np.concatenate(intervals).astype(int)
        self._break = np.concatenate(breaks).astype(int)
        self._band = np.concatenate(bands).astype(int)
        self._level = np.concatenate(levels_).astype(int)
        self.a_plus, self.a_minus = self._sides(contrast)
        v = self.vertices
        normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        if np.any(area <= 0):
            raise RuntimeError("the interface has a degenerate triangle")
        self.normals = normal / (2.0 * area[:, None])
        self.areas = area
        self.centroids = v.mean(axis=1)
        strike = np.cross(np.array([0.0, 0.0, 1.0]), self.normals)
        length = np.linalg.norm(strike, axis=1)
        flat = length < 1e-9
        if flat.any():
            if not np.all(self.kind[flat] == "depth break"):
                raise RuntimeError("the interface has a horizontal triangle outside a depth break")
            # A horizontal element's frame, as `_tde._element_basis` takes it.
            strike[flat] = np.column_stack([np.zeros(flat.sum()), self.normals[flat, 2],
                                            np.zeros(flat.sum())])
            length[flat] = 1.0
        strike /= length[:, None]
        dip = np.cross(self.normals, strike)
        #: Each element's (strike, dip, normal) unit vectors as rows, as `_tde` defines them.
        self.frames = np.stack([strike, dip, self.normals], axis=1)

    @property
    def n_elements(self):
        return int(self.vertices.shape[0])

    def _sides(self, contrast):
        """``a`` on each element's two sides, the normal's side first, for ``contrast``."""
        a_plus = np.empty(self._interval.size)
        a_minus = np.empty(self._interval.size)
        joint = (self._break < 0) & (self._level < 0)
        a_plus[joint], a_minus[joint] = contrast._left_right(self._interval[joint],
                                                             self._band[joint])
        stepping = contrast.stepping_block()
        for b in np.unique(self._break[self._break >= 0]):
            at = self._break == b
            # The tangent points to larger arc length: the next interval.
            a_plus[at] = contrast.a(stepping, b + 1, self._band[at])
            a_minus[at] = contrast.a(stepping, b, self._band[at])
        for k in np.unique(self._level[self._level >= 0]):
            at = self._level == k
            # The normal points up: the band above is on its side.
            a_plus[at] = contrast.a(stepping, self._interval[at], k)
            a_minus[at] = contrast.a(stepping, self._interval[at], k + 1)
        return a_plus, a_minus

    def with_contrast(self, contrast):
        """This interface with other ratios: the same geometry and reference field, a new solve.

        ``contrast`` must name the same side and breaks; only the ratios may differ.
        The expensive part of the exact mode, ``Fbar``, does not depend on them, so
        a ratio costs one dense LU of the system.
        """
        if (contrast.side != self.contrast.side or contrast.reference != self.contrast.reference
                or not np.array_equal(contrast.breaks, self.contrast.breaks)
                or not np.array_equal(contrast.depth_breaks, self.contrast.depth_breaks)):
            raise ValueError("with_contrast keeps the geometry: the side, the reference, the "
                             "breaks and the depth breaks must be the same")
        ratios = contrast.ratios
        built = self._break[self._break >= 0]
        if any(not np.array_equal(ratios[b], ratios[b + 1]) and b not in built
               for b in range(ratios.shape[0] - 1)):
            raise ValueError("a break with no boundary meshed (equal ratios when built) cannot step now")
        built = self._level[self._level >= 0]
        if any(not np.array_equal(ratios[:, k], ratios[:, k + 1]) and k not in built
               for k in range(ratios.shape[1] - 1)):
            raise ValueError("a depth break with no boundary meshed (equal ratios when built) cannot "
                             "step now")
        new = copy.copy(self)
        new.contrast = contrast
        new.a_plus, new.a_minus = new._sides(contrast)
        new._factor()
        new._x = None
        return new

    # -- the system ---------------------------------------------------------------
    def _pool(self):
        """A worker pool for a layered kernel (``workers`` > 1), else None."""
        if not self.workers or int(self.workers) < 2 or not hasattr(self.kernel, "tables"):
            return None
        from .layered_tde import engine_pool

        return engine_pool(self.kernel, self.workers)

    def _assemble(self, factor=True):
        n = self.n_elements
        offset = AVERAGE_OFFSET * np.sqrt(self.areas)[:, None] * self.normals
        cx, cy, cz = self.centroids.T
        fbar = np.empty((3 * n, 3 * n))
        pool = self._pool()
        if pool is None:
            for j in range(n):
                fbar[:, 3 * j: 3 * j + 3] = _fbar_columns(self.kernel, self.vertices[j:j + 1], cx, cy, cz, offset,
                                                          self.frames)
        else:
            from .layered_tde import pool_map

            chunks = [c for c in np.array_split(np.arange(n), max(1, 16 * int(self.workers))) if c.size]
            with pool:
                jobs = ((k, (self.vertices[c], cx, cy, cz, offset, self.frames)) for k, c in enumerate(chunks))
                for k, cols in pool_map(pool, _fbar_task, jobs):
                    c = chunks[k]
                    fbar[:, 3 * c[0]: 3 * (c[-1] + 1)] = cols
        self.fbar = fbar
        if factor:                  # ``factor=False``: Fbar alone, to be saved and factored elsewhere
            self._factor()

    def _factor(self):
        n = self.n_elements
        m = 0.5 * (self.a_plus + self.a_minus)
        da = self.a_plus - self.a_minus
        system = np.repeat(da, 3)[:, None] * self.fbar
        system.flat[:: 3 * n + 1] += np.repeat(m, 3)
        self._lu = sla.lu_factor(system, overwrite_a=True)

    def _rhs(self, slip):
        """The right-hand side for element slip ``(2 n_fault,)`` or ``(2 n_fault, k)``."""
        slip = np.asarray(slip, dtype=float)
        nf = self.n_fault
        out = np.zeros((3 * self.n_elements,) + slip.shape[1:])
        out[0: 3 * nf: 3] = slip[:nf]
        out[1: 3 * nf: 3] = slip[nf: 2 * nf]
        return out

    def discontinuity(self, slip):
        """``D`` on every interface element, ``(N, 3)``, for element slip ``(2 n_fault,)``."""
        return sla.lu_solve(self._lu, self._rhs(slip)).reshape(self.n_elements, 3)

    @property
    def solution(self):
        """``D`` per unit slip on each fault element: ``(3 N, 2 n_fault)``."""
        if self._x is None:
            self._x = sla.lu_solve(self._lu, self._rhs(np.eye(2 * self.n_fault)))
        return self._x

    # -- fields -------------------------------------------------------------------
    def _responses(self, x, y, z, look, block):
        """Per interface element, the field of each unit component at the points."""
        x, y, z = _as_points(x, y, z)
        for start in range(0, self.n_elements, block):
            stop = min(self.n_elements, start + block)
            out = np.empty((x.size, 3, 3 * (stop - start)) if look is None
                           else (x.size, 3 * (stop - start)))
            for k, j in enumerate(range(start, stop)):
                f = self.kernel.element_field(self.vertices[j], x, y, z)
                for comp in range(3):
                    if look is None:
                        out[:, :, 3 * k + comp] = f[:, comp]
                    else:
                        out[:, 3 * k + comp] = (f[:, comp, 0] * look[0] + f[:, comp, 1] * look[1]
                                                + f[:, comp, 2] * look[2])
            yield start, stop, out

    def response_matrix(self, x, y, z=None, look=None, block=64, workers=None):
        """The reference field of unit ``D`` on every element, ``(npts, 3 N)`` with ``look``.

        Without ``look``, east/north/up: ``(npts, 3, 3 N)``. It does not depend on the
        ratios, so with :meth:`with_contrast` a Green's matrix for another ratio is
        ``rows * (response_matrix @ solution)``, the rows being
        :meth:`RigidityContrast.observation_factor`.
        """
        if look is not None:
            look = tuple(np.asarray(v, dtype=float).ravel() for v in look)
        x_ = np.asarray(x, dtype=float).ravel()
        shape = (x_.size, 3 * self.n_elements) if look is not None else (x_.size, 3, 3 * self.n_elements)
        out = np.empty(shape)
        if workers is None:
            workers = self.workers
        saved, self.workers = self.workers, workers
        pool = self._pool()
        self.workers = saved
        if pool is None:
            for start, stop, cols in self._responses(x, y, z, look, block):
                out[..., 3 * start: 3 * stop] = cols
            return out
        from .layered_tde import pool_map

        px, py, pz = _as_points(x, y, z)
        chunks = [c for c in np.array_split(np.arange(self.n_elements), max(1, 16 * int(workers))) if c.size]
        with pool:
            jobs = ((k, (self.vertices[c], px, py, pz, look)) for k, c in enumerate(chunks))
            for k, cols in pool_map(pool, _response_task, jobs):
                c = chunks[k]
                out[..., 3 * c[0]: 3 * (c[-1] + 1)] = cols
        return out

    def los(self, x, y, z=None, look=None, block=64):
        """The reference-medium LOS of the whole interface per unit fault slip, ``(npts, 2 n_fault)``."""
        look = tuple(np.asarray(v, dtype=float).ravel() for v in look)
        x_ = np.asarray(x, dtype=float).ravel()
        out = np.zeros((x_.size, 2 * self.n_fault))
        for start, stop, cols in self._responses(x, y, z, look, block):
            out += cols @ self.solution[3 * start: 3 * stop]
        return out

    def enu(self, x, y, z=None, block=32):
        """As :meth:`los`, in east, north and up: ``(npts, 3, 2 n_fault)``."""
        x_ = np.asarray(x, dtype=float).ravel()
        out = np.zeros((x_.size, 3, 2 * self.n_fault))
        for start, stop, cols in self._responses(x, y, z, None, block):
            out += np.einsum("pcj,jk->pck", cols, self.solution[3 * start: 3 * stop])
        return out

    def field(self, discontinuity, x, y, z=None, look=None):
        """The reference-medium field of ``D`` (``(N, 3)``): LOS with ``look``, else ENU."""
        x, y, z = _as_points(x, y, z)
        total = np.zeros((x.size, 3))
        d = np.asarray(discontinuity, dtype=float)
        scale = np.abs(d).max() if d.size else 0.0
        for j in np.flatnonzero(np.abs(d).max(axis=1) > 1e-14 * scale):
            total += self.kernel.slip_field(self.vertices[j], d[j], x, y, z)
        if look is None:
            return total
        look = [np.asarray(v, dtype=float).ravel() for v in look]
        return total[:, 0] * look[0] + total[:, 1] * look[1] + total[:, 2] * look[2]

    def __repr__(self):
        counts = {k: int((self.kind == k).sum())
                  for k in ("fault", "extension", "break", "depth break")}
        extra = f", {counts['depth break']} depth break" if counts["depth break"] else ""
        return (f"<InterfaceSolve {self.n_elements} elements ({counts['fault']} fault, "
                f"{counts['extension']} extension, {counts['break']} break{extra}), "
                f"{3 * self.n_elements} unknowns>")


def _orient(triangles, want):
    """Reverse each triangle whose normal points against ``want``."""
    triangles = np.array(triangles, dtype=float)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", normal, want) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return triangles
