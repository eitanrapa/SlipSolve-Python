"""Green's functions: the linear operator from slip on the fault to what we see.

Column `k` of the design matrix is the displacement every observation would
record for one metre of slip on element `k` and none anywhere else. Because
elastic displacement is linear in the Burgers vector, a slip model is just a
weighted sum of those columns -- which is the whole reason the inversion is a
least-squares problem and not an optimisation over forward models.

Parameters are ordered `[strike-slip on every element, dip-slip on every
element]`, so a design matrix has `2 * n_elements` columns; that block layout
is what the smoothing operator and the bound vectors in
`slipsolve.regularize` assume.

The engine is deliberately split from the projection. `displacement` returns
east/north/up, and only `project_los` knows about a satellite -- so the
same engine serves a layered-rigidity replacement, or GPS, without the inversion
learning anything new.

Assembly is serial on purpose. A thread pool over elements is slower at every
worker count: each element is about fifty numpy calls on arrays of tens of
kilobytes, too small for the GIL-free stretch to outweigh the dispatch overhead.
To make it faster, evaluate many elements per numpy call; do not add threads.
"""

import numpy as np

from ._tde import tde_disp_hs, tde_disp_hs_unit_many  # noqa: F401  (tde_disp_hs: kept for importers)


#: Bytes of triangle fields `HalfSpaceTDE` holds at once (it evaluates elements in chunks of this size).
CHUNK_BYTES = 200e6


def _element_fields(mesh, x, y, z, nu):
    """`(elements, unit)` chunks: each element's field of unit strike-, dip- and tensile slip at the points,
    `(chunk, 3, 3, n_points)`, all of a chunk's triangles in one vectorised pass."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    per = max(1, int(CHUNK_BYTES // (72.0 * max(x.size, 1))))
    for start in range(0, vertices.shape[0], per):
        chunk = np.arange(start, min(start + per, vertices.shape[0]))
        yield chunk, tde_disp_hs_unit_many(x, y, z, vertices[chunk], nu)


class HalfSpaceTDE:
    """Triangular dislocations in a homogeneous elastic half-space.

    `nu` is Poisson's ratio; 0.25 is the usual crustal value and the reference
    implementation's default. A homogeneous half-space has no rigidity structure,
    so `nu` is the only material parameter the displacements depend on -- the
    shear modulus cancels, and enters only when slip is converted to moment.
    """

    name = "half_space_tde"

    def __init__(self, nu=0.25):
        self.nu = float(nu)

    def displacement(self, mesh, x, y, z=None):
        """ENU displacement per unit slip, shape `(npts, 3, 2 * n_elements)`.

        Memory-hungry by construction -- three components times every parameter.
        Use `los_matrix` for an inversion, which never materialises the
        third axis.
        """
        x, y, z = _as_points(x, y, z)
        n = mesh.n_elements
        out = np.zeros((x.size, 3, 2 * n))
        for chunk, unit in _element_fields(mesh, x, y, z, self.nu):
            # unit: (elements, slip, component, points); columns are strike slip then dip slip.
            out[:, :, chunk] = np.transpose(unit[:, 0], (2, 1, 0))
            out[:, :, n + chunk] = np.transpose(unit[:, 1], (2, 1, 0))
        _require_finite(out, mesh, x, y, z)
        return out

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None):
        """Line-of-sight design matrix, shape `(npts, 2 * n_elements)`.

        Each element's ENU response is projected onto that observation's own look
        vector as it is computed, so the `(npts, 3)` intermediate is transient
        and peak memory is the matrix itself.
        """
        x, y, z = _as_points(x, y, z)
        look_e = np.asarray(look_e, dtype=float).ravel()
        look_n = np.asarray(look_n, dtype=float).ravel()
        look_u = np.asarray(look_u, dtype=float).ravel()
        if not (look_e.size == look_n.size == look_u.size == x.size):
            raise ValueError("Look-vector components must match the observation count")

        n = mesh.n_elements
        g = np.zeros((x.size, 2 * n))
        look = np.stack([look_e, look_n, look_u])
        for chunk, unit in _element_fields(mesh, x, y, z, self.nu):
            projected = np.einsum("esct,ct->est", unit[:, :2], look)        # (elements, slip, points)
            g[:, chunk] = projected[:, 0].T
            g[:, n + chunk] = projected[:, 1].T
        _require_finite(g, mesh, x, y, z)
        return g

    def forward(self, mesh, slip, x, y, look=None, z=None):
        """Displacement produced by a given slip model.

        `slip` is either the `2 * n_elements` parameter vector or an
        `(n_elements, 2)` array of `(strike_slip, dip_slip)`. With `look`
        as a 3-tuple of arrays the result is scalar line-of-sight; without it,
        ENU of shape `(npts, 3)`.
        """
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        if slip.size != 2 * mesh.n_elements:
            raise ValueError(
                f"slip has {slip.size} values; expected {2 * mesh.n_elements}"
            )
        if look is None:
            return self.displacement(mesh, x, y, z) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z) @ slip


def project_los(disp_enu, look_e, look_n, look_u):
    """Project ENU displacement onto the line of sight.

    The look vector points **from the target to the sensor** and the result is
    positive **toward the sensor**, so this is a plain dot product with no sign
    to remember. `disp_enu` is `(npts, 3)` or `(npts, 3, nparam)`.
    """
    look = np.stack([np.asarray(look_e, dtype=float).ravel(),
                     np.asarray(look_n, dtype=float).ravel(),
                     np.asarray(look_u, dtype=float).ravel()], axis=1)
    disp_enu = np.asarray(disp_enu, dtype=float)
    if disp_enu.ndim == 2:
        return (disp_enu * look).sum(axis=1)
    return np.einsum("pcn,pc->pn", disp_enu, look)


def tde_greens(mesh, obs, nu=0.25):
    """The line-of-sight design matrix for a set of `Observations`.

    A thin convenience over `HalfSpaceTDE.los_matrix` that also checks the
    observations were built in the mesh's own frame -- mixing frames is a
    kilometre-scale error that produces a perfectly smooth, entirely wrong matrix.
    """
    if mesh.frame is not None and obs.ds.attrs.get("frame") is not None:
        mesh.frame.require_match(obs.ds.attrs["frame"], "Observations")
    ds = obs.ds
    return HalfSpaceTDE(nu).los_matrix(
        mesh, ds["x"].values, ds["y"].values,
        ds["look_e"].values, ds["look_n"].values, ds["look_u"].values,
    )


def _as_points(x, y, z):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.zeros(x.size) if z is None else np.asarray(z, dtype=float).ravel()
    if y.size != x.size or z.size != x.size:
        raise ValueError("x, y and z must have the same length")
    return x, y, z


def _require_finite(g, mesh, x, y, z=None):
    """Refuse a design matrix with non-finite entries, and say which observation.

    A dislocation solution is singular on the fault surface itself, so an
    observation sitting on the surface trace produces NaN. Silently zeroing it
    would leave a row of the inversion quietly meaningless; the caller wants to
    know to widen the exclusion buffer around the trace.

    The distance quoted is three-dimensional: in map view the deep nodes of a
    dipping fault sit kilometres to one side, so the nearest node in plan view
    can be one the observation is nowhere near.
    """
    bad = ~np.isfinite(g)
    if not bad.any():
        return
    rows = np.unique(np.nonzero(bad.any(axis=tuple(range(1, g.ndim))))[0])
    z = np.zeros(x.size) if z is None else np.asarray(z, dtype=float).ravel()
    nearest = np.sqrt(np.min(
        (x[rows, None] - mesh.nodes[None, :, 0]) ** 2
        + (y[rows, None] - mesh.nodes[None, :, 1]) ** 2
        + (z[rows, None] - mesh.nodes[None, :, 2]) ** 2,
        axis=1,
    ))
    raise ValueError(
        f"{rows.size} observation(s) gave a non-finite Green's function; the "
        f"closest sits {nearest.min():.0f} m from the fault. Dislocation "
        "solutions are singular on the fault surface -- exclude observations "
        "near the trace (Observations.from_los(exclude_within=...)). "
        f"Offending observation indices: {rows[:10].tolist()}"
    )
