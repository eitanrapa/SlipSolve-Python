"""Adaptive downsampling of a gridded LOS product to point observations.

An unwrapped interferogram has millions of pixels and a fault model has thousands
of parameters, so the inversion is fed a decimated set of samples: dense where
the displacement field varies, sparse where it is smooth. That is a quadtree: a
cell splits while the displacement inside it is rougher than a threshold, and
what survives is one representative sample per cell.

SlipSolve-Curve recurses over unstructured `(x, y, value)` lists. Here the
input is a regular raster, which allows something both faster and exactly
reproducible: cumulative sums of the valid mask and of the values let any
candidate cell's count, mean and variance be read in constant time, and the
recursion walks integer index rectangles instead of copying coordinate arrays.
Because a cell is an index rectangle, re-reading the look vectors over exactly
the same pixels is trivially consistent; in SlipSolve-Curve that consistency
has to be arranged by passing the cell index lists back in.

Samples too close to the fault trace are dropped rather than kept: a dislocation
solution is singular on the fault surface, so a Green's function evaluated on the
trace is not finite (see `slipsolve.greens._require_finite`), and
unwrapped InSAR is decorrelated there anyway.
"""

import numpy as np
import xarray as xr

from ._geo import open_stage


class Observations:
    """Downsampled line-of-sight samples with their look vectors and weights.

    Wraps an `xarray.Dataset` on a single `obs` dimension carrying
    `x`, `y` (local frame metres), `los` (metres, positive toward the
    sensor), `look_e`/`look_n`/`look_u` (target->sensor unit vector),
    `std` (the within-cell scatter), `weight` and `track`; and `region`
    when the track was sampled with `regions=` (see `from_los`).

    Attributes
    ----------
    ds : xarray.Dataset
        The samples, with the variables above.
    """

    STAGE = "slip_observations"

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_los(cls, los_stack, pair=0, name="track", frame=None, trace=None,
                 rms_min=0.005, width_min=1000.0, width_max=20000.0,
                 min_valid_fraction=0.5, exclude_within=0.0, stat="mean",
                 field=None, refine_within=0.0, regions=None, elevation=None, datum=0.0):
        """Quadtree-downsample one pair of a gridded LOS product.

        `rms_min` is the within-cell standard deviation (metres) below which a
        cell stops splitting; `width_min` and `width_max` bound the cell size
        in metres. `min_valid_fraction` drops cells that are mostly outside the
        swath. `exclude_within` drops samples closer than that many metres to
        `trace`; it is required, since the Green's functions are singular on the
        fault. `frame` defaults to one centred on the trace, or on the scene.

        The stack's `los` is taken as-is: it is already positive toward the
        sensor, because `slipsolve._geo.phase_to_los` applied `sign` when the
        stack was built. The attribute is validated and carried as provenance,
        not re-applied: applying it again would undo the correction `sign=-1`
        exists to make.

        **Model-based sampling.** `field` is an optional `(ny, nx)` array that
        decides *where* to split, in place of the observed displacement. Passing a
        model's predicted LOS (see `slipsolve.predict.predicted_los`)
        is the Wang & Fialko (2015) scheme: the observed field's within-cell
        scatter does not shrink as a cell shrinks, so once `rms_min` is below the
        noise the recursion cannot stop on information and simply runs down to
        `width_min`, sampling mostly noise. A predicted field is noise-free by
        construction, so cells split on signal. What is *reduced* is always the
        observed data: `field` chooses the cells, and the cells are then filled
        from `los`, exactly as the paper uses "the bounding coordinates of each
        resolution cell" to average the real interferogram.

        `refine_within` forces cells within that many metres of `trace` to keep
        splitting to `width_min` whatever `field` says there. It exists only for
        the model-based path and matters more than it looks: an initial model with
        little shallow slip predicts a smooth near-field, so the quadtree would
        coarsen precisely where shallow slip needs constraining, and the next
        iteration is then free to invent it. The paper keeps "a relatively dense
        sampling around the fault trace ... through all iterations" for this reason.

        `rms_min` means something different under `field`: a threshold on
        model curvature, not a noise level, so the recommendation of
        `slipsolve.diagnostics.scene_report` does not apply; use
        `slipsolve.predict.model_rms_min`.

        **Regions unwrapped apart.** Each connected region of an unwrapped
        interferogram carries its own arbitrary constant: an island, or a block
        cut off along the fault on purpose so the unwrapper cannot cross it.
        `regions` says where they are, and each region then gets its own offset
        column (`slipsolve.regularize.ramp_columns`), so no tie between
        regions is taken on trust. `"connected"` labels the 4-connected regions
        of the valid pixels, which is how SNAPHU's grid joins them; or pass an
        integer `(ny, nx)` label map (`<= 0` = unused). No cell mixes two
        regions: a mixed cell keeps splitting, and one that reaches `width_min`
        is reduced over its majority region's pixels only. Each sample records its
        label in `region`. `None` (the default) treats the scene as one region.

        **Topography.** `elevation` is an optional `(ny, nx)` DEM on the stack's grid, metres above sea level (or any
        datum, the same for every data set). It is averaged over each cell's valid pixels, as the look vectors are,
        ignoring pixels where it is not finite, and each sample records its height above `datum`, the elevation the
        model's free surface stands for, in `elevation`. The Green's functions then put each sample's free surface at
        that height (`slipsolve.inversion.assemble_design`).
        """
        if stat not in ("mean", "median"):
            raise ValueError("stat must be 'mean' or 'median'")
        if exclude_within > 0 and trace is None:
            raise ValueError("exclude_within needs a trace to measure distance from")
        if refine_within > 0 and trace is None:
            raise ValueError("refine_within needs a trace to measure distance from")

        ds = los_stack.ds
        epsg = stack_epsg(ds)
        if frame is None:
            frame = trace.local_frame() if trace is not None else _frame_for(ds)

        los = np.asarray(ds["los"].isel(pair=pair).values, dtype=float)
        # Validated and recorded below as provenance only. `phase_to_los` already
        # applied it, so multiplying again would square it back to +1 and return
        # the uncorrected field.
        sign = int(ds.attrs.get("sign", 1))
        if sign not in (1, -1):
            raise ValueError(f"Unexpected LOSStack sign attribute {sign!r}")

        look = np.stack([
            np.asarray(ds["los_east"].values, dtype=float),
            np.asarray(ds["los_north"].values, dtype=float),
            np.asarray(ds["los_up"].values, dtype=float),
        ])

        valid = np.isfinite(los) & np.all(np.isfinite(look), axis=0)
        region_map = _region_map(regions, valid)
        if region_map is not None:
            valid = valid & (region_map > 0)

        split_on = los
        if field is not None:
            split_on = np.asarray(
                field.values if hasattr(field, "values") else field, dtype=float
            )
            if split_on.shape != los.shape:
                raise ValueError(
                    f"field has shape {split_on.shape}, but the stack's grid is "
                    f"{los.shape}"
                )
            # A pixel with no prediction cannot be judged, so it cannot vote on
            # where to split -- and reducing it would mix modelled and observed
            # footprints inside one cell.
            valid = valid & np.isfinite(split_on)

        if not valid.any():
            raise ValueError("No pixel has both a finite displacement and a look vector")

        x_native = np.asarray(ds["x"].values, dtype=float)
        y_native = np.asarray(ds["y"].values, dtype=float)
        dx = abs(float(np.diff(x_native)[0]))
        dy = abs(float(np.diff(y_native)[0]))

        refine = None
        if refine_within > 0:
            gx, gy = np.meshgrid(x_native, y_native)
            fx_all, fy_all = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)
            refine = (trace.distance(fx_all, fy_all, frame)
                      .reshape(los.shape) <= refine_within)

        cells = _quadtree_cells(
            split_on, valid,
            rms_min=rms_min,
            min_rows=max(1, int(round(width_min / dy))),
            min_cols=max(1, int(round(width_min / dx))),
            max_rows=max(1, int(round(width_max / dy))),
            max_cols=max(1, int(round(width_max / dx))),
            min_valid_fraction=min_valid_fraction,
            refine=refine,
            regions=region_map,
        )
        if not cells:
            raise ValueError(
                "Quadtree produced no samples; loosen rms_min or min_valid_fraction"
            )

        dem = None
        if elevation is not None:
            dem = np.asarray(elevation.values if hasattr(elevation, "values") else elevation, dtype=float)
            if dem.shape != los.shape:
                raise ValueError(f"elevation has shape {dem.shape}, but the stack's grid is {los.shape}")
        rows = _reduce_cells(cells, los, look, valid, x_native, y_native, stat,
                             regions=region_map, min_valid_fraction=min_valid_fraction, elevation=dem)

        # Native grid metres -> the shared local frame. Averaging in the native
        # projection and transforming once is exact to well under a metre over a
        # cell, and avoids transforming every pixel of the raster.
        fx, fy = _to_frame(rows["x"], rows["y"], frame, epsg)

        keep = rows.pop("keep")
        if exclude_within > 0:
            keep = keep & (trace.distance(fx, fy, frame) > exclude_within)
            if not keep.any():
                raise ValueError(
                    f"exclude_within={exclude_within} m removed every sample"
                )

        out = xr.Dataset(
            {
                "x": ("obs", fx[keep]),
                "y": ("obs", fy[keep]),
                "los": ("obs", rows["los"][keep]),
                "look_e": ("obs", rows["look_e"][keep]),
                "look_n": ("obs", rows["look_n"][keep]),
                "look_u": ("obs", rows["look_u"][keep]),
                "std": ("obs", rows["std"][keep]),
                "cell_size": ("obs", rows["cell_size"][keep]),
                "count": ("obs", rows["count"][keep]),
                "weight": ("obs", np.ones(int(keep.sum()))),
                "track": ("obs", np.full(int(keep.sum()), name, dtype=object)),
            }
        )
        if region_map is not None:
            out["region"] = ("obs", rows["region"][keep].astype(np.int64))
        if dem is not None:
            heights = rows["elevation"][keep]
            if not np.all(np.isfinite(heights)):
                raise ValueError(f"{int((~np.isfinite(heights)).sum())} samples have no finite elevation in their cell")
            out["elevation"] = ("obs", heights - float(datum))
            out.attrs["datum"] = float(datum)
        quadtree = {
            "rms_min": float(rms_min),
            "width_min": float(width_min),
            "width_max": float(width_max),
            "min_valid_fraction": float(min_valid_fraction),
            "exclude_within": float(exclude_within),
            "stat": stat,
        }
        if field is not None:
            # Folded in only when used, so a data-driven run's params hash does
            # not depend on these options.
            quadtree["field"] = "model"
            quadtree["refine_within"] = float(refine_within)
        if region_map is not None:
            # Folded in only when used, like `field`.
            quadtree["regions"] = regions if isinstance(regions, str) else "labels"

        out.attrs.update(
            frame=frame.to_dict(),
            epsg=epsg,
            tracks=[name],
            source_epsg=epsg,
            pair=int(pair),
            sign=sign,
            direction=ds.attrs.get("direction"),
            wavelength=ds.attrs.get("wavelength"),
            quadtree=quadtree,
            n_raw_valid=int(valid.sum()),
        )
        return cls(out)

    @classmethod
    def from_points(cls, x, y, values, look, std, name="gnss", frame=None, epsg=4326, elevation=None, datum=0.0):
        """Observations that are already points: GNSS components, or any point measurement.

        `x`, `y` are station coordinates in `epsg` (lon/lat by default), or local
        metres with `epsg=None`; `frame` is required either way, since a handful of
        stations is no basis for choosing one: pass the frame the mesh was built in.
        `values` (metres) and `std` (metres, > 0) are one per row, and `look` is
        `(n, 3)` or one 3-vector for every row: a row measures `u . look` with `u`
        the (east, north, up) displacement. A three-component GNSS station is therefore
        three rows, with looks `(1, 0, 0)`, `(0, 1, 0)` and `(0, 0, 1)`. A look need
        not point upward (the upward check `from_los` inherits from radar
        geometry does not apply to a horizontal component), but it must be a unit vector.

        `weight` is `1 / std`: the diagonal noise weighting, which is what
        `concat` keeps with `normalize="none"`. Under `"sqrt_count"` the track's
        total influence is rescaled as for any other track and the relative station
        weights survive. `cell_size` is 0 and `count` 1, since nothing was averaged.

        `elevation` (m, one per row or one for all) is each row's height, recorded above `datum` as in `from_los`.
        """
        if frame is None:
            raise ValueError("from_points needs the frame the mesh was built in")
        values = np.asarray(values, dtype=float).ravel()
        n = values.size
        std = np.broadcast_to(np.asarray(std, dtype=float), (n,)).copy()
        look = np.asarray(look, dtype=float)
        look = np.broadcast_to(look, (n, 3)).copy() if look.ndim == 1 else look.reshape(n, 3)
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size != n or y.size != n:
            raise ValueError("x, y and values must have one entry per row")
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("from_points needs finite coordinates and values")
        if not np.all(std > 0):
            raise ValueError("every std must be positive: it sets weight = 1/std")
        norm = np.linalg.norm(look, axis=1)
        if not np.allclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"looks must be unit vectors (norms {norm.min():.4f}..{norm.max():.4f})")
        fx, fy = (x, y) if epsg is None else frame.from_epsg(x, y, int(epsg))
        out = xr.Dataset(
            {
                "x": ("obs", np.asarray(fx, dtype=float)),
                "y": ("obs", np.asarray(fy, dtype=float)),
                "los": ("obs", values),
                "look_e": ("obs", look[:, 0]),
                "look_n": ("obs", look[:, 1]),
                "look_u": ("obs", look[:, 2]),
                "std": ("obs", std),
                "cell_size": ("obs", np.zeros(n)),
                "count": ("obs", np.ones(n)),
                "weight": ("obs", 1.0 / std),
                "track": ("obs", np.full(n, name, dtype=object)),
            }
        )
        out.attrs.update(frame=frame.to_dict(), epsg=epsg, tracks=[name], source_epsg=epsg)
        if elevation is not None:
            heights = np.broadcast_to(np.asarray(elevation, dtype=float), (n,)).copy()
            if not np.all(np.isfinite(heights)):
                raise ValueError("every elevation must be finite")
            out["elevation"] = ("obs", heights - float(datum))
            out.attrs["datum"] = float(datum)
        return cls(out)

    @classmethod
    def concat(cls, items, normalize="sqrt_count", weights=None):
        """Combine several tracks into one observation set.

        `normalize` sets how a track's sample count feeds its influence:

        - `"sqrt_count"`: each row scaled by `1/sqrt(n)`, so a track's total
          contribution to the normal equations is independent of how densely it
          was sampled. The default, and almost always what is wanted.
        - `"count"`: each row scaled by `1/n`, which is what SlipSolve-Curve
          does, and which means a track with twice as many samples gets half
          the influence, since a row's weight enters the normal equations
          squared.
        - `"none"`: no count normalisation.

        `weights` optionally scales each named track on top of that.
        """
        if normalize not in ("sqrt_count", "count", "none"):
            raise ValueError("normalize must be 'sqrt_count', 'count' or 'none'")
        items = list(items)
        if not items:
            raise ValueError("Nothing to concatenate")

        frame = items[0].ds.attrs.get("frame")
        parts = []
        for item in items:
            if item.ds.attrs.get("frame") != frame:
                raise ValueError(
                    "All tracks must be sampled in the same LocalFrame; "
                    "pass frame= consistently to from_los."
                )
            ds = item.ds.copy(deep=True)
            n = ds.sizes["obs"]
            scale = {"sqrt_count": 1.0 / np.sqrt(n), "count": 1.0 / n, "none": 1.0}[normalize]

            # Build the scaled weights as a new array rather than editing in place,
            # so the caller's Observations are never modified and passing the same
            # object twice does not scale it twice.
            track = ds["track"].values
            factor = np.full(n, scale)
            if weights:
                for track_name in np.unique(track):
                    factor[track == track_name] *= float(weights.get(str(track_name), 1.0))
            ds["weight"] = ("obs", ds["weight"].values * factor)
            parts.append(ds)
        # Heights go in for every track or for none: a track without them would sit on the datum without saying so.
        with_heights = ["elevation" in p for p in parts]
        if any(with_heights) and not all(with_heights):
            missing = [str(item.ds.attrs.get("tracks")) for item, has in zip(items, with_heights) if not has]
            raise ValueError(f"some tracks carry elevations and some do not ({', '.join(missing)}): give every one, "
                             "or none")
        datums = {float(p.attrs.get("datum", 0.0)) for p in parts if "elevation" in p}
        if len(datums) > 1:
            raise ValueError(f"the tracks' elevations are above different datums: {sorted(datums)}")
        # A track sampled without `regions` is one region (label 0) beside those
        # that were; `xr.concat` would refuse the missing variable.
        if any("region" in p for p in parts):
            for p in parts:
                if "region" not in p:
                    p["region"] = ("obs", np.zeros(p.sizes["obs"], dtype=np.int64))

        out = xr.concat(parts, dim="obs", combine_attrs="drop_conflicts")
        out.attrs["frame"] = frame
        out.attrs["tracks"] = [t for item in items for t in item.ds.attrs.get("tracks", [])]
        out.attrs["normalize"] = normalize

        # `drop_conflicts` compares the per-track `quadtree` dicts whole, and the
        # per-scene thresholds differ between tracks, so the key would vanish and
        # take `field` (data- or model-driven, shared by every track in a round)
        # with it. Merge one level deeper and keep what the tracks agree on, or a
        # persisted model-driven set is indistinguishable from a data-driven one.
        quadtrees = [item.ds.attrs.get("quadtree") for item in items]
        if quadtrees and all(isinstance(q, dict) for q in quadtrees):
            shared = {k: v for k, v in quadtrees[0].items()
                      if all(k in q and q[k] == v for q in quadtrees[1:])}
            if shared:
                out.attrs["quadtree"] = shared
            # Keep the per-scene numbers too: they record how each track was sampled.
            out.attrs["quadtree_per_track"] = {
                str(name): dict(q)
                for item, q in zip(items, quadtrees)
                for name in item.ds.attrs.get("tracks", [])
            }
        if weights:
            out.attrs["track_weights"] = dict(weights)
        return cls(out)

    # -- accessors ---------------------------------------------------------
    @property
    def n(self):
        """Number of samples."""
        return int(self.ds.sizes["obs"])

    @property
    def frame(self):
        """The `LocalFrame` the samples are in."""
        from .frame import LocalFrame

        return LocalFrame.from_dict(self.ds.attrs["frame"])

    @property
    def tracks(self):
        """The track names, sorted."""
        return [str(t) for t in np.unique(self.ds["track"].values)]

    def track_mask(self, name):
        """Boolean mask of the samples from track `name`."""
        return self.ds["track"].values == name

    def frame_looks(self):
        """`(look_x, look_y, look_u)`: each row's look vector in the local frame's own axes."""
        ds = self.ds
        return frame_looks(self.frame, ds["x"].values, ds["y"].values,
                           ds["look_e"].values, ds["look_n"].values, ds["look_u"].values)

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write to a Zarr stage and return the reopened set.

        Worth caching: this is the one step that reads the full raster, and
        every later experiment with smoothing or bounds reuses its output
        unchanged.
        """
        name = name or self.STAGE
        full = {
            "stage": name,
            "epsg": self.ds.attrs.get("epsg"),
            "frame": self.ds.attrs["frame"],
            "tracks": self.ds.attrs.get("tracks"),
            "quadtree": self.ds.attrs.get("quadtree"),
            "normalize": self.ds.attrs.get("normalize"),
            **params,
        }
        ds = self.ds.copy()
        # Zarr has no object dtype; track names ride as fixed-width text.
        ds["track"] = ds["track"].astype(str)
        return Observations(workspace.store(name, ds, full, overwrite=overwrite))

    @classmethod
    def from_zarr(cls, path):
        """Reopen a set written by `persist`."""
        return cls(open_stage(path))

    def __repr__(self):
        return f"<Observations n={self.n} tracks={self.tracks}>"


def _frame_axes(frame, x, y):
    """True east and north at `(x, y)` as unit vectors along a `LocalFrame`'s axes: `(east, north)`, each `(2, ...)`.

    The frame's projection is conformal, so the two stay perpendicular; each is read off the projection by a central
    difference.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lon, lat = frame.to_lonlat(x, y)
    half = 5e-4                                     # degrees: about 50 m either side
    east = np.array(frame.to_local(lon + half, lat)) - np.array(frame.to_local(lon - half, lat))
    north = np.array(frame.to_local(lon, lat + half)) - np.array(frame.to_local(lon, lat - half))
    return east / np.hypot(*east), north / np.hypot(*north)


def frame_looks(frame, x, y, look_e, look_n, look_u):
    """Geographic east/north/up look vectors rotated into a `LocalFrame`'s axes.

    A look vector is measured against true east and north, but the Green's functions
    return displacement along the frame's x and y, and a transverse Mercator's grid
    north turns away from true north with distance from its central meridian (by about
    `d_lon * sin(lat)`): about 1 degree 150 km from the meridian at 37 N, 3 degrees at
    500 km from it at 36 N, which on 20 mm/yr of elastic signal is 1 mm/yr. The
    postseismic and interseismic modes rotate by default; `SlipInversion` does when
    asked (`frame_looks=True`). The vertical is untouched.
    """
    east, north = _frame_axes(frame, x, y)
    look_e, look_n = np.asarray(look_e, dtype=float), np.asarray(look_n, dtype=float)
    return (look_e * east[0] + look_n * north[0], look_e * east[1] + look_n * north[1],
            np.asarray(look_u, dtype=float))


def geographic_enu(frame, x, y, u_x, u_y, u_z):
    """Displacement along a `LocalFrame`'s axes turned to geographic east/north/up, `frame_looks`' inverse.

    The Green's functions give displacement along the frame's x and y; a model solved with
    rotated look vectors reports its east and north through this.
    """
    east, north = _frame_axes(frame, x, y)
    u_x, u_y = np.asarray(u_x, dtype=float), np.asarray(u_y, dtype=float)
    return u_x * east[0] + u_y * east[1], u_x * north[0] + u_y * north[1], np.asarray(u_z, dtype=float)


def stack_epsg(ds):
    """The EPSG code to transform a stack's grid from, or `None`.

    `None` means the stack was resampled onto a lattice in the local frame
    (`slipsolve.resample`) and its `x`/`y` are already local
    metres. Every helper that projects a stack's coordinates goes through this
    and `_to_frame`, so a frame-gridded stack works everywhere a UTM one
    does rather than raising `KeyError: 'epsg'` somewhere downstream.
    """
    if ds.attrs.get("frame") is not None:
        return None
    return int(ds.attrs["epsg"])


def _to_frame(x, y, frame, epsg):
    """Bring native grid coordinates into `frame`.

    `epsg is None` means the stack was already resampled onto a lattice in this
    frame (see `slipsolve.resample`), which grids in
    `slipsolve.frame.LocalFrame.local_crs`, so its `x`/`y` are local metres
    and there is nothing to do. That is also the only available answer: the
    frame's transverse Mercator has no EPSG code to transform from.
    """
    if epsg is None:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return frame.from_epsg(x, y, epsg)


def _frame_for(ds):
    """A local frame centred on a scene, when no trace was supplied."""
    from pyproj import Transformer

    from .frame import LocalFrame

    if ds.attrs.get("frame") is not None:
        return LocalFrame.from_dict(ds.attrs["frame"])

    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    t = Transformer.from_crs(f"EPSG:{int(ds.attrs['epsg'])}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x.mean(), y.mean())
    return LocalFrame(float(lon), float(lat))


def _region_map(regions, valid):
    """`regions` as an integer label map on the stack's grid, or `None`."""
    if regions is None:
        return None
    if isinstance(regions, str):
        if regions != "connected":
            raise ValueError(f"regions must be 'connected' or a label map, got {regions!r}")
        from scipy import ndimage

        # 4-connected (the default structure): SNAPHU's grid joins a pixel to its
        # four neighbours, so two areas touching only at a corner unwrap apart.
        return ndimage.label(valid)[0].astype(np.int64)
    labels = np.asarray(regions.values if hasattr(regions, "values") else regions)
    if labels.shape != valid.shape:
        raise ValueError(f"regions has shape {labels.shape}, but the stack's grid is "
                         f"{valid.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"regions must be an integer label map, got dtype {labels.dtype}")
    return labels.astype(np.int64)


def _quadtree_cells(values, valid, rms_min, min_rows, min_cols,
                    max_rows, max_cols, min_valid_fraction, refine=None, regions=None):
    """Index rectangles `(r0, r1, c0, c1)` surviving the variance split.

    Constant-time per candidate: the count, sum and sum of squares over any
    rectangle come from three cumulative-sum tables, so the recursion never
    touches the pixels themselves.

    The split statistic is the within-cell standard deviation. The published
    schemes use a curvature-based rule (Simons et al. 2002; Fialko 2004; Wang &
    Fialko 2015); on a smooth field the two behave alike, and the standard
    deviation is what the summed-area tables give in constant time.

    `refine` is an optional boolean mask of pixels that must be resolved to
    `min_rows`/`min_cols` regardless of how flat `values` is over them. It
    rides on a fourth table, so the test stays O(1) per candidate.

    `regions` (an integer label map) makes a cell holding valid pixels of two
    regions split as if it were rough. The test is exact: with `n`, `S1` and
    `S2` the count, sum and sum of squares of the labels in the cell, all equal
    iff `n S2 == S1**2`, read from integer tables and multiplied as Python
    integers, so no rounding can call a mixed cell pure.
    """
    v = np.where(valid, values, 0.0)
    m = valid.astype(np.float64)
    s0 = _sat(m)
    s1 = _sat(v)
    s2 = _sat(v * v)
    s3 = None if refine is None else _sat(np.asarray(refine, dtype=np.float64))
    if regions is not None:
        # Consecutive labels keep the integer sums small.
        dense = np.zeros(regions.shape, dtype=np.int64)
        used = valid & (regions > 0)
        dense[used] = np.unique(regions[used], return_inverse=True)[1] + 1
        t0 = _sat_int(used.astype(np.int64))
        t1 = _sat_int(dense)
        t2 = _sat_int(dense * dense)

    def block(sat, r0, r1, c0, c1):
        return sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]

    ny, nx = values.shape
    out = []
    stack = [(0, ny, 0, nx)]
    while stack:
        r0, r1, c0, c1 = stack.pop()
        rows, cols = r1 - r0, c1 - c0
        count = block(s0, r0, r1, c0, c1)

        rough = False
        if count > 1:
            mean = block(s1, r0, r1, c0, c1) / count
            var = block(s2, r0, r1, c0, c1) / count - mean * mean
            rough = np.sqrt(max(var, 0.0)) > rms_min
        if s3 is not None and block(s3, r0, r1, c0, c1) > 0:
            rough = True
        if regions is not None and not rough:
            n_lab = int(block(t0, r0, r1, c0, c1))
            if n_lab > 1:
                s_lab = int(block(t1, r0, r1, c0, c1))
                rough = n_lab * int(block(t2, r0, r1, c0, c1)) != s_lab * s_lab

        # Each axis is halved only if the halves stay above that axis's minimum
        # width -- decided per axis, not once for the cell. Deciding jointly lets
        # a cell that is splittable along strike also be halved across it, taking
        # the short side below width_min.
        oversize_rows = rows > max_rows
        oversize_cols = cols > max_cols
        want = rough or oversize_rows or oversize_cols
        split_rows = oversize_rows or (want and rows >= 2 * min_rows)
        split_cols = oversize_cols or (want and cols >= 2 * min_cols)

        if (split_rows and rows >= 2) or (split_cols and cols >= 2):
            rm = r0 + rows // 2
            cm = c0 + cols // 2
            row_spans = ((r0, rm), (rm, r1)) if split_rows and rows >= 2 else ((r0, r1),)
            col_spans = ((c0, cm), (cm, c1)) if split_cols and cols >= 2 else ((c0, c1),)
            for rr in row_spans:
                for cc in col_spans:
                    stack.append((rr[0], rr[1], cc[0], cc[1]))
            continue

        if count > 0 and count >= min_valid_fraction * rows * cols:
            out.append((r0, r1, c0, c1))
    return out


def _reduce_cells(cells, los, look, valid, x_native, y_native, stat, regions=None,
                  min_valid_fraction=0.0, elevation=None):
    """Collapse each surviving cell to one sample.

    Every quantity is averaged over the same valid pixels: the cell is an
    index rectangle, so the data, the look vector and the coordinates cannot
    disagree about which pixels they came from.

    With `regions`, a cell still holding two regions (one that reached the
    minimum width) is reduced over its majority region's pixels only, and dropped
    (`keep` False) if those fall under `min_valid_fraction` of the cell.
    """
    n = len(cells)
    out = {k: np.empty(n) for k in
           ("x", "y", "los", "look_e", "look_n", "look_u", "std", "cell_size",
            "count")}
    out["keep"] = np.ones(n, dtype=bool)
    if regions is not None:
        out["region"] = np.zeros(n, dtype=np.int64)
    if elevation is not None:
        out["elevation"] = np.full(n, np.nan)
    dx = abs(float(np.diff(x_native)[0]))
    dy = abs(float(np.diff(y_native)[0]))

    for i, (r0, r1, c0, c1) in enumerate(cells):
        m = valid[r0:r1, c0:c1]
        if regions is not None:
            lab = regions[r0:r1, c0:c1]
            values, counts = np.unique(lab[m], return_counts=True)
            major = values[np.argmax(counts)]
            if values.size > 1:
                m = m & (lab == major)
                out["keep"][i] = m.sum() >= min_valid_fraction * (r1 - r0) * (c1 - c0)
            out["region"][i] = major
        d = los[r0:r1, c0:c1][m]
        out["los"][i] = np.median(d) if stat == "median" else d.mean()
        out["std"][i] = d.std() if d.size > 1 else 0.0
        # How many pixels went into this average. Needed by
        # `slipsolve.covariance`, where the *uncorrelated* part of the
        # noise averages down as 1/n and the correlated part does not -- so a
        # cell's variance is not a function of its size alone.
        out["count"][i] = float(d.size)

        rr, cc = np.nonzero(m)
        out["x"][i] = x_native[c0:c1][cc].mean()
        out["y"][i] = y_native[r0:r1][rr].mean()
        for k, name in enumerate(("look_e", "look_n", "look_u")):
            out[name][i] = look[k, r0:r1, c0:c1][m].mean()
        if elevation is not None:
            heights = elevation[r0:r1, c0:c1][m]
            heights = heights[np.isfinite(heights)]
            if heights.size:
                out["elevation"][i] = heights.mean()
        out["cell_size"][i] = 0.5 * ((r1 - r0) * dy + (c1 - c0) * dx)

    # Averaging three components of a unit vector shortens it slightly; renormalise
    # so the projection stays a direction cosine and `los_up == cos(incidence)`
    # continues to hold.
    norm = np.sqrt(out["look_e"] ** 2 + out["look_n"] ** 2 + out["look_u"] ** 2)
    norm[norm == 0] = 1.0
    for name in ("look_e", "look_n", "look_u"):
        out[name] /= norm
    return out


def _sat(a):
    """Summed-area table with a zero first row and column."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(a, axis=0), axis=1, out=out[1:, 1:])
    return out


def _sat_int(a):
    """`_sat` in exact integers."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(a, axis=0, dtype=np.int64), axis=1, out=out[1:, 1:])
    return out
