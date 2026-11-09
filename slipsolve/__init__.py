"""Geodetic coseismic slip inversion from InSAR, pixel offsets and GNSS.

`slipsolve` turns surface displacement into slip on a fault. Gridded
line-of-sight observations are adaptively downsampled, a triangulated fault
surface is built from the fault's surface trace, elastic Green's functions relate
unit slip on the fault to what each observation would see, and a regularized,
bounded least-squares solve inverts for the slip distribution. The design is a
Python port and extension of
[SlipSolve-Curve](https://github.com/x3zou/SlipSolve-Curve) (MATLAB).

A typical run:

```python
from slipsolve import FaultMesh, FaultTrace, LosGrid, Observations, SlipInversion

trace = FaultTrace.from_file("fault.kml")
frame = trace.local_frame()
mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

asc = LosGrid.from_grd("asc_los.grd", "asc_e.grd", "asc_n.grd", "asc_u.grd")
desc = LosGrid.from_grd("desc_los.grd", "desc_e.grd", "desc_n.grd", "desc_u.grd")
obs = Observations.concat([
    Observations.from_los(asc, name="asc", frame=frame, trace=trace, exclude_within=1e3),
    Observations.from_los(desc, name="desc", frame=frame, trace=trace, exclude_within=1e3),
])

model = SlipInversion(mesh, obs, ramp="linear").solve(smoothing=0.3)
print(model.moment_magnitude)
model.save("model.slip.zip")
```

Two more modes rest on the same machinery: slip that changes with time from dated series
(`slipsolve.temporal.TimeDependentInversion`, postseismic), and rigid blocks plus
back-slip from velocities (`slipsolve.interseismic.InterseismicInversion`).

Main entry points:

- Geometry: `FaultTrace`, `LocalFrame` and `FaultMesh` (`vertical`, `curved`
  for a dipping or curved fault, `join` for several faults).
- Data: `LosGrid` (a gridded LOS product with its look vectors), `resample_all`
  and `Observations` (`from_los` quadtree-downsamples a scene, `from_points`
  takes GNSS, `concat` joins tracks).
- Green's functions: `HalfSpaceTDE` (the default), `LayeredTDE` (a layered
  crust, from a `VelocityModel`), `HeterogeneousEngine` (3-D
  heterogeneity on a `VoxelModel`) and `Bimaterial` (blocks of different
  rigidity either side of the fault).
- The solve: `SlipInversion` returns a `SlipModel`, which reports the fit and
  moment, forward-models, exports and saves.
- Choosing and judging a model: `blocked_cv_curve` and `gcv_curve` (the
  smoothing weight), `resolution_report` and `posterior_covariance`,
  `iterate_sampling` (model-driven resampling and remeshing), `DipSweep` (the
  dip of each segment) and `calibrate` (synthetic recovery tests).

Conventions:

- Displacement is **positive toward the sensor**, and the ENU look vector points
  from the target *to* the sensor, so the forward model is a plain dot product.
- Coordinates are metres in one `LocalFrame`; depth is negative, with the free
  surface at `z = 0`.
- Positive strike-slip is **left-lateral**: `FaultMesh` winds every element so
  that its normal is the trace's left-hand normal, which makes its strike
  direction, `cross(eZ, normal)`, minus the trace's tangent. A right-lateral
  fault wants `polarity=(-1, 0, 0)` in `SlipInversion.solve`.
- Positive dip-slip moves the block on the trace's left up the dip. That is
  reverse motion on a fault dipping to the left of the trace (`uniform_dip`
  below 90) and normal motion on one dipping to the right. Its sign flips with
  an element's winding, which is why the mesh pins the winding.
"""

from . import calibrate
from ._geo import LosGrid, read_grd
from .basis import ElementBasis, NodeBasis
from .bimaterial import Bimaterial, RigidityContrast
from .covariance import (
    correlated_field,
    CovarianceModel,
    empirical_variogram,
    fit_covariance,
    fit_reml,
    observation_covariance,
    scene_covariance,
    whitening_factors,
)
from .crust import VelocityModel
from .diagnostics import (
    cell_size_ladder,
    noise_floor,
    ramp_content,
    ramp_leakage,
    ramp_sweep,
    scene_report,
)
from .evidence import GeometryEvidence, smoothing_precision
from .frame import LocalFrame
from .gnss import (StationSeries, common_mode, fit_trajectory, gnss_observations, gnss_series, noise_levels,
                   prepare_stations, read_midas, read_steps, read_tenv3, screen_velocities, trajectory_design,
                   velocity_observations)
from .greens import HalfSpaceTDE
from .heterogeneity import HeterogeneousEngine, VoxelModel
from .interseismic import (EulerPole, InterseismicInversion, InterseismicModel, block_labels,
                           coupling_bounds, rotation_columns, side_labels, strain_from_coefficients)
from .inversion import SlipInversion, SlipModel
from .layered_tde import LayeredTDE
from .layout import MeshLayout, graded_layout, lloyd_layout, triangulate
from .mesh import FaultMesh
from .resample import ARCSEC_10, frame_lattice, resample_all, resample_to_frame
from .patterns import Pattern, poroelastic_pattern, separate_history
from .relaxation import RelaxationTables, Rheology, relaxation_history, relaxation_pattern
from .regional import RegionalModel, celeri_inputs, fit_frame_rotation, mesh_edges, west_first
from .predict import (iterate_sampling, model_rms_min, predicted_los,
                      sample_slip)
from .resolution import (
    adapt_layout,
    element_resolution,
    data_importance,
    resolution_diagonal,
    resolution_gain,
    resolution_kernels,
    resolution_matrix,
    epic_prior,
    blocked_cross_validation,
    blocked_cv_curve,
    blocked_folds,
    cell_folds,
    gcv_curve,
    generalized_cross_validation,
    resolution_bias,
    resolution_report,
    posterior_covariance,
    bound_fraction,
    size_field,
    target_element_size,
    weighted_design,
)
from .sampling import Observations
from .surface import FaultSurface, gridfit
from .sweep import (BootstrapResult, DipSweep, SweepResult, dip_profile_offsets,
                    geometry_key, segment_breaks)
from .temporal import SlipSnapshot, TimeBasis, TimeDependentInversion, TimeDependentModel, TrackSeries
from .trace import FaultSegment, FaultTrace, TraceSet
from .tsurf import TSurf, read_tsurf
from .uncertainty import Ensemble, checkerboard, monte_carlo, recovery, with_solution
from .wavenumber import RemainderTables

__all__ = [
    "ARCSEC_10",
    "Bimaterial",
    "BootstrapResult",
    "CovarianceModel",
    "DipSweep",
    "ElementBasis",
    "Ensemble",
    "EulerPole",
    "FaultMesh",
    "FaultSegment",
    "FaultSurface",
    "FaultTrace",
    "TraceSet",
    "GeometryEvidence",
    "HalfSpaceTDE",
    "HeterogeneousEngine",
    "InterseismicInversion",
    "InterseismicModel",
    "LayeredTDE",
    "LocalFrame",
    "LosGrid",
    "MeshLayout",
    "NodeBasis",
    "Observations",
    "Pattern",
    "RegionalModel",
    "RelaxationTables",
    "RemainderTables",
    "Rheology",
    "RigidityContrast",
    "SlipInversion",
    "SlipModel",
    "SlipSnapshot",
    "StationSeries",
    "SweepResult",
    "TSurf",
    "TimeBasis",
    "TimeDependentInversion",
    "TimeDependentModel",
    "TrackSeries",
    "VelocityModel",
    "VoxelModel",
    "adapt_layout",
    "block_labels",
    "bound_fraction",
    "calibrate",
    "celeri_inputs",
    "cell_folds",
    "cell_size_ladder",
    "checkerboard",
    "common_mode",
    "correlated_field",
    "coupling_bounds",
    "data_importance",
    "dip_profile_offsets",
    "element_resolution",
    "blocked_cross_validation",
    "blocked_cv_curve",
    "blocked_folds",
    "empirical_variogram",
    "epic_prior",
    "fit_covariance",
    "fit_frame_rotation",
    "fit_reml",
    "fit_trajectory",
    "frame_lattice",
    "geometry_key",
    "gcv_curve",
    "generalized_cross_validation",
    "gnss_observations",
    "gnss_series",
    "graded_layout",
    "gridfit",
    "iterate_sampling",
    "lloyd_layout",
    "mesh_edges",
    "model_rms_min",
    "monte_carlo",
    "noise_floor",
    "noise_levels",
    "observation_covariance",
    "posterior_covariance",
    "poroelastic_pattern",
    "relaxation_history",
    "relaxation_pattern",
    "predicted_los",
    "prepare_stations",
    "ramp_content",
    "ramp_leakage",
    "ramp_sweep",
    "read_grd",
    "read_midas",
    "read_steps",
    "read_tenv3",
    "read_tsurf",
    "recovery",
    "resample_all",
    "resample_to_frame",
    "resolution_bias",
    "resolution_diagonal",
    "resolution_gain",
    "resolution_kernels",
    "resolution_matrix",
    "resolution_report",
    "rotation_columns",
    "sample_slip",
    "scene_covariance",
    "scene_report",
    "screen_velocities",
    "segment_breaks",
    "separate_history",
    "side_labels",
    "strain_from_coefficients",
    "size_field",
    "smoothing_precision",
    "target_element_size",
    "trajectory_design",
    "triangulate",
    "velocity_observations",
    "weighted_design",
    "west_first",
    "whitening_factors",
    "with_solution",
]
