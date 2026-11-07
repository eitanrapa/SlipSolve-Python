# Faults and meshes

The fault is described by its surface trace and a model of how it dips. From those
the package builds a triangulated surface in the [local frame](conventions.md#one-local-frame),
and slip is expanded over it per element or per node.

## The fault trace

A [`FaultTrace`](../api/geometry.md#slipsolve.trace.FaultTrace) is an ordered
lon/lat polyline:

```python
from slipsolve import FaultTrace

trace = FaultTrace.from_file("fault.kml")                 # KML from Google Earth
trace = FaultTrace.from_file("fault.txt")                 # or "lon lat" rows
trace = FaultTrace.from_lonlat(lon, lat, name="main")     # or arrays
frame = trace.local_frame()
```

The text reader skips `#` comments and GMT `>` segment headers, so the output of
`gmt kml2gmt` can be read as it is. Repeated vertices are dropped.

**Vertex order matters.** It fixes the along-strike direction, which side of the
fault is "left", and therefore the side a dip below 90° leans toward and the sense
of positive dip-slip (see [Conventions](conventions.md#the-sign-of-slip)).

A trace also measures the geometry the mesh and the samplers need:
`trace.length(frame)`, `trace.distance(x, y, frame)` (perpendicular distance to the
trace), `trace.side(x, y, frame)` (+1 left, -1 right) and
`trace.to_curvilinear(x, y, frame)` (arc length and signed cross-strike offset).
`trace.min_curvature_radius(frame)` bounds how far the trace can be projected down
dip before a curved fault folds through itself.

## Building a mesh

A [`FaultMesh`](../api/geometry.md#slipsolve.mesh.FaultMesh) is a triangulated fault
surface in local metres, with \(z \le 0\). It is parameterized by arc length along
the trace and depth, \((s, z)\), so its edges (top, bottom, the two ends) are known
exactly and its triangles are wound consistently.

### Vertical faults

```python
from slipsolve import FaultMesh

mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)
```

`edge_length` sets the element size: the trace is resampled at that spacing along
strike, and the depth range is divided into levels of about that height.
`top_depth` (default 0) starts the fault below the surface. The extrusion is exact:
every element is vertical to machine precision.

### Dipping and curved faults

`FaultMesh.curved` builds a fault that leaves its trace at depth. Give exactly one
description of the geometry:

```python
# One dip for the whole fault. Below 90 it dips toward the trace's left-hand side.
mesh = FaultMesh.curved(trace, frame, uniform_dip=70, max_depth=20e3, edge_length=3e3)

# One dip per segment: the trace cut into equal chords, or FaultSegment objects.
mesh = FaultMesh.curved(trace, frame, segments=3, dips=(70, 80, 90),
                        max_depth=20e3, edge_length=3e3)

# A digitised bottom edge (a FaultTrace or a file): the dip is whatever joins
# the two traces, and may change along strike.
mesh = FaultMesh.curved(trace, frame, bottom_trace="bottom.kml", bottom_depth=15e3,
                        max_depth=20e3, edge_length=3e3)

# A cross(s, z) surface you fitted yourself (slipsolve.FaultSurface).
mesh = FaultMesh.curved(trace, frame, surface=surface, max_depth=20e3, edge_length=3e3)
```

A uniform dip is placed in closed form: every node steps
\(\text{depth} / \tan(\text{dip})\) along the trace's local left-hand normal.
Segment dips and bottom traces are joined to the surface trace by a smooth fitted
surface (a port of the `gridfit` gridder SlipSolve-Curve uses), so the fitted
surface, not only the dips, shapes the fault between the constraints. Dips within
0.01° of vertical are snapped to exactly vertical, because the triangular
dislocation solution is numerically unstable just beside 90°.

Check the result: `mesh.dip`, `mesh.dip_direction` and `mesh.strike` are per
element, and [`plot_mesh`](../api/plotting.md#slipsolve.plot.plot_mesh) draws the
map view beside the unrolled section.

### Where the nodes go

By default the depth levels are evenly spaced. Several options place the nodes
differently, all with the geometry arguments above unchanged:

- `down_dip_levels=` and `bias_w=` give a geometric progression of level heights,
  thickening downward (`bias_w=1` is even), as SlipSolve-Curve's `biasW` does;
- `depths=` gives the levels outright;
- `nodes_per_level=` or `along_spacing=` (a scalar, one value per level, or a
  callable of depth) gives each level its own node count, for elements that grow
  with depth;
- `layout=` takes a [`MeshLayout`](../api/geometry.md#slipsolve.layout.MeshLayout)
  built elsewhere: `graded_layout` (free depth levels, each with its own count),
  `lloyd_layout` (fully unstructured, relaxed toward a target size `h(s, depth)`),
  or `triangulate` on any \((s, z)\) point set.

```python
from slipsolve import lloyd_layout

layout = lloyd_layout(length=trace.length(frame), max_depth=20e3,
                      size=lambda s, depth: 2e3 + 0.2 * depth)
mesh = FaultMesh.from_layout(layout, trace, frame, uniform_dip=90, edge_length=3e3)
```

`FaultMesh.from_layout` takes the same geometry keywords as `curved`; the depth
extent comes from the layout. `balance` caps how fast the node count may change
between adjacent levels, which keeps Delaunay from making long thin triangles.

Element sizes can also be chosen from the data: see
[mesh adaptation](regularization.md#adapting-the-mesh-and-the-sampling).

### Several faults

`FaultMesh.join([mesh_a, mesh_b])` holds several disjoint faults in one mesh for a
joint inversion. Each keeps its own edges, so zero-slip boundaries apply per fault.
`SlipInversion(..., sees={track: [fault indices]})` restricts which faults a track
measures, for a sequence in which some data span every event and others only one.

## Slip per element or per node

The slip on the mesh is expanded in a basis, chosen with `SlipInversion(basis=...)`:

- **`"element"`** (default, [`ElementBasis`](../api/geometry.md#slipsolve.basis.ElementBasis)):
  constant slip on each triangle, the assumption of the triangular dislocation
  solution. Smoothing penalises differences between neighbouring elements.
- **`"node"`** ([`NodeBasis`](../api/geometry.md#slipsolve.basis.NodeBasis)): a
  continuous, piecewise-linear field with one value per node (tent functions).
  Roughly half the parameters, no slip discontinuity on element edges, and
  smoothing by a Laplace–Beltrami operator on the fault surface. The layered
  engines and `HeterogeneousEngine` integrate each tent directly; the half-space
  engine and the exact bimaterial mode compute per element, and the element columns
  are projected onto the tents.

Whatever the basis, `SlipModel.element_slip` gives slip per element and
`SlipModel.nodal_slip()` per node.

## Saving a mesh

`mesh.to_dataset()` and `FaultMesh.from_dataset(ds)` round-trip a mesh through an
xarray Dataset, and a saved [`SlipModel`](inversion.md#saving-and-exporting) carries
its mesh.
