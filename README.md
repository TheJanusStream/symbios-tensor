# symbios-tensor

Tensor-field-driven procedural urban layout generator for terrain-aware road
networks and building lots.

Roads follow the natural topography of a heightmap: **major** roads trace
elevation contours while **minor** roads run along the gradient, producing
organic street grids that adapt to hills and valleys. On flat terrain the
field falls back to an axis-aligned Manhattan grid.

## Pipeline

1. **Road generation** (`generate_roads`) — Seeds streamlines on a jittered
   grid and traces them bidirectionally through a `TensorField` using RK2
   (midpoint) integration. Traces snap to existing nodes and split edges via
   a spatial hash, forming T-junctions and 4-way intersections. Orthogonal
   branches are spawned at configurable intervals.

2. **Graph rationalization** (`rationalize_graph`) — Rewrites the raw tracer
   output into clean, civil-engineered geometry. Traces continuous arteries
   *through* intersections (by forward-vector alignment), applies
   Ramer-Douglas-Peucker decimation to remove unnecessary intermediate
   points, and replaces sharp bends with smooth quadratic Bézier fillet arcs.
   Elevation profiles are smoothed with global Laplacian passes and clamped
   to a maximum grade. Side-streets severed by moved intersection nodes are automatically
   reconnected.

3. **Block extraction** (`extract_blocks`) — Walks the planar road graph with
   a minimum-angle (left-most turn) algorithm to enumerate every bounded
   interior face as a `CityBlock` polygon. The unbounded exterior face is
   filtered out by winding direction.

4. **Lot subdivision** (`extract_lots`) — Recursively splits each block
   perpendicular to its longest edge (through the centroid) until sub-polygons
   fall below a configurable area threshold. A street-aligned inscribed
   rectangle is computed for each piece, with front/side/rear setbacks applied
   to produce `BuildingLot` footprints.

5. **Terrain carving** (`carve_roads`, `carve_lots`) — Flattens the heightmap
   under roads and building foundations with smooth embankment blending at the
   edges. `carve_roads` takes a `road_width` (total width of the flat road
   surface) and a `blend_radius` that controls how far the embankment zone
   extends beyond the road edge — larger values produce wider, gentler slopes
   on steep terrain. `carve_lots` similarly accepts a `blend_radius` for
   foundation embankments. `carve_roads` returns a
   boolean road-surface mask so that `carve_lots` can avoid overwriting
   already-flattened pavement. A two-pass road carving approach prevents
   embankments from overwriting previously flattened pavement at intersections.
   Road elevations use the rationalized (smoothed) node heights rather than
   raw terrain, so roads cut through hills and bridge dips.

6. **Road pruning** (`prune_unused_roads`) — Optionally removes roads that do
   not serve any building lot. Uses Dijkstra-based Steiner tree construction
   to keep only the minimal connected sub-network that reaches every lot's
   frontage edge, preferring major roads over minor ones.

7. **3D mesh generation** (`generate_road_meshes`) — Produces engine-agnostic
   `ProceduralMesh` vertex buffers for intersection hubs (flat N-gon polygons
   with embankment skirts) and street ribbons (extruded strips with flanking
   skirt meshes), using rationalized node elevations for correct height.

## Quick start

```rust
use symbios_ground::HeightMap;
use symbios_tensor::*;

let heightmap = HeightMap::new(128, 128, 4.0);
let config = TensorConfig::default();

// 1. Generate road network
let mut graph = generate_roads(&heightmap, &config).expect("invalid config");

// 2. Rationalize: straighten, fillet, and smooth elevations
rationalize_graph(&mut graph, &heightmap, &RationalizeConfig::default());

// 3. Extract city blocks
extract_blocks(&mut graph);

// 4. Subdivide blocks into building lots
let lots = extract_lots(&graph, &heightmap, config.water_level, &LotConfig::default());

// 5. Carve roads and lots into terrain
let mut hm = heightmap;
let road_mask = carve_roads(&graph, &mut hm, &RoadMeshConfig::default(), 4.0);
carve_lots(&lots, &mut hm, 2.0, Some(&road_mask));

// 6. (Optional) Prune roads that don't serve any lot
prune_unused_roads(&mut graph, &lots);

// 7. Generate 3D road meshes
let meshes = generate_road_meshes(&graph, &hm, &RoadMeshConfig::default());
// meshes.hubs    — intersection polygons
// meshes.ribbons — street ribbon strips
// meshes.skirts  — embankment skirt meshes
```

## Configuration

### `TensorConfig`

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | RNG seed for jittered seed placement |
| `step_size` | `2.0` | World-space distance per integration step |
| `major_road_dist` | `40.0` | Spacing between parallel major roads (avenues) |
| `minor_road_dist` | `15.0` | Spacing between parallel minor roads (streets) |
| `snap_radius` | `4.0` | Merge radius for snapping trace endpoints to existing geometry |
| `max_trace_steps` | `300` | Maximum integration steps per trace before abandoning |
| `tracer_inertia` | `0.8` | Momentum factor (0.0–0.99) for the tracer direction; higher values resist sharp changes from terrain noise, producing smoother roads |
| `water_level` | `-inf` | World-space Y height of the water plane; terrain at or below this is treated as underwater |

### `LotConfig`

| Field | Default | Description |
|---|---|---|
| `max_lot_area` | `400.0` | Maximum lot area (sqm) before recursive subdivision |
| `min_lot_area` | `50.0` | Minimum lot area — polygons below this are discarded |
| `front_setback` | `3.0` | Distance from street edge to building front |
| `side_setback` | `1.5` | Distance from side edges to building sides |
| `rear_setback` | `2.0` | Distance from back edge to building rear |
| `min_width` | `6.0` | Minimum building width (along street) |
| `min_depth` | `6.0` | Minimum building depth (perpendicular to street) |

### `RationalizeConfig`

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Master toggle for rationalization |
| `rdp_tolerance` | `2.0` | Ramer-Douglas-Peucker tolerance in world units — how aggressively to straighten |
| `major_fillet_radius` | `20.0` | Fillet radius for major (contour-following) roads |
| `minor_fillet_radius` | `10.0` | Fillet radius for minor (gradient-following) roads |
| `fillet_segments` | `6` | Number of line segments used to approximate each fillet arc |
| `elevation_smooth_passes` | `10` | Global Laplacian smoothing passes applied to the road graph's elevation profile before chain extraction; 0 disables |
| `max_grade` | `0.15` | Maximum allowed slope between adjacent nodes (e.g. 0.15 = 15% grade); 0.0 disables |

### `RoadMeshConfig`

| Field | Default | Description |
|---|---|---|
| `major_half_width` | `3.0` | Half-width of major roads (world units) |
| `minor_half_width` | `2.0` | Half-width of minor roads (world units) |
| `hub_sides` | `8` | Number of sides for dead-end cap polygons (e.g. 8 = octagon); degree-3+ intersections use procedural boundary polygons |
| `depth_bias` | `0.05` | Vertices are raised above terrain by this amount to prevent z-fighting |
| `texture_scale` | `0.1` | UV texture scale: world units per texture repeat |
| `spline_subdivisions` | `8` | Legacy: Catmull-Rom subdivisions per graph edge; ignored when the graph has been rationalized |
| `curb_radius` | `2.0` | Extra radius added to dead-end caps beyond the road half-width, creating a wider turning zone |
| `skirt` | `SkirtConfig::default()` | Embankment skirt configuration (see below) |

### `SkirtConfig`

| Field | Default | Description |
|---|---|---|
| `width` | `3.0` | Width of the skirt extending outward from the road edge (world units) |
| `bury_depth` | `0.5` | How far below the terrain surface the skirt buries itself |

## Module overview

| Module | Purpose |
|---|---|
| `tensor` | Tensor field sampling from heightmap normals (major/minor directions) |
| `tracer` | Streamline tracing with RK2 integration, branching, and snap logic |
| `graph` | Arena-based road graph (`RoadNode`, `RoadEdge`, `CityBlock`) |
| `spatial` | Spatial hash grid for O(1) proximity and intersection queries |
| `geometry` | Segment intersection and closest-point primitives |
| `rationalize` | RDP decimation, Bézier fillet smoothing, Laplacian elevation smoothing, grade clamping, and artery-through-intersection straightening |
| `topology` | Shared topology helpers: chain extraction, artery extraction, active degree computation |
| `polygons` | Minimum-cycle-basis block extraction and centroid computation |
| `lots` | Recursive subdivision, frontage detection, inscribed box, setbacks |
| `carve` | Heightmap flattening for roads and building foundations |
| `prune` | Steiner-tree road pruning to remove unused roads |
| `roads_3d` | Engine-agnostic 3D mesh generation (hubs, ribbons, embankment skirts) |

## Dependencies

- [`symbios-ground`](https://github.com/TheJanusStream/symbios-ground) — heightmap terrain (`HeightMap`)
- [`glam`](https://crates.io/crates/glam) — 2D/3D math (`Vec2`, `Vec3`)
- [`rand`](https://crates.io/crates/rand) + [`rand_pcg`](https://crates.io/crates/rand_pcg) — deterministic RNG for seed jitter
- [`serde`](https://crates.io/crates/serde) — serialization for configs, graph, and lots

## Known limitations

- **Flat terrain blocks**: On perfectly flat heightmaps the tracer produces
  mostly parallel streamlines that form degenerate zero-area slivers rather
  than enclosed blocks. Real enclosed blocks require terrain with slopes so
  that major and minor directions cross at non-trivial angles.

## License

MIT
