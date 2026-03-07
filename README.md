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

2. **Block extraction** (`extract_blocks`) — Walks the planar road graph with
   a minimum-angle (left-most turn) algorithm to enumerate every bounded
   interior face as a `CityBlock` polygon. The unbounded exterior face is
   filtered out by winding direction.

3. **Lot subdivision** (`extract_lots`) — Recursively splits each block along
   its longest edge until sub-polygons fall below a configurable area
   threshold. A street-aligned inscribed rectangle is computed for each piece,
   with front/side/rear setbacks applied to produce `BuildingLot` footprints.

4. **Terrain carving** (`carve_roads`, `carve_lots`) — Flattens the heightmap
   under roads and building foundations with smooth embankment blending at the
   edges. `carve_roads` returns a boolean road-surface mask so that
   `carve_lots` can avoid overwriting already-flattened pavement. A two-pass
   road carving approach prevents embankments from overwriting previously
   flattened pavement at intersections.

5. **Road pruning** (`prune_unused_roads`) — Optionally removes roads that do
   not serve any building lot. Uses Dijkstra-based Steiner tree construction
   to keep only the minimal connected sub-network that reaches every lot's
   frontage edge, preferring major roads over minor ones.

## Quick start

```rust
use symbios_ground::HeightMap;
use symbios_tensor::*;

let heightmap = HeightMap::new(128, 128, 4.0);
let config = TensorConfig::default();

// 1. Generate road network
let mut graph = generate_roads(&heightmap, &config);

// 2. Extract city blocks
extract_blocks(&mut graph);

// 3. Subdivide blocks into building lots
let lots = extract_lots(&graph, &heightmap, config.water_level, &LotConfig::default());

// 4. Carve roads and lots into terrain
let mut hm = heightmap;
let road_mask = carve_roads(&graph, &mut hm, 6.0);
carve_lots(&lots, &mut hm, 2.0, Some(&road_mask));

// 5. (Optional) Prune roads that don't serve any lot
prune_unused_roads(&mut graph, &lots);
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

## Module overview

| Module | Purpose |
|---|---|
| `tensor` | Tensor field sampling from heightmap normals (major/minor directions) |
| `tracer` | Streamline tracing with RK2 integration, branching, and snap logic |
| `graph` | Arena-based road graph (`RoadNode`, `RoadEdge`, `CityBlock`) |
| `spatial` | Spatial hash grid for O(1) proximity and intersection queries |
| `geometry` | Segment intersection and closest-point primitives |
| `polygons` | Minimum-cycle-basis block extraction and centroid computation |
| `lots` | Recursive subdivision, frontage detection, inscribed box, setbacks |
| `carve` | Heightmap flattening for roads and building foundations |
| `prune` | Steiner-tree road pruning to remove unused roads |

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
