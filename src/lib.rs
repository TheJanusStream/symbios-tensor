//! Tensor-field-driven procedural urban layout generator.
//!
//! This crate generates realistic road networks and building lots on terrain
//! defined by a [`symbios_ground::HeightMap`]. Roads follow the natural
//! topography: **major** roads trace elevation contours while **minor** roads
//! run along the gradient, producing organic street grids that adapt to hills
//! and valleys. On flat terrain the field falls back to an axis-aligned
//! Manhattan grid.
//!
//! # Pipeline
//!
//! 1. **Road generation** — [`generate_roads`] seeds streamlines on a jittered
//!    grid and traces them through a [`TensorField`] using RK2 integration,
//!    snapping and splitting edges to form a planar [`RoadGraph`].
//! 2. **Block extraction** — [`extract_blocks`] walks the planar graph with a
//!    minimum-angle (left-most turn) algorithm, producing closed [`CityBlock`]
//!    polygons for every bounded interior face.
//! 3. **Lot subdivision** — [`extract_lots`] recursively splits each block
//!    perpendicular to its longest edge (through the centroid) until pieces
//!    are below a configurable area threshold, then computes a street-aligned
//!    [`BuildingLot`] rectangle with front/side/rear setbacks.
//! 4. **Terrain carving** — [`carve_roads`] and [`carve_lots`] flatten the
//!    heightmap under roads and building foundations with smooth embankment
//!    blending at the edges. Both accept a configurable `blend_radius` that
//!    controls how far the embankment zone extends — larger values produce
//!    wider, gentler slopes on steep terrain. `carve_roads` returns a boolean
//!    road-surface mask so that `carve_lots` can avoid overwriting
//!    already-flattened pavement.
//! 5. **Road pruning** — [`prune_unused_roads`] optionally removes roads that
//!    do not serve any building lot, keeping only the minimal connected
//!    sub-network via Dijkstra-based Steiner tree construction.
//! 6. **3D mesh generation** — [`generate_road_meshes`] produces engine-agnostic
//!    [`ProceduralMesh`] vertex buffers for intersection hubs (flat N-gon
//!    polygons) and street ribbons (Catmull-Rom-smoothed extruded strips).
//!
//! # Quick start
//!
//! ```ignore
//! use symbios_ground::HeightMap;
//! use symbios_tensor::*;
//!
//! let heightmap = HeightMap::new(128, 128, 4.0);
//! let config = TensorConfig::default();
//!
//! // 1. Generate road network
//! let mut graph = generate_roads(&heightmap, &config).expect("invalid config");
//!
//! // 2. Extract city blocks
//! extract_blocks(&mut graph);
//!
//! // 3. Subdivide blocks into building lots
//! let lots = extract_lots(&graph, &heightmap, config.water_level, &LotConfig::default());
//!
//! // 4. Carve roads and lots into terrain
//! let mut hm = heightmap;
//! let road_mask = carve_roads(&graph, &mut hm, 6.0, 4.0);
//! carve_lots(&lots, &mut hm, 2.0, Some(&road_mask));
//!
//! // 5. (Optional) Prune roads that don't serve any lot
//! prune_unused_roads(&mut graph, &lots);
//!
//! // 6. Generate 3D road meshes
//! let meshes = generate_road_meshes(&graph, &hm, &RoadMeshConfig::default());
//! // meshes.hubs — intersection polygons
//! // meshes.ribbons — street ribbon strips
//! ```

pub mod carve;
pub mod geometry;
pub mod graph;
pub mod lots;
pub mod polygons;
pub mod prune;
pub mod rationalize;
pub mod roads_3d;
pub mod spatial;
pub mod tensor;
pub mod topology;
pub mod tracer;

pub use carve::{carve_lots, carve_roads};
pub use graph::{BlockId, CityBlock, EdgeId, NodeId, RoadEdge, RoadGraph, RoadNode, RoadType};
pub use lots::{BuildingLot, LotConfig, extract_lots};
pub use polygons::{block_centroid, extract_blocks};
pub use prune::prune_unused_roads;
pub use rationalize::{RationalizeConfig, rationalize_graph};
pub use roads_3d::{ProceduralMesh, RoadMeshConfig, RoadMeshes, generate_road_meshes};
pub use tensor::TensorField;
pub use tracer::{TensorConfig, TensorError, generate_roads};
