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
//! 1. **Road generation** — [`generate_roads`] seeds streamlines on a regular
//!    grid and traces them through a [`TensorField`] using RK2 integration,
//!    snapping and splitting edges to form a planar [`RoadGraph`].
//! 2. **Block extraction** — [`extract_blocks`] walks the planar graph with a
//!    minimum-angle (left-most turn) algorithm, producing closed [`CityBlock`]
//!    polygons for every bounded interior face.
//! 3. **Lot subdivision** — [`extract_lots`] recursively splits each block
//!    along its longest edge until pieces are below a configurable area
//!    threshold, then computes a street-aligned [`BuildingLot`] rectangle with
//!    front/side/rear setbacks.
//! 4. **Terrain carving** — [`carve_roads`] and [`carve_lots`] flatten the
//!    heightmap under roads and building foundations with smooth embankment
//!    blending at the edges.
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
//! let mut graph = generate_roads(&heightmap, &config);
//!
//! // 2. Extract city blocks
//! extract_blocks(&mut graph);
//!
//! // 3. Subdivide blocks into building lots
//! let lots = extract_lots(&graph, &LotConfig::default());
//!
//! // 4. Carve roads and lots into terrain
//! let mut hm = heightmap;
//! carve_roads(&graph, &mut hm, 6.0);
//! carve_lots(&lots, &mut hm, 2.0);
//! ```

pub mod carve;
pub mod geometry;
pub mod graph;
pub mod lots;
pub mod polygons;
pub mod spatial;
pub mod tensor;
pub mod tracer;

pub use carve::{carve_lots, carve_roads};
pub use graph::{BlockId, CityBlock, EdgeId, NodeId, RoadEdge, RoadGraph, RoadNode, RoadType};
pub use lots::{BuildingLot, LotConfig, extract_lots};
pub use polygons::{block_centroid, extract_blocks};
pub use tensor::TensorField;
pub use tracer::{TensorConfig, generate_roads};
