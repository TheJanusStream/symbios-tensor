//! On-demand tile-based city streaming for unbounded worlds.
//!
//! The full city pipeline (`generate_roads` → `rationalize_graph` →
//! `extract_blocks` → `extract_lots`) keeps the entire road graph in
//! memory, which is fine for fixed-size worlds but doesn't scale to
//! 10×10 km open-world regions. [`CityStreamer`] divides the world into
//! square tiles, runs the pipeline per tile on demand, caches the
//! results, and lets the caller evict distant tiles when memory grows.
//!
//! # Per-tile coordinate system
//!
//! Each tile is a square of side `tile_size` world units. Tile `(i, j)`
//! occupies world AABB `[i·tile_size, (i+1)·tile_size] × [j·tile_size,
//! (j+1)·tile_size]`. The caller supplies a heightmap callback that
//! returns a [`HeightMap`] in **tile-local coordinates** (i.e. world
//! coordinates `[0, tile_size]` along each axis). The streamer offsets
//! the resulting graph nodes back into world space before returning.
//!
//! # Seamlessness
//!
//! Each tile is generated with an independent tensor field over its own
//! heightmap. The tensor field itself is locally consistent because it
//! samples the same global heightmap (assuming the caller's callback is
//! consistent across tile boundaries), but **streamlines do not extend
//! across tile borders** — a road approaching the edge of tile `(i, j)`
//! ends there, and a separately-seeded road in tile `(i+1, j)` may or
//! may not align with it. Truly seamless cross-tile tracing is research
//! work and out of scope here. For most consumers this is acceptable
//! because tile borders are designed to fall on natural seams (rivers,
//! highway corridors) or are simply hidden by mesh LOD at distance.

use std::collections::HashMap;

use glam::Vec2;
use symbios_ground::HeightMap;

use crate::graph::RoadGraph;
use crate::lots::{BuildingLot, LotConfig, extract_lots};
use crate::polygons::extract_blocks;
use crate::rationalize::{RationalizeConfig, rationalize_graph};
use crate::tracer::{GenerationError, TensorConfig, generate_roads};

/// Configuration for a [`CityStreamer`].
#[derive(Debug, Clone)]
pub struct CityStreamerConfig {
    /// Side length of each tile in world units. All tiles are square.
    /// Must be positive.
    pub tile_size: f32,
    /// Base seed mixed with tile coordinates to derive per-tile seeds.
    /// Reproducibility: the same `base_seed` + tile coordinate always
    /// yields the same tile content.
    pub base_seed: u64,
    /// Tensor / road generation parameters. The `seed` field is overwritten
    /// per-tile; the `water_level` and `field` are forwarded as-is.
    pub tensor: TensorConfig,
    /// Graph rationalization parameters.
    pub rationalize: RationalizeConfig,
    /// Lot extraction parameters.
    pub lots: LotConfig,
}

/// One generated city tile: graph + lots + the heightmap that produced
/// them. Node positions are in **world** coordinates (offset by the
/// tile's origin).
#[derive(Debug, Clone)]
pub struct CityTile {
    /// Integer tile index along X.
    pub tile_x: i32,
    /// Integer tile index along Z.
    pub tile_z: i32,
    /// World-space origin of this tile (lower-left corner).
    pub origin: Vec2,
    /// Side length copied from [`CityStreamerConfig::tile_size`].
    pub size: f32,
    /// Road graph in **world** coordinates. Block perimeters are valid.
    pub graph: RoadGraph,
    /// Building lots in **world** coordinates.
    pub lots: Vec<BuildingLot>,
    /// The (potentially carved-by-lot-flush) heightmap used to generate
    /// this tile, kept for downstream meshing.
    pub heightmap: HeightMap,
}

/// Streams city tiles on demand.
///
/// `P` produces a heightmap for tile `(tile_x, tile_z)` whose world
/// coordinates run from `0` to `config.tile_size` along each axis.
/// Returning a heightmap of any other size is a programmer error; the
/// streamer asserts on world-size mismatch.
pub struct CityStreamer<P>
where
    P: FnMut(i32, i32) -> HeightMap,
{
    config: CityStreamerConfig,
    provider: P,
    cache: HashMap<(i32, i32), CityTile>,
}

impl<P> CityStreamer<P>
where
    P: FnMut(i32, i32) -> HeightMap,
{
    /// Creates a new streamer.
    ///
    /// # Panics
    ///
    /// Panics if `config.tile_size <= 0.0` or non-finite.
    pub fn new(config: CityStreamerConfig, provider: P) -> Self {
        assert!(
            config.tile_size.is_finite() && config.tile_size > 0.0,
            "tile_size must be positive and finite, got {}",
            config.tile_size
        );
        Self {
            config,
            provider,
            cache: HashMap::new(),
        }
    }

    /// Returns the configured tile size.
    pub fn tile_size(&self) -> f32 {
        self.config.tile_size
    }

    /// Number of tiles currently cached in memory.
    pub fn cached_tile_count(&self) -> usize {
        self.cache.len()
    }

    /// Returns the integer tile coordinate that contains `world_pos`.
    pub fn tile_coord_for(&self, world_pos: Vec2) -> (i32, i32) {
        let s = self.config.tile_size;
        (
            (world_pos.x / s).floor() as i32,
            (world_pos.y / s).floor() as i32,
        )
    }

    /// Generates tile `(tile_x, tile_z)` if not cached and returns a
    /// reference to it.
    pub fn ensure_tile(&mut self, tile_x: i32, tile_z: i32) -> Result<&CityTile, GenerationError> {
        if !self.cache.contains_key(&(tile_x, tile_z)) {
            let tile = self.generate_tile(tile_x, tile_z)?;
            self.cache.insert((tile_x, tile_z), tile);
        }
        Ok(&self.cache[&(tile_x, tile_z)])
    }

    /// Ensures every tile overlapping the AABB `[min, max]` is generated
    /// and returns references to all of them in unspecified order.
    pub fn query_region(
        &mut self,
        min: Vec2,
        max: Vec2,
    ) -> Result<Vec<&CityTile>, GenerationError> {
        let s = self.config.tile_size;
        if !min.x.is_finite() || !min.y.is_finite() || !max.x.is_finite() || !max.y.is_finite() {
            return Ok(Vec::new());
        }
        let tx_min = (min.x / s).floor() as i32;
        let tz_min = (min.y / s).floor() as i32;
        let tx_max = ((max.x / s).floor() as i32).max(tx_min);
        let tz_max = ((max.y / s).floor() as i32).max(tz_min);

        for tz in tz_min..=tz_max {
            for tx in tx_min..=tx_max {
                self.ensure_tile(tx, tz)?;
            }
        }

        let mut out = Vec::new();
        for tz in tz_min..=tz_max {
            for tx in tx_min..=tx_max {
                if let Some(t) = self.cache.get(&(tx, tz)) {
                    out.push(t);
                }
            }
        }
        Ok(out)
    }

    /// Evicts every cached tile whose center is farther than
    /// `max_distance` from `center`. Returns the number of tiles evicted.
    pub fn evict_outside(&mut self, center: Vec2, max_distance: f32) -> usize {
        let s = self.config.tile_size;
        let max_sq = max_distance * max_distance;
        let to_remove: Vec<(i32, i32)> = self
            .cache
            .keys()
            .filter(|&&(tx, tz)| {
                let tile_center = Vec2::new((tx as f32 + 0.5) * s, (tz as f32 + 0.5) * s);
                tile_center.distance_squared(center) > max_sq
            })
            .copied()
            .collect();
        let n = to_remove.len();
        for k in to_remove {
            self.cache.remove(&k);
        }
        n
    }

    fn generate_tile(&mut self, tile_x: i32, tile_z: i32) -> Result<CityTile, GenerationError> {
        let mut heightmap = (self.provider)(tile_x, tile_z);
        let s = self.config.tile_size;
        let provided_w = heightmap.world_width();
        let provided_d = heightmap.world_depth();
        debug_assert!(
            (provided_w - s).abs() < 1e-3 && (provided_d - s).abs() < 1e-3,
            "heightmap_provider returned heightmap of {provided_w}x{provided_d} world units, expected {s}x{s}",
        );

        let mut tensor_config = self.config.tensor.clone();
        tensor_config.seed = mix_seed(self.config.base_seed, tile_x, tile_z);

        let mut graph = generate_roads(&heightmap, &tensor_config)?;
        rationalize_graph(&mut graph, &heightmap, &self.config.rationalize);
        extract_blocks(&mut graph);
        let lots = extract_lots(&graph, &mut heightmap, &self.config.lots);

        // Translate everything from tile-local to world coordinates.
        let origin = Vec2::new(tile_x as f32 * s, tile_z as f32 * s);
        for node in &mut graph.nodes {
            node.position += origin;
        }
        let mut lots = lots;
        for lot in &mut lots {
            lot.position += origin;
            lot.frontage_center += origin;
        }

        Ok(CityTile {
            tile_x,
            tile_z,
            origin,
            size: s,
            graph,
            lots,
            heightmap,
        })
    }
}

/// Mixes a base seed with integer tile coordinates to produce a
/// deterministic per-tile seed. Uses splitmix64-style avalanche so that
/// adjacent tiles have uncorrelated seeds.
fn mix_seed(base: u64, tile_x: i32, tile_z: i32) -> u64 {
    let x = tile_x as i64 as u64;
    let z = tile_z as i64 as u64;
    let mut h = base ^ x.wrapping_mul(0x9E3779B97F4A7C15);
    h = h.wrapping_add(z.wrapping_mul(0xBF58476D1CE4E5B9));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^ (h >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_streamer() -> CityStreamer<impl FnMut(i32, i32) -> HeightMap> {
        let cfg = CityStreamerConfig {
            tile_size: 32.0,
            base_seed: 7,
            tensor: TensorConfig {
                step_size: 1.0,
                major_road_dist: 12.0,
                minor_road_dist: 6.0,
                snap_radius: 2.0,
                max_trace_steps: 80,
                ..Default::default()
            },
            rationalize: RationalizeConfig::default(),
            lots: LotConfig::default(),
        };

        // Heightmap provider: a simple slope that varies with tile so
        // each tile's tensor field is non-trivial.
        let provider = move |tile_x: i32, tile_z: i32| {
            let mut hm = HeightMap::new(32, 32, 1.0);
            for z in 0..32 {
                for x in 0..32 {
                    let global_x = tile_x as f32 * 32.0 + x as f32;
                    let global_z = tile_z as f32 * 32.0 + z as f32;
                    let h = (global_x * 0.05).sin() * 2.0 + (global_z * 0.07).cos();
                    hm.set(x, z, h);
                }
            }
            hm
        };

        CityStreamer::new(cfg, provider)
    }

    #[test]
    fn ensure_tile_caches() {
        let mut s = make_streamer();
        assert_eq!(s.cached_tile_count(), 0);
        s.ensure_tile(0, 0).expect("generate (0,0)");
        assert_eq!(s.cached_tile_count(), 1);
        // Re-requesting same tile must not regenerate.
        s.ensure_tile(0, 0).expect("re-request (0,0)");
        assert_eq!(s.cached_tile_count(), 1);
    }

    #[test]
    fn tile_nodes_in_world_coordinates() {
        let mut s = make_streamer();
        let tile = s.ensure_tile(2, 3).expect("generate").clone();
        let origin = tile.origin;
        let size = tile.size;
        for node in &tile.graph.nodes {
            assert!(
                node.position.x >= origin.x - 1e-3
                    && node.position.x <= origin.x + size + 1e-3
                    && node.position.y >= origin.y - 1e-3
                    && node.position.y <= origin.y + size + 1e-3,
                "node at {:?} outside tile (origin={origin:?}, size={size})",
                node.position
            );
        }
    }

    #[test]
    fn query_region_spans_multiple_tiles() {
        let mut s = make_streamer();
        let tiles = s
            .query_region(Vec2::new(-10.0, -10.0), Vec2::new(50.0, 50.0))
            .expect("query_region");
        // The AABB spans tiles (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0),
        // (0, 1), (1, 1), (-1, 1), (1, -1) — all 9 of the 3×3 block.
        assert_eq!(tiles.len(), 9, "expected 9 tiles, got {}", tiles.len());
    }

    #[test]
    fn evict_outside_drops_far_tiles() {
        let mut s = make_streamer();
        s.ensure_tile(0, 0).expect("(0,0)");
        s.ensure_tile(5, 5).expect("(5,5)");
        assert_eq!(s.cached_tile_count(), 2);

        // (0,0) center is at (16, 16); (5,5) center is at (176, 176).
        // Evict everything farther than 100 units from (16, 16).
        let evicted = s.evict_outside(Vec2::new(16.0, 16.0), 100.0);
        assert_eq!(evicted, 1);
        assert_eq!(s.cached_tile_count(), 1);
    }

    #[test]
    fn deterministic_seeds_per_tile() {
        let mut s1 = make_streamer();
        let mut s2 = make_streamer();
        let t1 = s1.ensure_tile(3, 4).unwrap().clone();
        let t2 = s2.ensure_tile(3, 4).unwrap().clone();
        assert_eq!(t1.graph.nodes.len(), t2.graph.nodes.len());
        for (a, b) in t1.graph.nodes.iter().zip(t2.graph.nodes.iter()) {
            assert!((a.position - b.position).length() < 1e-5);
        }
    }
}
