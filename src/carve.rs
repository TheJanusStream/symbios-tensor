//! Heightmap carving for roads and building lots.
//!
//! After the road network and lots are generated, these functions modify the
//! underlying [`symbios_ground::HeightMap`] to create flat surfaces where
//! roads and buildings sit, with smooth embankment blending at the edges.

use symbios_ground::HeightMap;

use crate::graph::{RoadGraph, RoadType};
use crate::lots::BuildingLot;
use crate::roads_3d::RoadMeshConfig;

/// Flattens the heightmap under each building lot with blended embankments.
///
/// Each lot's footprint is set to a uniform foundation height (sampled at the
/// lot center). Grid cells within `blend_radius` of the footprint edge are
/// linearly interpolated between the foundation and natural terrain height.
pub fn carve_lots(
    lots: &[BuildingLot],
    heightmap: &mut HeightMap,
    blend_radius: f32,
    road_surface: Option<&[bool]>,
) {
    let scale = heightmap.scale();
    let hw = heightmap.width();
    let hh = heightmap.height();
    if hw == 0 || hh == 0 {
        return;
    }

    if let Some(mask) = road_surface {
        assert_eq!(
            mask.len(),
            hw * hh,
            "road_surface mask length ({}) must match heightmap dimensions ({}x{} = {})",
            mask.len(),
            hw,
            hh,
            hw * hh,
        );
    }

    // Track cells that have been set to a lot foundation height so that
    // the embankment pass does not overwrite them (same pattern as carve_roads).
    let mut is_lot_surface = vec![false; hw * hh];

    // Pass 1: flatten all lot foundation surfaces.
    for lot in lots {
        let target_h = heightmap.get_height_at(lot.position.x, lot.position.y);

        let lp = LotParams {
            center: lot.position,
            cos: lot.rotation.cos(),
            sin: lot.rotation.sin(),
            half_w: lot.width * 0.5,
            half_d: lot.depth * 0.5,
        };

        let (gx_start, gx_end, gz_start, gz_end) =
            lot_grid_bounds(lot, lp.half_w, lp.half_d, blend_radius, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                if let Some(mask) = road_surface
                    && mask[gz * hw + gx]
                {
                    continue;
                }

                let dist_to_edge = lot_cell_distance(gx, gz, scale, &lp);

                if dist_to_edge <= 0.0 {
                    heightmap.set(gx, gz, target_h);
                    is_lot_surface[gz * hw + gx] = true;
                }
            }
        }
    }

    // Pass 2: embankments — for each cell, only the closest lot's embankment
    // applies. Protected cells (road surfaces and lot foundations) are skipped.
    let mut embankment_dist = vec![f32::MAX; hw * hh];
    let mut embankment_lot_h = vec![0.0_f32; hw * hh];

    for lot in lots {
        let target_h = heightmap.get_height_at(lot.position.x, lot.position.y);

        let lp = LotParams {
            center: lot.position,
            cos: lot.rotation.cos(),
            sin: lot.rotation.sin(),
            half_w: lot.width * 0.5,
            half_d: lot.depth * 0.5,
        };

        let (gx_start, gx_end, gz_start, gz_end) =
            lot_grid_bounds(lot, lp.half_w, lp.half_d, blend_radius, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let idx = gz * hw + gx;

                if let Some(mask) = road_surface
                    && mask[idx]
                {
                    continue;
                }
                if is_lot_surface[idx] {
                    continue;
                }

                let dist_to_edge = lot_cell_distance(gx, gz, scale, &lp);

                if dist_to_edge > 0.0 && dist_to_edge < blend_radius && dist_to_edge < embankment_dist[idx] {
                    embankment_dist[idx] = dist_to_edge;
                    embankment_lot_h[idx] = target_h;
                }
            }
        }
    }

    // Apply the winning embankment blend for each cell.
    for gz in 0..hh {
        for gx in 0..hw {
            let idx = gz * hw + gx;
            if embankment_dist[idx] < f32::MAX {
                let t = embankment_dist[idx] / blend_radius;
                let terrain_h = heightmap.get(gx, gz);
                let blended_h = embankment_lot_h[idx] + t * (terrain_h - embankment_lot_h[idx]);
                heightmap.set(gx, gz, blended_h);
            }
        }
    }
}

/// Flattens the heightmap along road edges and around intersection hubs,
/// creating smooth graded surfaces with blended embankments at the edges.
///
/// Uses `road_config` to determine per-type road widths (major vs minor) and
/// hub radii (including `curb_radius`), so the carved terrain matches the 3D
/// mesh exactly. `blend_radius` controls how far the embankment zone extends
/// beyond the road/hub surface.
///
/// Returns a boolean mask (`Vec<bool>`, one entry per heightmap cell) marking
/// cells that are part of a road surface. Pass this to [`carve_lots`] to
/// prevent building foundations from overwriting road pavement.
///
/// Uses a multi-pass approach: hub circles are flattened first, then edge
/// surfaces, then embankments are blended. This prevents later passes from
/// overwriting previously flattened pavement at intersections.
pub fn carve_roads(
    graph: &RoadGraph,
    heightmap: &mut HeightMap,
    road_config: &RoadMeshConfig,
    blend_radius: f32,
) -> Vec<bool> {
    let scale = heightmap.scale();
    let hw = heightmap.width();
    let hh = heightmap.height();
    if hw == 0 || hh == 0 {
        return Vec::new();
    }

    // Use the sovereign node elevations (smoothed by rationalization) as the
    // reference heights for carving. This makes roads slice through hills
    // and bridge dips rather than hugging every terrain bump.
    let node_heights: Vec<f32> = graph
        .nodes
        .iter()
        .map(|n| n.elevation)
        .collect();

    // Compute active degree for each node to identify hubs.
    let degrees = crate::topology::compute_active_degrees(graph);

    // Track cells that have been set to a road surface height so that
    // the embankment pass does not overwrite them.
    let mut is_road_surface = vec![false; hw * hh];

    // Pass 1a: flatten hub circles (nodes with degree != 2).
    for (nid, &deg) in degrees.iter().enumerate() {
        if deg == 0 || deg == 2 {
            continue;
        }
        let node = &graph.nodes[nid];
        let center = node.position;
        let center_h = node.elevation;

        // Hub radius: max half-width of connecting edges + curb radius
        // (mirrors the logic in roads_3d::generate_hub).
        let mut radius = road_config.minor_half_width;
        for &eid in &node.edges {
            let edge = &graph.edges[eid as usize];
            if !edge.active {
                continue;
            }
            let hw_edge = match edge.road_type {
                RoadType::Major => road_config.major_half_width,
                RoadType::Minor => road_config.minor_half_width,
            };
            if hw_edge > radius {
                radius = hw_edge;
            }
        }
        radius += road_config.curb_radius;

        // Flatten a circle of `radius` around the hub center.
        let expand = radius + scale;
        let gx_start = ((center.x - expand).max(0.0) / scale).floor() as usize;
        let gx_end = (((center.x + expand) / scale).ceil() as usize).min(hw - 1);
        let gz_start = ((center.y - expand).max(0.0) / scale).floor() as usize;
        let gz_end = (((center.y + expand) / scale).ceil() as usize).min(hh - 1);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let wx = gx as f32 * scale;
                let wz = gz as f32 * scale;
                let dist = ((wx - center.x).powi(2) + (wz - center.y).powi(2)).sqrt();
                if dist <= radius {
                    heightmap.set(gx, gz, center_h);
                    is_road_surface[gz * hw + gx] = true;
                }
            }
        }

        // Hub embankment ring.
        let outer = radius + blend_radius;
        let gx_start_e = ((center.x - outer - scale).max(0.0) / scale).floor() as usize;
        let gx_end_e = (((center.x + outer + scale) / scale).ceil() as usize).min(hw - 1);
        let gz_start_e = ((center.y - outer - scale).max(0.0) / scale).floor() as usize;
        let gz_end_e = (((center.y + outer + scale) / scale).ceil() as usize).min(hh - 1);

        for gz in gz_start_e..=gz_end_e {
            for gx in gx_start_e..=gx_end_e {
                let idx = gz * hw + gx;
                if is_road_surface[idx] {
                    continue;
                }
                let wx = gx as f32 * scale;
                let wz = gz as f32 * scale;
                let dist = ((wx - center.x).powi(2) + (wz - center.y).powi(2)).sqrt();
                if dist > radius && dist <= outer {
                    let blend = (dist - radius) / blend_radius;
                    let terrain_h = heightmap.get(gx, gz);
                    let blended = center_h + blend * (terrain_h - center_h);
                    heightmap.set(gx, gz, blended);
                    // Mark so edge embankments don't overwrite.
                    is_road_surface[idx] = true;
                }
            }
        }
    }

    // Pass 1b: flatten all road edge surfaces (per-type half-width).
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let half_w = match edge.road_type {
            RoadType::Major => road_config.major_half_width,
            RoadType::Minor => road_config.minor_half_width,
        };

        let (start_pos, end_pos, start_h, end_h, _dir, _length) =
            match cached_edge_params(graph, edge, &node_heights) {
                Some(v) => v,
                None => continue,
            };
        let ep = (start_pos, end_pos, start_h, end_h);

        let (gx_start, gx_end, gz_start, gz_end) =
            edge_grid_bounds(start_pos, end_pos, half_w + scale, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let idx = gz * hw + gx;
                if is_road_surface[idx] {
                    continue;
                }
                let (dist, road_h) = project_cell(gx, gz, scale, &ep);
                if dist <= half_w {
                    heightmap.set(gx, gz, road_h);
                    is_road_surface[idx] = true;
                }
            }
        }
    }

    // Pass 2: edge embankments — for each cell, only the closest road's
    // embankment applies, preventing overwrite artifacts at intersections.
    // Store the blend factor (0..1) directly so we don't need per-cell half-width.
    let mut embankment_blend = vec![f32::MAX; hw * hh];
    let mut embankment_road_h = vec![0.0_f32; hw * hh];

    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let half_w = match edge.road_type {
            RoadType::Major => road_config.major_half_width,
            RoadType::Minor => road_config.minor_half_width,
        };

        let (start_pos, end_pos, start_h, end_h, _dir, _length) =
            match cached_edge_params(graph, edge, &node_heights) {
                Some(v) => v,
                None => continue,
            };
        let ep = (start_pos, end_pos, start_h, end_h);

        let (gx_start, gx_end, gz_start, gz_end) =
            edge_grid_bounds(start_pos, end_pos, half_w + blend_radius, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let idx = gz * hw + gx;
                if is_road_surface[idx] {
                    continue;
                }
                let (dist, road_h) = project_cell(gx, gz, scale, &ep);
                if dist > half_w && dist <= half_w + blend_radius {
                    let blend = (dist - half_w) / blend_radius;
                    if blend < embankment_blend[idx] {
                        embankment_blend[idx] = blend;
                        embankment_road_h[idx] = road_h;
                    }
                }
            }
        }
    }

    // Apply the winning embankment blend for each cell.
    for gz in 0..hh {
        for gx in 0..hw {
            let idx = gz * hw + gx;
            if embankment_blend[idx] < f32::MAX {
                let current_h = heightmap.get(gx, gz);
                let blended = embankment_road_h[idx]
                    + embankment_blend[idx] * (current_h - embankment_road_h[idx]);
                heightmap.set(gx, gz, blended);
            }
        }
    }

    is_road_surface
}

use crate::graph::RoadEdge;

/// Reads node heights from a pre-cached array
/// instead of sampling the (potentially modified) heightmap.
fn cached_edge_params(
    graph: &RoadGraph,
    edge: &RoadEdge,
    node_heights: &[f32],
) -> Option<(glam::Vec2, glam::Vec2, f32, f32, glam::Vec2, f32)> {
    let start_pos = graph.nodes[edge.start as usize].position;
    let end_pos = graph.nodes[edge.end as usize].position;
    let dir = end_pos - start_pos;
    let length = dir.length();
    if length < 1e-6 {
        return None;
    }
    let start_h = node_heights[edge.start as usize];
    let end_h = node_heights[edge.end as usize];
    Some((start_pos, end_pos, start_h, end_h, dir, length))
}

fn edge_grid_bounds(
    start_pos: glam::Vec2,
    end_pos: glam::Vec2,
    expand: f32,
    scale: f32,
    hw: usize,
    hh: usize,
) -> (usize, usize, usize, usize) {
    let min_x = (start_pos.x.min(end_pos.x) - expand).max(0.0);
    let max_x = (start_pos.x.max(end_pos.x) + expand).min((hw - 1) as f32 * scale);
    let min_z = (start_pos.y.min(end_pos.y) - expand).max(0.0);
    let max_z = (start_pos.y.max(end_pos.y) + expand).min((hh - 1) as f32 * scale);
    (
        (min_x / scale).floor() as usize,
        ((max_x / scale).ceil() as usize).min(hw - 1),
        (min_z / scale).floor() as usize,
        ((max_z / scale).ceil() as usize).min(hh - 1),
    )
}

/// Projects grid cell `(gx, gz)` onto a road edge and returns `(dist, road_h)`.
fn project_cell(
    gx: usize,
    gz: usize,
    scale: f32,
    edge: &(glam::Vec2, glam::Vec2, f32, f32),
) -> (f32, f32) {
    let (start_pos, _, start_h, end_h) = *edge;
    let dir = edge.1 - start_pos;
    let length = dir.length();
    let world_x = gx as f32 * scale;
    let world_z = gz as f32 * scale;
    let ap = glam::Vec2::new(world_x - start_pos.x, world_z - start_pos.y);
    let t = (ap.dot(dir) / (length * length)).clamp(0.0, 1.0);
    let proj = start_pos + t * dir;
    let dist = glam::Vec2::new(world_x - proj.x, world_z - proj.y).length();
    let road_h = start_h + t * (end_h - start_h);
    (dist, road_h)
}

/// Computes the grid cell range for a lot's AABB (footprint + blend radius).
fn lot_grid_bounds(
    lot: &BuildingLot,
    half_w: f32,
    half_d: f32,
    blend_radius: f32,
    scale: f32,
    hw: usize,
    hh: usize,
) -> (usize, usize, usize, usize) {
    let max_radius = half_w.hypot(half_d) + blend_radius + scale * 2.0;
    let min_x = (lot.position.x - max_radius).max(0.0);
    let max_x = (lot.position.x + max_radius).min((hw - 1) as f32 * scale);
    let min_z = (lot.position.y - max_radius).max(0.0);
    let max_z = (lot.position.y + max_radius).min((hh - 1) as f32 * scale);
    (
        (min_x / scale).floor() as usize,
        ((max_x / scale).ceil() as usize).min(hw - 1),
        (min_z / scale).floor() as usize,
        ((max_z / scale).ceil() as usize).min(hh - 1),
    )
}

/// Precomputed parameters for a lot's OBB distance queries.
struct LotParams {
    center: glam::Vec2,
    cos: f32,
    sin: f32,
    half_w: f32,
    half_d: f32,
}

/// Returns the distance from a grid cell to the edge of a lot's OBB.
/// Zero means on the edge or inside the footprint, positive means outside.
fn lot_cell_distance(gx: usize, gz: usize, scale: f32, lp: &LotParams) -> f32 {
    let world_x = gx as f32 * scale;
    let world_z = gz as f32 * scale;
    let dx = world_x - lp.center.x;
    let dz = world_z - lp.center.y;
    let local_x = dx * lp.cos + dz * lp.sin;
    let local_z = -dx * lp.sin + dz * lp.cos;
    let dist_x = local_x.abs() - lp.half_w;
    let dist_z = local_z.abs() - lp.half_d;
    dist_x.max(0.0).hypot(dist_z.max(0.0))
}
