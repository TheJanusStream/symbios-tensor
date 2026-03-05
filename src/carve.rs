use symbios_ground::HeightMap;

use crate::graph::RoadGraph;
use crate::lots::BuildingLot;


pub fn carve_lots(lots: &[BuildingLot], heightmap: &mut HeightMap, blend_radius: f32) {
    let scale = heightmap.scale();
    let hw = heightmap.width();
    let hh = heightmap.height();
    if hw == 0 || hh == 0 {
        return;
    }

    for lot in lots {
        // 1. Determine the target foundation height (e.g., sample the center)
        let target_h = heightmap.get_height_at(lot.position.x, lot.position.y);
        
        let half_w = lot.width * 0.5;
        let half_d = lot.depth * 0.5;
        
        // Calculate a generous Axis-Aligned Bounding Box (AABB) to limit our grid search
        let max_radius = half_w.hypot(half_d) + blend_radius + scale * 2.0;
        let min_x = (lot.position.x - max_radius).max(0.0);
        let max_x = (lot.position.x + max_radius).min((hw - 1) as f32 * scale);
        let min_z = (lot.position.y - max_radius).max(0.0);
        let max_z = (lot.position.y + max_radius).min((hh - 1) as f32 * scale);

        let gx_start = (min_x / scale).floor() as usize;
        let gx_end = ((max_x / scale).ceil() as usize).min(hw - 1);
        let gz_start = (min_z / scale).floor() as usize;
        let gz_end = ((max_z / scale).ceil() as usize).min(hh - 1);

        let cos = lot.rotation.cos();
        let sin = lot.rotation.sin();

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let world_x = gx as f32 * scale;
                let world_z = gz as f32 * scale;

                // 2. Transform world coordinates to Lot Local Space
                let dx = world_x - lot.position.x;
                let dz = world_z - lot.position.y;
                
                // Inverse rotation (2D)
                let local_x = dx * cos + dz * sin;
                let local_z = -dx * sin + dz * cos;

                // 3. Calculate distance from the edge of the lot
                let dist_x = local_x.abs() - half_w;
                let dist_z = local_z.abs() - half_d;

                // Distance to the rectangle. 
                // If both are < 0, we are inside. 
                // If > 0, we use length() to get rounded corners on the blend zone.
                let dist_to_edge = dist_x.max(0.0).hypot(dist_z.max(0.0));

                if dist_to_edge <= 0.0 {
                    // Strictly inside the footprint: Flat foundation
                    heightmap.set(gx, gz, target_h);
                } else if dist_to_edge < blend_radius {
                    // Inside the blend zone: Smooth interpolation (e.g., smoothstep)
                    let t = dist_to_edge / blend_radius;
                    // Simple linear blend (or use a smoothstep for softer embankments)
                    let terrain_h = heightmap.get(gx, gz);
                    let blended_h = target_h + t * (terrain_h - target_h);
                    heightmap.set(gx, gz, blended_h);
                }
            }
        }
    }
}

/// Flattens the heightmap along road edges, creating smooth graded surfaces
/// where roads are placed, with blended embankments at the edges.
///
/// Uses a two-pass approach: first all road surfaces are flattened, then
/// embankments are blended. This prevents later embankments from overwriting
/// previously flattened road pavement at intersections.
pub fn carve_roads(graph: &RoadGraph, heightmap: &mut HeightMap, road_width: f32) {
    let scale = heightmap.scale();
    let hw = heightmap.width();
    let hh = heightmap.height();
    if hw == 0 || hh == 0 {
        return;
    }

    let half_w = road_width * 0.5;

    // Track cells that have been set to a road surface height so that
    // the embankment pass does not overwrite them.
    let mut is_road_surface = vec![false; hw * hh];

    // Pass 1: flatten all road surfaces
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let (start_pos, end_pos, start_h, end_h, _dir, _length) =
            match edge_params(graph, edge, heightmap) {
                Some(v) => v,
                None => continue,
            };
        let ep = (start_pos, end_pos, start_h, end_h);

        let (gx_start, gx_end, gz_start, gz_end) =
            edge_grid_bounds(start_pos, end_pos, half_w + scale, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let (dist, road_h) = project_cell(gx, gz, scale, &ep);
                if dist <= half_w {
                    heightmap.set(gx, gz, road_h);
                    is_road_surface[gz * hw + gx] = true;
                }
            }
        }
    }

    // Pass 2: embankments — skip cells already marked as road surface
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let (start_pos, end_pos, start_h, end_h, _dir, _length) =
            match edge_params(graph, edge, heightmap) {
                Some(v) => v,
                None => continue,
            };
        let ep = (start_pos, end_pos, start_h, end_h);

        let (gx_start, gx_end, gz_start, gz_end) =
            edge_grid_bounds(start_pos, end_pos, half_w + scale, scale, hw, hh);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                if is_road_surface[gz * hw + gx] {
                    continue;
                }
                let (dist, road_h) = project_cell(gx, gz, scale, &ep);
                if dist > half_w && dist <= half_w + scale {
                    let blend = (dist - half_w) / scale;
                    let current_h = heightmap.get(gx, gz);
                    let blended = road_h + blend * (current_h - road_h);
                    heightmap.set(gx, gz, blended);
                }
            }
        }
    }
}

use crate::graph::RoadEdge;

fn edge_params(
    graph: &RoadGraph,
    edge: &RoadEdge,
    heightmap: &HeightMap,
) -> Option<(glam::Vec2, glam::Vec2, f32, f32, glam::Vec2, f32)> {
    let start_pos = graph.nodes[edge.start as usize].position;
    let end_pos = graph.nodes[edge.end as usize].position;
    let dir = end_pos - start_pos;
    let length = dir.length();
    if length < 1e-6 {
        return None;
    }
    let start_h = heightmap.get_height_at(start_pos.x, start_pos.y);
    let end_h = heightmap.get_height_at(end_pos.x, end_pos.y);
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
