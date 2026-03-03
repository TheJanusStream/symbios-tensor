use symbios_ground::HeightMap;

use crate::graph::RoadGraph;

/// Flattens the heightmap along road edges, creating smooth graded surfaces
/// where roads are placed, with blended embankments at the edges.
pub fn carve_roads(graph: &RoadGraph, heightmap: &mut HeightMap, road_width: f32) {
    let scale = heightmap.scale();
    let hw = heightmap.width();
    let hh = heightmap.height();

    for edge in &graph.edges {
        if !edge.active {
            continue;
        }

        let start_pos = graph.nodes[edge.start as usize].position;
        let end_pos = graph.nodes[edge.end as usize].position;

        let start_h = heightmap.get_height_at(start_pos.x, start_pos.y);
        let end_h = heightmap.get_height_at(end_pos.x, end_pos.y);

        let dir = end_pos - start_pos;
        let length = dir.length();
        if length < 1e-6 {
            continue;
        }

        // Compute the bounding box of this edge in grid coordinates, expanded by road_width
        let half_w = road_width * 0.5;
        let min_x = (start_pos.x.min(end_pos.x) - half_w - scale).max(0.0);
        let max_x = (start_pos.x.max(end_pos.x) + half_w + scale).min((hw - 1) as f32 * scale);
        let min_z = (start_pos.y.min(end_pos.y) - half_w - scale).max(0.0);
        let max_z = (start_pos.y.max(end_pos.y) + half_w + scale).min((hh - 1) as f32 * scale);

        let gx_start = (min_x / scale).floor() as usize;
        let gx_end = ((max_x / scale).ceil() as usize).min(hw - 1);
        let gz_start = (min_z / scale).floor() as usize;
        let gz_end = ((max_z / scale).ceil() as usize).min(hh - 1);

        for gz in gz_start..=gz_end {
            for gx in gx_start..=gx_end {
                let world_x = gx as f32 * scale;
                let world_z = gz as f32 * scale;

                // Project this grid cell onto the road segment
                let ap = glam::Vec2::new(world_x - start_pos.x, world_z - start_pos.y);
                let t = (ap.dot(dir) / (length * length)).clamp(0.0, 1.0);
                let proj = start_pos + t * dir;

                let dist = glam::Vec2::new(world_x - proj.x, world_z - proj.y).length();

                if dist > half_w + scale {
                    continue;
                }

                // Interpolate the target road height along the segment
                let road_h = start_h + t * (end_h - start_h);

                if dist <= half_w {
                    // Under the road surface: force to road height
                    heightmap.set(gx, gz, road_h);
                } else {
                    // Embankment blend zone: smoothly transition between road and terrain
                    let blend = (dist - half_w) / scale;
                    let terrain_h = heightmap.get(gx, gz);
                    let blended = road_h + blend * (terrain_h - road_h);
                    heightmap.set(gx, gz, blended);
                }
            }
        }
    }
}
