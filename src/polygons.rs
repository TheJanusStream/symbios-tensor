//! City block extraction from the planar road graph.
//!
//! Uses a minimum-angle (left-most turn) walk to enumerate every bounded
//! interior face of the planar graph as a [`CityBlock`] polygon. The
//! unbounded exterior face is filtered out by its winding direction.

use glam::Vec2;

use crate::graph::{CityBlock, EdgeId, NodeId, RoadGraph};

/// Extracts enclosed city blocks from the planar road graph using the
/// minimum-angle walk algorithm.
///
/// The walk picks the smallest counter-clockwise turn from the reverse
/// incoming direction at each node, which traces faces in **clockwise**
/// (CW) winding order. CW polygons have negative signed area, identifying
/// them as bounded interior blocks. The unbounded exterior face winds CCW
/// and is filtered out by the area sign check.
pub fn extract_blocks(graph: &mut RoadGraph) {
    let mut visited_half_edges: std::collections::HashSet<(NodeId, NodeId)> =
        std::collections::HashSet::new();

    // Collect all directed half-edges from active edges
    let directed: Vec<(NodeId, NodeId, EdgeId)> = graph
        .edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.active)
        .flat_map(|(i, e)| {
            let eid = i as EdgeId;
            [(e.start, e.end, eid), (e.end, e.start, eid)]
        })
        .collect();

    // Build adjacency: for each node, sorted outgoing neighbours by angle
    let node_count = graph.nodes.len();
    let mut adjacency: Vec<Vec<(NodeId, EdgeId)>> = vec![Vec::new(); node_count];

    for &(from, to, eid) in &directed {
        adjacency[from as usize].push((to, eid));
    }

    // Sort each adjacency list by the angle of the outgoing direction
    for (nid, neighbours) in adjacency.iter_mut().enumerate() {
        let origin = graph.nodes[nid].position;
        neighbours.sort_by(|a, b| {
            let da = graph.nodes[a.0 as usize].position - origin;
            let db = graph.nodes[b.0 as usize].position - origin;
            let angle_a = da.y.atan2(da.x);
            let angle_b = db.y.atan2(db.x);
            angle_a.partial_cmp(&angle_b).unwrap()
        });
    }

    // Walk minimal cycles using left-most turn
    for &(start_from, start_to, _) in &directed {
        if visited_half_edges.contains(&(start_from, start_to)) {
            continue;
        }

        let mut cycle: Vec<NodeId> = vec![start_from];
        let mut prev = start_from;
        let mut curr = start_to;

        let max_cycle_len = node_count + 1;
        let mut valid = true;

        loop {
            if visited_half_edges.contains(&(prev, curr)) {
                valid = false;
                break;
            }
            visited_half_edges.insert((prev, curr));
            cycle.push(curr);

            if curr == start_from {
                break;
            }

            if cycle.len() > max_cycle_len {
                valid = false;
                break;
            }

            // Find next: the edge with the smallest CCW rotation from the reverse
            // incoming direction. Because incoming_dir points backwards, this
            // selects the sharpest right turn — tracing CW (interior) faces.
            let neighbours = &adjacency[curr as usize];
            if neighbours.is_empty() {
                valid = false;
                break;
            }

            let incoming_dir =
                graph.nodes[prev as usize].position - graph.nodes[curr as usize].position;
            let incoming_angle = incoming_dir.y.atan2(incoming_dir.x);

            let next = pick_next_face_edge(neighbours, graph, curr, prev, incoming_angle);

            match next {
                Some(n) => {
                    prev = curr;
                    curr = n;
                }
                None => {
                    valid = false;
                    break;
                }
            }
        }

        if valid && cycle.len() >= 4 {
            // Remove the duplicate closing node (last == first)
            cycle.pop();

            // Reject the outer (unbounded) face: if the polygon winds clockwise
            // (negative signed area) it is an interior block.
            if signed_area(&cycle, graph) < 0.0 {
                graph.blocks.push(CityBlock { perimeter: cycle });
            }
        }
    }
}

/// Picks the neighbour whose outgoing angle is the smallest positive
/// counter-clockwise rotation from `incoming_angle` (the reverse of our
/// travel direction). This selects the sharpest right turn relative to
/// our forward direction, tracing CW (interior) faces of the planar graph.
///
/// U-turns are detected topologically (`to == prev`) rather than via fragile
/// angular tolerances, so near-parallel edges from imperfect snapping cannot
/// trick the algorithm into reversing.
fn pick_next_face_edge(
    neighbours: &[(NodeId, EdgeId)],
    graph: &RoadGraph,
    current: NodeId,
    prev: NodeId,
    incoming_angle: f32,
) -> Option<NodeId> {
    let origin = graph.nodes[current as usize].position;
    let mut best: Option<NodeId> = None;
    let mut best_delta = f32::MAX;

    let has_alternatives = neighbours.len() > 1;
    for &(to, _) in neighbours {
        // Topological U-turn check: skip the node we came from, unless this
        // is a dead-end (degree-1 node) where a U-turn is required to walk
        // back out of an antenna edge during face traversal.
        if to == prev && has_alternatives {
            continue;
        }

        let d = graph.nodes[to as usize].position - origin;
        let out_angle = d.y.atan2(d.x);

        // Counter-clockwise delta from incoming_angle, in (0, TAU]
        let mut delta = (out_angle - incoming_angle).rem_euclid(std::f32::consts::TAU);
        if delta < 1e-5 {
            delta = std::f32::consts::TAU;
        }
        if delta < best_delta {
            best_delta = delta;
            best = Some(to);
        }
    }

    best
}

/// Signed area of the polygon via the shoelace formula (positive = CCW, negative = CW).
pub(crate) fn signed_area(nodes: &[NodeId], graph: &RoadGraph) -> f32 {
    let n = nodes.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f32;
    for i in 0..n {
        let a = graph.nodes[nodes[i] as usize].position;
        let b = graph.nodes[nodes[(i + 1) % n] as usize].position;
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

/// Returns the centroid of a [`CityBlock`] polygon using the shoelace-based
/// geometric centroid formula (robust for irregular polygons).
pub fn block_centroid(block: &CityBlock, graph: &RoadGraph) -> Vec2 {
    let n = block.perimeter.len();
    if n == 0 {
        return Vec2::ZERO;
    }
    if n < 3 {
        let sum: Vec2 = block
            .perimeter
            .iter()
            .map(|&nid| graph.nodes[nid as usize].position)
            .sum();
        return sum / n as f32;
    }
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let mut signed_area_2 = 0.0_f32;
    for i in 0..n {
        let a = graph.nodes[block.perimeter[i] as usize].position;
        let b = graph.nodes[block.perimeter[(i + 1) % n] as usize].position;
        let cross = a.x * b.y - b.x * a.y;
        cx += (a.x + b.x) * cross;
        cy += (a.y + b.y) * cross;
        signed_area_2 += cross;
    }
    if signed_area_2.abs() < 1e-8 {
        let sum: Vec2 = block
            .perimeter
            .iter()
            .map(|&nid| graph.nodes[nid as usize].position)
            .sum();
        return sum / n as f32;
    }
    let inv = 1.0 / (3.0 * signed_area_2);
    Vec2::new(cx * inv, cy * inv)
}
