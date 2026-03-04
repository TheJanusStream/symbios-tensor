use glam::Vec2;

use crate::graph::{CityBlock, EdgeId, NodeId, RoadGraph};

/// Extracts enclosed city blocks from the planar road graph using the
/// "left-most turn" (minimum angle) walk algorithm.
///
/// Each block is a minimal cycle — the smallest closed polygon formed by
/// road edges. These become the building footprint candidates for shape grammars.
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

            // Find next: the edge that makes the sharpest left turn from (prev → curr)
            let neighbours = &adjacency[curr as usize];
            if neighbours.is_empty() {
                valid = false;
                break;
            }

            let incoming_dir =
                graph.nodes[prev as usize].position - graph.nodes[curr as usize].position;
            let incoming_angle = incoming_dir.y.atan2(incoming_dir.x);

            // Find the neighbour whose outgoing angle is the smallest counter-clockwise
            // rotation from the incoming angle. This is the "left-most turn".
            let next = pick_left_turn(neighbours, graph, curr, incoming_angle);

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

/// Picks the neighbour that represents the smallest counter-clockwise turn
/// from `incoming_angle` (the direction we arrived from).
fn pick_left_turn(
    neighbours: &[(NodeId, EdgeId)],
    graph: &RoadGraph,
    current: NodeId,
    incoming_angle: f32,
) -> Option<NodeId> {
    let origin = graph.nodes[current as usize].position;
    let mut best: Option<NodeId> = None;
    let mut best_delta = f32::MAX;

    for &(to, _) in neighbours {
        let d = graph.nodes[to as usize].position - origin;
        let out_angle = d.y.atan2(d.x);

        // Counter-clockwise delta from incoming_angle
        let mut delta = out_angle - incoming_angle;
        if delta <= 0.0 {
            delta += std::f32::consts::TAU;
        }
        // Skip the exact reverse (delta ≈ π means going straight back)
        // We want the smallest positive rotation that isn't a U-turn
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

/// Returns the centroid of a [`CityBlock`] polygon.
pub fn block_centroid(block: &CityBlock, graph: &RoadGraph) -> Vec2 {
    if block.perimeter.is_empty() {
        return Vec2::ZERO;
    }
    let sum: Vec2 = block
        .perimeter
        .iter()
        .map(|&nid| graph.nodes[nid as usize].position)
        .sum();
    sum / block.perimeter.len() as f32
}
