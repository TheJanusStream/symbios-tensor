use crate::geometry::closest_point_on_segment;
use crate::graph::{EdgeId, NodeId, RoadGraph, RoadType};
use crate::lots::BuildingLot;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f32,
    node: NodeId,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn prune_unused_roads(graph: &mut RoadGraph, lots: &[BuildingLot]) {
    if lots.is_empty() {
        for edge in &mut graph.edges {
            edge.active = false;
        }
        return;
    }

    // --- PHASE 0: Component Analysis ---
    // Identify connected components to prevent houses from snapping to
    // tiny disconnected "garbage" artifacts left by the tensor tracer.
    let mut comp_ids = vec![0; graph.nodes.len()];
    let mut comp_sizes = HashMap::new();
    let mut current_comp = 1;

    for i in 0..graph.nodes.len() {
        if comp_ids[i] == 0 {
            let mut stack = vec![i as NodeId];
            let mut size = 0;
            while let Some(node) = stack.pop() {
                if comp_ids[node as usize] == 0 {
                    comp_ids[node as usize] = current_comp;
                    size += 1;
                    for &eid in &graph.nodes[node as usize].edges {
                        let e = &graph.edges[eid as usize];
                        if e.active {
                            let opp = graph.opposite(eid, node);
                            if comp_ids[opp as usize] == 0 {
                                stack.push(opp);
                            }
                        }
                    }
                }
            }
            comp_sizes.insert(current_comp, size);
            current_comp += 1;
        }
    }

    let max_comp_size = comp_sizes.values().copied().max().unwrap_or(0);

    // --- PHASE 1: Identify Lot Access ---
    let mut essential_edges = HashSet::new();
    let mut essential_nodes = HashSet::new();

    for lot in lots {
        let mut best_edge = None;
        let mut best_score = f32::MAX;

        for (i, edge) in graph.edges.iter().enumerate() {
            if !edge.active {
                continue;
            }
            let a = graph.nodes[edge.start as usize].position;
            let b = graph.nodes[edge.end as usize].position;
            let proj = closest_point_on_segment(lot.position, a, b);
            let dist = lot.position.distance(proj);

            let comp_id = comp_ids[edge.start as usize];
            let comp_size = comp_sizes[&comp_id];

            // Ignore isolated floating lines if a real network exists
            let penalty = if comp_size < 10 && max_comp_size > 20 {
                1000.0
            } else {
                0.0
            };

            let score = dist + penalty;

            if score < best_score {
                best_score = score;
                best_edge = Some(i as EdgeId);
            }
        }

        if let Some(eid) = best_edge {
            essential_edges.insert(eid);
            let edge = &graph.edges[eid as usize];
            essential_nodes.insert(edge.start);
            essential_nodes.insert(edge.end);
        }
    }

    // --- PHASE 2: The Steiner Tree ---
    let mut keep_edges = essential_edges.clone();
    let mut connected_nodes = HashSet::new();

    // CRITICAL FIX: Track exactly which essential nodes we still need to reach
    let mut unreached_essential = essential_nodes.clone();

    // Seed the network
    if let Some(&start_node) = essential_nodes.iter().next() {
        connected_nodes.insert(start_node);
        unreached_essential.remove(&start_node);
    }

    // Loop until EVERY house is connected
    while !unreached_essential.is_empty() {
        let mut heap = BinaryHeap::new();
        let mut distances = vec![f32::MAX; graph.nodes.len()];
        let mut came_from: Vec<Option<(NodeId, EdgeId)>> = vec![None; graph.nodes.len()];

        for &node in &connected_nodes {
            distances[node as usize] = 0.0;
            heap.push(State { cost: 0.0, node });
        }

        let mut found_target = None;

        while let Some(State { cost, node }) = heap.pop() {
            // Target is found if it's an essential node we haven't reached yet
            if unreached_essential.contains(&node) {
                found_target = Some(node);
                break;
            }

            if cost > distances[node as usize] {
                continue;
            }

            for &edge_id in &graph.nodes[node as usize].edges {
                let edge = &graph.edges[edge_id as usize];
                if !edge.active {
                    continue;
                }

                let next_node = graph.opposite(edge_id, node);
                let dist = graph.nodes[node as usize]
                    .position
                    .distance(graph.nodes[next_node as usize].position);

                let weight = if edge.road_type == RoadType::Major {
                    dist * 1.0
                } else {
                    dist * 5.0
                };

                let next_cost = cost + weight;

                if next_cost < distances[next_node as usize] {
                    distances[next_node as usize] = next_cost;
                    came_from[next_node as usize] = Some((node, edge_id));
                    heap.push(State {
                        cost: next_cost,
                        node: next_node,
                    });
                }
            }
        }

        if let Some(mut curr) = found_target {
            // Trace back and preserve the path
            while let Some((prev, edge_id)) = came_from[curr as usize] {
                keep_edges.insert(edge_id);
                connected_nodes.insert(curr);
                unreached_essential.remove(&curr);
                curr = prev;
            }
            connected_nodes.insert(curr);
            unreached_essential.remove(&curr);
        } else {
            // --- PHASE 3: The Bridge Builder ---
            // Dijkstra failed. We have isolated islands. Build a physical bridge
            // to the closest unreached house.
            let mut best_bridge = None;
            let mut min_bridge_dist = f32::MAX;

            for &a in &connected_nodes {
                let pos_a = graph.nodes[a as usize].position;
                for &b in &unreached_essential {
                    let pos_b = graph.nodes[b as usize].position;
                    let d = pos_a.distance_squared(pos_b);
                    if d < min_bridge_dist {
                        min_bridge_dist = d;
                        best_bridge = Some((a, b));
                    }
                }
            }

            if let Some((a, b)) = best_bridge {
                // Spawn a new major road bridging the gap
                let new_edge_id = graph.add_edge(a, b, RoadType::Major);
                keep_edges.insert(new_edge_id);
                connected_nodes.insert(b);
                unreached_essential.remove(&b);
            } else {
                break; // Failsafe
            }
        }
    }

    // --- PHASE 4: Pruning ---
    for (i, edge) in graph.edges.iter_mut().enumerate() {
        if !keep_edges.contains(&(i as EdgeId)) {
            edge.active = false;
        }
    }
}
