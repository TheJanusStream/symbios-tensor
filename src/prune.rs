//! Road pruning via Steiner tree construction.
//!
//! After lots are placed, many road segments may not serve any building.
//! [`prune_unused_roads`] identifies the minimal connected sub-network that
//! reaches every lot's frontage edge using Dijkstra shortest paths (preferring
//! major roads), then deactivates all edges outside that sub-network.
//! Disconnected components are handled independently so lots on isolated
//! islands still retain their access roads.

use crate::geometry::closest_point_on_segment;
use crate::graph::{EdgeId, NodeId, RoadGraph, RoadType};
use crate::lots::BuildingLot;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

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

/// Deactivates road edges that are not needed to reach any building lot.
///
/// For each lot the closest active edge (by frontage center) is marked
/// essential, then a Dijkstra-based Steiner tree connects all essential
/// nodes through the cheapest path (major roads are 5× cheaper than minor
/// roads). Edges outside the resulting tree are deactivated. If no lots
/// exist, all edges are deactivated.
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
    let mut essential_nodes = BTreeSet::new();

    for lot in lots {
        let mut best_edge = None;
        let mut best_score = f32::MAX;

        for (i, edge) in graph.edges.iter().enumerate() {
            if !edge.active {
                continue;
            }
            let a = graph.nodes[edge.start as usize].position;
            let b = graph.nodes[edge.end as usize].position;
            let proj = closest_point_on_segment(lot.frontage_center, a, b);
            let dist = lot.frontage_center.distance(proj);

            let comp_id = comp_ids[edge.start as usize];
            let comp_size = comp_sizes[&comp_id];

            // Penalise only truly degenerate fragments (≤2 nodes can't
            // form a usable road loop). Small but valid island communities
            // (e.g. 5-9 nodes) should keep their lots.
            let penalty = if comp_size < 3 && max_comp_size > 20 {
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
    let mut connected_nodes = BTreeSet::new();

    // CRITICAL FIX: Track exactly which essential nodes we still need to reach
    let mut unreached_essential = essential_nodes.clone();

    // Seed the network
    if let Some(&start_node) = essential_nodes.iter().next() {
        connected_nodes.insert(start_node);
        unreached_essential.remove(&start_node);
    }

    // Pre-allocate Dijkstra buffers with generation counters for O(1) resets.
    // A distance/came_from entry is only valid when its generation matches
    // `current_gen`, avoiding costly O(V) fills each iteration.
    let num_nodes = graph.nodes.len();
    let mut heap = BinaryHeap::new();
    let mut dist_gen = vec![(0_u32, f32::MAX); num_nodes];
    let mut from_gen: Vec<(u32, Option<(NodeId, EdgeId)>)> = vec![(0, None); num_nodes];
    let mut current_gen = 0_u32;

    // Loop until EVERY house is connected
    while !unreached_essential.is_empty() {
        heap.clear();
        current_gen += 1;

        for &node in &connected_nodes {
            dist_gen[node as usize] = (current_gen, 0.0);
            heap.push(State { cost: 0.0, node });
        }

        let mut found_target = None;

        while let Some(State { cost, node }) = heap.pop() {
            // Target is found if it's an essential node we haven't reached yet
            if unreached_essential.contains(&node) {
                found_target = Some(node);
                break;
            }

            let cur_dist = dist_gen[node as usize];
            if cur_dist.0 != current_gen || cost > cur_dist.1 {
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

                let nd = &mut dist_gen[next_node as usize];
                if nd.0 != current_gen || next_cost < nd.1 {
                    *nd = (current_gen, next_cost);
                    from_gen[next_node as usize] = (current_gen, Some((node, edge_id)));
                    heap.push(State {
                        cost: next_cost,
                        node: next_node,
                    });
                }
            }
        }

        if let Some(mut curr) = found_target {
            // Trace back and preserve the path
            while from_gen[curr as usize].0 == current_gen {
                let Some((prev, edge_id)) = from_gen[curr as usize].1 else {
                    break;
                };
                keep_edges.insert(edge_id);
                connected_nodes.insert(curr);
                unreached_essential.remove(&curr);
                curr = prev;
            }
            connected_nodes.insert(curr);
            unreached_essential.remove(&curr);
        } else {
            // Dijkstra failed — unreached nodes are on a disconnected
            // island. Instead of creating planarity-violating bridge
            // edges, run a local Steiner tree within the island: seed
            // from the unreached essential nodes on this component and
            // connect them through existing active edges.
            let target_comp = comp_ids[*unreached_essential.iter().next().unwrap() as usize];
            let island_seed: Vec<NodeId> = unreached_essential
                .iter()
                .copied()
                .filter(|&n| comp_ids[n as usize] == target_comp)
                .collect();

            if island_seed.is_empty() {
                break;
            }

            // BFS/Dijkstra within the island to connect its essential nodes
            let island_comp = comp_ids[island_seed[0] as usize];
            let mut island_connected: BTreeSet<NodeId> = BTreeSet::new();
            island_connected.insert(island_seed[0]);
            let mut island_remaining: BTreeSet<NodeId> = island_seed.iter().copied().collect();
            island_remaining.remove(&island_seed[0]);

            while !island_remaining.is_empty() {
                heap.clear();
                current_gen += 1;

                for &node in &island_connected {
                    dist_gen[node as usize] = (current_gen, 0.0);
                    heap.push(State { cost: 0.0, node });
                }

                let mut found = None;
                while let Some(State { cost, node }) = heap.pop() {
                    if island_remaining.contains(&node) {
                        found = Some(node);
                        break;
                    }
                    let cur_dist = dist_gen[node as usize];
                    if cur_dist.0 != current_gen || cost > cur_dist.1 {
                        continue;
                    }
                    for &edge_id in &graph.nodes[node as usize].edges {
                        let edge = &graph.edges[edge_id as usize];
                        if !edge.active {
                            continue;
                        }
                        // Stay within this component
                        let next = graph.opposite(edge_id, node);
                        if comp_ids[next as usize] != island_comp {
                            continue;
                        }
                        let d = graph.nodes[node as usize]
                            .position
                            .distance(graph.nodes[next as usize].position);
                        let next_cost = cost + d;
                        let nd = &mut dist_gen[next as usize];
                        if nd.0 != current_gen || next_cost < nd.1 {
                            *nd = (current_gen, next_cost);
                            from_gen[next as usize] = (current_gen, Some((node, edge_id)));
                            heap.push(State {
                                cost: next_cost,
                                node: next,
                            });
                        }
                    }
                }

                if let Some(mut curr) = found {
                    while from_gen[curr as usize].0 == current_gen {
                        let Some((prev, edge_id)) = from_gen[curr as usize].1 else {
                            break;
                        };
                        keep_edges.insert(edge_id);
                        island_connected.insert(curr);
                        island_remaining.remove(&curr);
                        curr = prev;
                    }
                    island_connected.insert(curr);
                    island_remaining.remove(&curr);
                } else {
                    // Truly unreachable within this component —
                    // keep whatever essential edges they have and move on
                    break;
                }
            }

            // Mark all island essential nodes as handled so the
            // outer loop doesn't retry them
            for n in &island_seed {
                connected_nodes.insert(*n);
                unreached_essential.remove(n);
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
