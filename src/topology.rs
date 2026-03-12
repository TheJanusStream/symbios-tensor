//! Shared topology helpers for the road graph.
//!
//! Extracts chain (degree-2 path) information from a [`RoadGraph`] so that
//! both the rationalizer and the 3D mesher can operate on the same topology.

use crate::graph::{EdgeId, NodeId, RoadGraph, RoadType};

/// A chain is a maximal path of degree-2 nodes between two junction/dead-end
/// endpoints, all sharing the same [`RoadType`].
pub struct Chain {
    /// Ordered node IDs from one junction to another (inclusive at both ends).
    pub nodes: Vec<NodeId>,
    /// The road classification shared by every edge in the chain.
    pub road_type: RoadType,
    /// The edge IDs that make up this chain (in order).
    pub edges: Vec<EdgeId>,
}

/// Computes the active degree (number of active incident edges) for every node.
pub fn compute_active_degrees(graph: &RoadGraph) -> Vec<u32> {
    let mut degrees = vec![0u32; graph.nodes.len()];
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        degrees[edge.start as usize] += 1;
        degrees[edge.end as usize] += 1;
    }
    degrees
}

/// Extracts all maximal chains of degree-2 nodes from the graph.
///
/// Each chain runs from one non-degree-2 node to another (or forms a cycle),
/// following edges of a single [`RoadType`]. The returned chains include
/// the edge IDs so callers can deactivate them during graph mutation.
pub fn extract_chains(graph: &RoadGraph, degrees: &[u32]) -> Vec<Chain> {
    let mut visited_edges = vec![false; graph.edges.len()];
    let mut chains = Vec::new();

    for (eid, edge) in graph.edges.iter().enumerate() {
        if !edge.active || visited_edges[eid] {
            continue;
        }

        let road_type = edge.road_type;
        let mut chain_nodes = Vec::new();
        let mut chain_edges = Vec::new();

        // Mark seed edge visited before walking so pure cycles can't loop back.
        visited_edges[eid] = true;

        // Walk backwards from edge.start.
        let (head_nodes, head_edges) = walk_chain(
            graph,
            degrees,
            edge.start,
            eid as u32,
            &mut visited_edges,
            road_type,
        );
        head_nodes.into_iter().rev().for_each(|n| chain_nodes.push(n));
        head_edges.into_iter().rev().for_each(|e| chain_edges.push(e));

        // Seed edge endpoints.
        chain_nodes.push(edge.start);
        chain_edges.push(eid as EdgeId);
        chain_nodes.push(edge.end);

        // Walk forward from edge.end.
        let (tail_nodes, tail_edges) = walk_chain(
            graph,
            degrees,
            edge.end,
            eid as u32,
            &mut visited_edges,
            road_type,
        );
        chain_nodes.extend(tail_nodes);
        chain_edges.extend(tail_edges);

        // Deduplicate consecutive nodes.
        chain_nodes.dedup();

        if chain_nodes.len() >= 2 {
            chains.push(Chain {
                nodes: chain_nodes,
                road_type,
                edges: chain_edges,
            });
        }
    }

    chains
}

/// Walks from `start_node` through degree-2 nodes of matching road type.
/// Returns `(nodes_visited, edges_traversed)` — `start_node` is NOT included.
fn walk_chain(
    graph: &RoadGraph,
    degrees: &[u32],
    start_node: NodeId,
    from_edge: u32,
    visited_edges: &mut [bool],
    road_type: RoadType,
) -> (Vec<NodeId>, Vec<EdgeId>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut current = start_node;
    let mut prev_edge = from_edge;

    if degrees[current as usize] != 2 {
        return (nodes, edges);
    }

    loop {
        let node = &graph.nodes[current as usize];
        let mut next_edge = None;
        for &eid in &node.edges {
            if eid == prev_edge {
                continue;
            }
            let e = &graph.edges[eid as usize];
            if !e.active || visited_edges[eid as usize] {
                continue;
            }
            if e.road_type != road_type {
                continue;
            }
            next_edge = Some(eid);
            break;
        }

        let Some(ne) = next_edge else { break };
        visited_edges[ne as usize] = true;
        edges.push(ne);

        let next_node = graph.opposite(ne, current);
        nodes.push(next_node);

        if degrees[next_node as usize] != 2 {
            break;
        }
        prev_edge = ne;
        current = next_node;
    }

    (nodes, edges)
}
