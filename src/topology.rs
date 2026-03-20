//! Shared topology helpers for the road graph.
//!
//! Extracts chain (degree-2 path) information from a [`RoadGraph`] so that
//! both the rationalizer and the 3D mesher can operate on the same topology.
//!
//! Also provides **artery extraction** — tracing continuous paths *through*
//! intersections by picking the most forward-aligned exit edge of matching
//! road type. This allows the rationalizer to straighten entire avenues
//! globally rather than treating each intersection as an immovable anchor.

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
        head_nodes
            .into_iter()
            .rev()
            .for_each(|n| chain_nodes.push(n));
        head_edges
            .into_iter()
            .rev()
            .for_each(|e| chain_edges.push(e));

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

/// Extracts all maximal chains of degree-2 nodes, **ignoring road type**.
///
/// Unlike [`extract_chains`], this walks through degree-2 nodes regardless of
/// whether adjacent edges differ in [`RoadType`]. Each returned chain contains
/// the edges as they appear in the graph — the caller can inspect or overwrite
/// their types. Used by the road-type unification pass.
pub fn extract_chains_any_type(graph: &RoadGraph, degrees: &[u32]) -> Vec<Chain> {
    let mut visited_edges = vec![false; graph.edges.len()];
    let mut chains = Vec::new();

    for (eid, edge) in graph.edges.iter().enumerate() {
        if !edge.active || visited_edges[eid] {
            continue;
        }

        visited_edges[eid] = true;

        let mut chain_nodes = Vec::new();
        let mut chain_edges = Vec::new();

        // Walk backwards from edge.start (type-agnostic).
        let (head_nodes, head_edges) =
            walk_chain_any_type(graph, degrees, edge.start, eid as u32, &mut visited_edges);
        head_nodes
            .into_iter()
            .rev()
            .for_each(|n| chain_nodes.push(n));
        head_edges
            .into_iter()
            .rev()
            .for_each(|e| chain_edges.push(e));

        // Seed edge endpoints.
        chain_nodes.push(edge.start);
        chain_edges.push(eid as EdgeId);
        chain_nodes.push(edge.end);

        // Walk forward from edge.end.
        let (tail_nodes, tail_edges) =
            walk_chain_any_type(graph, degrees, edge.end, eid as u32, &mut visited_edges);
        chain_nodes.extend(tail_nodes);
        chain_edges.extend(tail_edges);

        chain_nodes.dedup();

        if chain_nodes.len() >= 2 {
            // Use the first edge's road type as the nominal type; the caller
            // will overwrite it after majority-vote anyway.
            chains.push(Chain {
                nodes: chain_nodes,
                road_type: edge.road_type,
                edges: chain_edges,
            });
        }
    }

    chains
}

/// Walks from `start_node` through degree-2 nodes regardless of road type.
fn walk_chain_any_type(
    graph: &RoadGraph,
    degrees: &[u32],
    start_node: NodeId,
    from_edge: u32,
    visited_edges: &mut [bool],
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
            // No road_type check — walk through any type.
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

// ---------------------------------------------------------------------------
// Artery extraction (traces through intersections by alignment)
// ---------------------------------------------------------------------------

/// An artery is a continuous path of same-type edges that passes *through*
/// intersections by choosing the most forward-aligned exit. This allows
/// global straightening of avenues that would otherwise be chopped into
/// many tiny chains at every T-junction.
pub struct Artery {
    /// Ordered node IDs along the artery (inclusive at both ends).
    pub nodes: Vec<NodeId>,
    /// The road classification shared by every edge in the artery.
    pub road_type: RoadType,
    /// The edge IDs that make up this artery (in order).
    pub edges: Vec<EdgeId>,
}

/// Cosine threshold below which we refuse to continue through a junction.
/// cos(60°) ≈ 0.5 — anything sharper than a 60° turn is not "the same road".
const ARTERY_MIN_ALIGNMENT: f32 = 0.5;

/// Extracts arteries of the given `road_type` from the graph.
///
/// Each artery traces through intersections as long as there is a same-type
/// exit edge whose direction aligns within `ARTERY_MIN_ALIGNMENT` of the
/// incoming forward vector. Dead-ends and sharp turns terminate the artery.
pub fn extract_arteries(graph: &RoadGraph, degrees: &[u32], road_type: RoadType) -> Vec<Artery> {
    let mut visited_edges = vec![false; graph.edges.len()];
    let mut arteries = Vec::new();

    for (eid, edge) in graph.edges.iter().enumerate() {
        if !edge.active || visited_edges[eid] || edge.road_type != road_type {
            continue;
        }

        visited_edges[eid] = true;

        // Track visited nodes to prevent figure-8 self-crossings.
        // If the tensor field causes a road to loop back through a
        // previously visited node, continuing would produce a
        // self-intersecting polyline that breaks graph planarity when
        // replaced by rationalize_artery's smoothed geometry.
        let mut visited_nodes = vec![false; graph.nodes.len()];
        visited_nodes[edge.start as usize] = true;
        visited_nodes[edge.end as usize] = true;

        // Walk backwards from edge.start.
        let (mut head_nodes, mut head_edges) = walk_artery(
            graph,
            degrees,
            edge.start,
            edge.end, // "came from" direction
            eid as EdgeId,
            &mut visited_edges,
            &mut visited_nodes,
            road_type,
        );
        head_nodes.reverse();
        head_edges.reverse();

        // Seed edge.
        head_nodes.push(edge.start);
        head_edges.push(eid as EdgeId);
        head_nodes.push(edge.end);

        // Walk forwards from edge.end.
        let (tail_nodes, tail_edges) = walk_artery(
            graph,
            degrees,
            edge.end,
            edge.start, // "came from" direction
            eid as EdgeId,
            &mut visited_edges,
            &mut visited_nodes,
            road_type,
        );
        head_nodes.extend(tail_nodes);
        head_edges.extend(tail_edges);

        head_nodes.dedup();

        if head_nodes.len() >= 2 {
            arteries.push(Artery {
                nodes: head_nodes,
                road_type,
                edges: head_edges,
            });
        }
    }

    arteries
}

/// Walks from `current_node` through junctions by picking the most
/// forward-aligned exit edge of matching road type.
///
/// `came_from_node` is the node we arrived from (for computing forward direction).
/// Returns `(nodes_visited, edges_traversed)` — `current_node` is NOT included.
#[allow(clippy::too_many_arguments)]
fn walk_artery(
    graph: &RoadGraph,
    degrees: &[u32],
    current_node: NodeId,
    came_from_node: NodeId,
    from_edge: EdgeId,
    visited_edges: &mut [bool],
    visited_nodes: &mut [bool],
    road_type: RoadType,
) -> (Vec<NodeId>, Vec<EdgeId>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut current = current_node;
    let mut prev_node = came_from_node;
    let mut prev_edge = from_edge;

    loop {
        let deg = degrees[current as usize];

        // Dead-end — stop.
        if deg <= 1 {
            break;
        }

        // Degree-2: follow the only other edge (same as chain walking).
        if deg == 2 {
            let node = &graph.nodes[current as usize];
            let mut next_edge = None;
            for &eid in &node.edges {
                if eid == prev_edge {
                    continue;
                }
                let e = &graph.edges[eid as usize];
                if !e.active || visited_edges[eid as usize] || e.road_type != road_type {
                    continue;
                }
                next_edge = Some(eid);
                break;
            }
            let Some(ne) = next_edge else { break };
            let next_node = graph.opposite(ne, current);
            // Prevent figure-8: stop if we would revisit a node.
            if visited_nodes[next_node as usize] {
                break;
            }
            visited_edges[ne as usize] = true;
            visited_nodes[next_node as usize] = true;
            edges.push(ne);
            nodes.push(next_node);
            prev_node = current;
            prev_edge = ne;
            current = next_node;
            continue;
        }

        // Junction (degree >= 3): pick the most forward-aligned same-type exit.
        let forward = (graph.node_pos(current) - graph.node_pos(prev_node)).normalize_or_zero();
        if forward.length_squared() < 1e-12 {
            break;
        }

        let node = &graph.nodes[current as usize];
        let mut best_edge: Option<EdgeId> = None;
        let mut best_cos = ARTERY_MIN_ALIGNMENT; // minimum threshold

        for &eid in &node.edges {
            if eid == prev_edge {
                continue;
            }
            let e = &graph.edges[eid as usize];
            if !e.active || visited_edges[eid as usize] || e.road_type != road_type {
                continue;
            }
            let neighbor = graph.opposite(eid, current);
            // Skip edges leading to already-visited nodes (figure-8 prevention).
            if visited_nodes[neighbor as usize] {
                continue;
            }
            let dir = (graph.node_pos(neighbor) - graph.node_pos(current)).normalize_or_zero();
            let cos = forward.dot(dir);
            if cos > best_cos {
                best_cos = cos;
                best_edge = Some(eid);
            }
        }

        let Some(ne) = best_edge else { break };
        let next_node = graph.opposite(ne, current);
        visited_edges[ne as usize] = true;
        visited_nodes[next_node as usize] = true;
        edges.push(ne);
        nodes.push(next_node);
        prev_node = current;
        prev_edge = ne;
        current = next_node;
    }

    (nodes, edges)
}
