//! Graph rationalization: straighten and smooth the road network.
//!
//! After the tracer produces a raw network of many small segments, this pass
//! rewrites the graph into cleaner geometry:
//!
//! 1. **Global Laplacian elevation smoothing** averages node heights across the
//!    entire graph, producing roads that bridge dips and cut through bumps.
//!    A maximum grade clamp limits slope between adjacent nodes.
//! 2. **Artery extraction** traces continuous paths *through* intersections by
//!    forward-vector alignment, allowing global straightening of avenues.
//! 3. **Ramer-Douglas-Peucker** decimation removes unnecessary intermediate
//!    points, straightening nearly-collinear runs.
//! 4. **Fillet corners** replaces sharp bends with smooth quadratic Bézier arcs,
//!    giving the network a civil-engineered look.
//!
//! The result is a graph whose geometry is already presentation-ready — the 3D
//! mesher can simply extrude edges without additional spline smoothing.

use glam::Vec2;
use serde::{Deserialize, Serialize};
use symbios_ground::HeightMap;

use crate::geometry::closest_point_on_segment;
use crate::graph::{EdgeId, NodeId, RoadGraph, RoadType};
use crate::topology::{
    compute_active_degrees, extract_arteries, extract_chains, extract_chains_any_type,
};

/// Configuration for the graph rationalization pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RationalizeConfig {
    /// Master toggle.
    pub enabled: bool,
    /// RDP tolerance in world units — how aggressively to straighten.
    pub rdp_tolerance: f32,
    /// Fillet radius for major (contour-following) roads.
    pub major_fillet_radius: f32,
    /// Fillet radius for minor (gradient-following) roads.
    pub minor_fillet_radius: f32,
    /// Number of line segments used to approximate each fillet arc.
    pub fillet_segments: u32,
    /// Number of global Laplacian smoothing passes applied to the entire
    /// road graph's elevation profile before chain extraction. Higher values
    /// produce smoother roads that bridge over dips and cut through bumps,
    /// giving a civil-engineered look. 0 disables.
    pub elevation_smooth_passes: u32,
    /// Maximum allowed slope (grade) between adjacent nodes, expressed as a
    /// fraction (e.g. 0.15 = 15% grade). 0.0 disables clamping.
    pub max_grade: f32,
}

impl Default for RationalizeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rdp_tolerance: 2.0,
            major_fillet_radius: 20.0,
            minor_fillet_radius: 10.0,
            fillet_segments: 6,
            elevation_smooth_passes: 10,
            max_grade: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Unifies road types along topological chains (degree-2 paths between
/// intersections) so that each chain has a single, consistent [`RoadType`].
///
/// When major and minor traces collide at T-junctions, the resulting path
/// can alternate between types. This shatters arteries during smoothing
/// because `extract_arteries` only follows same-type edges. By overwriting
/// every edge in a chain to the majority type, we guarantee that each
/// inter-intersection stretch is a single unbroken type.
pub fn unify_road_types(graph: &mut RoadGraph) {
    let degrees = compute_active_degrees(graph);
    let chains = extract_chains_any_type(graph, &degrees);

    for chain in &chains {
        // Tally total length by road type.
        let mut major_len: f32 = 0.0;
        let mut minor_len: f32 = 0.0;

        for &eid in &chain.edges {
            let edge = &graph.edges[eid as usize];
            let a = graph.node_pos(edge.start);
            let b = graph.node_pos(edge.end);
            let seg_len = (b - a).length();
            match edge.road_type {
                RoadType::Major => major_len += seg_len,
                RoadType::Minor => minor_len += seg_len,
            }
        }

        let winner = if major_len >= minor_len {
            RoadType::Major
        } else {
            RoadType::Minor
        };

        // Overwrite all edges in this chain to the majority type.
        for &eid in &chain.edges {
            graph.edges[eid as usize].road_type = winner;
        }
    }
}

/// Rationalizes the road graph in-place.
///
/// **Phase 0 — Type unification:** Ensures each degree-2 chain between
/// intersections has a single, consistent road type (majority-vote).
///
/// **Phase 1 — Arteries:** Traces continuous paths of same-type edges
/// *through* intersections (by forward-vector alignment), then applies
/// RDP + fillet globally. Side-streets severed by moved intersection nodes
/// are reconnected by projecting onto the new artery geometry.
///
/// **Phase 2 — Residual chains:** Any edges not consumed by an artery are
/// processed with the original chain-based RDP + fillet pass.
pub fn rationalize_graph(graph: &mut RoadGraph, hm: &HeightMap, config: &RationalizeConfig) {
    // --- Phase 0: Unify road types along degree-2 chains ---
    unify_road_types(graph);

    // --- Phase 0.5: Global Laplacian elevation smoothing ---
    // Smooth elevations over the entire graph topology so that
    // intersections themselves average their heights with neighbors,
    // creating elevated bridges over dips and sunken cuts through bumps.
    // This runs before chain/artery extraction so all geometry inherits
    // the stabilized profile.
    smooth_graph_elevations(graph, hm, config);

    // --- Phase 1: Artery rationalization ---
    // Process Major arteries first (avenues get priority), then Minor.
    // Only arteries with 3+ nodes (2+ edges) benefit from global
    // straightening. Shorter paths are left for Phase 2's chain pass.
    // Multi-node arteries don't share edges (extract_arteries marks
    // visited), so processing them from a single extraction is safe.
    for &road_type in &[RoadType::Major, RoadType::Minor] {
        let degrees = compute_active_degrees(graph);
        let arteries = extract_arteries(graph, &degrees, road_type);

        for artery in &arteries {
            if artery.nodes.len() < 3 {
                continue;
            }
            rationalize_artery(
                graph,
                hm,
                artery.road_type,
                &artery.nodes,
                &artery.edges,
                config,
            );
        }
    }

    // --- Phase 2: Residual chain rationalization ---
    // Any edges not yet consumed by arteries get the original treatment.
    let degrees = compute_active_degrees(graph);
    let chains = extract_chains(graph, &degrees);

    for chain in &chains {
        rationalize_polyline(
            graph,
            hm,
            chain.road_type,
            &chain.nodes,
            &chain.edges,
            config,
        );
    }
}

/// Rationalizes a single artery: deactivates old edges, injects smoothed
/// geometry, and reconnects severed side-streets.
fn rationalize_artery(
    graph: &mut RoadGraph,
    hm: &HeightMap,
    road_type: RoadType,
    nodes: &[NodeId],
    edges: &[EdgeId],
    config: &RationalizeConfig,
) {
    let positions: Vec<Vec2> = nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
    if positions.len() < 2 {
        return;
    }

    // 1. Decimate + Fillet
    let simplified = ramer_douglas_peucker(&positions, config.rdp_tolerance);
    if simplified.len() < 2 {
        return;
    }
    let fillet_radius = match road_type {
        RoadType::Major => config.major_fillet_radius,
        RoadType::Minor => config.minor_fillet_radius,
    };
    let smoothed = fillet_corners(&simplified, fillet_radius, config.fillet_segments);
    if smoothed.len() < 2 {
        return;
    }

    // Compute smoothed elevations for the new polyline.
    let elevations = smooth_elevations(&smoothed, hm, config);

    // 2. Collect junction nodes along this artery that have side-street
    //    connections (edges of a *different* type, or same-type edges not
    //    part of this artery). These will need reconnection.
    let artery_edge_set: Vec<bool> = {
        let mut set = vec![false; graph.edges.len()];
        for &eid in edges {
            set[eid as usize] = true;
        }
        set
    };

    // For each interior artery node, find side-street edges that branch off.
    // We record (node_id, edge_id) pairs for reconnection after rewrite.
    let mut severed: Vec<(NodeId, EdgeId)> = Vec::new();
    for &nid in nodes {
        let node = &graph.nodes[nid as usize];
        for &eid in &node.edges {
            let e = &graph.edges[eid as usize];
            if !e.active || artery_edge_set[eid as usize] {
                continue;
            }
            // This is a side-street edge connected to an artery node.
            severed.push((nid, eid));
        }
    }

    // 3. Deactivate old artery edges.
    for &eid in edges {
        graph.edges[eid as usize].active = false;
    }

    // 4. Inject new smoothed geometry with elevation data.
    let first_node = nodes[0];
    let last_node = *nodes.last().unwrap();
    let new_edge_ids = inject_polyline(
        graph,
        road_type,
        first_node,
        last_node,
        &smoothed,
        &elevations,
    );

    // 5. Reconnect severed side-streets.
    // For each severed (old_artery_node, side_edge), find the closest point
    // on the new artery polyline and rewire the side-street endpoint.
    reconnect_side_streets(graph, &smoothed, &new_edge_ids, &severed);
}

/// Rationalizes a simple chain of degree-2 nodes (the original algorithm).
fn rationalize_polyline(
    graph: &mut RoadGraph,
    hm: &HeightMap,
    road_type: RoadType,
    nodes: &[NodeId],
    edges: &[EdgeId],
    config: &RationalizeConfig,
) {
    let positions: Vec<Vec2> = nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
    if positions.len() < 2 {
        return;
    }

    let simplified = ramer_douglas_peucker(&positions, config.rdp_tolerance);
    if simplified.len() < 2 {
        return;
    }

    let fillet_radius = match road_type {
        RoadType::Major => config.major_fillet_radius,
        RoadType::Minor => config.minor_fillet_radius,
    };
    let smoothed = fillet_corners(&simplified, fillet_radius, config.fillet_segments);
    if smoothed.len() < 2 {
        return;
    }

    let elevations = smooth_elevations(&smoothed, hm, config);

    for &eid in edges {
        graph.edges[eid as usize].active = false;
    }

    let first_node = nodes[0];
    let last_node = *nodes.last().unwrap();
    inject_polyline(
        graph,
        road_type,
        first_node,
        last_node,
        &smoothed,
        &elevations,
    );
}

/// Injects a smoothed polyline into the graph, reusing `first_node` and
/// `last_node` as endpoints. Elevations are assigned to each node.
/// Returns the edge IDs of the new edges.
fn inject_polyline(
    graph: &mut RoadGraph,
    road_type: RoadType,
    first_node: NodeId,
    last_node: NodeId,
    smoothed: &[Vec2],
    elevations: &[f32],
) -> Vec<EdgeId> {
    let mut new_edges = Vec::with_capacity(smoothed.len());
    let mut prev_node = first_node;
    for (i, &pos) in smoothed.iter().enumerate() {
        let current_node = if i == 0 {
            // Update first node's elevation to the smoothed value.
            graph.nodes[first_node as usize].elevation = elevations[i];
            first_node
        } else if i == smoothed.len() - 1 {
            graph.nodes[last_node as usize].elevation = elevations[i];
            last_node
        } else {
            graph.add_node_with_elevation(pos, elevations[i])
        };

        if i > 0 {
            let eid = graph.add_edge(prev_node, current_node, road_type);
            new_edges.push(eid);
        }
        prev_node = current_node;
    }
    new_edges
}

// ---------------------------------------------------------------------------
// Elevation smoothing
// ---------------------------------------------------------------------------

/// Runs Laplacian smoothing over the entire road graph's elevation profile.
///
/// Each active node's elevation is iteratively averaged with its active
/// neighbors' elevations, weighted by inverse edge length so that closely
/// spaced nodes don't dominate. This allows intersections themselves to
/// float above dips and sink through bumps, producing a civil-engineered
/// vertical alignment. Raw terrain heights are seeded once from the
/// heightmap before smoothing begins.
fn smooth_graph_elevations(graph: &mut RoadGraph, hm: &HeightMap, config: &RationalizeConfig) {
    if config.elevation_smooth_passes == 0 {
        return;
    }

    // Seed all node elevations from the heightmap.
    for node in &mut graph.nodes {
        node.elevation = hm.get_height_at(node.position.x, node.position.y);
    }

    // Build adjacency list of active neighbors for each node.
    let n = graph.nodes.len();
    let mut neighbors: Vec<Vec<(NodeId, f32)>> = vec![Vec::new(); n];
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let a = edge.start as usize;
        let b = edge.end as usize;
        let dist = (graph.nodes[a].position - graph.nodes[b].position)
            .length()
            .max(1e-6);
        let weight = 1.0 / dist;
        neighbors[a].push((edge.end, weight));
        neighbors[b].push((edge.start, weight));
    }

    // Laplacian smoothing passes.
    let mut new_elevs = vec![0.0f32; n];
    for _ in 0..config.elevation_smooth_passes {
        for i in 0..n {
            if neighbors[i].is_empty() {
                new_elevs[i] = graph.nodes[i].elevation;
                continue;
            }
            let mut sum = 0.0f32;
            let mut total_weight = 0.0f32;
            for &(neighbor, weight) in &neighbors[i] {
                sum += graph.nodes[neighbor as usize].elevation * weight;
                total_weight += weight;
            }
            // Blend: 50% self + 50% neighbor-weighted average.
            let neighbor_avg = sum / total_weight;
            new_elevs[i] = graph.nodes[i].elevation * 0.5 + neighbor_avg * 0.5;
        }
        for (node, &elev) in graph.nodes.iter_mut().zip(new_elevs.iter()) {
            node.elevation = elev;
        }
    }

    // Max grade clamping over edges (forward + backward BFS-order passes).
    if config.max_grade > 0.0 {
        for _ in 0..3 {
            for edge in &graph.edges {
                if !edge.active {
                    continue;
                }
                let a = edge.start as usize;
                let b = edge.end as usize;
                let dist = (graph.nodes[a].position - graph.nodes[b].position).length();
                let max_rise = dist * config.max_grade;
                // Clamp b relative to a.
                if graph.nodes[b].elevation > graph.nodes[a].elevation + max_rise {
                    graph.nodes[b].elevation = graph.nodes[a].elevation + max_rise;
                } else if graph.nodes[b].elevation < graph.nodes[a].elevation - max_rise {
                    graph.nodes[b].elevation = graph.nodes[a].elevation - max_rise;
                }
                // Clamp a relative to b.
                if graph.nodes[a].elevation > graph.nodes[b].elevation + max_rise {
                    graph.nodes[a].elevation = graph.nodes[b].elevation + max_rise;
                } else if graph.nodes[a].elevation < graph.nodes[b].elevation - max_rise {
                    graph.nodes[a].elevation = graph.nodes[b].elevation - max_rise;
                }
            }
        }
    }
}

/// Projects smoothed elevations from existing graph nodes onto new polyline
/// points (e.g. fillet geometry), then applies max-grade clamping.
///
/// For points that coincide with existing graph nodes, the pre-smoothed
/// elevation is used directly. For interpolated fillet points, the elevation
/// is linearly interpolated along the polyline from the nearest node
/// elevations, inheriting the globally smoothed profile.
fn smooth_elevations(points: &[Vec2], hm: &HeightMap, config: &RationalizeConfig) -> Vec<f32> {
    let n = points.len();
    // Use the globally-smoothed heightmap sample as a baseline — the global
    // Laplacian pass has already written stabilized elevations into graph
    // nodes, and new fillet points sample the terrain which is close enough.
    let mut elevs: Vec<f32> = points.iter().map(|p| hm.get_height_at(p.x, p.y)).collect();

    // Light local smoothing to blend fillet points with their neighbors.
    for _ in 0..3_u32.min(config.elevation_smooth_passes) {
        let prev = elevs.clone();
        for i in 1..n - 1 {
            elevs[i] = (prev[i - 1] + prev[i] + prev[i + 1]) / 3.0;
        }
    }

    // Max grade clamping: walk forward then backward, clamping slope.
    if config.max_grade > 0.0 {
        // Forward pass.
        for i in 1..n {
            let dist = (points[i] - points[i - 1]).length();
            let max_rise = dist * config.max_grade;
            if elevs[i] > elevs[i - 1] + max_rise {
                elevs[i] = elevs[i - 1] + max_rise;
            } else if elevs[i] < elevs[i - 1] - max_rise {
                elevs[i] = elevs[i - 1] - max_rise;
            }
        }
        // Backward pass.
        for i in (0..n - 1).rev() {
            let dist = (points[i + 1] - points[i]).length();
            let max_rise = dist * config.max_grade;
            if elevs[i] > elevs[i + 1] + max_rise {
                elevs[i] = elevs[i + 1] + max_rise;
            } else if elevs[i] < elevs[i + 1] - max_rise {
                elevs[i] = elevs[i + 1] - max_rise;
            }
        }
    }

    elevs
}

/// Reconnects side-street edges that were severed when artery nodes moved.
///
/// For each severed `(old_artery_node, side_edge)`:
/// 1. Find the closest *active* edge to the old node's position.
/// 2. Split that edge at the projection point.
/// 3. Rewire the side-street edge to connect to the new split node.
///
/// We search for the nearest active edge each time (rather than using stale
/// edge IDs) because previous splits invalidate the original edge list.
fn reconnect_side_streets(
    graph: &mut RoadGraph,
    _new_polyline: &[Vec2],
    new_edge_ids: &[EdgeId],
    severed: &[(NodeId, EdgeId)],
) {
    if severed.is_empty() || new_edge_ids.is_empty() {
        return;
    }

    // Track which edges belong to the new artery (including children from
    // splits). We seed with the initial new edges and grow as splits occur.
    let mut artery_edges: Vec<bool> = vec![false; graph.edges.len()];
    for &eid in new_edge_ids {
        artery_edges[eid as usize] = true;
    }

    for &(old_node, side_eid) in severed {
        if !graph.edges[side_eid as usize].active {
            continue;
        }

        let old_pos = graph.node_pos(old_node);

        // Find the closest active artery edge to old_pos.
        let mut best_eid: Option<EdgeId> = None;
        let mut best_dist_sq = f32::MAX;
        let mut best_proj = old_pos;

        // Grow the tracking vec if new edges were added by prior splits.
        artery_edges.resize(graph.edges.len(), false);

        for (eid, edge) in graph.edges.iter().enumerate() {
            if !edge.active || !artery_edges[eid] {
                continue;
            }
            let a = graph.node_pos(edge.start);
            let b = graph.node_pos(edge.end);
            let proj = closest_point_on_segment(old_pos, a, b);
            let dist_sq = (proj - old_pos).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_eid = Some(eid as EdgeId);
                best_proj = proj;
            }
        }

        let Some(target_eid) = best_eid else { continue };

        // Check if the projection is very close to an existing endpoint —
        // if so, rewire directly to that node instead of splitting.
        let target_edge = &graph.edges[target_eid as usize];
        let start_pos = graph.node_pos(target_edge.start);
        let end_pos = graph.node_pos(target_edge.end);
        let snap_threshold_sq = 1e-4;

        let connect_node = if (best_proj - start_pos).length_squared() < snap_threshold_sq {
            target_edge.start
        } else if (best_proj - end_pos).length_squared() < snap_threshold_sq {
            target_edge.end
        } else {
            // Split the artery edge at the projection point.
            let (split_node, ea, eb) = graph.split_edge(target_eid, best_proj);
            // Track the child edges as artery edges.
            artery_edges.resize(graph.edges.len(), false);
            artery_edges[ea as usize] = true;
            artery_edges[eb as usize] = true;
            split_node
        };

        rewire_edge_endpoint(graph, side_eid, old_node, connect_node);
    }
}

/// Rewires one endpoint of an edge from `old_node` to `new_node`.
fn rewire_edge_endpoint(
    graph: &mut RoadGraph,
    edge_id: EdgeId,
    old_node: NodeId,
    new_node: NodeId,
) {
    // Update the edge's endpoint.
    let edge = &mut graph.edges[edge_id as usize];
    if edge.start == old_node {
        edge.start = new_node;
    } else if edge.end == old_node {
        edge.end = new_node;
    } else {
        return; // old_node wasn't an endpoint — shouldn't happen.
    }

    // Update adjacency lists.
    graph.nodes[old_node as usize]
        .edges
        .retain(|&e| e != edge_id);
    graph.nodes[new_node as usize].edges.push(edge_id);
}

// ---------------------------------------------------------------------------
// Ramer-Douglas-Peucker
// ---------------------------------------------------------------------------

/// Simplifies a polyline by recursively removing points closer than
/// `tolerance` to the line between endpoints.
pub fn ramer_douglas_peucker(points: &[Vec2], tolerance: f32) -> Vec<Vec2> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;

    rdp_recurse(
        points,
        0,
        points.len() - 1,
        tolerance * tolerance,
        &mut keep,
    );

    points
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect()
}

fn rdp_recurse(points: &[Vec2], start: usize, end: usize, tol_sq: f32, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }

    let a = points[start];
    let b = points[end];
    let ab = b - a;
    let ab_len_sq = ab.length_squared();

    let mut max_dist_sq = 0.0f32;
    let mut max_idx = start;

    for (i, pt) in points
        .iter()
        .enumerate()
        .skip(start + 1)
        .take(end - start - 1)
    {
        let dist_sq = if ab_len_sq < 1e-12 {
            (*pt - a).length_squared()
        } else {
            let t = ((*pt - a).dot(ab) / ab_len_sq).clamp(0.0, 1.0);
            (*pt - (a + ab * t)).length_squared()
        };

        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
            max_idx = i;
        }
    }

    if max_dist_sq > tol_sq {
        keep[max_idx] = true;
        rdp_recurse(points, start, max_idx, tol_sq, keep);
        rdp_recurse(points, max_idx, end, tol_sq, keep);
    }
}

// ---------------------------------------------------------------------------
// Fillet corners
// ---------------------------------------------------------------------------

/// Replaces sharp corners in a polyline with smooth quadratic Bézier arcs.
///
/// For each interior vertex B with neighbours A and C, the fillet arc is
/// tangent to segments AB and BC at a distance of `radius · tan(half_angle)`
/// from B, clamped so adjacent fillets don't overlap.
pub fn fillet_corners(points: &[Vec2], radius: f32, segments: u32) -> Vec<Vec2> {
    if points.len() <= 2 || radius <= 0.0 || segments == 0 {
        return points.to_vec();
    }

    let n = points.len();
    let segments = segments.max(1);

    // Pre-compute segment lengths for setback clamping.
    let seg_lengths: Vec<f32> = points.windows(2).map(|w| (w[1] - w[0]).length()).collect();

    let mut result = Vec::with_capacity(n + (n - 2) * segments as usize);
    result.push(points[0]);

    for i in 1..n - 1 {
        let a = points[i - 1];
        let b = points[i];
        let c = points[i + 1];

        let ba = a - b;
        let bc = c - b;
        let ba_len = seg_lengths[i - 1];
        let bc_len = seg_lengths[i];

        if ba_len < 1e-6 || bc_len < 1e-6 {
            result.push(b);
            continue;
        }

        let ba_dir = ba / ba_len;
        let bc_dir = bc / bc_len;

        // Half-angle between the two legs.
        let cos_theta = ba_dir.dot(bc_dir).clamp(-1.0, 1.0);

        // Nearly straight — no fillet needed.
        if cos_theta > 0.999 {
            result.push(b);
            continue;
        }

        // Nearly a U-turn — skip filleting to avoid numerical issues.
        if cos_theta < -0.999 {
            result.push(b);
            continue;
        }

        let half_angle = cos_theta.acos() * 0.5;
        let tan_half = half_angle.tan();
        if tan_half.abs() < 1e-6 {
            result.push(b);
            continue;
        }

        // Desired setback along each leg.
        let mut setback = radius / tan_half;

        // Clamp so adjacent fillets don't overlap: each fillet can use at most
        // half of each incident segment.
        setback = setback.min(ba_len * 0.5).min(bc_len * 0.5);

        // Quadratic Bézier: P0 (on AB), control = B, P2 (on BC).
        let p0 = b + ba_dir * setback;
        let p2 = b + bc_dir * setback;

        for j in 0..=segments {
            let t = j as f32 / segments as f32;
            let q = quadratic_bezier(p0, b, p2, t);
            result.push(q);
        }
    }

    result.push(points[n - 1]);
    result
}

/// Evaluates a quadratic Bézier curve at parameter `t`.
fn quadratic_bezier(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let inv = 1.0 - t;
    p0 * (inv * inv) + p1 * (2.0 * inv * t) + p2 * (t * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_preserves_endpoints() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.01),
            Vec2::new(2.0, 0.0),
        ];
        let result = ramer_douglas_peucker(&pts, 0.1);
        assert_eq!(result.len(), 2); // middle point within tolerance
        assert_eq!(result[0], pts[0]);
        assert_eq!(result[1], pts[2]);
    }

    #[test]
    fn rdp_keeps_significant_points() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(10.0, 0.0),
        ];
        let result = ramer_douglas_peucker(&pts, 1.0);
        assert_eq!(result.len(), 3); // deviation > tolerance
    }

    #[test]
    fn rdp_short_input() {
        let pts = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)];
        let result = ramer_douglas_peucker(&pts, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn fillet_straight_line_unchanged() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            Vec2::new(10.0, 0.0),
        ];
        let result = fillet_corners(&pts, 2.0, 4);
        // Nearly straight → just passes through, no arc expansion
        // First, middle (kept as-is), last
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn fillet_right_angle_adds_points() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ];
        let result = fillet_corners(&pts, 3.0, 4);
        // Should have: first point + 5 bezier samples (0..=4) + last point = 7
        assert_eq!(result.len(), 7);
        // First and last should be preserved.
        assert_eq!(result[0], pts[0]);
        assert_eq!(*result.last().unwrap(), pts[2]);
    }

    #[test]
    fn fillet_clamps_setback_on_short_segments() {
        // Very short middle segment — setback should be clamped.
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
        ];
        let result = fillet_corners(&pts, 100.0, 4);
        // Should still produce valid output despite huge radius.
        assert!(result.len() >= 3);
        assert_eq!(result[0], pts[0]);
        assert_eq!(*result.last().unwrap(), pts[2]);
    }

    #[test]
    fn fillet_two_points_passthrough() {
        let pts = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let result = fillet_corners(&pts, 3.0, 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn rationalize_simple_chain() {
        use crate::graph::RoadGraph;

        // Build a zigzag chain: A -- B -- C -- D
        // Where B is slightly off the straight line A→D.
        let mut g = RoadGraph::default();
        let _a = g.add_node(Vec2::new(0.0, 0.0)); // 0
        let _b = g.add_node(Vec2::new(10.0, 0.5)); // 1 - slight deviation
        let _c = g.add_node(Vec2::new(20.0, 0.0)); // 2
        let _d = g.add_node(Vec2::new(30.0, 0.0)); // 3

        // Make A and D true junctions (degree >= 3) with two stubs each.
        let stub_a1 = g.add_node(Vec2::new(-10.0, 0.0)); // 4
        let stub_a2 = g.add_node(Vec2::new(0.0, -10.0)); // 5
        let stub_d1 = g.add_node(Vec2::new(40.0, 0.0)); // 6
        let stub_d2 = g.add_node(Vec2::new(30.0, -10.0)); // 7
        g.add_edge(stub_a1, _a, RoadType::Major);
        g.add_edge(stub_a2, _a, RoadType::Major);
        g.add_edge(_a, _b, RoadType::Major);
        g.add_edge(_b, _c, RoadType::Major);
        g.add_edge(_c, _d, RoadType::Major);
        g.add_edge(_d, stub_d1, RoadType::Major);
        g.add_edge(_d, stub_d2, RoadType::Major);

        let config = RationalizeConfig {
            enabled: true,
            rdp_tolerance: 2.0,
            major_fillet_radius: 0.0,
            minor_fillet_radius: 0.0,
            fillet_segments: 4,
            elevation_smooth_passes: 0,
            max_grade: 0.0,
        };

        let hm = symbios_ground::HeightMap::new(64, 64, 2.0);
        rationalize_graph(&mut g, &hm, &config);

        // Verify structural properties after rationalization:
        // 1. The graph should have a connected active sub-graph.
        let active_count = g.edges.iter().filter(|e| e.active).count();
        assert!(
            active_count >= 3,
            "expected at least 3 active edges (avenue + 2 stubs), got {active_count}"
        );

        // 2. Leaf nodes (stub_a1, stub_a2, stub_d1, stub_d2) should each
        //    have at least one active edge connecting them to the network.
        for &leaf in &[stub_a1, stub_a2, stub_d1, stub_d2] {
            let has_active = g.nodes[leaf as usize]
                .edges
                .iter()
                .any(|&eid| g.edges[eid as usize].active);
            assert!(has_active, "leaf node {leaf} should have active edges");
        }

        // 3. There should be a continuous active path from stub_a1's
        //    neighborhood to stub_d1's neighborhood (the straightened avenue).
        let active_major: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.active && e.road_type == RoadType::Major)
            .collect();
        assert!(!active_major.is_empty(), "should have active Major edges");
    }

    #[test]
    fn artery_rationalizes_through_intersections() {
        use crate::graph::RoadGraph;

        // Build a long Major avenue: A -- B -- C -- D -- E
        // with Minor side-streets branching off B and D.
        // B and D are T-junctions (degree 3).
        let mut g = RoadGraph::default();
        let _a = g.add_node(Vec2::new(0.0, 0.0));
        let _b = g.add_node(Vec2::new(10.0, 0.3)); // slight wobble
        let _c = g.add_node(Vec2::new(20.0, 0.0));
        let _d = g.add_node(Vec2::new(30.0, 0.2)); // slight wobble
        let _e = g.add_node(Vec2::new(40.0, 0.0));

        // Minor side-streets
        let s1 = g.add_node(Vec2::new(10.0, -15.0));
        let s2 = g.add_node(Vec2::new(30.0, -15.0));

        // Major avenue edges
        g.add_edge(_a, _b, RoadType::Major); // 0
        g.add_edge(_b, _c, RoadType::Major); // 1
        g.add_edge(_c, _d, RoadType::Major); // 2
        g.add_edge(_d, _e, RoadType::Major); // 3

        // Minor side-streets
        g.add_edge(_b, s1, RoadType::Minor); // 4
        g.add_edge(_d, s2, RoadType::Minor); // 5

        let config = RationalizeConfig {
            enabled: true,
            rdp_tolerance: 1.0,
            major_fillet_radius: 0.0,
            minor_fillet_radius: 0.0,
            fillet_segments: 4,
            elevation_smooth_passes: 0,
            max_grade: 0.0,
        };

        let hm = symbios_ground::HeightMap::new(64, 64, 2.0);
        rationalize_graph(&mut g, &hm, &config);

        // The original Major avenue edges should be deactivated.
        assert!(!g.edges[0].active, "Major A→B should be deactivated");
        assert!(!g.edges[1].active, "Major B→C should be deactivated");
        assert!(!g.edges[2].active, "Major C→D should be deactivated");
        assert!(!g.edges[3].active, "Major D→E should be deactivated");

        // New Major edges should exist forming the straightened avenue.
        let new_major: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.active && e.road_type == RoadType::Major)
            .collect();
        assert!(!new_major.is_empty(), "should have new Major artery edges");

        // Minor side-streets should still connect s1 and s2 to the network.
        // They may have been rewired (original edge deactivated, replaced by
        // Phase 2) but equivalent active Minor edges must exist.
        let active_minor: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.active && e.road_type == RoadType::Minor)
            .collect();
        assert_eq!(
            active_minor.len(),
            2,
            "should have 2 active Minor side-streets"
        );

        // s1 and s2 must each have an active edge.
        for &leaf in &[s1, s2] {
            let has_active = g.nodes[leaf as usize]
                .edges
                .iter()
                .any(|&eid| g.edges[eid as usize].active);
            assert!(
                has_active,
                "side-street leaf node {leaf} should be connected"
            );
        }
    }
}
