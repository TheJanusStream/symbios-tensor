//! Engine-agnostic 3D road mesh generation.
//!
//! Takes a [`RoadGraph`] and [`HeightMap`] and produces [`ProceduralMesh`]
//! vertex buffers for hub (intersection) and ribbon (street) geometry.
//!
//! **Hubs** are flat polygons (regular N-gons) placed at intersections and dead
//! ends (degree ≠ 2 nodes). **Ribbons** are extruded strips that follow
//! chains of degree-2 nodes, smoothed with Centripetal Catmull-Rom splines
//! and truncated at hub boundaries.

use glam::Vec2;
use symbios_ground::HeightMap;

use crate::graph::{NodeId, RoadGraph, RoadType};

/// Engine-agnostic mesh container.
#[derive(Debug, Clone, Default)]
pub struct ProceduralMesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl ProceduralMesh {
    /// Merge another mesh into this one, offsetting indices.
    pub fn append(&mut self, other: &ProceduralMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}

/// Configuration for 3D road mesh generation.
#[derive(Debug, Clone)]
pub struct RoadMeshConfig {
    /// Half-width of major roads (world units).
    pub major_half_width: f32,
    /// Half-width of minor roads (world units).
    pub minor_half_width: f32,
    /// Number of sides for hub polygons (e.g. 8 = octagon, 16 = near-circle).
    pub hub_sides: u32,
    /// Depth bias: vertices are raised above the terrain by this amount to
    /// prevent z-fighting.
    pub depth_bias: f32,
    /// UV texture scale: world units per texture repeat.
    pub texture_scale: f32,
    /// Number of subdivisions per graph edge when generating Catmull-Rom
    /// spline points for ribbons.
    pub spline_subdivisions: u32,
}

impl Default for RoadMeshConfig {
    fn default() -> Self {
        Self {
            major_half_width: 3.0,
            minor_half_width: 2.0,
            hub_sides: 8,
            depth_bias: 0.05,
            texture_scale: 0.1,
            spline_subdivisions: 8,
        }
    }
}

/// Generated road meshes split by component type.
#[derive(Debug, Clone, Default)]
pub struct RoadMeshes {
    /// Intersection / dead-end hub polygons.
    pub hubs: ProceduralMesh,
    /// Street ribbon strips.
    pub ribbons: ProceduralMesh,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generates 3D road meshes from a road graph and heightmap.
pub fn generate_road_meshes(
    graph: &RoadGraph,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> RoadMeshes {
    let mut meshes = RoadMeshes::default();

    // Classify nodes by active degree.
    let degrees = compute_active_degrees(graph);

    // --- Hubs (degree != 2) ---
    for (node_id, &deg) in degrees.iter().enumerate() {
        if deg == 0 {
            continue;
        }
        if deg == 2 {
            continue;
        }
        let hub = generate_hub(graph, node_id as NodeId, deg, heightmap, config);
        meshes.hubs.append(&hub);
    }

    // --- Ribbons (chains of degree-2 nodes) ---
    let chains = extract_chains(graph, &degrees);
    for chain in &chains {
        let ribbon = generate_ribbon(graph, chain, &degrees, heightmap, config);
        meshes.ribbons.append(&ribbon);
    }

    meshes
}

// ---------------------------------------------------------------------------
// Node degree computation
// ---------------------------------------------------------------------------

fn compute_active_degrees(graph: &RoadGraph) -> Vec<u32> {
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

// ---------------------------------------------------------------------------
// Hub generation
// ---------------------------------------------------------------------------

fn generate_hub(
    graph: &RoadGraph,
    node_id: NodeId,
    _degree: u32,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> ProceduralMesh {
    let center = graph.node_pos(node_id);
    let node = &graph.nodes[node_id as usize];

    // Find max half-width of all connecting active edges.
    let mut radius = config.minor_half_width;
    for &eid in &node.edges {
        let edge = &graph.edges[eid as usize];
        if !edge.active {
            continue;
        }
        let hw = match edge.road_type {
            RoadType::Major => config.major_half_width,
            RoadType::Minor => config.minor_half_width,
        };
        if hw > radius {
            radius = hw;
        }
    }

    let sides = config.hub_sides.max(3);
    let center_y = heightmap.get_height_at(center.x, center.y) + config.depth_bias;

    let mut mesh = ProceduralMesh::default();

    // Center vertex.
    mesh.vertices.push([center.x, center_y, center.y]);
    mesh.normals.push([0.0, 1.0, 0.0]);
    mesh.uvs.push([
        center.x * config.texture_scale,
        center.y * config.texture_scale,
    ]);

    // Perimeter vertices.
    let angle_step = std::f32::consts::TAU / sides as f32;
    for i in 0..sides {
        let angle = angle_step * i as f32;
        let (sin, cos) = angle.sin_cos();
        let px = center.x + cos * radius;
        let pz = center.y + sin * radius;
        let py = heightmap.get_height_at(px, pz) + config.depth_bias;

        mesh.vertices.push([px, py, pz]);
        mesh.normals.push([0.0, 1.0, 0.0]);
        mesh.uvs
            .push([px * config.texture_scale, pz * config.texture_scale]);
    }

    // Fan triangles: center(0) + perimeter.
    for i in 0..sides {
        let a = 1 + i;
        let b = 1 + (i + 1) % sides;
        mesh.indices.push(0);
        mesh.indices.push(b);
        mesh.indices.push(a);
    }

    mesh
}

// ---------------------------------------------------------------------------
// Chain extraction (paths of degree-2 nodes)
// ---------------------------------------------------------------------------

/// A chain is a sequence of NodeIds from one junction/dead-end to another,
/// passing through only degree-2 interior nodes.
struct Chain {
    nodes: Vec<NodeId>,
    road_type: RoadType,
}

fn extract_chains(graph: &RoadGraph, degrees: &[u32]) -> Vec<Chain> {
    let mut visited_edges = vec![false; graph.edges.len()];
    let mut chains = Vec::new();

    for (eid, edge) in graph.edges.iter().enumerate() {
        if !edge.active || visited_edges[eid] {
            continue;
        }

        // Start a chain from this edge.
        let mut chain_nodes = Vec::new();
        let road_type = edge.road_type;

        // Walk backwards from edge.start as far as degree-2 nodes go.
        let head = walk_chain(
            graph,
            degrees,
            edge.start,
            eid as u32,
            &mut visited_edges,
            road_type,
        );
        head.into_iter().rev().for_each(|n| chain_nodes.push(n));

        // Now walk forward from edge.end.
        visited_edges[eid] = true;
        chain_nodes.push(edge.start);
        chain_nodes.push(edge.end);

        let tail = walk_chain(
            graph,
            degrees,
            edge.end,
            eid as u32,
            &mut visited_edges,
            road_type,
        );
        chain_nodes.extend(tail);

        // Deduplicate the start node if walk_chain returned it.
        dedup_consecutive(&mut chain_nodes);

        if chain_nodes.len() >= 2 {
            chains.push(Chain {
                nodes: chain_nodes,
                road_type,
            });
        }
    }

    chains
}

/// Walk along degree-2 nodes starting from `start_node`, not going back
/// through `from_edge`. Returns the sequence of nodes visited (not including
/// `start_node`).
fn walk_chain(
    graph: &RoadGraph,
    degrees: &[u32],
    start_node: NodeId,
    from_edge: u32,
    visited_edges: &mut [bool],
    road_type: RoadType,
) -> Vec<NodeId> {
    let mut result = Vec::new();
    let mut current = start_node;
    let mut prev_edge = from_edge;

    // Only walk through degree-2 nodes.
    if degrees[current as usize] != 2 {
        return result;
    }

    loop {
        // Find the other active edge at this degree-2 node.
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
            // Only follow edges of the same road type for a clean chain.
            if e.road_type != road_type {
                continue;
            }
            next_edge = Some(eid);
            break;
        }

        let Some(ne) = next_edge else { break };
        visited_edges[ne as usize] = true;

        let next_node = graph.opposite(ne, current);
        result.push(next_node);

        if degrees[next_node as usize] != 2 {
            break;
        }
        prev_edge = ne;
        current = next_node;
    }

    result
}

fn dedup_consecutive(v: &mut Vec<NodeId>) {
    v.dedup();
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline
// ---------------------------------------------------------------------------

/// Centripetal Catmull-Rom interpolation between p1 and p2, given surrounding
/// control points p0 and p3.
fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let alpha = 0.5; // centripetal

    fn knot(ti: f32, pi: Vec2, pj: Vec2, alpha: f32) -> f32 {
        let d = pj - pi;
        let len = d.length_squared().powf(alpha * 0.5);
        ti + len
    }

    let t0 = 0.0f32;
    let t1 = knot(t0, p0, p1, alpha);
    let t2 = knot(t1, p1, p2, alpha);
    let t3 = knot(t2, p2, p3, alpha);

    let tt = t1 + t * (t2 - t1);

    let a1 = lerp2(p0, p1, t0, t1, tt);
    let a2 = lerp2(p1, p2, t1, t2, tt);
    let a3 = lerp2(p2, p3, t2, t3, tt);

    let b1 = lerp2(a1, a2, t0, t2, tt);
    let b2 = lerp2(a2, a3, t1, t3, tt);

    lerp2(b1, b2, t1, t2, tt)
}

fn lerp2(a: Vec2, b: Vec2, t0: f32, t1: f32, t: f32) -> Vec2 {
    let denom = t1 - t0;
    if denom.abs() < 1e-10 {
        return a;
    }
    a + (b - a) * ((t - t0) / denom)
}

/// Generates a smooth polyline from a chain of 2D points using Catmull-Rom
/// splines. Returns a dense sequence of 2D points.
fn smooth_chain(points: &[Vec2], subdivisions: u32) -> Vec<Vec2> {
    if points.len() < 2 {
        return points.to_vec();
    }
    if points.len() == 2 {
        // Just linearly subdivide.
        let mut result = Vec::with_capacity(subdivisions as usize + 1);
        for i in 0..=subdivisions {
            let t = i as f32 / subdivisions as f32;
            result.push(points[0].lerp(points[1], t));
        }
        return result;
    }

    let n = points.len();
    let mut result = Vec::new();

    for seg in 0..n - 1 {
        let p0 = if seg == 0 {
            // Mirror first point.
            points[0] * 2.0 - points[1]
        } else {
            points[seg - 1]
        };
        let p1 = points[seg];
        let p2 = points[seg + 1];
        let p3 = if seg + 2 < n {
            points[seg + 2]
        } else {
            // Mirror last point.
            points[n - 1] * 2.0 - points[n - 2]
        };

        let count = if seg == n - 2 {
            subdivisions + 1
        } else {
            subdivisions
        };
        for i in 0..count {
            let t = i as f32 / subdivisions as f32;
            result.push(catmull_rom(p0, p1, p2, p3, t));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Ribbon generation
// ---------------------------------------------------------------------------

fn generate_ribbon(
    graph: &RoadGraph,
    chain: &Chain,
    degrees: &[u32],
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> ProceduralMesh {
    let half_width = match chain.road_type {
        RoadType::Major => config.major_half_width,
        RoadType::Minor => config.minor_half_width,
    };

    // Gather 2D positions along the chain.
    let raw_points: Vec<Vec2> = chain.nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
    if raw_points.len() < 2 {
        return ProceduralMesh::default();
    }

    // Smooth with Catmull-Rom.
    let smooth_pts = smooth_chain(&raw_points, config.spline_subdivisions);
    if smooth_pts.len() < 2 {
        return ProceduralMesh::default();
    }

    // Truncate at hub boundaries.
    let first_node = chain.nodes[0];
    let last_node = *chain.nodes.last().unwrap();
    let start_radius = hub_radius_for_node(graph, first_node, degrees, config);
    let end_radius = hub_radius_for_node(graph, last_node, degrees, config);

    let truncated = truncate_polyline(&smooth_pts, start_radius, end_radius);
    if truncated.len() < 2 {
        return ProceduralMesh::default();
    }

    // Extrude ribbon.
    extrude_ribbon(&truncated, half_width, heightmap, config)
}

/// Returns the hub radius for a node, or 0.0 if the node is degree-2 (no hub).
fn hub_radius_for_node(
    graph: &RoadGraph,
    node_id: NodeId,
    degrees: &[u32],
    config: &RoadMeshConfig,
) -> f32 {
    let deg = degrees[node_id as usize];
    if deg == 2 {
        return 0.0;
    }
    // Same logic as hub generation: max half-width of connecting edges.
    let node = &graph.nodes[node_id as usize];
    let mut radius = config.minor_half_width;
    for &eid in &node.edges {
        let edge = &graph.edges[eid as usize];
        if !edge.active {
            continue;
        }
        let hw = match edge.road_type {
            RoadType::Major => config.major_half_width,
            RoadType::Minor => config.minor_half_width,
        };
        if hw > radius {
            radius = hw;
        }
    }
    radius
}

/// Truncates a polyline by removing length from the start and end.
fn truncate_polyline(points: &[Vec2], start_trim: f32, end_trim: f32) -> Vec<Vec2> {
    if points.len() < 2 {
        return points.to_vec();
    }

    // Compute cumulative arc lengths.
    let mut arc_lengths = Vec::with_capacity(points.len());
    arc_lengths.push(0.0f32);
    for i in 1..points.len() {
        let seg_len = (points[i] - points[i - 1]).length();
        arc_lengths.push(arc_lengths[i - 1] + seg_len);
    }
    let total = *arc_lengths.last().unwrap();

    let t_start = start_trim;
    let t_end = total - end_trim;
    if t_start >= t_end {
        return Vec::new();
    }

    let mut result = Vec::new();

    // Find start point.
    result.push(point_at_arc_length(points, &arc_lengths, t_start));

    // Add interior points.
    for i in 1..points.len() - 1 {
        if arc_lengths[i] > t_start && arc_lengths[i] < t_end {
            result.push(points[i]);
        }
    }

    // Find end point.
    result.push(point_at_arc_length(points, &arc_lengths, t_end));

    result
}

/// Returns the 2D point at a given arc length along a polyline.
fn point_at_arc_length(points: &[Vec2], arc_lengths: &[f32], target: f32) -> Vec2 {
    for i in 1..points.len() {
        if arc_lengths[i] >= target {
            let seg_len = arc_lengths[i] - arc_lengths[i - 1];
            if seg_len < 1e-6 {
                return points[i];
            }
            let t = (target - arc_lengths[i - 1]) / seg_len;
            return points[i - 1].lerp(points[i], t);
        }
    }
    *points.last().unwrap()
}

/// Extrudes a 2D polyline into a ribbon mesh with height sampling.
fn extrude_ribbon(
    points: &[Vec2],
    half_width: f32,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> ProceduralMesh {
    let n = points.len();
    let mut mesh = ProceduralMesh {
        vertices: Vec::with_capacity(n * 2),
        normals: Vec::with_capacity(n * 2),
        uvs: Vec::with_capacity(n * 2),
        indices: Vec::with_capacity((n - 1) * 6),
    };

    let mut accum_dist = 0.0f32;

    for i in 0..n {
        // Tangent: forward difference, backward at end, average in middle.
        let tangent = if i == 0 {
            (points[1] - points[0]).normalize_or_zero()
        } else if i == n - 1 {
            (points[n - 1] - points[n - 2]).normalize_or_zero()
        } else {
            (points[i + 1] - points[i - 1]).normalize_or_zero()
        };

        // 2D right normal (perpendicular to tangent).
        let right = Vec2::new(-tangent.y, tangent.x);

        // Left and right extrusion.
        let left_pt = points[i] - right * half_width;
        let right_pt = points[i] + right * half_width;

        // Sample heights.
        let left_y = heightmap.get_height_at(left_pt.x, left_pt.y) + config.depth_bias;
        let right_y = heightmap.get_height_at(right_pt.x, right_pt.y) + config.depth_bias;

        mesh.vertices.push([right_pt.x, right_y, right_pt.y]);
        mesh.vertices.push([left_pt.x, left_y, left_pt.y]);

        mesh.normals.push([0.0, 1.0, 0.0]);
        mesh.normals.push([0.0, 1.0, 0.0]);

        // UV: U = accumulated arc distance (scaled), V = 0 left, 1 right.
        if i > 0 {
            accum_dist += (points[i] - points[i - 1]).length();
        }
        let u = accum_dist * config.texture_scale;
        mesh.uvs.push([u, 0.0]);
        mesh.uvs.push([u, 1.0]);
    }

    // Triangle strip → index buffer (two triangles per quad).
    for i in 0..n as u32 - 1 {
        let bl = i * 2; // bottom-left
        let br = i * 2 + 1; // bottom-right
        let tl = (i + 1) * 2; // top-left
        let tr = (i + 1) * 2 + 1; // top-right

        // Triangle 1
        mesh.indices.push(bl);
        mesh.indices.push(tl);
        mesh.indices.push(br);

        // Triangle 2
        mesh.indices.push(br);
        mesh.indices.push(tl);
        mesh.indices.push(tr);
    }

    mesh
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RoadGraph;

    /// Helper: builds a simple cross intersection graph.
    fn cross_graph() -> RoadGraph {
        let mut g = RoadGraph::default();
        // Center node
        let c = g.add_node(Vec2::new(50.0, 50.0));
        // Four arms
        let n = g.add_node(Vec2::new(50.0, 20.0));
        let s = g.add_node(Vec2::new(50.0, 80.0));
        let e = g.add_node(Vec2::new(80.0, 50.0));
        let w = g.add_node(Vec2::new(20.0, 50.0));

        g.add_edge(c, n, RoadType::Major);
        g.add_edge(c, s, RoadType::Major);
        g.add_edge(c, e, RoadType::Minor);
        g.add_edge(c, w, RoadType::Minor);

        g
    }

    fn flat_heightmap() -> HeightMap {
        HeightMap::new(64, 64, 2.0)
    }

    #[test]
    fn hub_mesh_has_correct_vertex_count() {
        let g = cross_graph();
        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();
        let degrees = compute_active_degrees(&g);

        // Center node has degree 4 → hub.
        let hub = generate_hub(&g, 0, degrees[0], &hm, &config);
        // 1 center + hub_sides perimeter vertices.
        assert_eq!(hub.vertices.len(), (1 + config.hub_sides) as usize);
        assert_eq!(hub.indices.len(), (config.hub_sides * 3) as usize);
    }

    #[test]
    fn ribbon_mesh_nonempty() {
        // Two-node chain (single edge, both endpoints are degree-1 dead ends).
        let mut g = RoadGraph::default();
        let a = g.add_node(Vec2::new(10.0, 10.0));
        let b = g.add_node(Vec2::new(90.0, 10.0));
        g.add_edge(a, b, RoadType::Major);

        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();
        let meshes = generate_road_meshes(&g, &hm, &config);

        // Both endpoints are degree-1 → hubs exist.
        assert!(!meshes.hubs.vertices.is_empty());
        // Ribbon should also exist.
        assert!(!meshes.ribbons.vertices.is_empty());
        // Indices should be divisible by 3 (triangles).
        assert_eq!(meshes.ribbons.indices.len() % 3, 0);
    }

    #[test]
    fn smooth_chain_preserves_endpoints() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 5.0),
            Vec2::new(20.0, 0.0),
        ];
        let smoothed = smooth_chain(&pts, 4);
        assert!((smoothed[0] - pts[0]).length() < 1e-4);
        assert!((*smoothed.last().unwrap() - *pts.last().unwrap()).length() < 1e-4);
    }

    #[test]
    fn truncate_polyline_shortens() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 0.0),
        ];
        let truncated = truncate_polyline(&pts, 3.0, 3.0);
        assert!(!truncated.is_empty());
        // First point should be at x≈3.
        assert!((truncated[0].x - 3.0).abs() < 1e-4);
        // Last point should be at x≈17.
        assert!((truncated.last().unwrap().x - 17.0).abs() < 1e-4);
    }

    #[test]
    fn full_pipeline_cross_graph() {
        let g = cross_graph();
        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();

        let meshes = generate_road_meshes(&g, &hm, &config);

        // 5 nodes: center = degree 4 (hub), 4 arms = degree 1 (hubs).
        // All 5 should generate hubs.
        let degrees = compute_active_degrees(&g);
        let hub_count = degrees.iter().filter(|&&d| d > 0 && d != 2).count();
        assert_eq!(hub_count, 5);

        // Hubs mesh should have vertices.
        assert!(!meshes.hubs.vertices.is_empty());
        // Ribbons should exist (4 edges, each is a single-edge chain).
        assert!(!meshes.ribbons.vertices.is_empty());
    }
}
