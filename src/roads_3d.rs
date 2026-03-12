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
use crate::topology;

/// Engine-agnostic mesh container.
///
/// Vertices use a Y-up coordinate system: `[x, y, z]` where Y is the
/// world-space height sampled from the [`HeightMap`].
#[derive(Debug, Clone, Default)]
pub struct ProceduralMesh {
    /// Vertex positions as `[x, y, z]` (Y-up).
    pub vertices: Vec<[f32; 3]>,
    /// Per-vertex normals (currently always `[0, 1, 0]` — flat upward).
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex texture coordinates `[u, v]`.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices into the vertex/normal/uv arrays.
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
    /// Legacy: subdivisions per graph edge for Catmull-Rom spline ribbons.
    /// Ignored when the graph has been rationalized (geometry is already smooth).
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
// Node degree computation (delegated to topology module)
// ---------------------------------------------------------------------------

fn compute_active_degrees(graph: &RoadGraph) -> Vec<u32> {
    topology::compute_active_degrees(graph)
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
// Chain extraction (delegated to topology module)
// ---------------------------------------------------------------------------

/// Local alias — the ribbon generator only needs nodes and road type.
struct Chain {
    nodes: Vec<NodeId>,
    road_type: RoadType,
}

fn extract_chains(graph: &RoadGraph, degrees: &[u32]) -> Vec<Chain> {
    topology::extract_chains(graph, degrees)
        .into_iter()
        .map(|c| Chain {
            nodes: c.nodes,
            road_type: c.road_type,
        })
        .collect()
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

    // Gather 2D positions along the chain. When the graph has been
    // rationalized the geometry is already smooth — no spline needed.
    let smooth_pts: Vec<Vec2> = chain.nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
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
