//! Engine-agnostic 3D road mesh generation.
//!
//! Takes a [`RoadGraph`] and [`HeightMap`] and produces [`ProceduralMesh`]
//! vertex buffers for hub (intersection) and ribbon (street) geometry.
//!
//! **Hubs** are flat polygons (regular N-gons) placed at intersections and dead
//! ends (degree ≠ 2 nodes). **Ribbons** are extruded strips that follow
//! chains of degree-2 nodes, smoothed with Centripetal Catmull-Rom splines
//! and truncated at hub boundaries.

use glam::{Vec2, Vec3};
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
    /// Extra radius added to intersection hubs beyond the road half-width,
    /// creating a wider turning zone (world units).
    pub curb_radius: f32,
    /// Embankment skirt configuration.
    pub skirt: SkirtConfig,
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
            curb_radius: 2.0,
            skirt: SkirtConfig::default(),
        }
    }
}

/// Generated road meshes split by component type.
#[derive(Debug, Clone, Default)]
pub struct RoadMeshes {
    /// Intersection / dead-end hub polygons.
    pub hubs: ProceduralMesh,
    /// Street ribbon strips (flat asphalt surface).
    pub ribbons: ProceduralMesh,
    /// Embankment skirts that taper from the road edge down to terrain.
    pub skirts: ProceduralMesh,
}

/// Configuration for the embankment skirts flanking roads.
#[derive(Debug, Clone)]
pub struct SkirtConfig {
    /// Width of the skirt extending outward from the road edge (world units).
    pub width: f32,
    /// How far below the terrain surface the skirt buries itself.
    pub bury_depth: f32,
}

impl Default for SkirtConfig {
    fn default() -> Self {
        Self {
            width: 3.0,
            bury_depth: 0.5,
        }
    }
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
        let (hub, hub_skirt) = generate_hub(graph, node_id as NodeId, deg, heightmap, config);
        meshes.hubs.append(&hub);
        meshes.skirts.append(&hub_skirt);
    }

    // --- Ribbons (chains of degree-2 nodes) ---
    let chains = extract_chains(graph, &degrees);
    for chain in &chains {
        let (ribbon, skirt) = generate_ribbon(graph, chain, &degrees, heightmap, config);
        meshes.ribbons.append(&ribbon);
        meshes.skirts.append(&skirt);
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
) -> (ProceduralMesh, ProceduralMesh) {
    let center = graph.node_pos(node_id);
    let node = &graph.nodes[node_id as usize];

    // Find max half-width of all connecting active edges + curb radius.
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
    radius += config.curb_radius;

    let sides = config.hub_sides.max(3);
    // Flat hub: use the sovereign node elevation (smoothed by rationalization).
    let center_y = node.elevation + config.depth_bias;

    let mut mesh = ProceduralMesh::default();

    // Center vertex.
    mesh.vertices.push([center.x, center_y, center.y]);
    mesh.normals.push([0.0, 1.0, 0.0]);
    mesh.uvs.push([
        center.x * config.texture_scale,
        center.y * config.texture_scale,
    ]);

    // Perimeter vertices — all at centerline height for a flat intersection.
    let angle_step = std::f32::consts::TAU / sides as f32;
    let mut perimeter_pts = Vec::with_capacity(sides as usize);
    for i in 0..sides {
        let angle = angle_step * i as f32;
        let (sin, cos) = angle.sin_cos();
        let px = center.x + cos * radius;
        let pz = center.y + sin * radius;

        mesh.vertices.push([px, center_y, pz]);
        mesh.normals.push([0.0, 1.0, 0.0]);
        mesh.uvs
            .push([px * config.texture_scale, pz * config.texture_scale]);
        perimeter_pts.push((px, pz));
    }

    // Fan triangles: center(0) + perimeter.
    for i in 0..sides {
        let a = 1 + i;
        let b = 1 + (i + 1) % sides;
        mesh.indices.push(0);
        mesh.indices.push(b);
        mesh.indices.push(a);
    }

    // --- Hub skirt: ring of quads extruding outward from perimeter ---
    let skirt_w = config.skirt.width;
    let bury = config.skirt.bury_depth;
    let mut skirt = ProceduralMesh::default();

    for &(px, pz) in &perimeter_pts {
        // Direction from center outward (for extrusion).
        let dx = px - center.x;
        let dz = pz - center.y;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        let nx = dx / len;
        let nz = dz / len;

        let outer_x = px + nx * skirt_w;
        let outer_z = pz + nz * skirt_w;
        let outer_y = heightmap.get_height_at(outer_x, outer_z) - bury;

        // Inner vertex (at road edge, hub height).
        skirt.vertices.push([px, center_y, pz]);
        skirt.normals.push([0.0, 1.0, 0.0]);
        skirt.uvs.push([px * config.texture_scale, pz * config.texture_scale]);

        // Outer vertex (skirt width outward, terrain - bury).
        skirt.vertices.push([outer_x, outer_y, outer_z]);
        skirt.normals.push([0.0, 1.0, 0.0]);
        skirt.uvs.push([outer_x * config.texture_scale, outer_z * config.texture_scale]);
    }

    // Skirt index buffer: quads connecting adjacent perimeter edges.
    for i in 0..sides {
        let next = (i + 1) % sides;
        let i0 = i * 2;       // inner current
        let o0 = i * 2 + 1;   // outer current
        let i1 = next * 2;    // inner next
        let o1 = next * 2 + 1; // outer next

        // Two triangles per quad, CCW winding so normals face outward.
        skirt.indices.push(i0);
        skirt.indices.push(i1);
        skirt.indices.push(o0);

        skirt.indices.push(o0);
        skirt.indices.push(i1);
        skirt.indices.push(o1);
    }

    (mesh, skirt)
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
) -> (ProceduralMesh, ProceduralMesh) {
    let half_width = match chain.road_type {
        RoadType::Major => config.major_half_width,
        RoadType::Minor => config.minor_half_width,
    };

    // Gather 2D positions and sovereign elevations along the chain.
    let smooth_pts: Vec<Vec2> = chain.nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
    let node_elevs: Vec<f32> = chain.nodes.iter().map(|&nid| graph.nodes[nid as usize].elevation).collect();
    if smooth_pts.len() < 2 {
        return (ProceduralMesh::default(), ProceduralMesh::default());
    }

    // Truncate at hub boundaries.
    let first_node = chain.nodes[0];
    let last_node = *chain.nodes.last().unwrap();
    let start_radius = hub_radius_for_node(graph, first_node, degrees, config);
    let end_radius = hub_radius_for_node(graph, last_node, degrees, config);

    let (truncated, truncated_elevs) = truncate_polyline_with_elevations(
        &smooth_pts, &node_elevs, start_radius, end_radius,
    );
    if truncated.len() < 2 {
        return (ProceduralMesh::default(), ProceduralMesh::default());
    }

    // Extrude asphalt ribbon (sovereign elevation) + embankment skirts (heightmap).
    extrude_ribbon(&truncated, &truncated_elevs, half_width, heightmap, config)
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
    // Same logic as hub generation: max half-width of connecting edges + curb radius.
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
    radius + config.curb_radius
}

/// Truncates a polyline and its associated elevations by removing length
/// from the start and end, interpolating elevations at the cut points.
fn truncate_polyline_with_elevations(
    points: &[Vec2],
    elevations: &[f32],
    start_trim: f32,
    end_trim: f32,
) -> (Vec<Vec2>, Vec<f32>) {
    if points.len() < 2 {
        return (points.to_vec(), elevations.to_vec());
    }

    let mut arc_lengths = Vec::with_capacity(points.len());
    arc_lengths.push(0.0f32);
    for i in 1..points.len() {
        let seg_len = (points[i] - points[i - 1]).length();
        arc_lengths.push(arc_lengths[i - 1] + seg_len);
    }
    let total = *arc_lengths.last().unwrap();

    // Clamp combined trim so it never exceeds 98% of the segment length.
    // This prevents ribbons from vanishing when two hubs are very close.
    let max_trim = total * 0.98;
    let (adj_start, adj_end) = if start_trim + end_trim > max_trim {
        let scale = max_trim / (start_trim + end_trim);
        (start_trim * scale, end_trim * scale)
    } else {
        (start_trim, end_trim)
    };

    let t_start = adj_start;
    let t_end = total - adj_end;
    if t_start >= t_end {
        return (Vec::new(), Vec::new());
    }

    let mut result_pts = Vec::new();
    let mut result_elevs = Vec::new();

    // Start point.
    result_pts.push(point_at_arc_length(points, &arc_lengths, t_start));
    result_elevs.push(elevation_at_arc_length(elevations, &arc_lengths, t_start));

    // Interior points.
    for i in 1..points.len() - 1 {
        if arc_lengths[i] > t_start && arc_lengths[i] < t_end {
            result_pts.push(points[i]);
            result_elevs.push(elevations[i]);
        }
    }

    // End point.
    result_pts.push(point_at_arc_length(points, &arc_lengths, t_end));
    result_elevs.push(elevation_at_arc_length(elevations, &arc_lengths, t_end));

    (result_pts, result_elevs)
}

/// Returns the interpolated elevation at a given arc length along a polyline.
fn elevation_at_arc_length(elevations: &[f32], arc_lengths: &[f32], target: f32) -> f32 {
    for i in 1..elevations.len() {
        if arc_lengths[i] >= target {
            let seg_len = arc_lengths[i] - arc_lengths[i - 1];
            if seg_len < 1e-6 {
                return elevations[i];
            }
            let t = (target - arc_lengths[i - 1]) / seg_len;
            return elevations[i - 1] + t * (elevations[i] - elevations[i - 1]);
        }
    }
    *elevations.last().unwrap()
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

/// Extrudes a 2D polyline into a flat asphalt ribbon and tapered embankment
/// skirt meshes.
///
/// **Asphalt**: Both left and right edges share the centerline height, producing
/// a perfectly flat, horizontal driving surface.
///
/// **Skirts**: Two strips (one per side) that attach to the asphalt edge and
/// taper outward and downward to the terrain surface minus `bury_depth`,
/// creating a smooth cut-and-fill embankment.
fn extrude_ribbon(
    points: &[Vec2],
    elevations: &[f32],
    half_width: f32,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> (ProceduralMesh, ProceduralMesh) {
    let n = points.len();
    let skirt_w = config.skirt.width;
    let bury = config.skirt.bury_depth;

    // --- Asphalt (flat surface) ---
    let mut asphalt = ProceduralMesh {
        vertices: Vec::with_capacity(n * 2),
        normals: Vec::with_capacity(n * 2),
        uvs: Vec::with_capacity(n * 2),
        indices: Vec::with_capacity((n - 1) * 6),
    };

    // --- Skirts (left + right embankment strips) ---
    // Each side has n * 2 vertices (inner edge at road height, outer edge at terrain).
    let mut skirts = ProceduralMesh {
        vertices: Vec::with_capacity(n * 4),
        normals: Vec::with_capacity(n * 4),
        uvs: Vec::with_capacity(n * 4),
        indices: Vec::with_capacity((n - 1) * 12),
    };

    let mut accum_dist = 0.0f32;

    for i in 0..n {
        let tangent = if i == 0 {
            (points[1] - points[0]).normalize_or_zero()
        } else if i == n - 1 {
            (points[n - 1] - points[n - 2]).normalize_or_zero()
        } else {
            (points[i + 1] - points[i - 1]).normalize_or_zero()
        };

        let right = Vec2::new(-tangent.y, tangent.x);

        let left_pt = points[i] - right * half_width;
        let right_pt = points[i] + right * half_width;

        // Centerline height — sovereign elevation from rationalized graph.
        let center_y = elevations[i] + config.depth_bias;

        // Compute slope-aware normal via cross product of 3D forward and right.
        let elev_delta = if i == 0 {
            elevations[1] - elevations[0]
        } else if i == n - 1 {
            elevations[n - 1] - elevations[n - 2]
        } else {
            elevations[i + 1] - elevations[i - 1]
        };
        let forward_3d = Vec3::new(tangent.x, elev_delta, tangent.y).normalize_or_zero();
        let right_3d = Vec3::new(right.x, 0.0, right.y);
        let normal = right_3d.cross(forward_3d).normalize_or_zero();
        // Ensure normal points upward; flip if necessary.
        let normal = if normal.y < 0.0 { -normal } else { normal };
        let norm_arr = [normal.x, normal.y, normal.z];

        // Asphalt vertices: right (idx 2i), left (idx 2i+1).
        asphalt.vertices.push([right_pt.x, center_y, right_pt.y]);
        asphalt.vertices.push([left_pt.x, center_y, left_pt.y]);
        asphalt.normals.push(norm_arr);
        asphalt.normals.push(norm_arr);

        if i > 0 {
            accum_dist += (points[i] - points[i - 1]).length();
        }
        let u = accum_dist * config.texture_scale;
        asphalt.uvs.push([u, 0.0]);
        asphalt.uvs.push([u, 1.0]);

        // Skirt geometry — 4 vertices per cross-section:
        //   [0] right inner (at road edge, road height)
        //   [1] right outer (skirt_w outward, terrain - bury)
        //   [2] left inner  (at road edge, road height)
        //   [3] left outer  (skirt_w outward, terrain - bury)
        let right_outer_pt = points[i] + right * (half_width + skirt_w);
        let left_outer_pt = points[i] - right * (half_width + skirt_w);

        let right_outer_y =
            heightmap.get_height_at(right_outer_pt.x, right_outer_pt.y) - bury;
        let left_outer_y =
            heightmap.get_height_at(left_outer_pt.x, left_outer_pt.y) - bury;

        // Right skirt: inner then outer.
        skirts.vertices.push([right_pt.x, center_y, right_pt.y]);
        skirts.vertices.push([right_outer_pt.x, right_outer_y, right_outer_pt.y]);
        // Left skirt: inner then outer.
        skirts.vertices.push([left_pt.x, center_y, left_pt.y]);
        skirts.vertices.push([left_outer_pt.x, left_outer_y, left_outer_pt.y]);

        skirts.normals.push([0.0, 1.0, 0.0]);
        skirts.normals.push([0.0, 1.0, 0.0]);
        skirts.normals.push([0.0, 1.0, 0.0]);
        skirts.normals.push([0.0, 1.0, 0.0]);

        let skirt_u = u;
        skirts.uvs.push([skirt_u, 0.0]);
        skirts.uvs.push([skirt_u, 1.0]);
        skirts.uvs.push([skirt_u, 0.0]);
        skirts.uvs.push([skirt_u, 1.0]);
    }

    // Asphalt index buffer.
    for i in 0..n as u32 - 1 {
        let bl = i * 2;
        let br = i * 2 + 1;
        let tl = (i + 1) * 2;
        let tr = (i + 1) * 2 + 1;

        asphalt.indices.push(bl);
        asphalt.indices.push(tl);
        asphalt.indices.push(br);

        asphalt.indices.push(br);
        asphalt.indices.push(tl);
        asphalt.indices.push(tr);
    }

    // Skirt index buffer — two quads per segment (right side + left side).
    for i in 0..n as u32 - 1 {
        let base = i * 4;
        let next = (i + 1) * 4;

        // Right skirt quad: inner=[base+0], outer=[base+1].
        // Winding produces outward+upward normals on the right side.
        let ri0 = base;
        let ro0 = base + 1;
        let ri1 = next;
        let ro1 = next + 1;

        skirts.indices.push(ri0);
        skirts.indices.push(ro0);
        skirts.indices.push(ri1);
        skirts.indices.push(ro0);
        skirts.indices.push(ro1);
        skirts.indices.push(ri1);

        // Left skirt quad: inner=[base+2], outer=[base+3].
        // Winding is reversed so face normal points outward (left).
        let li0 = base + 2;
        let lo0 = base + 3;
        let li1 = next + 2;
        let lo1 = next + 3;

        skirts.indices.push(li0);
        skirts.indices.push(li1);
        skirts.indices.push(lo0);
        skirts.indices.push(lo0);
        skirts.indices.push(li1);
        skirts.indices.push(lo1);
    }

    (asphalt, skirts)
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
        let (hub, hub_skirt) = generate_hub(&g, 0, degrees[0], &hm, &config);
        // 1 center + hub_sides perimeter vertices.
        assert_eq!(hub.vertices.len(), (1 + config.hub_sides) as usize);
        assert_eq!(hub.indices.len(), (config.hub_sides * 3) as usize);
        // Skirt: 2 vertices per perimeter side (inner + outer).
        assert_eq!(hub_skirt.vertices.len(), (config.hub_sides * 2) as usize);
        // 2 triangles per quad = 6 indices per side.
        assert_eq!(hub_skirt.indices.len(), (config.hub_sides * 6) as usize);
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
        let elevs = vec![0.0, 5.0, 10.0];
        let (truncated, trunc_elevs) = truncate_polyline_with_elevations(&pts, &elevs, 3.0, 3.0);
        assert!(!truncated.is_empty());
        // First point should be at x≈3.
        assert!((truncated[0].x - 3.0).abs() < 1e-4);
        // Last point should be at x≈17.
        assert!((truncated.last().unwrap().x - 17.0).abs() < 1e-4);
        // Elevations should be interpolated: at x=3 → 1.5, at x=17 → 8.5.
        assert!((trunc_elevs[0] - 1.5).abs() < 1e-4);
        assert!((*trunc_elevs.last().unwrap() - 8.5).abs() < 1e-4);
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
