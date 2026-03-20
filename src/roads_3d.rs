//! Engine-agnostic 3D road mesh generation.
//!
//! Takes a [`RoadGraph`] and [`HeightMap`] and produces [`ProceduralMesh`]
//! vertex buffers for hub (intersection) and ribbon (street) geometry.
//!
//! **Hubs (degree 3+)** are procedural polygons derived from the 2D
//! intersection of incoming road boundaries — each ribbon is truncated
//! exactly where adjacent road boundaries meet, and the hub fills the
//! remaining polygon with a triangle fan. Skirts are generated only in
//! the angular gaps between roads.
//!
//! **Dead-end caps (degree 1)** retain the legacy rounded N-gon style.
//!
//! **Ribbons** are extruded strips that follow chains of degree-2 nodes,
//! truncated at hub / cap boundaries using the truncation map.

use std::collections::HashMap;

use glam::{Vec2, Vec3};
use symbios_ground::HeightMap;

use crate::graph::{EdgeId, NodeId, RoadGraph, RoadType};
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
    /// Number of sides for dead-end cap polygons (e.g. 8 = octagon).
    pub hub_sides: u32,
    /// Depth bias: vertices are raised above the terrain by this amount to
    /// prevent z-fighting.
    pub depth_bias: f32,
    /// UV texture scale: world units per texture repeat.
    pub texture_scale: f32,
    /// Legacy: subdivisions per graph edge for Catmull-Rom spline ribbons.
    /// Ignored when the graph has been rationalized (geometry is already smooth).
    pub spline_subdivisions: u32,
    /// Extra radius added to dead-end caps beyond the road half-width,
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

    let degrees = compute_active_degrees(graph);
    let truncations = compute_truncations(graph, &degrees, config);

    // --- Hubs (degree != 2) ---
    for (node_id, &deg) in degrees.iter().enumerate() {
        if deg == 0 || deg == 2 {
            continue;
        }
        let nid = node_id as NodeId;
        if deg == 1 {
            let (hub, hub_skirt) = generate_hub_cap(graph, nid, heightmap, config);
            meshes.hubs.append(&hub);
            meshes.skirts.append(&hub_skirt);
        } else {
            let (hub, hub_skirt) =
                generate_hub_procedural(graph, nid, &truncations, heightmap, config);
            meshes.hubs.append(&hub);
            meshes.skirts.append(&hub_skirt);
        }
    }

    // --- Ribbons (chains of degree-2 nodes) ---
    let chains = extract_chains(graph, &degrees);
    for chain in &chains {
        let (ribbon, skirt) = generate_ribbon(graph, chain, &truncations, heightmap, config);
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
// Truncation computation
// ---------------------------------------------------------------------------

/// Computes per-edge truncation distances at every non-degree-2 node.
///
/// For dead ends (degree 1), truncation = half_width + curb_radius (cap radius).
///
/// For intersections (degree 3+), incoming edges are sorted radially and
/// adjacent boundary lines are intersected to find the exact distance along
/// each edge's centerline where the ribbon must be cut. The boundary width
/// is `half_width + skirt_width` so that neither asphalt nor skirt overlaps.
fn compute_truncations(
    graph: &RoadGraph,
    degrees: &[u32],
    config: &RoadMeshConfig,
) -> HashMap<(NodeId, EdgeId), f32> {
    let mut truncations = HashMap::new();
    let skirt_w = config.skirt.width;

    for (node_idx, &deg) in degrees.iter().enumerate() {
        if deg == 0 || deg == 2 {
            continue;
        }

        let nid = node_idx as NodeId;
        let center = graph.node_pos(nid);
        let node = &graph.nodes[node_idx];

        // Collect active edges with their geometry.
        let mut arms: Vec<(EdgeId, Vec2, Vec2, f32)> = Vec::new(); // (eid, dir, right, half_width)
        for &eid in &node.edges {
            let edge = &graph.edges[eid as usize];
            if !edge.active {
                continue;
            }
            let neighbor = graph.opposite(eid, nid);
            let dir = (graph.node_pos(neighbor) - center).normalize_or_zero();
            if dir.length_squared() < 1e-12 {
                continue;
            }
            let right = Vec2::new(-dir.y, dir.x);
            let hw = match edge.road_type {
                RoadType::Major => config.major_half_width,
                RoadType::Minor => config.minor_half_width,
            };
            arms.push((eid, dir, right, hw));
        }

        if arms.is_empty() {
            continue;
        }

        // Dead end: simple cap radius.
        if deg == 1 {
            let (eid, _, _, hw) = arms[0];
            truncations.insert((nid, eid), hw + config.curb_radius);
            continue;
        }

        // Intersection (degree 3+): sort arms by angle.
        arms.sort_by(|a, b| {
            let angle_a = (-a.1.y).atan2(a.1.x);
            let angle_b = (-b.1.y).atan2(b.1.x);
            angle_a.partial_cmp(&angle_b).unwrap()
        });

        let n = arms.len();
        // Initialize truncations with a minimum (half_width ensures some volume).
        let mut trunc: Vec<f32> = arms.iter().map(|a| a.3).collect();

        // For each adjacent pair, intersect their outer boundary lines.
        for i in 0..n {
            let j = (i + 1) % n;

            let (_eid_a, dir_a, right_a, hw_a) = arms[i];
            let (_eid_b, dir_b, right_b, hw_b) = arms[j];

            // Total boundary width (asphalt + skirt).
            let w_a = hw_a + skirt_w;
            let w_b = hw_b + skirt_w;

            // Edge A's left boundary: center - right_A * w_A + dir_A * t_A
            // Edge B's right boundary: center + right_B * w_B + dir_B * t_B
            //
            // Setting equal:
            //   -right_A * w_A + dir_A * t_A = right_B * w_B + dir_B * t_B
            //   dir_A * t_A - dir_B * t_B = right_A * w_A + right_B * w_B
            //
            // 2x2 system: [dir_A.x  -dir_B.x] [t_A]   [right_A.x * w_A + right_B.x * w_B]
            //              [dir_A.y  -dir_B.y] [t_B] = [right_A.y * w_A + right_B.y * w_B]
            let rhs = right_a * w_a + right_b * w_b;
            let det = dir_a.x * (-dir_b.y) - (-dir_b.x) * dir_a.y;

            if det.abs() < 1e-6 {
                // Nearly parallel — use a generous fallback.
                let fallback = (w_a + w_b) * 0.5;
                trunc[i] = trunc[i].max(fallback);
                trunc[j] = trunc[j].max(fallback);
                continue;
            }

            let t_a = (rhs.x * (-dir_b.y) - (-dir_b.x) * rhs.y) / det;
            let t_b = (dir_a.x * rhs.y - dir_a.y * rhs.x) / det;

            // Only use positive truncations (intersection is in front).
            if t_a > 0.0 {
                trunc[i] = trunc[i].max(t_a);
            }
            if t_b > 0.0 {
                trunc[j] = trunc[j].max(t_b);
            }
        }

        // Store results.
        for (idx, &(eid, _, _, _)) in arms.iter().enumerate() {
            truncations.insert((nid, eid), trunc[idx]);
        }
    }

    truncations
}

// ---------------------------------------------------------------------------
// Dead-end cap (N-gon, degree 1)
// ---------------------------------------------------------------------------

fn generate_hub_cap(
    graph: &RoadGraph,
    node_id: NodeId,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> (ProceduralMesh, ProceduralMesh) {
    let center = graph.node_pos(node_id);
    let node = &graph.nodes[node_id as usize];

    // Find max half-width of connecting active edges + curb radius.
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
    let center_y = node.elevation + config.depth_bias;

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

    // Fan triangles.
    for i in 0..sides {
        let a = 1 + i;
        let b = 1 + (i + 1) % sides;
        mesh.indices.push(0);
        mesh.indices.push(b);
        mesh.indices.push(a);
    }

    // Skirt ring.
    let skirt_w = config.skirt.width;
    let bury = config.skirt.bury_depth;
    let mut skirt = ProceduralMesh::default();

    for &(px, pz) in &perimeter_pts {
        let dx = px - center.x;
        let dz = pz - center.y;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        let nx = dx / len;
        let nz = dz / len;

        let outer_x = px + nx * skirt_w;
        let outer_z = pz + nz * skirt_w;
        let outer_y = heightmap.get_height_at(outer_x, outer_z) - bury;

        skirt.vertices.push([px, center_y, pz]);
        skirt.normals.push([0.0, 1.0, 0.0]);
        skirt
            .uvs
            .push([px * config.texture_scale, pz * config.texture_scale]);

        skirt.vertices.push([outer_x, outer_y, outer_z]);
        skirt.normals.push([0.0, 1.0, 0.0]);
        skirt.uvs.push([
            outer_x * config.texture_scale,
            outer_z * config.texture_scale,
        ]);
    }

    for i in 0..sides {
        let next = (i + 1) % sides;
        let i0 = i * 2;
        let o0 = i * 2 + 1;
        let i1 = next * 2;
        let o1 = next * 2 + 1;

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
// Procedural intersection hub (degree 3+)
// ---------------------------------------------------------------------------

/// Generates a procedural intersection polygon from the truncated ribbon
/// corners, plus gap-only skirts between adjacent roads.
fn generate_hub_procedural(
    graph: &RoadGraph,
    node_id: NodeId,
    truncations: &HashMap<(NodeId, EdgeId), f32>,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> (ProceduralMesh, ProceduralMesh) {
    let center = graph.node_pos(node_id);
    let node = &graph.nodes[node_id as usize];
    let center_y = node.elevation + config.depth_bias;
    let skirt_w = config.skirt.width;
    let bury = config.skirt.bury_depth;

    // Collect active edges with geometry.
    struct Arm {
        dir: Vec2,
        right: Vec2,
        half_width: f32,
        truncation: f32,
        angle: f32,
    }

    let mut arms: Vec<Arm> = Vec::new();
    for &eid in &node.edges {
        let edge = &graph.edges[eid as usize];
        if !edge.active {
            continue;
        }
        let neighbor = graph.opposite(eid, node_id);
        let dir = (graph.node_pos(neighbor) - center).normalize_or_zero();
        if dir.length_squared() < 1e-12 {
            continue;
        }
        let right = Vec2::new(-dir.y, dir.x);
        let hw = match edge.road_type {
            RoadType::Major => config.major_half_width,
            RoadType::Minor => config.minor_half_width,
        };
        let trunc = truncations
            .get(&(node_id, eid))
            .copied()
            .unwrap_or(hw + config.curb_radius);

        arms.push(Arm {
            dir,
            right,
            half_width: hw,
            truncation: trunc,
            angle: (-dir.y).atan2(dir.x),
        });
    }

    if arms.is_empty() {
        return (ProceduralMesh::default(), ProceduralMesh::default());
    }

    // Sort by angle (CCW).
    arms.sort_by(|a, b| a.angle.partial_cmp(&b.angle).unwrap());

    // Build perimeter: for each arm, emit right corner then left corner.
    // Going CCW, the perimeter order is:
    //   arm[0].right, arm[0].left, arm[1].right, arm[1].left, ...
    let mut perimeter: Vec<Vec2> = Vec::with_capacity(arms.len() * 2);
    for arm in &arms {
        let right_corner = center + arm.dir * arm.truncation + arm.right * arm.half_width;
        let left_corner = center + arm.dir * arm.truncation - arm.right * arm.half_width;
        perimeter.push(right_corner);
        perimeter.push(left_corner);
    }

    // --- Asphalt mesh: triangle fan from center ---
    let mut mesh = ProceduralMesh::default();

    // Center vertex (index 0).
    mesh.vertices.push([center.x, center_y, center.y]);
    mesh.normals.push([0.0, 1.0, 0.0]);
    mesh.uvs.push([
        center.x * config.texture_scale,
        center.y * config.texture_scale,
    ]);

    // Perimeter vertices.
    for pt in &perimeter {
        mesh.vertices.push([pt.x, center_y, pt.y]);
        mesh.normals.push([0.0, 1.0, 0.0]);
        mesh.uvs
            .push([pt.x * config.texture_scale, pt.y * config.texture_scale]);
    }

    // Fan triangles around perimeter (CCW winding for upward-facing +Y normal).
    let peri_count = perimeter.len() as u32;
    for i in 0..peri_count {
        let a = 1 + i;
        let b = 1 + (i + 1) % peri_count;
        mesh.indices.push(0);
        mesh.indices.push(a);
        mesh.indices.push(b);
    }

    // --- Skirt mesh: quads only in angular gaps between adjacent roads ---
    let mut skirt = ProceduralMesh::default();
    let n_arms = arms.len();

    for i in 0..n_arms {
        let j = (i + 1) % n_arms;

        // Gap runs from arm[i]'s left corner to arm[j]'s right corner.
        let left_corner =
            center + arms[i].dir * arms[i].truncation - arms[i].right * arms[i].half_width;
        let right_corner =
            center + arms[j].dir * arms[j].truncation + arms[j].right * arms[j].half_width;

        // Angular span of this gap.
        let left_angle = arms[i].angle + std::f32::consts::FRAC_PI_2; // left side angle
        let right_angle = arms[j].angle - std::f32::consts::FRAC_PI_2; // right side angle

        // Compute the angular gap; handle wrap-around.
        let mut gap_angle = right_angle - left_angle;
        if gap_angle < 0.0 {
            gap_angle += std::f32::consts::TAU;
        }
        if gap_angle > std::f32::consts::TAU {
            gap_angle -= std::f32::consts::TAU;
        }

        // Number of subdivisions for a smooth arc (at least 1).
        let subdivs = ((gap_angle / (std::f32::consts::FRAC_PI_4)).ceil() as u32).max(1);

        // Generate arc points by interpolating between the two corners.
        let base_vert = skirt.vertices.len() as u32;

        for s in 0..=subdivs {
            let t = s as f32 / subdivs as f32;

            // Linearly interpolate inner point along the gap.
            let inner = left_corner.lerp(right_corner, t);

            // Outer direction: from center through inner, pushed out by skirt_w.
            let out_dir = (inner - center).normalize_or_zero();
            let outer = inner + out_dir * skirt_w;
            let outer_y = heightmap.get_height_at(outer.x, outer.y) - bury;

            skirt.vertices.push([inner.x, center_y, inner.y]);
            skirt.normals.push([0.0, 1.0, 0.0]);
            skirt.uvs.push([
                inner.x * config.texture_scale,
                inner.y * config.texture_scale,
            ]);

            skirt.vertices.push([outer.x, outer_y, outer.y]);
            skirt.normals.push([0.0, 1.0, 0.0]);
            skirt.uvs.push([
                outer.x * config.texture_scale,
                outer.y * config.texture_scale,
            ]);
        }

        // Quad strip: connect adjacent cross-sections.
        for s in 0..subdivs {
            let i0 = base_vert + s * 2; // inner current
            let o0 = base_vert + s * 2 + 1; // outer current
            let i1 = base_vert + (s + 1) * 2;
            let o1 = base_vert + (s + 1) * 2 + 1;

            skirt.indices.push(i0);
            skirt.indices.push(i1);
            skirt.indices.push(o0);

            skirt.indices.push(o0);
            skirt.indices.push(i1);
            skirt.indices.push(o1);
        }
    }

    (mesh, skirt)
}

// ---------------------------------------------------------------------------
// Chain extraction (delegated to topology module)
// ---------------------------------------------------------------------------

/// Local alias — the ribbon generator needs nodes, road type, and edge IDs.
struct Chain {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    road_type: RoadType,
}

fn extract_chains(graph: &RoadGraph, degrees: &[u32]) -> Vec<Chain> {
    topology::extract_chains(graph, degrees)
        .into_iter()
        .map(|c| Chain {
            nodes: c.nodes,
            edges: c.edges,
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
    truncations: &HashMap<(NodeId, EdgeId), f32>,
    heightmap: &HeightMap,
    config: &RoadMeshConfig,
) -> (ProceduralMesh, ProceduralMesh) {
    let half_width = match chain.road_type {
        RoadType::Major => config.major_half_width,
        RoadType::Minor => config.minor_half_width,
    };

    let smooth_pts: Vec<Vec2> = chain.nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
    let node_elevs: Vec<f32> = chain
        .nodes
        .iter()
        .map(|&nid| graph.nodes[nid as usize].elevation)
        .collect();
    if smooth_pts.len() < 2 {
        return (ProceduralMesh::default(), ProceduralMesh::default());
    }

    // Look up truncation at each endpoint from the precomputed map.
    let first_node = chain.nodes[0];
    let last_node = *chain.nodes.last().unwrap();
    let first_edge = chain.edges[0];
    let last_edge = *chain.edges.last().unwrap();

    let start_trim = truncations
        .get(&(first_node, first_edge))
        .copied()
        .unwrap_or(0.0);
    let end_trim = truncations
        .get(&(last_node, last_edge))
        .copied()
        .unwrap_or(0.0);

    let (truncated, truncated_elevs) =
        truncate_polyline_with_elevations(&smooth_pts, &node_elevs, start_trim, end_trim);
    if truncated.len() < 2 {
        return (ProceduralMesh::default(), ProceduralMesh::default());
    }

    extrude_ribbon(&truncated, &truncated_elevs, half_width, heightmap, config)
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

    result_pts.push(point_at_arc_length(points, &arc_lengths, t_start));
    result_elevs.push(elevation_at_arc_length(elevations, &arc_lengths, t_start));

    for i in 1..points.len() - 1 {
        if arc_lengths[i] > t_start && arc_lengths[i] < t_end {
            result_pts.push(points[i]);
            result_elevs.push(elevations[i]);
        }
    }

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

    let mut asphalt = ProceduralMesh {
        vertices: Vec::with_capacity(n * 2),
        normals: Vec::with_capacity(n * 2),
        uvs: Vec::with_capacity(n * 2),
        indices: Vec::with_capacity((n - 1) * 6),
    };

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

        let center_y = elevations[i] + config.depth_bias;

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
        let normal = if normal.y < 0.0 { -normal } else { normal };
        let norm_arr = [normal.x, normal.y, normal.z];

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

        let right_outer_pt = points[i] + right * (half_width + skirt_w);
        let left_outer_pt = points[i] - right * (half_width + skirt_w);

        let right_outer_y = heightmap.get_height_at(right_outer_pt.x, right_outer_pt.y) - bury;
        let left_outer_y = heightmap.get_height_at(left_outer_pt.x, left_outer_pt.y) - bury;

        skirts.vertices.push([right_pt.x, center_y, right_pt.y]);
        skirts
            .vertices
            .push([right_outer_pt.x, right_outer_y, right_outer_pt.y]);
        skirts.vertices.push([left_pt.x, center_y, left_pt.y]);
        skirts
            .vertices
            .push([left_outer_pt.x, left_outer_y, left_outer_pt.y]);

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

    // Skirt index buffer.
    for i in 0..n as u32 - 1 {
        let base = i * 4;
        let next = (i + 1) * 4;

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
    fn hub_cap_mesh_has_correct_vertex_count() {
        // Build a dead-end node (degree 1).
        let mut g = RoadGraph::default();
        let a = g.add_node(Vec2::new(50.0, 50.0));
        let b = g.add_node(Vec2::new(80.0, 50.0));
        g.add_edge(a, b, RoadType::Major);

        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();

        // Node `a` has degree 1 → cap.
        let (hub, hub_skirt) = generate_hub_cap(&g, a, &hm, &config);
        let sides = config.hub_sides;
        assert_eq!(hub.vertices.len(), (1 + sides) as usize);
        assert_eq!(hub.indices.len(), (sides * 3) as usize);
        assert_eq!(hub_skirt.vertices.len(), (sides * 2) as usize);
        assert_eq!(hub_skirt.indices.len(), (sides * 6) as usize);
    }

    #[test]
    fn procedural_hub_has_correct_vertex_count() {
        let g = cross_graph();
        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();
        let degrees = compute_active_degrees(&g);
        let truncations = compute_truncations(&g, &degrees, &config);

        // Center node (0) has degree 4 → procedural hub.
        let (hub, _hub_skirt) = generate_hub_procedural(&g, 0, &truncations, &hm, &config);
        // 1 center + 4 arms × 2 corners = 9 vertices.
        assert_eq!(hub.vertices.len(), 1 + 4 * 2);
        // 8 fan triangles × 3 indices = 24.
        assert_eq!(hub.indices.len(), 8 * 3);
    }

    #[test]
    fn truncations_computed_for_all_edges() {
        let g = cross_graph();
        let degrees = compute_active_degrees(&g);
        let config = RoadMeshConfig::default();
        let truncations = compute_truncations(&g, &degrees, &config);

        // Center node (0) has 4 active edges.
        for eid in 0..4u32 {
            assert!(
                truncations.contains_key(&(0, eid)),
                "truncation missing for (0, {eid})"
            );
            let t = truncations[&(0, eid)];
            assert!(t > 0.0, "truncation should be positive, got {t}");
        }

        // Leaf nodes (1..4) each have 1 edge → dead-end truncation.
        for nid in 1..5u32 {
            let node = &g.nodes[nid as usize];
            for &eid in &node.edges {
                assert!(
                    truncations.contains_key(&(nid, eid)),
                    "truncation missing for ({nid}, {eid})"
                );
            }
        }
    }

    #[test]
    fn ribbon_mesh_nonempty() {
        let mut g = RoadGraph::default();
        let a = g.add_node(Vec2::new(10.0, 10.0));
        let b = g.add_node(Vec2::new(90.0, 10.0));
        g.add_edge(a, b, RoadType::Major);

        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();
        let meshes = generate_road_meshes(&g, &hm, &config);

        assert!(!meshes.hubs.vertices.is_empty());
        assert!(!meshes.ribbons.vertices.is_empty());
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
        assert!((truncated[0].x - 3.0).abs() < 1e-4);
        assert!((truncated.last().unwrap().x - 17.0).abs() < 1e-4);
        assert!((trunc_elevs[0] - 1.5).abs() < 1e-4);
        assert!((*trunc_elevs.last().unwrap() - 8.5).abs() < 1e-4);
    }

    #[test]
    fn full_pipeline_cross_graph() {
        let g = cross_graph();
        let hm = flat_heightmap();
        let config = RoadMeshConfig::default();

        let meshes = generate_road_meshes(&g, &hm, &config);

        let degrees = compute_active_degrees(&g);
        let hub_count = degrees.iter().filter(|&&d| d > 0 && d != 2).count();
        assert_eq!(hub_count, 5);

        assert!(!meshes.hubs.vertices.is_empty());
        assert!(!meshes.ribbons.vertices.is_empty());
    }
}
