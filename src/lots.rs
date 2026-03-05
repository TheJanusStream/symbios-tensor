use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::geometry::segment_intersection;
use crate::graph::RoadGraph;

/// Configuration for lot subdivision and building footprint extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotConfig {
    /// Maximum lot area before recursive subdivision (sqm).
    pub max_lot_area: f32,
    /// Minimum lot area — polygons below this are discarded.
    pub min_lot_area: f32,
    /// Distance from the street edge to the building front.
    pub front_setback: f32,
    /// Distance from side edges to the building sides.
    pub side_setback: f32,
    /// Distance from the back edge to the building rear.
    pub rear_setback: f32,
    /// Minimum building width (along street).
    pub min_width: f32,
    /// Minimum building depth (perpendicular to street).
    pub min_depth: f32,
}

impl Default for LotConfig {
    fn default() -> Self {
        Self {
            max_lot_area: 400.0,
            min_lot_area: 50.0,
            front_setback: 3.0,
            side_setback: 1.5,
            rear_setback: 2.0,
            min_width: 6.0,
            min_depth: 6.0,
        }
    }
}

/// A rectangular building footprint aligned to the nearest street edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingLot {
    /// World-space center of the footprint.
    pub position: Vec2,
    /// Rotation angle in radians (around the Y axis in 3D; around Z in 2D top-down).
    pub rotation: f32,
    /// Extent along the street frontage.
    pub width: f32,
    /// Extent perpendicular to the street.
    pub depth: f32,
}

/// Extracts building lots from city blocks in the road graph.
///
/// Each block polygon is recursively subdivided until pieces are below
/// `config.max_lot_area`, then a street-aligned inscribed rectangle is
/// computed with setbacks applied.
pub fn extract_lots(graph: &RoadGraph, config: &LotConfig) -> Vec<BuildingLot> {
    let mut lots = Vec::new();
    for block in &graph.blocks {
        let polygon: Vec<Vec2> = block
            .perimeter
            .iter()
            .map(|&nid| graph.node_pos(nid))
            .collect();

        let sub_polys = subdivide_polygon(&polygon, config.max_lot_area, config.min_lot_area, 10);

        for poly in sub_polys {
            if let Some(lot) = polygon_to_lot(&poly, &polygon, config) {
                lots.push(lot);
            }
        }
    }
    lots
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn polygon_area(vertices: &[Vec2]) -> f32 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f32;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    (area * 0.5).abs()
}

fn polygon_centroid(vertices: &[Vec2]) -> Vec2 {
    let n = vertices.len();
    if n == 0 {
        return Vec2::ZERO;
    }
    if n < 3 {
        return vertices.iter().copied().sum::<Vec2>() / n as f32;
    }
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let mut signed_area_2 = 0.0_f32;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let cross = a.x * b.y - b.x * a.y;
        cx += (a.x + b.x) * cross;
        cy += (a.y + b.y) * cross;
        signed_area_2 += cross;
    }
    if signed_area_2.abs() < 1e-8 {
        return vertices.iter().copied().sum::<Vec2>() / n as f32;
    }
    let inv = 1.0 / (3.0 * signed_area_2);
    Vec2::new(cx * inv, cy * inv)
}

fn longest_edge_index(vertices: &[Vec2]) -> usize {
    let n = vertices.len();
    let mut best_idx = 0;
    let mut best_len_sq = 0.0_f32;
    for i in 0..n {
        let len_sq = (vertices[(i + 1) % n] - vertices[i]).length_squared();
        if len_sq > best_len_sq {
            best_len_sq = len_sq;
            best_idx = i;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Polygon splitting
// ---------------------------------------------------------------------------

fn split_polygon_by_line(
    poly: &[Vec2],
    line_origin: Vec2,
    line_dir: Vec2,
) -> Option<(Vec<Vec2>, Vec<Vec2>)> {
    let n = poly.len();
    let half_extent = 100_000.0;
    let line_a = line_origin - line_dir * half_extent;
    let line_b = line_origin + line_dir * half_extent;

    let mut intersections: Vec<(usize, Vec2)> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(pt) = segment_intersection(line_a, line_b, poly[i], poly[j]) {
            let dominated = intersections.iter().any(|(_, p)| p.distance(pt) < 1e-4);
            if !dominated {
                intersections.push((i, pt));
            }
        }
    }

    if intersections.len() != 2 {
        return None;
    }

    intersections.sort_by_key(|(idx, _)| *idx);
    let (idx_a, pt_a) = intersections[0];
    let (idx_b, pt_b) = intersections[1];

    // Poly A: pt_a -> vertices (idx_a+1)..=idx_b -> pt_b
    let mut poly_a = vec![pt_a];
    for vertex in &poly[(idx_a + 1)..=idx_b] {
        poly_a.push(*vertex);
    }
    poly_a.push(pt_b);

    // Poly B: pt_b -> vertices after idx_b wrapping to idx_a -> pt_a
    let mut poly_b = vec![pt_b];
    let mut k = (idx_b + 1) % n;
    loop {
        poly_b.push(poly[k]);
        if k == idx_a {
            break;
        }
        k = (k + 1) % n;
    }
    poly_b.push(pt_a);

    // Filter degenerate results
    if poly_a.len() < 3 || poly_b.len() < 3 {
        return None;
    }

    Some((poly_a, poly_b))
}

fn subdivide_polygon(
    poly: &[Vec2],
    max_area: f32,
    min_area: f32,
    depth_limit: u32,
) -> Vec<Vec<Vec2>> {
    let area = polygon_area(poly);

    if area <= max_area || area <= min_area * 2.0 || depth_limit == 0 {
        return vec![poly.to_vec()];
    }

    let longest = longest_edge_index(poly);
    let n = poly.len();
    let edge_a = poly[longest];
    let edge_b = poly[(longest + 1) % n];
    let edge_dir = (edge_b - edge_a).normalize();

    // Perpendicular to longest edge, through centroid
    let perp = Vec2::new(-edge_dir.y, edge_dir.x);
    let centroid = polygon_centroid(poly);

    match split_polygon_by_line(poly, centroid, perp) {
        Some((left, right)) => {
            let mut result = Vec::new();
            result.extend(subdivide_polygon(&left, max_area, min_area, depth_limit - 1));
            result.extend(subdivide_polygon(&right, max_area, min_area, depth_limit - 1));
            result
        }
        None => vec![poly.to_vec()],
    }
}

// ---------------------------------------------------------------------------
// Frontage + inscribed box + setbacks
// ---------------------------------------------------------------------------

/// Finds the frontage edge: the longest edge that lies on the original block
/// perimeter (a street edge). Falls back to the longest edge overall if no
/// boundary edge is found (e.g. heavily subdivided interiors).
fn find_frontage(poly: &[Vec2], perimeter: &[Vec2]) -> (usize, f32) {
    let n = poly.len();
    let mut best_idx = 0;
    let mut best_len = 0.0_f32;
    let mut best_boundary_idx = 0;
    let mut best_boundary_len = 0.0_f32;

    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let len = (b - a).length();

        if len > best_len {
            best_len = len;
            best_idx = i;
        }
        if edge_on_perimeter(a, b, perimeter) && len > best_boundary_len {
            best_boundary_len = len;
            best_boundary_idx = i;
        }
    }

    if best_boundary_len > 0.0 {
        (best_boundary_idx, best_boundary_len)
    } else {
        (best_idx, best_len)
    }
}

/// Returns true if both endpoints of edge (a, b) lie on some edge of `perimeter`.
fn edge_on_perimeter(a: Vec2, b: Vec2, perimeter: &[Vec2]) -> bool {
    let m = perimeter.len();
    for j in 0..m {
        let pa = perimeter[j];
        let pb = perimeter[(j + 1) % m];
        if point_on_segment(a, pa, pb) && point_on_segment(b, pa, pb) {
            return true;
        }
    }
    false
}

fn point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> bool {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return p.distance(a) < 1e-3;
    }
    let t = (p - a).dot(ab) / len_sq;
    if !(-1e-3..=1.0 + 1e-3).contains(&t) {
        return false;
    }
    let proj = a + ab * t.clamp(0.0, 1.0);
    p.distance(proj) < 1e-3
}

fn inscribed_box(poly: &[Vec2], frontage_idx: usize) -> Option<(Vec2, f32, f32, f32)> {
    let n = poly.len();
    if n < 3 {
        return None;
    }

    let fa = poly[frontage_idx];
    let fb = poly[(frontage_idx + 1) % n];

    let street_dir = (fb - fa).normalize();
    let mut inward_dir = Vec2::new(-street_dir.y, street_dir.x);

    // Ensure inward points toward polygon interior
    let centroid = polygon_centroid(poly);
    if (centroid - fa).dot(inward_dir) < 0.0 {
        inward_dir = -inward_dir;
    }

    let rotation = street_dir.y.atan2(street_dir.x);
    let width = (fb - fa).length();

    // Cast rays inward from several points along the frontage edge to find
    // the minimum depth before hitting the opposite polygon boundary.
    let num_samples = 7;
    let mut min_depth = f32::MAX;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let ray_origin = fa.lerp(fb, t);
        let ray_end = ray_origin + inward_dir * 10_000.0;

        let mut closest_dist = f32::MAX;
        for j in 0..n {
            if j == frontage_idx {
                continue;
            }
            let ea = poly[j];
            let eb = poly[(j + 1) % n];
            if let Some(hit) = segment_intersection(ray_origin, ray_end, ea, eb) {
                let d = (hit - ray_origin).dot(inward_dir);
                if d > 1e-4 && d < closest_dist {
                    closest_dist = d;
                }
            }
        }
        if closest_dist < min_depth {
            min_depth = closest_dist;
        }
    }

    if min_depth <= 0.0 || min_depth == f32::MAX {
        return None;
    }

    // Verify side edges: cast rays along ±street_dir from the back corners
    // to clamp width so the rectangle stays inside the polygon.
    let back_center = (fa + fb) * 0.5 + inward_dir * min_depth;
    let mut half_width_limit = width * 0.5;

    for &sign in &[1.0_f32, -1.0] {
        let ray_origin = back_center;
        let ray_end = ray_origin + street_dir * sign * 10_000.0;
        let mut closest = f32::MAX;
        for j in 0..n {
            let ea = poly[j];
            let eb = poly[(j + 1) % n];
            if let Some(hit) = segment_intersection(ray_origin, ray_end, ea, eb) {
                let d = (hit - ray_origin).dot(street_dir * sign);
                if d > 1e-4 && d < closest {
                    closest = d;
                }
            }
        }
        if closest < half_width_limit {
            half_width_limit = closest;
        }
    }

    let width = half_width_limit * 2.0;

    let street_midpoint = (fa + fb) * 0.5;
    let center = street_midpoint + inward_dir * (min_depth * 0.5);

    Some((center, rotation, width, min_depth))
}

fn apply_setbacks(
    center: Vec2,
    rotation: f32,
    width: f32,
    depth: f32,
    config: &LotConfig,
) -> Option<BuildingLot> {
    let new_width = width - 2.0 * config.side_setback.max(0.0);
    let new_depth = depth - config.front_setback.max(0.0) - config.rear_setback.max(0.0);

    if new_width < config.min_width || new_depth < config.min_depth {
        return None;
    }

    // Shift center inward by (front - rear) / 2 to account for asymmetric setbacks
    let street_dir = Vec2::new(rotation.cos(), rotation.sin());
    let inward_dir = Vec2::new(-street_dir.y, street_dir.x);
    let depth_shift = (config.front_setback - config.rear_setback) * 0.5;
    let adjusted_center = center + inward_dir * depth_shift;

    Some(BuildingLot {
        position: adjusted_center,
        rotation,
        width: new_width,
        depth: new_depth,
    })
}

fn polygon_to_lot(poly: &[Vec2], perimeter: &[Vec2], config: &LotConfig) -> Option<BuildingLot> {
    if poly.len() < 3 {
        return None;
    }
    let area = polygon_area(poly);
    if area < config.min_lot_area {
        return None;
    }

    let (frontage_idx, _) = find_frontage(poly, perimeter);
    let (center, rotation, width, depth) = inscribed_box(poly, frontage_idx)?;
    apply_setbacks(center, rotation, width, depth, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RoadGraph;

    fn square(x: f32, y: f32, size: f32) -> Vec<Vec2> {
        vec![
            Vec2::new(x, y),
            Vec2::new(x + size, y),
            Vec2::new(x + size, y + size),
            Vec2::new(x, y + size),
        ]
    }

    #[test]
    fn polygon_area_square() {
        let sq = square(0.0, 0.0, 10.0);
        let area = polygon_area(&sq);
        assert!((area - 100.0).abs() < 1e-3);
    }

    #[test]
    fn polygon_centroid_unit_square() {
        let sq = square(0.0, 0.0, 1.0);
        let c = polygon_centroid(&sq);
        assert!((c.x - 0.5).abs() < 1e-3);
        assert!((c.y - 0.5).abs() < 1e-3);
    }

    #[test]
    fn split_rectangle_into_halves() {
        // 20x10 rectangle, longest edge is along X (20 units)
        let rect = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(20.0, 0.0),
            Vec2::new(20.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let longest = longest_edge_index(&rect);
        let edge_a = rect[longest];
        let edge_b = rect[(longest + 1) % rect.len()];
        let edge_dir = (edge_b - edge_a).normalize();
        let perp = Vec2::new(-edge_dir.y, edge_dir.x);
        let centroid = polygon_centroid(&rect);

        let result = split_polygon_by_line(&rect, centroid, perp);
        assert!(result.is_some());
        let (a, b) = result.unwrap();
        let area_a = polygon_area(&a);
        let area_b = polygon_area(&b);
        assert!((area_a - 100.0).abs() < 1.0, "half A should be ~100 sqm, got {area_a}");
        assert!((area_b - 100.0).abs() < 1.0, "half B should be ~100 sqm, got {area_b}");
    }

    #[test]
    fn subdivide_large_block() {
        let big = square(0.0, 0.0, 40.0); // 1600 sqm
        let sub = subdivide_polygon(&big, 400.0, 50.0, 10);
        assert!(sub.len() >= 4, "1600 sqm block should yield at least 4 lots, got {}", sub.len());
        for poly in &sub {
            let area = polygon_area(poly);
            assert!(area <= 400.0 + 1.0, "sub-polygon area {area} exceeds max");
        }
    }

    #[test]
    fn inscribed_box_rectangle() {
        let rect = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 5.0),
            Vec2::new(0.0, 5.0),
        ];
        let (frontage_idx, _) = find_frontage(&rect, &rect);
        let result = inscribed_box(&rect, frontage_idx);
        assert!(result.is_some());
        let (center, _rotation, width, depth) = result.unwrap();
        assert!((width - 10.0).abs() < 0.5, "width should be ~10, got {width}");
        assert!((depth - 5.0).abs() < 0.5, "depth should be ~5, got {depth}");
        assert!((center.x - 5.0).abs() < 0.5);
        assert!((center.y - 2.5).abs() < 0.5);
    }

    #[test]
    fn setbacks_filter_tiny_lots() {
        let config = LotConfig {
            min_width: 6.0,
            min_depth: 6.0,
            side_setback: 1.5,
            front_setback: 3.0,
            rear_setback: 2.0,
            ..Default::default()
        };
        // Width 5 - 2*1.5 = 2 < 6 → filtered
        let result = apply_setbacks(Vec2::ZERO, 0.0, 5.0, 20.0, &config);
        assert!(result.is_none());
    }

    #[test]
    fn extract_lots_empty_graph() {
        let graph = RoadGraph::default();
        let lots = extract_lots(&graph, &LotConfig::default());
        assert!(lots.is_empty());
    }
}
