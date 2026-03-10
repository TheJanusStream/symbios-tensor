//! Building lot extraction from city blocks.
//!
//! Each [`CityBlock`](crate::graph::CityBlock) polygon is recursively split
//! perpendicular to its longest edge (through the centroid) until sub-polygons
//! fall below a configurable area threshold. A street-aligned inscribed
//! rectangle is then computed for each
//! piece, with front/side/rear setbacks applied to produce the final
//! [`BuildingLot`] footprint.

use glam::Vec2;
use serde::{Deserialize, Serialize};
use symbios_ground::HeightMap;

use crate::geometry::segment_intersection;
use crate::graph::RoadGraph;

/// Minimum distance between consecutive polygon vertices after splitting.
const DEDUP_TOLERANCE: f32 = 1e-4;

/// Epsilon for detecting degenerate zero-area centroids.
const CENTROID_AREA_EPS: f32 = 1e-8;

/// Epsilon for degenerate zero-length segments in point-on-segment tests.
const DEGENERATE_SEG_LEN_SQ: f32 = 1e-10;

/// Tolerance for point-on-segment proximity checks (world units).
const POINT_ON_SEG_TOLERANCE: f32 = 1e-3;

/// Minimum positive ray hit distance to avoid self-intersection artifacts.
const RAY_HIT_EPS: f32 = 1e-4;

/// Maximum bounding-box aspect ratio for polygon subdivision. Polygons
/// whose OBB is more skewed than this are treated as degenerate slivers
/// and skipped to avoid wasting cycles on recursive splitting.
const MAX_SUBDIVISION_ASPECT_RATIO: f32 = 20.0;

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
    /// Midpoint of the street frontage edge (pre-setback), used for road
    /// access queries so that pruning doesn't misidentify the access road.
    pub frontage_center: Vec2,
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
/// computed with setbacks applied. Lots whose center falls below the
/// water level are discarded.
pub fn extract_lots(
    graph: &RoadGraph,
    heightmap: &HeightMap,
    water_level: f32,
    config: &LotConfig,
) -> Vec<BuildingLot> {
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
                let hw = lot.width * 0.5;
                let hd = lot.depth * 0.5;
                let cos = lot.rotation.cos();
                let sin = lot.rotation.sin();

                let corners = [
                    Vec2::new(hw * cos - hd * sin, hw * sin + hd * cos),
                    Vec2::new(hw * cos - (-hd) * sin, hw * sin + (-hd) * cos),
                    Vec2::new(-hw * cos - hd * sin, -hw * sin + hd * cos),
                    Vec2::new(-hw * cos - (-hd) * sin, -hw * sin + (-hd) * cos),
                ];

                let is_above_water = heightmap.get_height_at(lot.position.x, lot.position.y)
                    > water_level
                    && corners.iter().all(|c| {
                        heightmap.get_height_at(lot.position.x + c.x, lot.position.y + c.y)
                            > water_level
                    });

                if is_above_water {
                    lots.push(lot);
                }
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
    // Translate to local origin to avoid f32 cancellation at large coordinates.
    let origin = vertices[0];
    let mut area = 0.0_f32;
    for i in 0..n {
        let a = vertices[i] - origin;
        let b = vertices[(i + 1) % n] - origin;
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
    // Translate to local origin to avoid f32 cancellation at large coordinates.
    let origin = vertices[0];
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    let mut signed_area_2 = 0.0_f32;
    for i in 0..n {
        let a = vertices[i] - origin;
        let b = vertices[(i + 1) % n] - origin;
        let cross = a.x * b.y - b.x * a.y;
        cx += (a.x + b.x) * cross;
        cy += (a.y + b.y) * cross;
        signed_area_2 += cross;
    }
    if signed_area_2.abs() < CENTROID_AREA_EPS {
        return vertices.iter().copied().sum::<Vec2>() / n as f32;
    }
    let inv = 1.0 / (3.0 * signed_area_2);
    Vec2::new(cx * inv, cy * inv) + origin
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

/// Removes consecutive vertices that are closer than [`DEDUP_TOLERANCE`] apart.
fn dedup_consecutive(poly: &mut Vec<Vec2>) {
    poly.dedup_by(|a, b| a.distance(*b) < DEDUP_TOLERANCE);
    // Also check wrap-around (last vs first)
    if poly.len() > 1 && poly.first().unwrap().distance(*poly.last().unwrap()) < DEDUP_TOLERANCE {
        poly.pop();
    }
}

fn split_polygon_by_line(
    poly: &[Vec2],
    line_origin: Vec2,
    line_dir: Vec2,
) -> Option<(Vec<Vec2>, Vec<Vec2>)> {
    let n = poly.len();
    // Compute half-extent from the polygon's bounding box diagonal so the
    // splitting line is always long enough to cross the polygon, without
    // relying on a hardcoded constant that could cause floating-point issues
    // on very large or very small maps.
    let (mut min_pt, mut max_pt) = (poly[0], poly[0]);
    for &v in &poly[1..] {
        min_pt = min_pt.min(v);
        max_pt = max_pt.max(v);
    }
    let half_extent = (max_pt - min_pt).length() + 1.0;
    let line_a = line_origin - line_dir * half_extent;
    let line_b = line_origin + line_dir * half_extent;

    let mut intersections: Vec<(usize, Vec2)> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        if let Some(pt) = segment_intersection(line_a, line_b, poly[i], poly[j]) {
            let dominated = intersections
                .iter()
                .any(|(_, p)| p.distance(pt) < DEDUP_TOLERANCE);
            if !dominated {
                intersections.push((i, pt));
            }
        }
    }

    if intersections.len() < 2 {
        return None;
    }

    // For concave polygons the splitting line may intersect >2 edges.
    // Pick the pair of intersections whose midpoint is closest to the
    // line origin (centroid), which splits through the polygon's core
    // rather than clipping an outer lobe.
    if intersections.len() > 2 {
        let mut best_pair: Option<(usize, usize)> = None;
        let mut best_dist = f32::MAX;
        for i in 0..intersections.len() {
            for j in (i + 1)..intersections.len() {
                let mid = (intersections[i].1 + intersections[j].1) * 0.5;
                let d = mid.distance_squared(line_origin);
                if d < best_dist {
                    best_dist = d;
                    best_pair = Some((i, j));
                }
            }
        }
        let (i, j) = best_pair.unwrap();
        intersections = vec![intersections[i], intersections[j]];
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

    // Remove consecutive duplicate vertices (from split points coinciding
    // with existing polygon vertices), which would produce zero-length
    // edges and NaN in subsequent normalizations.
    dedup_consecutive(&mut poly_a);
    dedup_consecutive(&mut poly_b);

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

    // Early-exit for degenerate slivers: if the polygon's axis-aligned
    // bounding box is excessively skewed, further subdivision will only
    // waste cycles producing sub-threshold fragments.
    let (mut bb_min, mut bb_max) = (poly[0], poly[0]);
    for &v in &poly[1..] {
        bb_min = bb_min.min(v);
        bb_max = bb_max.max(v);
    }
    let extent = bb_max - bb_min;
    let (short, long) = if extent.x < extent.y {
        (extent.x, extent.y)
    } else {
        (extent.y, extent.x)
    };
    if short > 0.0 && long / short > MAX_SUBDIVISION_ASPECT_RATIO {
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
            result.extend(subdivide_polygon(
                &left,
                max_area,
                min_area,
                depth_limit - 1,
            ));
            result.extend(subdivide_polygon(
                &right,
                max_area,
                min_area,
                depth_limit - 1,
            ));
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
    if len_sq < DEGENERATE_SEG_LEN_SQ {
        return p.distance(a) < POINT_ON_SEG_TOLERANCE;
    }
    let t = (p - a).dot(ab) / len_sq;
    let t_tol = POINT_ON_SEG_TOLERANCE / len_sq.sqrt();
    if !(-t_tol..=1.0 + t_tol).contains(&t) {
        return false;
    }
    let proj = a + ab * t.clamp(0.0, 1.0);
    p.distance(proj) < POINT_ON_SEG_TOLERANCE
}

fn inscribed_box(poly: &[Vec2], frontage_idx: usize) -> Option<(Vec2, f32, f32, f32)> {
    let n = poly.len();
    if n < 3 {
        return None;
    }

    let fa = poly[frontage_idx];
    let fb = poly[(frontage_idx + 1) % n];

    let street_dir = (fb - fa).normalize();
    // Blocks from extract_blocks() have guaranteed CW winding, so the
    // interior is always to the right of each edge direction.
    let inward_dir = Vec2::new(street_dir.y, -street_dir.x);

    let rotation = street_dir.y.atan2(street_dir.x);
    let width = (fb - fa).length();

    // Compute ray extent from polygon bounding box so rays always reach
    // the far side without relying on a hardcoded constant.
    let (mut min_pt, mut max_pt) = (poly[0], poly[0]);
    for &v in &poly[1..] {
        min_pt = min_pt.min(v);
        max_pt = max_pt.max(v);
    }
    let ray_extent = (max_pt - min_pt).length() + 1.0;

    // Cast rays inward from points along the frontage edge to find the
    // minimum depth before hitting the opposite polygon boundary.
    //
    // We use uniform samples PLUS projections of every polygon vertex onto
    // the frontage line so that vertex-induced notches are never missed.
    let num_uniform = 7;
    let mut sample_ts: Vec<f32> = (0..=num_uniform)
        .map(|i| i as f32 / num_uniform as f32)
        .collect();

    // Project each non-frontage vertex onto the frontage line and add its
    // parametric position if it falls within the edge span.
    let frontage_vec = fb - fa;
    let frontage_len_sq = frontage_vec.length_squared();
    if frontage_len_sq > DEGENERATE_SEG_LEN_SQ {
        for (k, &vertex) in poly.iter().enumerate() {
            if k == frontage_idx || k == (frontage_idx + 1) % n {
                continue;
            }
            let t = (vertex - fa).dot(frontage_vec) / frontage_len_sq;
            if (0.0..=1.0).contains(&t) {
                sample_ts.push(t);
            }
        }
    }

    let mut min_depth = f32::MAX;

    for t in &sample_ts {
        let ray_origin = fa.lerp(fb, *t);
        let ray_end = ray_origin + inward_dir * ray_extent;

        let mut closest_dist = f32::MAX;
        for j in 0..n {
            if j == frontage_idx {
                continue;
            }
            let ea = poly[j];
            let eb = poly[(j + 1) % n];
            if let Some(hit) = segment_intersection(ray_origin, ray_end, ea, eb) {
                let d = (hit - ray_origin).dot(inward_dir);
                if d > RAY_HIT_EPS && d < closest_dist {
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
        let ray_end = ray_origin + street_dir * sign * ray_extent;
        let mut closest = f32::MAX;
        for j in 0..n {
            let ea = poly[j];
            let eb = poly[(j + 1) % n];
            if let Some(hit) = segment_intersection(ray_origin, ray_end, ea, eb) {
                let d = (hit - ray_origin).dot(street_dir * sign);
                if d > RAY_HIT_EPS && d < closest {
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
    frontage_center: Vec2,
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
    let inward_dir = Vec2::new(street_dir.y, -street_dir.x);
    let depth_shift = (config.front_setback.max(0.0) - config.rear_setback.max(0.0)) * 0.5;
    let adjusted_center = center + inward_dir * depth_shift;

    Some(BuildingLot {
        position: adjusted_center,
        frontage_center,
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
    let n = poly.len();
    let frontage_center = (poly[frontage_idx] + poly[(frontage_idx + 1) % n]) * 0.5;
    let (center, rotation, width, depth) = inscribed_box(poly, frontage_idx)?;
    apply_setbacks(center, frontage_center, rotation, width, depth, config)
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
        assert!(
            (area_a - 100.0).abs() < 1.0,
            "half A should be ~100 sqm, got {area_a}"
        );
        assert!(
            (area_b - 100.0).abs() < 1.0,
            "half B should be ~100 sqm, got {area_b}"
        );
    }

    #[test]
    fn subdivide_large_block() {
        let big = square(0.0, 0.0, 40.0); // 1600 sqm
        let sub = subdivide_polygon(&big, 400.0, 50.0, 10);
        assert!(
            sub.len() >= 4,
            "1600 sqm block should yield at least 4 lots, got {}",
            sub.len()
        );
        for poly in &sub {
            let area = polygon_area(poly);
            assert!(area <= 400.0 + 1.0, "sub-polygon area {area} exceeds max");
        }
    }

    #[test]
    fn inscribed_box_rectangle() {
        // CW winding — matches extract_blocks() output
        let rect = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 5.0),
            Vec2::new(10.0, 5.0),
            Vec2::new(10.0, 0.0),
        ];
        let (frontage_idx, _) = find_frontage(&rect, &rect);
        let result = inscribed_box(&rect, frontage_idx);
        assert!(result.is_some());
        let (center, _rotation, width, depth) = result.unwrap();
        assert!(
            (width - 10.0).abs() < 0.5,
            "width should be ~10, got {width}"
        );
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
        let result = apply_setbacks(Vec2::ZERO, Vec2::ZERO, 0.0, 5.0, 20.0, &config);
        assert!(result.is_none());
    }

    #[test]
    fn inscribed_box_notched_polygon() {
        // A polygon with an inward notch between uniform sample points.
        // The notch vertex at (5, 2) should limit depth to 2, not the
        // far edge at y=5. Without vertex-projection sampling, uniform
        // rays could miss this notch entirely.
        //
        // CW winding: frontage along bottom edge (y=0), interior upward.
        let poly = vec![
            Vec2::new(0.0, 0.0),   // 0 — frontage start
            Vec2::new(0.0, 5.0),   // 1
            Vec2::new(5.0, 2.0),   // 2 — notch vertex
            Vec2::new(10.0, 5.0),  // 3
            Vec2::new(10.0, 0.0),  // 4 — frontage end
        ];
        let frontage_idx = 4; // edge 4→0: (10,0)→(0,0)
        let result = inscribed_box(&poly, frontage_idx);
        assert!(result.is_some());
        let (_center, _rotation, _width, depth) = result.unwrap();
        // Depth must be ≤ 2.0 (the notch), not ~5.0 (the far edges).
        assert!(
            depth <= 2.1,
            "depth should be clamped by notch vertex at y=2, got {depth}"
        );
    }

    #[test]
    fn extract_lots_empty_graph() {
        let hm = symbios_ground::HeightMap::new(8, 8, 1.0);
        let graph = RoadGraph::default();
        let lots = extract_lots(&graph, &hm, 0.0, &LotConfig::default());
        assert!(lots.is_empty());
    }
}
