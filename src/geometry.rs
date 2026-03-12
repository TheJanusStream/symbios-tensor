//! Low-level 2D geometry primitives used by the tracer and spatial hash.

use glam::Vec2;

/// Epsilon for treating a segment as degenerate (length² below this).
const DEGENERATE_LEN_SQ: f32 = 1e-6;

/// Relative epsilon for treating two lines as parallel. The raw cross
/// product scales with segment lengths, so we compare against
/// `PARALLEL_REL_EPS * |s1| * |s2|` to remain robust at any world scale.
const PARALLEL_REL_EPS: f32 = 1e-6;

/// Returns the closest point on segment `a`–`b` to point `p`.
pub fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < DEGENERATE_LEN_SQ {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + t * ab
}

/// Returns the intersection point of segments `a1`–`a2` and `b1`–`b2`, if any.
pub fn segment_intersection(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Option<Vec2> {
    let s1 = a2 - a1;
    let s2 = b2 - b1;

    let denom = -s2.x * s1.y + s1.x * s2.y;
    // Use a relative threshold so the parallel check works at any scale.
    let len_product = s1.length() * s2.length();
    if denom.abs() < PARALLEL_REL_EPS * len_product {
        return None;
    }

    let d = a1 - b1;
    let s = (-s1.y * d.x + s1.x * d.y) / denom;
    let t = (s2.x * d.y - s2.y * d.x) / denom;

    if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) {
        Some(a1 + t * s1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_midpoint() {
        let p = Vec2::new(1.0, 1.0);
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = closest_point_on_segment(p, a, b);
        assert!((c - Vec2::new(1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn closest_point_clamped_start() {
        let p = Vec2::new(-5.0, 0.0);
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = closest_point_on_segment(p, a, b);
        assert!((c - a).length() < 1e-5);
    }

    #[test]
    fn closest_point_degenerate() {
        let p = Vec2::new(3.0, 4.0);
        let a = Vec2::new(1.0, 1.0);
        let c = closest_point_on_segment(p, a, a);
        assert!((c - a).length() < 1e-5);
    }

    #[test]
    fn segments_cross() {
        let hit = segment_intersection(
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
            Vec2::new(2.0, 0.0),
        );
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h - Vec2::new(1.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn parallel_segments_no_intersection() {
        let hit = segment_intersection(
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(2.0, 1.0),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn non_overlapping_segments() {
        let hit = segment_intersection(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, -1.0),
            Vec2::new(2.0, 1.0),
        );
        assert!(hit.is_none());
    }
}
