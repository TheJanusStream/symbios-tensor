//! Graph rationalization: straighten and smooth the road network.
//!
//! After the tracer produces a raw network of many small segments, this pass
//! rewrites chains of degree-2 nodes into cleaner geometry:
//!
//! 1. **Ramer-Douglas-Peucker** decimation removes unnecessary intermediate
//!    points, straightening nearly-collinear runs.
//! 2. **Fillet corners** replaces sharp bends with smooth quadratic Bézier arcs,
//!    giving the network a civil-engineered look.
//!
//! The result is a graph whose geometry is already presentation-ready — the 3D
//! mesher can simply extrude edges without additional spline smoothing.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::graph::{RoadGraph, RoadType};
use crate::topology::{compute_active_degrees, extract_chains};

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
}

impl Default for RationalizeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rdp_tolerance: 2.0,
            major_fillet_radius: 20.0,
            minor_fillet_radius: 10.0,
            fillet_segments: 6,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Rationalizes the road graph in-place: straightens chains with RDP then
/// smooths corners with fillet arcs.
pub fn rationalize_graph(graph: &mut RoadGraph, config: &RationalizeConfig) {
    let degrees = compute_active_degrees(graph);
    let chains = extract_chains(graph, &degrees);

    for chain in &chains {
        let positions: Vec<Vec2> = chain.nodes.iter().map(|&nid| graph.node_pos(nid)).collect();
        if positions.len() < 2 {
            continue;
        }

        // 1. Decimate
        let simplified = ramer_douglas_peucker(&positions, config.rdp_tolerance);
        if simplified.len() < 2 {
            continue;
        }

        // 2. Fillet
        let fillet_radius = match chain.road_type {
            RoadType::Major => config.major_fillet_radius,
            RoadType::Minor => config.minor_fillet_radius,
        };
        let smoothed = fillet_corners(&simplified, fillet_radius, config.fillet_segments);
        if smoothed.len() < 2 {
            continue;
        }

        // 3. Deactivate old edges
        for &eid in &chain.edges {
            graph.edges[eid as usize].active = false;
        }

        // 4. Inject new geometry
        //    The first and last nodes of the chain are junction/dead-end nodes
        //    that must be preserved — we connect the new interior nodes to them.
        let first_junction = chain.nodes[0];
        let last_junction = *chain.nodes.last().unwrap();

        // Skip the first and last positions in `smoothed` — they correspond to
        // the existing junction nodes. Create new interior nodes for everything
        // in between.
        let mut prev_node = first_junction;
        for (i, &pos) in smoothed.iter().enumerate() {
            let current_node = if i == 0 {
                first_junction
            } else if i == smoothed.len() - 1 {
                last_junction
            } else {
                graph.add_node(pos)
            };

            if i > 0 {
                graph.add_edge(prev_node, current_node, chain.road_type);
            }
            prev_node = current_node;
        }
    }
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

    rdp_recurse(points, 0, points.len() - 1, tolerance * tolerance, &mut keep);

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

    for (i, pt) in points.iter().enumerate().skip(start + 1).take(end - start - 1) {
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
    let seg_lengths: Vec<f32> = points
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .collect();

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
        let a = g.add_node(Vec2::new(0.0, 0.0));       // 0
        let b = g.add_node(Vec2::new(10.0, 0.5));      // 1 - slight deviation
        let c = g.add_node(Vec2::new(20.0, 0.0));      // 2
        let d = g.add_node(Vec2::new(30.0, 0.0));      // 3

        // Make A and D true junctions (degree >= 3) with two stubs each.
        let stub_a1 = g.add_node(Vec2::new(-10.0, 0.0));  // 4
        let stub_a2 = g.add_node(Vec2::new(0.0, -10.0));  // 5
        let stub_d1 = g.add_node(Vec2::new(40.0, 0.0));   // 6
        let stub_d2 = g.add_node(Vec2::new(30.0, -10.0)); // 7
        g.add_edge(stub_a1, a, RoadType::Major);  // edge 0
        g.add_edge(stub_a2, a, RoadType::Major);  // edge 1
        g.add_edge(a, b, RoadType::Major);         // edge 2
        g.add_edge(b, c, RoadType::Major);         // edge 3
        g.add_edge(c, d, RoadType::Major);         // edge 4
        g.add_edge(d, stub_d1, RoadType::Major);  // edge 5
        g.add_edge(d, stub_d2, RoadType::Major);  // edge 6

        let config = RationalizeConfig {
            enabled: true,
            rdp_tolerance: 2.0,        // B's deviation of 0.5 is within tolerance
            major_fillet_radius: 0.0,   // disable filleting for this test
            minor_fillet_radius: 0.0,
            fillet_segments: 4,
        };

        rationalize_graph(&mut g, &config);

        // The original chain edges (A→B, B→C, C→D) should be deactivated.
        assert!(!g.edges[2].active, "edge A→B should be deactivated");
        assert!(!g.edges[3].active, "edge B→C should be deactivated");
        assert!(!g.edges[4].active, "edge C→D should be deactivated");

        // After rationalization, the total number of active edges should be
        // at least as many as before (stubs get replaced 1:1, chain gets
        // simplified). Verify the graph has active connectivity.
        let active_edges: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.active)
            .collect();
        // We had 4 stubs + 3 chain = 7 original. Chain (3 edges through B,C)
        // collapses to 1 edge (A→D). Stubs get replaced 1:1 = 4 edges.
        // Total: at least 5 active edges.
        assert!(
            active_edges.len() >= 5,
            "expected at least 5 active edges, got {}",
            active_edges.len()
        );

        // Verify junction nodes A and D still participate in active edges.
        let a_active = g.nodes[a as usize]
            .edges
            .iter()
            .any(|&eid| g.edges[eid as usize].active);
        let d_active = g.nodes[d as usize]
            .edges
            .iter()
            .any(|&eid| g.edges[eid as usize].active);
        assert!(a_active, "node A should have active edges");
        assert!(d_active, "node D should have active edges");
    }
}
