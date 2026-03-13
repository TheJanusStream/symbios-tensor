//! Streamline tracer — the core road generation algorithm.
//!
//! Seeds are placed on a jittered grid and traced bidirectionally (±major,
//! ±minor) through the tensor field using RK2 (midpoint) integration.
//! Traces snap to existing nodes and edges via the spatial hash, creating
//! T-junctions and 4-way intersections. Orthogonal branches are spawned at
//! configurable intervals to fill the network.

use std::collections::VecDeque;
use std::fmt;

use glam::Vec2;
use rand::Rng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use symbios_ground::HeightMap;

use crate::graph::{RoadGraph, RoadType};
use crate::spatial::{SpatialHash, TraceResult, resolve_trace_step};
use crate::tensor::TensorField;

/// Error returned when [`generate_roads`] receives an invalid configuration.
#[derive(Debug, Clone)]
pub struct TensorError {
    /// Human-readable description of the invalid parameter.
    pub message: String,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorError: {}", self.message)
    }
}

impl std::error::Error for TensorError {}

/// Configuration for tensor-field city generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorConfig {
    /// RNG seed for jittered seed placement.
    pub seed: u64,
    /// World-space distance per integration step.
    pub step_size: f32,
    /// Desired spacing between parallel major roads (avenues).
    pub major_road_dist: f32,
    /// Desired spacing between parallel minor roads (streets).
    pub minor_road_dist: f32,
    /// Snap radius for merging trace endpoints with existing geometry.
    pub snap_radius: f32,
    /// Maximum number of integration steps per trace before it is abandoned.
    pub max_trace_steps: u32,
    /// Momentum factor for the tracer direction (range `0.0` to `0.99`).
    /// Higher values resist sharp direction changes caused by high-frequency
    /// terrain noise, producing smoother, less zig-zaggy roads.
    pub tracer_inertia: f32,
    /// Absolute world-space Y coordinate for the water plane.
    /// Terrain at or below this height is treated as underwater.
    /// Defaults to [`f32::NEG_INFINITY`] (no water).
    pub water_level: f32,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            step_size: 2.0,
            major_road_dist: 40.0,
            minor_road_dist: 15.0,
            snap_radius: 4.0,
            max_trace_steps: 300,
            tracer_inertia: 0.8,
            water_level: f32::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Seed {
    position: Vec2,
    direction: Vec2,
    road_type: RoadType,
    /// Accumulated distance since the last orthogonal branch was spawned.
    branch_accum: f32,
    /// If set, reuse this existing node instead of creating a new one at `position`.
    existing_node: Option<u32>,
}

/// Generates a [`RoadGraph`] by tracing streamlines through the tensor field
/// derived from the given heightmap.
///
/// # Errors
///
/// Returns [`TensorError`] if any `TensorConfig` parameter is non-positive
/// (step_size, major_road_dist, minor_road_dist, snap_radius must all be > 0).
pub fn generate_roads(
    heightmap: &HeightMap,
    config: &TensorConfig,
) -> Result<RoadGraph, TensorError> {
    if !config.step_size.is_finite() || config.step_size <= 0.0 {
        return Err(TensorError {
            message: format!(
                "step_size must be finite and positive, got {}",
                config.step_size
            ),
        });
    }
    if !config.major_road_dist.is_finite() || config.major_road_dist <= 0.0 {
        return Err(TensorError {
            message: format!(
                "major_road_dist must be finite and positive, got {}",
                config.major_road_dist
            ),
        });
    }
    if !config.minor_road_dist.is_finite() || config.minor_road_dist <= 0.0 {
        return Err(TensorError {
            message: format!(
                "minor_road_dist must be finite and positive, got {}",
                config.minor_road_dist
            ),
        });
    }
    if !config.snap_radius.is_finite() || config.snap_radius <= 0.0 {
        return Err(TensorError {
            message: format!(
                "snap_radius must be finite and positive, got {}",
                config.snap_radius
            ),
        });
    }

    let field = TensorField::new(heightmap);
    let mut graph = RoadGraph::default();

    let world_w = heightmap.world_width();
    let world_d = heightmap.world_depth();
    let cell_size = config.snap_radius * 2.0;
    let mut spatial = SpatialHash::new(world_w, world_d, cell_size);

    let mut rng = Pcg64::new(config.seed.into(), 0xa02bdbf7bb3c0a7_u128);

    // --- Seed generation ---
    // Drop seeds along a grid at `major_road_dist` spacing, jittered slightly.
    let mut active: VecDeque<Seed> = VecDeque::new();

    let margin = config.major_road_dist * 0.5;
    let mut x = margin;
    while x < world_w - margin {
        let mut z = margin;
        while z < world_d - margin {
            let jitter_x: f32 = rng.random_range(-config.step_size..config.step_size);
            let jitter_z: f32 = rng.random_range(-config.step_size..config.step_size);
            let pos = Vec2::new(x + jitter_x, z + jitter_z);

            // Do not spawn seeds underwater (at or below water level)
            if heightmap.get_height_at(pos.x, pos.y) <= config.water_level {
                z += config.major_road_dist;
                continue;
            }

            let (major, minor) = field.sample(pos.x, pos.y);

            // Create a shared starting node for all traces from this seed
            let elev = heightmap.get_height_at(pos.x, pos.y);
            let shared_node = graph.add_node_with_elevation(pos, elev);
            spatial.insert_node(shared_node, pos);

            // Trace both directions along each axis to form full through-lines
            for &dir in &[major, -major] {
                active.push_back(Seed {
                    position: pos,
                    direction: dir,
                    road_type: RoadType::Major,
                    branch_accum: 0.0,
                    existing_node: Some(shared_node),
                });
            }
            for &dir in &[minor, -minor] {
                active.push_back(Seed {
                    position: pos,
                    direction: dir,
                    road_type: RoadType::Minor,
                    branch_accum: 0.0,
                    existing_node: Some(shared_node),
                });
            }

            z += config.major_road_dist;
        }
        x += config.major_road_dist;
    }

    // --- Trace each seed ---
    let bounds = Vec2::new(world_w, world_d);
    // Cap total traces to prevent runaway branching in circular tensor flows.
    // Use the larger of seed-proportional and area-proportional limits so that
    // sparse seeds on large maps don't prematurely abort city growth.
    let area_based = ((world_w * world_d) / config.minor_road_dist) as usize;
    let max_traces = (active.len() * 50).max(area_based);
    let mut trace_count = 0_usize;
    while let Some(seed) = active.pop_front() {
        trace_count += 1;
        if trace_count > max_traces {
            break;
        }
        trace_streamline(
            &field,
            &mut graph,
            &mut spatial,
            &mut active,
            seed,
            config,
            bounds,
        );
    }

    Ok(graph)
}

fn trace_streamline(
    field: &TensorField<'_>,
    graph: &mut RoadGraph,
    spatial: &mut SpatialHash,
    active: &mut VecDeque<Seed>,
    seed: Seed,
    config: &TensorConfig,
    bounds: Vec2,
) {
    let start_node = match seed.existing_node {
        Some(id) => id,
        None => {
            let elev = field.heightmap.get_height_at(seed.position.x, seed.position.y);
            let id = graph.add_node_with_elevation(seed.position, elev);
            spatial.insert_node(id, seed.position);
            id
        }
    };

    let mut current_node = start_node;
    let mut dir = seed.direction;
    let mut branch_accum = seed.branch_accum;

    for _ in 0..config.max_trace_steps {
        let current_pos = graph.node_pos(current_node);

        // RK2 (midpoint method): sample k1 at current position
        let (k1_major, k1_minor) = field.sample(current_pos.x, current_pos.y);
        let k1 = match seed.road_type {
            RoadType::Major => k1_major,
            RoadType::Minor => k1_minor,
        };
        let k1 = if k1.dot(dir) < 0.0 { -k1 } else { k1 };

        // Sample k2 at midpoint
        let mid = current_pos + k1 * (config.step_size * 0.5);
        let (k2_major, k2_minor) = field.sample(mid.x, mid.y);
        let k2 = match seed.road_type {
            RoadType::Major => k2_major,
            RoadType::Minor => k2_minor,
        };
        let k2 = if k2.dot(k1) < 0.0 { -k2 } else { k2 };

        // Blend with previous direction for momentum (anti-zig-zag).
        let inertia = config.tracer_inertia.clamp(0.0, 0.99);
        dir = (dir * inertia + k2 * (1.0 - inertia)).normalize_or_zero();
        if dir.length_squared() < 1e-12 {
            break;
        }
        let proposed = current_pos + dir * config.step_size;

        // Bounds check (NaN coordinates fail is_finite and abort the trace)
        if !proposed.x.is_finite()
            || !proposed.y.is_finite()
            || proposed.x < 0.0
            || proposed.x >= bounds.x
            || proposed.y < 0.0
            || proposed.y >= bounds.y
        {
            break;
        }

        // Coastline collision. If the proposed step dips underwater, abort the trace.
        if field.heightmap.get_height_at(proposed.x, proposed.y) <= config.water_level {
            break;
        }

        match resolve_trace_step(
            graph,
            spatial,
            current_pos,
            proposed,
            config.snap_radius,
            current_node,
        ) {
            TraceResult::Clear(pos) => {
                let elev = field.heightmap.get_height_at(pos.x, pos.y);
                let new_node = graph.add_node_with_elevation(pos, elev);
                spatial.insert_node(new_node, pos);
                let edge_id = graph.add_edge(current_node, new_node, seed.road_type);
                spatial.insert_edge(edge_id, current_pos, pos);
                current_node = new_node;
            }
            TraceResult::SnappedToNode(n_id) => {
                // Avoid duplicate edges
                let already_connected =
                    graph.nodes[current_node as usize].edges.iter().any(|&eid| {
                        let e = &graph.edges[eid as usize];
                        e.active && (e.start == n_id || e.end == n_id)
                    });
                if !already_connected {
                    let n_pos = graph.node_pos(n_id);

                    // The snapped endpoint may differ from the proposed
                    // position, rotating the committed segment so it crosses
                    // an edge the original ray missed. Re-check the adjusted
                    // trajectory for crossings to maintain planarity.
                    let crossing =
                        find_crossing(graph, spatial, current_pos, n_pos, current_node, n_id);
                    if let Some((cross_eid, cross_pt)) = crossing {
                        let mid_node = split_or_snap_edge(graph, spatial, cross_eid, cross_pt);
                        // split_or_snap_edge may return an existing endpoint;
                        // guard against creating a duplicate edge.
                        let already = graph.nodes[current_node as usize]
                            .edges
                            .iter()
                            .any(|&eid| {
                                let e = &graph.edges[eid as usize];
                                e.active && (e.start == mid_node || e.end == mid_node)
                            });
                        if !already {
                            let mid_pos = graph.node_pos(mid_node);
                            let connecting =
                                graph.add_edge(current_node, mid_node, seed.road_type);
                            spatial.insert_edge(connecting, current_pos, mid_pos);
                        }
                    } else {
                        let edge_id = graph.add_edge(current_node, n_id, seed.road_type);
                        spatial.insert_edge(edge_id, current_pos, n_pos);
                    }
                }
                break;
            }
            TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            } => {
                let split_edge = &graph.edges[edge_id as usize];
                let old_start_pos_pre = graph.node_pos(split_edge.start);
                let old_end_pos_pre = graph.node_pos(split_edge.end);
                let edge_dir = (old_end_pos_pre - old_start_pos_pre).normalize_or_zero();

                let mid_node = split_or_snap_edge(graph, spatial, edge_id, intersection_pos);
                let mid_pos = graph.node_pos(mid_node);

                // The snap target (intersection_pos) can be up to snap_radius
                // away from proposed_pos, sweeping the connecting segment
                // through unverified space. Re-check for crossings to
                // maintain planarity, mirroring the SnappedToNode branch.
                let crossing =
                    find_crossing(graph, spatial, current_pos, mid_pos, current_node, mid_node);
                if let Some((cross_eid, cross_pt)) = crossing {
                    let cross_mid = split_or_snap_edge(graph, spatial, cross_eid, cross_pt);
                    // split_or_snap_edge may return an existing endpoint;
                    // guard against creating a duplicate edge.
                    let already = graph.nodes[current_node as usize]
                        .edges
                        .iter()
                        .any(|&eid| {
                            let e = &graph.edges[eid as usize];
                            e.active && (e.start == cross_mid || e.end == cross_mid)
                        });
                    if !already {
                        let cross_pos = graph.node_pos(cross_mid);
                        let connecting_edge =
                            graph.add_edge(current_node, cross_mid, seed.road_type);
                        spatial.insert_edge(connecting_edge, current_pos, cross_pos);
                    }
                    // Terminate: we hit a crossing before reaching the
                    // snapped edge, so continuing from mid_node would
                    // leave a gap in the graph.
                    break;
                }

                let connecting_edge = graph.add_edge(current_node, mid_node, seed.road_type);
                spatial.insert_edge(connecting_edge, graph.node_pos(current_node), mid_pos);

                // If the trace direction is nearly parallel to the edge
                // we just split, continuing would create overlapping
                // geometry invisible to resolve_trace_step (the split
                // halves are connected to mid_node and thus skipped).
                let alignment = dir.dot(edge_dir).abs();
                if alignment > 0.9 {
                    break;
                }

                // Continue tracing through the intersection so that
                // 4-way crossings form naturally instead of dead-ending
                // at every T-junction.
                current_node = mid_node;
            }
        }

        // Branching: spawn an orthogonal trace at regular intervals
        branch_accum += config.step_size;
        let branch_dist = match seed.road_type {
            RoadType::Major => config.minor_road_dist,
            RoadType::Minor => config.major_road_dist,
        };
        if branch_accum >= branch_dist {
            branch_accum -= branch_dist;
            let (field_major, field_minor) = field.sample(proposed.x, proposed.y);
            let branch_dir = match seed.road_type {
                RoadType::Major => field_minor,
                RoadType::Minor => field_major,
            };
            let branch_type = match seed.road_type {
                RoadType::Major => RoadType::Minor,
                RoadType::Minor => RoadType::Major,
            };
            for &dir_sign in &[1.0_f32, -1.0] {
                active.push_back(Seed {
                    position: graph.node_pos(current_node),
                    direction: branch_dir * dir_sign,
                    road_type: branch_type,
                    branch_accum: 0.0,
                    existing_node: Some(current_node),
                });
            }
        }
    }
}

/// Minimum squared distance between a split point and an existing endpoint.
/// If a cross-point falls closer than this to an edge's start or end node,
/// we snap to that node instead of inserting a degenerate micro-segment.
const SPLIT_SNAP_DIST_SQ: f32 = 1.0;

/// Splits `edge_id` at `split_pos`, or snaps to an existing endpoint if
/// `split_pos` is within `SPLIT_SNAP_DIST_SQ` of one. Returns the node
/// at the split (or snapped endpoint). When a real split occurs the two
/// new half-edge IDs are returned; when snapping, no new edges are created.
fn split_or_snap_edge(
    graph: &mut RoadGraph,
    spatial: &mut SpatialHash,
    edge_id: u32,
    split_pos: Vec2,
) -> u32 {
    let edge = &graph.edges[edge_id as usize];
    let start = edge.start;
    let end = edge.end;
    let start_pos = graph.node_pos(start);
    let end_pos = graph.node_pos(end);

    // Snap to existing endpoint if the split point is very close
    if split_pos.distance_squared(start_pos) < SPLIT_SNAP_DIST_SQ {
        return start;
    }
    if split_pos.distance_squared(end_pos) < SPLIT_SNAP_DIST_SQ {
        return end;
    }

    let (mid_node, ea, eb) = graph.split_edge(edge_id, split_pos);
    spatial.remove_edge(edge_id, start_pos, end_pos);
    spatial.insert_node(mid_node, split_pos);
    spatial.insert_edge(ea, start_pos, split_pos);
    spatial.insert_edge(eb, split_pos, end_pos);
    mid_node
}

/// Checks whether the segment `from -> to` crosses any active edge not
/// incident to `from_node` or `to_node`. Returns the closest crossing if
/// one exists.
fn find_crossing(
    graph: &RoadGraph,
    spatial: &SpatialHash,
    from: Vec2,
    to: Vec2,
    from_node: u32,
    to_node: u32,
) -> Option<(u32, Vec2)> {
    use crate::geometry::segment_intersection;

    let edge_ids = spatial.edges_in_region(from, to, 0.0);
    let mut best: Option<(u32, Vec2)> = None;
    let mut best_dist = f32::MAX;

    for e_id in edge_ids {
        let edge = &graph.edges[e_id as usize];
        if !edge.active {
            continue;
        }
        if edge.start == from_node || edge.end == from_node {
            continue;
        }
        if edge.start == to_node || edge.end == to_node {
            continue;
        }
        let e_start = graph.nodes[edge.start as usize].position;
        let e_end = graph.nodes[edge.end as usize].position;
        if let Some(pt) = segment_intersection(from, to, e_start, e_end) {
            let d = from.distance_squared(pt);
            if d < best_dist {
                best_dist = d;
                best = Some((e_id, pt));
            }
        }
    }
    best
}
