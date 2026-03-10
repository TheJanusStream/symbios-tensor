//! Spatial hash grid for fast proximity and intersection queries during tracing.
//!
//! [`SpatialHash`] partitions world space into a uniform grid of cells, each
//! storing references to nearby nodes and edges. [`resolve_trace_step`]
//! evaluates a proposed trace step against the graph, detecting node snaps,
//! edge crossings, and T-junction proximity — all in O(1) expected time per
//! query.

use glam::Vec2;

use crate::geometry::{closest_point_on_segment, segment_intersection};
use crate::graph::{EdgeId, NodeId, RoadGraph};

#[derive(Clone, Default)]
struct HashCell {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

/// A flat 2D spatial hash grid for O(1) proximity queries against road nodes and edges.
pub struct SpatialHash {
    cell_size: f32,
    cols: usize,
    rows: usize,
    cells: Vec<HashCell>,
}

impl SpatialHash {
    /// Maximum number of cells the grid will allocate. If the requested
    /// `cell_size` would exceed this, it is automatically enlarged.
    const MAX_CELLS: usize = 4_000_000;

    /// Creates a new spatial hash covering a `world_width` × `world_depth`
    /// area with cells of the given size. If the resulting grid would exceed
    /// `MAX_CELLS`, `cell_size` is automatically increased to fit.
    pub fn new(world_width: f32, world_depth: f32, cell_size: f32) -> Self {
        let mut cs = cell_size;
        loop {
            let cols = (world_width / cs).ceil() as usize;
            let rows = (world_depth / cs).ceil() as usize;
            if cols.saturating_mul(rows) <= Self::MAX_CELLS {
                return Self {
                    cell_size: cs,
                    cols,
                    rows,
                    cells: vec![HashCell::default(); cols * rows],
                };
            }
            cs *= 2.0;
        }
    }

    fn coords_to_index(&self, pos: Vec2) -> Option<usize> {
        let col = (pos.x / self.cell_size).floor() as isize;
        let row = (pos.y / self.cell_size).floor() as isize;

        if col >= 0 && col < self.cols as isize && row >= 0 && row < self.rows as isize {
            Some(row as usize * self.cols + col as usize)
        } else {
            None
        }
    }

    /// Registers a node in the cell containing `pos`.
    pub fn insert_node(&mut self, id: NodeId, pos: Vec2) {
        if let Some(idx) = self.coords_to_index(pos) {
            self.cells[idx].nodes.push(id);
        }
    }

    /// Removes an edge from every cell it was registered in.
    pub fn remove_edge(&mut self, id: EdgeId, start: Vec2, end: Vec2) {
        let min_col = ((start.x.min(end.x) / self.cell_size).floor() as isize).max(0);
        let max_col =
            ((start.x.max(end.x) / self.cell_size).floor() as isize).min(self.cols as isize - 1);
        let min_row = ((start.y.min(end.y) / self.cell_size).floor() as isize).max(0);
        let max_row =
            ((start.y.max(end.y) / self.cell_size).floor() as isize).min(self.rows as isize - 1);

        for r in min_row..=max_row {
            for c in min_col..=max_col {
                let idx = r as usize * self.cols + c as usize;
                self.cells[idx].edges.retain(|&e| e != id);
            }
        }
    }

    /// Registers an edge in every cell its axis-aligned bounding box overlaps.
    pub fn insert_edge(&mut self, id: EdgeId, start: Vec2, end: Vec2) {
        let min_col = ((start.x.min(end.x) / self.cell_size).floor() as isize).max(0);
        let max_col =
            ((start.x.max(end.x) / self.cell_size).floor() as isize).min(self.cols as isize - 1);
        let min_row = ((start.y.min(end.y) / self.cell_size).floor() as isize).max(0);
        let max_row =
            ((start.y.max(end.y) / self.cell_size).floor() as isize).min(self.rows as isize - 1);

        for r in min_row..=max_row {
            for c in min_col..=max_col {
                let idx = r as usize * self.cols + c as usize;
                self.cells[idx].edges.push(id);
            }
        }
    }

    /// Returns deduplicated edge IDs from all cells overlapping the region.
    pub(crate) fn edges_in_region(&self, start: Vec2, end: Vec2, padding: f32) -> Vec<EdgeId> {
        let cells = self.cells_for_region(start, end, padding);
        let mut ids = Vec::new();
        for cell in &cells {
            for &eid in &cell.edges {
                if !ids.contains(&eid) {
                    ids.push(eid);
                }
            }
        }
        ids
    }

    /// Collects all unique cells whose bounding box overlaps `[start, end]` expanded by `padding`.
    fn cells_for_region(&self, start: Vec2, end: Vec2, padding: f32) -> Vec<&HashCell> {
        let min_col = (((start.x.min(end.x) - padding) / self.cell_size).floor() as isize).max(0);
        let max_col = (((start.x.max(end.x) + padding) / self.cell_size).floor() as isize)
            .min(self.cols as isize - 1);
        let min_row = (((start.y.min(end.y) - padding) / self.cell_size).floor() as isize).max(0);
        let max_row = (((start.y.max(end.y) + padding) / self.cell_size).floor() as isize)
            .min(self.rows as isize - 1);

        let mut result = Vec::new();
        for r in min_row..=max_row {
            for c in min_col..=max_col {
                result.push(&self.cells[r as usize * self.cols + c as usize]);
            }
        }
        result
    }
}

/// Result of evaluating a proposed trace step against the existing graph.
pub enum TraceResult {
    /// Path is clear — create a new node at this position.
    Clear(Vec2),
    /// Landed near an existing intersection — snap to it.
    SnappedToNode(NodeId),
    /// Hit or landed near an existing edge — split it to form an intersection.
    SnappedToEdge {
        edge_id: EdgeId,
        intersection_pos: Vec2,
    },
}

/// Evaluates a single trace step `start_pos` → `proposed_pos` against the graph.
///
/// Resolution order:
/// 1. **Edge crossing** — did our path physically cross an existing road?
/// 2. **Node snap** — did we land close to an existing intersection?
/// 3. **Edge proximity (T-junction)** — did we land near the side of a road?
pub fn resolve_trace_step(
    graph: &RoadGraph,
    spatial: &SpatialHash,
    start_pos: Vec2,
    proposed_pos: Vec2,
    snap_radius: f32,
    current_node_id: NodeId,
) -> TraceResult {
    let cells = spatial.cells_for_region(start_pos, proposed_pos, snap_radius);
    let snap_sq = snap_radius * snap_radius;

    // Collect direct neighbours of the current node so we don't snap back
    // into the immediate adjacency of where we just came from.
    let current_neighbours: Vec<NodeId> = graph.nodes[current_node_id as usize]
        .edges
        .iter()
        .filter(|&&eid| graph.edges[eid as usize].active)
        .map(|&eid| graph.opposite(eid, current_node_id))
        .collect();

    // --- CHECK 1: EDGE CROSSING (physical intersection) ---
    // Must be checked BEFORE node snaps: if the trace segment physically
    // crosses an existing road, that crossing must be detected even if the
    // proposed position happens to land near a node on the far side.
    let mut closest_crossing: Option<(EdgeId, Vec2)> = None;
    let mut closest_crossing_dist = f32::MAX;

    for cell in &cells {
        for &e_id in &cell.edges {
            let edge = &graph.edges[e_id as usize];
            if !edge.active {
                continue;
            }
            if edge.start == current_node_id || edge.end == current_node_id {
                continue;
            }

            let e_start = graph.nodes[edge.start as usize].position;
            let e_end = graph.nodes[edge.end as usize].position;

            if let Some(intersect) = segment_intersection(start_pos, proposed_pos, e_start, e_end) {
                let dist = start_pos.distance_squared(intersect);
                if dist < closest_crossing_dist {
                    closest_crossing_dist = dist;
                    closest_crossing = Some((e_id, intersect));
                }
            }
        }
    }

    if let Some((edge_id, intersection_pos)) = closest_crossing {
        let edge = &graph.edges[edge_id as usize];
        let e_start = graph.nodes[edge.start as usize].position;
        let e_end = graph.nodes[edge.end as usize].position;

        let dist_to_start = intersection_pos.distance_squared(e_start);
        let dist_to_end = intersection_pos.distance_squared(e_end);

        if dist_to_start < snap_sq && dist_to_start <= dist_to_end {
            let nid = edge.start;
            if nid != current_node_id {
                return TraceResult::SnappedToNode(nid);
            }
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        } else if dist_to_end < snap_sq {
            let nid = edge.end;
            if nid != current_node_id {
                return TraceResult::SnappedToNode(nid);
            }
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        } else {
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        }
    }

    // --- CHECK 2: NODE SNAPPING ---
    let mut closest_node: Option<NodeId> = None;
    let mut closest_node_dist = f32::MAX;

    for cell in &cells {
        for &n_id in &cell.nodes {
            if n_id == current_node_id || current_neighbours.contains(&n_id) {
                continue;
            }
            let dist_sq = graph.nodes[n_id as usize]
                .position
                .distance_squared(proposed_pos);
            if dist_sq < snap_sq && dist_sq < closest_node_dist {
                closest_node_dist = dist_sq;
                closest_node = Some(n_id);
            }
        }
    }

    if let Some(n_id) = closest_node {
        return TraceResult::SnappedToNode(n_id);
    }

    // --- CHECK 3: EDGE PROXIMITY (T-junction) ---
    let mut closest_edge_hit: Option<(EdgeId, Vec2)> = None;
    let mut closest_edge_dist = f32::MAX;

    for cell in &cells {
        for &e_id in &cell.edges {
            let edge = &graph.edges[e_id as usize];
            if !edge.active {
                continue;
            }
            if edge.start == current_node_id || edge.end == current_node_id {
                continue;
            }
            if current_neighbours.contains(&edge.start) || current_neighbours.contains(&edge.end) {
                continue;
            }

            let e_start = graph.nodes[edge.start as usize].position;
            let e_end = graph.nodes[edge.end as usize].position;

            let proj = closest_point_on_segment(proposed_pos, e_start, e_end);
            let dist_to_edge_sq = proposed_pos.distance_squared(proj);

            if dist_to_edge_sq < snap_sq {
                let dist = start_pos.distance_squared(proj);
                if dist < closest_edge_dist {
                    closest_edge_dist = dist;
                    closest_edge_hit = Some((e_id, proj));
                }
            }
        }
    }

    if let Some((edge_id, intersection_pos)) = closest_edge_hit {
        let edge = &graph.edges[edge_id as usize];
        let e_start = graph.nodes[edge.start as usize].position;
        let e_end = graph.nodes[edge.end as usize].position;

        let dist_to_start = intersection_pos.distance_squared(e_start);
        let dist_to_end = intersection_pos.distance_squared(e_end);

        if dist_to_start < snap_sq && dist_to_start <= dist_to_end {
            let nid = edge.start;
            if nid != current_node_id {
                return TraceResult::SnappedToNode(nid);
            }
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        } else if dist_to_end < snap_sq {
            let nid = edge.end;
            if nid != current_node_id {
                return TraceResult::SnappedToNode(nid);
            }
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        } else {
            return TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            };
        }
    }

    TraceResult::Clear(proposed_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RoadType;

    #[test]
    fn insert_and_find_node() {
        let mut graph = RoadGraph::default();
        let origin = graph.add_node(Vec2::new(0.0, 0.0)); // tracer's current node
        let target = graph.add_node(Vec2::new(5.0, 5.0));

        let mut sh = SpatialHash::new(100.0, 100.0, 10.0);
        sh.insert_node(origin, Vec2::new(0.0, 0.0));
        sh.insert_node(target, Vec2::new(5.0, 5.0));

        let result = resolve_trace_step(
            &graph,
            &sh,
            Vec2::new(0.0, 0.0),
            Vec2::new(5.5, 5.5),
            2.0,
            origin,
        );
        assert!(matches!(result, TraceResult::SnappedToNode(1)));
    }

    #[test]
    fn crossing_edge_detected() {
        let mut graph = RoadGraph::default();
        let origin = graph.add_node(Vec2::new(5.0, 0.0)); // tracer's current node
        let a = graph.add_node(Vec2::new(0.0, 5.0));
        let b = graph.add_node(Vec2::new(10.0, 5.0));
        let e = graph.add_edge(a, b, RoadType::Major);

        let mut sh = SpatialHash::new(100.0, 100.0, 10.0);
        sh.insert_node(origin, graph.node_pos(origin));
        sh.insert_node(a, graph.node_pos(a));
        sh.insert_node(b, graph.node_pos(b));
        sh.insert_edge(e, graph.node_pos(a), graph.node_pos(b));

        let result = resolve_trace_step(
            &graph,
            &sh,
            Vec2::new(5.0, 0.0),
            Vec2::new(5.0, 10.0),
            1.0,
            origin,
        );
        match result {
            TraceResult::SnappedToEdge {
                edge_id,
                intersection_pos,
            } => {
                assert_eq!(edge_id, e);
                assert!((intersection_pos - Vec2::new(5.0, 5.0)).length() < 1e-4);
            }
            _ => panic!("expected SnappedToEdge"),
        }
    }

    #[test]
    fn clear_when_no_obstacles() {
        let mut graph = RoadGraph::default();
        let origin = graph.add_node(Vec2::new(40.0, 40.0));
        let sh = SpatialHash::new(100.0, 100.0, 10.0);

        let target = Vec2::new(50.0, 50.0);
        let result = resolve_trace_step(&graph, &sh, Vec2::new(40.0, 40.0), target, 1.0, origin);
        match result {
            TraceResult::Clear(pos) => assert!((pos - target).length() < 1e-5),
            _ => panic!("expected Clear"),
        }
    }
}
