//! Arena-based road graph with soft-deletion support for edge splitting.
//!
//! The graph stores nodes, edges, and extracted city blocks in flat `Vec`s
//! indexed by [`NodeId`], [`EdgeId`], and [`BlockId`] (all `u32`). Edges
//! carry an `active` flag so that [`RoadGraph::split_edge`] can deactivate
//! the original while inserting two replacement segments without invalidating
//! existing indices.

use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Index into [`RoadGraph::nodes`].
pub type NodeId = u32;
/// Index into [`RoadGraph::edges`].
pub type EdgeId = u32;
/// Index into [`RoadGraph::blocks`].
pub type BlockId = u32;

/// Classification of a road segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoadType {
    /// Contour-following avenue.
    Major,
    /// Gradient-following street.
    Minor,
}

/// A road intersection or endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadNode {
    /// World-space position (X, Z in a Y-up coordinate system).
    pub position: Vec2,
    /// Indices of all edges incident to this node (both active and inactive).
    pub edges: Vec<EdgeId>,
}

/// A directed road segment connecting two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadEdge {
    /// Source node index.
    pub start: NodeId,
    /// Destination node index.
    pub end: NodeId,
    /// Whether this is a major (contour) or minor (gradient) road.
    pub road_type: RoadType,
    /// `false` after the edge has been split — superseded by two child edges.
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityBlock {
    /// Ordered list of node indices forming a closed polygon perimeter.
    pub perimeter: Vec<NodeId>,
}

/// The complete road network: nodes, edges, and extracted city blocks.
///
/// All public fields are arena `Vec`s — indices ([`NodeId`], [`EdgeId`],
/// [`BlockId`]) are stable for the lifetime of the graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoadGraph {
    /// Road intersections and endpoints.
    pub nodes: Vec<RoadNode>,
    /// Road segments (check [`RoadEdge::active`] before traversal).
    pub edges: Vec<RoadEdge>,
    /// Enclosed city blocks extracted by [`crate::polygons::extract_blocks`].
    pub blocks: Vec<CityBlock>,
}

impl RoadGraph {
    /// Inserts a new node and returns its [`NodeId`].
    pub fn add_node(&mut self, position: Vec2) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(RoadNode {
            position,
            edges: Vec::new(),
        });
        id
    }

    /// Inserts a new active edge between two nodes and returns its [`EdgeId`].
    pub fn add_edge(&mut self, start: NodeId, end: NodeId, road_type: RoadType) -> EdgeId {
        let id = self.edges.len() as EdgeId;
        self.edges.push(RoadEdge {
            start,
            end,
            road_type,
            active: true,
        });
        self.nodes[start as usize].edges.push(id);
        self.nodes[end as usize].edges.push(id);
        id
    }

    /// Deactivates an edge and splits it at `split_pos`, returning `(new_node, edge_a, edge_b)`.
    ///
    /// The original edge is marked inactive. Two new edges are created connecting
    /// the original endpoints through the new split node.
    pub fn split_edge(&mut self, edge_id: EdgeId, split_pos: Vec2) -> (NodeId, EdgeId, EdgeId) {
        let edge = &self.edges[edge_id as usize];
        let start = edge.start;
        let end = edge.end;
        let road_type = edge.road_type;

        self.edges[edge_id as usize].active = false;

        let mid = self.add_node(split_pos);
        let ea = self.add_edge(start, mid, road_type);
        let eb = self.add_edge(mid, end, road_type);

        (mid, ea, eb)
    }

    /// Returns the other endpoint of an edge relative to `node_id`.
    pub fn opposite(&self, edge_id: EdgeId, node_id: NodeId) -> NodeId {
        let edge = &self.edges[edge_id as usize];
        if edge.start == node_id {
            edge.end
        } else {
            edge.start
        }
    }

    /// Returns position of a node by id.
    pub fn node_pos(&self, id: NodeId) -> Vec2 {
        self.nodes[id as usize].position
    }
}
