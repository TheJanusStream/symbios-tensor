use glam::Vec2;
use serde::{Deserialize, Serialize};

pub type NodeId = u32;
pub type EdgeId = u32;
pub type BlockId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoadType {
    Major,
    Minor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadNode {
    pub position: Vec2,
    pub edges: Vec<EdgeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadEdge {
    pub start: NodeId,
    pub end: NodeId,
    pub road_type: RoadType,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityBlock {
    /// Ordered list of node indices forming a closed polygon perimeter.
    pub perimeter: Vec<NodeId>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoadGraph {
    pub nodes: Vec<RoadNode>,
    pub edges: Vec<RoadEdge>,
    pub blocks: Vec<CityBlock>,
}

impl RoadGraph {
    pub fn add_node(&mut self, position: Vec2) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(RoadNode {
            position,
            edges: Vec::new(),
        });
        id
    }

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
