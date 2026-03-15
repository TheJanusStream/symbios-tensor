use symbios_ground::HeightMap;
use symbios_tensor::{
    LotConfig, RoadMeshConfig, RoadType, TensorConfig, carve_roads, extract_blocks, extract_lots,
    generate_roads,
};

fn flat_heightmap() -> HeightMap {
    HeightMap::new(64, 64, 2.0)
}

#[test]
fn generates_nonempty_graph_on_flat_terrain() {
    let hm = flat_heightmap();
    let config = TensorConfig {
        seed: 1,
        step_size: 2.0,
        major_road_dist: 20.0,
        minor_road_dist: 10.0,
        snap_radius: 3.0,
        max_trace_steps: 100,
        ..Default::default()
    };
    let graph = generate_roads(&hm, &config).expect("generate_roads");

    assert!(!graph.nodes.is_empty(), "should produce nodes");
    assert!(!graph.edges.is_empty(), "should produce edges");

    // All active edges should reference valid nodes
    for edge in &graph.edges {
        if edge.active {
            assert!(
                (edge.start as usize) < graph.nodes.len(),
                "edge.start out of bounds"
            );
            assert!(
                (edge.end as usize) < graph.nodes.len(),
                "edge.end out of bounds"
            );
        }
    }
}

#[test]
fn road_types_are_present() {
    let hm = flat_heightmap();
    let config = TensorConfig {
        seed: 7,
        major_road_dist: 25.0,
        minor_road_dist: 12.0,
        ..Default::default()
    };
    let graph = generate_roads(&hm, &config).expect("generate_roads");

    let has_major = graph
        .edges
        .iter()
        .any(|e| e.active && e.road_type == RoadType::Major);
    let has_minor = graph
        .edges
        .iter()
        .any(|e| e.active && e.road_type == RoadType::Minor);

    assert!(has_major, "should have major roads");
    assert!(has_minor, "should have minor roads");
}

#[test]
fn extract_blocks_finds_polygons() {
    let hm = flat_heightmap();
    let config = TensorConfig {
        seed: 42,
        major_road_dist: 30.0,
        minor_road_dist: 15.0,
        ..Default::default()
    };
    let mut graph = generate_roads(&hm, &config).expect("generate_roads");
    extract_blocks(&mut graph);

    // On a flat grid the tracer should form a regular grid → enclosed blocks
    // We don't assert an exact count but there should be at least one.
    assert!(
        !graph.blocks.is_empty(),
        "should extract at least one city block from the grid"
    );

    for block in &graph.blocks {
        assert!(
            block.perimeter.len() >= 3,
            "block perimeter must be a polygon"
        );
    }
}

#[test]
fn carve_modifies_heightmap() {
    let mut hm = HeightMap::new(32, 32, 1.0);
    // Create a simple slope
    for z in 0..32 {
        for x in 0..32 {
            hm.set(x, z, z as f32 * 0.1);
        }
    }
    let original_sum: f32 = hm.data().iter().sum();

    let config = TensorConfig {
        seed: 99,
        step_size: 1.0,
        major_road_dist: 10.0,
        minor_road_dist: 5.0,
        snap_radius: 2.0,
        max_trace_steps: 50,
        tracer_inertia: 0.8,
        water_level: 0.0,
    };
    let graph = generate_roads(&hm, &config).expect("generate_roads");

    if graph.edges.iter().any(|e| e.active) {
        let _ = carve_roads(&graph, &mut hm, &RoadMeshConfig::default(), 1.0);
        let carved_sum: f32 = hm.data().iter().sum();
        assert!(
            (carved_sum - original_sum).abs() > 1e-3,
            "carving should modify the heightmap"
        );
    }
}

#[test]
fn graph_serialization_roundtrip() {
    let hm = flat_heightmap();
    let config = TensorConfig::default();
    let graph = generate_roads(&hm, &config).expect("generate_roads");

    let json = serde_json::to_string(&graph).expect("serialize");
    let restored: symbios_tensor::RoadGraph = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(graph.nodes.len(), restored.nodes.len());
    assert_eq!(graph.edges.len(), restored.edges.len());
}

#[test]
fn extract_lots_produces_buildings() {
    use symbios_tensor::{CityBlock, RoadGraph, RoadType};

    // Build a manual graph with a known enclosed rectangular block (30x20)
    let mut graph = RoadGraph::default();
    let n0 = graph.add_node(glam::Vec2::new(0.0, 0.0));
    let n1 = graph.add_node(glam::Vec2::new(30.0, 0.0));
    let n2 = graph.add_node(glam::Vec2::new(30.0, 20.0));
    let n3 = graph.add_node(glam::Vec2::new(0.0, 20.0));
    graph.add_edge(n0, n1, RoadType::Minor);
    graph.add_edge(n1, n2, RoadType::Minor);
    graph.add_edge(n2, n3, RoadType::Minor);
    graph.add_edge(n3, n0, RoadType::Minor);
    // CW winding (negative signed area) matches extract_blocks convention
    graph.blocks.push(CityBlock {
        perimeter: vec![n0, n3, n2, n1],
    });

    // Heightmap large enough to cover the block, no water
    let hm = HeightMap::new(32, 32, 2.0);
    let lot_config = LotConfig::default();
    let lots = extract_lots(&graph, &hm, f32::NEG_INFINITY, &lot_config);

    assert!(!lots.is_empty(), "should produce at least one building lot");

    for lot in &lots {
        assert!(lot.width > 0.0, "lot width must be positive");
        assert!(lot.depth > 0.0, "lot depth must be positive");
        assert!(
            lot.width >= lot_config.min_width,
            "lot width {} below minimum {}",
            lot.width,
            lot_config.min_width
        );
        assert!(
            lot.depth >= lot_config.min_depth,
            "lot depth {} below minimum {}",
            lot.depth,
            lot_config.min_depth
        );
    }
}
