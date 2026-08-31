use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use symbios_ground::HeightMap;
use symbios_tensor::{
    LotConfig, RationalizeConfig, RoadMeshConfig, RoadType, TensorConfig, TensorFieldConfig,
    carve_lots, carve_roads, extract_blocks, extract_lots, generate_road_meshes, generate_roads,
    rationalize_graph,
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
        ..Default::default()
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

/// Property test: drives the full pipeline with a randomized population of
/// tensor fields (heightmaps + configs) and confirms no panic occurs. This
/// is the regression net for the unwrap audit (#59).
///
/// We use 200 generations for CI runtime budget; the scenarios below cover
/// the categories the audit was concerned with: degenerate intersections,
/// near-flat slopes, tiny worlds, large step sizes relative to features,
/// and the new water-policy code paths.
#[test]
fn pipeline_survives_random_inputs() {
    let mut rng = Pcg64::seed_from_u64(0x0C1A_55E5_DEAD_BEEF);

    for trial in 0..200u32 {
        // Vary heightmap size and topology.
        let cells = rng.random_range(8..=32);
        let scale = rng.random_range(0.5..=4.0_f32);
        let mut hm = HeightMap::new(cells, cells, scale);
        let mode = trial % 4;
        for z in 0..cells {
            for x in 0..cells {
                let h = match mode {
                    0 => 0.0, // flat
                    1 => x as f32 * rng.random_range(-0.05..0.05),
                    2 => ((x + z) as f32 * 0.1).sin() * rng.random_range(0.0..2.0),
                    _ => rng.random_range(-1.0..2.0),
                };
                hm.set(x, z, h);
            }
        }

        let world_w = hm.world_width();
        let cfg = TensorConfig {
            seed: trial as u64,
            step_size: rng.random_range(0.5..3.0),
            major_road_dist: rng.random_range(5.0..(world_w * 0.5).max(6.0)),
            minor_road_dist: rng.random_range(2.0..(world_w * 0.25).max(3.0)),
            snap_radius: rng.random_range(1.0..4.0),
            max_trace_steps: rng.random_range(20..150),
            tracer_inertia: rng.random_range(0.0..0.95),
            water_level: if trial % 3 == 0 {
                -0.1
            } else {
                f32::NEG_INFINITY
            },
            field: TensorFieldConfig {
                jitter_amplitude: if trial % 2 == 0 { 0.0 } else { 0.2 },
                ..Default::default()
            },
        };

        let Ok(mut graph) = generate_roads(&hm, &cfg) else {
            // Invalid randomized config — that's a Result, not a panic. OK.
            continue;
        };
        rationalize_graph(&mut graph, &hm, &RationalizeConfig::default());
        extract_blocks(&mut graph);
        let mut hm_copy = hm.clone();
        let lots = extract_lots(&graph, &mut hm_copy, &LotConfig::default());
        let mesh_cfg = RoadMeshConfig::default();
        let _ = carve_roads(&graph, &mut hm_copy, &mesh_cfg, 1.0);
        carve_lots(&lots, &mut hm_copy, 1.0, None);
        let _meshes = generate_road_meshes(&graph, &hm_copy, &mesh_cfg);
    }
}

#[test]
fn no_nodes_outside_world_bounds() {
    // Regression: RK2 midpoint sampling could fall outside world bounds and
    // emit a node just outside the world. With the fix, no node should ever
    // sit outside [0, world_w] x [0, world_d].
    let mut hm = HeightMap::new(32, 32, 2.0);
    // Build a slope so the tensor field is well-defined and traces actually
    // run to the boundary instead of exiting on the flat-fallback.
    for z in 0..32 {
        for x in 0..32 {
            hm.set(x, z, x as f32 * 0.5);
        }
    }
    let world_w = hm.world_width();
    let world_d = hm.world_depth();

    let config = TensorConfig {
        seed: 17,
        step_size: 2.0,
        major_road_dist: 8.0,
        minor_road_dist: 4.0,
        snap_radius: 2.5,
        max_trace_steps: 200,
        tracer_inertia: 0.5,
        water_level: f32::NEG_INFINITY,
        ..Default::default()
    };
    let graph = generate_roads(&hm, &config).expect("generate_roads");

    for (i, node) in graph.nodes.iter().enumerate() {
        let p = node.position;
        assert!(
            p.x >= 0.0 && p.x < world_w && p.y >= 0.0 && p.y < world_d,
            "node {i} at {:?} outside world bounds [{}, {})x[{}, {})",
            p,
            0.0,
            world_w,
            0.0,
            world_d
        );
    }
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
    let mut hm = HeightMap::new(32, 32, 2.0);
    let lot_config = LotConfig::default();
    let lots = extract_lots(&graph, &mut hm, &lot_config);

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
