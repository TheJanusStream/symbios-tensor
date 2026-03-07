use symbios_ground::HeightMap;
use symbios_tensor::{
    LotConfig, RoadType, TensorConfig, block_centroid, carve_roads, extract_blocks, extract_lots,
    generate_roads,
};

fn main() {
    // 1. Create a flat 128×128 heightmap
    let mut hm = HeightMap::new(64, 64, 2.0);

    // 2. Generate the road graph
    let config = TensorConfig {
        seed: 42,
        major_road_dist: 30.0,
        minor_road_dist: 15.0,
        ..Default::default()
    };
    let mut graph = generate_roads(&hm, &config);

    let active = graph.edges.iter().filter(|e| e.active).count();
    let major = graph
        .edges
        .iter()
        .filter(|e| e.active && e.road_type == RoadType::Major)
        .count();
    let minor = graph
        .edges
        .iter()
        .filter(|e| e.active && e.road_type == RoadType::Minor)
        .count();

    println!("Road Graph:");
    println!("  nodes: {}", graph.nodes.len());
    println!(
        "  edges: {} active ({} major, {} minor)",
        active, major, minor
    );

    // 3. Extract city blocks
    extract_blocks(&mut graph);
    println!("  blocks: {}", graph.blocks.len());

    for (i, block) in graph.blocks.iter().enumerate() {
        let centroid = block_centroid(block, &graph);
        println!(
            "  block {}: {} vertices, centroid ({:.1}, {:.1})",
            i,
            block.perimeter.len(),
            centroid.x,
            centroid.y
        );
    }

    // 4. Extract building lots
    let lot_config = LotConfig::default();
    let lots = extract_lots(&graph, &hm, config.water_level, &lot_config);
    println!("\nBuilding lots: {}", lots.len());
    for (i, lot) in lots.iter().enumerate() {
        println!(
            "  lot {i}: pos=({:.1}, {:.1}), rot={:.2}rad, {:.1}x{:.1}",
            lot.position.x, lot.position.y, lot.rotation, lot.width, lot.depth
        );
    }

    // 5. Carve roads into the terrain
    let _ = carve_roads(&graph, &mut hm, 2.0, 1.0);
    println!("\nTerrain carved along {} active road edges.", active);
}
