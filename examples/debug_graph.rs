use symbios_ground::HeightMap;
use symbios_tensor::{
    block_centroid, carve_roads, extract_blocks, generate_roads, RoadType, TensorConfig,
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
    println!("  edges: {} active ({} major, {} minor)", active, major, minor);

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

    // 4. Carve roads into the terrain
    carve_roads(&graph, &mut hm, 2.0);
    println!("\nTerrain carved along {} active road edges.", active);
}
