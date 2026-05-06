use criterion::{Criterion, criterion_group, criterion_main};
use symbios_ground::HeightMap;
use symbios_tensor::{RationalizeConfig, TensorConfig, generate_roads, rationalize_graph};

fn bench_generate_roads(c: &mut Criterion) {
    let hm = HeightMap::new(64, 64, 2.0);
    let config = TensorConfig {
        seed: 42,
        major_road_dist: 25.0,
        minor_road_dist: 12.0,
        ..Default::default()
    };

    c.bench_function("generate_roads_64x64", |b| {
        b.iter(|| generate_roads(&hm, &config).expect("generate_roads"))
    });
}

fn bench_rationalize(c: &mut Criterion) {
    let mut hm = HeightMap::new(64, 64, 2.0);
    for z in 0..64 {
        for x in 0..64 {
            hm.set(x, z, ((x + z) as f32 * 0.1).sin() * 5.0);
        }
    }
    let cfg = TensorConfig {
        seed: 42,
        major_road_dist: 25.0,
        minor_road_dist: 12.0,
        ..Default::default()
    };
    let base_graph = generate_roads(&hm, &cfg).expect("generate_roads");

    // No early termination (always runs full pass count).
    c.bench_function("rationalize_no_early_term", |b| {
        let rcfg = RationalizeConfig {
            convergence_tolerance: 0.0,
            ..Default::default()
        };
        b.iter(|| {
            let mut g = base_graph.clone();
            rationalize_graph(&mut g, &hm, &rcfg);
            g
        })
    });

    // Early termination at default tolerance.
    c.bench_function("rationalize_early_term", |b| {
        let rcfg = RationalizeConfig::default();
        b.iter(|| {
            let mut g = base_graph.clone();
            rationalize_graph(&mut g, &hm, &rcfg);
            g
        })
    });
}

criterion_group!(benches, bench_generate_roads, bench_rationalize);
criterion_main!(benches);
