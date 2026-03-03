use criterion::{Criterion, criterion_group, criterion_main};
use symbios_ground::HeightMap;
use symbios_tensor::{TensorConfig, generate_roads};

fn bench_generate_roads(c: &mut Criterion) {
    let hm = HeightMap::new(64, 64, 2.0);
    let config = TensorConfig {
        seed: 42,
        major_road_dist: 25.0,
        minor_road_dist: 12.0,
        ..Default::default()
    };

    c.bench_function("generate_roads_64x64", |b| {
        b.iter(|| generate_roads(&hm, &config))
    });
}

criterion_group!(benches, bench_generate_roads);
criterion_main!(benches);
