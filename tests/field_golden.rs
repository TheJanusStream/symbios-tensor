//! Golden traces: the raw road graph `generate_roads` produces for a fixed
//! heightmap and configs, pinned as a hash of every node position (to the
//! bit) and every edge. A consumer that re-traces a saved network at load
//! (Overlands does, under buildings it saved on the old layout) depends on
//! an unchanged config tracing the same streets across releases, so a change
//! to the field that is meant to be opt-in must leave these hashes alone.
//! The values were recorded at c3f2875, the commit before the basis fields
//! and the smoothing scale (#68).
//!
//! Only arithmetic that is correctly rounded on every IEEE platform reaches
//! these hashes: the heightmap is integer-hash value noise (no `sin`), the
//! configs carry no jitter (its `sin` is the platform's libm), and the graph
//! is hashed before `rationalize_graph` (its fillets call `acos` and `tan`).
//! A libm that differs by an ulp - CI's glibc and a developer's can - would
//! otherwise fail this test for a reason that has nothing to do with it.

use symbios_ground::HeightMap;
use symbios_tensor::{
    LotConfig, MathMode, RationalizeConfig, RoadGraph, RoadType, TensorConfig, WaterPolicy,
    extract_blocks, extract_lots, generate_roads, rationalize_graph,
};

/// A lattice hash in `[0, 1)`: integer mixing only.
fn lattice(x: i64, z: i64, salt: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841) ^ salt;
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Value noise: the lattice smoothly interpolated (a smoothstep blend),
/// with a period of `cell` world units.
fn value_noise(x: f32, z: f32, cell: f32, salt: u32) -> f32 {
    let (gx, gz) = (x / cell, z / cell);
    let (ix, iz) = (gx.floor(), gz.floor());
    let (fx, fz) = (gx - ix, gz - iz);
    let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (ix, iz) = (ix as i64, iz as i64);
    let a = lattice(ix, iz, salt);
    let b = lattice(ix + 1, iz, salt);
    let c = lattice(ix, iz + 1, salt);
    let d = lattice(ix + 1, iz + 1, salt);
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sz
}

/// Rolling terrain with features at three scales and a fine roughness, so
/// the field has slopes, near-flats and noise. Heights span about
/// `amplitude * [0, 1.75]`.
pub fn rolling_heightmap(cells: usize, scale: f32, amplitude: f32) -> HeightMap {
    let mut hm = HeightMap::new(cells, cells, scale);
    for z in 0..cells {
        for x in 0..cells {
            let (fx, fz) = (x as f32 * scale, z as f32 * scale);
            let h = value_noise(fx, fz, 96.0, 1)
                + 0.5 * value_noise(fx, fz, 37.0, 2)
                + 0.2 * value_noise(fx, fz, 11.0, 3)
                + 0.05 * value_noise(fx, fz, 3.0, 4);
            hm.set(x, z, amplitude * h);
        }
    }
    hm
}

/// FNV-1a over the graph's nodes (positions to the bit) and edges.
pub fn fingerprint(graph: &RoadGraph) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for byte in v.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(graph.nodes.len() as u64);
    for node in &graph.nodes {
        eat(u64::from(node.position.x.to_bits()));
        eat(u64::from(node.position.y.to_bits()));
    }
    eat(graph.edges.len() as u64);
    for edge in &graph.edges {
        eat(u64::from(edge.start));
        eat(u64::from(edge.end));
        eat(match edge.road_type {
            RoadType::Major => 1,
            RoadType::Minor => 2,
        });
        eat(u64::from(edge.active));
    }
    hash
}

fn traced(hm: &HeightMap, config: &TensorConfig) -> RoadGraph {
    generate_roads(hm, config).expect("the golden configs are valid")
}

/// The configs pinned, by name: the default terrain blend, the
/// axis-aligned grid Overlands' Grid style asks for, wide flat thresholds
/// that put much of the map inside the terrain-to-grid blend, and a
/// shoreline.
pub fn golden_configs(hm: &HeightMap) -> Vec<(&'static str, TensorConfig)> {
    let base = TensorConfig {
        seed: 7,
        major_road_dist: 60.0,
        minor_road_dist: 30.0,
        ..TensorConfig::default()
    };
    let mut grid = base.clone();
    grid.field.flat_threshold_low = f32::MAX;
    grid.field.flat_threshold_high = f32::MAX;
    let mut blend = base.clone();
    blend.field.flat_threshold_low = 0.03;
    blend.field.flat_threshold_high = 0.12;
    let mut shore = base.clone();
    // A line through the middle of the map's heights floods its low parts.
    let (low, high) = hm
        .data()
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    shore.water_level = low + 0.4 * (high - low);
    vec![
        ("default", base),
        ("grid", grid),
        ("blend", blend),
        ("shore", shore),
    ]
}

/// The pinned hashes, in `golden_configs` order, for a 128-cell map at 2 m
/// and 12 m of amplitude.
const GOLDEN: [u64; 4] = [
    0xe32a_f262_babb_c4d7,
    0x5029_7cc4_6b1f_edb0,
    0xd36a_9c1b_ccbb_970d,
    0x3cfa_1ac1_8776_0f83,
];

#[test]
fn an_unchanged_config_traces_the_same_streets() {
    let hm = rolling_heightmap(128, 2.0, 12.0);
    let configs = golden_configs(&hm);
    let got: Vec<u64> = configs
        .iter()
        .map(|(_, c)| fingerprint(&traced(&hm, c)))
        .collect();
    for ((name, c), value) in configs.iter().zip(&got) {
        let g = traced(&hm, c);
        println!(
            "golden {name}: {value:#018x} ({} nodes, {} edges, water {})",
            g.nodes.len(),
            g.edges.len(),
            c.water_level
        );
    }
    assert_eq!(got, GOLDEN, "an unchanged config traced different streets");
}

/// The four configs must each trace something different, or a hash could
/// be pinning one graph four times (the shore's water, say, flooding
/// nothing) and prove less than it claims.
#[test]
fn the_golden_configs_are_four_different_traces() {
    let hm = rolling_heightmap(128, 2.0, 12.0);
    let mut got: Vec<u64> = golden_configs(&hm)
        .iter()
        .map(|(_, c)| fingerprint(&traced(&hm, c)))
        .collect();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got.len(), 4, "two golden configs trace the same graph");
}

// --- Portable layouts (#70) -------------------------------------------------

/// Everything a consumer grows buildings from, for `config` traced with
/// [`MathMode::Portable`]: the graph after `rationalize_graph`, its blocks
/// and its lots, hashed to the bit - every platform transcendental the
/// derivation calls (the jitter, the fillets, the edge order, the lots'
/// frames) on the way.
fn portable_layout(hm: &HeightMap, config: &TensorConfig, lots: &LotConfig) -> u64 {
    let config = TensorConfig {
        math: MathMode::Portable,
        ..config.clone()
    };
    let mut graph = traced(hm, &config);
    assert_eq!(graph.math, MathMode::Portable, "the trace records its math");
    rationalize_graph(&mut graph, hm, &RationalizeConfig::default());
    extract_blocks(&mut graph);
    let mut ground = hm.clone();
    let lots = extract_lots(&graph, &mut ground, lots);
    let mut hash = fingerprint(&graph);
    let mut eat = |v: u64| {
        for byte in v.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(graph.blocks.len() as u64);
    for block in &graph.blocks {
        eat(block.perimeter.len() as u64);
        block.perimeter.iter().for_each(|&n| eat(u64::from(n)));
    }
    eat(lots.len() as u64);
    for lot in &lots {
        for v in [
            lot.position.x,
            lot.position.y,
            lot.frontage_center.x,
            lot.frontage_center.y,
            lot.rotation,
            lot.width,
            lot.depth,
        ] {
            eat(u64::from(v.to_bits()));
        }
        eat(u64::from(lot.is_shoreline));
    }
    hash
}

/// The portable layouts pinned: the base config with the jitter
/// Overlands' organic streets trace with, and the same over a shoreline
/// whose lots are tagged where they touch the water.
fn portable_configs(hm: &HeightMap) -> Vec<(&'static str, TensorConfig, LotConfig)> {
    let configs = golden_configs(hm);
    let mut jittered = configs[0].1.clone();
    jittered.field.jitter_amplitude = 0.15;
    jittered.tracer_inertia = 0.6;
    let mut shore = configs[3].1.clone();
    shore.field.jitter_amplitude = 0.15;
    let shore_lots = LotConfig {
        water_level: shore.water_level,
        water_policy: WaterPolicy::TagShoreline,
        ..LotConfig::default()
    };
    vec![
        ("jittered", jittered, LotConfig::default()),
        ("shore", shore, shore_lots),
    ]
}

/// The pinned hashes, in `portable_configs` order, on the 128-cell map -
/// recorded with every platform maths function interposed and nudged three
/// ulps (an `LD_PRELOAD` shim) and without it, alike, and no platform call
/// made on the way.
const PORTABLE_GOLDEN: [u64; 2] = [0x41c7_dc6e_9879_450c, 0x1522_dc0b_249e_8766];

/// A layout traced with [`MathMode::Portable`] derives the same streets,
/// blocks and lots on every platform. This machine's glibc and CI's answer
/// `sin`, `acos` and `atan2` differently in the last bit, so a call that
/// escaped the portable mode would fail this pin on one of them.
#[test]
fn a_portable_layout_derives_the_same_lots_on_every_platform() {
    let hm = rolling_heightmap(128, 2.0, 12.0);
    let got: Vec<u64> = portable_configs(&hm)
        .iter()
        .map(|(name, c, lots)| {
            let value = portable_layout(&hm, c, lots);
            println!("portable {name}: {value:#018x}");
            value
        })
        .collect();
    assert_eq!(
        got, PORTABLE_GOLDEN,
        "a portable layout derived differently"
    );
}

/// The portable pins hold lots, or they would prove nothing about the lot
/// frames.
#[test]
fn the_portable_layouts_grow_lots() {
    let hm = rolling_heightmap(128, 2.0, 12.0);
    for (name, c, lots) in portable_configs(&hm) {
        let mut graph = traced(
            &hm,
            &TensorConfig {
                math: MathMode::Portable,
                ..c
            },
        );
        rationalize_graph(&mut graph, &hm, &RationalizeConfig::default());
        extract_blocks(&mut graph);
        let mut ground = hm.clone();
        let grown = extract_lots(&graph, &mut ground, &lots);
        assert!(grown.len() > 10, "{name} grew {} lots", grown.len());
    }
}
