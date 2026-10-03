//! Basis fields and the smoothing scale (#68), judged on what the tracer
//! draws with them: the streets of a whole traced graph, not one sample.

use glam::Vec2;
use symbios_ground::HeightMap;
use symbios_tensor::{
    BasisField, RoadGraph, RoadType, TensorConfig, TensorFieldConfig, generate_roads,
};

/// Each active edge of `road_type` as (midpoint, unit direction).
fn edges(graph: &RoadGraph, road_type: RoadType) -> Vec<(Vec2, Vec2)> {
    graph
        .edges
        .iter()
        .filter(|e| e.active && e.road_type == road_type)
        .filter_map(|e| {
            let a = graph.nodes[e.start as usize].position;
            let b = graph.nodes[e.end as usize].position;
            let d = b - a;
            (d.length() > 1e-3).then(|| ((a + b) * 0.5, d.normalize()))
        })
        .collect()
}

/// The share of `edges` that `keep` accepts.
fn share(edges: &[(Vec2, Vec2)], keep: impl Fn(&(Vec2, Vec2)) -> bool) -> f32 {
    edges.iter().filter(|e| keep(e)).count() as f32 / edges.len().max(1) as f32
}

fn base_config(field: TensorFieldConfig) -> TensorConfig {
    TensorConfig {
        seed: 3,
        major_road_dist: 40.0,
        minor_road_dist: 20.0,
        field,
        ..TensorConfig::default()
    }
}

/// #68: on flat ground a radial field traces ring roads round its centre
/// (major) and spokes out from it (minor). The control is the same ground
/// without it, where the field falls back to an axis grid and the same
/// measure fails.
#[test]
fn a_radial_field_traces_rings_and_spokes() {
    let hm = HeightMap::new(128, 128, 2.0);
    let centre = Vec2::new(128.0, 128.0);
    let radial = TensorFieldConfig {
        terrain_weight: 0.0,
        basis: vec![BasisField::Radial {
            center: centre,
            radius: 160.0,
            strength: 1.0,
        }],
        ..TensorFieldConfig::default()
    };
    // Edges in the band 30-90 m out, where the rings are well formed and
    // short of the field's fading edge.
    let in_band = |(mid, _): &(Vec2, Vec2)| (30.0..90.0).contains(&mid.distance(centre));
    let ringness = |graph: &RoadGraph| {
        let majors: Vec<_> = edges(graph, RoadType::Major)
            .into_iter()
            .filter(in_band)
            .collect();
        let minors: Vec<_> = edges(graph, RoadType::Minor)
            .into_iter()
            .filter(in_band)
            .collect();
        let out = |mid: Vec2| (mid - centre).normalize();
        (
            majors.len(),
            share(&majors, |(mid, dir)| dir.dot(out(*mid)).abs() < 0.3),
            minors.len(),
            share(&minors, |(mid, dir)| dir.dot(out(*mid)).abs() > 0.95),
        )
    };

    let graph = generate_roads(&hm, &base_config(radial)).expect("traces");
    let (majors, rings, minors, spokes) = ringness(&graph);
    assert!(
        majors > 30 && minors > 30,
        "a city's worth of streets: {majors} / {minors}"
    );
    assert!(rings > 0.85, "major roads must ring the centre: {rings}");
    assert!(spokes > 0.85, "minor roads must run out from it: {spokes}");

    let plain = generate_roads(&hm, &base_config(TensorFieldConfig::default())).expect("traces");
    let (_, rings, _, spokes) = ringness(&plain);
    assert!(
        rings < 0.6 || spokes < 0.6,
        "the control: an axis grid is no ring city"
    );
}

/// The angle of a direction from +X, folded into [0, 90): a grid's two
/// families read as one.
fn grid_angle(dir: Vec2) -> f32 {
    dir.y.atan2(dir.x).to_degrees().rem_euclid(90.0)
}

/// #68: a grid field turns a district's streets to its angle; without it
/// the flat-ground fallback lays them square to the axes.
#[test]
fn a_grid_field_turns_the_streets() {
    let hm = HeightMap::new(128, 128, 2.0);
    let turned = TensorFieldConfig {
        terrain_weight: 0.0,
        basis: vec![BasisField::Grid {
            center: Vec2::new(128.0, 128.0),
            angle: 30.0_f32.to_radians(),
            radius: 400.0,
            strength: 1.0,
        }],
        ..TensorFieldConfig::default()
    };
    let at = |config: TensorFieldConfig, angle: f32| {
        let graph = generate_roads(&hm, &base_config(config)).expect("traces");
        let all: Vec<_> = edges(&graph, RoadType::Major)
            .into_iter()
            .chain(edges(&graph, RoadType::Minor))
            .collect();
        share(&all, |(_, dir)| {
            let off = (grid_angle(*dir) - angle).abs();
            off.min(90.0 - off) < 5.0
        })
    };
    let on_grid = at(turned, 30.0);
    assert!(
        on_grid > 0.85,
        "streets must follow the turned grid: {on_grid}"
    );
    assert!(
        at(TensorFieldConfig::default(), 30.0) < 0.2,
        "the control: the plain flat-ground grid is square to the axes"
    );
}

/// A slope along +X roughened by a fine deterministic noise.
fn rough_slope(cells: usize) -> HeightMap {
    let mut hm = HeightMap::new(cells, cells, 2.0);
    for z in 0..cells {
        for x in 0..cells {
            let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841);
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1_e995);
            h ^= h >> 15;
            let noise = (h & 0xffff) as f32 / 65_535.0 - 0.5;
            hm.set(x, z, x as f32 * 0.25 + 1.2 * noise);
        }
    }
    hm
}

/// Mean turn, in degrees, where one major road runs on into another of
/// its own kind (a node with exactly two active major edges): how much the
/// streets wander between junctions.
fn mean_major_turn(graph: &RoadGraph) -> f32 {
    let mut through: Vec<Vec<Vec2>> = vec![Vec::new(); graph.nodes.len()];
    for e in graph
        .edges
        .iter()
        .filter(|e| e.active && e.road_type == RoadType::Major)
    {
        let a = graph.nodes[e.start as usize].position;
        let b = graph.nodes[e.end as usize].position;
        if a.distance(b) > 1e-3 {
            through[e.start as usize].push((b - a).normalize());
            through[e.end as usize].push((a - b).normalize());
        }
    }
    let turns: Vec<f32> = through
        .iter()
        .filter(|dirs| dirs.len() == 2)
        .map(|dirs| 180.0 - dirs[0].dot(dirs[1]).clamp(-1.0, 1.0).acos().to_degrees())
        .collect();
    turns.iter().sum::<f32>() / turns.len().max(1) as f32
}

/// #68: on rough ground a smoothed field traces streets that run on
/// steadily where the raw field's streets wander at every bump.
#[test]
fn smoothing_straightens_streets_on_rough_ground() {
    let hm = rough_slope(128);
    let raw = generate_roads(&hm, &base_config(TensorFieldConfig::default())).expect("traces");
    let smooth = generate_roads(
        &hm,
        &base_config(TensorFieldConfig {
            smoothing: 12.0,
            ..TensorFieldConfig::default()
        }),
    )
    .expect("traces");
    let (raw_turn, smooth_turn) = (mean_major_turn(&raw), mean_major_turn(&smooth));
    assert!(
        smooth_turn < 0.5 * raw_turn,
        "smoothed streets must wander less: raw {raw_turn} deg, smoothed {smooth_turn} deg"
    );
}

/// #68: smoothing changes only the directions. Water is judged on the
/// heightmap itself: a smoothed field traces no seed under the real water
/// line even where the blurred copy would stand above it.
#[test]
fn smoothing_leaves_the_water_line_to_the_real_heights() {
    let hm = rough_slope(96);
    let config = TensorConfig {
        water_level: 12.0,
        ..base_config(TensorFieldConfig {
            smoothing: 12.0,
            ..TensorFieldConfig::default()
        })
    };
    let graph = generate_roads(&hm, &config).expect("traces");
    assert!(!graph.nodes.is_empty(), "the dry half still traces");
    for node in &graph.nodes {
        let h = hm.get_height_at(node.position.x, node.position.y);
        assert!(
            h > 12.0 - 1.0,
            "a node stands deep under the water: {h} at {:?}",
            node.position
        );
    }
}

/// #68: a field configuration that cannot be traced is an InvalidConfig
/// error from `generate_roads`, not a panic or a silent default.
#[test]
fn generate_roads_refuses_a_bad_basis_field() {
    let hm = HeightMap::new(32, 32, 2.0);
    let config = base_config(TensorFieldConfig {
        basis: vec![BasisField::Radial {
            center: Vec2::new(10.0, 10.0),
            radius: -5.0,
            strength: 1.0,
        }],
        ..TensorFieldConfig::default()
    });
    let err = generate_roads(&hm, &config).expect_err("refused");
    assert!(format!("{err}").contains("radius"), "{err}");
}

/// #69: a keep-out disc keeps every street out - no node stands inside it
/// beyond a snap radius of its rim - while streets still come right up to
/// it. The control is the same map without the disc, which runs streets
/// through its middle.
#[test]
fn a_keep_out_disc_keeps_the_streets_out() {
    use symbios_tensor::KeepOut;
    let hm = rough_slope(128);
    let disc = KeepOut {
        center: Vec2::new(128.0, 128.0),
        radius: 40.0,
    };
    let config = TensorConfig {
        keep_out: vec![disc],
        ..base_config(TensorFieldConfig::default())
    };
    let graph = generate_roads(&hm, &config).expect("traces");
    let snap = config.snap_radius;
    let inside: Vec<Vec2> = graph
        .nodes
        .iter()
        .map(|n| n.position)
        .filter(|p| p.distance(disc.center) < disc.radius - snap)
        .collect();
    assert!(inside.is_empty(), "streets inside the disc: {inside:?}");
    let at_rim = graph
        .nodes
        .iter()
        .filter(|n| (n.position.distance(disc.center) - disc.radius).abs() < 2.0 * config.step_size)
        .count();
    assert!(at_rim > 4, "streets must still reach the rim: {at_rim}");

    let open = generate_roads(&hm, &base_config(TensorFieldConfig::default())).expect("traces");
    let through = open
        .nodes
        .iter()
        .filter(|n| n.position.distance(disc.center) < disc.radius - snap)
        .count();
    assert!(
        through > 10,
        "the control: without the disc streets cross it ({through})"
    );
}

/// #69: a keep-out disc that cannot be traced is an InvalidConfig error.
#[test]
fn generate_roads_refuses_a_bad_keep_out_disc() {
    use symbios_tensor::KeepOut;
    let hm = HeightMap::new(32, 32, 2.0);
    for disc in [
        KeepOut {
            center: Vec2::new(10.0, 10.0),
            radius: 0.0,
        },
        KeepOut {
            center: Vec2::new(f32::NAN, 10.0),
            radius: 5.0,
        },
        KeepOut {
            center: Vec2::new(10.0, 10.0),
            radius: f32::INFINITY,
        },
    ] {
        let config = TensorConfig {
            keep_out: vec![disc],
            ..base_config(TensorFieldConfig::default())
        };
        let err = generate_roads(&hm, &config).expect_err("refused");
        assert!(format!("{err}").contains("keep_out[0]"), "{err}");
    }
}
