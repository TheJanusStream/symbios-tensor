//! End-to-end example: heightmap → tensor field → tracer → rationalize
//! → blocks → lots → meshes, with hand-rolled writers for a top-down
//! PPM image, road OBJ mesh, and lot SVG footprint diagram.
//!
//! Run with: `cargo run --release --example full_city`
//!
//! Outputs in the current working directory:
//!   - `full_city_graph.ppm`  — top-down RGB visualization of the road
//!     network and lot footprints
//!   - `full_city_roads.obj`  — Wavefront OBJ of the 3D road meshes
//!   - `full_city_lots.svg`   — SVG diagram of building lot footprints
//!
//! No external image/3D crates are pulled in; every writer is plain
//! text and ~30 lines.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use symbios_ground::HeightMap;
use symbios_tensor::{
    BuildingLot, LotConfig, RationalizeConfig, RoadGraph, RoadMeshConfig, RoadMeshes, RoadType,
    TensorConfig, carve_lots, carve_roads, extract_blocks, extract_lots, generate_road_meshes,
    generate_roads, prune_unused_roads, rationalize_graph,
};

fn main() {
    let total = Instant::now();

    // --- 1. Build a heightmap with topographical variety ----------------
    // 96×96 cells at scale=4 → 384×384 world units. Terrain combines a
    // diagonal ridge, a basin, and small noise so the tensor field has
    // both contour-following and gradient-following directions to work
    // with — producing curving major roads and steeper minor connectors.
    let stage = Instant::now();
    let cells = 96;
    let scale = 4.0_f32;
    let mut hm = HeightMap::new(cells, cells, scale);
    for z in 0..cells {
        for x in 0..cells {
            let fx = x as f32;
            let fz = z as f32;
            let ridge = ((fx - fz) * 0.05).sin() * 8.0;
            let basin = -(((fx - 48.0).powi(2) + (fz - 48.0).powi(2)).sqrt() * 0.05).cos() * 4.0;
            let noise = ((fx * 0.27).sin() + (fz * 0.31).cos()) * 0.5;
            hm.set(x, z, ridge + basin + noise);
        }
    }
    println!(
        "[1] Heightmap: {}×{} cells, world {}×{} ({:?})",
        cells,
        cells,
        hm.world_width(),
        hm.world_depth(),
        stage.elapsed()
    );

    // --- 2. Tracer: streamlines through the tensor field ----------------
    let stage = Instant::now();
    let tensor_config = TensorConfig {
        seed: 1337,
        major_road_dist: 36.0,
        minor_road_dist: 14.0,
        snap_radius: 4.0,
        max_trace_steps: 250,
        ..Default::default()
    };
    let mut graph = generate_roads(&hm, &tensor_config).expect("generate_roads");
    println!(
        "[2] Tracer: {} nodes, {} edges ({:?})",
        graph.nodes.len(),
        graph.edges.iter().filter(|e| e.active).count(),
        stage.elapsed()
    );

    // --- 3. Rationalize: RDP, fillet, elevation smoothing ---------------
    let stage = Instant::now();
    rationalize_graph(&mut graph, &hm, &RationalizeConfig::default());
    println!(
        "[3] Rationalize: {} active edges after smoothing ({:?})",
        graph.edges.iter().filter(|e| e.active).count(),
        stage.elapsed()
    );

    // --- 4. Block extraction: minimum-cycle-basis face walk -------------
    let stage = Instant::now();
    extract_blocks(&mut graph);
    println!("[4] Blocks: {} ({:?})", graph.blocks.len(), stage.elapsed());

    // --- 5. Lot subdivision: OBB slicing + frontage + setbacks ----------
    let stage = Instant::now();
    let mut hm = hm;
    let lot_config = LotConfig::default();
    let lots = extract_lots(&graph, &mut hm, &lot_config);
    println!("[5] Lots: {} ({:?})", lots.len(), stage.elapsed());

    // --- 6. Optional pruning: drop roads that don't serve any lot -------
    let stage = Instant::now();
    prune_unused_roads(&mut graph, &lots);
    println!(
        "[6] Prune: {} active edges after pruning ({:?})",
        graph.edges.iter().filter(|e| e.active).count(),
        stage.elapsed()
    );

    // --- 7. Carve roads + lots into the heightmap -----------------------
    let stage = Instant::now();
    let mesh_config = RoadMeshConfig::default();
    let road_mask = carve_roads(&graph, &mut hm, &mesh_config, 4.0);
    carve_lots(&lots, &mut hm, 2.0, Some(&road_mask));
    println!(
        "[7] Carve: heightmap flattened under network ({:?})",
        stage.elapsed()
    );

    // --- 8. 3D road meshes ---------------------------------------------
    let stage = Instant::now();
    let meshes = generate_road_meshes(&graph, &hm, &mesh_config);
    let total_verts =
        meshes.hubs.vertices.len() + meshes.ribbons.vertices.len() + meshes.skirts.vertices.len();
    let total_tris =
        (meshes.hubs.indices.len() + meshes.ribbons.indices.len() + meshes.skirts.indices.len())
            / 3;
    println!(
        "[8] Meshes: {} verts / {} triangles ({:?})",
        total_verts,
        total_tris,
        stage.elapsed()
    );

    // --- 9. Write outputs ----------------------------------------------
    let stage = Instant::now();
    write_graph_ppm(
        &graph,
        &lots,
        hm.world_width(),
        hm.world_depth(),
        "full_city_graph.ppm",
    )
    .expect("write ppm");
    write_meshes_obj(&meshes, "full_city_roads.obj").expect("write obj");
    write_lots_svg(
        &lots,
        hm.world_width(),
        hm.world_depth(),
        "full_city_lots.svg",
    )
    .expect("write svg");
    println!("[9] Outputs written ({:?})", stage.elapsed());

    println!("--- Done in {:?} ---", total.elapsed());
}

/// Renders a top-down P6 PPM image: roads in white, major roads slightly
/// brighter, lot footprints filled in pale blue.
fn write_graph_ppm(
    graph: &RoadGraph,
    lots: &[BuildingLot],
    world_w: f32,
    world_d: f32,
    path: &str,
) -> std::io::Result<()> {
    let res = 512_usize;
    let scale_x = res as f32 / world_w;
    let scale_z = res as f32 / world_d;
    let mut buf = vec![[20u8, 24, 32]; res * res]; // dark background

    // Lots: filled rectangles.
    for lot in lots {
        let cos = lot.rotation.cos();
        let sin = lot.rotation.sin();
        let hw = lot.width * 0.5;
        let hd = lot.depth * 0.5;
        let corners = [
            local_corner(hw, hd),
            local_corner(hw, -hd),
            local_corner(-hw, -hd),
            local_corner(-hw, hd),
        ];
        let world_corners: Vec<(f32, f32)> = corners
            .iter()
            .map(|&(x, y)| {
                (
                    lot.position.x + x * cos - y * sin,
                    lot.position.y + x * sin + y * cos,
                )
            })
            .collect();
        let color = if lot.is_shoreline {
            [180u8, 200, 240]
        } else {
            [120u8, 140, 180]
        };
        fill_polygon(&mut buf, res, &world_corners, scale_x, scale_z, color);
    }

    // Edges: white lines, brighter for major.
    for edge in &graph.edges {
        if !edge.active {
            continue;
        }
        let a = graph.nodes[edge.start as usize].position;
        let b = graph.nodes[edge.end as usize].position;
        let color = match edge.road_type {
            RoadType::Major => [255u8, 240, 200],
            RoadType::Minor => [180u8, 180, 180],
        };
        draw_line(
            &mut buf,
            res,
            a.x * scale_x,
            a.y * scale_z,
            b.x * scale_x,
            b.y * scale_z,
            color,
        );
    }

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    write!(w, "P6\n{res} {res}\n255\n")?;
    for px in &buf {
        w.write_all(px)?;
    }
    Ok(())
}

fn local_corner(x: f32, y: f32) -> (f32, f32) {
    (x, y)
}

fn draw_line(buf: &mut [[u8; 3]], res: usize, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 3]) {
    // Bresenham over the canvas grid.
    let mut x0 = x0.round() as i32;
    let mut y0 = y0.round() as i32;
    let x1 = x1.round() as i32;
    let y1 = y1.round() as i32;
    let dx = (x1 - x0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && (x0 as usize) < res && y0 >= 0 && (y0 as usize) < res {
            buf[(y0 as usize) * res + (x0 as usize)] = color;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn fill_polygon(
    buf: &mut [[u8; 3]],
    res: usize,
    corners_world: &[(f32, f32)],
    scale_x: f32,
    scale_z: f32,
    color: [u8; 3],
) {
    if corners_world.is_empty() {
        return;
    }
    let (min_x, min_y, max_x, max_y) = corners_world.iter().fold(
        (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
        |(mn_x, mn_y, mx_x, mx_y), &(x, y)| (mn_x.min(x), mn_y.min(y), mx_x.max(x), mx_y.max(y)),
    );
    let px_min_x = ((min_x * scale_x).floor() as i32).max(0);
    let px_max_x = ((max_x * scale_x).ceil() as i32).min(res as i32 - 1);
    let px_min_y = ((min_y * scale_z).floor() as i32).max(0);
    let px_max_y = ((max_y * scale_z).ceil() as i32).min(res as i32 - 1);

    for py in px_min_y..=px_max_y {
        for px in px_min_x..=px_max_x {
            let wx = px as f32 / scale_x;
            let wy = py as f32 / scale_z;
            if point_in_polygon(wx, wy, corners_world) {
                buf[(py as usize) * res + (px as usize)] = color;
            }
        }
    }
}

fn point_in_polygon(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) {
            let t = (y - yi) / (yj - yi);
            let x_cross = xi + t * (xj - xi);
            if x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Hand-rolled OBJ writer: dumps every mesh's vertices and triangle
/// indices grouped by hubs/ribbons/skirts. No materials, no normals —
/// just enough for a viewer to load the geometry.
fn write_meshes_obj(meshes: &RoadMeshes, path: &str) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "# symbios-tensor full_city example")?;

    let mut vertex_offset = 1_usize; // OBJ is 1-indexed
    for (name, mesh) in [
        ("hubs", &meshes.hubs),
        ("ribbons", &meshes.ribbons),
        ("skirts", &meshes.skirts),
    ] {
        writeln!(w, "g {name}")?;
        for v in &mesh.vertices {
            writeln!(w, "v {} {} {}", v[0], v[1], v[2])?;
        }
        for tri in mesh.indices.chunks(3) {
            if tri.len() == 3 {
                writeln!(
                    w,
                    "f {} {} {}",
                    tri[0] as usize + vertex_offset,
                    tri[1] as usize + vertex_offset,
                    tri[2] as usize + vertex_offset
                )?;
            }
        }
        vertex_offset += mesh.vertices.len();
    }
    Ok(())
}

/// SVG with one `<rect>` per lot (rotated about its center). Top-down
/// view with the heightmap origin at the SVG origin.
fn write_lots_svg(
    lots: &[BuildingLot],
    world_w: f32,
    world_d: f32,
    path: &str,
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {world_w} {world_d}" width="800" height="800">
  <rect width="100%" height="100%" fill="#1c1f26"/>"##
    )?;
    for lot in lots {
        let deg = lot.rotation.to_degrees();
        let fill = if lot.is_shoreline {
            "#bbd5ff"
        } else {
            "#7a8ab5"
        };
        writeln!(
            w,
            r##"  <rect x="{x}" y="{y}" width="{ww}" height="{hh}" fill="{fill}" stroke="#222" stroke-width="0.4" transform="rotate({deg} {cx} {cy})"/>"##,
            x = lot.position.x - lot.width * 0.5,
            y = lot.position.y - lot.depth * 0.5,
            ww = lot.width,
            hh = lot.depth,
            cx = lot.position.x,
            cy = lot.position.y,
            deg = deg,
            fill = fill,
        )?;
    }
    writeln!(w, "</svg>")?;
    Ok(())
}
