//! Tensor field derived from heightmap surface normals.
//!
//! The field decomposes each surface normal into orthogonal 2D directions:
//! a **major** (contour) axis that follows elevation lines and a **minor**
//! (gradient) axis that points up- or down-slope. Road traces integrate
//! along these axes to produce terrain-adaptive street layouts.

use glam::{Vec2, Vec3};
use symbios_ground::HeightMap;

/// Evaluates a tensor field over a [`HeightMap`], producing orthogonal major/minor
/// direction vectors at any world-space coordinate.
///
/// - **Major** (contour): follows elevation lines — ideal for winding mountain roads.
/// - **Minor** (gradient): points up/down the slope — ideal for steep connecting streets.
pub struct TensorField<'a> {
    heightmap: &'a HeightMap,
}

impl<'a> TensorField<'a> {
    /// Creates a tensor field backed by the given heightmap.
    pub fn new(heightmap: &'a HeightMap) -> Self {
        Self { heightmap }
    }

    /// Samples the tensor field, returning `(major, minor)` unit direction vectors.
    ///
    /// On perfectly flat terrain the field falls back to an axis-aligned Manhattan grid.
    pub fn sample(&self, world_x: f32, world_z: f32) -> (Vec2, Vec2) {
        let n_arr = self.heightmap.get_normal_at(world_x, world_z);
        let normal = Vec3::from_array(n_arr);

        // Minor axis: projection of surface normal onto the XZ plane (gradient direction)
        let minor = Vec2::new(normal.x, normal.z);

        if minor.length_squared() < 1e-5 {
            return (Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        }

        let minor = minor.normalize();
        // Major axis: perpendicular to gradient (contour direction)
        let major = Vec2::new(-minor.y, minor.x);

        (major, minor)
    }
}
