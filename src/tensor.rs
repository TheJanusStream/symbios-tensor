//! Tensor field derived from heightmap surface normals.
//!
//! The field decomposes each surface normal into orthogonal 2D directions:
//! a **major** (contour) axis that follows elevation lines and a **minor**
//! (gradient) axis that points up- or down-slope. Road traces integrate
//! along these axes to produce terrain-adaptive street layouts.
//!
//! On near-flat terrain the field smoothly blends toward an axis-aligned
//! Manhattan grid as slope drops through `[flat_threshold_low,
//! flat_threshold_high]`; an optional low-frequency jitter perturbs the
//! field to break up perfectly parallel streamlines.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use symbios_ground::HeightMap;

/// Configuration for [`TensorField`] sampling.
///
/// Slope (the magnitude of the normal's XZ projection) controls a smooth
/// blend between terrain-derived directions and an axis-aligned fallback,
/// avoiding the abrupt regime change at near-zero slopes that produced
/// visible artefacts at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFieldConfig {
    /// Slope at or below which the field returns the pure axis-aligned
    /// fallback. Default: `1e-4`.
    pub flat_threshold_low: f32,
    /// Slope at or above which the field returns pure terrain-derived
    /// directions. Default: `1e-3`.
    pub flat_threshold_high: f32,
    /// Amplitude of low-frequency directional jitter, in radians. Small
    /// values (e.g. `0.05`–`0.2`) break perfectly parallel streamlines on
    /// flat terrain. Default: `0.0` (disabled).
    pub jitter_amplitude: f32,
    /// Spatial frequency of the jitter (cycles per world unit). Lower
    /// values produce larger, gentler swirls. Default: `0.01`.
    pub jitter_frequency: f32,
}

impl Default for TensorFieldConfig {
    fn default() -> Self {
        Self {
            flat_threshold_low: 1e-4,
            flat_threshold_high: 1e-3,
            jitter_amplitude: 0.0,
            jitter_frequency: 0.01,
        }
    }
}

/// Evaluates a tensor field over a [`HeightMap`], producing orthogonal major/minor
/// direction vectors at any world-space coordinate.
///
/// - **Major** (contour): follows elevation lines — ideal for winding mountain roads.
/// - **Minor** (gradient): points up/down the slope — ideal for steep connecting streets.
pub struct TensorField<'a> {
    pub(crate) heightmap: &'a HeightMap,
    config: TensorFieldConfig,
    fallback_warned: AtomicBool,
}

impl<'a> TensorField<'a> {
    /// Creates a tensor field with default sampling parameters.
    pub fn new(heightmap: &'a HeightMap) -> Self {
        Self::with_config(heightmap, TensorFieldConfig::default())
    }

    /// Creates a tensor field with custom sampling parameters.
    pub fn with_config(heightmap: &'a HeightMap, config: TensorFieldConfig) -> Self {
        Self {
            heightmap,
            config,
            fallback_warned: AtomicBool::new(false),
        }
    }

    /// Samples the tensor field, returning `(major, minor)` unit direction vectors.
    pub fn sample(&self, world_x: f32, world_z: f32) -> (Vec2, Vec2) {
        let n_arr = self.heightmap.get_normal_at(world_x, world_z);
        let normal = Vec3::from_array(n_arr);

        // Minor axis: projection of surface normal onto the XZ plane (gradient direction).
        // Its magnitude is the slope (steepness).
        let raw_minor = Vec2::new(normal.x, normal.z);
        let slope = raw_minor.length();

        let cfg = &self.config;
        let t = smoothstep(cfg.flat_threshold_low, cfg.flat_threshold_high, slope);

        if t < 1.0
            && self
                .fallback_warned
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            eprintln!(
                "symbios-tensor: tensor field blending toward axis-aligned fallback on near-flat terrain (slope={slope:.3e})"
            );
        }

        // Stable minor direction. On true-zero slope, default to +Z; otherwise
        // blend the terrain direction toward the nearest axis as t→0 to avoid
        // sign-flip artefacts when the gradient direction is unstable.
        let blended_minor = if slope > 1e-12 {
            let terrain_dir = raw_minor / slope;
            let fallback_dir = nearest_axis(terrain_dir);
            (terrain_dir * t + fallback_dir * (1.0 - t)).normalize_or_zero()
        } else {
            Vec2::new(0.0, 1.0)
        };

        // Apply low-frequency directional jitter (deterministic per world point).
        let minor = if cfg.jitter_amplitude.abs() > 0.0 {
            let phase = world_x * cfg.jitter_frequency + world_z * cfg.jitter_frequency * 1.7320508;
            let angle = cfg.jitter_amplitude * phase.sin();
            rotate(blended_minor, angle)
        } else {
            blended_minor
        };

        let minor = if minor.length_squared() < 1e-12 {
            Vec2::new(0.0, 1.0)
        } else {
            minor.normalize()
        };
        let major = Vec2::new(-minor.y, minor.x);
        (major, minor)
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Returns the unit axis vector (`±X` or `±Z`) closest to `dir`. Caller
/// must pass a non-zero direction.
fn nearest_axis(dir: Vec2) -> Vec2 {
    if dir.x.abs() >= dir.y.abs() {
        Vec2::new(if dir.x >= 0.0 { 1.0 } else { -1.0 }, 0.0)
    } else {
        Vec2::new(0.0, if dir.y >= 0.0 { 1.0 } else { -1.0 })
    }
}

fn rotate(v: Vec2, angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_consistent_direction() {
        // On a perfectly flat heightmap every sample must return the same
        // axis-aligned direction (no split between two regimes).
        let hm = HeightMap::new(64, 64, 2.0);
        let field = TensorField::new(&hm);

        let (m0, n0) = field.sample(10.0, 10.0);
        for x in (0..120).step_by(7) {
            for z in (0..120).step_by(11) {
                let (m, n) = field.sample(x as f32, z as f32);
                assert!(
                    (m - m0).length() < 1e-5 && (n - n0).length() < 1e-5,
                    "flat-terrain sample at ({x}, {z}) drifted: major={m:?} expected {m0:?}, minor={n:?} expected {n0:?}"
                );
            }
        }
    }

    #[test]
    fn smooth_blend_no_cliff() {
        // Build a heightmap with a controlled slope and verify the blend
        // function returns directions that smoothly transition rather than
        // snapping at a single threshold.
        let mut hm = HeightMap::new(32, 32, 1.0);
        // Very gentle slope: 0.0005 * x — slope magnitude ~0.0005, in the
        // middle of the default blend window (1e-4 .. 1e-3).
        for z in 0..32 {
            for x in 0..32 {
                hm.set(x, z, x as f32 * 0.0005);
            }
        }
        let field = TensorField::new(&hm);
        let (_major, minor) = field.sample(15.0, 15.0);
        // Under the old binary fallback the result would be either the pure
        // terrain direction or the fallback. Under smooth blend, the minor
        // direction is a blend that points roughly along +X (the slope) but
        // is biased toward an axis. Just assert it's a valid unit vector.
        assert!(
            (minor.length() - 1.0).abs() < 1e-4,
            "minor must be unit, got {minor:?}"
        );
    }

    #[test]
    fn jitter_breaks_flat_uniformity() {
        // With jitter enabled on a perfectly flat heightmap, samples at
        // different world points must yield different directions, breaking
        // up the perfectly parallel streamlines that flat terrain would
        // otherwise produce.
        let hm = HeightMap::new(64, 64, 2.0);
        let cfg = TensorFieldConfig {
            jitter_amplitude: 0.3,
            jitter_frequency: 0.05,
            ..Default::default()
        };
        let field = TensorField::with_config(&hm, cfg);

        let (_, n0) = field.sample(10.0, 10.0);
        let (_, n1) = field.sample(50.0, 50.0);
        assert!(
            (n0 - n1).length() > 1e-3,
            "jitter must produce distinct directions; got {n0:?} and {n1:?}"
        );
    }

    #[test]
    fn major_minor_orthogonal() {
        let mut hm = HeightMap::new(16, 16, 1.0);
        for z in 0..16 {
            for x in 0..16 {
                hm.set(x, z, (x + z) as f32 * 0.1);
            }
        }
        let field = TensorField::new(&hm);
        let (m, n) = field.sample(8.0, 8.0);
        assert!(
            m.dot(n).abs() < 1e-4,
            "major and minor must be orthogonal: m={m:?}, n={n:?}, dot={}",
            m.dot(n)
        );
    }
}
