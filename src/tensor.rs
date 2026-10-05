//! Tensor field derived from heightmap surface normals, optionally shaped
//! by designer basis fields.
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
//!
//! Two opt-in controls shape the field further (#68):
//!
//! * **Smoothing** ([`TensorFieldConfig::smoothing`]) derives the directions
//!   from a blurred copy of the heightmap, so streets follow the land's
//!   broad form instead of turning at every bump. Heights, water and
//!   everything else the tracer reads still come from the heightmap itself.
//! * **Basis fields** ([`BasisField`], after Chen et al. 2008, "Interactive
//!   Procedural Street Modeling") are laid over the terrain's field and
//!   summed with it *as tensors*: each direction `θ` is the symmetric
//!   traceless tensor `(cos 2θ, sin 2θ)`, which makes a direction and its
//!   reverse the same thing, and the summed tensor's major eigenvector is
//!   the road direction. A [`BasisField::Radial`] rings major roads round a
//!   centre and runs minor roads out from it - the field a lone hill would
//!   give, without the hill - and a [`BasisField::Grid`] lays a straight grid
//!   at any angle. Each reaches a finite `radius` and fades smoothly to
//!   nothing at its edge; outside every one of them the terrain's own field
//!   is returned bit for bit.
//!
//! With both left at their defaults the field is the one earlier releases
//! traced, to the bit: a saved layout re-traced under a new release keeps
//! its streets.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use symbios_ground::HeightMap;

use crate::math::MathMode;

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
    /// Scale, in world units, below which the terrain's relief does not
    /// steer the field: the directions come from a copy of the heightmap
    /// blurred over about this radius (three box passes, close to a
    /// Gaussian), so a street follows a hillside rather than every hummock
    /// on it. `0.0` (the default) reads the heightmap itself. Must be
    /// finite and non-negative.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub smoothing: f32,
    /// Weight of the terrain-derived field against the [`Self::basis`]
    /// fields where they reach. `1.0` (the default) is a full partner;
    /// `0.0` lets the basis fields alone decide inside their reach (the
    /// terrain still decides outside it). Must be finite and non-negative.
    #[serde(
        default = "default_terrain_weight",
        skip_serializing_if = "is_default_terrain_weight"
    )]
    pub terrain_weight: f32,
    /// Designer fields summed with the terrain's, as tensors. Empty (the
    /// default) is the terrain's field alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub basis: Vec<BasisField>,
}

fn default_terrain_weight() -> f32 {
    1.0
}

// The new fields stay off the wire at their defaults, so a configuration
// serialised by this release reads exactly as one serialised before it.
fn is_zero(v: &f32) -> bool {
    *v == 0.0
}

fn is_default_terrain_weight(v: &f32) -> bool {
    *v == default_terrain_weight()
}

impl Default for TensorFieldConfig {
    fn default() -> Self {
        Self {
            flat_threshold_low: 1e-4,
            flat_threshold_high: 1e-3,
            jitter_amplitude: 0.0,
            jitter_frequency: 0.01,
            smoothing: 0.0,
            terrain_weight: default_terrain_weight(),
            basis: Vec::new(),
        }
    }
}

impl TensorFieldConfig {
    /// Why this configuration cannot be traced, if it cannot: a non-finite
    /// or negative smoothing or terrain weight, or a basis field with a
    /// non-finite centre, angle or strength, a negative strength, or a
    /// radius that is not finite and positive.
    pub fn validate(&self) -> Result<(), String> {
        if !self.smoothing.is_finite() || self.smoothing < 0.0 {
            return Err(format!(
                "field.smoothing must be finite and non-negative, got {}",
                self.smoothing
            ));
        }
        if !self.terrain_weight.is_finite() || self.terrain_weight < 0.0 {
            return Err(format!(
                "field.terrain_weight must be finite and non-negative, got {}",
                self.terrain_weight
            ));
        }
        for (i, basis) in self.basis.iter().enumerate() {
            basis
                .validate()
                .map_err(|why| format!("field.basis[{i}]: {why}"))?;
        }
        Ok(())
    }
}

/// A designer field laid over the terrain's ([`TensorFieldConfig::basis`]).
///
/// Coordinates are the heightmap's world frame, the frame the tracer works
/// in: `(0, 0)` is the heightmap's first cell. Each field reaches `radius`
/// world units from its `center`, its weight falling smoothly from
/// `strength` there to nothing at the edge (`strength · (1 - (d/r)²)²`, a
/// compact bump with no `exp`, so the weight is the same on every
/// platform).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BasisField {
    /// Major roads ring `center` and minor roads run straight out from it:
    /// the field a lone round hill there would give, without the hill.
    Radial {
        center: Vec2,
        radius: f32,
        strength: f32,
    },
    /// A straight grid: major roads along `angle` radians (the direction
    /// `(cos angle, sin angle)` in the XZ plane, from +X toward +Z), minor
    /// roads square to them.
    Grid {
        center: Vec2,
        angle: f32,
        radius: f32,
        strength: f32,
    },
}

impl BasisField {
    fn center(&self) -> Vec2 {
        match self {
            Self::Radial { center, .. } | Self::Grid { center, .. } => *center,
        }
    }

    fn reach(&self) -> (f32, f32) {
        match self {
            Self::Radial {
                radius, strength, ..
            }
            | Self::Grid {
                radius, strength, ..
            } => (*radius, *strength),
        }
    }

    fn validate(&self) -> Result<(), String> {
        let (radius, strength) = self.reach();
        if !self.center().is_finite() {
            return Err(format!("center must be finite, got {:?}", self.center()));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(format!("radius must be finite and positive, got {radius}"));
        }
        if !strength.is_finite() || strength < 0.0 {
            return Err(format!(
                "strength must be finite and non-negative, got {strength}"
            ));
        }
        if let Self::Grid { angle, .. } = self
            && !angle.is_finite()
        {
            return Err(format!("angle must be finite, got {angle}"));
        }
        Ok(())
    }

    /// This field's weight at `p`: `strength` at the centre, falling
    /// smoothly to `0.0` at `radius` and staying there beyond it.
    pub fn weight_at(&self, p: Vec2) -> f32 {
        let (radius, strength) = self.reach();
        let s = (p - self.center()).length_squared() / (radius * radius);
        if s >= 1.0 {
            return 0.0;
        }
        let falloff = 1.0 - s;
        strength * falloff * falloff
    }

    /// This field's own direction at `p` as the unit tensor
    /// `(cos 2θ, sin 2θ)` of its major road direction `θ`, or zero where it
    /// has none (a radial field's centre).
    pub fn tensor_at(&self, p: Vec2) -> Vec2 {
        match self {
            Self::Radial { center, .. } => {
                // The tangent to the ring through p is (-d.y, d.x) / |d|, and
                // its tensor (t.x² - t.y², 2 t.x t.y) is rational in d.
                let d = p - *center;
                let len2 = d.length_squared();
                if len2 < 1e-12 {
                    return Vec2::ZERO;
                }
                Vec2::new(d.y * d.y - d.x * d.x, -2.0 * d.x * d.y) / len2
            }
            // The `libm` port rather than the platform's: the same bits on
            // every peer that traces this layout.
            Self::Grid { angle, .. } => Vec2::new(libm::cosf(2.0 * angle), libm::sinf(2.0 * angle)),
        }
    }
}

/// The unit tensor `(cos 2θ, sin 2θ)` of the unit direction `(cos θ, sin θ)`.
fn direction_tensor(v: Vec2) -> Vec2 {
    Vec2::new(v.x * v.x - v.y * v.y, 2.0 * v.x * v.y)
}

/// The major eigenvector of the tensor `t = r (cos 2θ, sin 2θ)`: the unit
/// direction `(cos θ, sin θ)`, by the half-angle identities rather than an
/// `atan2` (`(1 + cos 2θ, sin 2θ)` and `(sin 2θ, 1 - cos 2θ)` are both that
/// direction scaled, by `2 cos θ` and `2 sin θ`; the longer is the
/// well-conditioned one). `None` for a tensor too small to have a
/// direction - opposing fields that cancel.
fn major_eigenvector(t: Vec2) -> Option<Vec2> {
    let r = t.length();
    if r.is_nan() || r <= 1e-6 {
        return None;
    }
    let (c, s) = (t.x / r, t.y / r);
    let a = Vec2::new(1.0 + c, s);
    let b = Vec2::new(s, 1.0 - c);
    let v = if a.length_squared() >= b.length_squared() {
        a
    } else {
        b
    };
    Some(v.normalize())
}

/// A copy of `hm` blurred over about `radius` world units: three passes of a
/// separable box filter, each window cut off at the map's edge so it
/// averages only the cells it covers (a clamped edge would weight its edge
/// cells many times over). A window as wide as the map is the map's mean,
/// so the half-width stops there: any finite smoothing costs the same, and
/// nothing can overflow. Prefix sums in `f64` keep each pass linear in the
/// map's size whatever the radius. Only the field reads it.
fn smoothed_copy(hm: &HeightMap, radius: f32) -> HeightMap {
    let (w, h) = (hm.width(), hm.height());
    let r = ((radius / hm.scale()).round().min(w.max(h) as f32) as usize).max(1);
    let mut out = HeightMap::new(w, h, hm.scale());
    let mut cur: Vec<f32> = hm.data().to_vec();
    let mut tmp = vec![0.0_f32; cur.len()];
    let mut prefix = vec![0.0_f64; w.max(h) + 1];
    let mut box_pass = |src: &[f32], dst: &mut [f32], along_x: bool| {
        let (n, lines) = if along_x { (w, h) } else { (h, w) };
        let at = |line: usize, i: usize| if along_x { line * w + i } else { i * w + line };
        for line in 0..lines {
            for i in 0..n {
                prefix[i + 1] = prefix[i] + f64::from(src[at(line, i)]);
            }
            for i in 0..n {
                let lo = i.saturating_sub(r);
                let hi = (i + r).min(n - 1);
                dst[at(line, i)] = ((prefix[hi + 1] - prefix[lo]) / (hi - lo + 1) as f64) as f32;
            }
        }
    };
    for _ in 0..3 {
        box_pass(&cur, &mut tmp, true);
        box_pass(&tmp, &mut cur, false);
    }
    out.data_mut().copy_from_slice(&cur);
    out
}

/// Evaluates a tensor field over a [`HeightMap`], producing orthogonal major/minor
/// direction vectors at any world-space coordinate.
///
/// - **Major** (contour): follows elevation lines — ideal for winding mountain roads.
/// - **Minor** (gradient): points up/down the slope — ideal for steep connecting streets.
pub struct TensorField<'a> {
    pub(crate) heightmap: &'a HeightMap,
    /// The heightmap the directions are read from when
    /// [`TensorFieldConfig::smoothing`] is set: a blurred copy of
    /// `heightmap`, which stays the one the tracer reads heights from.
    smoothed: Option<HeightMap>,
    config: TensorFieldConfig,
    /// Whose `sin` and `cos` the jitter turns directions with.
    math: MathMode,
    fallback_warned: AtomicBool,
}

impl<'a> TensorField<'a> {
    /// Creates a tensor field with default sampling parameters.
    pub fn new(heightmap: &'a HeightMap) -> Self {
        Self::with_config(heightmap, TensorFieldConfig::default())
    }

    /// Creates a tensor field with custom sampling parameters.
    pub fn with_config(heightmap: &'a HeightMap, config: TensorFieldConfig) -> Self {
        let smoothed = (config.smoothing > 0.0 && config.smoothing.is_finite())
            .then(|| smoothed_copy(heightmap, config.smoothing));
        Self {
            heightmap,
            smoothed,
            config,
            math: MathMode::Platform,
            fallback_warned: AtomicBool::new(false),
        }
    }

    /// The same field with its jitter turned by `math`'s functions (see
    /// [`crate::math`]); the default is [`MathMode::Platform`].
    pub fn with_math(mut self, math: MathMode) -> Self {
        self.math = math;
        self
    }

    /// Samples the tensor field, returning `(major, minor)` unit direction vectors.
    pub fn sample(&self, world_x: f32, world_z: f32) -> (Vec2, Vec2) {
        let terrain = self.sample_terrain(world_x, world_z);
        if self.config.basis.is_empty() {
            return terrain;
        }
        let p = Vec2::new(world_x, world_z);
        let mut sum = Vec2::ZERO;
        let mut reached = false;
        for basis in &self.config.basis {
            let weight = basis.weight_at(p);
            if weight > 0.0 {
                sum += basis.tensor_at(p) * weight;
                reached = true;
            }
        }
        // Outside every basis field's reach the terrain decides alone, bit
        // for bit.
        if !reached {
            return terrain;
        }
        sum += direction_tensor(terrain.0) * self.config.terrain_weight;
        let Some(mut major) = major_eigenvector(sum) else {
            // Fields that cancel (two grids square to each other at equal
            // weight) have no direction: keep the terrain's.
            return terrain;
        };
        // A tensor has no sign; keep the terrain's sense so a sample does not
        // flip for no reason (the tracer aligns signs anyway).
        if major.dot(terrain.0) < 0.0 {
            major = -major;
        }
        (major, Vec2::new(major.y, -major.x))
    }

    /// The field the terrain alone gives at a point: its contour and slope
    /// directions, blended toward the axes on the flat and jittered when
    /// asked to - read from the smoothed copy when there is one.
    fn sample_terrain(&self, world_x: f32, world_z: f32) -> (Vec2, Vec2) {
        let source = self.smoothed.as_ref().unwrap_or(self.heightmap);
        let n_arr = source.get_normal_at(world_x, world_z);
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
            let angle = cfg.jitter_amplitude * self.math.sin(phase);
            rotate(blended_minor, angle, self.math)
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

fn rotate(v: Vec2, angle: f32, math: MathMode) -> Vec2 {
    let (s, c) = math.sin_cos(angle);
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

    /// A flat map 128 m across, and a field over it with `basis` alone
    /// deciding where it reaches (terrain weight 0).
    fn designer_field(hm: &HeightMap, basis: Vec<BasisField>) -> TensorField<'_> {
        TensorField::with_config(
            hm,
            TensorFieldConfig {
                terrain_weight: 0.0,
                basis,
                ..Default::default()
            },
        )
    }

    /// A gentle slope rising along +X, so the terrain's own major direction
    /// is ±Z everywhere.
    fn slope_along_x() -> HeightMap {
        let mut hm = HeightMap::new(64, 64, 2.0);
        for z in 0..64 {
            for x in 0..64 {
                hm.set(x, z, x as f32 * 0.2);
            }
        }
        hm
    }

    /// #68: a radial field rings its major roads round its centre and runs
    /// its minor roads straight out from it, at every bearing and distance
    /// inside its reach.
    #[test]
    fn a_radial_field_rings_major_roads_round_its_centre() {
        let hm = HeightMap::new(64, 64, 2.0);
        let centre = Vec2::new(64.0, 64.0);
        let field = designer_field(
            &hm,
            vec![BasisField::Radial {
                center: centre,
                radius: 60.0,
                strength: 1.0,
            }],
        );
        for step in 0..12 {
            let bearing = step as f32 * std::f32::consts::FRAC_PI_6;
            for d in [8.0_f32, 25.0, 50.0] {
                let p = centre + Vec2::new(bearing.cos(), bearing.sin()) * d;
                let (major, minor) = field.sample(p.x, p.y);
                let out = (p - centre).normalize();
                assert!(
                    major.dot(out).abs() < 1e-4,
                    "major must ring the centre at {p:?}: {major:?}"
                );
                assert!(
                    minor.dot(out).abs() > 1.0 - 1e-4,
                    "minor must run out from the centre at {p:?}: {minor:?}"
                );
            }
        }
    }

    /// #68: a grid field lays its major roads along its angle, wherever it
    /// reaches.
    #[test]
    fn a_grid_field_lays_its_roads_at_its_angle() {
        let hm = HeightMap::new(64, 64, 2.0);
        let angle = 30.0_f32.to_radians();
        let field = designer_field(
            &hm,
            vec![BasisField::Grid {
                center: Vec2::new(64.0, 64.0),
                angle,
                radius: 80.0,
                strength: 1.0,
            }],
        );
        let along = Vec2::new(angle.cos(), angle.sin());
        for (x, z) in [(64.0, 64.0), (30.0, 90.0), (100.0, 40.0)] {
            let (major, minor) = field.sample(x, z);
            assert!(
                major.dot(along).abs() > 1.0 - 1e-5,
                "major must lie along the grid at ({x}, {z}): {major:?}"
            );
            assert!(major.dot(minor).abs() < 1e-5, "minor must be square to it");
        }
    }

    /// #68: outside every basis field's reach the field is the terrain's,
    /// to the bit - the property that keeps a layout's far streets where
    /// they were when a field is added near its centre.
    #[test]
    fn beyond_its_reach_a_basis_field_changes_nothing() {
        let hm = slope_along_x();
        let plain = TensorField::new(&hm);
        let shaped = TensorField::with_config(
            &hm,
            TensorFieldConfig {
                basis: vec![BasisField::Radial {
                    center: Vec2::new(20.0, 20.0),
                    radius: 15.0,
                    strength: 5.0,
                }],
                ..Default::default()
            },
        );
        let bits =
            |(a, b): (Vec2, Vec2)| [a.x.to_bits(), a.y.to_bits(), b.x.to_bits(), b.y.to_bits()];
        for (x, z) in [(100.0, 100.0), (60.0, 20.0), (20.0, 36.0)] {
            assert_eq!(bits(plain.sample(x, z)), bits(shaped.sample(x, z)));
        }
        // The control: inside its reach it does turn the field. North of
        // the centre the ring runs along X, across the slope's contour (Z);
        // east of it the two would agree and prove nothing.
        assert_ne!(
            bits(plain.sample(20.0, 28.0)),
            bits(shaped.sample(20.0, 28.0))
        );
    }

    /// #68: two fields that cancel (grids square to each other at equal
    /// weight) leave no direction to follow; the terrain's is kept and
    /// nothing turns to NaN.
    #[test]
    fn fields_that_cancel_fall_back_to_the_terrain() {
        let hm = slope_along_x();
        let centre = Vec2::new(64.0, 64.0);
        let grid = |angle: f32| BasisField::Grid {
            center: centre,
            angle,
            radius: 50.0,
            strength: 1.0,
        };
        let field = designer_field(&hm, vec![grid(0.0), grid(std::f32::consts::FRAC_PI_2)]);
        let (major, minor) = field.sample(64.0, 64.0);
        let (t_major, t_minor) = TensorField::new(&hm).sample(64.0, 64.0);
        assert!(major.is_finite() && minor.is_finite());
        assert_eq!((major, minor), (t_major, t_minor));
    }

    /// #68: the terrain weight trades the terrain's direction against a
    /// basis field's: at 0 the grid decides, at 1 the two meet between, and
    /// a heavy terrain all but overrules the grid.
    #[test]
    fn the_terrain_weight_trades_against_a_basis_field() {
        let hm = slope_along_x();
        let angle = 45.0_f32.to_radians();
        let at_weight = |terrain_weight: f32| {
            let field = TensorField::with_config(
                &hm,
                TensorFieldConfig {
                    terrain_weight,
                    basis: vec![BasisField::Grid {
                        center: Vec2::new(64.0, 64.0),
                        angle,
                        radius: 200.0,
                        strength: 1.0,
                    }],
                    ..Default::default()
                },
            );
            // The angle of the major road from +X, folded into [0, 180).
            let (major, _) = field.sample(64.0, 64.0);
            major.y.atan2(major.x).to_degrees().rem_euclid(180.0)
        };
        let grid_only = at_weight(0.0);
        let terrain_heavy = at_weight(100.0);
        let even = at_weight(1.0);
        assert!((grid_only - 45.0).abs() < 0.01, "grid alone: {grid_only}");
        assert!(
            (terrain_heavy - 90.0).abs() < 1.0,
            "terrain heavy: {terrain_heavy}"
        );
        assert!(
            even > 46.0 && even < 89.0,
            "an even mix lies between: {even}"
        );
    }

    /// A slope along +X roughened by a fine deterministic noise.
    fn rough_slope() -> HeightMap {
        let mut hm = HeightMap::new(96, 96, 2.0);
        for z in 0..96 {
            for x in 0..96 {
                let mut h =
                    (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841);
                h ^= h >> 13;
                h = h.wrapping_mul(0x5bd1_e995);
                h ^= h >> 15;
                let noise = (h & 0xffff) as f32 / 65_535.0 - 0.5;
                hm.set(x, z, x as f32 * 0.25 + 1.2 * noise);
            }
        }
        hm
    }

    /// Mean turn, in radians, of the field's major direction between
    /// samples 2 m apart along a line across the map.
    fn mean_turn(field: &TensorField<'_>) -> f32 {
        let samples: Vec<Vec2> = (0..80)
            .map(|i| field.sample(20.0 + i as f32 * 2.0, 96.0).0)
            .collect();
        let turns: Vec<f32> = samples
            .windows(2)
            .map(|w| w[0].dot(w[1]).abs().min(1.0).acos())
            .collect();
        turns.iter().sum::<f32>() / turns.len() as f32
    }

    /// #68: on rough ground the smoothed field barely turns where the raw
    /// one swings at every bump, and it still runs along the hillside.
    #[test]
    fn smoothing_steadies_the_field_on_rough_ground() {
        let hm = rough_slope();
        let raw = TensorField::new(&hm);
        let smooth = TensorField::with_config(
            &hm,
            TensorFieldConfig {
                smoothing: 12.0,
                ..Default::default()
            },
        );
        let (raw_turn, smooth_turn) = (mean_turn(&raw), mean_turn(&smooth));
        assert!(
            smooth_turn < 0.25 * raw_turn,
            "smoothing must steady the field: raw {raw_turn}, smoothed {smooth_turn}"
        );
        // Still the hillside's contour: the slope rises along +X, so the
        // major road runs along Z.
        let (major, _) = smooth.sample(96.0, 96.0);
        assert!(
            major.y.abs() > 0.99,
            "the smoothed major must follow the contour: {major:?}"
        );
        // And the heights the tracer reads are the heightmap's own.
        assert!(std::ptr::eq(smooth.heightmap, &hm));
    }

    /// #68: a configuration that cannot be traced is refused with the field
    /// that is wrong named.
    #[test]
    fn a_bad_field_configuration_is_refused_by_name() {
        let radial = |center: Vec2, radius: f32, strength: f32| TensorFieldConfig {
            basis: vec![BasisField::Radial {
                center,
                radius,
                strength,
            }],
            ..Default::default()
        };
        let why = |c: TensorFieldConfig| c.validate().expect_err("refused");
        assert!(why(radial(Vec2::ZERO, 0.0, 1.0)).contains("basis[0]: radius"));
        assert!(why(radial(Vec2::new(f32::NAN, 0.0), 5.0, 1.0)).contains("center"));
        assert!(why(radial(Vec2::ZERO, 5.0, -1.0)).contains("strength"));
        let grid = TensorFieldConfig {
            basis: vec![BasisField::Grid {
                center: Vec2::ZERO,
                angle: f32::INFINITY,
                radius: 5.0,
                strength: 1.0,
            }],
            ..Default::default()
        };
        assert!(why(grid).contains("angle"));
        let smoothing = TensorFieldConfig {
            smoothing: -1.0,
            ..Default::default()
        };
        assert!(why(smoothing).contains("smoothing"));
        let weight = TensorFieldConfig {
            terrain_weight: f32::NAN,
            ..Default::default()
        };
        assert!(why(weight).contains("terrain_weight"));
        assert!(TensorFieldConfig::default().validate().is_ok());
    }

    /// #68: a configuration written before the new fields reads with them
    /// at their defaults, and a basis field round-trips by its `kind`.
    #[test]
    fn an_older_field_configuration_reads_and_basis_fields_round_trip() {
        let old: TensorFieldConfig = serde_json::from_value(serde_json::json!({
            "flat_threshold_low": 0.0001,
            "flat_threshold_high": 0.001,
            "jitter_amplitude": 0.0,
            "jitter_frequency": 0.01
        }))
        .expect("an older configuration reads");
        assert_eq!(old.smoothing, 0.0);
        assert_eq!(old.terrain_weight, 1.0);
        assert!(old.basis.is_empty());

        let basis = BasisField::Grid {
            center: Vec2::new(3.0, 4.0),
            angle: 0.5,
            radius: 20.0,
            strength: 2.0,
        };
        let json = serde_json::to_value(&basis).expect("writes");
        assert_eq!(json["kind"], "grid");
        let back: BasisField = serde_json::from_value(json).expect("reads back");
        assert_eq!(back, basis);
    }

    /// #68, the end review: any finite smoothing is safe - the kernel stops
    /// at the map's size, where a wider box would give the same mean - so a
    /// hostile or mistaken value can neither overflow nor run for hours.
    #[test]
    fn a_huge_smoothing_is_the_map_mean_not_a_hang() {
        let hm = rough_slope();
        let start = std::time::Instant::now();
        let field = TensorField::with_config(
            &hm,
            TensorFieldConfig {
                smoothing: f32::MAX,
                ..Default::default()
            },
        );
        let (major, minor) = field.sample(96.0, 96.0);
        assert!(major.is_finite() && minor.is_finite());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "a huge smoothing took {:?}",
            start.elapsed()
        );
        // And it IS the map's mean: in a release build an uncapped kernel
        // wraps its width silently and flattens the map to nothing instead.
        let mean = hm.data().iter().sum::<f32>() / hm.data().len() as f32;
        let flat = smoothed_copy(&hm, f32::MAX);
        assert!(
            flat.data().iter().all(|h| (h - mean).abs() < 1.0),
            "the widest smoothing is not the map's mean {mean}"
        );
    }

    /// #68: the new fields stay off the wire at their defaults, so a
    /// configuration written by this release reads as one written before.
    #[test]
    fn the_new_field_settings_stay_off_the_wire_at_their_defaults() {
        let v = serde_json::to_value(TensorFieldConfig::default()).expect("writes");
        let keys: Vec<&str> = v
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys.len(),
            4,
            "only the four fields that predate #68: {keys:?}"
        );
        let v = serde_json::to_value(crate::TensorConfig::default()).expect("writes");
        assert!(v.get("keep_out").is_none(), "{v}");
    }
}
