//! Where the transcendental functions a layout depends on come from.
//!
//! `f32::sin` and its kin call whatever libm the target links - glibc's on
//! x86-64 Linux, a port of musl's on `wasm32-unknown-unknown`, the system's
//! on macOS - and those differ in the last bit. A trace, its fillets, the
//! order of a block's edges and a lot's frame all turn such bits into
//! decisions: a lot kept on one peer is dropped on another, and a building
//! stands somewhere else. [`MathMode::Portable`] routes every one of those
//! calls through the pure-Rust [`libm`] crate, which gives the same bits on
//! every target, so peers on different platforms derive the same layout.
//! [`MathMode::Platform`], the default, keeps the target's own functions and
//! the bits this crate has always produced.
//!
//! The mode covers the layout - the trace, rationalization, blocks and
//! lots. Terrain carving ([`crate::carve_roads`], [`crate::carve_lots`]) and
//! the 3D meshes ([`crate::generate_road_meshes`]) keep the platform's
//! functions: they shape what is drawn, not where anything stands.

use serde::{Deserialize, Serialize};

/// Which implementation of `sin`, `cos`, `tan`, `acos` and `atan2` a layout
/// is derived with - see the [module documentation](self). Its methods are
/// those functions, by the mode's implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MathMode {
    /// The target's own libm: what every release before 0.6 used, bit for
    /// bit, and different in the last bit from one platform to the next.
    #[default]
    Platform,
    /// The [`libm`] crate's: the same bits on every target.
    Portable,
}

impl MathMode {
    /// Whether this is [`MathMode::Platform`] - the serde skip predicate
    /// that keeps the default off the wire.
    pub fn is_platform(&self) -> bool {
        *self == Self::Platform
    }

    /// `x.sin()`, by this mode's implementation. Public so a consumer
    /// deriving more from a layout - its own geometry over the lots, say -
    /// can do so with the layout's functions.
    pub fn sin(self, x: f32) -> f32 {
        match self {
            Self::Platform => x.sin(),
            Self::Portable => libm::sinf(x),
        }
    }

    /// `x.cos()`, by this mode's implementation.
    pub fn cos(self, x: f32) -> f32 {
        match self {
            Self::Platform => x.cos(),
            Self::Portable => libm::cosf(x),
        }
    }

    /// `x.sin_cos()`, by this mode's implementation.
    pub fn sin_cos(self, x: f32) -> (f32, f32) {
        match self {
            Self::Platform => x.sin_cos(),
            Self::Portable => (libm::sinf(x), libm::cosf(x)),
        }
    }

    /// `x.tan()`, by this mode's implementation.
    pub fn tan(self, x: f32) -> f32 {
        match self {
            Self::Platform => x.tan(),
            Self::Portable => libm::tanf(x),
        }
    }

    /// `x.acos()`, by this mode's implementation.
    pub fn acos(self, x: f32) -> f32 {
        match self {
            Self::Platform => x.acos(),
            Self::Portable => libm::acosf(x),
        }
    }

    /// The angle of `(x, y)`: `y.atan2(x)`, by this mode's implementation
    /// (argument order as `f32::atan2`'s receiver and argument).
    pub fn atan2(self, y: f32, x: f32) -> f32 {
        match self {
            Self::Platform => y.atan2(x),
            Self::Portable => libm::atan2f(y, x),
        }
    }

    /// `x.hypot(y)`, by this mode's implementation. Not needed by this
    /// crate's own derivation; offered for a consumer's.
    pub fn hypot(self, x: f32, y: f32) -> f32 {
        match self {
            Self::Platform => x.hypot(y),
            Self::Portable => libm::hypotf(x, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Portable is the `libm` crate's, bit for bit, at awkward arguments.
    #[test]
    fn portable_math_is_the_libm_crates() {
        let m = MathMode::Portable;
        for x in [0.3_f32, -1.2, 2.0, 1.0e-3, 3.0e3, -0.999_9] {
            assert_eq!(m.sin(x).to_bits(), libm::sinf(x).to_bits());
            assert_eq!(m.cos(x).to_bits(), libm::cosf(x).to_bits());
            assert_eq!(m.sin_cos(x).0.to_bits(), libm::sinf(x).to_bits());
            assert_eq!(m.sin_cos(x).1.to_bits(), libm::cosf(x).to_bits());
            assert_eq!(m.tan(x).to_bits(), libm::tanf(x).to_bits());
            assert_eq!(m.atan2(x, 0.7).to_bits(), libm::atan2f(x, 0.7).to_bits());
            assert_eq!(m.hypot(x, 0.7).to_bits(), libm::hypotf(x, 0.7).to_bits());
        }
        for x in [0.0_f32, 0.5, -0.999_9, 1.0] {
            assert_eq!(m.acos(x).to_bits(), libm::acosf(x).to_bits());
        }
    }

    /// Platform is the target's own `f32` functions, bit for bit.
    #[test]
    fn platform_math_is_the_targets() {
        let m = MathMode::Platform;
        for x in [0.3_f32, -1.2, 2.0, 1.0e-3] {
            assert_eq!(m.sin(x).to_bits(), x.sin().to_bits());
            assert_eq!(m.cos(x).to_bits(), x.cos().to_bits());
            assert_eq!(m.tan(x).to_bits(), x.tan().to_bits());
            assert_eq!(m.atan2(x, 0.7).to_bits(), x.atan2(0.7).to_bits());
            assert_eq!(m.hypot(x, 0.7).to_bits(), x.hypot(0.7).to_bits());
        }
        assert_eq!(MathMode::default(), MathMode::Platform);
    }
}
