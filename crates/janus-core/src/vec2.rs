//! Tiny POD 2-vector type used in hot loops instead of pulling in `nalgebra`.
//!
//! DESIGN: per ENGINEERING_SPEC.md §5, "a tiny Vec2/Vec3 POD type with
//! `#[repr(C)]` + `bytemuck::Pod`; do not pull nalgebra." `glam` is on the
//! approved whitelist and already provides a SIMD-friendly Vec2, but the spec
//! explicitly sketches a hand-rolled repr(C) POD struct for field storage
//! (mmap/bytemuck casting requires a stable, exact byte layout we own). We
//! define our own `Vec2` here for that reason; `glam::Vec2/DVec2` may still be
//! used internally for scratch math where convenient, but is not part of any
//! on-disk or SoA storage layout.

use bytemuck::{Pod, Zeroable};

/// Plain-old-data 2D vector of `f64`, used as an element type in SoA arrays
/// and for bytemuck zero-copy casting to/from raw `.jvtk` blocks.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

// SAFETY: Vec2 is #[repr(C)], contains only f64 fields, has no padding (two
// 8-byte fields, naturally 8-byte aligned), and all bit patterns of f64 are
// valid (NaN included), so it satisfies the Pod/Zeroable requirements: no
// invalid bit patterns, no interior mutability, no padding bytes.
unsafe impl Pod for Vec2 {}
// SAFETY: the all-zero bit pattern is a valid Vec2 (0.0, 0.0).
unsafe impl Zeroable for Vec2 {}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    #[inline]
    pub fn scale(self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_is_pod_layout() {
        assert_eq!(std::mem::size_of::<Vec2>(), 16);
        assert_eq!(std::mem::align_of::<Vec2>(), 8);
    }

    #[test]
    fn basic_ops() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0);
        assert!((a.norm() - 5f64.sqrt()).abs() < 1e-12);
        assert_eq!(a.add(b), Vec2::new(4.0, 6.0));
    }
}
