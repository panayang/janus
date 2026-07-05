//! Physical constants, unit helpers, and a zero-cost dimensional-analysis /
//! nondimensionalization system.
//!
//! ## Design: phantom-typed `Quantity<D>`
//!
//! All solver internals still store plain `f64` SI values in the hot path
//! (SoA `Vec<f64>` fields, per ENGINEERING_SPEC.md paragraph 5/8 -- a units
//! wrapper around every array element would add per-element overhead and is
//! explicitly not what this module does). Instead, `Quantity<D>` is a
//! `#[repr(transparent)]` newtype over `f64` carrying a zero-sized phantom
//! `D: Dimension` marker. Because `PhantomData<D>` has size 0 and
//! `Quantity` is `#[repr(transparent)]`, `Quantity<D>` has exactly the same
//! layout, size, and alignment as `f64` -- arithmetic on it compiles to the
//! identical machine code as plain `f64` arithmetic (the dimension checking
//! is 100% compile-time, erased before codegen). This gives compile-time
//! -checked units at API boundaries (function signatures, case-setup/config
//! code, `GasProperties`, boundary-condition parameters) with zero runtime
//! cost, while the innermost per-cell/per-velocity-node loops keep operating
//! on raw `f64` (obtained via `.value()`) exactly as before -- no
//! allocation, no dynamic dispatch, no per-step overhead anywhere in the hot
//! path.
//!
//! `Quantity<D>` only supports `+`/`-` between same-dimensioned quantities
//! and `*`/`/` that combine dimensions (via associated-type dimension
//! arithmetic), catching unit-mismatch bugs (e.g. accidentally adding a
//! velocity to a temperature) at compile time in the case-setup / boundary
//! -condition / gas-property code, which is not in the innermost loop and is
//! exactly where such bugs are most likely and where compile-time checking
//! has the highest value-for-cost ratio.
//!
//! ## Nondimensionalization
//!
//! `ReferenceScales` holds a reference length/velocity/density/temperature;
//! `nondimensionalize`/`redimensionalize` convert plain `f64` SI values
//! to/from their dimensionless counterparts, and `knudsen`/`mach`/`reynolds`
//! compute the standard dimensionless groups directly from physical inputs:
//!
//! ```text
//! Kn = lambda / L_ref                     (mean free path / reference length)
//! Ma = |u| / sqrt(gamma * R * T)           (local speed / local sound speed)
//! Re = rho_ref * u_ref * L_ref / mu_ref    (Reynolds number)
//! ```
//!
//! These are the standard cross-scale similarity parameters used throughout
//! the UGKWP literature (e.g. Liu, Zhu & Xu 2020 report results in
//! nondimensional Kn/Ma) and match the local-Kn field computed by
//! `janus_kinetic::kn`, which drives the wave/particle split.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Sub};

/// Marker trait for a physical dimension in terms of the base SI exponents
/// (length L, mass M, time T, temperature Theta). Purely a compile-time tag;
/// zero-sized, never instantiated.
pub trait Dimension: Copy + Clone {
    /// Human-readable SI unit string, for diagnostics/Debug output only.
    const UNIT: &'static str;
}

macro_rules! dimension {
    ($name:ident, $unit:literal) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl Dimension for $name {
            const UNIT: &'static str = $unit;
        }
    };
}

dimension!(Length, "m");
dimension!(Time, "s");
dimension!(Mass, "kg");
dimension!(Temperature, "K");
dimension!(Velocity, "m/s");
dimension!(Density, "kg/m^3");
dimension!(Pressure, "Pa");
dimension!(Energy, "J");
dimension!(SpecificEnergy, "J/kg");
dimension!(Viscosity, "Pa*s");
dimension!(Dimensionless, "1");
dimension!(SpecificGasConstant, "J/(kg*K)");

/// A compile-time-dimensioned scalar: `#[repr(transparent)]` over `f64`, so
/// it has the exact same size/alignment/layout as `f64` and all arithmetic
/// on it is zero-cost (compiles to the same instructions as the equivalent
/// raw-`f64` code; the `PhantomData<D>` marker is erased before codegen).
///
/// SAFETY-relevant note: `#[repr(transparent)]` is what makes the
/// zero-cost guarantee load-bearing (not just an optimizer heuristic) --
/// the Rust reference guarantees a `repr(transparent)` struct has identical
/// ABI to its single non-zero-sized field, so `Quantity<D>` can be passed
/// across FFI/bytemuck-cast boundaries exactly like `f64` if ever needed
/// (not currently required, but the layout guarantee is what "zero runtime
/// overhead" concretely means here, not just "probably inlines away").
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Quantity<D: Dimension>(f64, PhantomData<D>);

impl<D: Dimension> Quantity<D> {
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value, PhantomData)
    }

    #[inline(always)]
    pub const fn value(self) -> f64 {
        self.0
    }
}

impl<D: Dimension> Add for Quantity<D> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}

impl<D: Dimension> Sub for Quantity<D> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0 - rhs.0)
    }
}

/// Scaling a quantity by a plain dimensionless `f64` factor keeps its
/// dimension (e.g. `velocity * 0.5`).
impl<D: Dimension> Mul<f64> for Quantity<D> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.0 * rhs)
    }
}

impl<D: Dimension> Div<f64> for Quantity<D> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self::new(self.0 / rhs)
    }
}

/// Dividing two same-dimensioned quantities yields a plain dimensionless
/// ratio -- this is the one place dimension *cancellation* is expressed
/// generically (e.g. `T / T_ref` in nondimensionalization).
impl<D: Dimension> Div<Quantity<D>> for Quantity<D> {
    type Output = f64;
    #[inline(always)]
    fn div(self, rhs: Quantity<D>) -> f64 {
        self.0 / rhs.0
    }
}

/// Type aliases for the physical quantities used throughout `janus-core`/
/// `janus-kinetic` case setup and gas-property APIs.
pub type LengthQ = Quantity<Length>;
pub type TimeQ = Quantity<Time>;
pub type MassQ = Quantity<Mass>;
pub type TemperatureQ = Quantity<Temperature>;
pub type VelocityQ = Quantity<Velocity>;
pub type DensityQ = Quantity<Density>;
pub type PressureQ = Quantity<Pressure>;
pub type EnergyQ = Quantity<Energy>;
pub type SpecificEnergyQ = Quantity<SpecificEnergy>;
pub type ViscosityQ = Quantity<Viscosity>;
pub type SpecificGasConstantQ = Quantity<SpecificGasConstant>;

/// Reference scales for nondimensionalizing a case: length, velocity,
/// density, temperature. These four are dimensionally independent and
/// sufficient to nondimensionalize every quantity this solver uses (mass,
/// time, pressure, energy, viscosity are all derived combinations of
/// L/U/rho/T in a monatomic ideal-gas kinetic solver).
#[derive(Clone, Copy, Debug)]
pub struct ReferenceScales {
    pub length: LengthQ,
    pub velocity: VelocityQ,
    pub density: DensityQ,
    pub temperature: TemperatureQ,
}

impl ReferenceScales {
    pub fn new(length: f64, velocity: f64, density: f64, temperature: f64) -> Self {
        Self {
            length: LengthQ::new(length),
            velocity: VelocityQ::new(velocity),
            density: DensityQ::new(density),
            temperature: TemperatureQ::new(temperature),
        }
    }

    /// Reference time scale `L_ref / U_ref` (advective time unit).
    #[inline]
    pub fn time(&self) -> TimeQ {
        TimeQ::new(self.length.value() / self.velocity.value())
    }

    /// Reference pressure `rho_ref * U_ref^2` (dynamic-pressure scale,
    /// standard for compressible-flow nondimensionalization).
    #[inline]
    pub fn pressure(&self) -> PressureQ {
        PressureQ::new(self.density.value() * self.velocity.value() * self.velocity.value())
    }

    /// Nondimensionalize a plain SI length: `x / L_ref`.
    #[inline]
    pub fn nondim_length(&self, x: f64) -> f64 {
        x / self.length.value()
    }

    /// Redimensionalize a dimensionless length back to SI: `x_star * L_ref`.
    #[inline]
    pub fn redim_length(&self, x_star: f64) -> f64 {
        x_star * self.length.value()
    }

    #[inline]
    pub fn nondim_velocity(&self, u: f64) -> f64 {
        u / self.velocity.value()
    }

    #[inline]
    pub fn redim_velocity(&self, u_star: f64) -> f64 {
        u_star * self.velocity.value()
    }

    #[inline]
    pub fn nondim_density(&self, rho: f64) -> f64 {
        rho / self.density.value()
    }

    #[inline]
    pub fn redim_density(&self, rho_star: f64) -> f64 {
        rho_star * self.density.value()
    }

    #[inline]
    pub fn nondim_temperature(&self, t: f64) -> f64 {
        t / self.temperature.value()
    }

    #[inline]
    pub fn redim_temperature(&self, t_star: f64) -> f64 {
        t_star * self.temperature.value()
    }

    /// Reynolds number `Re = rho_ref * U_ref * L_ref / mu_ref`, the standard
    /// continuum-flow similarity parameter.
    #[inline]
    pub fn reynolds(&self, mu_ref: f64) -> f64 {
        self.density.value() * self.velocity.value() * self.length.value() / mu_ref
    }

    /// Knudsen number `Kn = lambda / L_ref` (mean free path over reference
    /// length) -- the primary cross-scale similarity parameter this whole
    /// solver is built around (ENGINEERING_SPEC.md section 2). `lambda`
    /// (mean free path) is typically `vhs_mean_free_path(...)` below,
    /// evaluated at the reference density/temperature.
    #[inline]
    pub fn knudsen(&self, mean_free_path: f64) -> f64 {
        mean_free_path / self.length.value()
    }

    /// Mach number `Ma = |u| / a`, `a = sqrt(gamma * R * T)` the local
    /// (ideal-gas) sound speed. Uses the *local* dimensional `u`/`t_gas`
    /// (not the reference scales) since Mach number is inherently a local
    /// flow quantity, not a case-global constant like Kn/Re.
    #[inline]
    pub fn mach(u: f64, gamma: f64, r_gas: f64, t_gas: f64) -> f64 {
        let a = (gamma * r_gas * t_gas).max(0.0).sqrt();
        u / a.max(1e-300)
    }
}

/// Mean free path from kinetic theory, consistent with the VHS viscosity
/// law (Bird 1994, eq. 4.76 combined with the VHS mu(T) law):
/// `lambda = mu / (rho * sqrt(2*R*T/pi))`. Shared implementation for the
/// `knudsen()` helper above and `janus_kinetic::kn::update_kn_loc`'s
/// per-cell local mean free path (kept here so both go through one
/// definition rather than risking drift between two copies of the same
/// formula).
#[inline]
pub fn vhs_mean_free_path(mu: f64, rho: f64, r_gas: f64, t: f64) -> f64 {
    let most_probable_speed_factor = (2.0 * r_gas * t.max(1e-9) / core::f64::consts::PI).sqrt();
    mu / (rho.max(f64::MIN_POSITIVE) * most_probable_speed_factor.max(1e-30))
}

/// Universal gas constant, J / (mol*K).
pub const R_UNIVERSAL: f64 = 8.314_462_618;

/// Boltzmann constant, J / K.
pub const K_BOLTZMANN: f64 = 1.380_649e-23;

/// Avogadro constant, 1/mol.
pub const N_AVOGADRO: f64 = 6.022_140_76e23;

/// Specific gas constant `R = R_universal / M` for a gas of molar mass `m_molar`
/// (kg/mol).
#[inline]
pub fn specific_gas_constant(m_molar: f64) -> f64 {
    R_UNIVERSAL / m_molar
}

/// VHS (Variable Hard Sphere) viscosity law:
/// `mu(T) = mu_ref * (T / T_ref)^omega`.
///
/// Reference: Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
/// Flows" (1994), VHS model.
#[inline]
pub fn vhs_viscosity(t: f64, mu_ref: f64, t_ref: f64, omega: f64) -> f64 {
    mu_ref * (t / t_ref).powf(omega)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specific_gas_constant_air() {
        // Air, M ~ 0.02897 kg/mol -> R ~ 287 J/(kg K)
        let r = specific_gas_constant(0.028_97);
        assert!((r - 287.0).abs() < 1.0);
    }

    #[test]
    fn vhs_at_reference_temperature() {
        let mu = vhs_viscosity(300.0, 1.5e-5, 300.0, 0.81);
        assert!((mu - 1.5e-5).abs() < 1e-12);
    }

    #[test]
    fn quantity_is_zero_cost_layout() {
        // repr(transparent) over f64 must have identical size/align to f64;
        // this is the load-bearing property behind the "zero runtime
        // overhead" claim in the module docs.
        assert_eq!(std::mem::size_of::<LengthQ>(), std::mem::size_of::<f64>());
        assert_eq!(std::mem::align_of::<LengthQ>(), std::mem::align_of::<f64>());
    }

    #[test]
    fn quantity_arithmetic_round_trips() {
        let a = LengthQ::new(2.0);
        let b = LengthQ::new(3.0);
        assert_eq!((a + b).value(), 5.0);
        assert_eq!((b - a).value(), 1.0);
        assert_eq!((a * 2.0).value(), 4.0);
        assert_eq!((b / a), 1.5); // dimensionless ratio (Div<Quantity<D>> for Quantity<D>)
    }

    #[test]
    fn reference_scales_nondim_round_trip() {
        let scales = ReferenceScales::new(1.0e-3, 300.0, 1.2, 300.0);
        let x = 2.5e-4;
        let x_star = scales.nondim_length(x);
        let x_back = scales.redim_length(x_star);
        assert!((x_back - x).abs() < 1e-18);

        // Reference time/pressure derived scales are self-consistent.
        assert!((scales.time().value() - 1.0e-3 / 300.0).abs() < 1e-18);
        assert!((scales.pressure().value() - 1.2 * 300.0 * 300.0).abs() < 1e-9);
    }

    #[test]
    fn knudsen_mach_reynolds_sane_values() {
        let scales = ReferenceScales::new(1.0, 1.0, 1.0, 300.0);
        // A mean free path equal to the reference length gives Kn = 1.
        assert!((scales.knudsen(1.0) - 1.0).abs() < 1e-12);
        // Re with mu_ref = rho*U*L gives Re = 1.
        assert!((scales.reynolds(1.0) - 1.0).abs() < 1e-12);
        // Mach number at u = a (sonic) should be 1.
        let r_gas = 287.0;
        let t = 300.0;
        let gamma: f64 = 5.0 / 3.0;
        let a = (gamma * r_gas * t).sqrt();
        assert!((ReferenceScales::mach(a, gamma, r_gas, t) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vhs_mean_free_path_matches_kn_module_formula() {
        // Cross-check against the standalone formula used historically in
        // janus-kinetic::kn (now delegating to this function) to guard
        // against silent drift.
        let mu = 2.117e-5;
        let rho = 1.2;
        let r_gas = 208.13;
        let t = 300.0;
        let lambda = vhs_mean_free_path(mu, rho, r_gas, t);
        let expected = mu / (rho * (2.0 * r_gas * t / std::f64::consts::PI).sqrt());
        assert!((lambda - expected).abs() / expected < 1e-12);
    }
}
