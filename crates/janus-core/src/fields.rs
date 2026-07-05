//! Macroscopic field storage, Structure-of-Arrays, one entry per cell.
//!
//! Layout follows ENGINEERING_SPEC.md §5 exactly: SoA with `[Vec<f64>; 2]`
//! for 2-component vector fields (momentum, heat flux) rather than a faer or
//! nalgebra vector type, per the spec's explicit instruction to keep the
//! hot-path element type minimal.

/// Cell-centered macroscopic (moment) fields over a `Grid2D`, SoA layout.
///
/// All `Vec`s have the same length `ncells` and are indexed by the same
/// linear cell index as `Grid2D::idx`.
#[derive(Clone, Debug)]
pub struct MacroFields {
    pub rho: Vec<f64>,
    pub mom: [Vec<f64>; 2],
    pub energy: Vec<f64>,
    /// Optional higher moments (viscous stress components: xx, yy, xy) for
    /// the transition regime / diagnostics. Populated lazily; empty Vec of
    /// `[0.0;3]` is a valid "unused" state.
    pub stress: Vec<[f64; 3]>,
    pub heat: [Vec<f64>; 2],
    /// Local Knudsen number, drives the wave/particle split in M2. Unused
    /// (all zero) in the M1 pure-wave solver but kept for forward
    /// compatibility with the field layout the spec mandates.
    pub kn_loc: Vec<f64>,
}

impl MacroFields {
    pub fn zeros(ncells: usize) -> Self {
        Self {
            rho: vec![0.0; ncells],
            mom: [vec![0.0; ncells], vec![0.0; ncells]],
            energy: vec![0.0; ncells],
            stress: vec![[0.0; 3]; ncells],
            heat: [vec![0.0; ncells], vec![0.0; ncells]],
            kn_loc: vec![0.0; ncells],
        }
    }

    pub fn ncells(&self) -> usize {
        self.rho.len()
    }

    /// Bulk velocity `u = mom / rho` for cell `c`.
    #[inline]
    pub fn velocity(&self, c: usize) -> [f64; 2] {
        let rho = self.rho[c].max(f64::MIN_POSITIVE);
        [self.mom[0][c] / rho, self.mom[1][c] / rho]
    }

    /// Temperature from ideal-gas EOS: `E = rho*(u^2/2) + rho*R*T/(gamma-1)`
    /// for a monatomic gas in 2D velocity space (specific gas constant `r`,
    /// D=2 translational DOF used in the DVM reduction, see janus-kinetic).
    /// `dof` is the total effective degrees of freedom (2 for a 2D DVM
    /// reduction of a monatomic gas without internal energy).
    #[inline]
    pub fn temperature(&self, c: usize, r_gas: f64, dof: f64) -> f64 {
        let rho = self.rho[c].max(f64::MIN_POSITIVE);
        let u = self.velocity(c);
        let kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let internal = self.energy[c] - kinetic;
        (2.0 * internal) / (rho * dof * r_gas)
    }

    /// Pressure from ideal gas law `p = rho * r_gas * T`.
    #[inline]
    pub fn pressure(&self, c: usize, r_gas: f64, dof: f64) -> f64 {
        self.rho[c] * r_gas * self.temperature(c, r_gas, dof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_consistent_lengths() {
        let f = MacroFields::zeros(10);
        assert_eq!(f.rho.len(), 10);
        assert_eq!(f.mom[0].len(), 10);
        assert_eq!(f.mom[1].len(), 10);
        assert_eq!(f.energy.len(), 10);
        assert_eq!(f.stress.len(), 10);
        assert_eq!(f.heat[0].len(), 10);
        assert_eq!(f.kn_loc.len(), 10);
    }

    #[test]
    fn velocity_and_temperature() {
        let mut f = MacroFields::zeros(1);
        f.rho[0] = 2.0;
        f.mom[0][0] = 4.0; // u_x = 2
        f.mom[1][0] = 0.0;
        let r_gas = 287.0;
        let dof = 2.0;
        // pick energy so kinetic + internal matches a target T
        let u = f.velocity(0);
        let kinetic = 0.5 * f.rho[0] * (u[0] * u[0] + u[1] * u[1]);
        let target_t = 300.0;
        let internal = 0.5 * f.rho[0] * dof * r_gas * target_t;
        f.energy[0] = kinetic + internal;
        let t = f.temperature(0, r_gas, dof);
        assert!((t - target_t).abs() < 1e-9);
    }
}
