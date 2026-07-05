//! janus-kinetic: UGKWP solver (DUGKS wave part + stochastic particle part)
//! with Shakhov collision model. This crate must never depend on any UI/
//! visualization crate.
//!
//! ## 2D vs 3D modules
//!
//! The original (M1-M3) 2D solver (`bc`, `collision`, `maxwellian`,
//! `particles`, `solver`, `velocity_grid`, `coupled`) uses the *reduced*
//! (g,h) two-distribution formulation (gamma = 5/3) to recover the correct
//! monatomic 3-DOF energy from a 2-component discretized velocity space. The
//! M4 3D extension (`bc3d`, `collision3d`, `maxwellian3d`, `particles3d`,
//! `solver3d`, `velocity_grid3d`) discretizes the full 3-component velocity
//! space directly and therefore drops the (g,h) reduction entirely — see
//! `maxwellian3d`'s module docs for the detailed justification (it is a
//! legitimate simplification, not a shortcut: the reduction's entire purpose
//! was patching a DOF deficit that does not exist once velocity space is
//! genuinely 3D).
//!
//! Algorithm references:
//! - DUGKS: Guo, Xu & Wang, "Discrete unified gas kinetic scheme for all
//!   Knudsen number flows: Low-speed isothermal case", Phys. Rev. E 88,
//!   033305 (2013).
//! - UGKWP: Liu, S., Zhu, Y., Xu, K., "A unified gas kinetic wave-particle
//!   method I: Continuum and rarefied gas dynamics", J. Comput. Phys. 401,
//!   108977 (2020).
//! - Shakhov: Shakhov, "Generalization of the Krook kinetic relaxation
//!   equation", Fluid Dynamics 3, 95 (1968).
//! - Reduced (g,h) (2D solver only): Xu, K., Huang, J.-C., J. Comput. Phys.
//!   229, 7747-7764 (2010).
//! - Gauss-Hermite quadrature (shared 1D generator, reused by both the 2D
//!   and 3D velocity grids): Golub, G. H., Welsch, J. H., Math. Comp. 23,
//!   221-230 (1969).

pub mod bc;
pub mod bc3d;
pub mod collision;
pub mod collision3d;
pub mod coupled;
pub mod coupled3d;
pub mod fft;
pub mod gas_model;
pub mod kn;
pub mod maxwellian;
pub mod maxwellian3d;
pub mod particles;
pub mod particles3d;
pub mod solver;
pub mod solver3d;
pub mod spectral_collision;
pub mod velocity_grid;
pub mod velocity_grid3d;

pub use bc::{BoundaryCondition, BoundaryConditionKernel};
pub use bc3d::{BoundaryCondition3D, BoundaryConditionKernel3D};
pub use collision::{Collision, Shakhov};
pub use collision3d::{Collision3D, Shakhov3D};
pub use coupled::{FluxKernel, UgkwpSolver};
pub use coupled3d::{FluxKernel3D, UgkwpSolver3D};
pub use gas_model::{CustomGasModel, GasModel, IdealVhsGasModel, VirialGasModel};
pub use particles::Particles;
pub use particles3d::Particles3D;
pub use solver::{DugksSolver, TimeScheme, TimeStepper};
pub use solver3d::DugksSolver3D;
pub use spectral_collision::{FastSpectralCollision, SpectralGrid};
pub use velocity_grid::VelocityGrid2D;
pub use velocity_grid3d::VelocityGrid3D;
