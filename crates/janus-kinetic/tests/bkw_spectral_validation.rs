//! Validation of `FastSpectralCollision` against the analytic Bobylev-
//! Krook-Wu (BKW) solution of the spatially-homogeneous Boltzmann equation
//! for Maxwell molecules (`gamma = 0`).
//!
//! Reference: Bobylev, A. V., Sov. Phys. Dokl. 20, 820 (1975); Krook, M.,
//! Wu, T. T., "Formation of Maxwellian Tails", Phys. Rev. Lett. 36, 1107
//! (1976) and Phys. Fluids 20, 1589 (1977). The BKW solution is the unique
//! known closed-form time-dependent solution of the nonlinear spatially-
//! homogeneous Boltzmann equation for Maxwell molecules, making it the
//! standard analytic benchmark for any Boltzmann collision-operator
//! implementation (used this way in essentially every fast-spectral-method
//! paper since Pareschi & Perthame 1996).
//!
//! The BKW distribution (isotropic, zero bulk velocity, unit density,
//! normalized so `R*T=1` at a reference time) is:
//!
//! ```text
//! f_BKW(v, t) = 1 / (2 pi K)^{3/2} * exp(-|v|^2 / (2K))
//!               * [ (5K-3)/(2K) + (1-K)/(2K^2) |v|^2 ]
//! K(t) = 1 - exp(-t/6)   (for the normalization used here; the standard
//!                          BKW time constant for Maxwell molecules)
//! ```
//!
//! and it satisfies `df/dt = Q(f,f)` EXACTLY for this `K(t)`. We therefore
//! validate the discrete `FastSpectralCollision::apply_to_distribution` by:
//! checking that `Q(f_BKW(., t0), f_BKW(., t0))` (evaluated by the spectral
//! operator) matches the analytic time-derivative `dK/dt * df_BKW/dK` at a
//! sample time `t0`, to within the truncation/discretization error expected
//! of a modest-resolution spectral grid.

use janus_kinetic::spectral_collision::{FastSpectralCollision, SpectralGrid};

/// BKW distribution at parameter `k` (writing `K` as `k` to avoid clashing
/// with Rust's `K` type-parameter convention), unit density, zero bulk
/// velocity, `R*T` normalized into `k` itself (dimensionless velocity units
/// consistent with the reference's convention: velocities here are in units
/// where the "temperature" scale is absorbed into `k`).
fn bkw_f(v: [f64; 3], k: f64) -> f64 {
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let norm = 1.0 / (2.0 * std::f64::consts::PI * k).powf(1.5);
    let bracket = (5.0 * k - 3.0) / (2.0 * k) + (1.0 - k) / (2.0 * k * k) * v2;
    norm * (-v2 / (2.0 * k)).exp() * bracket
}

/// Analytic `df_BKW/dt = dK/dt * df_BKW/dK` at parameter `k`, `dK/dt =
/// (1/6) exp(-t/6) = (1-k)/6` (from `k(t) = 1 - exp(-t/6)` =>
/// `dk/dt = (1-k)/6`), with `df/dK` obtained by direct differentiation of
/// `bkw_f`'s closed form.
fn bkw_dfdt(v: [f64; 3], k: f64) -> f64 {
    let dk_dt = (1.0 - k) / 6.0;
    // Numerical derivative in k (central difference) is simpler and just as
    // exact here (bkw_f is smooth in k away from k=0) than re-deriving and
    // hand-coding the analytic df/dK bracket term, and avoids a second
    // hand-differentiation bug surface independent from `bkw_f` itself.
    let h = 1e-5;
    let df_dk = (bkw_f(v, k + h) - bkw_f(v, k - h)) / (2.0 * h);
    dk_dt * df_dk
}

// IGNORED — intentionally, and permanently at the unit-test level: the BKW
// exact solution is specific to MAXWELL molecules (gamma=0), whereas the
// decoupled *fast* Mouhot-Pareschi scheme in 3D is the HARD-SPHERE kernel
// (Maxwell molecules decouple only in 2D; see spectral_collision.rs module
// docs). The operator is otherwise validated by structural checks that hold for
// any physical kernel — Q(M,M)=0, mass/momentum/energy conservation, and the
// H-theorem — in the `spectral_collision.rs` unit tests and
// `fast_spectral_collision_conserves_mass_on_bkw_state` below. Reproducing the
// Maxwell-molecule BKW *rate* would require the paper's Appendix non-decoupled
// construction (a separate, larger undertaking).
#[test]
#[ignore = "BKW rate is Maxwell-molecule-specific; the fast decoupled 3D scheme is hard-sphere — see note"]
fn fast_spectral_collision_matches_bkw_analytic_rate() {
    // Modest grid: fast-spectral methods converge quickly (spectral
    // accuracy) for smooth kernels/distributions like BKW's, so even a
    // coarse-by-DVM-standards grid should track the analytic rate to within
    // a few tens of percent — the point of this test is to catch gross
    // implementation errors (sign flips, wrong convolution structure,
    // mass-conservation violations), not to achieve production-grade
    // quantitative accuracy from a unit test.
    let l = 8.0; // velocity truncation radius, in BKW's dimensionless units
    let grid = SpectralGrid::new(16, l);
    let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 48);

    let t0 = 3.0; // sample time, well inside the smooth (k < 1) BKW regime
    let k0 = 1.0 - (-t0 / 6.0f64).exp();

    let n = grid.ntotal();
    let mut f = vec![0.0; n];
    let mut expected = vec![0.0; n];
    for i in 0..n {
        let v = grid.velocity_at(i);
        f[i] = bkw_f(v, k0);
        expected[i] = bkw_dfdt(v, k0);
    }

    let mut q = vec![0.0; n];
    op.apply_to_distribution(&f, &mut q);

    // Compare on a moderate-|v| sub-region (the BKW solution's high-|v| tail
    // is where a truncated/discretized velocity grid's error is largest and
    // least representative of the operator's core correctness — restricting
    // the comparison to |v| < l/2 keeps the check meaningful without
    // requiring an enormous grid to pass in a unit test).
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let v = grid.velocity_at(i);
        let vmag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if vmag < l * 0.5 {
            num += (q[i] - expected[i]).powi(2);
            den += expected[i].powi(2);
        }
    }
    let rel_l2_error = (num / den.max(1e-300)).sqrt();
    assert!(
        rel_l2_error < 0.5,
        "fast-spectral collision rate deviates too far from analytic BKW rate: rel L2 error = {rel_l2_error}"
    );
}

// Structural mass conservation holds for ANY kernel/state, so this is a valid
// check of the (hard-sphere) fast-spectral operator even though the state is the
// (Maxwell-molecule) BKW distribution — it exercises the operator on a nontrivial
// non-Maxwellian input.
#[test]
fn fast_spectral_collision_conserves_mass_on_bkw_state() {
    let l = 8.0;
    let grid = SpectralGrid::new(16, l);
    let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 6);
    let k0 = 0.5;
    let n = grid.ntotal();
    let mut f = vec![0.0; n];
    for i in 0..n {
        let v = grid.velocity_at(i);
        f[i] = bkw_f(v, k0);
    }
    let mut q = vec![0.0; n];
    op.apply_to_distribution(&f, &mut q);
    let dv3 = grid.dv3();
    let mass_rate: f64 = q.iter().sum::<f64>() * dv3;
    let mass: f64 = f.iter().sum::<f64>() * dv3;
    assert!(
        mass_rate.abs() < 0.1 * mass.abs().max(1e-9),
        "collision operator should conserve mass on the BKW state: d(mass)/dt = {mass_rate}, mass = {mass}"
    );
}
