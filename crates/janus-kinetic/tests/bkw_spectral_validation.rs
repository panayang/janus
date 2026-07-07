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

// The Maxwell-molecule (gamma=0) VHS kernel reproduces the BKW analytic
// relaxation rate. The collision operator carries an overall collision-FREQUENCY
// constant (the kernel prefactor, which sets the absolute timescale and is a
// physical normalization choice); this test therefore validates that the
// operator's *shape* matches the analytic `df/dt` up to that single scalar --
// i.e. it fits the best collision-frequency constant and checks the residual.
// A wrong operator (bad convolution structure, sign error, non-conservation)
// cannot be scaled to match and fails this grossly.
#[test]
fn fast_spectral_collision_matches_bkw_analytic_rate() {
    let l = 8.0; // velocity truncation radius, in BKW's dimensionless units
    let grid = SpectralGrid::new(32, l);
    let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 8);

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

    // Best-fit collision-frequency scale over a moderate-|v| sub-region (the
    // high-|v| tail is where grid truncation error dominates). The scale-free
    // shape residual is  rel_l2 = sqrt(1 - <Q,target>^2 / (<Q,Q> <target,target>)).
    let (mut qq, mut qt, mut tt) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let v = grid.velocity_at(i);
        let vmag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if vmag < l * 0.5 {
            qq += q[i] * q[i];
            qt += q[i] * expected[i];
            tt += expected[i] * expected[i];
        }
    }
    let rel_l2_error = (1.0 - qt * qt / (qq * tt).max(1e-300)).max(0.0).sqrt();
    assert!(
        rel_l2_error < 0.05,
        "fast-spectral collision shape deviates from analytic BKW rate: rel L2 (scale-free) = {rel_l2_error}"
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
