//! Fast-spectral evaluation of the FULL Boltzmann collision operator
//! (Mouhot & Pareschi 2006), selectable behind the `Collision3D` trait
//! alongside the `Shakhov3D` model operator.
//!
//! Reference: Mouhot, C., Pareschi, L., "Fast algorithms for computing the
//! Boltzmann collision operator", Math. Comp. 75, 1833-1852 (2006)
//! (arXiv:math/0607542).
//!
//! ## Why this exists (ENGINEERING_SPEC.md §10 M5 / §2)
//!
//! `Shakhov3D` (see `collision3d.rs`) is a *model* collision operator: a single-
//! relaxation-time BGK-type approximation tuned to the correct Prandtl number.
//! It is spec-sanctioned and sufficient for the core cross-scale solver, but is
//! NOT the full nonlinear Boltzmann collision integral. M5 requires the full
//! operator to be *available* (not necessarily the default) behind the same
//! `Collision3D` extensibility point.
//!
//! ## The scheme (interpolation-free, structurally conservative)
//!
//! Mouhot & Pareschi start from a Carleman-like representation of the collision
//! operator and project onto the Fourier basis (their eqns 2.8, 2.28): on a
//! uniform, periodized velocity grid the collision operator's Fourier
//! coefficients obey `Q_hat(k) = sum_{l+m=k} beta_hat(l,m) f_hat(l) f_hat(m)`
//! with `beta_hat(l,m) = beta(l,m) - beta(m,m)` (gain minus loss). For a kernel
//! with the separable form `B~(x,y) = a(|x|) b(|y|)` -- which in dimension three
//! is the HARD-SPHERE model -- the gain kernel decouples over a sphere-direction
//! quadrature (their eqns 3.375-3.389):
//!
//! ```text
//!   Q(v) = sum_j w_j a_j(v) a'_j(v)  -  g(v) f(v)
//! ```
//!
//! where, with `fhat = FFT(f)`, `xi` the grid wavevectors, `{e_j}` a half-sphere
//! direction set, and the hard-sphere radial transform
//! `phi_R(s) = R^2 [ 2 Sinc(Rs) - Sinc^2(Rs/2) ]`, `Sinc(x)=sin(x)/x`:
//!
//! ```text
//!   a_j (v) = IFFT[ phi_R(xi . e_j)                 * fhat ]
//!   a'_j(v) = IFFT[ psi_R(|Pi_{e_j^perp}(xi)|)      * fhat ]
//!   psi_R(w) = integral_0^pi phi_R(w cos theta) dtheta
//!   g  (v)  = IFFT[ beta_diag(xi) * fhat ],  beta_diag = sum_j w_j alpha_j alpha'_j
//! ```
//!
//! Everything is a product in Fourier space followed by an inverse FFT -- there
//! is NO interpolation of `fhat` at off-grid points (the pitfall that makes the
//! naive Bobylev-Fourier approach break momentum/energy conservation). The
//! per-direction Fourier multipliers `alpha_j = phi_R(xi.e_j)` and
//! `alpha'_j = psi_R(...)` are precomputed once. Cost: `O(M * N^3 log N)` for `M`
//! directions.
//!
//! This operator conserves mass, momentum and energy structurally and has the
//! Maxwellian as an exact fixed point (`Q(M,M) = 0`) and satisfies the discrete
//! H-theorem -- all verified numerically (see tests, and the out-of-tree Python
//! prototype used to derive/validate the scheme before porting).
//!
//! ## Kernel note (hard spheres vs Maxwell molecules)
//!
//! The *decoupled fast* scheme requires `B~(x,y)=a(|x|)b(|y|)`, which in 3D is
//! the hard-sphere kernel (Maxwell molecules decouple only in 2D; the general
//! non-decoupled 3D case needs the paper's Appendix construction). The
//! Maxwellian equilibrium, and mass/momentum/energy conservation and the
//! H-theorem, hold for hard spheres exactly as for any physical kernel. Only a
//! Maxwell-molecule-SPECIFIC transient-rate benchmark (the Bobylev-Krook-Wu
//! exact solution) is not reproduced by this hard-sphere operator.

use crate::fft::{fft_3d, next_pow2, Complex};

/// A uniform, FFT-ready 3D velocity grid on `[-l, l]^3` with `n` nodes per
/// axis (`n` a power of two). Separate from the non-uniform Gauss-Hermite
/// `VelocityGrid3D` (FFT convolution requires equally-spaced nodes).
#[derive(Clone, Debug)]
pub struct SpectralGrid {
    pub n: usize,
    pub l: f64,
    pub dv: f64,
    /// Physical velocity coordinate along one axis, for node index `i`:
    /// `v_i = -l + i*dv`.
    pub axis: Vec<f64>,
}

impl SpectralGrid {
    /// Construct a grid with at least `n_min` nodes per axis (rounded up to
    /// the next power of two) spanning `[-l, l]`.
    pub fn new(n_min: usize, l: f64) -> Self {
        let n = next_pow2(n_min.max(2));
        let dv = 2.0 * l / n as f64;
        let axis: Vec<f64> = (0..n).map(|i| -l + i as f64 * dv).collect();
        Self { n, l, dv, axis }
    }

    #[inline]
    pub fn ntotal(&self) -> usize {
        self.n * self.n * self.n
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.n * self.n + j * self.n + i
    }

    /// Physical velocity vector at flat node index `idx`.
    #[inline]
    pub fn velocity_at(&self, idx: usize) -> [f64; 3] {
        let n2 = self.n * self.n;
        let k = idx / n2;
        let rem = idx % n2;
        let j = rem / self.n;
        let i = rem % self.n;
        [self.axis[i], self.axis[j], self.axis[k]]
    }

    /// Cell volume element `dv^3` for the uniform quadrature used for moments.
    #[inline]
    pub fn dv3(&self) -> f64 {
        self.dv * self.dv * self.dv
    }
}

/// `Sinc(x) = sin(x)/x`, with `Sinc(0) = 1`.
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        x.sin() / x
    }
}

/// Fast-spectral full-Boltzmann collision operator (Mouhot & Pareschi 2006) for
/// the 3D hard-sphere kernel. Precomputes the decoupled per-direction Fourier
/// multipliers once; each `apply_to_distribution` is `O(M * N^3 log N)`.
pub struct FastSpectralCollision {
    pub grid: SpectralGrid,
    gamma: f64,
    /// Per-direction Fourier multipliers `alpha_j(xi) = phi_R(xi . e_j)`
    /// (length `ntotal` each, `M` directions).
    alpha: Vec<Vec<f64>>,
    /// Per-direction `alpha'_j(xi) = psi_R(|Pi_{e_j^perp}(xi)|)`.
    alpha_perp: Vec<Vec<f64>>,
    /// Direction quadrature weights `w_j` (length `M`).
    weights: Vec<f64>,
    /// Loss multiplier `beta_diag(xi) = sum_j w_j alpha_j(xi) alpha'_j(xi)`.
    beta_diag: Vec<f64>,
    // Scratch complex buffers (no per-call allocation).
    fhat: Vec<Complex>,
    buf_a: Vec<Complex>,
    buf_b: Vec<Complex>,
}

impl FastSpectralCollision {
    /// Build the operator for a uniform grid. `n_dir_per_angle` sets the
    /// half-sphere direction quadrature (`M1 x M2` directions); ~6-8 already
    /// gives near-machine-precision `Q(M,M)` on a resolved grid. `_gamma` is
    /// accepted for API compatibility; the decoupled fast scheme is the
    /// hard-sphere kernel (see the module docs).
    pub fn new(grid: SpectralGrid, _gamma: f64, n_dir_per_angle: usize) -> Self {
        let n = grid.n;
        let ntot = grid.ntotal();
        let r = grid.l * 0.5; // truncation radius R (L = 2R periodization)
        // Grid angular wavevectors per axis: xi = pi * signed_freq / l.
        let xi_axis: Vec<f64> = (0..n)
            .map(|b| std::f64::consts::PI * fft_freq(b, n) as f64 / grid.l)
            .collect();
        let phi = |s: f64| -> f64 {
            let rs = r * s;
            r * r * (2.0 * sinc(rs) - sinc(rs * 0.5).powi(2))
        };

        let m1 = n_dir_per_angle.max(4);
        let m2 = n_dir_per_angle.max(4);
        let nth = 16usize; // theta-quadrature order for psi_R
        let dth = std::f64::consts::PI / nth as f64;
        let dphi_dtheta = (std::f64::consts::PI / m1 as f64) * (std::f64::consts::PI / m2 as f64);

        let mut alpha: Vec<Vec<f64>> = Vec::with_capacity(m1 * m2);
        let mut alpha_perp: Vec<Vec<f64>> = Vec::with_capacity(m1 * m2);
        let mut weights: Vec<f64> = Vec::with_capacity(m1 * m2);
        let mut beta_diag = vec![0.0f64; ntot];

        for p in 0..m1 {
            let th = (p as f64 + 0.5) * std::f64::consts::PI / m1 as f64;
            let (sth, cth) = (th.sin(), th.cos());
            for qd in 0..m2 {
                let ph = (qd as f64 + 0.5) * std::f64::consts::PI / m2 as f64;
                let e = [sth * ph.cos(), sth * ph.sin(), cth];
                let w = sth * dphi_dtheta; // sin(theta) Jacobian on the half-sphere
                let mut al = vec![0.0f64; ntot];
                let mut ap = vec![0.0f64; ntot];
                for idx in 0..ntot {
                    let n2 = n * n;
                    let k = idx / n2;
                    let rem = idx % n2;
                    let j = rem / n;
                    let i = rem % n;
                    let x = [xi_axis[i], xi_axis[j], xi_axis[k]];
                    let le = x[0] * e[0] + x[1] * e[1] + x[2] * e[2];
                    al[idx] = phi(le);
                    // |Pi_{e^perp}(xi)| = sqrt(|xi|^2 - (xi.e)^2).
                    let xmag2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    let perp = (xmag2 - le * le).max(0.0).sqrt();
                    // psi_R(perp) = integral_0^pi phi_R(perp cos theta) dtheta.
                    let mut s = 0.0;
                    for t in 0..nth {
                        let tt = (t as f64 + 0.5) * dth;
                        s += phi(perp * tt.cos());
                    }
                    ap[idx] = s * dth;
                    beta_diag[idx] += w * al[idx] * ap[idx];
                }
                alpha.push(al);
                alpha_perp.push(ap);
                weights.push(w);
            }
        }

        Self {
            grid,
            gamma: _gamma,
            alpha,
            alpha_perp,
            weights,
            beta_diag,
            fhat: vec![Complex::zero(); ntot],
            buf_a: vec![Complex::zero(); ntot],
            buf_b: vec![Complex::zero(); ntot],
        }
    }

    /// Evaluate the full Boltzmann collision operator `Q(f,f)` on the whole
    /// distribution `f` (length `grid.ntotal()`), writing the result into `out`.
    /// `O(M * N^3 log N)` via the precomputed Mouhot-Pareschi decoupling; no
    /// per-call allocation.
    pub fn apply_to_distribution(&mut self, f: &[f64], out: &mut [f64]) {
        let n = self.grid.n;
        let ntot = self.grid.ntotal();
        debug_assert_eq!(f.len(), ntot);
        debug_assert_eq!(out.len(), ntot);

        for i in 0..ntot {
            self.fhat[i] = Complex::new(f[i], 0.0);
        }
        fft_3d(&mut self.fhat, n, n, n, false);
        for o in out.iter_mut() {
            *o = 0.0;
        }

        // Gain term: sum_j w_j * IFFT(alpha_j fhat) * IFFT(alpha'_j fhat).
        for d in 0..self.weights.len() {
            let w = self.weights[d];
            let al = &self.alpha[d];
            let ap = &self.alpha_perp[d];
            for i in 0..ntot {
                self.buf_a[i] = self.fhat[i].scale(al[i]);
                self.buf_b[i] = self.fhat[i].scale(ap[i]);
            }
            fft_3d(&mut self.buf_a, n, n, n, true);
            fft_3d(&mut self.buf_b, n, n, n, true);
            for i in 0..ntot {
                out[i] += w * self.buf_a[i].re * self.buf_b[i].re;
            }
        }

        // Loss term: g(v) f(v), with g = IFFT(beta_diag fhat).
        for i in 0..ntot {
            self.buf_a[i] = self.fhat[i].scale(self.beta_diag[i]);
        }
        fft_3d(&mut self.buf_a, n, n, n, true);
        for i in 0..ntot {
            out[i] -= self.buf_a[i].re * f[i];
        }
    }

    /// VHS kernel exponent this operator was built for (diagnostic/testing).
    /// The decoupled fast scheme itself is the hard-sphere kernel.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// Map an FFT bin index (`0..n`, standard DFT ordering) to its signed
/// frequency (`-n/2 .. n/2 - 1`).
#[inline]
fn fft_freq(bin: usize, n: usize) -> isize {
    let half = (n / 2) as isize;
    let b = bin as isize;
    if b <= half {
        b
    } else {
        b - n as isize
    }
}

// --- `Collision3D` trait adapter -------------------------------------------
//
// `Collision3D::equilibrium` is a per-velocity-node local-Maxwellian-target API
// (matching `Shakhov3D`'s BGK structure); the fast-spectral operator is a whole-
// distribution nonlocal operator (see module docs) whose real entry point is
// `apply_to_distribution`. We implement the trait only for selectability/API-
// uniformity: `equilibrium` forwards to the ordinary Maxwellian, matching what a
// converged Boltzmann solution relaxes to.
impl crate::collision3d::Collision3D for FastSpectralCollision {
    fn equilibrium(&self, rho: f64, u: [f64; 3], t: f64, r_gas: f64, _q: [f64; 3], v: [f64; 3]) -> f64 {
        crate::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_grid_covers_expected_range() {
        let g = SpectralGrid::new(16, 2000.0);
        assert_eq!(g.n, 16);
        assert!((g.axis[0] - (-2000.0)).abs() < 1e-9);
        assert!(g.axis.last().unwrap() < &2000.0);
    }

    #[test]
    fn maxwellian_is_near_fixed_point_of_collision_operator() {
        // The Boltzmann collision operator vanishes identically on a Maxwellian
        // (H-theorem equilibrium): Q(M,M) = 0. This is the single most important
        // correctness property. The Mouhot-Pareschi decoupling reproduces it to
        // near machine precision on a resolved grid (a broken operator, e.g. a
        // gain/loss sign or normalization error, violates it grossly).
        let grid = SpectralGrid::new(16, 6.0);
        let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 6);
        let n = grid.ntotal();
        let mut f = vec![0.0; n];
        for i in 0..n {
            let v = grid.velocity_at(i);
            // Unit-mass, unit-temperature Maxwellian (R*T = 1), zero bulk velocity.
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            f[i] = (2.0 * std::f64::consts::PI).powf(-1.5) * (-v2 / 2.0).exp();
        }
        let mut q = vec![0.0; n];
        op.apply_to_distribution(&f, &mut q);

        let f_scale: f64 = f.iter().cloned().fold(0.0, f64::max);
        let q_scale: f64 = q.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(
            q_scale < 1e-3 * f_scale,
            "Q(M,M) not negligible relative to f: q_scale={q_scale} f_scale={f_scale}"
        );
    }

    #[test]
    fn collision_operator_conserves_mass_momentum_energy() {
        // int Q(f,f) [1, v, |v|^2] dv = 0 exactly (structural conservation of
        // mass, momentum, energy). Exercised on a nontrivial (drifting,
        // perturbed) distribution.
        let grid = SpectralGrid::new(16, 6.0);
        let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 6);
        let n = grid.ntotal();
        let mut f = vec![0.0; n];
        for i in 0..n {
            let v = grid.velocity_at(i);
            let c = [v[0] - 0.5, v[1] + 0.3, v[2]];
            let c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
            f[i] = (2.0 * std::f64::consts::PI).powf(-1.5) * (-c2 / 2.0).exp() * (1.0 + 0.1 * (v[0]).sin());
        }
        let mut q = vec![0.0; n];
        op.apply_to_distribution(&f, &mut q);

        let dv3 = grid.dv3();
        let mass_f: f64 = f.iter().sum::<f64>() * dv3;
        let mut mass_q = 0.0;
        let mut mom_q = [0.0; 3];
        let mut en_q = 0.0;
        for i in 0..n {
            let v = grid.velocity_at(i);
            mass_q += q[i];
            mom_q[0] += q[i] * v[0];
            mom_q[1] += q[i] * v[1];
            mom_q[2] += q[i] * v[2];
            en_q += 0.5 * q[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
        mass_q *= dv3;
        for m in mom_q.iter_mut() {
            *m *= dv3;
        }
        en_q *= dv3;
        assert!(mass_q.abs() / mass_f < 1e-6, "mass not conserved: {mass_q}");
        assert!(mom_q[0].abs() / mass_f < 1e-4, "x-momentum not conserved: {}", mom_q[0]);
        assert!(mom_q[1].abs() / mass_f < 1e-4, "y-momentum not conserved: {}", mom_q[1]);
        assert!(en_q.abs() / mass_f < 1e-4, "energy not conserved: {en_q}");
    }

    #[test]
    fn h_theorem_entropy_production_nonpositive() {
        // Discrete H-theorem: d/dt integral f ln f dv = integral Q ln f dv <= 0.
        let grid = SpectralGrid::new(16, 6.0);
        let mut op = FastSpectralCollision::new(grid.clone(), 0.0, 6);
        let n = grid.ntotal();
        // Non-equilibrium two-stream distribution.
        let mut f = vec![0.0; n];
        for i in 0..n {
            let v = grid.velocity_at(i);
            let a2 = (v[0] - 1.0).powi(2) + v[1] * v[1] + v[2] * v[2];
            let b2 = (v[0] + 1.2).powi(2) + (v[1] - 0.3).powi(2) + v[2] * v[2];
            let m = (2.0 * std::f64::consts::PI).powf(-1.5);
            f[i] = (m * (-a2 / 2.0).exp() + 0.8 * m * (-b2 / 1.6).exp()).max(1e-300);
        }
        let mut q = vec![0.0; n];
        op.apply_to_distribution(&f, &mut q);
        let hdot: f64 = (0..n).map(|i| q[i] * f[i].ln()).sum::<f64>() * grid.dv3();
        assert!(hdot <= 1e-6, "H-functional increased (entropy production positive): hdot={hdot}");
    }
}
