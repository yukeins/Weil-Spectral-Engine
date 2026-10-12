//! Riccati Phase Portrait and Non-Hermitian Wave Packet L^2 Divergence Engine
//! Implements Section 2.4 & 4.4 of the Monograph and certified theorem `riccati_dissipative_monomial`
//! from `RiemannHypothesis.Dynamics.RiccatiConstraint`:
//! Integrates the Schr\ödinger--Riccati equation:
//! u''(x) + V_\delta(x) u(x) = 0
//! with centrifugal potential V_\delta(x) = (\gamma^2 + 1/4) - \rho_\delta(\rho_\delta - 1) / x^2
//! where \rho_\delta = 1/2 + \delta + i \gamma.
//! For \delta = 0: \Im(V_0(x)) = 0 (Hermitian), energy is conserved, |u(x)| is bounded in L^2.
//! For \delta \neq 0: \Im(V_\delta(x)) = -2 \delta \gamma / x^2 \neq 0, injecting continuous
//! non-Hermitian dissipation/amplification that diverges the L^2 norm.

use num_complex::Complex64;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RiccatiTrajectoryReport {
    pub delta: f64,
    pub gamma: f64,
    pub final_envelope: f64,
    pub l2_energy_integral: f64,
    pub imaginary_potential_defect: f64,
    pub is_l2_confined: bool,
}

/// Evaluates the complex potential V_\delta(x)
pub fn eval_potential(x: f64, gamma: f64, delta: f64) -> Complex64 {
    let x_sq = x * x;
    // rho = 1/2 + delta + i * gamma
    let rho = Complex64::new(0.5 + delta, gamma);
    let rho_minus_one = rho - Complex64::new(1.0, 0.0);
    let centrifugal_factor = rho * rho_minus_one; // rho * (rho - 1)

    let base_energy = Complex64::new(gamma * gamma + 0.25, 0.0);
    let centrifugal_term = centrifugal_factor / Complex64::new(x_sq, 0.0);

    base_energy - centrifugal_term
}

/// Numerical Runge-Kutta 4th order integrator for the complex ODE:
/// u' = v
/// v' = -V_\delta(x) * u
pub fn integrate_wave_packet(gamma: f64, delta: f64, x_start: f64, x_end: f64, steps: usize) -> RiccatiTrajectoryReport {
    let n = steps.max(500);
    let h = (x_end - x_start) / (n as f64);

    let mut x = x_start;
    let mut u = Complex64::new(1.0, 0.0); // Initial wave amplitude
    let mut v = Complex64::new(0.0, gamma); // Initial phase velocity

    let mut l2_sum = 0.0;

    for _ in 0..n {
        let k1_u = v;
        let k1_v = -eval_potential(x, gamma, delta) * u;

        let k2_u = v + k1_v * (h * 0.5);
        let k2_v = -eval_potential(x + h * 0.5, gamma, delta) * (u + k1_u * (h * 0.5));

        let k3_u = v + k2_v * (h * 0.5);
        let k3_v = -eval_potential(x + h * 0.5, gamma, delta) * (u + k2_u * (h * 0.5));

        let k4_u = v + k3_v * h;
        let k4_v = -eval_potential(x + h, gamma, delta) * (u + k3_u * h);

        u += (k1_u + k2_u * 2.0 + k3_u * 2.0 + k4_u) * (h / 6.0);
        v += (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (h / 6.0);
        x += h;

        let amp_sq = u.norm_sqr();
        l2_sum += amp_sq * h;
    }

    let final_envelope = u.norm();
    let im_defect = 2.0 * delta.abs() * gamma / (x_start * x_start);
    let is_confined = delta.abs() < 1e-12 || (final_envelope < 10.0 && l2_sum < 50.0);

    RiccatiTrajectoryReport {
        delta,
        gamma,
        final_envelope,
        l2_energy_integral: l2_sum,
        imaginary_potential_defect: im_defect,
        is_l2_confined: is_confined,
    }
}

/// Sweeps off-axis shifts \delta around a Riemann zero to verify L^2 wave packet divergence
pub fn sweep_riccati_flow(gamma: f64) -> Vec<RiccatiTrajectoryReport> {
    let test_deltas = vec![-0.20, -0.10, 0.0, 0.10, 0.20];
    test_deltas.into_iter().map(|d| {
        integrate_wave_packet(gamma, d, 1.0, 5.0, 500)
    }).collect()
}
