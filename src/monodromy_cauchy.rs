//! Homotopic Monodromy and Cauchy Winding Index Integrator (\nu \in Z)
//! Implements Section 8.3 of the Monograph and certified theorem `inertial_winding_zero` from
//! `RiemannHypothesis.Invariants.MonodromyWinding`:
//! Evaluates the Cauchy contour integral around the resolvent determinant in the upper half-plane:
//! \nu = \frac{1}{2\pi i} \oint_{\Gamma} \frac{d}{dz} \ln \det(z \mathbf{I} - \mathbf{H}_\delta) \, dz
//! For a contour \Gamma lying in the upper half-plane C^+ (e.g. centered at gamma + i * y_0 with radius R < y_0):
//! If the spectrum is purely real (\delta = 0), the pole is at (gamma, 0), outside \Gamma, so \nu = 0 (Sector A).
//! If an off-axis pole existed with imaginary part \delta = y_0 inside \Gamma, then \nu = 1 or 2 (Sector B).

use std::f64::consts::PI;
use num_complex::Complex64;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MonodromyReport {
    pub delta: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub contour_center_im: f64,
    pub contour_radius: f64,
    pub phase_accumulated_radians: f64,
    pub winding_index_nu: i64,
    pub is_sector_a_inertial: bool,
}

/// Evaluates det(z I - H_\delta) for a single zero mode displaced by \delta:
/// Mode eigenvalue: \lambda = \gamma + i \delta
/// det(z - \lambda) = z - (\gamma + i \delta)
pub fn eval_mode_determinant(z: Complex64, gamma: f64, delta: f64) -> Complex64 {
    let pole = Complex64::new(gamma, delta);
    z - pole
}

/// Computes the Cauchy winding number \nu by numerical line integration along a circle
/// in the upper half plane C^+ centered at (gamma, y_0):
/// \Gamma(\theta) = (gamma + i y_0) + R e^{i \theta},  \theta \in [0, 2\pi].
/// - If \delta = 0 (real spectrum), the pole at (gamma, 0) is outside \Gamma (distance y_0 > R) => \nu = 0.
/// - If \delta = y_0 (off-axis pole), the pole is at the center of \Gamma => \nu = 1 (Monodromy Defect).
pub fn integrate_cauchy_winding(gamma: f64, delta: f64, y0: f64, radius: f64, steps: usize) -> MonodromyReport {
    let sigma = 0.5 + delta;
    let n_steps = steps.max(200);
    let mut total_d_arg = 0.0;

    let center = Complex64::new(gamma, y0);
    let mut prev_val = eval_mode_determinant(
        center + Complex64::from_polar(radius, 0.0),
        gamma,
        delta
    );

    for s in 1..=n_steps {
        let theta = 2.0 * PI * (s as f64) / (n_steps as f64);
        let z = center + Complex64::from_polar(radius, theta);
        let curr_val = eval_mode_determinant(z, gamma, delta);

        // Incremental argument difference: arg(curr / prev)
        let ratio = curr_val / prev_val;
        total_d_arg += ratio.arg();
        prev_val = curr_val;
    }

    // Winding index: total_d_arg / (2\pi) rounded to nearest integer
    let winding_raw = total_d_arg / (2.0 * PI);
    let winding_nu = winding_raw.round() as i64;
    let is_sector_a = winding_nu == 0;

    MonodromyReport {
        delta,
        sigma,
        gamma,
        contour_center_im: y0,
        contour_radius: radius,
        phase_accumulated_radians: total_d_arg,
        winding_index_nu: winding_nu,
        is_sector_a_inertial: is_sector_a,
    }
}

/// Sweeps delta across [-0.4, 0.4] testing whether the pole enters the test domain C^+
pub fn sweep_monodromy_sectors(gamma: f64, y0: f64, radius: f64) -> Vec<MonodromyReport> {
    let test_deltas = vec![-0.30, 0.0, 0.10, y0, 0.40];
    test_deltas.into_iter().map(|d| {
        integrate_cauchy_winding(gamma, d, y0, radius, 300)
    }).collect()
}
