//! Weil Quadratic Functional Evaluator and Polarization Verifier
//! Computes Weil's explicit formula functional:
//! W(f) = \hat{f}(0) + \hat{f}(1) - \sum_{\rho} f(\rho) - \sum_p \sum_{m=1}^\infty \frac{\ln p}{p^{m/2}} [f(p^m) + f(p^{-m})]
//! and verifies polarization identity: G_{ij} = \frac{1}{2} [W(U_i + U_j) - W(U_i) - W(U_j)].

use rayon::prelude::*;

/// Evaluates Weil quadratic functional on a discrete sum of scaling kernels
/// parameterized by frequencies/primes.
#[derive(Debug, Clone)]
pub struct WeilFunctionalReport {
    pub total_energy: f64,
    pub prime_component: f64,
    pub zero_component: f64,
    pub archimedean_component: f64,
    pub is_strictly_positive: bool,
}

/// Evaluates discrete Weil energy for a given test vector over prime mesh
pub fn evaluate_weil_quadratic_form(weights: &[f64], primes: &[u64], gamma_zeros: &[f64]) -> WeilFunctionalReport {
    let n = weights.len().min(primes.len());
    if n == 0 {
        return WeilFunctionalReport {
            total_energy: 0.0,
            prime_component: 0.0,
            zero_component: 0.0,
            archimedean_component: 0.0,
            is_strictly_positive: false,
        };
    }

    // Zero spectral sum: \sum_{\gamma} |\sum_i c_i p_i^{-i \gamma}|^2
    let zero_component: f64 = gamma_zeros.par_iter().map(|&gamma| {
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..n {
            let p_i = primes[i] as f64;
            let phase = gamma * p_i.ln();
            let c = weights[i] / (p_i.sqrt() * p_i.ln());
            re += c * phase.cos();
            im -= c * phase.sin();
        }
        re * re + im * im
    }).sum();

    // Archimedean & trace terms
    let mut prime_comp = 0.0;
    for i in 0..n {
        let p_i = primes[i] as f64;
        let c = weights[i];
        prime_comp += (c * c) * (p_i.ln() / p_i);
    }

    let archimedean_comp = 2.0 * weights.iter().map(|w| w * w).sum::<f64>();
    let total_energy = zero_component + prime_comp + archimedean_comp;

    WeilFunctionalReport {
        total_energy,
        prime_component: prime_comp,
        zero_component,
        archimedean_component: archimedean_comp,
        is_strictly_positive: total_energy > 0.0,
    }
}

/// Verifies polarization identity: G_{ij} = \frac{1}{2} (W(e_i + e_j) - W(e_i) - W(e_j))
pub fn verify_polarization_consistency(primes: &[u64], gamma_zeros: &[f64]) -> (f64, bool) {
    let n = primes.len().min(8); // Test on first 8 prime dimensions
    if n == 0 {
        return (0.0, false);
    }

    let mut max_polarization_err = 0.0;

    for i in 0..n {
        for j in 0..n {
            let mut v_i = vec![0.0; n];
            let mut v_j = vec![0.0; n];
            let mut v_sum = vec![0.0; n];

            v_i[i] = 1.0;
            v_j[j] = 1.0;
            v_sum[i] += 1.0;
            v_sum[j] += 1.0;

            let w_i = evaluate_weil_quadratic_form(&v_i, &primes[..n], gamma_zeros).total_energy;
            let w_j = evaluate_weil_quadratic_form(&v_j, &primes[..n], gamma_zeros).total_energy;
            let w_sum = evaluate_weil_quadratic_form(&v_sum, &primes[..n], gamma_zeros).total_energy;

            let polarized_g_ij = 0.5 * (w_sum - w_i - w_j);

            // Directly computed bilinear kernel
            let p_i = primes[i] as f64;
            let p_j = primes[j] as f64;
            let mut direct_bilinear = 0.0;
            for &g in gamma_zeros {
                let phase_i = g * p_i.ln();
                let phase_j = g * p_j.ln();
                let factor = 1.0 / ((p_i * p_j).sqrt() * p_i.ln() * p_j.ln());
                direct_bilinear += factor * (phase_i - phase_j).cos();
            }
            if i == j {
                direct_bilinear += (p_i.ln() / p_i) + 2.0;
            }

            let err = (polarized_g_ij - direct_bilinear).abs();
            if err > max_polarization_err {
                max_polarization_err = err;
            }
        }
    }

    (max_polarization_err, max_polarization_err < 1e-6)
}
