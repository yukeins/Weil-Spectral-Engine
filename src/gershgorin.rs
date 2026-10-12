//! Gershgorin Disc Certified Bounds for Arithmetic Zero-Mesh Gram Matrices
//! Implements Section 7.1, Equation (497) and Theorem 7.1 of the Monograph:
//! G_{jk} = \sum_{p \le N} \frac{1}{\sqrt{p}\ln p} \cos((\gamma_j - \gamma_k)\ln p)
//! Strictly verifies: G_{jj} \gg \sum_{k \neq j} |G_{jk}| and \lambda_{\min}(G) > 0.

use rayon::prelude::*;

/// Computes the discrete arithmetic Gram matrix G_{jk} of dimension M x M over N primes.
pub fn build_zero_gram_matrix(zeros: &[f64], primes: &[u64]) -> Vec<Vec<f64>> {
    let m = zeros.len();
    let mut g = vec![vec![0.0; m]; m];

    // Precompute prime weights w_p = 1 / (sqrt(p) * ln(p))
    let weights: Vec<(f64, f64)> = primes.iter().map(|&p| {
        let pf = p as f64;
        let lnp = pf.ln();
        let w = 1.0 / (pf.sqrt() * lnp);
        (lnp, w)
    }).collect();

    // Diagonal entry: G_{jj} = \sum_{p \le N} w_p
    let diag_val: f64 = weights.iter().map(|(_, w)| *w).sum();

    for j in 0..m {
        g[j][j] = diag_val;
        for k in (j + 1)..m {
            let d_gamma = zeros[j] - zeros[k];
            let mut off_diag = 0.0;
            for (lnp, w) in &weights {
                off_diag += w * (d_gamma * lnp).cos();
            }
            g[j][k] = off_diag;
            g[k][j] = off_diag;
        }
    }

    g
}

/// Certifies strict diagonal dominance and Gershgorin lower bound:
/// \lambda_{\min}(G) \ge \min_j (G_{jj} - \sum_{k \neq j} |G_{jk}|) > 0
pub fn certify_zero_gram_matrix(zeros: &[f64], primes: &[u64]) -> (f64, f64, bool) {
    let m = zeros.len();
    if m == 0 {
        return (0.0, 0.0, false);
    }

    let g = build_zero_gram_matrix(zeros, primes);
    let diag_val = g[0][0];

    // Compute Gershgorin disc lower bounds in parallel across zero rows
    let min_disc = (0..m).into_par_iter().map(|j| {
        let mut row_off_sum = 0.0;
        for k in 0..m {
            if j != k {
                row_off_sum += g[j][k].abs();
            }
        }
        diag_val - row_off_sum
    }).reduce(|| f64::INFINITY, f64::min);

    let is_strictly_dominant = min_disc > 0.0;
    (diag_val, min_disc, is_strictly_dominant)
}
