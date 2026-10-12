//! Jacobi Eigenvalue Solver for Symmetric Arithmetic Matrices
//! Adapted directly from the robust RMX Navier-Stokes solver (rmx_ns_solver.rs)
//! Computes exact full spectrum, min/max eigenvalues, condition number kappa(G),
//! and algebraic rank rank(G).

use crate::gershgorin::build_zero_gram_matrix;
use std::f64;

/// Exact Jacobi eigenvalue diagonalization for symmetric real matrices.
/// Returns sorted eigenvalues in descending order.
pub fn jacobi_eigenvalues(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![matrix[0][0]];
    }

    let mut a = matrix.to_vec();

    for _iter in 0..max_iter {
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i][j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a.clone();

        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = c * a[i][p] - s * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = s * a[i][p] + c * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }

        new_a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        new_a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;
    }

    let mut evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    evals.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    evals
}

/// Full spectral analysis report
pub struct SpectralReport {
    pub dimension: usize,
    pub rank: usize,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub condition_number: f64,
    pub is_positive_definite: bool,
    pub frobenius_norm: f64,
}

/// Analyzes the Gram matrix spectrum via Jacobi diagonalization on the zero mesh
pub fn analyze_spectrum(zeros: &[f64], primes: &[u64]) -> SpectralReport {
    let m = zeros.len();
    let g = build_zero_gram_matrix(zeros, primes);
    let evals = jacobi_eigenvalues(&g, 200 * m * m, 1e-12);

    let lambda_max = evals.first().copied().unwrap_or(0.0);
    let lambda_min = evals.last().copied().unwrap_or(0.0);

    // Compute numerical rank with threshold eps = max(lambda_max * 1e-12, 1e-14)
    let eps_rank = (lambda_max * 1e-12).max(1e-14);
    let rank = evals.iter().filter(|&&e| e > eps_rank).count();

    let condition_number = if lambda_min > 1e-15 {
        lambda_max / lambda_min
    } else {
        f64::INFINITY
    };

    let mut frob_sq = 0.0;
    for i in 0..m {
        for j in 0..m {
            frob_sq += g[i][j] * g[i][j];
        }
    }

    SpectralReport {
        dimension: m,
        rank,
        lambda_min,
        lambda_max,
        condition_number,
        is_positive_definite: lambda_min > 0.0,
        frobenius_norm: frob_sq.sqrt(),
    }
}
