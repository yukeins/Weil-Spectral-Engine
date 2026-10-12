//! Gaussian Unitary Ensemble (GUE) Spectral Statistics (Section 9.3 of Monograph)
//! Verifies Montgomery-Odlyzko pair correlation conjecture and level repulsion
//! R_2(r) = 1 - (\sin(\pi r) / (\pi r))^2 for Riemann zeros.

use std::f64::consts::PI;

pub struct GueReport {
    pub num_zeros: usize,
    pub mean_normalized_spacing: f64,
    pub spacing_variance: f64,
    pub level_repulsion_ratio: f64, // P(s < 0.2) showing absence of clustering (Wigner surmise vs Poisson)
    pub pair_correlation_rmse: f64,
}

pub fn analyze_gue_spacings(zeros: &[f64]) -> GueReport {
    let m = zeros.len();
    if m < 3 {
        return GueReport {
            num_zeros: m,
            mean_normalized_spacing: 1.0,
            spacing_variance: 0.0,
            level_repulsion_ratio: 0.0,
            pair_correlation_rmse: 0.0,
        };
    }

    // Compute normalized unfolded spacings: s_n = (gamma_{n+1} - gamma_n) * ln(gamma_n / 2pi) / (2pi)
    let mut spacings = Vec::with_capacity(m - 1);
    let mut small_spacing_count = 0;

    for i in 0..(m - 1) {
        let gamma = zeros[i];
        let diff = zeros[i + 1] - gamma;
        let density = (gamma / (2.0 * PI)).max(1.0).ln() / (2.0 * PI);
        let s = diff * density;
        spacings.push(s);
        if s < 0.2 {
            small_spacing_count += 1;
        }
    }

    let mean_s: f64 = spacings.iter().sum::<f64>() / (spacings.len() as f64);
    let var_s: f64 = spacings.iter().map(|s| (s - mean_s).powi(2)).sum::<f64>() / (spacings.len() as f64);
    let repulsion = (small_spacing_count as f64) / (spacings.len() as f64);

    // Compute empirical pair correlation in [0, 2.0] in 20 bins
    let num_bins = 20;
    let max_r = 2.0;
    let bin_width = max_r / (num_bins as f64);
    let mut hist = vec![0.0; num_bins];
    let mut total_pairs = 0.0;

    for i in 0..spacings.len().min(200) {
        for j in (i + 1)..spacings.len().min(200) {
            let r: f64 = spacings[i..j].iter().sum();
            if r < max_r {
                let bin = (r / bin_width).floor() as usize;
                if bin < num_bins {
                    hist[bin] += 1.0;
                    total_pairs += 1.0;
                }
            }
        }
    }

    let mut rmse_sq = 0.0;
    for b in 0..num_bins {
        let r_center = (b as f64 + 0.5) * bin_width;
        let theoretical = if r_center > 1e-6 {
            1.0 - ((PI * r_center).sin() / (PI * r_center)).powi(2)
        } else {
            0.0
        };
        let empirical = if total_pairs > 0.0 {
            hist[b] / (total_pairs * bin_width)
        } else {
            theoretical
        };
        rmse_sq += (empirical - theoretical).powi(2);
    }

    let pair_correlation_rmse = (rmse_sq / (num_bins as f64)).sqrt();

    GueReport {
        num_zeros: m,
        mean_normalized_spacing: mean_s,
        spacing_variance: var_s,
        level_repulsion_ratio: repulsion,
        pair_correlation_rmse,
    }
}
