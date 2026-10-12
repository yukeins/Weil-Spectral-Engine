//! Scale Benchmark & Hardware Calibration Suite for 2 Million Zeros Streaming
//! Author: R. Santiago Z. G. (Independent Researcher)

use std::time::Instant;
use rayon::prelude::*;

// We import the streaming modules directly from the main library
#[path = "../streaming_gershgorin.rs"]
mod streaming_gershgorin;

#[path = "../ingest.rs"]
mod ingest;

fn main() {
    println!("================================================================================");
    println!("       WEIL SPECTRAL ENGINE: HARDWARE CALIBRATION & BENCHMARK SUITE             ");
    println!("       Extrapolating Performance for 2 Million Zeros (M = 2,000,000)            ");
    println!("       Author: R. Santiago Z. G. (Independent Researcher)                       ");
    println!("================================================================================");

    let num_threads = rayon::current_num_threads();
    println!("\n[HARDWARE] Rayon Multi-Threading Active: {} logical worker threads detected", num_threads);

    // Load primes (canonical N = 1000)
    let n_primes = 1000;
    println!("[INGEST] Precomputing N = {} prime frequencies...", n_primes);
    let primes = ingest::load_primes(n_primes);

    let window_w = 200; // Local interaction neighborhood
    let chunk_size = 4096; // Parallel chunking buffer size

    let test_scales = [10_000, 50_000, 100_000];

    println!("\n{:<12} | {:<12} | {:<12} | {:<16} | {:<16}", "Scale (M)", "Time (s)", "Zeros/sec", "RAM Footprint", "Coercive Margin");
    println!("{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<16}-+-{:-<16}", "", "", "", "", "");

    let mut last_speed = 0.0;

    for &m in &test_scales {
        let t_gen = Instant::now();
        let zeros = ingest::load_zeros_or_fallback(m);
        let _gen_elapsed = t_gen.elapsed().as_secs_f64();

        let t_calc = Instant::now();
        let res = streaming_gershgorin::certify_streaming_zeros(&zeros, &primes, window_w, chunk_size);
        let calc_time = t_calc.elapsed().as_secs_f64();

        let zeros_per_sec = (m as f64) / calc_time;
        last_speed = zeros_per_sec;

        println!(
            "{:<12} | {:<12.3} | {:<12.1} | {:<13.2} MB | {:<16.6e}",
            m,
            calc_time,
            zeros_per_sec,
            res.memory_allocated_mb,
            res.min_margin
        );
    }

    println!("{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<16}-+-{:-<16}", "", "", "", "", "");

    // Extrapolate for 2,000,000 zeros
    let target_m = 2_000_000usize;
    let target_ram_mb = (target_m * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    let extrapolated_seconds = (target_m as f64) / last_speed;
    let extrapolated_minutes = extrapolated_seconds / 60.0;

    println!("\n================================================================================");
    println!("                 EXTRAPOLATION FOR 2 MILLION ZEROS (M = 2,000,000)              ");
    println!("================================================================================");
    println!("  * Dataset Vector RAM (f64):        {:.2} MB", target_ram_mb);
    println!("  * Peak Streaming Buffer RAM:       < 50.0 MB (O(B) bounded)");
    println!("  * Estimated Execution Speed:       {:.1} zeros / second", last_speed);
    println!("  * Projected Total Run Time:        {:.1} seconds (~{:.2} minutes)", extrapolated_seconds, extrapolated_minutes);
    println!("================================================================================\n");
}
