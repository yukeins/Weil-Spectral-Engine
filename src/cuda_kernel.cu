// CUDA Kernel for High-Performance Gram Gershgorin Disc Evaluation
// Optimized for NVIDIA GeForce RTX 2050 (Ampere Architecture, compute capability 8.6)
// Author: R. Santiago Z. G. (Independent Researcher)

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// CUDA Kernel evaluating Gershgorin Disc Radii for M zeros over N primes
// Each GPU thread calculates the off-diagonal sum for one zero index j
extern "C" __global__ void gershgorin_streaming_kernel(
    const double* __restrict__ zeros,        // Array of M zeros [M]
    const double* __restrict__ lnp,          // Array of N log-primes [N]
    const double* __restrict__ weights,      // Array of N prime weights [N]
    double* __restrict__ out_radii,          // Output array of disc radii [M]
    int m,                                   // Total zeros M
    int n,                                   // Total primes N
    int window_w,                            // Local window width W
    double tail_decay_cota                   // Analytical Montgomery-Vaughan tail bound
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;

    double gamma_j = zeros[j];

    // Interaction window: k in [max(0, j - W), min(M - 1, j + W)]
    int k_start = (j > window_w) ? (j - window_w) : 0;
    int k_end = (j + window_w + 1 < m) ? (j + window_w + 1) : m;

    double row_radius = 0.0;

    for (int k = k_start; k < k_end; ++k) {
        if (k != j) {
            double d_gamma = gamma_j - zeros[k];
            double g_jk = 0.0;

            #pragma unroll 4
            for (int p_idx = 0; p_idx < n; ++p_idx) {
                double phase = d_gamma * lnp[p_idx];
                g_jk += weights[p_idx] * cos(phase);
            }

            row_radius += fabs(g_jk);
        }
    }

    // Add Montgomery-Vaughan tail bound for frequencies outside window
    out_radii[j] = row_radius + tail_decay_cota;
}

// C-compatible wrapper to be invoked from Rust via FFI
extern "C" int launch_gpu_gershgorin(
    const double* h_zeros,
    const double* h_lnp,
    const double* h_weights,
    double* h_radii,
    int m,
    int n,
    int window_w,
    double tail_decay_cota
) {
    double *d_zeros = NULL, *d_lnp = NULL, *d_weights = NULL, *d_radii = NULL;

    cudaError_t err;

    // Allocate GPU Device Memory
    err = cudaMalloc((void**)&d_zeros, m * sizeof(double));
    if (err != cudaSuccess) return 1;

    err = cudaMalloc((void**)&d_lnp, n * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_zeros); return 2; }

    err = cudaMalloc((void**)&d_weights, n * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_zeros); cudaFree(d_lnp); return 3; }

    err = cudaMalloc((void**)&d_radii, m * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_zeros); cudaFree(d_lnp); cudaFree(d_weights); return 4; }

    // Copy data from Host CPU to GPU Device VRAM
    cudaMemcpy(d_zeros, h_zeros, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lnp, h_lnp, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch CUDA Grid: 256 threads per block
    int threads_per_block = 256;
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;

    gershgorin_streaming_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_zeros, d_lnp, d_weights, d_radii, m, n, window_w, tail_decay_cota
    );

    cudaDeviceSynchronize();

    // Copy computed radii back to Host CPU
    cudaMemcpy(h_radii, d_radii, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU VRAM
    cudaFree(d_zeros);
    cudaFree(d_lnp);
    cudaFree(d_weights);
    cudaFree(d_radii);

    return 0;
}
