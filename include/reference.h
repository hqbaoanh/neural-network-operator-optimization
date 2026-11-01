// float64 reference for checking kernels. Its own rounding error is tiny
// compared to float32, so the difference is basically all the kernel's error.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

inline void gemm_reference_f64(int M, int N, int K, const float* A, const float* B,
                               std::vector<double>& C) {
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            for (int j = 0; j < N; ++j) {
                C[static_cast<size_t>(i) * N + j] += a * B[static_cast<size_t>(k) * N + j];
            }
        }
    }
}

// max |C - ref| / max |ref| (relative, so it doesn't depend on the scale of the values)
inline double max_relative_error(const std::vector<float>& C, const std::vector<double>& ref) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(C[i]) - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

// float32 error grows roughly like sqrt(K) * eps (eps ~1.2e-7), and kernels add in
// different orders. 1e-5 * sqrt(K) is well above that; real bugs give errors ~1.
inline double error_tolerance(int K) { return 1e-5 * std::sqrt(static_cast<double>(K)); }

inline std::vector<float> random_matrix(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(rows) * cols);
    for (float& x : m) x = dist(rng);
    return m;
}
