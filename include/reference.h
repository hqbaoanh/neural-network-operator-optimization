// Ground truth for correctness checks.
//
// The reference accumulates in double precision, so its own rounding error
// is far below float32's. Comparing a float32 kernel against it measures the
// kernel's error, not the reference's.
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

// max |C - ref| / max |ref|. A relative measure so the threshold doesn't
// depend on the size of the numbers in the matrices.
inline double max_relative_error(const std::vector<float>& C, const std::vector<double>& ref) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(C[i]) - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

// Expected float32 error grows roughly like sqrt(K) * machine epsilon, and
// the order kernels add things in differs. 1e-5 * sqrt(K) leaves a safe margin
// over that (float32 epsilon is ~1.2e-7) while still catching real bugs, which
// show up as errors of order 1.
inline double error_tolerance(int K) { return 1e-5 * std::sqrt(static_cast<double>(K)); }

inline std::vector<float> random_matrix(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(rows) * cols);
    for (float& x : m) x = dist(rng);
    return m;
}
