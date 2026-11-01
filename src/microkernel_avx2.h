// AVX2 6 x 16 micro-kernel (x86-64)
// 8 floats per register, 16 registers -> 12 accumulators (6 rows x 2) + 2 for B + 1 for A = 15/16
// per k: 2 B loads, 6 A broadcasts, 12 FMAs, 192 flops
#pragma once

#include <immintrin.h>

#include <cstddef>

namespace simd {

constexpr const char* kName = "avx2";
constexpr int MR = 6;
constexpr int NR = 16;

// default block sizes (multiples of MR, NR)
constexpr int kDefaultMC = 144;
constexpr int kDefaultKC = 256;
constexpr int kDefaultNC = 1024;

// C[0:6, 0:16] += a_panel * b_panel (C row stride ldc). b must be 32-byte aligned (_mm256_load_ps)
inline void micro_kernel(int kc, const float* a, const float* b, float* C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_load_ps(b);
        const __m256 b1 = _mm256_load_ps(b + 8);
        __m256 a_k;

        a_k = _mm256_broadcast_ss(a + 0);
        c00 = _mm256_fmadd_ps(a_k, b0, c00);
        c01 = _mm256_fmadd_ps(a_k, b1, c01);
        a_k = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(a_k, b0, c10);
        c11 = _mm256_fmadd_ps(a_k, b1, c11);
        a_k = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(a_k, b0, c20);
        c21 = _mm256_fmadd_ps(a_k, b1, c21);
        a_k = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(a_k, b0, c30);
        c31 = _mm256_fmadd_ps(a_k, b1, c31);
        a_k = _mm256_broadcast_ss(a + 4);
        c40 = _mm256_fmadd_ps(a_k, b0, c40);
        c41 = _mm256_fmadd_ps(a_k, b1, c41);
        a_k = _mm256_broadcast_ss(a + 5);
        c50 = _mm256_fmadd_ps(a_k, b0, c50);
        c51 = _mm256_fmadd_ps(a_k, b1, c51);

        a += MR;
        b += NR;
    }

    auto accumulate = [&](int row, __m256 lo, __m256 hi) {
        float* c = C + static_cast<std::size_t>(row) * ldc;
        _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), lo));
        _mm256_storeu_ps(c + 8, _mm256_add_ps(_mm256_loadu_ps(c + 8), hi));
    };
    accumulate(0, c00, c01);
    accumulate(1, c10, c11);
    accumulate(2, c20, c21);
    accumulate(3, c30, c31);
    accumulate(4, c40, c41);
    accumulate(5, c50, c51);
}

}  // namespace simd
