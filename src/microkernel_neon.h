// NEON micro-kernel for 64-bit ARM (Apple Silicon, Graviton, ...): an 8 x 12 tile of C.
//
// A NEON register holds 4 floats (half of AVX2), but there are 32 of them
// (twice AVX2). So the tile is taller and narrower than the AVX2 one:
// 24 accumulators (8 rows x 3 vectors of 4 columns), 3 registers for the
// current row of B and 2 for the 8 values of A: 29 of 32.
//
// Instead of broadcasting A from memory like the AVX2 kernel does, it loads
// all 8 values of A into two registers and uses the "by element" form of FMA
// (vfmaq_laneq_f32), which multiplies by one lane of a register directly.
// Per step of k: 5 vector loads, 24 FMAs = 192 flops.
#pragma once

#include <arm_neon.h>

#include <cstddef>

namespace simd {

constexpr const char* kName = "neon";
constexpr int MR = 8;
constexpr int NR = 12;

// Default cache block sizes (multiples of MR and NR). Apple M-series cores
// have a 128 KB L1 data cache and a large shared L2, so KC can be a bit larger.
constexpr int kDefaultMC = 128;
constexpr int kDefaultKC = 384;
constexpr int kDefaultNC = 1008;

namespace detail {
// One row of the tile: (c0 | c1 | c2) += (b0 | b1 | b2) * a[Lane].
// The lane has to be a compile-time constant, hence the template parameter.
// The accumulators are separate named variables passed by reference (not an
// array) because GCC keeps those in registers but spills arrays to the stack
// on every iteration.
template <int Lane>
inline void fma_row(float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t b0,
                    float32x4_t b1, float32x4_t b2, float32x4_t a) {
    c0 = vfmaq_laneq_f32(c0, b0, a, Lane);
    c1 = vfmaq_laneq_f32(c1, b1, a, Lane);
    c2 = vfmaq_laneq_f32(c2, b2, a, Lane);
}

inline void add_row(float* row, float32x4_t c0, float32x4_t c1, float32x4_t c2) {
    vst1q_f32(row, vaddq_f32(vld1q_f32(row), c0));
    vst1q_f32(row + 4, vaddq_f32(vld1q_f32(row + 4), c1));
    vst1q_f32(row + 8, vaddq_f32(vld1q_f32(row + 8), c2));
}
}  // namespace detail

// C[0:8, 0:12] += a_panel * b_panel, where C has row stride ldc.
// a: kc steps of MR floats. b: kc steps of NR floats.
inline void micro_kernel(int kc, const float* a, const float* b, float* C, int ldc) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t c00 = zero, c01 = zero, c02 = zero;
    float32x4_t c10 = zero, c11 = zero, c12 = zero;
    float32x4_t c20 = zero, c21 = zero, c22 = zero;
    float32x4_t c30 = zero, c31 = zero, c32 = zero;
    float32x4_t c40 = zero, c41 = zero, c42 = zero;
    float32x4_t c50 = zero, c51 = zero, c52 = zero;
    float32x4_t c60 = zero, c61 = zero, c62 = zero;
    float32x4_t c70 = zero, c71 = zero, c72 = zero;

    for (int k = 0; k < kc; ++k) {
        const float32x4_t b0 = vld1q_f32(b);
        const float32x4_t b1 = vld1q_f32(b + 4);
        const float32x4_t b2 = vld1q_f32(b + 8);
        const float32x4_t a_lo = vld1q_f32(a);      // A rows 0-3 at this k
        const float32x4_t a_hi = vld1q_f32(a + 4);  // A rows 4-7

        detail::fma_row<0>(c00, c01, c02, b0, b1, b2, a_lo);
        detail::fma_row<1>(c10, c11, c12, b0, b1, b2, a_lo);
        detail::fma_row<2>(c20, c21, c22, b0, b1, b2, a_lo);
        detail::fma_row<3>(c30, c31, c32, b0, b1, b2, a_lo);
        detail::fma_row<0>(c40, c41, c42, b0, b1, b2, a_hi);
        detail::fma_row<1>(c50, c51, c52, b0, b1, b2, a_hi);
        detail::fma_row<2>(c60, c61, c62, b0, b1, b2, a_hi);
        detail::fma_row<3>(c70, c71, c72, b0, b1, b2, a_hi);

        a += MR;
        b += NR;
    }

    const auto row = [&](int r) { return C + static_cast<std::size_t>(r) * ldc; };
    detail::add_row(row(0), c00, c01, c02);
    detail::add_row(row(1), c10, c11, c12);
    detail::add_row(row(2), c20, c21, c22);
    detail::add_row(row(3), c30, c31, c32);
    detail::add_row(row(4), c40, c41, c42);
    detail::add_row(row(5), c50, c51, c52);
    detail::add_row(row(6), c60, c61, c62);
    detail::add_row(row(7), c70, c71, c72);
}

}  // namespace simd
