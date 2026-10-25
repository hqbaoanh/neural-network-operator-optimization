// Stage 3: cache blocking (tiling).
//
// In stage 2, for each row i we stream through ALL of B (K x N floats, up to
// 4 MB for the 2048 x 512 shape). That is bigger than L1 and L2, so B gets
// re-fetched from far away for every single row of C.
//
// Blocking fixes this: cut the problem into a block of B that is KC x NC
// (here 256 x 256 floats = 256 KB, sized for L2) and reuse that block for
// MC rows of A before moving on. Inside the block the loop order is the same
// cache-friendly (i, k, j) as stage 2.
//
// The block sizes are compile-time constants; see the README for how to tune them.
#include <algorithm>
#include <cstring>

#include "gemm.h"
#include "gemm_internal.h"

namespace {
constexpr int MC = 64;   // rows of A / C per block
constexpr int KC = 256;  // depth per block
constexpr int NC = 256;  // columns of B / C per block
}  // namespace

// Computes rows [row_begin, row_end) of C. Shared with the threaded version.
void gemm_blocked_rows(int row_begin, int row_end, int N, int K,
                       const float* A, const float* B, float* C) {
    for (int i = row_begin; i < row_end; ++i) {
        std::memset(C + static_cast<size_t>(i) * N, 0, sizeof(float) * N);
    }
    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            for (int ic = row_begin; ic < row_end; ic += MC) {
                const int mc = std::min(MC, row_end - ic);
                // One block: C[ic:ic+mc, jc:jc+nc] += A[ic:ic+mc, pc:pc+kc] * B[pc:pc+kc, jc:jc+nc]
                for (int i = ic; i < ic + mc; ++i) {
                    float* c_row = C + static_cast<size_t>(i) * N + jc;
                    for (int k = pc; k < pc + kc; ++k) {
                        const float a = A[static_cast<size_t>(i) * K + k];
                        const float* b_row = B + static_cast<size_t>(k) * N + jc;
                        for (int j = 0; j < nc; ++j) {
                            c_row[j] += a * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    gemm_blocked_rows(0, M, N, K, A, B, C);
}
