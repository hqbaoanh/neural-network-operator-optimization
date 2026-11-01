// stage 3: cache blocking
// stage 2 reads all of B (up to 4 MB) for every row of C. here B is cut into
// KC x NC blocks (256 x 256 floats = 256 KB) and each block is reused for MC rows
// before moving on. same i, k, j order inside a block.
// note: slower on the M3 Pro (B already fits in its 16 MB L2), helps on the Xeon (1 MB L2).
#include <algorithm>
#include <cstring>

#include "gemm.h"
#include "gemm_internal.h"

namespace {
constexpr int MC = 64;   // rows of A / C per block
constexpr int KC = 256;  // depth per block
constexpr int NC = 256;  // columns of B / C per block
}  // namespace

// rows [row_begin, row_end) of C; stage 4 calls this per thread
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
                // C[ic:ic+mc, jc:jc+nc] += A[ic:ic+mc, pc:pc+kc] * B[pc:pc+kc, jc:jc+nc]
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
