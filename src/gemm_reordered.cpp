// stage 2: i, k, j
// row i of C += A[i][k] * row k of B. inner loop now reads B and C in a straight line,
// and it's just y += a*x so the compiler vectorizes it by itself (NEON / AVX2).
#include <cstring>

#include "gemm.h"

void gemm_reordered(int M, int N, int K, const float* A, const float* B, float* C) {
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
    for (int i = 0; i < M; ++i) {
        float* c_row = C + static_cast<size_t>(i) * N;
        for (int k = 0; k < K; ++k) {
            const float a = A[i * K + k];
            const float* b_row = B + static_cast<size_t>(k) * N;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}
