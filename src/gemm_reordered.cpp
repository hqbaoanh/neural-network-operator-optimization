// Stage 2: loop reordering (i, k, j).
//
// Same multiplications and additions as stage 1, different order. Fix a row i
// and a k, take the scalar a = A[i][k], and add a * (row k of B) into row i of C.
// Now the innermost loop reads B[k][0..N) and C[i][0..N) contiguously:
// every cache line is fully used, and the loop is a plain "y += a*x" that
// the compiler turns into 8-wide AVX2 FMA instructions on its own.
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
