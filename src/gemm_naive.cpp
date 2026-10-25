// Stage 1: naive GEMM.
//
// For every output element C[i][j], walk down row i of A and column j of B.
// The access to B is the problem: B[k*N + j] jumps N floats (2-8 KB here)
// every iteration of k, so almost every load touches a new cache line, and
// the compiler cannot vectorize the reduction over k without reordering
// floating-point additions (which it will not do without -ffast-math).
#include "gemm.h"

void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
