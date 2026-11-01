// stage 1: naive i, j, k
// B[k*N + j] jumps N floats (2-8 KB here) every k step, so B is read down a column.
// Also can't vectorize: sum over k is one long chain of adds and the compiler
// won't reorder float adds without -ffast-math.
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
