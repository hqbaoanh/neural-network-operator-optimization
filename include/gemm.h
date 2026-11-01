// C = A * B. A is M x K, B is K x N, C is M x N, float32, row-major, no padding.
// Every kernel overwrites C (no need to zero it first).
// New kernel: declare it here and add it to kernels.cpp so bench and tests run it.
#pragma once

#include <string>
#include <vector>

using GemmFn = void (*)(int M, int N, int K, const float* A, const float* B, float* C);

// stage 1: i, j, k triple loop (baseline)
void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C);

// stage 2: i, k, j order
void gemm_reordered(int M, int N, int K, const float* A, const float* B, float* C);

// stage 3: cache blocking
void gemm_blocked(int M, int N, int K, const float* A, const float* B, float* C);

// stage 4: stage 3 split across threads by rows of C
void gemm_blocked_mt(int M, int N, int K, const float* A, const float* B, float* C);

// stage 5: packing + SIMD micro-kernel (6x16 AVX2 on x86, 8x12 NEON on ARM, picked at compile time)
void gemm_simd(int M, int N, int K, const float* A, const float* B, float* C);
void gemm_simd_mt(int M, int N, int K, const float* A, const float* B, float* C);
const char* simd_isa_name();  // "avx2" or "neon"

struct Kernel {
    std::string name;
    GemmFn fn;
    bool threaded;
};

// all kernels in stage order (kernels.cpp)
const std::vector<Kernel>& all_kernels();

// GEMM_THREADS if set, else hardware_concurrency()
int gemm_num_threads();
