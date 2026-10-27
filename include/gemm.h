// gemm.h -- the one interface every kernel in this project implements.
//
//   C = A * B
//
//   A is M x K, B is K x N, C is M x N, all float32, all row-major and
//   densely packed (so the leading dimensions are K, N and N).
//   Every kernel OVERWRITES C; the caller never has to zero it first.
//
// Each optimization stage adds one function here and one entry to the
// registry in kernels.cpp, so the benchmark and tests pick it up
// automatically.
#pragma once

#include <string>
#include <vector>

using GemmFn = void (*)(int M, int N, int K, const float* A, const float* B, float* C);

// Stage 1: textbook triple loop (i, j, k). The baseline everything is measured against.
void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C);

// Stage 2: same arithmetic, loops reordered to (i, k, j) so the inner loop
// walks B and C contiguously and the compiler can vectorize it.
void gemm_reordered(int M, int N, int K, const float* A, const float* B, float* C);

// Stage 3: cache blocking (tiling). Works on blocks of A, B and C small
// enough to stay in cache while they are reused.
void gemm_blocked(int M, int N, int K, const float* A, const float* B, float* C);

// Stage 4: stage 3 split across threads by rows of C.
void gemm_blocked_mt(int M, int N, int K, const float* A, const float* B, float* C);

// Stage 5: packed A/B panels + a register-tiled SIMD micro-kernel, picked at
// compile time: 6x16 AVX2/FMA on x86-64, 8x12 NEON on 64-bit ARM.
// Single-threaded and multithreaded versions.
void gemm_simd(int M, int N, int K, const float* A, const float* B, float* C);
void gemm_simd_mt(int M, int N, int K, const float* A, const float* B, float* C);
const char* simd_isa_name();  // "avx2" or "neon"

struct Kernel {
    std::string name;
    GemmFn fn;
    bool threaded;
};

// All kernels, in stage order. Defined in kernels.cpp.
const std::vector<Kernel>& all_kernels();

// Number of worker threads for the *_mt kernels: $GEMM_THREADS if set,
// otherwise std::thread::hardware_concurrency(). Defined in threads.cpp.
int gemm_num_threads();
