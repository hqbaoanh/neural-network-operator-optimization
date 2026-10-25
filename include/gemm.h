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
