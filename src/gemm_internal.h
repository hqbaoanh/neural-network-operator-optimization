// Helpers shared between kernel files. Not part of the public interface.
#pragma once

#include <functional>

// Stage 3 kernel restricted to rows [row_begin, row_end) of C.
void gemm_blocked_rows(int row_begin, int row_end, int N, int K,
                       const float* A, const float* B, float* C);

// Runs work(t, num_threads) on num_threads threads (t = 0 .. num_threads-1)
// and waits for all of them. Thread 0 is the calling thread.
void parallel_for_threads(int num_threads, const std::function<void(int, int)>& work);
