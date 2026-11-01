// internal helpers shared by the kernel files
#pragma once

#include <functional>

// stage 3 on rows [row_begin, row_end) of C only
void gemm_blocked_rows(int row_begin, int row_end, int N, int K,
                       const float* A, const float* B, float* C);

// runs work(t, num_threads) for t = 0..num_threads-1 and waits. t = 0 runs on the calling thread.
void parallel_for_threads(int num_threads, const std::function<void(int, int)>& work);
