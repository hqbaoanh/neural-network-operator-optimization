// stage 4: each thread gets a range of rows of C and runs stage 3 on it.
// row i of C only needs row i of A (+ all of B), so threads never write the same
// element -> no locks.
#include <algorithm>

#include "gemm.h"
#include "gemm_internal.h"

void gemm_blocked_mt(int M, int N, int K, const float* A, const float* B, float* C) {
    const int num_threads = std::max(1, std::min(gemm_num_threads(), M));
    parallel_for_threads(num_threads, [&](int t, int nt) {
        const int rows_per_thread = (M + nt - 1) / nt;
        const int begin = t * rows_per_thread;
        const int end = std::min(M, begin + rows_per_thread);
        if (begin < end) {
            gemm_blocked_rows(begin, end, N, K, A, B, C);
        }
    });
}
