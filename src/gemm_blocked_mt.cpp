// Stage 4: multithreading.
//
// Rows of C are independent: row i only reads row i of A (and all of B) and
// only writes row i of C. So we hand each thread a contiguous range of rows
// and let it run the stage 3 blocked kernel on that range. No locks needed,
// since no two threads ever write the same element of C.
//
// Thread count comes from GEMM_THREADS (or the number of cores).
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
