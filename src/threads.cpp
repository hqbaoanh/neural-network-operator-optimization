#include <cstdlib>
#include <thread>
#include <vector>

#include "gemm.h"
#include "gemm_internal.h"

int gemm_num_threads() {
    if (const char* env = std::getenv("GEMM_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) return n;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

void parallel_for_threads(int num_threads, const std::function<void(int, int)>& work) {
    std::vector<std::thread> workers;
    workers.reserve(num_threads > 1 ? num_threads - 1 : 0);
    for (int t = 1; t < num_threads; ++t) {
        workers.emplace_back(work, t, num_threads);
    }
    work(0, num_threads);  // the calling thread does a share too
    for (auto& w : workers) w.join();
}
