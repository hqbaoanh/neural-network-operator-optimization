#include <cstdlib>
#include <thread>

#include "gemm.h"

int gemm_num_threads() {
    if (const char* env = std::getenv("GEMM_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) return n;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}
