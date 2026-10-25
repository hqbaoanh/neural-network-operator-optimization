#include "gemm.h"

const std::vector<Kernel>& all_kernels() {
    static const std::vector<Kernel> kernels = {
        {"naive", gemm_naive, false},
        {"reordered", gemm_reordered, false},
    };
    return kernels;
}
