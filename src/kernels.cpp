#include "gemm.h"

const std::vector<Kernel>& all_kernels() {
    static const std::vector<Kernel> kernels = {
        {"naive", gemm_naive, false},
    };
    return kernels;
}
