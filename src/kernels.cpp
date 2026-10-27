#include "gemm.h"

const std::vector<Kernel>& all_kernels() {
    static const std::vector<Kernel> kernels = {
        {"naive", gemm_naive, false},
        {"reordered", gemm_reordered, false},
        {"blocked", gemm_blocked, false},
        {"blocked_mt", gemm_blocked_mt, true},
        {simd_isa_name(), gemm_simd, false},                         // "avx2" or "neon"
        {std::string(simd_isa_name()) + "_mt", gemm_simd_mt, true},
    };
    return kernels;
}
