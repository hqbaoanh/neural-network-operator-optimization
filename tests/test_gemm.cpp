// Correctness tests: every kernel against the float64 reference.
//
// The shapes are picked to hit edge cases rather than to be fast:
// sizes of 1, sizes that are not multiples of the micro-kernel tile (6 x 16
// for AVX2, 8 x 12 for NEON), sizes just over a cache block, and the real
// transformer shapes.
#include <cstdio>
#include <vector>

#include "gemm.h"
#include "reference.h"

namespace {
struct Shape {
    int M, N, K;
};
}  // namespace

int main() {
    const std::vector<Shape> shapes = {
        {1, 1, 1},       {1, 17, 3},       {7, 5, 13},       {6, 16, 1},
        {6, 16, 256},    {12, 32, 257},    {13, 33, 7},      {8, 12, 1},
        {9, 13, 385},    {17, 25, 129},    {73, 1025, 300},  {127, 129, 511},
        {300, 300, 300}, {128, 1536, 512}, {128, 512, 2048},
    };

    int failures = 0;
    for (const Shape& s : shapes) {
        const auto A = random_matrix(s.M, s.K, 1);
        const auto B = random_matrix(s.K, s.N, 2);
        std::vector<double> ref;
        gemm_reference_f64(s.M, s.N, s.K, A.data(), B.data(), ref);

        for (const Kernel& kernel : all_kernels()) {
            // Fill C with garbage to check that kernels overwrite it instead of adding to it.
            std::vector<float> C(static_cast<size_t>(s.M) * s.N, 6767.0f);
            kernel.fn(s.M, s.N, s.K, A.data(), B.data(), C.data());

            const double err = max_relative_error(C, ref);
            const bool ok = err <= error_tolerance(s.K);
            if (!ok) ++failures;
            std::printf("%-4s %-11s M=%-4d N=%-5d K=%-5d rel_err=%.2e\n", ok ? "ok" : "FAIL",
                        kernel.name.c_str(), s.M, s.N, s.K, err);
        }
    }

    if (failures > 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
