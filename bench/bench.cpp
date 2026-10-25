// Benchmarks every kernel on the GEMM shapes that ONNX Runtime runs for the
// 2-layer encoder (see profile_ort.py), plus a square 1024 case for reference.
//
//   ./bench                     all kernels, all shapes
//   ./bench --csv results.csv   also write the numbers to a CSV file
//   ./bench --skip-naive        skip the slow baseline (speedups then can't be computed)
//
// Each measurement is the median of several runs after a warm-up run. Every
// kernel's output is also checked against the float64 reference, so a fast
// but wrong kernel shows up immediately.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gemm.h"
#include "reference.h"

namespace {

struct Shape {
    std::string name;
    int M, N, K;
    int calls_per_forward;  // how many times one forward pass runs this shape
};

double seconds_now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Median time of one call, in seconds. Runs at least 3 times and keeps going
// until roughly min_total seconds have passed, so fast kernels get many samples.
double time_kernel(const Kernel& kernel, const Shape& s, const float* A, const float* B,
                   float* C, double min_total = 0.5) {
    kernel.fn(s.M, s.N, s.K, A, B, C);  // warm-up: caches, page faults, thread start-up
    std::vector<double> samples;
    const double start = seconds_now();
    while (samples.size() < 3 || (seconds_now() - start < min_total && samples.size() < 200)) {
        const double t0 = seconds_now();
        kernel.fn(s.M, s.N, s.K, A, B, C);
        samples.push_back(seconds_now() - t0);
    }
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const char* csv_path = nullptr;
    bool skip_naive = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--skip-naive") == 0) {
            skip_naive = true;
        } else {
            std::fprintf(stderr, "usage: %s [--csv FILE] [--skip-naive]\n", argv[0]);
            return 2;
        }
    }

    // M = 128 tokens. Two encoder layers, so each weight GEMM runs twice per forward pass.
    const std::vector<Shape> shapes = {
        {"qkv_proj", 128, 1536, 512, 2},
        {"attn_out", 128, 512, 512, 2},
        {"ffn_up", 128, 2048, 512, 2},
        {"ffn_down", 128, 512, 2048, 2},
        {"square_1024", 1024, 1024, 1024, 0},
    };

    std::printf("threads for *_mt kernels: %d\n\n", gemm_num_threads());
    std::printf("%-12s %-11s %10s %10s %9s %10s\n", "shape", "kernel", "ms", "GFLOP/s", "speedup",
                "rel_err");

    FILE* csv = nullptr;
    if (csv_path) {
        csv = std::fopen(csv_path, "w");
        if (!csv) {
            std::perror(csv_path);
            return 1;
        }
        std::fprintf(csv, "shape,M,N,K,kernel,threads,ms,gflops,speedup_vs_naive,rel_err\n");
    }

    std::map<std::string, double> forward_ms;  // kernel -> time for all GEMMs in one forward pass
    bool all_correct = true;

    for (const Shape& s : shapes) {
        const auto A = random_matrix(s.M, s.K, 1);
        const auto B = random_matrix(s.K, s.N, 2);
        std::vector<double> ref;
        gemm_reference_f64(s.M, s.N, s.K, A.data(), B.data(), ref);
        std::vector<float> C(static_cast<size_t>(s.M) * s.N);
        const double flops = 2.0 * s.M * s.N * s.K;

        double naive_seconds = 0.0;
        for (const Kernel& kernel : all_kernels()) {
            if (skip_naive && kernel.name == "naive") continue;

            const double sec = time_kernel(kernel, s, A.data(), B.data(), C.data());
            const double err = max_relative_error(C, ref);
            if (err > error_tolerance(s.K)) all_correct = false;
            if (kernel.name == "naive") naive_seconds = sec;
            const double speedup = naive_seconds > 0.0 ? naive_seconds / sec : 0.0;
            const int threads = kernel.threaded ? gemm_num_threads() : 1;

            char speedup_str[16] = "-";
            if (speedup > 0.0) std::snprintf(speedup_str, sizeof(speedup_str), "%.1fx", speedup);
            std::printf("%-12s %-11s %10.3f %10.2f %9s %10.1e\n", s.name.c_str(),
                        kernel.name.c_str(), sec * 1e3, flops / sec * 1e-9, speedup_str, err);
            if (csv) {
                std::fprintf(csv, "%s,%d,%d,%d,%s,%d,%.4f,%.3f,%.2f,%.3e\n", s.name.c_str(), s.M,
                             s.N, s.K, kernel.name.c_str(), threads, sec * 1e3,
                             flops / sec * 1e-9, speedup, err);
            }
            forward_ms[kernel.name] += s.calls_per_forward * sec * 1e3;
        }
        std::printf("\n");
    }

    std::printf("All GEMMs in one forward pass (2 layers):\n");
    for (const Kernel& kernel : all_kernels()) {
        if (forward_ms.count(kernel.name)) {
            std::printf("  %-11s %9.3f ms\n", kernel.name.c_str(), forward_ms[kernel.name]);
        }
    }

    if (csv) std::fclose(csv);
    if (!all_correct) {
        std::printf("\nWARNING: at least one kernel exceeded the error tolerance\n");
        return 1;
    }
    return 0;
}
