// stage 5: packing + register-tiled micro-kernel (BLIS / Goto loop order)
//
//   jc over N by NC       B block KC x NC   (L2/L3)
//    pc over K by KC      pack B into NR-wide panels
//     ic over M by MC     A block MC x KC   (L2), pack A into MR-tall panels
//      jr over NC by NR   one B panel KC x NR (L1)
//       ir over MC by MR  micro-kernel: C tile MR x NR stays in registers for all of k
//
// only the micro-kernel is ISA-specific (microkernel_avx2.h / microkernel_neon.h).
// packing puts A and B in the order the kernel reads them. panels are zero-padded,
// so edge tiles use the same kernel into a scratch tile and only the valid part is copied.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "gemm.h"
#include "gemm_internal.h"

#if defined(__AVX2__) && defined(__FMA__)
#include "microkernel_avx2.h"
#elif defined(__ARM_NEON) && defined(__aarch64__)
#include "microkernel_neon.h"
#else
#error "No SIMD micro-kernel for this target: build with -mavx2 -mfma on x86-64, or for 64-bit ARM"
#endif

// block sizes, override with e.g. cmake -B build -DGEMM_KC=384
// aim: KC x NR B panel in L1, MC x KC A block in L2, KC x NC B block in L2/L3
#ifndef GEMM_MC
#define GEMM_MC simd::kDefaultMC
#endif
#ifndef GEMM_KC
#define GEMM_KC simd::kDefaultKC
#endif
#ifndef GEMM_NC
#define GEMM_NC simd::kDefaultNC
#endif

namespace {

using simd::MR;
using simd::NR;
constexpr int MC = GEMM_MC;
constexpr int KC = GEMM_KC;
constexpr int NC = GEMM_NC;

static_assert(MC % MR == 0, "GEMM_MC must be a multiple of the micro-tile height MR");
static_assert(NC % NR == 0, "GEMM_NC must be a multiple of the micro-tile width NR");
static_assert(KC > 0, "GEMM_KC must be positive");

struct AlignedFree {
    void operator()(float* p) const { std::free(p); }
};
using AlignedBuffer = std::unique_ptr<float[], AlignedFree>;

AlignedBuffer alloc_aligned(size_t count) {
    size_t bytes = count * sizeof(float);
    bytes = (bytes + 63) / 64 * 64;  // aligned_alloc needs size % alignment == 0
    return AlignedBuffer(static_cast<float*>(std::aligned_alloc(64, bytes)));
}

// B[0:kc, 0:nc] -> panels NR columns wide, each stored as kc rows of NR floats
void pack_b(int kc, int nc, const float* B, int ldb, float* packed) {
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        for (int k = 0; k < kc; ++k) {
            const float* src = B + static_cast<size_t>(k) * ldb + j;
            int c = 0;
            for (; c < cols; ++c) packed[c] = src[c];
            for (; c < NR; ++c) packed[c] = 0.0f;
            packed += NR;
        }
    }
}

// A[0:mc, 0:kc] -> panels MR rows tall, stored by column (for each k, MR values together)
void pack_a(int mc, int kc, const float* A, int lda, float* packed) {
    for (int i = 0; i < mc; i += MR) {
        const int rows = std::min(MR, mc - i);
        for (int k = 0; k < kc; ++k) {
            int r = 0;
            for (; r < rows; ++r) packed[r] = A[static_cast<size_t>(i + r) * lda + k];
            for (; r < MR; ++r) packed[r] = 0.0f;
            packed += MR;
        }
    }
}

// columns [col_begin, col_end) of C
void gemm_simd_cols(int M, int N, int K, int col_begin, int col_end,
                    const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        std::memset(C + static_cast<size_t>(i) * N + col_begin, 0,
                    sizeof(float) * (col_end - col_begin));
    }

    AlignedBuffer packed_a = alloc_aligned(static_cast<size_t>(MC) * KC);
    AlignedBuffer packed_b = alloc_aligned(static_cast<size_t>(KC) * NC);
    alignas(64) float edge_tile[MR * NR];

    for (int jc = col_begin; jc < col_end; jc += NC) {
        const int nc = std::min(NC, col_end - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * N + jc, N, packed_b.get());

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + static_cast<size_t>(ic) * K + pc, K, packed_a.get());

                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min(NR, nc - jr);
                    const float* b_panel = packed_b.get() + static_cast<size_t>(jr) * kc;

                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        const float* a_panel = packed_a.get() + static_cast<size_t>(ir) * kc;
                        float* c_tile = C + static_cast<size_t>(ic + ir) * N + jc + jr;

                        if (mr == MR && nr == NR) {
                            simd::micro_kernel(kc, a_panel, b_panel, c_tile, N);
                        } else {
                            // edge tile: full tile into scratch, then add the valid part
                            std::memset(edge_tile, 0, sizeof(edge_tile));
                            simd::micro_kernel(kc, a_panel, b_panel, edge_tile, NR);
                            for (int r = 0; r < mr; ++r) {
                                for (int c = 0; c < nr; ++c) {
                                    c_tile[static_cast<size_t>(r) * N + c] += edge_tile[r * NR + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

const char* simd_isa_name() { return simd::kName; }

void gemm_simd(int M, int N, int K, const float* A, const float* B, float* C) {
    gemm_simd_cols(M, N, K, 0, N, A, B, C);
}

// split by columns (whole NR panels), not rows: M is only 128 here but N is 512-2048.
// each thread packs its own A and B, nothing shared.
// TODO: every thread packs the same A again -- maybe why 11 threads barely beat 5?
void gemm_simd_mt(int M, int N, int K, const float* A, const float* B, float* C) {
    const int panels = (N + NR - 1) / NR;
    const int num_threads = std::max(1, std::min(gemm_num_threads(), panels));
    parallel_for_threads(num_threads, [&](int t, int nt) {
        const int begin = static_cast<int>(static_cast<long>(panels) * t / nt) * NR;
        const int end = std::min(N, static_cast<int>(static_cast<long>(panels) * (t + 1) / nt) * NR);
        if (begin < end) {
            gemm_simd_cols(M, N, K, begin, end, A, B, C);
        }
    });
}
