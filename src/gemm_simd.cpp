// Stage 5: packing + a register-tiled SIMD micro-kernel.
//
// This follows the structure of BLIS / GotoBLAS:
//
//   for jc in N by NC:            B block  (KC x NC)  -> stays in L2/L3
//     for pc in K by KC:
//       pack B[pc:pc+KC, jc:jc+NC] into panels of NR columns
//       for ic in M by MC:        A block  (MC x KC)  -> stays in L2
//         pack A[ic:ic+MC, pc:pc+KC] into panels of MR rows
//         for jr in NC by NR:     one B micro-panel (KC x NR) -> L1
//           for ir in MC by MR:
//             micro-kernel: C[MR x NR] += A_panel(MR x KC) * B_panel(KC x NR)
//
// The micro-kernel keeps an MR x NR tile of C in vector registers for the
// whole k loop, so each value of A and B it loads gets used many times
// before it is thrown away. That ratio of arithmetic to memory traffic is
// what lets it get close to the CPU's peak.
//
// Everything in this file is the same on every CPU. Only the micro-kernel and
// its tile size depend on the instruction set, and those live in
// microkernel_avx2.h (x86-64) and microkernel_neon.h (64-bit ARM).
//
// Packing copies the blocks into the exact order the micro-kernel reads
// them, so its loads are contiguous. Panels are zero-padded on the edges,
// which means the micro-kernel never needs a special case: edge tiles are
// computed in full into a scratch tile and only the valid part is added to C.
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

// Cache block sizes. Override at configure time to tune for a specific CPU:
//   cmake -B build -DGEMM_MC=96 -DGEMM_KC=384 -DGEMM_NC=2048
// Rough guide: a KC x NR panel of B should fit in L1, the MC x KC block of A
// in L2, and the KC x NC block of B in L2/L3.
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
    bytes = (bytes + 63) / 64 * 64;  // aligned_alloc wants a multiple of the alignment
    return AlignedBuffer(static_cast<float*>(std::aligned_alloc(64, bytes)));
}

// Copy B[0:kc, 0:nc] (row stride ldb) into panels of NR columns.
// Panel p holds columns [p*NR, p*NR+NR) as kc consecutive rows of NR floats.
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

// Copy A[0:mc, 0:kc] (row stride lda) into panels of MR rows.
// Panel p holds rows [p*MR, p*MR+MR) stored column by column: for each k,
// the MR values A[r][k] are next to each other.
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

// Computes columns [col_begin, col_end) of C = A * B.
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
                            // Edge of the matrix: compute the full padded tile
                            // into scratch space, then add only the valid part.
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

// The threaded version splits the COLUMNS of C, in whole NR-wide panels.
// For the transformer shapes M is only 128 but N is 512-2048, so columns
// give every thread plenty of work. Each thread packs its own buffers, so
// they never share writable memory.
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
