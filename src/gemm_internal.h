// Helpers shared between kernel files. Not part of the public interface.
#pragma once

// Stage 3 kernel restricted to rows [row_begin, row_end) of C.
void gemm_blocked_rows(int row_begin, int row_end, int N, int K,
                       const float* A, const float* B, float* C);
