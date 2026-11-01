"""Time numpy matmul (OpenBLAS / Accelerate) on the bench.cpp shapes.

usage: python bench/compare_blas.py [threads]

On Mac this uses Accelerate, which runs on the AMX coprocessor, not the CPU
cores (~1500 GFLOP/s), so it's not a fair comparison for my kernel.
I compare against ORT's MLAS instead.
"""
import os
import sys

THREADS = sys.argv[1] if len(sys.argv) > 1 else "1"
# has to be set before importing numpy (BLAS makes its thread pool on import)
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = THREADS

import time  # noqa: E402

import numpy as np  # noqa: E402

SHAPES = [  # (name, M, N, K), same as bench.cpp
    ("qkv_proj", 128, 1536, 512),
    ("attn_out", 128, 512, 512),
    ("ffn_up", 128, 2048, 512),
    ("ffn_down", 128, 512, 2048),
    ("square_1024", 1024, 1024, 1024),
]


def blas_description():
    """which BLAS numpy is using (+ OpenBLAS's CPU kernel if threadpoolctl is there)"""
    try:
        name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        name = "unknown"
    try:
        import threadpoolctl  # optional: pip install threadpoolctl

        for pool in threadpoolctl.threadpool_info():
            if pool.get("architecture"):
                return f"{name} ({pool['architecture']} kernels)"
    except ImportError:
        pass
    return name


def median_seconds(fn, min_total=0.5):
    fn()  # warm-up
    samples = []
    start = time.perf_counter()
    while len(samples) < 3 or (time.perf_counter() - start < min_total and len(samples) < 200):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main():
    rng = np.random.default_rng(0)
    print(f"numpy {np.__version__}, BLAS: {blas_description()}, threads={THREADS}")
    print(f"{'shape':12s} {'ms':>10s} {'GFLOP/s':>10s}")
    for name, m, n, k in SHAPES:
        a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
        b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
        c = np.empty((m, n), dtype=np.float32)
        sec = median_seconds(lambda: np.matmul(a, b, out=c))
        print(f"{name:12s} {sec * 1e3:10.3f} {2 * m * n * k / sec * 1e-9:10.2f}")


if __name__ == "__main__":
    main()
