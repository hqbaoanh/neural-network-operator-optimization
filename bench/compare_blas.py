"""Times NumPy's matmul on the same shapes as bench.cpp.

NumPy hands float32 matmul to a production BLAS library: usually OpenBLAS on
Linux and Windows, and Apple's Accelerate on recent Apple Silicon Macs (the
script prints which one it found). This is the "how close are we to a library
people actually ship" number. Run it with the same thread count you give the
C++ benchmark:

    python bench/compare_blas.py          # 1 thread
    python bench/compare_blas.py 2        # 2 threads

Note: Accelerate can run matmul on Apple's matrix coprocessor (AMX) rather
than the CPU's NEON units, so on a Mac this is not a like-for-like comparison
with a CPU kernel. It shows what dedicated matrix hardware buys you.
"""
import os
import sys

THREADS = sys.argv[1] if len(sys.argv) > 1 else "1"
# The BLAS thread pool is created when NumPy is imported, so set these first.
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
    """Which BLAS NumPy uses, and (for OpenBLAS) which CPU kernels it picked at runtime."""
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
