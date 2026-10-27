# Neural Network Operator Optimization

Where does the time go when a CPU runs a Transformer, and how far can a hand written kernel close the gap to a production library? 

I exported a small Transformer encoder from PyTorch to ONNX and profiled it in ONNX Runtime (ORT). Matrix multiplication (GEMM) takes about 85% of the kernel time, so I'm writing a C++ GEMM from scratch in stages, each one fixing the main bottleneck of the one before, and measuring every stage on the exact matrix shapes the model runs. 

## Profiling: why GEMM 

`export_model.py` builds a 2 layer encoder (d_model 512, 8 heads, FFN 2048, sequence length 128) with random weights, since only speed matters here. 

`profile_ort.py` runs it 100 times in ONNX Runtime and sums kernel time by operator. 

On my MacBook (M3 Pro, 11 core CPU, 18 GB, 1 thread), `Gemm` is about 85% of kernel time across runs, and about 89% including the attention matmuls. The four weight GEMMs ($M \times K \times N$), each run twice per forward pass: 

| Layer | Shape |
|---|---|
| QKV projection | 128 x 512 x 1536 |
| Attention output | 128 x 512 x 512 |
| FFN up | 128 x 512 x 2048 |
| FFN down | 128 x 2048 x 512 |

## Results so far
 
Same machine, 1 thread, Apple clang 17, `-O3`. Every kernel is checked against
a float64 reference.
 
| Kernel | All GEMMs in one forward pass | GFLOP/s | vs naive |
|---|---:|---:|---:|
| ONNX Runtime (MLAS), the target | 15.4 ms | ~104 | 44x |
| Stage 1: naive triple loop | 675.6 ms | 2.3 to 2.5 | 1x |
| Stage 2: loop reordering (i, k, j) | 54.0 ms | 29.6 to 29.9 | 12.5x |
| Stage 3: cache blocking | 62.1 ms | 24.6 to 27.5 | 10.9x |
| Stage 4: cache blocking + 5 threads | 16.3 ms | 92.9 to 106.3 |
| Stage 4: cache blocking + 11 threads | 12.6 ms | 123.9 to 139.5 |
 
### What the numbers show

The naive kernel is about 44x slower than ONNX Runtime. On this Mac its speed
is nearly the same for every shape, which suggests it's limited mainly by the
chain of dependent additions into one running sum (each add waits about 3 to 4
cycles for the previous one), rather than by memory access.

Reordering the loops to (i, k, j) makes the inner loop walk rows of B and C
and update independent elements, so the compiler vectorizes it with NEON.
That gives 12.5x over naive (54.0 ms per forward pass), at almost exactly
30 GFLOP/s for every shape. A flat number across shapes points to a limit
inside the core rather than cache misses: each FMA also loads B, loads C and
stores C, which works out to about one vector FMA per cycle, roughly a
quarter of peak.

Cache blocking (64 x 256 x 256 blocks) made things about 15% slower on this
Mac. The largest B in the model is 4 MB, which already fits in the M3 Pro's
16 MB L2, so there was no cache problem to fix and the extra loops only add
overhead. The kernel was limited by loading and storing C, which blocking
doesn't change. On a 2-core x86 cloud VM with smaller caches, the same code
was 1.3x to 1.4x faster than reordering on most shapes.

Splitting the rows of C across threads needs no locks, since each thread
writes only its own rows. With the 5 performance cores it is 4.0x faster
than one thread (16.3 ms). All 11 cores reach 5.1x (12.6 ms): the M3 Pro's
6 efficiency cores are roughly half as fast on this kernel, and because the
rows are split evenly, the performance cores finish early and wait for them.
18 threads gave the same 12.6 ms, so threads beyond the core count add
nothing. These are multithreaded numbers, so they are not comparable with
ONNX Runtime's single-threaded 15.4 ms.

## Progress

- [x] Stage 1: benchmark harness, float64 correctness tests, CI, naive kernel
- [x] Stage 2: loop reordering so the inner loop vectorizes
- [x] Stage 3: cache blocking
- [x] Stage 4: multithreading
- [ ] Stage 5: packed panels + SIMD micro-kernels (NEON on Apple Silicon, AVX2 on x86)

## Build and run
 
Needs a C++17 compiler and CMake 3.16+, on Apple Silicon or x86-64 with AVX2.
 
```bash
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure    # correctness tests
./build/bench                                  # benchmark
```
 
To redo the profiling:
 
```bash
pip install -r requirements.txt
python export_model.py && python profile_ort.py 1
```

## References

- MIT 6.172 Performance Engineering of Software Systems (Fall 2018),
  Lecture 1: Introduction and Matrix Multiplication, by Charles Leiserson and
  Julian Shun. The loop reordering, tiling, parallelism and vectorization
  steps follow the same progression.
  https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/