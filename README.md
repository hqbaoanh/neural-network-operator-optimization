# Neural Network Operator Optimization

Where does the time go when a CPU runs a Transformer, and how far can a hand written kernel close the gap to a production library? 

I exported a small Transformer encoder from PyTorch to ONNX and profiled it in ONNX Runtime (ORT). Matrix multiplication (GEMM) takes about 85% of the runtime, so I'm writing a C++ GEMM from scratch in stages, each one fixing the main bottleneck of the one before, and measuring every stage on the exact matrix shapes the model runs. 

## Profiling: why GEMM 

`export_model.py` builds a 2 layer encoder (d_model 512, 8 heads, FFN 2048, sequence length 128) with random weights, since only speed matters here. 

`profile_ort.py` runs it 100 times in ONNX Runtime and sums kernel time by operator. 

On my MacBook (M3 Pro, 11 core CPU, 18 GB), `Gemm` is about 85% of kernel time across runs, and about 89% including the attention matmuls. The four weight GEMMs ($M \times K \times N$), each run twice per foward pass: 

| Layer | Shape |
|---|---|
| QKV projection | 128 x 512 x 1536 |
| Attention output | 128 x 512 x 512 |
| FFN up | 128 x 512 x 2048 |
| FFN down | 128 x 2048 x 512 |

## Results so far
 
Same machine, 1 thread, Apple clang 17, `-O3`. Every kernel is checked against
a float64 reference.
 
| Kernel | All GEMMs in one forward pass | GFLOP/s |
|---|---:|---:|
| ONNX Runtime (MLAS), the target | 15.4 ms | ~104 |
| Stage 1: naive triple loop | 675.6 ms | 2.3 to 2.5 |

 
The naive kernel is about 44x slower than ONNX Runtime. On this Mac its speed
is nearly the same for every shape, which suggests it's limited mainly by the
chain of dependent additions into one running sum (each add waits about 3 to 4
cycles for the previous one), rather than by memory access.


## References

- MIT 6.172 Performance Engineering of Software Systems (Fall 2018),
  Lecture 1: Introduction and Matrix Multiplication, by Charles Leiserson and
  Julian Shun. The loop reordering, tiling, parallelism and vectorization
  steps follow the same progression.
  https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/