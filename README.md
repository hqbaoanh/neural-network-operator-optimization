# neural-network-operator-optimization

Where does the time go when a CPU runs a Transformer, and how far can a hand written kernel close the gap to a production library? 

I exported a small Transformer encoder from PyTorch to ONNX and profiled it in ONNX Runtime (ORT). Matrix multiplication (GEMM) takes about 85% of the runtime, so I'm writing a C++ GEMM from scratch in stages, each one fixing the main bottleneck of the one before, and measuring every stage on the exact matrix shapes the model runs. 

## Profiling: why GEMM 

`export_model.py` builds a 2 layer encoder (d_model 512, 8 heads, FFN 2048, sequence length 128) with random weights, since only speed matters here. 

`profile_ort.py` runs it 100 times in ONNX Runtime and sums kernel time by operator. 

On my Macbook (M3 Pro, 11 core CPU, 18 GB), 

## References

- MIT 6.172 Performance Engineering of Software Systems (Fall 2018),
  Lecture 1: Introduction and Matrix Multiplication, by Charles Leiserson and
  Julian Shun. The loop reordering, tiling, parallelism and vectorization
  steps follow the same progression.
  https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/