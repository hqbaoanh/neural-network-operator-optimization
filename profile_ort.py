
import collections
import json
import sys

import numpy as np
import onnxruntime as ort

MODEL = "model.onnx"
RUNS = 100
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def _dims(type_shape):
    return list(next(iter(type_shape.values())))


def gemm_shape(args):
    """Returns 'M x K x N' for a 2-D Gemm/MatMul, or the raw input shapes otherwise.

    ORT pre-packs constant weights, so the weight matrix is usually missing
    from input_type_shape (for Gemm the second entry is then the bias). M and K
    come from the activation input and N from the output instead.
    """
    inputs = [_dims(d) for d in args.get("input_type_shape", [])]
    outputs = [_dims(d) for d in args.get("output_type_shape", [])]
    if inputs and outputs and len(inputs[0]) == 2 and len(outputs[0]) == 2:
        m, k = inputs[0]
        return f"{m} x {k} x {outputs[0][1]}"
    return str(inputs[:2])

def main():
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    x = np.load("ref_input.npy")
    for _ in range(10):  
        sess.run(None, {"x": x}) #warm up 
    for _ in range(RUNS):
        sess.run(None, {"x": x})
    trace_path = sess.end_profiling()

    with open(trace_path) as f:
        events = json.load(f)

    per_op = collections.defaultdict(float)
    shapes = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        args = e.get("args", {})
        op = args.get("op_name", "?")
        per_op[op] += e["dur"]  
        if op in ("MatMul", "Gemm", "FusedMatMul"):
            shapes[(op, gemm_shape(args))] += 1

    total = sum(per_op.values())
    print(f"threads={THREADS}  runs={RUNS}  total kernel time per run: {total / RUNS / 1000:.2f} ms")
    print(f"{'op':20s} {'ms/run':>10s} {'share':>8s}")
    for op, us in sorted(per_op.items(), key=lambda kv: -kv[1])[:10]:
        print(f"{op:20s} {us / RUNS / 1000:10.3f} {100 * us / total:7.1f}%")
    print("\nMatMul/Gemm shapes M x K x N (attention matmuls are batched, shown raw):")
    for (op, shp), n in shapes.most_common(8):
        print(f"  {op:12s} {shp}  x{n // RUNS} per run")


if __name__ == "__main__":
    main()
