"""Profiles model.onnx in ONNX Runtime and reports time per operator type.

Also prints the (M, K) x (K, N) shapes of the MatMul/Gemm nodes, which are the
shapes the custom C++ GEMM is benchmarked on afterwards.
"""
import collections
import json
import sys

import numpy as np
import onnxruntime as ort

MODEL = "model.onnx"
RUNS = 100
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    x = np.load("ref_input.npy")
    for _ in range(10):  # warm-up
        sess.run(None, {"x": x})
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
        per_op[op] += e["dur"]  # microseconds
        if op in ("MatMul", "Gemm", "FusedMatMul"):
            shp = [list(next(iter(d.values()))) for d in args.get("input_type_shape", [])[:2]]
            shapes[(op, str(shp))] += 1

    total = sum(per_op.values())
    print(f"threads={THREADS}  runs={RUNS}  total kernel time per run: {total / RUNS / 1000:.2f} ms")
    print(f"{'op':20s} {'ms/run':>10s} {'share':>8s}")
    for op, us in sorted(per_op.items(), key=lambda kv: -kv[1])[:10]:
        print(f"{op:20s} {us / RUNS / 1000:10.3f} {100 * us / total:7.1f}%")
    print("\nMatMul/Gemm shapes (calls across all runs):")
    for (op, shp), n in shapes.most_common(8):
        print(f"  {op:12s} {shp}  x{n // RUNS} per run")


if __name__ == "__main__":
    main()
