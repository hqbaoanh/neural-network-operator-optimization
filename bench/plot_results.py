"""Plots GFLOP/s per kernel for each shape from a bench.cpp CSV.

    ./build/bench --csv results/my_machine.csv
    python bench/plot_results.py results/my_machine.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    gflops = defaultdict(dict)
    for r in rows:
        gflops[r["kernel"]][r["shape"]] = float(r["gflops"])

    width = 0.8 / len(kernels)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, kernel in enumerate(kernels):
        xs = [s + (i - (len(kernels) - 1) / 2) * width for s in range(len(shapes))]
        ax.bar(xs, [gflops[kernel].get(s, 0) for s in shapes], width, label=kernel)

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes)
    ax.set_ylabel("GFLOP/s (higher is better)")
    ax.set_title(f"GEMM throughput by optimization stage ({Path(csv_path).stem})")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path(csv_path).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python bench/plot_results.py results/<machine>.csv")
    main(sys.argv[1])
