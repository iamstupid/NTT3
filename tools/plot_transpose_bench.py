#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 11:
                continue
            rows.append(
                {
                    "rows": int(parts[0]),
                    "cols": int(parts[1]),
                    "iters": int(parts[2]),
                    "plain_ms": float(parts[3]),
                    "plain_gbps": float(parts[4]),
                    "blocked_ms": float(parts[5]),
                    "blocked_gbps": float(parts[6]),
                    "co_ms": float(parts[7]),
                    "co_gbps": float(parts[8]),
                    "blocked_speedup": float(parts[9]),
                    "co_speedup": float(parts[10]),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Plot transpose benchmark curves from bench_transpose CSV output."
    )
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--output", required=True, help="Output PNG file")
    ap.add_argument(
        "--title",
        default="AVX2 Vec Transpose Benchmark",
        help="Figure title",
    )
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit("No data rows found in input.")

    labels = [f"{r['rows']}x{r['cols']}" for r in rows]
    x = list(range(len(rows)))

    plain_ms = [r["plain_ms"] for r in rows]
    block_ms = [r["blocked_ms"] for r in rows]
    co_ms = [r["co_ms"] for r in rows]

    plain_gbps = [r["plain_gbps"] for r in rows]
    block_gbps = [r["blocked_gbps"] for r in rows]
    co_gbps = [r["co_gbps"] for r in rows]

    block_sp = [r["blocked_speedup"] for r in rows]
    co_sp = [r["co_speedup"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(args.title)

    ax = axes[0]
    ax.plot(x, plain_ms, marker="o", label="Plain")
    ax.plot(x, block_ms, marker="o", label="Blocked")
    ax.plot(x, co_ms, marker="o", label="Cache-Oblivious")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(x, plain_gbps, marker="o", label="Plain")
    ax.plot(x, block_gbps, marker="o", label="Blocked")
    ax.plot(x, co_gbps, marker="o", label="Cache-Oblivious")
    ax.set_ylabel("Throughput (GB/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(x, block_sp, marker="o", label="Blocked / Plain")
    ax.plot(x, co_sp, marker="o", label="Cache-Oblivious / Plain")
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Matrix shape (rows_vec x cols_vec)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=25, ha="right")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote plot: {out}")


if __name__ == "__main__":
    main()

