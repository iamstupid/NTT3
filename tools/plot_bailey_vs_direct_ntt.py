#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "pow2": int(row["pow2"]),
                    "limbs": int(row["limbs"]),
                    "iters": int(row["iters"]),
                    "direct_ms": float(row["direct_ms"]),
                    "bailey_ms": float(row["bailey_ms"]),
                    "ratio": float(row["bailey_over_direct"]),
                }
            )
    rows.sort(key=lambda x: x["pow2"])
    return rows


def estimate_cross(rows):
    for i in range(len(rows) - 1):
        x0, y0 = rows[i]["pow2"], rows[i]["ratio"]
        x1, y1 = rows[i + 1]["pow2"], rows[i + 1]["ratio"]
        if y0 >= 1.0 and y1 <= 1.0 and y1 != y0:
            t = (1.0 - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Plot Direct vs Bailey single-prime NTT timings."
    )
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Single-Prime NTT: Direct vs Bailey")
    ap.add_argument(
        "--time-log",
        action="store_true",
        default=True,
        help="Use log scale for time axis (enabled by default).",
    )
    ap.add_argument(
        "--time-linear",
        action="store_true",
        help="Force linear scale for time axis.",
    )
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        raise SystemExit("No rows found in CSV.")

    p = [r["pow2"] for r in rows]
    xlbl = [f"2^{k}" for k in p]
    d = [r["direct_ms"] for r in rows]
    b = [r["bailey_ms"] for r in rows]
    q = [r["ratio"] for r in rows]
    cross = estimate_cross(rows)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(args.title, fontsize=14)

    ax0.plot(xlbl, d, marker="o", label="Direct")
    ax0.plot(xlbl, b, marker="o", label="Bailey")
    if args.time_log and not args.time_linear:
        ax0.set_yscale("log")
    ax0.set_ylabel("Time per forward NTT (ms)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(xlbl, q, marker="o", color="tab:purple", label="Bailey / Direct")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    if cross is not None:
        ax1.text(
            0.02,
            0.92,
            f"Estimated crossover near 2^{cross:.2f} limbs",
            transform=ax1.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    ax1.set_ylabel("Ratio")
    ax1.set_xlabel("Transform length (limbs)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
