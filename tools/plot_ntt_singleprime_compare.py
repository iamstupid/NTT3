import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    n = []
    pair = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line]))
            n.append(int(row[0]))
            pair.append(float(row[3]))
    return n, pair


def to_map(xs, ys):
    return {x: y for x, y in zip(xs, ys)}


root = Path(__file__).resolve().parent.parent
bench_dir = root / "benchmarks"

flint_n, flint_pair = load_csv(bench_dir / "bench_flint_sdfft_singleprime.csv")
old_n, old_pair = load_csv(bench_dir / "bench_our_singleprime_old.csv")
v2_n, v2_pair = load_csv(bench_dir / "bench_ntt_v2_lazy256.csv")

flint_m = to_map(flint_n, flint_pair)
old_m = to_map(old_n, old_pair)
v2_m = to_map(v2_n, v2_pair)

common_old = sorted(set(old_m.keys()) & set(flint_m.keys()))
common_v2 = sorted(set(v2_m.keys()) & set(flint_m.keys()))

old_ratio = [old_m[x] / flint_m[x] for x in common_old]
v2_ratio = [v2_m[x] / flint_m[x] for x in common_v2]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True
)

ax1.plot(flint_n, flint_pair, marker="o", label="FLINT sd_fft (single-prime)")
ax1.plot(old_n, old_pair, marker="o", label="Our old single-prime")
ax1.plot(v2_n, v2_pair, marker="o", label="Our v2 lazy+256")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("Transform length n")
ax1.set_ylabel("Forward+Inverse time (ms)")
ax1.set_title("Single-Prime NTT Pair Time Comparison")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

ax2.plot(common_old, old_ratio, marker="o", label="Old / FLINT")
ax2.plot(common_v2, v2_ratio, marker="o", label="v2 / FLINT")
ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1)
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Transform length n")
ax2.set_ylabel("Ratio")
ax2.set_title("Relative to FLINT")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

out = bench_dir / "ntt_singleprime_compare.png"
fig.savefig(out, dpi=170)
print(out)

