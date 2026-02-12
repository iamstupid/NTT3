#!/usr/bin/env python3
"""
Plot benchmark results from bench_extended.cpp CSV output.
Usage: python plot_bench.py [bench_results.csv]
"""

import sys
import csv
import os
from collections import defaultdict

def load_csv(path):
    """Load CSV into dict of benchmark_name -> list of rows."""
    data = defaultdict(list)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['n1'] = int(row['n1'])
            row['n2'] = int(row['n2'])
            row['zint_ns'] = float(row['zint_ns'])
            row['gmp_ns'] = float(row['gmp_ns'])
            row['ratio'] = float(row['ratio'])
            data[row['benchmark']].append(row)
    return data

def fmt_size(n):
    if n >= 1048576: return f"{n // 1048576}M"
    if n >= 1024: return f"{n // 1024}K"
    return str(n)

def fmt_ns(ns):
    if ns < 1e3: return f"{ns:.0f} ns"
    if ns < 1e6: return f"{ns/1e3:.1f} us"
    if ns < 1e9: return f"{ns/1e6:.2f} ms"
    return f"{ns/1e9:.3f} s"

def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator
        return plt, FuncFormatter, LogLocator
    except ImportError:
        print("matplotlib not found. Install with: pip install matplotlib")
        print("Generating text summary instead.")
        return None, None, None

def plot_ratio(plt, FuncFormatter, LogLocator, rows, title, filename, xlabel='n (limbs)'):
    """Plot ratio (zint/GMP) vs size. Ratio < 1 = zint faster."""
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = [r['n1'] for r in rows]
    ratios = [r['ratio'] for r in rows]

    ax.plot(xs, ratios, 'o-', color='#2196F3', markersize=4, linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.fill_between(xs, ratios, 1.0,
                    where=[r < 1.0 for r in ratios],
                    alpha=0.15, color='green', interpolate=True)
    ax.fill_between(xs, ratios, 1.0,
                    where=[r > 1.0 for r in ratios],
                    alpha=0.15, color='red', interpolate=True)

    ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Ratio (zint / GMP)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_size(int(x))))

    # Add annotations
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    min_idx = ratios.index(min_ratio)
    max_idx = ratios.index(max_ratio)

    ax.annotate(f'Best: {min_ratio:.2f}x @ {fmt_size(xs[min_idx])}',
                xy=(xs[min_idx], min_ratio), fontsize=8,
                xytext=(10, -15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold')

    if max_ratio > 1.1:
        ax.annotate(f'Worst: {max_ratio:.2f}x @ {fmt_size(xs[max_idx])}',
                    xy=(xs[max_idx], max_ratio), fontsize=8,
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red')

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_absolute(plt, FuncFormatter, LogLocator, rows, title, filename, xlabel='n (limbs)'):
    """Plot absolute times for both libraries."""
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = [r['n1'] for r in rows]
    zint_ns = [r['zint_ns'] for r in rows]
    gmp_ns = [r['gmp_ns'] for r in rows]

    ax.plot(xs, zint_ns, 'o-', color='#2196F3', markersize=3, linewidth=1.5, label='zint')
    ax.plot(xs, gmp_ns, 's-', color='#FF5722', markersize=3, linewidth=1.5, label='GMP')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Time (ns)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_size(int(x))))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: fmt_ns(y) if y > 0 else '0'))

    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_unbalanced_heatmap(plt, FuncFormatter, LogLocator, rows, filename):
    """Plot unbalanced multiply ratios as a grouped bar chart by large operand size."""
    from collections import defaultdict

    # Group by large operand (n1)
    by_large = defaultdict(list)
    for r in rows:
        by_large[r['n1']].append(r)

    fig, ax = plt.subplots(figsize=(12, 6))

    large_sizes = sorted(by_large.keys())
    colors = plt.cm.viridis_r([0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    all_ratios_set = set()
    for sz in large_sizes:
        for r in by_large[sz]:
            all_ratios_set.add(r['n1'] // r['n2'])
    ratio_vals = sorted(all_ratios_set)

    bar_width = 0.8 / max(len(ratio_vals), 1)
    x_pos = list(range(len(large_sizes)))

    for i, rv in enumerate(ratio_vals):
        ys = []
        for sz in large_sizes:
            matching = [r for r in by_large[sz] if r['n1'] // r['n2'] == rv]
            ys.append(matching[0]['ratio'] if matching else 0)
        offsets = [x + i * bar_width for x in x_pos]
        color = colors[i % len(colors)]
        ax.bar(offsets, ys, bar_width, label=f'1:{rv}', color=color, edgecolor='white', linewidth=0.5)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Large operand (limbs)', fontsize=11)
    ax.set_ylabel('Ratio (zint / GMP)', fontsize=11)
    ax.set_title('Unbalanced Multiply: zint/GMP Ratio by Size and Imbalance', fontsize=13, fontweight='bold')
    ax.set_xticks([x + bar_width * len(ratio_vals) / 2 for x in x_pos])
    ax.set_xticklabels([fmt_size(s) for s in large_sizes], fontsize=9)
    ax.legend(title='Ratio (an:bn)', fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_combined_overview(plt, FuncFormatter, LogLocator, data, filename):
    """Combined ratio plot for all balanced operations on one chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bench_configs = [
        ('mul_balanced', 'Multiply', '#2196F3', 'o'),
        ('sqr', 'Squaring', '#4CAF50', 's'),
        ('div', 'Division', '#FF9800', '^'),
        ('bigint_mul', 'BigInt Mul', '#9C27B0', 'D'),
    ]

    for bench_name, label, color, marker in bench_configs:
        if bench_name not in data:
            continue
        rows = data[bench_name]
        xs = [r['n1'] for r in rows]
        ratios = [r['ratio'] for r in rows]
        ax.plot(xs, ratios, f'{marker}-', color=color, markersize=4,
                linewidth=1.5, label=label, alpha=0.85)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('n (limbs)', fontsize=11)
    ax.set_ylabel('Ratio (zint / GMP)', fontsize=11)
    ax.set_title('zint vs GMP: All Operations (ratio < 1 = zint faster)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_size(int(x))))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")

def text_summary(data):
    """Print text summary when matplotlib is not available."""
    for bench_name, rows in sorted(data.items()):
        print(f"\n=== {bench_name} ===")
        print(f"  {'n1':>10}  {'n2':>10}  {'zint':>12}  {'GMP':>12}  {'Ratio':>8}")
        for r in rows:
            print(f"  {r['n1']:>10}  {r['n2']:>10}  {fmt_ns(r['zint_ns']):>12}  "
                  f"{fmt_ns(r['gmp_ns']):>12}  {r['ratio']:>7.2f}x")

        ratios = [r['ratio'] for r in rows]
        print(f"  Range: {min(ratios):.2f}x - {max(ratios):.2f}x")

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'bench_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run bench_extended first.")
        return 1

    print(f"Loading {csv_path}...")
    data = load_csv(csv_path)
    print(f"Loaded {sum(len(v) for v in data.values())} data points across {len(data)} benchmarks")

    plt, FuncFormatter, LogLocator = try_import_matplotlib()
    if plt is None:
        text_summary(data)
        return 0

    out_dir = 'plots'
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots in {out_dir}/...")

    # Individual ratio plots
    plot_configs = [
        ('addmul_1', 'addmul_1: zint/GMP Ratio', 'addmul1_ratio.png'),
        ('mul_balanced', 'Balanced Multiply: zint/GMP Ratio', 'mul_balanced_ratio.png'),
        ('sqr', 'Squaring: zint/GMP Ratio', 'sqr_ratio.png'),
        ('div', 'Division (2n/n): zint/GMP Ratio', 'div_ratio.png'),
        ('bigint_mul', 'BigInt Multiply: zint/GMP Ratio', 'bigint_mul_ratio.png'),
        ('to_string', 'to_string: zint/GMP Ratio', 'to_string_ratio.png'),
        ('from_string', 'from_string: zint/GMP Ratio', 'from_string_ratio.png'),
    ]

    for bench_name, title, fname in plot_configs:
        if bench_name not in data:
            print(f"  Skipping {bench_name} (no data)")
            continue
        plot_ratio(plt, FuncFormatter, LogLocator, data[bench_name],
                   title, os.path.join(out_dir, fname))

    # Absolute time plots
    abs_configs = [
        ('mul_balanced', 'Balanced Multiply: Absolute Time', 'mul_balanced_abs.png'),
        ('sqr', 'Squaring: Absolute Time', 'sqr_abs.png'),
        ('div', 'Division (2n/n): Absolute Time', 'div_abs.png'),
    ]

    for bench_name, title, fname in abs_configs:
        if bench_name not in data:
            continue
        plot_absolute(plt, FuncFormatter, LogLocator, data[bench_name],
                      title, os.path.join(out_dir, fname))

    # Unbalanced multiply
    if 'mul_unbalanced' in data:
        plot_unbalanced_heatmap(plt, FuncFormatter, LogLocator,
                                data['mul_unbalanced'],
                                os.path.join(out_dir, 'mul_unbalanced.png'))

    # Combined overview
    plot_combined_overview(plt, FuncFormatter, LogLocator, data,
                           os.path.join(out_dir, 'overview.png'))

    # Print summary table
    print("\n=== Summary ===")
    print(f"  {'Benchmark':<20}  {'Best':>10}  {'Worst':>10}  {'Median':>10}")
    print(f"  {'--------':<20}  {'----':>10}  {'-----':>10}  {'------':>10}")
    for bench_name in ['addmul_1', 'mul_balanced', 'mul_unbalanced', 'sqr',
                        'div', 'bigint_mul', 'to_string', 'from_string']:
        if bench_name not in data:
            continue
        ratios = sorted([r['ratio'] for r in data[bench_name]])
        median = ratios[len(ratios) // 2]
        print(f"  {bench_name:<20}  {min(ratios):>9.2f}x  {max(ratios):>9.2f}x  {median:>9.2f}x")

    print(f"\nAll plots saved to {out_dir}/")
    return 0

if __name__ == '__main__':
    sys.exit(main() or 0)
