#!/usr/bin/env python3
"""
ObjectNav Results Analyzer
==========================

Compare evaluation runs, generate ablation tables, analyze failures.

Usage:
    python -m eval.analyze results/val_baseline_100ep.json results/val_clip_100ep.json
    python -m eval.analyze results/*.json --latex
    python -m eval.analyze results/val_clip_100ep.json --failures
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_pct(v: float) -> str:
    return f"{v * 100:.1f}"


def format_f(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


# ============================================================
# Comparison Table
# ============================================================

def print_comparison(results_list: List[dict], names: List[str]):
    """Print side-by-side comparison of multiple runs."""

    metrics = ['sr', 'spl', 'soft_spl', 'mean_dts', 'mean_steps', 'mean_time']
    labels = ['SR (%)', 'SPL', 'SoftSPL', 'Mean DTS (m)', 'Mean Steps', 'Mean Time (s)']

    # Header
    col_w = 14
    name_w = max(len(n) for n in names) + 2
    print()
    print("=" * (name_w + col_w * len(names) + 4))
    print("ABLATION COMPARISON")
    print("=" * (name_w + col_w * len(names) + 4))
    print()

    header = f"{'Metric':<{name_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))

    for metric, label in zip(metrics, labels):
        row = f"{label:<{name_w}}"
        for r in results_list:
            agg = r.get('aggregate', {})
            v = agg.get(metric, 0)
            if metric == 'sr':
                row += f"{format_pct(v) + '%':>{col_w}}"
            elif metric in ('mean_dts', 'mean_time'):
                row += f"{format_f(v, 2):>{col_w}}"
            elif metric == 'mean_steps':
                row += f"{int(v):>{col_w}}"
            else:
                row += f"{format_f(v):>{col_w}}"
        print(row)

    # Episode counts
    row = f"{'Episodes':<{name_w}}"
    for r in results_list:
        row += f"{r.get('aggregate', {}).get('n', 0):>{col_w}}"
    print(row)

    print()


# ============================================================
# Per-Category Breakdown
# ============================================================

def print_category_comparison(results_list: List[dict], names: List[str]):
    """Per-category SR/SPL comparison across runs."""
    # Collect all categories
    all_cats = set()
    for r in results_list:
        cats = r.get('aggregate', {}).get('per_category', {})
        all_cats.update(cats.keys())

    all_cats = sorted(all_cats)

    col_w = 12
    print("PER-CATEGORY SUCCESS RATE")
    print("-" * 60)

    header = f"{'Category':<15}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))

    for cat in all_cats:
        row = f"{cat:<15}"
        for r in results_list:
            cats = r.get('aggregate', {}).get('per_category', {})
            if cat in cats:
                sr = cats[cat].get('sr', 0)
                n = cats[cat].get('n', 0)
                row += f"{format_pct(sr) + f'% ({n})':>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    print()


# ============================================================
# Failure Analysis
# ============================================================

def analyze_failures(results: dict, name: str = ""):
    """Detailed failure analysis for a single run."""
    episodes = results.get('episodes', [])
    if not episodes:
        print("No episode data available.")
        return

    print(f"\nFAILURE ANALYSIS: {name or results.get('policy', '?')}")
    print("=" * 70)

    # Categorize outcomes
    successes = [e for e in episodes if e.get('success')]
    false_stops = [e for e in episodes
                   if not e.get('success')
                   and e.get('termination') == 'stop']
    max_steps = [e for e in episodes
                 if not e.get('success')
                 and e.get('termination') == 'max_steps']
    errors = [e for e in episodes
              if not e.get('success')
              and e.get('termination') not in ('stop', 'max_steps')]

    total = len(episodes)
    print(f"\n  Outcomes ({total} episodes):")
    print(f"    Success:      {len(successes):3d} ({len(successes)/total*100:.1f}%)")
    print(f"    False stop:   {len(false_stops):3d} ({len(false_stops)/total*100:.1f}%) ← agent stopped but wasn't at goal")
    print(f"    Max steps:    {len(max_steps):3d} ({len(max_steps)/total*100:.1f}%) ← ran out of steps")
    if errors:
        print(f"    Error/other:  {len(errors):3d} ({len(errors)/total*100:.1f}%)")

    # False stop analysis
    if false_stops:
        dts_values = [e['distance_to_goal'] for e in false_stops]
        print(f"\n  False stops (stopped but DTS > 1.0m):")
        print(f"    Mean DTS:  {sum(dts_values)/len(dts_values):.2f}m")
        print(f"    Min DTS:   {min(dts_values):.2f}m")
        print(f"    Max DTS:   {max(dts_values):.2f}m")
        print(f"    Steps <100: {sum(1 for e in false_stops if e['num_steps'] < 100)} "
              f"(premature stops)")

        # By category
        cat_counts = defaultdict(int)
        for e in false_stops:
            cat_counts[e.get('goal_object', '?')] += 1
        print(f"    By category: {dict(sorted(cat_counts.items()))}")

    # Max steps analysis (exploration failures)
    if max_steps:
        dts_values = [e['distance_to_goal'] for e in max_steps]
        print(f"\n  Max steps (exploration exhausted):")
        print(f"    Mean DTS:  {sum(dts_values)/len(dts_values):.2f}m")
        close = [e for e in max_steps if e['distance_to_goal'] < 2.0]
        if close:
            print(f"    Near-misses (DTS < 2m): {len(close)}")
            for e in close:
                print(f"      {e.get('goal_object', '?'):15s} DTS={e['distance_to_goal']:.2f}m")

    # Success analysis
    if successes:
        steps = [e['num_steps'] for e in successes]
        spl_vals = [e.get('spl', 0) for e in successes]
        print(f"\n  Successes:")
        print(f"    Mean steps: {sum(steps)/len(steps):.0f}")
        print(f"    Mean SPL:   {sum(spl_vals)/len(spl_vals):.3f}")
        by_cat = defaultdict(int)
        for e in successes:
            by_cat[e.get('goal_object', '?')] += 1
        print(f"    By category: {dict(sorted(by_cat.items()))}")

    print()


# ============================================================
# DTS Distribution
# ============================================================

def print_dts_distribution(results: dict, name: str = ""):
    """Show DTS distribution to understand how close the agent gets."""
    episodes = results.get('episodes', [])
    if not episodes:
        return

    dts_values = [e['distance_to_goal'] for e in episodes if 'distance_to_goal' in e]
    if not dts_values:
        return

    print(f"\nDTS DISTRIBUTION: {name}")
    print("-" * 40)

    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 999)]
    for lo, hi in bins:
        count = sum(1 for d in dts_values if lo <= d < hi)
        pct = count / len(dts_values) * 100
        bar = "█" * int(pct / 2)
        label = f"{lo}-{hi}m" if hi < 999 else f"{lo}m+"
        print(f"  {label:>6s}: {count:3d} ({pct:5.1f}%) {bar}")

    print()


# ============================================================
# LaTeX Table
# ============================================================

def print_latex_table(results_list: List[dict], names: List[str]):
    """Generate LaTeX table for thesis."""
    print()
    print("% LaTeX ablation table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{ObjectNav ablation results on HM3D val split}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Configuration & Episodes & SR (\\%) & SPL & SoftSPL & Mean DTS (m) \\\\")
    print("\\midrule")

    for name, r in zip(names, results_list):
        agg = r.get('aggregate', {})
        n = agg.get('n', 0)
        sr = agg.get('sr', 0) * 100
        spl = agg.get('spl', 0)
        sspl = agg.get('soft_spl', 0)
        dts = agg.get('mean_dts', 0)
        print(f"{name} & {n} & {sr:.1f} & {spl:.3f} & {sspl:.3f} & {dts:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Per-category table
    all_cats = set()
    for r in results_list:
        all_cats.update(r.get('aggregate', {}).get('per_category', {}).keys())
    all_cats = sorted(all_cats)

    ncols = len(names)
    col_spec = "l" + "cc" * ncols

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-category success rates on HM3D val split}")
    print("\\label{tab:per_category}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header = "Category"
    for n in names:
        header += f" & \\multicolumn{{2}}{{c}}{{{n}}}"
    header += " \\\\"
    print(header)

    subheader = ""
    for _ in names:
        subheader += " & SR (\\%) & SPL"
    subheader += " \\\\"
    print(subheader)
    print("\\midrule")

    for cat in all_cats:
        row = cat.replace('_', '\\_')
        for r in results_list:
            cats = r.get('aggregate', {}).get('per_category', {})
            if cat in cats:
                sr = cats[cat].get('sr', 0) * 100
                spl = cats[cat].get('spl', 0)
                row += f" & {sr:.1f} & {spl:.3f}"
            else:
                row += " & — & —"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


# ============================================================
# Episode-level CSV export
# ============================================================

def export_csv(results: dict, output_path: str):
    """Export per-episode results as CSV for external analysis."""
    episodes = results.get('episodes', [])
    if not episodes:
        print("No episodes to export.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fields = ['goal_object', 'success', 'spl', 'soft_spl', 'distance_to_goal',
              'num_steps', 'termination', 'elapsed_time']

    with open(output_path, 'w') as f:
        f.write(','.join(fields) + '\n')
        for e in episodes:
            row = []
            for field in fields:
                v = e.get(field, '')
                if isinstance(v, float):
                    v = f"{v:.4f}"
                row.append(str(v))
            f.write(','.join(row) + '\n')

    print(f"CSV exported to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze ObjectNav results")
    parser.add_argument('results', nargs='+', help="JSON result files to analyze")
    parser.add_argument('--latex', action='store_true', help="Generate LaTeX tables")
    parser.add_argument('--failures', action='store_true', help="Detailed failure analysis")
    parser.add_argument('--dts', action='store_true', help="DTS distribution")
    parser.add_argument('--csv', type=str, default=None, help="Export first result as CSV")
    parser.add_argument('--all', action='store_true', help="Show everything")
    args = parser.parse_args()

    # Load results
    results_list = []
    names = []
    for path in args.results:
        r = load_results(path)
        results_list.append(r)
        # Name from policy or filename
        name = r.get('policy', Path(path).stem)
        names.append(name)

    # Always show comparison
    print_comparison(results_list, names)
    print_category_comparison(results_list, names)

    if args.failures or args.all:
        for r, n in zip(results_list, names):
            analyze_failures(r, n)

    if args.dts or args.all:
        for r, n in zip(results_list, names):
            print_dts_distribution(r, n)

    if args.latex or args.all:
        print_latex_table(results_list, names)

    if args.csv:
        export_csv(results_list[0], args.csv)


if __name__ == '__main__':
    main()