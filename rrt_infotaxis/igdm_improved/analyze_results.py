"""Aggregate per-run summary.json files and print comparison table.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir /path/to/results/comparison
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for summary_file in sorted(results_dir.glob("*/*/run_*/summary.json")):
        with open(summary_file) as f:
            summaries.append(json.load(f))
    return summaries


def analyze(summaries: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in summaries:
        key = (s['map'], s['threshold_mode'])
        groups[key].append(s)

    results = {}
    for key, runs in groups.items():
        n = len(runs)
        successes = [r for r in runs if r['success']]
        success_rate = len(successes) / n if n > 0 else float('nan')

        all_steps = [r['steps'] for r in runs]
        mean_steps_all = float(np.mean(all_steps)) if all_steps else float('nan')
        std_steps_all = float(np.std(all_steps, ddof=1)) if len(all_steps) > 1 else float('nan')

        success_steps = [r['steps'] for r in successes]
        mean_steps_success = float(np.mean(success_steps)) if success_steps else float('nan')
        std_steps_success = float(np.std(success_steps, ddof=1)) if len(success_steps) > 1 else float('nan')

        results[key] = {
            'n': n,
            'n_success': len(successes),
            'success_rate': success_rate,
            'mean_steps_all': mean_steps_all,
            'std_steps_all': std_steps_all,
            'mean_steps_success': mean_steps_success,
            'std_steps_success': std_steps_success,
        }
    return results


def print_table(results: dict) -> None:
    header = f"{'Map':<12} {'Mode':<10} {'N':>4} {'Success':>10} {'Steps (all)':>18} {'Steps (success)':>20}"
    print(header)
    print("-" * len(header))
    for (map_name, mode), r in sorted(results.items()):
        success_str = f"{r['n_success']}/{r['n']} ({r['success_rate']*100:.0f}%)"
        steps_all_str = f"{r['mean_steps_all']:.1f} ± {r['std_steps_all']:.1f}"
        steps_ok_str = f"{r['mean_steps_success']:.1f} ± {r['std_steps_success']:.1f}"
        print(f"{map_name:<12} {mode:<10} {r['n']:>4} {success_str:>10} {steps_all_str:>18} {steps_ok_str:>20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path(__file__).parent / "results" / "comparison",
        help="Root directory containing <map>/<mode>/run_*/ structure"
    )
    args = parser.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        print(f"No summary.json files found under {args.results_dir}")
        raise SystemExit(1)

    print(f"Loaded {len(summaries)} run summaries from {args.results_dir}\n")
    results = analyze(summaries)
    print_table(results)
