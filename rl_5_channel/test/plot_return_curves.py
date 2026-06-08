"""
Plot return and success-rate curves from checkpoint evaluation results.

Output files (all saved to --output-dir):
  overall_comparison.png      mean ± std across all 100 envs, all agents
  per_template_return.png     6-subplot grid, one per template (all agents)
  per_template_success.png    6-subplot grid, one per template (all agents)
  template_<id>_return.png    individual figure per template — return
  template_<id>_success.png   individual figure per template — success rate

Usage:
    # Load all .npz files from a directory:
    python3 plot_return_curves.py --results-dir eval_results/

    # Load specific files with custom labels:
    python3 plot_return_curves.py \\
        --results eval_results/run1.npz eval_results/run2.npz \\
        --labels "Agent A" "Agent B"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_plots")

TEMPLATE_NAMES = [
    "Empty", "Single Wall", "U-Shape",
    "Three Walls", "Complex Maze", "Multi-Room",
]
SMOOTH_WINDOW = 3  # rolling-mean window applied to all curves


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _smooth(y, w):
    if w <= 1 or len(y) < w:
        return y, np.arange(len(y))
    kernel  = np.ones(w) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    indices  = np.arange(w - 1, len(y))
    return smoothed, indices


def _plot_metric(ax, steps, mean, std, label, color):
    """Thin raw line + smoothed line with ± std shading."""
    ax.plot(steps, mean, color=color, alpha=0.2, linewidth=0.7)
    sm, idx = _smooth(mean, SMOOTH_WINDOW)
    ss, _   = _smooth(std,  SMOOTH_WINDOW)
    s_steps = steps[idx]
    ax.plot(s_steps, sm, color=color, linewidth=1.8, label=label)
    ax.fill_between(s_steps, sm - ss, sm + ss, color=color, alpha=0.15)


def _finalise(ax, ylabel, ylim=None):
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_results(paths, labels):
    runs = []
    for path, label in zip(paths, labels):
        data = np.load(path)
        runs.append({
            "label":        label,
            "steps":        data["steps"],
            "returns":      data["returns"],       # (n_ckpts, n_envs)
            "successes":    data["successes"],     # (n_ckpts, n_envs)
            "template_ids": data["template_ids"],  # (n_envs,)
        })
    return runs


# ------------------------------------------------------------------
# Plot: overall
# ------------------------------------------------------------------

def plot_overall(runs, output_dir):
    fig, (ax_ret, ax_suc) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        c = colors[i % len(colors)]
        _plot_metric(ax_ret, run["steps"],
                     run["returns"].mean(axis=1),
                     run["returns"].std(axis=1),
                     run["label"], c)
        _plot_metric(ax_suc, run["steps"],
                     run["successes"].mean(axis=1),
                     run["successes"].std(axis=1),
                     run["label"], c)

    ax_ret.set_title("Return — all environments", fontweight="bold")
    ax_suc.set_title("Success Rate — all environments", fontweight="bold")
    _finalise(ax_ret, "Mean Return")
    _finalise(ax_suc, "Success Rate", ylim=(0, 1))

    fig.tight_layout()
    path = os.path.join(output_dir, "overall_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ------------------------------------------------------------------
# Plot: per-template grid (6 subplots in one figure)
# ------------------------------------------------------------------

def plot_per_template_grid(runs, metric, output_dir):
    ylabel = "Mean Return" if metric == "return" else "Success Rate"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{ylabel} — Per Template", fontsize=14, fontweight="bold")
    colors = plt.cm.tab10.colors

    for tid, ax in enumerate(axes.flatten()):
        ax.set_title(f"Template {tid}: {TEMPLATE_NAMES[tid]}", fontweight="bold")
        ylim = (0, 1) if metric == "success" else None

        for i, run in enumerate(runs):
            mask = run["template_ids"] == tid
            if not mask.any():
                continue
            data = (run["returns"][:, mask] if metric == "return"
                    else run["successes"][:, mask].astype(float))
            _plot_metric(ax, run["steps"],
                         data.mean(axis=1), data.std(axis=1),
                         run["label"], colors[i % len(colors)])

        _finalise(ax, ylabel, ylim=ylim)

    fig.tight_layout()
    path = os.path.join(output_dir, f"per_template_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ------------------------------------------------------------------
# Plot: one figure per template
# ------------------------------------------------------------------

def plot_individual_templates(runs, metric, output_dir):
    ylabel = "Mean Return" if metric == "return" else "Success Rate"
    colors = plt.cm.tab10.colors

    for tid in range(6):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(
            f"Template {tid}: {TEMPLATE_NAMES[tid]} — {ylabel}",
            fontweight="bold",
        )
        ylim = (0, 1) if metric == "success" else None

        for i, run in enumerate(runs):
            mask = run["template_ids"] == tid
            if not mask.any():
                continue
            data = (run["returns"][:, mask] if metric == "return"
                    else run["successes"][:, mask].astype(float))
            _plot_metric(ax, run["steps"],
                         data.mean(axis=1), data.std(axis=1),
                         run["label"], colors[i % len(colors)])

        _finalise(ax, ylabel, ylim=ylim)
        fig.tight_layout()
        path = os.path.join(output_dir, f"template_{tid}_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", nargs="+",
                       help="One or more .npz result files")
    group.add_argument("--results-dir", type=str,
                       help="Directory of .npz result files (loads all)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display name for each run (default: file stem)")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
                        help="Directory to write plots into")
    args = parser.parse_args()

    if args.results_dir:
        result_files = sorted(str(p) for p in Path(args.results_dir).glob("*.npz"))
        if not result_files:
            print(f"No .npz files found in {args.results_dir}")
            return
    else:
        result_files = args.results

    labels = args.labels or [Path(p).stem for p in result_files]
    if len(labels) != len(result_files):
        parser.error(f"--labels count ({len(labels)}) must match "
                     f"number of result files ({len(result_files)})")

    runs = load_results(result_files, labels)
    print(f"Loaded {len(runs)} run(s): {[r['label'] for r in runs]}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating plots → {args.output_dir}/")

    plot_overall(runs, args.output_dir)
    for metric in ("return", "success"):
        plot_per_template_grid(runs, metric, args.output_dir)
        plot_individual_templates(runs, metric, args.output_dir)

    # 1 overall + 2 grids + 12 individual = 15 files
    print(f"\nDone. 15 figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
