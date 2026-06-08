"""
Plot return, success-rate, and wall-distance curves from nav checkpoint evaluation.

Output files (all saved to --output-dir):
  overall_comparison.png      mean ± std across all 60 envs, all runs
  per_size_return.png         3-subplot grid (small / medium / large)
  per_size_success.png        3-subplot grid
  per_size_wall_dist.png      3-subplot grid — wall-following quality vs step
                              (dashed reference line at WALL_TARGET_DIST = 0.5 m)

Usage:
    python3 plot_nav_curves.py --results-dir eval_results/
    python3 plot_nav_curves.py --results eval_results/nav_lidar.npz --labels "run 1"
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
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "eval_plots", "nav")

WALL_TARGET_DIST = 0.50   # metres — reference line on wall_dist plots

SIZE_NAMES = {
    0: "Small (<120 m²)",
    1: "Medium (120–190 m²)",
    2: "Large (≥190 m²)",
    3: "Extra-large (22–25×16–20 m) [extrap]",
}


# ---------------------------------------------------------------------------
# Helpers (mirrored from plot_return_curves.py)
# ---------------------------------------------------------------------------

def _plot_metric(ax, steps, mean, std, label, color):
    """Line with ± std shading."""
    ax.plot(steps, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)


def _finalise(ax, ylabel, ylim=None):
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(paths, labels):
    runs = []
    for path, label in zip(paths, labels):
        data = np.load(path)
        runs.append({
            "label":         label,
            "steps":         data["steps"],
            "returns":       data["returns"],        # (n_ckpts, n_envs)
            "successes":     data["successes"],      # (n_ckpts, n_envs)
            "ep_lengths":    data["ep_lengths"],     # (n_ckpts, n_envs)
            "avg_wall_dists":data["avg_wall_dists"], # (n_ckpts, n_envs)
            "size_cats":     data["size_cats"],      # (n_envs,)
        })
    return runs


# ---------------------------------------------------------------------------
# Plot: overall (return + success, side by side)
# ---------------------------------------------------------------------------

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
                     run["successes"].mean(axis=1).astype(float),
                     run["successes"].std(axis=1).astype(float),
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


# ---------------------------------------------------------------------------
# Plot: per-size grid (3 subplots in one figure)
# ---------------------------------------------------------------------------

def plot_per_size_grid(runs, metric, output_dir):
    """metric: 'return' | 'success' | 'wall_dist'"""
    if metric == "return":
        ylabel = "Mean Return"
        ylim   = None
    elif metric == "success":
        ylabel = "Success Rate"
        ylim   = (0, 1)
    else:
        ylabel = "Avg Wall Distance (m)"
        ylim   = (0, None)

    n_cats = len(SIZE_NAMES)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5))
    fig.suptitle(f"{ylabel} — Per Room Size", fontsize=14, fontweight="bold")
    colors = plt.cm.tab10.colors

    for size_id, ax in enumerate(axes):
        ax.set_title(SIZE_NAMES[size_id], fontweight="bold")

        if metric == "wall_dist":
            ax.axhline(WALL_TARGET_DIST, color="gray", linestyle="--",
                       linewidth=1.2, label=f"target {WALL_TARGET_DIST:.2f} m")

        for i, run in enumerate(runs):
            mask = run["size_cats"] == size_id
            if not mask.any():
                continue

            if metric == "return":
                data = run["returns"][:, mask]
            elif metric == "success":
                data = run["successes"][:, mask].astype(float)
            else:
                data = run["avg_wall_dists"][:, mask]

            _plot_metric(ax, run["steps"],
                         data.mean(axis=1), data.std(axis=1),
                         run["label"], colors[i % len(colors)])

        _finalise(ax, ylabel, ylim=ylim)

    fig.tight_layout()
    path = os.path.join(output_dir, f"per_size_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    for metric in ("return", "success", "wall_dist"):
        plot_per_size_grid(runs, metric, args.output_dir)

    # 1 overall + 3 per-size = 4 files
    print(f"\nDone. 4 figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
