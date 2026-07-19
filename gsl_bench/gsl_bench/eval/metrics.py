"""Per-run result.json -> per-scenario and overall metrics. Pure functions, no ROS.

TT = mean sim_time_s over SUCCESSFUL runs (time-to-source).
TD = mean travel_distance_m over SUCCESSFUL runs (traveled distance).
Both are only meaningful over successes: a failed run's time is just the step cap.
"""
import json
import random
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

# Fixed seed so the same result tree always yields the same CI bands (reports are
# regenerated and diffed; a non-deterministic bootstrap would churn them).
_BOOT_SEED = 20260718
_BOOT_ITERS = 2000


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n. Robust near 0% / 100%
    and at small n (5 runs), unlike the normal approximation. Returns (lo, hi)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def bootstrap_ci(vals, iters: int = _BOOT_ITERS):
    """95% bootstrap percentile CI for the mean of `vals`. Returns (lo, hi), or
    None when there are too few samples to resample meaningfully (<2)."""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(_BOOT_SEED)
    n = len(vals)
    means = []
    for _ in range(iters):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (round(lo, 2), round(hi, 2))


def load_runs(root: Path) -> List[dict]:
    runs = []
    for p in sorted(Path(root).rglob('result.json')):
        try:
            with open(p) as f:
                r = json.load(f)
            r['_path'] = str(p)
            runs.append(r)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 2) if vals else None


def _median(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(median(vals), 2) if vals else None


def _summarize(runs: List[dict]) -> dict:
    # `success` is the budget-gated verdict (reached within 5x oracle distance).
    # `reached_source` is the raw reach; on older result trees it may be absent,
    # in which case it defaults to `success` so pre-budget runs still summarize.
    succ = [r for r in runs if r.get('success')]
    fail = [r for r in runs if not r.get('success')]
    n = len(runs)
    n_reached = sum(1 for r in runs if r.get('reached_source', r.get('success')))
    steps = [r.get('steps') for r in succ]
    times = [r.get('sim_time_s') for r in succ]
    dists = [r.get('travel_distance_m') for r in succ]
    return {
        'n': n,
        'n_success': len(succ),
        'success_rate': round(len(succ) / n, 3) if n else 0.0,
        'success_ci': wilson_interval(len(succ), n),
        'n_reached': n_reached,
        'reached_rate': round(n_reached / n, 3) if n else 0.0,
        'reached_ci': wilson_interval(n_reached, n),
        'mean_steps': _mean(steps),
        'median_steps': _median(steps),
        'steps_ci': bootstrap_ci(steps),
        'mean_time_s': _mean(times),                                        # TT
        'time_ci': bootstrap_ci(times),
        'mean_distance_m': _mean(dists),                                    # TD
        'distance_ci': bootstrap_ci(dists),
        'mean_final_distance_m_failures': _mean(r.get('final_distance_m') for r in fail),
        'status_counts': dict(Counter(r.get('status', 'unknown') for r in runs)),
    }


def aggregate(runs: List[dict]) -> Dict[str, dict]:
    """{'per_scenario': {name: summary}, 'overall': summary, 'harness_mismatch': [...]}"""
    per_scenario: Dict[str, dict] = {}
    by_scenario: Dict[str, List[dict]] = {}
    for r in runs:
        by_scenario.setdefault(r.get('scenario', '?'), []).append(r)
    for scen in sorted(by_scenario):
        per_scenario[scen] = _summarize(by_scenario[scen])
    return {
        'per_scenario': per_scenario,
        'overall': _summarize(runs),
        'harness_mismatch': harness_mismatch(runs),
    }


def harness_mismatch(runs: List[dict]) -> List[str]:
    """Keys whose harness setting is not identical across all runs — the fairness guard.

    Comparing methods scored under different success radii, step caps, or with escape
    on for one and off for another is meaningless, so we make it loud instead of silent.
    """
    blocks = [r.get('harness', {}) for r in runs if r.get('harness')]
    if len(blocks) < 2:
        return []
    # These are oracle-derived per-scenario budgets: they are *supposed* to differ
    # across scenarios, so a difference here is not an unfairness (the multipliers
    # behind them are what must match, and those live in the run's budget fields).
    per_scenario = {'max_travel_distance_m', 'max_sim_time_s'}
    keys = set().union(*(b.keys() for b in blocks)) - per_scenario
    return sorted(k for k in keys if len({json.dumps(b.get(k), sort_keys=True)
                                          for b in blocks}) > 1)
