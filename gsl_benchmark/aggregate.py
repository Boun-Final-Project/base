#!/usr/bin/env python3
"""Re-aggregate a gsl_bench results directory with 95% confidence intervals.

`ros2 run gsl_bench eval` already writes report.md + results.csv at the end of a
sweep. Use this when you want to (re)aggregate an *existing* results tree — e.g.
one produced before the 5x-budget labelling existed, or to recompute after a
different success budget.

    source ../../install/setup.bash        # puts gsl_bench on PYTHONPATH
    python3 aggregate.py <results_dir>
    python3 aggregate.py <results_dir> --relabel        # apply 5x budget to old runs
    python3 aggregate.py <results_dir> --relabel --success-budget-multiplier 3

With --relabel each run's result.json is rewritten: `reached_source` = raw reach,
`success` = reached within (multiplier x oracle distance). Without it, the runs
are aggregated as-is (they already carry the labels from the eval run).
"""
import argparse
import json
from pathlib import Path

from gsl_bench.eval import metrics, report

HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE = HERE / 'oracle_budgets_nav2_reliable.json'


def relabel_dir(root: Path, oracle: dict, multiplier: float) -> int:
    """Overwrite success in every result.json under `root` with the budget verdict."""
    n = 0
    for p in sorted(root.rglob('result.json')):
        try:
            res = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        reached = bool(res.get('reached_source', res.get('success')))
        res['reached_source'] = reached
        rec = oracle.get(f"{res.get('scenario')}_{res.get('sim')}")
        if rec is None:
            res['success_within_budget'] = reached
            res['success'] = reached
        else:
            budget = multiplier * float(rec['travel_distance_m'])
            travel = res.get('travel_distance_m')
            within = travel is not None and travel <= budget
            res['oracle_travel_distance_m'] = round(float(rec['travel_distance_m']), 3)
            res['success_budget_distance_m'] = round(budget, 3)
            res['success_budget_multiplier'] = multiplier
            res['success_within_budget'] = bool(reached and within)
            res['success'] = res['success_within_budget']
        p.write_text(json.dumps(res, indent=2))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(prog='gsl_benchmark aggregate')
    ap.add_argument('results_dir', type=Path)
    ap.add_argument('--relabel', action='store_true',
                    help='rewrite success in each result.json using the 5x oracle budget')
    ap.add_argument('--oracle-budgets', type=Path, default=DEFAULT_ORACLE)
    ap.add_argument('--success-budget-multiplier', type=float, default=5.0)
    ap.add_argument('--no-write', action='store_true',
                    help='print the report but do not write report.md / results.csv')
    args = ap.parse_args()

    if args.relabel:
        oracle = json.loads(args.oracle_budgets.read_text())
        n = relabel_dir(args.results_dir, oracle, args.success_budget_multiplier)
        print(f'relabelled {n} runs against {args.success_budget_multiplier}x oracle\n')

    if args.no_write:
        print(report.build_report(args.results_dir))
    else:
        print(report.write_all(args.results_dir))

    runs = metrics.load_runs(args.results_dir)
    o = metrics.aggregate(runs)['overall']
    ci = o.get('success_ci')
    band = f' (95% CI {ci[0] * 100:.0f}-{ci[1] * 100:.0f}%)' if ci else ''
    print(f"Overall success: {o['n_success']}/{o['n']} "
          f"({o['success_rate'] * 100:.0f}%){band}  |  "
          f"reached: {o.get('n_reached', o['n_success'])}/{o['n']}")


if __name__ == '__main__':
    main()
