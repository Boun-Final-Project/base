"""Aggregate a results directory into a markdown table + CSV.

    ros2 run gsl_bench report <results_dir>
"""
import argparse
import csv
import sys
from pathlib import Path

from gsl_bench.eval import metrics

COLUMNS = ['scenario', 'success (CI)', 'reached', 'TT (s) [CI]', 'TD (m) [CI]',
           'steps', 'fail modes']


def _fail_modes(summary: dict) -> str:
    bad = {k: v for k, v in summary['status_counts'].items() if k != 'success'}
    return ', '.join(f'{k}x{v}' for k, v in sorted(bad.items())) or '—'


def _rate(k: int, n: int, ci) -> str:
    """e.g. '3/5 (30–79%)' — count plus 95% Wilson band."""
    s = f'{k}/{n}'
    if ci:
        s += f' ({ci[0] * 100:.0f}–{ci[1] * 100:.0f}%)'
    return s


def _val(v, ci) -> str:
    """e.g. '42.1 [38–47]' — mean plus 95% bootstrap band."""
    if v is None:
        return '—'
    s = f'{v:.1f}'
    if ci:
        s += f' [{ci[0]:.0f}–{ci[1]:.0f}]'
    return s


def _row(name: str, s: dict) -> list:
    return [
        name,
        _rate(s['n_success'], s['n'], s.get('success_ci')),
        _rate(s.get('n_reached', s['n_success']), s['n'], s.get('reached_ci')),
        _val(s['mean_time_s'], s.get('time_ci')),
        _val(s['mean_distance_m'], s.get('distance_ci')),
        '—' if s['mean_steps'] is None else f"{s['mean_steps']:.0f}",
        _fail_modes(s),
    ]


def build_report(root: Path) -> str:
    runs = metrics.load_runs(root)
    if not runs:
        return f'No result.json files under {root}\n'
    agg = metrics.aggregate(runs)

    lines = []
    agents = sorted({r.get('agent', '?') for r in runs})
    lines.append(f"# gsl_bench report — {', '.join(agents)}")
    lines.append('')
    lines.append(f'Runs: {agg["overall"]["n"]}  |  Source: `{root}`')
    lines.append('')

    if agg['harness_mismatch']:
        lines.append('> **WARNING: mixed harness configs.** These runs were NOT scored '
                     'under identical conditions, so the aggregate is not a fair '
                     'comparison. Differing keys: '
                     f'`{", ".join(agg["harness_mismatch"])}`')
        lines.append('')

    lines.append('| ' + ' | '.join(COLUMNS) + ' |')
    lines.append('|' + '|'.join(['---'] * len(COLUMNS)) + '|')
    for scen, s in agg['per_scenario'].items():
        lines.append('| ' + ' | '.join(_row(scen, s)) + ' |')
    lines.append('| **TOTAL** | ' + ' | '.join(_row('', agg['overall'])[1:]) + ' |')
    lines.append('')

    h = next((r.get('harness') for r in runs if r.get('harness')), None)
    if h and not agg['harness_mismatch']:
        lines.append('Harness: ' + ', '.join(f'{k}={v}' for k, v in sorted(h.items())))
        lines.append('')
    lines.append('success = reached the source within 5x the oracle distance; '
                 'reached = reached it at all (up to the 10x/20x termination budget). '
                 'TT/TD = mean time / traveled distance over successes. '
                 'Bands are 95% CIs (Wilson for rates, bootstrap for means).')
    lines.append('')
    return '\n'.join(lines)


def write_csv(root: Path, path: Path) -> None:
    runs = metrics.load_runs(root)
    fields = ['scenario', 'sim', 'agent', 'status', 'success', 'reached_source',
              'success_within_budget', 'steps', 'sim_time_s',
              'wall_time_s', 'travel_distance_m', 'final_distance_m',
              'oracle_travel_distance_m', 'success_budget_distance_m',
              'escapes', 'clamped_steps', 'drive_timeouts']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in runs:
            w.writerow(r)


def write_all(root: Path) -> str:
    """Write results.csv + report.md into root and return the report text."""
    text = build_report(root)
    (Path(root) / 'report.md').write_text(text)
    write_csv(Path(root), Path(root) / 'results.csv')
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gsl_bench report')
    ap.add_argument('results_dir', type=Path)
    ap.add_argument('--write', action='store_true',
                    help='also write report.md and results.csv into the directory')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.write:
        print(write_all(args.results_dir))
    else:
        print(build_report(args.results_dir))


if __name__ == '__main__':
    main()
