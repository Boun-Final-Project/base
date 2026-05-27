"""Aggregate stats on a wind library: which cases succeeded, summary stats,
flag degenerate fields (near-zero, single-direction, etc.).

Usage:
    python library_stats.py --library-dir $CFD_DATA_ROOT/library_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np


def field_stats(field, free_mask):
    speeds = np.linalg.norm(field, axis=-1)
    free_speeds = speeds[free_mask]
    dirs = np.arctan2(field[..., 1], field[..., 0])
    free_dirs = dirs[free_mask]
    sin_m = np.mean(np.sin(free_dirs))
    cos_m = np.mean(np.cos(free_dirs))
    R = np.sqrt(sin_m**2 + cos_m**2)
    circ_std = float(np.sqrt(-2 * np.log(R))) if R > 1e-8 else float('inf')
    return {
        'mean_speed': float(free_speeds.mean()),
        'std_speed': float(free_speeds.std()),
        'max_speed': float(free_speeds.max()),
        'circ_std': circ_std,
    }


def is_degenerate(s):
    if s['mean_speed'] < 0.05: return 'near-zero wind'
    if s['std_speed'] < 0.02: return 'uniform wind (no spatial variance)'
    if s['circ_std'] < 0.3:   return 'single-direction (no recirculation)'
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library-dir', required=True)
    p.add_argument('--write-summary', action='store_true',
                   help='Write summary.json with per-case stats + reject list')
    args = p.parse_args()

    lib = Path(args.library_dir)
    manifest = json.loads((lib / 'manifest.json').read_text())

    n_total = len(manifest)
    n_complete = 0
    n_degenerate = 0
    summary = []
    for entry in manifest:
        case = Path(entry['case_dir'])
        wind_path = case / 'wind_field.npz'
        if not wind_path.exists():
            summary.append({**entry, 'status': 'incomplete'})
            continue
        n_complete += 1
        d = np.load(wind_path)
        field = d['field']
        grid = np.load(case / 'grid.npz')['grid']
        free = (grid == 0)
        s = field_stats(field, free)
        reason = is_degenerate(s)
        if reason:
            n_degenerate += 1
        summary.append({**entry, 'status': 'ok' if reason is None else 'degenerate',
                        'reject_reason': reason, **s})

    print(f"Library: {lib}")
    print(f"  total cases:    {n_total}")
    print(f"  completed:      {n_complete}  ({100*n_complete/n_total:.0f}%)")
    print(f"  degenerate:     {n_degenerate}  ({100*n_degenerate/max(n_complete,1):.0f}% of complete)")
    print(f"  usable:         {n_complete - n_degenerate}")

    ok = [s for s in summary if s.get('status') == 'ok']
    if ok:
        ms = np.array([s['mean_speed'] for s in ok])
        ss = np.array([s['std_speed'] for s in ok])
        cs = np.array([s['circ_std'] for s in ok])
        print(f"  good-case stats (n={len(ok)}):")
        print(f"    mean |U|:    {ms.mean():.3f} ± {ms.std():.3f}  [{ms.min():.3f}, {ms.max():.3f}]")
        print(f"    speed std:   {ss.mean():.3f} ± {ss.std():.3f}  [{ss.min():.3f}, {ss.max():.3f}]")
        print(f"    circ std:    {cs.mean():.3f} ± {cs.std():.3f}  [{cs.min():.3f}, {cs.max():.3f}]")
        print(f"    GADEN target: speed std 0.29-0.65, circ std 1.02-2.00")

    if args.write_summary:
        (lib / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f"Wrote {lib / 'summary.json'}")


if __name__ == '__main__':
    main()
