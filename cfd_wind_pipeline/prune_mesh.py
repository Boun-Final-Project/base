"""Prune OpenFOAM mesh artifacts from a library, keeping only what training needs.

Drops ~130MB/case → ~400KB/case (the wind_field.npz, grid.npz, meta.json).
After pruning, the cases can no longer be re-postprocessed — they'd need to
be regenerated from the manifest entry (~15 min/case).

Usage:
    python prune_mesh.py --library-dir $CFD_DATA_ROOT/library_v1 [--dry-run]
"""
import argparse
import shutil
import subprocess
from pathlib import Path


def prune_one(case_dir: Path, dry_run: bool):
    if not (case_dir / 'wind_field.npz').exists():
        return 0  # never pruned — case didn't finish
    bytes_freed = 0
    targets = ['constant', 'system', '0', '0.orig', 'postProcessing']
    for t in targets:
        p = case_dir / t
        if p.exists():
            bytes_freed += int(subprocess.check_output(['du', '-sb', str(p)]).split()[0])
            if not dry_run:
                shutil.rmtree(p)
    # Time-step dirs (numeric names): "100", "200", "50.5", etc.
    for sub in case_dir.iterdir():
        if sub.is_dir() and sub.name.replace('.', '', 1).isdigit():
            bytes_freed += int(subprocess.check_output(['du', '-sb', str(sub)]).split()[0])
            if not dry_run:
                shutil.rmtree(sub)
    return bytes_freed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library-dir', required=True)
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be freed; do not delete.')
    args = p.parse_args()

    lib = Path(args.library_dir)
    cases = sorted(lib.glob('case_*'))
    total = 0
    n_pruned = 0
    for c in cases:
        freed = prune_one(c, args.dry_run)
        if freed > 0:
            n_pruned += 1
            total += freed
    print(f"{'[DRY-RUN] would free' if args.dry_run else 'Freed'} "
          f"{total/1e9:.2f} GB across {n_pruned}/{len(cases)} cases.")


if __name__ == '__main__':
    main()
