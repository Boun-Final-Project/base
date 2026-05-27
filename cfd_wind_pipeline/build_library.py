"""Generate a batch of OpenFOAM case directories for the wind library.

For each case:
1. Sample procedural map (template_id, seed) and valid inlet/outlet openings
   using placement.sample_map_with_openings.
2. Shell out to gen_case.py to write the full case dir.
3. Append to manifest.json.

The cases are written but NOT executed. Run them with:
    sbatch sbatch/run_array.sh <library-dir>
which uses SLURM array job indexing into the manifest.

Usage:
    python build_library.py --library-dir $CFD_DATA_ROOT/library_v1 \\
        --n-cases 200 --templates 5,6,7,8,10 \\
        --inlet-speeds 0.3,0.5,0.7 --start-seed 1000
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PYTHON_BIN
from placement import sample_map_with_openings


def opening_str(specs, side):
    """Format openings on a given side as gen_case.py expects: 'lo-hi,lo-hi'."""
    parts = [f"{o.lo:.3f}-{o.hi:.3f}" for o in specs if o.side == side]
    return ','.join(parts)


def build_one(case_dir: Path, template_id: int, seed: int, inlet_speed: float,
              gen_case_script: Path, bg_cells_per_meter: float):
    """Try to generate a single case. Returns dict with params, or None on failure."""
    # Force inlet_side='west' (W↔E placements only): gen_case uses fixed
    # inlet/outlet patches on x=0 / x=W bg-mesh faces, so S↔N openings would
    # leave those patches empty → simpleFoam pRefCell crash.
    map_data, ops = sample_map_with_openings(template_id, seed, inlet_side='west')
    if ops is None:
        return None

    cmd = [
        PYTHON_BIN, str(gen_case_script),
        '--out-dir', str(case_dir),
        '--template-id', str(template_id),
        '--seed', str(seed),
        '--inlet-speed', f"{inlet_speed:.3f}",
        '--bg-cells-per-meter', f"{bg_cells_per_meter:.2f}",
        '--openings-west', opening_str(ops, 'west'),
        '--openings-east', opening_str(ops, 'east'),
        '--openings-south', opening_str(ops, 'south'),
        '--openings-north', opening_str(ops, 'north'),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  gen_case failed for {case_dir.name}: {result.stderr[-500:]}")
        return None

    return {
        'case_dir': str(case_dir),
        'template_id': template_id,
        'seed': seed,
        'inlet_speed': inlet_speed,
        'openings': [
            {'side': o.side, 'lo': o.lo, 'hi': o.hi, 'role': o.role}
            for o in ops
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library-dir', required=True,
                   help='Output directory; cases written as case_NNNN/')
    p.add_argument('--n-cases', type=int, default=100)
    p.add_argument('--templates', type=str, default='5,6,7,8,10',
                   help='Comma-separated template IDs to sample from')
    p.add_argument('--inlet-speeds', type=str, default='0.3,0.5,0.7',
                   help='Comma-separated inlet speed values [m/s]')
    p.add_argument('--start-seed', type=int, default=1000)
    p.add_argument('--bg-cells-per-meter', type=float, default=2.0,
                   help='Background mesh resolution. 2.0 gives ~5cm cells, '
                        '7x faster than 4.0 with ~3-5%% accuracy delta.')
    args = p.parse_args()

    templates = [int(x) for x in args.templates.split(',')]
    speeds = [float(x) for x in args.inlet_speeds.split(',')]

    lib = Path(args.library_dir)
    lib.mkdir(parents=True, exist_ok=True)
    gen_case_script = Path(__file__).parent / 'gen_case.py'

    rng = np.random.default_rng(args.start_seed)

    manifest = []
    attempt = 0
    case_idx = 0
    t0 = time.time()
    while case_idx < args.n_cases:
        template_id = int(rng.choice(templates))
        seed = args.start_seed + attempt
        inlet_speed = float(rng.choice(speeds))
        attempt += 1

        case_dir = lib / f"case_{case_idx:04d}"
        if case_dir.exists():
            case_idx += 1
            continue

        entry = build_one(case_dir, template_id, seed, inlet_speed, gen_case_script,
                          args.bg_cells_per_meter)
        if entry is not None:
            entry['bg_cells_per_meter'] = args.bg_cells_per_meter
        if entry is None:
            print(f"[{case_idx}/{args.n_cases}] skip: T{template_id} seed={seed} (no valid openings)")
            continue
        manifest.append(entry)
        case_idx += 1
        if case_idx % 10 == 0 or case_idx == args.n_cases:
            elapsed = time.time() - t0
            rate = case_idx / max(elapsed, 1e-6)
            eta = (args.n_cases - case_idx) / max(rate, 1e-6)
            print(f"[{case_idx}/{args.n_cases}] built (attempts={attempt}, "
                  f"rate={rate:.2f}/s, eta={eta:.0f}s)")

    (lib / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest.json with {len(manifest)} entries to {lib}")
    print(f"Submit with:  sbatch --array=0-{len(manifest)-1}%24 "
          f"{Path(__file__).parent}/sbatch/run_array.sh {lib}")


if __name__ == '__main__':
    main()
