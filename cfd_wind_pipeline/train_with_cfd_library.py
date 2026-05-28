"""Launch PPO training using a CFD wind library as the training distribution.

Monkey-patches the training script's `make_env` so the VecEnv builds
CFDLibraryEnv-wrapped envs that sample (map, wind) from `--library-dir`
at every reset. Otherwise this is an exact pass-through to the underlying
train.py main() — all of train.py's existing args still work.

Usage:
    python train_with_cfd_library.py \\
        --library-dir /comp04-storage/.../cfd_test/library_v2 \\
        --rl-package-path /comp04-storage/.../friend_base_loop_spatial \\
        --mix-synthetic 0.2 \\
        -- <any train.py args, e.g. --arch dual --num-envs 48 ...>

Args before `--` are this launcher's. Args after `--` go to train.py untouched.
"""
import argparse
import os
import sys


def main():
    # Split argv on the first '--' so the launcher's args don't collide
    # with train.py's argparse.
    argv = sys.argv[1:]
    if '--' in argv:
        i = argv.index('--')
        launcher_argv = argv[:i]
        passthrough = argv[i+1:]
    else:
        launcher_argv = argv
        passthrough = []

    p = argparse.ArgumentParser()
    p.add_argument('--library-dir', required=True,
                   help='CFD library dir, or comma-separated list to pool '
                        '(e.g. hard T4-9 + easy T0-3).')
    p.add_argument('--rl-package-path', required=True,
                   help='Path to the RL package providing train.py / GasSourceEnv')
    p.add_argument('--mix-synthetic', type=float, default=0.0,
                   help='Fraction of episodes that reset to procedural+synthetic '
                        '(default 0 = always use library)')
    p.add_argument('--template-filter', type=str, default=None,
                   help='Comma-separated template_ids to keep (e.g. "0,1,2,3,4,5" '
                        'to match the champ and avoid OOD template shock).')
    args = p.parse_args(launcher_argv)

    rl_pkg = os.path.abspath(args.rl_package_path)
    cfd_pkg = os.path.dirname(os.path.abspath(__file__))
    for p in (rl_pkg, cfd_pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import + patch the training module's env factory.
    from reinforcement_learning.training import train as train_mod
    from cfd_library_env import make_cfd_env

    library_dirs = [os.path.abspath(d) for d in args.library_dir.split(',')]
    mix = args.mix_synthetic
    tmpl_filter = ([int(x) for x in args.template_filter.split(',')]
                   if args.template_filter else None)

    def patched_make_env(seed, rank, template_id=None):
        return make_cfd_env(seed=seed, rank=rank,
                            library_dirs=library_dirs,
                            rl_package_path=rl_pkg,
                            mix_synthetic=mix,
                            template_id=template_id,
                            template_filter=tmpl_filter)

    train_mod.make_env = patched_make_env
    print(f"[train_with_cfd_library] Patched make_env → CFD libraries "
          f"({library_dirs}, mix_synthetic={mix}, template_filter={tmpl_filter})")

    # Hand off to train.main() with the remaining argv.
    sys.argv = [sys.argv[0]] + passthrough
    train_mod.main()


if __name__ == '__main__':
    main()
