"""Thin gym.Env wrapper that always resets from a CFD library.

The training VecEnv calls `env.reset()` (no options) on auto-reset, so to
swap the training distribution we need to intercept reset() and inject
`options={"map_data": ..., "wind_field": ...}` from the library.

Drop-in replacement: build the wrapper inside the env factory (`make_env`)
and the rest of the training code is unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


class CFDLibraryEnv:
    """Wraps a GasSourceEnv and reroutes reset() to a CFD library sample.

    If `mix_synthetic` > 0, that fraction of resets falls back to the env's
    original procedural-map + synthetic-wind reset (no options). This is a
    safety valve so the policy still sees the original distribution if the
    library is small.
    """

    def __init__(self, env, sampler, mix_synthetic: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self._env = env
        self._sampler = sampler
        self._mix = float(mix_synthetic)
        self._rng = rng or np.random.default_rng()

    # All non-reset attributes pass through to the wrapped env.
    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *, seed=None, options=None):
        # Honor explicit options (e.g. eval-time GADEN map_data) — do not override.
        if options is not None and ("map_data" in options or "wind_field" in options):
            return self._env.reset(seed=seed, options=options)
        # Stochastic mix with original distribution.
        if self._mix > 0.0 and self._rng.random() < self._mix:
            return self._env.reset(seed=seed)
        # Default: sample from the library.
        lib_opts = self._sampler.sample()
        if options:
            lib_opts = {**options, **lib_opts}
        return self._env.reset(seed=seed, options=lib_opts)

    def step(self, action):
        return self._env.step(action)


def make_cfd_env(seed: int, rank: int, library_dir: str,
                 rl_package_path: str,
                 mix_synthetic: float = 0.0,
                 template_id: Optional[int] = None):
    """Factory replacing train.py's `make_env`. Returns a thunk that
    constructs a CFDLibraryEnv ready for the VecEnv."""
    # Lazy imports because this module is also used outside training contexts.
    if rl_package_path not in sys.path:
        sys.path.insert(0, rl_package_path)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    from reinforcement_learning.envs.gas_source_env import GasSourceEnv
    from cfd_library_loader import CFDLibrarySampler

    def _init():
        env = GasSourceEnv(template_id=template_id)
        rng = np.random.default_rng(seed + rank)
        sampler = CFDLibrarySampler(library_dir, rng, rl_package_path=rl_package_path)
        wrapped = CFDLibraryEnv(env, sampler, mix_synthetic=mix_synthetic, rng=rng)
        wrapped.reset(seed=seed + rank)
        return wrapped

    return _init
