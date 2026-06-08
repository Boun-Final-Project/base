"""
Side-by-side visual comparison of wall-encoding options:
  GT (0.1 m)  |  OLD center-only  |  NEW max-pool  |  ALT mean-pool

Saves one figure per template to eval_results/wall_encodings/.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from rl_5_channel.envs.gas_source_env import GasSourceEnv
from rl_5_channel.envs.spatial_obs_wrapper import SpatialObsWrapper

_OUT_DIR = os.path.join(_SCRIPT_DIR, "obs_viz")

TEMPLATE_NAMES = ["empty", "single_wall", "u_shape", "three_walls",
                  "complex_maze", "multi_room"]


def compute_old_max_mean(env):
    gt = env._env._grid.grid
    true_res = env._env._grid.resolution
    cres = env.CELL_RES
    h, w = env._map_h_cells, env._map_w_cells
    gh, gw = gt.shape
    sub = int(round(cres / true_res))

    # OLD: centre-only
    cols = np.arange(w); rows = np.arange(h)
    WX, WY = np.meshgrid((cols + 0.5) * cres, (rows + 0.5) * cres)
    cx = np.floor(WX / true_res).astype(np.int32)
    cy = np.floor(WY / true_res).astype(np.int32)
    old = (gt[np.clip(cy, 0, gh - 1), np.clip(cx, 0, gw - 1)] != 0).astype(np.float32)

    # Pad for block reduction
    gh_pad = ((gh + sub - 1) // sub) * sub
    gw_pad = ((gw + sub - 1) // sub) * sub
    padded = np.zeros((gh_pad, gw_pad), dtype=np.float32)
    padded[:gh, :gw] = (gt != 0).astype(np.float32)
    blocks = padded.reshape(gh_pad // sub, sub, gw_pad // sub, sub)

    max_pool  = blocks.max(axis=(1, 3))[:h, :w]
    mean_pool = blocks.mean(axis=(1, 3))[:h, :w]

    return gt, old, max_pool, mean_pool


def plot_one(template_id, output_dir):
    name = TEMPLATE_NAMES[template_id]
    env = SpatialObsWrapper(GasSourceEnv(template_id=template_id))
    env.reset(seed=42)
    gt, old, max_pool, mean_pool = compute_old_max_mean(env)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Template {template_id} — {name}   "
                 f"[GT=0.1m, others=0.5m wrapper resolution]",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(gt != 0, origin="lower", cmap="gray_r", interpolation="nearest")
    axes[0].set_title(f"GT walls (0.1 m, {gt.shape})\n"
                      f"density: {(gt != 0).mean():.1%}")

    axes[1].imshow(old, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[1].set_title(f"OLD center-only (binary)\n"
                      f"density: {old.mean():.1%}")

    axes[2].imshow(max_pool, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].set_title(f"NEW max-pool (binary)\n"
                      f"density: {max_pool.mean():.1%}")

    im = axes[3].imshow(mean_pool, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[3].set_title(f"ALT mean-pool (continuous)\n"
                      f"density: {mean_pool.mean():.1%}")
    fig.colorbar(im, ax=axes[3], fraction=0.045, pad=0.04)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out = os.path.join(output_dir, f"wall_compare_t{template_id}_{name}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    for t in range(6):
        plot_one(t, _OUT_DIR)


if __name__ == "__main__":
    main()
