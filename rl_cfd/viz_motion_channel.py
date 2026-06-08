"""Roll an episode with random actions, dump the 6 channels every K steps.

Each frame shows the full obs as the policy sees it. Useful to see the motion
trail evolve over an episode and confirm it captures direction-of-motion.
"""
import os, sys
sys.path.insert(0, '/home/efe/ros2_ws')

import numpy as np
import matplotlib.pyplot as plt
from test_rl_fast.envs import GasSourceEnv, SpatialObsWrapper

OUT_DIR = '/tmp/fast_bundle_viz/motion'
os.makedirs(OUT_DIR, exist_ok=True)
TEMPLATE_ID = 4   # complex maze — most informative
SEED = 7
N_STEPS = 80
DUMP_EVERY = 10

env = SpatialObsWrapper(GasSourceEnv(seed=SEED, template_id=TEMPLATE_ID))
(spatial, ctx), _ = env.reset(seed=SEED)

names = ['is_known', 'is_wall', 'gas', 'recency', 'det_count', 'motion (NEW)']

def render(spatial, ctx, step, robot_pos):
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
    fig.suptitle(
        f'Step {step}  ·  pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f})  ·  '
        f'ctx=[speed={ctx[0]:.3f}, cos={ctx[1]:+.2f}, sin={ctx[2]:+.2f}, t={ctx[3]:.2f}]',
        fontsize=12)
    from matplotlib import colors as mcolors
    for i, ax in enumerate(axes):
        ch = spatial[i]
        if i == 5:
            # Motion: use power norm to lift the dim trail values into view.
            cmap = 'plasma'
            norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=1.0)
            im = ax.imshow(ch, origin='lower', cmap=cmap, norm=norm)
        elif i == 2:
            im = ax.imshow(ch, origin='lower', cmap='RdYlGn', vmin=-1, vmax=1)
        else:
            im = ax.imshow(ch, origin='lower', cmap='viridis', vmin=0, vmax=1.0)
        ax.plot(49, 49, 'r+', markersize=14, markeredgewidth=2)
        ax.set_title(names[i], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'step_{step:04d}.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out

# Render initial frame
render(spatial, ctx, 0, env._env._robot_pos)

# Rollout with smooth-ish actions so motion trail is visible
rng = np.random.default_rng(0)
theta = 0.0
last_paths = []
for step in range(1, N_STEPS + 1):
    # Slow random walk in direction so trail is interpretable
    theta += rng.normal(0, 0.3)
    a = np.array([(theta % (2 * np.pi)) / (2 * np.pi)], dtype=np.float32)
    (spatial, ctx), r, term, trunc, info = env.step(a)
    if step % DUMP_EVERY == 0 or step == N_STEPS:
        path = render(spatial, ctx, step, env._env._robot_pos)
        last_paths.append(path)
        print(f'Step {step}: motion nonzero={int((spatial[5] > 0.01).sum())} cells, '
              f'peak={spatial[5].max():.3f}  →  {os.path.basename(path)}')
    if term or trunc:
        break

print('\nRendered', len(os.listdir(OUT_DIR)), 'frames in', OUT_DIR)
