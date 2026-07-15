# Deployment note — agent_188416000.pt (dual_div_rlpkg_resume @ 82.9% GADEN)

## What this is
Best checkpoint from `friend_base/reinforcement_learning/runs/ppo_dual_rlpkg_resume_gentle_20260509_201717_job4876/`,
saved at training step 188,416,000.

- **Architecture**: dual (gas_gru + lidar_conv + gated fusion + actor/critic with residual blocks)
- **Parameter count**: 499,829
- **Observation**: 107-dim flat = `gas_history(30) + lidar(72) + pos(2) + wind(2) + time(1)`
- **Action**: 2-D continuous (cos θ, sin θ); decoded via arctan2 to a heading
- **Lidar frame**: heading-relative (rays measured from robot's current heading) — load-bearing!

## Lineage
- Started from friend's pretrain `agent_183500800.pt` (62.9% GADEN baseline)
- 5M steps of gentle fine-tune (`--lr 5e-5 --no-anneal-lr`) on the 10-template
  curriculum (T0–T9 incl. dead_end_corridor, serpentine, dense_multi_room, hybrid)
- Reward profile: R_SUCCESS=200, R_DETECTION=2.0, R_NEW_CELL=0.5, R_STEP=-0.3,
  R_REVISIT=-0.2, R_COLLISION=-5.0, R_MAX_STEPS=-20.0

## Use this deployment script
**`gaden_transfer/gaden_transfer/gaden_transfer_lidar/gaden_rl_node.py`**

Why this one and not the image variants:
- This policy uses the 107-dim flat observation (lidar + gas history + pos + wind + time)
- The `gaden_transfer_lidar/` variant is the only one that builds that obs format
- The `_image_5ch` and `_image_6ch` variants are for ego-centric grid policies (different arch)

### Launch
```bash
ros2 run gaden_transfer gaden_rl_node \
    --ros-args \
    -p checkpoint:=/home/efe/ros2_ws/src/base/agent_188416000.pt \
    -p arch:=dual \
    -p true_source_x:=<x> \
    -p true_source_y:=<y> \
    -p max_steps:=600
```

Or via the existing `launch/params.yaml` — point its `checkpoint` field at the new file.

### Topics it subscribes to
Same four as efe_igdm: lidar scan, gas reading, wind reading, robot pose.

### Critical: lidar resampling
`gaden_transfer_lidar/lidar_resampler.py` must be active. The policy expects
exactly **72 rays** in **heading-relative** frame. If the live scan has a
different ray count or absolute frame, the resampler must remap before
feeding the policy.

## GADEN per-map performance (5 eps each)
| Map | Success | Notes |
|---|---|---|
| 4rooms | 100% | up from 60% on friend's pretrain |
| uleft | 100% | matches friend |
| uright | 100% | matches friend |
| labyrinth_left | 100% | matches friend |
| labyrinth_right | 100% | matches friend |
| many_rooms | 0% | structurally hard — tight doorways, swirling wind, no point-wind in obs |
| ultimate | 80% | up from 20% on friend's pretrain |

Overall 82.9%, +20pp over friend's 62.9%.

## Known limitations
- many_rooms (and any layout with similar tight doorways + intermittent
  plume) will fail. The policy has no search fallback when no detections.
  Hybrid frontier+RL deploy controller may help — see project notes.
- Wind in obs is the **spatial mean of the field**, set once per episode,
  repeated every step. In multi-room envs with swirls, this is misleading.
- Trained on uniform per-episode wind; deploy environments with strong
  spatial structure will be out-of-distribution for the wind channel.
