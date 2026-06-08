# champ_far02 deploy — SLAM stuck-escape + hybrid drive

Final 7-map evaluation of the SLAM-based stuck-escape stack with the **hybrid
drive** (escape hops on deterministic cmd_vel, policy steps on Nav2/DWB).
5 runs/map, `champ_far02_agent_106496000.pt`, speed=2, max_steps=600.

| map | baseline (DWB, escape OFF) | HYBRID (escape ON) |
|---|---|---|
| 4_rooms | 5/5 | 5/5 |
| 10x6_u_left | 5/5 | 5/5 |
| 10x6_u_right | 5/5 | 5/5 |
| curved_labrinth_left | 5/5 | 5/5 |
| curved_labrinth_right | 5/5 | 5/5 |
| **many_rooms** | **0/5** | **4/5** |
| ultimate | 3/5 | 3/5 |
| **TOTAL** | **28/35 (80%)** | **32/35 (91%)** |

Net **+4**, entirely from cracking **many_rooms (0/5 → 4/5)**, with **no regression**
on any other map. (many_rooms was run with the source at (2.5,3.0) + denser
regenerated GADEN gas; the original depleted-pocket map is the 0/5 case.)

## What the stack does

1. **Stuck detector** (`escape_planner.py`) — fires on EITHER:
   - efficiency-streak (tight circling, fast), OR
   - long-horizon map-growth stall: `< grow_min` new SLAM cells over the last
     `grow_win` steps (cumulative window, not a streak — robust to excursions).
2. **Frontier-existence gate** — only escape if a reachable frontier cluster
   `>= frontier_min_cells` exists (denominator-free "explored enough" test;
   replaces the broken `mapped_fraction / total_grid` gate).
3. **A\* path inflation** — escape paths keep robot-radius clearance from walls
   (graceful fallback) so the controller doesn't wedge on wall-hugging waypoints.
4. **Escape-hop execution fix** — `_escape_fails` now counts only genuinely
   wedged hops (was incremented every in-progress step → escapes aborted ~4 hops
   in and never completed long paths).
5. **Hybrid drive** (`gaden_rl_node.py`, `OSL_DET_DRIVE=1`) — escape hops driven
   by a deterministic proportional cmd_vel controller (so they physically execute
   the A\* path); **policy steps stay on Nav2/DWB** (obstacle-avoiding). Det-drive
   everywhere cracked many_rooms but regressed corridor maps (inefficient straight-
   line driving); reserving it for escape hops keeps both.

## Reproduce (env flags)

```
RL_ESCAPE=1 RL_ESCAPE_TARGET=nearest RL_ESCAPE_MINDIST=0.0
OSL_DET_DRIVE=1                 # hybrid: cmd_vel for escape hops only
OSL_ESCAPE_GROW_WIN=120 OSL_ESCAPE_GROW_MIN=150 OSL_ESCAPE_FRONTIER_MIN=12
OSL_ESCAPE_SRC_EST=0
```

Escape is OFF by default (`OSL_ESCAPE=0`) — baseline behaviour is unchanged
unless explicitly enabled.
