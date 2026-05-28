# Experiment log — CFD-wind transfer campaign

Chronological record of training runs in the CFD-wind / GADEN-transfer
investigation, and *why* runs were killed. Goal: never re-run a dead end.

## Killed 2026-05-28

### 5193 — `ppo_dual_obs50_noloop` (uniform wind, fresh)
- **Killed at:** ~132M / 200M steps (66%), after 23.5h.
- **Why killed:** training success collapsed from 96% → 14% and stayed there;
  GADEN eval @115M = 0% on all maps. Unrecoverable.
- **Root cause (not "bad hyperparameters" in general — a bad *combination*):**
  it inherited the **post-May-25 aggressive curriculum** (feature/rl commit
  `6205763` added T8 @40% and T9 @60%) AND ran without the champ's
  stabilizers — `--num-envs 48` (small, noisy gradients) and **no
  `target_kl`** (destructive updates allowed). Hard templates (T8 dense
  rooms, T9 hybrid) have sparse reward → mostly destabilizing gradient noise;
  with a small batch and no KL guard the policy degraded faster than it could
  learn them, eventually failing even trivial maps.
- **Contrast — why the champ (76.4%) did NOT collapse with the same arch:**
  gentle curriculum stopping at T5, `--num-envs 256`, `--target-kl 0.05`,
  `--clip-epsilon 0.3`. Trained on feature/rl *before* `6205763`.
- **Keeper:** `agent_49152000.pt` (~49M, pre-collapse) scored **39.3% GADEN,
  30% many_rooms**. many_rooms-30% is rare (most runs are 0% there) — a lead
  worth its own follow-up. Preserved before killing.

### 5667 — `ppo_cfd_lib_v3` (CFD library, MEAN wind, fresh)
- **Killed at:** ~early (≈42M), GADEN @42M = 0%.
- **Why killed:** confounded — same risky knobs as 5193 (48 envs, no
  `target_kl`) and trains on hard CFD maps (T4-9) from step 0, so it was
  collapse-prone AND couldn't cleanly test the CFD hypothesis. Also,
  observationally it's *identical to uniform training*: the env collapses the
  spatial wind to its mean for the policy obs (`set_uniform(spatial_mean)`),
  so only the plume shape differs — the wind input is unchanged.
- **Lesson:** a mean-wind CFD run cannot beat uniform training *through the
  wind input* by construction. Not a useful experiment.

### 6651 — `ppo_cfd_localwind` (CFD library, LOCAL wind, fresh)
- **Killed at:** early.
- **Why killed:** right *idea* (policy observes local point wind via
  `query(robot_pos)` — the one genuinely novel lever), but wrong *recipe*:
  fresh-from-scratch, 48 envs, no `target_kl`, hard maps from step 0. Same
  collapse risk as 5193. The local-wind variable was confounded with the
  unstable recipe, so its result would be uninterpretable.

## What we concluded

1. The collapses were a **curriculum/stability artifact, not evidence about
   CFD or spatial wind.** The CFD hypothesis remains *untested*.
2. Any clean test must hold the recipe at the **champ's stable settings**
   (≤T5 curriculum, `target_kl 0.05`, `clip 0.3`, large batch) and vary
   *only* the wind. Full code-verified root cause: the champ came from
   feature/rl *before* commit `6205763` (2026-05-25), which is what
   introduced the aggressive T8/T9 curriculum the later runs inherited.
3. The only genuinely unexplored lever is **local-wind observation**
   (`OSL_LOCAL_WIND_OBS=1`). Mean-wind CFD is observationally identical to
   uniform training.

## Replacement plan

Resume from the champ (`ali_champ/agent_91750400.pt`, 76.4%), keep its
stabilizers, restrict to its known templates (T0-5 via `--template-filter`),
and A/B *only* the wind observation: mean vs local. This isolates the single
unexplored variable on a foundation known to be stable.
