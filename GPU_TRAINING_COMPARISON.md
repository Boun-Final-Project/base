# gpulaika Checkpoint Trials — Comparison

*Generated 2026-07-15. Compares gpulaika RL checkpoints on the `benchmark_env`
29-scenario suite (29 scenarios × 5 runs), physical Nav2 drive, real mature
GADEN plume, spawn at each scenario's `empty_point` (no corrective teleport),
`max_steps=600`, `wall_timeout` 1800–3600 s.*

## Checkpoints under test

| Tag | Checkpoint | Job | Update | Gas physics | Cluster job |
|---|---|---|---|---|---|
| **control** | `gpu_cfd_localwind_step05_..._upd4000_gaden80_ult85_many0.pt` | 25311 | 4000/5000 | **buggy** `FILAMENT_K=0.02` (40× too fast dispersion) | 25855/26002/26022 |
| **s2 (gasfix)** | `gpu_gasfix_s2_job25964_upd3800_val91.7.pt` | 25964 | 3800/8000 | **fixed** `K=0.0005`, σ 0.10, 5 filaments/step | **26323** |
| s1 (gasfix) | `gpu_gasfix_s1_job25963_upd5100_val83.9.pt` | 25963 | 5100/8000 | fixed (under-trained) | not yet run on ROS/GADEN |

All are dual-backbone, 107-dim obs, `local_wind_obs=1`, `lidar_frame=heading`,
device cpu, `RL_USE_NAV2=true` + SLAM stuck-escape. s2 and s1 are the
corrected-gas-physics A/B arms; the control still carries the `K=0.02`
dispersion bug.

## 1. Overall success (real ROS/GADEN eval, n=145 runs each)

| | control | **s2 (gasfix)** |
|---|---:|---:|
| **Success rate** | 87/145 (60%) | **87/145 (60%)** |

**A dead heat on aggregate.** The corrected gas physics did **not** raise the
overall score on real GADEN + Nav2 — it *redistributed* where the policy wins.

## 2. Per-family success rate

| Family | scenarios | control | **s2** | Δ |
|---|:--:|---:|---:|:--:|
| **many_rooms** | 4 | 7/20 (35%) | **12/20 (60%)** | 🔺 **+5** |
| 4_rooms | 5 | 17/25 (68%) | 19/25 (76%) | 🔺 +2 |
| 10x6 | 8 | 36/40 (90%) | 34/40 (85%) | 🔻 −2 |
| curved_labrinth | 6 | 22/30 (73%) | 20/30 (67%) | 🔻 −2 |
| ultimate | 6 | 5/30 (17%) | 2/30 (7%) | 🔻 −3 |
| **Total** | **29** | **87/145 (60%)** | **87/145 (60%)** | **0** |

**Read:** s2 delivers the pre-registered **many_rooms** gain (+25 pts, nearly
closing the gap to ADSM's 14/20 on that family) and a smaller 4_rooms gain,
but pays it back on **ultimate** (−10 pts) and the labyrinths. Net zero.

## 3. Per-scenario success rate (x/5)

| Scenario | family | control | **s2** | Δ |
|---|---|:--:|:--:|:--:|
| 10x6_u_left_1 | 10x6 | 5 | 5 | — |
| 10x6_u_left_2 | 10x6 | 5 | 5 | — |
| 10x6_u_left_3 | 10x6 | 4 | **0** | 🔻 −4 |
| 10x6_u_left_4 | 10x6 | 5 | 5 | — |
| 10x6_u_right_1 | 10x6 | 4 | 5 | 🔺 +1 |
| 10x6_u_right_2 | 10x6 | 5 | 5 | — |
| 10x6_u_right_3 | 10x6 | 5 | 5 | — |
| 10x6_u_right_4 | 10x6 | 3 | 4 | 🔺 +1 |
| curved_labrinth_left_1 | curved | 4 | 5 | 🔺 +1 |
| curved_labrinth_left_2 | curved | 1 | 3 | 🔺 +2 |
| curved_labrinth_left_3 | curved | 5 | 5 | — |
| curved_labrinth_right_1 | curved | 5 | 5 | — |
| curved_labrinth_right_2 | curved | 3 | 2 | 🔻 −1 |
| curved_labrinth_right_3 | curved | 4 | **0** | 🔻 −4 |
| 4_rooms_start_a | 4_rooms | 3 | 1 | 🔻 −2 |
| 4_rooms_start_b | 4_rooms | 3 | 5 | 🔺 +2 |
| 4_rooms_start_c | 4_rooms | 3 | 4 | 🔺 +1 |
| 4_rooms_start_d | 4_rooms | 4 | 5 | 🔺 +1 |
| 4_rooms_start_e | 4_rooms | 4 | 4 | — |
| many_rooms_1 | many_rooms | 0 | 2 | 🔺 +2 |
| many_rooms_2 | many_rooms | 0 | 4 | 🔺 +4 |
| many_rooms_3 | many_rooms | 3 | 2 | 🔻 −1 |
| many_rooms_4 | many_rooms | 4 | 4 | — |
| ultimate_1 | ultimate | 0 | 0 | — |
| ultimate_2 | ultimate | 0 | 1 | 🔺 +1 |
| ultimate_3 | ultimate | 1 | 0* | 🔻 −1 |
| ultimate_4 | ultimate | 3 | 0 | 🔻 −3 |
| ultimate_5 | ultimate | 1 | 0 | 🔻 −1 |
| ultimate_6 | ultimate | 0 | 1 | 🔺 +1 |

\* `ultimate_3` s2 hit the SLURM 2 h wall cap mid-run-5 (0/4 scored; success was
already very unlikely). A clean re-run needs `--time=03:00:00`.

## 4. Where s2's failures actually land — near-miss analysis

For every **failed** s2 run, the closest it ever got to the source (from the
per-step `d2src` in `node.log`). Success threshold ≈ 0.5 m (closest success
0.30 m; closest failure 0.58 m):

| Failed run | scenario | closest approach |
|---|---|---:|
| curved_labrinth_right_2 (×2) | curved | 0.58 m, 0.58 m |
| 4_rooms_start_a | 4_rooms | 0.62 m |
| 10x6_u_left_3 | 10x6 | 0.67 m |
| many_rooms_3 | many_rooms | 0.74 m |
| curved_labrinth_right_3 | curved | 0.75 m |

**8 of 58 failures reached ≤1.0 m, 14 reached ≤1.5 m.** These are
*terminal-behavior* misses — the robot arrives, touches ~0.6 m, then keeps
stepping and never commits to the last few cm — not navigation or plume-loss
failures. This is the single highest-ROI improvement target (a source-commit /
fine-approach behavior, no retraining required).

Two clean regressions are genuine, not flakes (verified in `node.log`):
- **10x6_u_left_3** (4→0): approaches to ~2.3 m then the now-sparser corrected
  plume drops to gas=0; robot casts in a gas-free pocket west of source.
- **curved_labrinth_right_3** (4→0): reaches 0.75 m, then gets pulled back to a
  spurious high-gas pocket at the east wall and collides repeatedly.

## 5. Caveat — the Python surrogate harness disagrees

The pre-deploy Python A/B (surrogate gas, 20 eps/map) predicted a **net** win:

| arm | Python-harness overall | many_rooms | ultimate |
|---|---:|---:|---:|
| control 25311 | 81% | 54% | 63% |
| **s2 (gasfix)** | 86% | 82% | 74% |
| s1 (gasfix) | 67% | 59% | 28% |

On real ROS/GADEN + Nav2, only the **many_rooms** gain transferred (35→60%,
directionally matching 54→82%). The claimed **ultimate** gain (63→74%)
**inverted** (17→7%). This is the proxy-eval gas-model gap: the surrogate
harness synthesizes its own plume and over-credits the hardest maps. **The
defensible claim is "s2 improves sparse/open-plume (many_rooms) at the cost of
ultimate," not a net success-rate win.**

s1 has no ROS/GADEN cluster run yet — its 67% is Python-harness only and not
comparable to the two ROS columns above.

---

*Data: `benchmark_results/` (s2 = job 26323, control = jobs 25855/26002/26022)
under `/comp04-storage/efe-mantaroglu/osl/gsl_eval_data/`. Baselines (ADSM/EESA)
in `../benchmark_env/docs/BENCHMARK_RESULTS.md`.*
