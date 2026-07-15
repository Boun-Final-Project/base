# GPU local-wind-observation trainer

## Status (2026-07-15)

- **Built and running end-to-end**: batched GPU env, wind/lidar/filament kernels ported from the
  validated CPU GPU-port, a CFD-library loader (`gpu_cfd_loader.py`), a procedural synth-wind
  fallback (`synth_wind.py` + `wind_field_potflow.py`), a shard-based train/val split, and a
  from-scratch GADEN eval harness (`gpu_fromscratch_gaden_eval.py`).
- **Evaluated on real ROS2/Nav2 + GADEN** (29-scenario benchmark suite, 5 runs each = 145 runs):
  the `control` checkpoint (trained with a buggy gas-dispersion constant, `FILAMENT_K=0.02`, 40x
  too fast) and the `s2 gasfix` checkpoint (corrected `K=0.0005`) both land at **60% (87/145)**
  overall — the corrected physics redistributes *which* map families win rather than raising the
  aggregate. Full per-family breakdown in [`GPU_TRAINING_COMPARISON.md`](GPU_TRAINING_COMPARISON.md).
- **Not yet done**: reflection physics in the GPU env, a full GADEN re-validation pass beyond the
  comparison above, and wiring this trainer into the top-level project docs/workflow.
- **CFD scenario generation is out of scope here.** This trainer only *consumes* a pre-baked CFD
  wind-field library (`grid.npz` / `wind_field.npz` / `meta.json` per case, see
  `gpu_cfd_loader.py`); it does not generate one. The actual scenario/library-generation pipeline
  (mesh gen from GADEN scenarios, multi-direction OpenFOAM wind bake, library sharding) lives in
  `cfd_wind_pipeline/` and is already pushed to GitHub on its own branch, `feature/cfd-wind-pipeline`
  — not merged to `main` and not part of this PR.


A GPU-vectorized trainer where the policy **observes the local wind at the robot's position**
(a point anemometer) rather than a single mean wind vector. This is the `local2` lever that
cracked the `many_rooms` GADEN map — it only matters when wind is **spatially varying**.

## Two wind sources (`--wind`)

| `--wind` | source | cost | maps |
|----------|--------|------|------|
| `synth` (default) | procedural maps + **potential-flow + curl-noise** wind | ~70 ms/map, no CFD bake | **unlimited** (`MapGenerator`) |
| `cfd` | the real **CFD wind-field library** (`cfd_test/library_v3{,b}`) | precomputed (expensive to make) | finite case set |

**`synth` is the cheap "uniform wind with varying" path.** Per the GADEN wind report
(`/comp04-storage/efe-mantaroglu/osl/gaden_wind_report.md`), the project's canonical
spatially-varying generator (`rl_gaden_wind/envs/wind_field.py`, vendored here as
`wind_field_potflow.py`) is exactly that idea: sample a uniform `(speed, direction)`, solve 2D
potential flow `∇²φ=0` with the room's outer wall ring as open Dirichlet inflow/outflow and
interior obstacles as Neumann no-penetration, take `(Ux,Uy)=-∇φ`, then overlay a divergence-free
**curl-noise** field for stationary eddies. `synth_wind.build_synth_pool()` pairs that with
champ's procedural maps and packs them into the same pool dict `load_cfd_pool` returns, so the
rest of the trainer is unchanged.

GADEN-tuned defaults (from `rl_gaden_wind/config.py`): `--speed-lo 0.05 --speed-hi 0.5`
(GADEN measured mean 0.06–0.34 m/s), `--curl-amp 0.6` (sharper eddies to match CFD),
`--curl-scale 4.0`. At this low speed range the curl term is comparable to the mean flow, giving
~7% of free cells a local wind >60° off the map mean — the signal the local-wind obs reads.
Pure potential flow alone (`--curl-amp 0`) is near-uniform in open interiors (~1%); the eddies
are what make local≠mean.

**Caveat (validate against GADEN):** even with curl noise, `synth` is a *training distribution*,
not a faithful GADEN reproduction — the `many_rooms` trap is partly geometry-locked recirculation.
Confirm a synth-trained ckpt against the CPU `local2` / GADEN baseline before trusting it.

## How it works
Reuses the **validated spatial-wind path** of `GpuVecEnvMulti` (`gpu_env.py`) + `gpu_cfd_loader`:
- **advection**: per-filament bilinear query of the CFD field (`gpu_wind.wind_query_bilinear_pool`)
- **observation**: local wind at the robot (`gpu_wind.faithful_obs_wind_pool`), encoded to [0,1]

Both kernels were validated EXACT vs the CPU `WindModel` in the original GPU port
(see memory `project_gpu_env_port`). Same on-device speed as the champ port — no per-env CFD
field is materialized; fields are pooled and indexed per env.

## Relation to the champ trainer
This is the **finetune / local-wind** counterpart of the faithful champ trainer in
`../train_gpu` (which trains on procedural maps + uniform-in-free wind, where local==mean).
Local-wind is a delta on a competent base, so the intended use is `--resume <champ_ckpt>`
(lr 1e-4). It also runs from scratch.

## Run
```
sbatch train_localwind_gpu.sh                                   # synth wind, from scratch
sbatch train_localwind_gpu.sh --resume <champ_ckpt>            # synth wind, finetune (intended)
sbatch train_localwind_gpu.sh --wind cfd --resume <champ_ckpt> # real CFD library instead
# local smoke (validated working):
PYTHONPATH=$PWD python train_localwind_gpu.py --wind synth --envs 64 --rollout 32 --updates 3 --cfd-cases 40
```

## Obs-convention note (verify before trusting checkpoints)
The obs wind uses the **nearest-cell + optional anemometer flip** convention
(`faithful_obs_wind_pool`, `--flip` default off). The GADEN wind report confirms this matches the
deploy encoding exactly: `clip((Ux/2 + 1)/2, 0, 1)` with `WIND_MAX_SPEED=2.0` (report items
#10/#11/#12). The CPU training `get_local_wind` uses **bilinear** interpolation (no flip), which
differs slightly near walls. The report also flags the **mean-vs-local** trap: this trainer uses
LOCAL wind, so a checkpoint from it must be evaluated with `OSL_LOCAL_WIND_OBS=1` (and the
matching `--flip`/`OSL_DEPLOY_ANEMO` setting) — the default champ deploy uses the locked CSV mean,
an incompatible encoding. Before trusting a GPU-trained local-wind checkpoint, run the
deterministic eval (`../train_gpu/eval_maps.py`) and/or a GADEN eval and confirm it matches the
CPU local2 baseline — the same release-gate discipline as the champ port.
