# gsl_benchmark

One place for the pieces that turn `gsl_bench` into **the** published GSL benchmark:
a fixed 29-scenario suite, per-scenario oracle budgets, a 5×/10×/20× success-and-
termination envelope, and confidence intervals over repeats.

This folder does **not** fork any code. The engine is the `gsl_bench` ROS 2 package
(the harness owns sim + Nav2 + metrics). Here we only wire in the defaults.

```
gsl_benchmark/
├── run_benchmark.sh                  # run: pick agent, scenarios, repeats (headless)
├── watch.sh                          # watch ONE scenario live in RViz (SLAM mode)
├── aggregate.py                      # (re)aggregate a results dir with 95% CIs
├── oracle_budgets_nav2_reliable.json # 29 per-scenario oracle distance/time budgets
└── README.md
```

## What a benchmark run is

- **Scenario** — one of 29 (`benchmark29` suite): map + gas source + robot start,
  from `benchmark_env`. House09 is excluded. Realistic perception (SLAM pose +
  online map) is available via `--realistic`.
- **Method ("robot")** — a `gsl_bench` agent implementing `observe()`/`act()`
  (`gpulaika`, `adsm`, `surge_cast`, …). Any checkpoint is loaded by the agent
  from its `configs/agents/<agent>.yaml`.
- **Repeats** — each scenario is run *N* times (default 5) for statistics.

## Success & termination

Per scenario the oracle (a privileged A*-to-source drive) gives a near-optimal
`travel_distance_m` and `sim_time_s`. Every run is scored against a multiple of it:

| threshold | multiplier | effect |
|---|---|---|
| **success** | 5× oracle distance | reaching within 0.5 m **and** ≤ 5× counts as success |
| **terminate (distance)** | 10× oracle distance | run ends as failure past this |
| **terminate (time)** | 20× oracle sim-time | run ends as failure past this |
| **terminate (other)** | — | stuck / `max_steps` (600) / env-dead |

`reached_source` (reached at all, up to 10×) is recorded separately from
`success` (reached efficiently, ≤ 5×), so an inefficient reach is visible, not
hidden.

## Run it

```bash
cd /home/efe/ros2_ws/src/gsl_benchmark
./run_benchmark.sh gpulaika          # 29 × 5 runs
./run_benchmark.sh gpulaika 3        # 29 × 3 runs
./run_benchmark.sh adsm 5 --realistic
```

Results land in `Results/<agent>_<timestamp>/` with per-run `result.json`,
a `report.md` (per-scenario table with 95% CIs), and `results.csv`.

## Watch a run live (RViz)

```bash
cd /home/efe/ros2_ws/src/gsl_benchmark
./watch.sh 4_rooms_start_a            # one run, live, in RViz
./watch.sh ultimate_1 gpulaika
```

Runs a single scenario in **realistic/SLAM** mode and opens RViz on the virtual
display `:99` (it starts `Xvfb :99` + `x11vnc` on port 5900 for you). From your
Mac, tunnel 5900 and open `vnc://localhost:5900` (see `../../REMOTE_VIEWING_README.md`).

You see: the **SLAM map** (built online), the **gas plume**, the **robot** (axes +
lidar), the current **goal** (cyan arrow) and per-step target, and the Nav2 plan.

**Origin fix:** GADEN publishes gas in world coordinates under a frame also named
`map`, while SLAM's `map` is anchored at the robot start — so raw, the plume would
be offset. `watch.sh` (via `--visual`) launches `gsl_bench world_align`, which
broadcasts a static TF `map → gaden_world` at `-start` and relays the gas/source
markers into `gaden_world`, so plume, map, robot and goal all line up. It reads the
per-scenario start from the scenario, so it is correct for every map (not a
hardcoded offset).

The same behaviour is available as a flag on the batch driver:
`ros2 run gsl_bench eval --agent <a> --realistic --visual --scenario <s> --runs 1`.

## Aggregate an existing results dir

```bash
source ../../install/setup.bash
python3 aggregate.py <results_dir>              # recompute report + CIs
python3 aggregate.py <results_dir> --relabel    # apply the 5× budget to old runs
```

CIs: **Wilson** score interval for success/reached rates (robust near 0/100 % and
at N=5), **bootstrap** percentile interval for mean time / distance / steps.

## Regenerating the oracle budgets

`oracle_budgets_nav2_reliable.json` was produced by the privileged oracle
(reliable Nav2 mode) and copied here from `Results/`. To rebuild:

```bash
ros2 run gsl_bench oracle --suite benchmark29 --drive-mode nav2 \
  --out /home/efe/ros2_ws/Results/oracle_budgets.json
```

then copy the manifest into this folder.

## Next step

Docker deployment (packaging this as a reproducible container) — not done here yet.
