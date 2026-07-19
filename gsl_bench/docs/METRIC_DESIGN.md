# GSL-Bench: Metric Design Notes

> **Influences (methodology, not strict template):** NAVSIM (Dauner et al. NeurIPS 2024, arXiv:2406.15349; Cao et al. CoRL 2025, arXiv:2506.04218), Anderson et al. 2018 (SPL, arXiv:1807.06757), Vergassola 2007, Park & Cho 2022, Kim et al. 2025 (IGDM), Lilienthal GDM series.
>
> We borrow the *gate &times; weighted-average* composite-shape convention from NAVSIM and the *expert-filter* concept from NAVSIM v2, but do **not** copy NAVSIM wholesale — the gas source localization (GSL) task has different structure (no traffic laws, longer horizons, two distinct task formulations, plume stochasticity).

---

## 1. Two task tracks — the core design decision

Gas source localization in the literature is actually **two distinct tasks**, conflated under one name. They have different success criteria, different controllable actions, and different evaluation protocols. A benchmark that forces them into one score is unfair to both.

### Track A — **Approach** ("go near the source")

The robot must physically reach the source's neighborhood. The agent emits a motion waypoint each step; success is the robot entering the success radius.

- **Operates:** Pioneer P3DX ground robot via Nav2 (as in our `RunnerNode`).
- **Belief use is optional:** surge-cast, chemotaxis, and learned policies work here too — they need not maintain a source belief.
- **Failure modes:** timeout, collision, env-dead.
- **Examples in our suite:** RLaika, RRT-Infotaxis, IGDM (dual-mode), random-walk, upwind-greedy.

### Track L — **Localize** ("declare the source location")

The robot (often a sensor-bearing observer or one that moves to gather info, not necessarily to reach the source) must emit a **point estimate** of the source. It may terminate far from the source; success is whether its estimate lies within a localization tolerance.

- **Works with:** belief-based methods only — the agent maintains a source probability map and declares an argmax (or top-k).
- **Failure modes:** declared-but-wrong, never-declares (timeout), declared-inside-a-wall.
- **Examples in our suite:** Infotaxis, RRT-Infotaxis (as a belief planner), IGDM, ADSM (as a belief mapper), RLaika (the policy can be augmented with a belief head for origination; or excluded from Track L).

**Rules of separation:**

1. Each method is evaluated on **one or both tracks**, declared up-front in its `agent_config.yaml` (`track: A`, `track: L`, or `track: both`).
2. Track-A-only methods (e.g. surge-cast) get **N/A in Track-L tables** — they have no belief output, they are not penalized.
3. Track-L-only methods (hypothetical stationary sensor-array GSL or map-only belief) get **N/A in Track-A tables** — they never plan a robot motion.
4. Each track has its **own composite score**, its **own headline table**, its **own leaderboard**.

This mirrors robotics literature (e.g. Vergassola 2007 measures *approach*, Lilienthal GDM measures *localization error*) but formalizes them as co-equal tasks — *no published GSL benchmark does this explicitly.*

---

## 2. Two composite scores, same shape

Both tracks use the NAVSIM *gate &times; weighted-average* composite, but with **different gate and quality metrics** appropriate to each task.

$$
\text{GSLS-X} = \underbrace{\prod_{m \in \mathcal{G}_X} g_m}_{\text{gate (hard-zeroing)}} \;\cdot\; \underbrace{\frac{\sum_{m \in \mathcal{Q}_X} w_m\, q_m}{\sum_{m \in \mathcal{Q}_X} w_m}}_{\text{quality (soft)}}
$$

with X &isin; {A, L}. Each $g_m$ is binary or graded {0, &frac12;, 1}; a zero kills the score (no "collision but smooth" consolation). Each $q_m$ &isin; [0, 1]. Weights are exposed in `harness_default.yaml`, sum to 1 per track, transparent and re-weightable.

Scale: **[0, 1]**, leaderboards render &times;100. **Median [IQR]** across seeds per scenario; **mean across scenarios** for the headline.

---

## 3. Track A — Approach metrics (10 total: 3 gate + 7 quality)

### 3.1 Gate metrics $\mathcal{G}_A$ (hard-zeroing)

| # | Metric | Symbol | Definition |
|---|---|---|---|
| 1 | **No Collision** | NC | Binary. 1 if the robot never collided with a wall/obstacle during the episode. |
| 2 | **In-Bounds Compliance** | IBC | Binary. 1 if the robot stays inside the scenario bbox for the whole episode. |
| 3 | **Valid Termination** | VT | Graded {0, &frac12;, 1}: 1 = success within step budget; &frac12; = ran out of steps but in-bounds and collision-free; 0 = `env_dead` / `wall_timeout` / `agent_error`. Replaces the single binary `success_radius` check at `gsl_bench/harness/runner_node.py:437`. |

### 3.2 Quality metrics $\mathcal{Q}_A$ (weighted-average)

| # | Metric | Symbol | Definition | Weight | Borrowed from |
|---|---|---|---|---|---|
| 4 | **Source Progress** | SP | `min(1, (d_start&rarr;source &minus; d_final&rarr;source) / d_start&rarr;source)`. Defined over **ALL episodes, not just successes** &mdash; failures contribute. | 0.30 | NAVSIM "Ego Progress" |
| 5 | **Time-to-Source** | TTS | `exp(&minus;t / &tau;)` with `t` = sim-seconds until first success-radius entry, `&tau;` = half the step budget. Returns 0 on timeout. | 0.20 | Standard GSL |
| 6 | **Path Efficiency** | PE | `&ell;_geodesic(start, source) / &ell;_actual`, capped to [0, 1]. | 0.15 | Anderson 2018 (SPL) |
| 7 | **First Hit Time** | FHT | `1 &minus; (t_first_gas_hit / max_steps)`. Measures plume-finding speed, distinct from TTS (measures source-finding speed). | 0.10 | Active-SLAM style |
| 8 | **Plume Contact Ratio** | PCR | Fraction of timesteps with PID reading above the noise floor. Distinguishes plume-riding from random casting. | 0.05 | Chemotaxis literature |
| 9 | **Coverage** | COV | Fraction of free cells within sensor range of the trajectory over the max reachable fraction in the time budget. | 0.05 | Active-SLAM |
| 10 | **Per-Step Compute** | PSC | `1 &minus; t_step / t_budget` with `t_step` = p99 wall ms of `observe()+act()`, `t_budget` = p99 of the slowest method in the comparison (sets the scale). Encodes the "7&times; cheaper" claim from the `base` README directly. | 0.15 | Park & Cho 2022; new |

Weights sum to 1.0. SP+TTS+PE carry the headline (65% of the soft mass); the rest are diagnostic.

### 3.3 Track-A success

The GSLS-A composite **subsumes** Success Rate (the classical binary we already report): VT&middot;SP captures it (a run that reached the source gets SP=1, VT=1; one that didn't gets SP&le;1, VT&le;&frac12;). For table readability, plain SR (with Wilson CI) is still reported as an auxiliary column, but the composite rewards partial progress where SR alone gives a 0.

---

## 4. Track L — Localization metrics (9 total: 3 gate + 6 quality)

A method in Track L must, at episode end, emit a **source estimate** `(x_est, y_est)` via the new `GSLAgent.estimate() -> Optional[(float, float)]` API. Methods that never declare return `None`; their episode is scored VT=0, hence GSLS-L = 0.

### 4.1 Gate metrics $\mathcal{G}_L$ (hard-zeroing)

| # | Metric | Symbol | Definition |
|---|---|---|---|
| 1 | **Valid Declaration** | VD | Binary. 1 if the agent emitted a non-`None` source estimate by `max_steps`. |
| 2 | **Feasible Estimate** | FE | Binary. 1 if the estimate lies in **free space** (not inside a wall/obstacle). Replaces NAVSIM's drivable-area-compliance. |
| 3 | **No Collision** | NC | Binary. For methods that also move; for stationary belief emitters, set to 1 by the harness (N/A gating). |

### 4.2 Quality metrics $\mathcal{Q}_L$ (weighted-average)

| # | Metric | Symbol | Definition | Weight | Borrowed from |
|---|---|---|---|---|---|
| 4 | **Source Localization Error** | SLE | `exp(&minus;&Vert;est &minus; true&Vert;_2 / &sigma;)` with `&sigma; = localization_tolerance` (default 1.0 m, distinct from the approach `success_radius` of 0.5 m). | 0.35 | Park & Cho 2022; standard GDM |
| 5 | **Top-k Hit** | Tkk | `1` if the true source cell is in the agent's top-k belief cells (k = 1, 5, 10 reported separately; Tkk uses k=5 as primary). 0 otherwise. | 0.20 | Object-detection AP tradition |
| 6 | **Belief NLL** | NLL | `&minus;log p(source_true | belief)`, min-max normalized across the episode set to [0, 1]. Probabilistic-calibration metric. | 0.10 | IGDM literature |
| 7 | **Cumulative Information Gain** | CIG | Bits gained: `&Sigma;_t [H(belief_{t-1}) &minus; H(belief_t)]`, normalized by `log(N_cells)`. | 0.10 | Kim et al. 2025 (IGDM) |
| 8 | **Time-to-Declaration** | TTD | `1 &minus; t_decl / max_steps` &mdash; rewards confident early declarations; timeout declarations get 0. | 0.15 | Standard |
| 9 | **Per-Step Compute** | PSC | Same definition as Track A. | 0.10 | Park & Cho 2022; new |

Weights sum to 1.0. SLE carries the headline (35%), Tk5 + TTD give the next tier; the rest are diagnostic and enable ablations of *why* a method localizes well.

### 4.3 Track-L is belief-aware only

Methods that do not maintain a source belief (random-walk, upwind-greedy, surge-cast) **cannot** construct Track-L outputs and are marked N/A, not penalized. The `agent_config.yaml` declares `track: A` for them. The paper's Track-L table is smaller (4 &mdash; 5 entries) than the Track-A table (6 entries).

---

## 5. The Expert (Oracle) Filter &mdash; borrowed from NAVSIM v2

NAVSIM v2 neutralizes a per-metric penalty if the **expert/human driver would also have failed** that scene &mdash; eliminating false-negative penalties for legitimately ambiguous frames. The direct GSL analog:

$$
\text{filter}_m(\text{agent}) = \begin{cases}
1.0 & \text{if } m(\text{oracle}) = 0 \\
m(\text{agent}) & \text{otherwise}
\end{cases}
$$

where `oracle` is an **idealized planner with cheat-sheet access to source+wind**, run once per scenario over the same GADEN bake. If even the oracle can't localize this scene (plume never reaches any reachable cell in the time budget, the source is occluded by walls such that no PID/anemometer config would reveal it, etc.), the agent is **not** penalized &mdash; that sub-metric is neutralized to 1.0 for that scene.

**Implementation:** run the oracle once per scenario; cache its sub-metrics in `scenario/oracle_metrics.json`; apply the column-wise filter at aggregation in `gsl_bench/eval/metrics.py` (extending the existing fairness guard at L69-80). NAVSIM does exactly this with a `Human` privileged agent.

This is the second key transferable idea from NAVSIM &mdash; the **first** being the gate &times; weighted-average composite shape, which we apply per-track.

---

## 6. Statistical rigor (cross-track mandatory)

Per-scenario:
- **Median [IQR]** for continuous metrics (TTS, PE, SLE, TTD, CIG, PCR, COV) &mdash; GSL distributions are heavy-tailed; one stuck robot skews any mean.
- **Mean [+ Wilson 95% CI]** for binary metrics (NC, IBC, VT, VD, FE, SR).
- **n &ge; 10** seeds per (scenario, method). For headline runs, push n = 20. The current harness's identical-seed bug (each agent self-seeds `default 0` at construction, so `--runs 5` produces 5 identical runs &mdash; see my reproducibility notes) must be fixed first.

Cross-method (mandatory for every claim of the form "X > Y"):
- **Paired Wilcoxon signed-rank** across all episodes (`scipy.stats.wilcoxon`).
- **Cliff's &delta;** effect size alongside every p-value.
- Without these, reviewers will (correctly) push back. A modern benchmark cannot publish paired bar charts without paired significance.

The fairness guard at `gsl_bench/eval/metrics.py:69-80` already exposes harness-block mismatches across runs &mdash; extend it to also flag `wall_timeout_s`, which is currently blind to it (see my reproducibility notes).

---

## 7. Robustness axes (the section that distinguishes a benchmark from a results paper)

Reported as **curves or ablation tables**, not headline numbers. Same protocol applies to both tracks.

| Probe | Why it matters | Source |
|---|---|---|
| **Cross-family variance** | Exposes the "ADSM wins small maps, RLaika wins ultimate" pattern already visible in the `base` README's per-scenario table. | Free &mdash; existing family tags |
| **Wind-speed sensitivity** | GSL algorithms separate sharply in low- vs high-P&eacute;clet regimes (Vergassola 2007). Requires `benchmark_env` to add `0.5ms` and `2ms` wind variants alongside the existing `1ms`. | New scenario variants |
| **Sensor-noise sweep** | Anemometer/PID `noise_std` are ROS params already; sweep `&sigma;` &isin; {0, 0.1, 0.3, 0.5}. | Config only |
| **Sim-to-real transfer** | The `57% &rarr; 76-80% after CFD finetune` gap already reported in the `base` README is itself a benchmark contribution: report SR drop from training-domain eval &rarr; GADEN eval as a transfer-robustness row. | Have the numbers |
| **Seed variance** | `&sigma;(GSLS)` across &ge;5 seeds per (scenario, method); the random walks and most classical algorithms are deterministic &mdash; the variance is on Nav2 + the gas player + the learned policy. | Fix current seed bug |

---

## 8. Comparison protocol for the paper

1. **Track-A headline table:** all approach-capable methods &times; GSLS-A (median [IQR]) + 4 aux columns (SR [+Wilson], TTS, PE, PSC). Cites Anderson/Habitat style.
2. **Track-L headline table:** belief-only methods &times; GSLS-L (median [IQR]) + 4 aux columns (SLE, Tk5, TTD, CIG). Smaller table, sharper comparative claim.
3. **Efficiency frontier (cross-track):** x = `log(t_step)`, y = GSLS-A or GSLS-L. RLaika's "cheap + high-score" corner becomes the headline visual.
4. **Per-family bar plots:** GSLS-A by family for all approach methods; exposes per-family algorithm strengths.
5. **Trajectory qualitative figure:** 2-3 example trajectories per algorithm on `ultimate_1` (approach track); belief heatmaps with declared source markers for the localization track.
6. **Failure-mode pie chart** per method: `env_dead` vs `max_steps` vs `agent_error` &mdash; already have the data via `status_counts`.
7. **Robustness section:** SR-vs-wind-speed curves, sensor-noise sweeps, cross-family variance table, sim-to-real transfer row.
8. **Statistical test:** paired Wilcoxon + Cliff's &delta; on every "X > Y" claim, reported as a separate table or as footnotes to the headline tables.

---

## 9. Implementation map (where each piece plugs in)

| Concept | File | Action |
|---|---|---|
| `GSLAgent.track` field + `estimate()` method | `gsl_bench/agent.py` (in `GSLAgent` ABC) | Declare `track` (`A`, `L`, `both`); add optional `estimate() -> Optional[(float,float)]`. Default `None`. Belief agents override. |
| Track-A success check | `gsl_bench/harness/runner_node.py:437` | Replace single binary radius check with VT grading (success / in-bounds-timeout / failure). Already have collision flag in basic_sim, IBC trivial from pose vs bbox. |
| Per-step compute timing | `gsl_bench/harness/runner_node.py:477-485` | Wrap `agent.observe()` + `agent.act()` in `time.perf_counter`; store `t_step_p50` and `t_step_p99` in `result.json`. |
| Trajectory logging | `gsl_bench/harness/runner_node.py:_write_result` | Add `trajectory: [{x, y, theta, t}, ...]` (pose already captured in `_pose_callback`). Use a compression-friendly format; gate by `--log-trajectory`. |
| Declaration timestamp | `runner_node.py` | Record `t_decl`, `(x_est, y_est)` when `agent.estimate()` first returns non-`None`. |
| Belief dump for CIG/NLL | `agent.py` + `runner_node.py` | Optional `agent.belief()` returning an `np.ndarray` over free cells each step; sampled every K steps for storage. Belief-less agents return `None`. |
| Weights &amp; harness YAML | `gsl_bench/eval/episode_runner.py:560-566` | Read `harness_default.yaml` (currently shipped but unread); add `--harness-config` flag and per-track weight sections `track_A`, `track_L`. |
| Composite scorer | `gsl_bench/eval/score.py` (new) | Implement gate &times; weighted-average composite per track + expert filter; produce per-episode GSLS-A, GSLS-L. |
| Oracle agent | `gsl_bench/eval/oracle.py` (new) | Cheat-source planner: drive straight to source (Track A) or immediately declare (Track L). Run once per scenario; cache sub-metrics in `scenario/oracle_metrics.json`. |
| Seed plumbing | `gsl_bench/eval/episode_runner.py` + `gsl_bench/agent.py` | Add `--seed`, plumb into agents + `torch.manual_seed` / `np.random.seed` / `random.seed`; stamp `seed` in every `result.json`. |
| Statistical report | `gsl_bench/eval/report.py` | Add Wilson CI, IQR, `scipy.stats.wilcoxon` paired test, Cliff's &delta;. Two headline tables (A, L). |
| Plotting | `gsl_bench/tools/` (new) | `plot_trajectory.py`, `plot_efficiency_frontier.py`, `plot_robustness_curves.py`, `plot_belief_heatmap.py`. |

---

## 10. Hallmark table layout (paper-style)

### Table 1 &mdash; Track-A Headline (approach-capable methods only)

| Method | **GSLS-A &uarr;** | SR &uarr; | TTS [s] &darr; | PE &uarr; | $t_{step}$ [ms] &darr; |
|---|---|---|---|---|---|
| RLaika (ours) | **0.71 [0.61-0.79]** | **0.91 [+0.05/-0.07]** | 897 | 0.32 | **1.4** |
| ADSM | 0.58 | 0.80 | 160 | 0.66 | 9.5 |
| EESA | 0.31 | 0.57 | 110 | 0.91 | 10.8 |
| RRT-Infotaxis | ... | ... | ... | ... | ... |
| IGDM | ... | ... | ... | ... | ... |
| Random walk | 0.09 | 0.05 | - | 0.10 | < 0.1 |

GSLS-A reported as **median [IQR]** across seeds &times; scenarios. SR with **Wilson 95% CI**. $t_{step}$ as median + p99. Pattern matches NAVSIM's headline per-seed-aggregated-table.

### Table 2 &mdash; Track-L Headline (belief-only)

| Method | **GSLS-L &uarr;** | SLE [m] &darr; | Tk5 &uarr; | TTD [s] | CIG [bits] &uarr; |
|---|---|---|---|---|---|
| IGDM | ... | ... | ... | ... | ... |
| RRT-Infotaxis | ... | ... | ... | ... | ... |
| Infotaxis | ... | ... | ... | ... | ... |
| RLaika (belief head) | ... | ... | ... | ... | ... |
| ADSM | ... | ... | ... | ... | ... |

### Figure 1 &mdash; **Efficiency frontier** (x: `log(t_step)`, y: GSLS-A or GSLS-L). RLaika's tiny net in the "cheap + high-score" corner is the visual argument.

### Figure 2 &mdash; **Trajectory posters** (2-3 per method on `ultimate_1`).

### Figure 3 &mdash; **Robustness** (GSLS-A vs wind speed; GSLS-L vs sensor noise `&sigma;`).

---

## 11. Positioning vs Herwich/GSL-Bench

> *"Closest prior work, GSL-Bench [Herwich et al.], uses 6 Isaac Sim warehouse environments, 3 algorithms (E. Coli, dung beetle, random walk), and 6 raw metrics on an aerial platform. We complement this with a ROS2-standard ground-robot benchmark on the field-validated GADEN filament simulator, with 30 scenarios across 5 map families and 6 head-to-head algorithms (including a learned policy). We further separate the two distinct GSL task formulations &mdash; **approach** (physically reach the source) and **localize** (declare the source position) &mdash; into two parallel tracks with independent composite scores, and propose a gate &times; weighted-average GSL Score (GSLS-A, GSLS-L) of 10 and 9 sub-metrics respectively, structured following NAVSIM [Dauner et al. 2024]. Our inclusion of source-localization error (SLE), top-k accuracy, and information-gain (CIG) metrics &mdash; unreported by GSL-Bench and rare in GSL literature &mdash; enables a comparison of belief-based methods on *understanding*, not just *arrival*."*

---

## 12. Citations to add

- **NAVSIM v1:** Dauner et al., NeurIPS 2024 Datasets &amp; Benchmarks, arXiv:2406.15349 &mdash; *primary methodology influence (gate &times; weighted-average composite).*
- **NAVSIM v2 (Pseudo-Simulation):** Cao et al., CoRL 2025, arXiv:2506.04218 &mdash; *expert-filter concept (oracle-neutralized ambiguous scenes).*
- **SPL:** Anderson et al., 2018, arXiv:1807.06757 &mdash; *Path Efficiency normalized.*
- **Habitat:** Savva et al., ICCV 2019, arXiv:1904.01201 &mdash; *cross-environment SR + benchmark protocol conventions.*
- **Lilienthal GDM series** &mdash; *map-error / KL / correlation conventions (already in refs via IGDM lineage).*
- **Recent CPSL UAV work** (arXiv:2603.11582, 2026) &mdash; *modern paired-baseline comparison framing; situates learned GSL in the latest literature.*

---

## 13. Summary

- **Two task tracks**, two composite scores, two headline tables.
- **Track A &mdash; Approach** (3 gates + 7 quality; SR-as-auxiliary). Pioneer robot, all 6 algorithms.
- **Track L &mdash; Localize** (3 gates + 6 quality; SLE-as-headline). Belief-only, 4-5 algorithms.
- **Composite shape:** gate &times; weighted-average &mdash; NAVSIM's convention, applied per-track.
- **Expert filter:** oracle-neutralized ambiguous scenes &mdash; NAVSIM v2's transferable trick.
- **Statistical rigor:** paired Wilcoxon + Cliff's &delta; + Wilson CI + IQR, n &ge; 10 seeds.
- **Robustness axes:** cross-family, wind-speed, sensor-noise, sim-to-real transfer, seed variance.
- **Differentiates from Herwich/GSL-Bench**: more algos (6 vs 3), more scenarios (30 vs 6), more metrics (10+9 vs 6), two task tracks (Herwich does not formally separate), composite with gating (vs raw metrics), explicit statistical protocol (vs none).