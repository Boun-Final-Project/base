# Multi-Source Gas Source Localization: Extension Plan

## Executive Summary

This document outlines a plan to extend `igdm_multiple` from single-source to multi-source gas source localization. The core idea, drawn from three reference papers (PSPF, PID-STE, ADE-PSPF), is a **multi-layer particle filter** where each layer independently estimates one source, combined with **concentration superposition** decomposition and **peak suppression** to prevent all layers from collapsing onto the same source.

---

## 1. Problem Statement

The current system estimates a single source θ = [x₀, y₀, Q₀]. When multiple sources exist, the sensor reads the **superimposed concentration field**:

```
z(rk) = Σᵢ Rᵢ(rk|θᵢ) + noise
```

The single-source particle filter cannot decompose this sum. We need to:
1. Estimate the number of sources N_s
2. Estimate parameters θᵢ = [xᵢ, yᵢ, Qᵢ] for each source
3. Drive exploration to discover and resolve all sources

---

## 2. Literature Summary

### 2.1 PSPF (Gao et al., Sensors 2018)

**Key contributions:**
- **Multi-layer particle swarms**: L layers (L ≥ expected N_s), each a standard particle filter tracking one source
- **Synthesized weight**: `w = w_obs · w_ps · w_dist`
  - `w_obs`: Standard observation likelihood
  - `w_ps`: **Peak suppression** — reduces likelihood around other layers' estimates to prevent convergence to the same source
  - `w_dist`: Swarm distance correction — prevents layers that are far from any real source from accumulating weight
- **Mean-shift clustering**: Used to determine if a layer has converged (tight cluster = likely real source)
- **Source count estimation**: Number of tight clusters = estimated N_s

**Strengths:** Elegant anti-collapse mechanism, non-parametric source counting
**Weaknesses:** Assumes known/bounded source count, no exploration strategy

### 2.2 PID-STE (Bai et al., Building & Environment 2023)

**Key contributions:**
- **Dynamic State Space**: Uses 3D octomap to construct search space; expands as robot explores
- **Parallel particle filtering**: Multiple layers like PSPF, but with pseudo-source verification
- **Pseudo-source verification**: After convergence, uses **Poisson kriging interpolation** to check if concentration at estimated location matches observations. Non-matching estimates are flagged as pseudo-sources (artifacts of superposition)
- **DBSCAN clustering**: Extracts valid observation sets from measurement history

**Strengths:** Rigorous verification step eliminates false positives
**Weaknesses:** Computationally expensive verification, requires dense measurement history

### 2.3 ADE-PSPF / MRSS (Bai et al., Robotics & Autonomous Systems 2023)

**Key contributions:**
- **Adaptive Differential Evolution + PSPF**: Uses DE crossover/mutation instead of standard MCMC for particle diversification
- **Radiation gain model for RRT**: Extends information gain to account for multi-source superposition
- **Three correction factors**:
  - **OIC (Observation Intensity Correction)**: Adjusts observed intensity by subtracting estimated contributions from already-found sources
  - **RSC (Redundant Sampling Correction)**: Penalizes revisiting already-sampled locations
  - **REC (Repeat Exploring Correction)**: Modifies exploration gain near confirmed sources
- **Observation-Estimation-Exploration loop**: Structured 3-phase approach

**Strengths:** Best exploration strategy, correction factors directly applicable to our RRT planner
**Weaknesses:** DE optimization adds complexity, radiation-specific terminology needs adaptation

---

## 3. Proposed Architecture

### 3.1 Overview

```
                    ┌─────────────────────┐
                    │   MultiSourceNode   │  (extends RRTInfotaxisBasicNode)
                    │   igdm_multi.py     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    │ MultiLayer  │  │ Superposition│  │ Multi-Source  │
    │ ParticleFilt│  │ Decomposer   │  │ RRT Planner  │
    └─────────────┘  └──────────────┘  └──────────────┘
    L layers of PF    Corrects obs.    Modified BI with
    + peak suppress.  per layer        OIC + RSC + REC
```

### 3.2 Component Design

#### A. `MultiLayerParticleFilter` (new class in `estimation/`)

Wraps L instances of the existing `ParticleFilter`, adding inter-layer interactions.

```python
class MultiLayerParticleFilter:
    def __init__(self, num_layers, num_particles_per_layer, search_bounds,
                 sensor_model, dispersion_model, peak_suppression_radius):
        self.layers: List[ParticleFilter]  # L independent PFs
        self.num_layers = num_layers
        self.ps_radius = peak_suppression_radius  # r_ps
        self.confirmed_sources = []  # Sources that passed verification

    def update(self, measurement, sensor_position):
        """
        Update all layers with synthesized weights.

        For each layer l:
        1. Compute standard observation likelihood: w_obs
        2. Compute peak suppression factor: w_ps (suppress near other layers' modes)
        3. Compute combined weight: w = w_obs * w_ps
        4. Update layer l
        """

    def get_all_estimates(self) -> List[Dict]:
        """Return estimates from all converged layers."""

    def get_source_count_estimate(self) -> int:
        """Use clustering to estimate number of real sources."""

    def get_corrected_measurement(self, measurement, sensor_position, exclude_layer) -> float:
        """
        OIC: Subtract estimated contributions from confirmed sources.
        Returns corrected measurement for a specific layer's update.
        """
```

**Peak Suppression Weight** (from PSPF):
```
w_ps^l(θ) = Π_{j≠l} [1 - exp(-||θ_xy - μ_j||² / (2·r_ps²))]
```
where μ_j is the weighted mean of layer j. This makes weights near other layers' estimates approach 0, forcing each layer to find a different source.

#### B. `SuperpositionModel` (new class in `estimation/`)

Handles the concentration superposition principle.

```python
class SuperpositionModel:
    def __init__(self, dispersion_model):
        self.dispersion_model = dispersion_model

    def compute_total_concentration(self, sensor_pos, all_source_params):
        """Sum concentrations from all estimated sources."""
        total = 0.0
        for params in all_source_params:
            total += self.dispersion_model.compute_concentration(
                sensor_pos, (params['x'], params['y']), params['Q']
            )
        return total

    def compute_residual(self, measurement, sensor_pos, confirmed_sources):
        """
        OIC: z_corrected = z_observed - Σ R(rk|θ_confirmed)

        Subtracts contributions from confirmed/converged sources so that
        remaining layers can focus on undiscovered sources.
        """

    def compute_superimposed_likelihood(self, measurement, sensor_pos, layer_particles, other_estimates):
        """
        Likelihood accounting for superposition:
        z_expected = R(rk|θ_layer) + Σ R(rk|θ_others)
        p(z | θ_layer) = N(z; z_expected, σ²)
        """
```

#### C. Modified RRT Planner (changes to `planning/rrt.py`)

Extend branch information calculation with multi-source corrections from ADE-PSPF.

```python
def calculate_branch_information_multi(self, path, multi_pf, confirmed_sources):
    """
    Modified BI with three correction factors:

    BI_multi = Σ γ^i · [I(v_i) · OIC(v_i) · RSC(v_i) · REC(v_i)]

    OIC: Observation Intensity Correction
      - Higher gain in areas where residual (unexplained) concentration is high
      - Encourages exploring areas not explained by current estimates

    RSC: Redundant Sampling Correction
      - Penalizes positions already visited (measurement history)
      - Encourages exploring NEW areas

    REC: Repeat Exploring Correction
      - Reduces exploration gain near confirmed sources
      - Focuses exploration away from already-found sources
    """
```

#### D. Source Verification (new module `estimation/source_verifier.py`)

Validates whether a converged layer represents a real source or a pseudo-source.

```python
class SourceVerifier:
    def __init__(self, dispersion_model, measurement_history):
        self.dispersion_model = dispersion_model
        self.history = measurement_history  # List of (position, measurement)

    def verify_source(self, estimated_params, other_confirmed_sources):
        """
        Check if estimated source is real by comparing predicted vs observed
        concentrations across measurement history.

        Method (simplified from PID-STE):
        1. For each historical measurement, predict concentration from this source
           + contributions from other confirmed sources
        2. Compute RMSE between predicted and observed
        3. If RMSE < threshold: confirmed
        4. If RMSE > threshold: pseudo-source (reject)
        """

    def cross_validate(self, all_estimates):
        """
        Check if the set of all estimates collectively explains the observation
        history better than subsets (helps detect redundant estimates).
        """
```

---

## 4. Implementation Plan

### Phase 1: Multi-Layer Particle Filter (Core)

**Files to create/modify:**
- **NEW**: `estimation/multi_layer_pf.py` — `MultiLayerParticleFilter` class
- **NEW**: `estimation/superposition_model.py` — concentration superposition logic
- **MODIFY**: `estimation/particle_filter.py` — minimal changes, add hook for external weight modifiers

**Key decisions:**
- Number of layers L: Start with L=4 (configurable). Redundant layers naturally spread out and don't converge.
- Particles per layer: 500 (total 2000, comparable to current 1000 with overhead)
- Peak suppression radius r_ps: Start at 1.0m (configurable), should be ~2× the convergence threshold

**The weight update for each layer becomes:**
```python
# For layer l:
# 1. Standard likelihood from corrected measurement
z_corrected = z_observed - Σ_{j ∈ confirmed, j≠l} R(rk|θ_j)
w_obs = sensor_model.probability(z_corrected, predicted_concentration_layer_l)

# 2. Peak suppression
w_ps = Π_{j≠l} [1 - exp(-dist(particle_xy, mean_j)² / (2·r_ps²))]

# 3. Combined
w_total = w_obs * w_ps
```

### Phase 2: Exploration Strategy

**Files to modify:**
- **MODIFY**: `planning/rrt.py` — Add multi-source branch information with OIC/RSC/REC
- **MODIFY**: `planning/global_planner.py` — Frontier utility considers unexplained concentration areas

**OIC (Observation Intensity Correction):**
```python
# At candidate position v:
residual = max(0, z_predicted_total - Σ R(v|θ_confirmed))
OIC(v) = 1 + β · residual  # β is a tuning parameter
# Higher residual = more unexplained concentration = more interesting
```

**RSC (Redundant Sampling Correction):**
```python
# measurement_positions: history of all positions visited
min_dist = min(||v - p|| for p in measurement_positions)
RSC(v) = 1 - exp(-min_dist² / (2·r_rsc²))
# Positions far from previous measurements get higher score
```

**REC (Repeat Exploring Correction):**
```python
# confirmed_sources: list of confirmed source locations
min_dist_to_confirmed = min(||v - s|| for s in confirmed_sources)
REC(v) = 1 - exp(-min_dist_to_confirmed² / (2·r_rec²))
# Positions far from confirmed sources get higher score
```

### Phase 3: Source Verification & Convergence

**Files to create:**
- **NEW**: `estimation/source_verifier.py` — Pseudo-source detection

**Convergence criteria for multi-source:**
1. Individual layer convergence: σ_x, σ_y < threshold (same as current)
2. Source verification: Converged layer passes verification → marked as confirmed
3. Global convergence: All layers either confirmed or dispersed (non-clustered)

**Simplified verification (practical approach):**
```python
def verify_source(self, estimate, measurement_history, confirmed_sources):
    errors = []
    for pos, z_obs in measurement_history[-N_recent:]:
        # Predicted from this source + confirmed sources
        z_pred = dispersion_model.compute_concentration(pos, estimate_loc, estimate_Q)
        for cs in confirmed_sources:
            z_pred += dispersion_model.compute_concentration(pos, cs_loc, cs_Q)
        errors.append((z_obs - z_pred) ** 2)
    rmse = sqrt(mean(errors))
    return rmse < verification_threshold
```

### Phase 4: Main Node Integration

**Files to modify:**
- **NEW**: `igdm_multi.py` — New main node for multi-source mode
- **MODIFY**: `setup.py` — Add `start_multi` entry point

**Main loop changes:**
```python
def take_step(self):
    # 1. Update multi-layer PF
    self.multi_pf.update(sensor_value, position)

    # 2. Check for newly converged layers
    for layer in self.multi_pf.layers:
        if layer_converged(layer) and not layer.confirmed:
            if self.verifier.verify_source(layer.estimate, self.history):
                layer.confirmed = True
                self.multi_pf.confirmed_sources.append(layer.estimate)
                self.get_logger().info(f"Source confirmed: {layer.estimate}")

    # 3. Plan (with multi-source corrections)
    if mode == 'LOCAL':
        next_pos = rrt.get_next_move_multi(position, self.multi_pf)
    elif mode == 'GLOBAL':
        next_pos = global_planner.plan_multi(position, self.multi_pf)

    # 4. Check global convergence
    if self.multi_pf.all_sources_resolved():
        self.finish()
```

---

## 5. New Parameters

```yaml
# Multi-source parameters
num_layers: 4                    # Number of particle filter layers (L ≥ expected sources)
particles_per_layer: 500         # Particles per layer
peak_suppression_radius: 1.0     # r_ps for peak suppression (meters)
verification_threshold: 5.0      # RMSE threshold for source verification
verification_history_size: 20    # Number of recent measurements for verification
oic_beta: 0.5                   # OIC sensitivity parameter
rsc_radius: 1.0                 # RSC redundant sampling radius
rec_radius: 2.0                 # REC repeat exploring radius
max_sources: 4                   # Upper bound on expected sources
layer_convergence_sigma: 0.5     # Per-layer convergence threshold
```

---

## 6. Computational Cost Analysis

| Component | Current (1 source) | Multi-source (4 layers) | Notes |
|-----------|-------------------|------------------------|-------|
| PF update | 30ms (1000 particles) | 4×20ms = 80ms (500/layer) | Parallelizable |
| Peak suppression | N/A | 4×2ms = 8ms | Simple distance calc |
| Dijkstra | 15ms (cached) | 15ms (shared cache) | Same map, shared |
| RRT MI calc | 380ms | ~500ms | OIC/RSC/REC are cheap |
| Verification | N/A | 10ms (when triggered) | Only at convergence |
| **Total step** | **~750ms** | **~1000ms** | ~33% increase |

The computational overhead is manageable because:
- Dijkstra distance maps are shared across all layers (same sensor position)
- Peak suppression is a simple O(N×L) distance computation
- OIC/RSC/REC are scalar multipliers on existing BI values

---

## 7. Implementation Order & Milestones

### Milestone 1: Multi-Layer PF (without exploration changes)
1. Create `MultiLayerParticleFilter` wrapping existing `ParticleFilter`
2. Implement peak suppression weights
3. Implement OIC (observation intensity correction)
4. Test with known multi-source simulation data
5. Visualize multiple layer estimates in RViz (different colors per layer)

### Milestone 2: Source Verification
1. Create `SourceVerifier` with RMSE-based verification
2. Add measurement history tracking
3. Implement confirmed source management
4. Test pseudo-source rejection

### Milestone 3: Exploration Strategy
1. Modify RRT branch information with OIC/RSC/REC corrections
2. Modify global planner frontier utility for multi-source
3. Add "post-confirmation exploration" mode — after confirming a source, encourage movement away from it

### Milestone 4: Integration & Testing
1. Create `igdm_multi.py` main node
2. Multi-source RViz visualization (color-coded layers, confirmed source markers)
3. End-to-end testing with GADEN multi-source scenarios
4. Parameter tuning

---

## 8. Which Paper's Approach Fits Best?

**Recommended: Hybrid of PSPF + ADE-PSPF (RAS)**

| Aspect | Best Source | Rationale |
|--------|-----------|-----------|
| Multi-layer PF | PSPF | Cleanest formulation, minimal changes to existing PF |
| Peak suppression | PSPF | Well-tested, simple to implement |
| Exploration strategy | ADE-PSPF (RAS) | OIC/RSC/REC directly extend our existing RRT planner |
| Source verification | PID-STE (simplified) | RMSE check is practical; skip full Poisson kriging |
| DE optimization | Skip for now | Our MCMC + regularized resampling is sufficient |

The PSPF paper gives us the multi-layer estimation foundation. The ADE-PSPF paper gives us the exploration strategy that maps naturally onto our existing RRT-Infotaxis + Global Planner architecture. PID-STE contributes the verification concept but we simplify it from Poisson kriging to direct RMSE comparison.

---

## 9. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Peak suppression too aggressive → layers avoid real sources | High | Adaptive r_ps that decreases as layers converge |
| Superposition makes likelihood multimodal → slow convergence | Medium | More particles per layer, DE mutation (Phase 2) |
| Pseudo-sources not caught | Medium | Conservative verification + cross-validation |
| Computational overhead too high | Low | Reduce particles/layer, share Dijkstra cache |
| Source count unknown | Medium | Start with L > expected, unused layers naturally disperse |

---

## References

- Kim et al., "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search", IEEE RA-L, 2025
- Gao et al., "Robust Radiation Sources Localization Based on the Peak Suppressed Particle Filter for Mixed Multi-Modal Environments", Sensors, 2018
- Bai et al., "Multi-source term estimation based on parallel particle filtering and dynamic state space in unknown radiation environments", Building & Environment, 2023
- Bai et al., "Autonomous radiation source searching using an adaptive differential evolution particle-suppressed particle filter", Robotics & Autonomous Systems, 2023

### Web Sources
- [Active Sensing Strategy: Multi-Modal, Multi-Robot Source Localization](https://arxiv.org/html/2407.01308v1)
- [Robotic Gas Source Localization With Probabilistic Mapping](https://dl.acm.org/doi/10.1109/TRO.2024.3426368)
- [Multi-gas source localisation and mapping by flocking robots](https://www.researchgate.net/publication/365241827_Multi-gas_source_localisation_and_mapping_by_flocking_robots)
- [PSPF Paper (Open Access)](https://www.mdpi.com/1424-8220/18/11/3784)
- [PID-STE Paper](https://www.sciencedirect.com/science/article/abs/pii/S0360132323003086)
