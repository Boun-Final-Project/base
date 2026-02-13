# Multi-Source Gas Source Localization: Comprehensive Recommendations

**Target**: Support 3-5 sources with unknown number of sources
**Constraint**: No wind (IGDM dispersion model)
**Author**: Claude Analysis
**Date**: February 2026

---

## Table of Contents

1. [Existing Research & Implementations](#existing-research--implementations)
2. [Recommended Approaches](#my-recommended-approaches-ranked)
3. [Implementation Roadmap](#implementation-roadmap-pspf-approach)
4. [Key Design Decisions](#key-design-decisions-needed)
5. [References](#references)

---

## Existing Research & Implementations

### Key Papers (Directly Relevant)

1. **Peak Suppressed Particle Filter (PSPF)** - [MDPI Sensors 2018](https://www.mdpi.com/1424-8220/18/11/3784)
   - Addresses multi-modal radiation fields with **unknown source count**
   - Uses: Multi-layer PF + Mean-shift clustering + Peak suppression
   - Solves particle degeneracy in narrow multi-source regions
   - **Highly relevant** to your indoor multi-source problem

2. **Parallel Particle Filtering with Dynamic State Space** - [ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0360132323003086)
   - **Incrementally constructs state space** to adapt to varying source numbers
   - Parallel particle weights across layers
   - Good for unknown number of sources

3. **Regularized Particle Filter for Unknown Sources** - [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S1738573324004194)
   - Sources whose **number and parameters are not known in advance**
   - Information-driven trajectory (similar to your RRT-Infotaxis)
   - Dual-mode: entropy minimization + variance reduction

4. **Multi-Robot PSO-PF** - [OAEP 2023](https://www.oaepublish.com/articles/ir.2023.38)
   - PSO moves particles to high-likelihood areas
   - Overcomes particle degradation
   - Applicable to single robot with swarm-inspired resampling

5. **Multi-Robot Multi-Source Term Estimation (MRMSTE)** - [ResearchGate 2022](https://www.researchgate.net/publication/365241827_Multi-gas_source_localisation_and_mapping_by_flocking_robots)
   - Hybrid Bayesian inference with **source birth, removal, and merging**
   - Physics-informed state transitions
   - Active sensing for optimal measurement positions

### Review Papers

- [Gas Source Localization Review - Wiley 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22109) - Comprehensive overview
- [Odor Source Localization Review - PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9748100/) - Multi-robot swarm algorithms

---

## My Recommended Approaches (Ranked)

### Approach 1: Peak Suppressed Particle Filter (PSPF) ⭐ RECOMMENDED

**Based on**: MDPI 2018 paper - proven for multi-modal unknown source count

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                    PSPF Framework                        │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Standard PF      → Find strongest mode        │
│  Mean-Shift Clustering     → Extract source estimate    │
│  Peak Suppression          → Subtract contribution      │
│  Layer 2: PF on residual   → Find next mode             │
│  ... Repeat until residual < threshold                  │
└─────────────────────────────────────────────────────────┘
```

**Key Innovation**: Instead of particles representing ALL sources (exponential complexity), use **layered particle filters** where each layer targets one source.

**Algorithm**:
1. Run standard particle filter until convergence
2. Apply mean-shift clustering to extract dominant mode
3. Confirm source: add (x, y, Q) to confirmed list
4. Apply peak suppression: subtract confirmed source contribution from future measurements
5. If residual energy > threshold, start new PF layer
6. Repeat until residual < threshold or max_sources reached

**Implementation**:
```python
class PeakSuppressedParticleFilter:
    def __init__(self, num_particles, max_sources=5):
        self.layers = []  # List of ParticleFilter instances
        self.confirmed_sources = []
        self.max_sources = max_sources
        self.residual_threshold = 1.0  # ppm

    def update(self, measurement, position):
        # 1. Compute residual (subtract confirmed sources)
        residual = self._compute_residual(measurement, position)

        # 2. Update active layer
        if not self.layers:
            self.layers.append(ParticleFilter(...))

        self.layers[-1].update(residual, position)

        # 3. Check for mode convergence using Mean-Shift
        modes = self._mean_shift_clustering(self.layers[-1])

        if self._mode_converged(modes[0]):
            # 4. Confirm source and apply peak suppression
            self.confirmed_sources.append(modes[0])

            # 5. Check if more sources exist
            new_residual = self._compute_residual(measurement, position)
            if new_residual > self.threshold and len(self.confirmed_sources) < self.max_sources:
                self.layers.append(ParticleFilter(...))  # New layer

    def _compute_residual(self, measurement, position):
        """Subtract contribution from all confirmed sources."""
        predicted = 0.0
        for source in self.confirmed_sources:
            predicted += self.igdm.compute_concentration(
                position, (source['x'], source['y']), source['Q']
            )
        return max(0, measurement - predicted)

    def _mean_shift_clustering(self, pf):
        """Extract dominant modes from particle distribution."""
        from sklearn.cluster import MeanShift
        particles, weights = pf.get_particles()
        # Weight-aware mean-shift
        ms = MeanShift(bandwidth=0.5)
        ms.fit(particles[:, :2], sample_weight=weights)
        return ms.cluster_centers_
```

**Pros**:
- Proven in literature for multi-modal fields
- Scales O(K) not O(K^max) like multi-hypothesis approaches
- Mean-shift provides robust mode extraction
- Reuses your existing ParticleFilter class
- Natural termination via residual threshold

**Cons**:
- Sequential (greedy) - can miss weak sources near strong ones
- Requires mean-shift clustering implementation
- Error propagation if early estimates are poor

**Complexity**: O(K × N × D) where K=sources, N=particles, D=Dijkstra

---

### Approach 2: Parallel Multi-Layer Particle Filter

**Based on**: ScienceDirect 2023 - Dynamic state space construction

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│              Parallel Multi-Layer PF                     │
├─────────────────────────────────────────────────────────┤
│  Hypothesis 1: 1 source  →  PF_1 (N particles)          │
│  Hypothesis 2: 2 sources →  PF_2 (N particles each)     │
│  Hypothesis 3: 3 sources →  PF_3 (N particles each)     │
│  ...                                                     │
│  Model Evidence: Compare marginal likelihoods            │
│  Output: Hypothesis with highest evidence                │
└─────────────────────────────────────────────────────────┘
```

**Key Innovation**: Run **parallel hypotheses** for different source counts, use Bayesian model comparison to select.

**Implementation**:
```python
class ParallelMultiLayerPF:
    def __init__(self, num_particles, max_sources=5):
        self.hypotheses = {}  # {k: MultiSourcePF for k sources}
        self.model_weights = {}  # Bayesian model evidence

    def update(self, measurement, position):
        # Update all hypotheses in parallel
        for k in range(1, self.max_sources + 1):
            if k not in self.hypotheses:
                self.hypotheses[k] = self._create_k_source_pf(k)

            # Update and compute marginal likelihood
            likelihood = self.hypotheses[k].update(measurement, position)
            self.model_weights[k] *= likelihood

        # Normalize model weights
        self._normalize_model_weights()

    def get_best_estimate(self):
        best_k = max(self.model_weights, key=self.model_weights.get)
        return self.hypotheses[best_k].get_estimate()

    def _create_k_source_pf(self, k):
        """Create a particle filter for k sources."""
        # Each particle has k*3 dimensions: [x1,y1,Q1, x2,y2,Q2, ...]
        return MultiSourceParticleFilter(
            num_particles=self.num_particles,
            num_sources=k,
            search_bounds=self.bounds
        )
```

**Pros**:
- Principled Bayesian model selection
- Non-greedy (considers all hypotheses)
- Can revise source count estimates as more data arrives
- Provides posterior over number of sources

**Cons**:
- Higher memory: O(K_max × N × 3K)
- More computation: O(K_max × N × K × D)
- Particle representation for k sources needs careful design
- Symmetry/label switching for multi-source particles

**Complexity**: O(K_max × N × K × D) - can be parallelized

---

### Approach 3: Sequential Detection with Information-Driven Exploration

**Based on**: Your existing RRT-Infotaxis + Residual analysis

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│           Sequential + Information-Driven                │
├─────────────────────────────────────────────────────────┤
│  Phase 1: RRT-Infotaxis → Converge on strongest source  │
│  Phase 2: Confirm & Store source estimate               │
│  Phase 3: Compute residual field                        │
│  Phase 4: RRT-Infotaxis on residual → Next source       │
│  Phase 5: Repeat until residual energy < threshold      │
└─────────────────────────────────────────────────────────┘
```

**Key Innovation**: Reuse your entire existing pipeline, just wrap it.

**Implementation**:
```python
class SequentialInfotaxisLocalizer:
    def __init__(self, base_params):
        self.confirmed_sources = []
        self.current_localizer = None
        self.residual_threshold = 1.0  # ppm
        self.base_params = base_params

    def initialize(self):
        """Start search for first source."""
        self.current_localizer = RRTInfotaxisNode(self.base_params)

    def take_step(self):
        # Modify measurement with residual
        residual_measurement = self._apply_residual_correction(
            self.current_localizer.sensor_raw_value,
            self.current_localizer.current_position
        )

        # Inject residual measurement
        self.current_localizer.sensor_raw_value = residual_measurement

        # Run standard Infotaxis step
        self.current_localizer.take_step()

        # Check convergence
        if self.current_localizer.converged:
            self._confirm_current_source()

            if self._residual_energy() > self.residual_threshold:
                self._start_new_search()
            else:
                self.all_sources_found = True

    def _apply_residual_correction(self, measurement, position):
        predicted = 0.0
        for src in self.confirmed_sources:
            predicted += self.igdm.compute_concentration(
                position, (src['x'], src['y']), src['Q']
            )
        return max(0, measurement - predicted)

    def _confirm_current_source(self):
        means, _ = self.current_localizer.particle_filter.get_estimate()
        self.confirmed_sources.append({
            'x': means['x'],
            'y': means['y'],
            'Q': means['Q']
        })

    def _start_new_search(self):
        # Reset particle filter for next source
        self.current_localizer.particle_filter.reset()
        # Optionally: exclude confirmed regions from search
```

**Pros**:
- Minimal code changes (wraps existing system)
- Leverages proven RRT-Infotaxis planning
- Easy to implement and test
- Fast path to working multi-source system

**Cons**:
- Greedy/sequential - order dependent
- May need to re-explore areas after each confirmation
- Cannot revise earlier estimates

**Complexity**: O(K × T_converge) - same as running K separate searches

---

### Approach 4: RJMCMC (Reversible Jump MCMC)

**Based on**: Theoretical gold standard for trans-dimensional inference

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                    RJMCMC Framework                      │
├─────────────────────────────────────────────────────────┤
│  State: [k, θ₁, θ₂, ..., θₖ] where k = num sources     │
│  Birth Move: Add new source at random location          │
│  Death Move: Remove low-weight source                   │
│  Update Move: Perturb existing source parameters        │
│  Acceptance: Metropolis-Hastings with Jacobian          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Sketch**:
```python
class RJMCMCMultiSourceFilter:
    def __init__(self, max_sources=5):
        self.max_sources = max_sources
        self.current_k = 1  # Current number of sources
        self.sources = [self._sample_prior()]  # Initial source

    def update(self, measurement, position):
        for _ in range(self.num_mcmc_iterations):
            move_type = np.random.choice(
                ['birth', 'death', 'update'],
                p=[0.2, 0.2, 0.6]
            )

            if move_type == 'birth' and self.current_k < self.max_sources:
                self._birth_move(measurement, position)
            elif move_type == 'death' and self.current_k > 1:
                self._death_move(measurement, position)
            else:
                self._update_move(measurement, position)

    def _birth_move(self, measurement, position):
        """Add a new source - requires careful Jacobian calculation."""
        new_source = self._sample_birth_proposal()
        proposal_sources = self.sources + [new_source]

        # Compute acceptance ratio with dimension-matching Jacobian
        log_acceptance = (
            self._log_likelihood(measurement, position, proposal_sources) -
            self._log_likelihood(measurement, position, self.sources) +
            self._log_birth_jacobian(new_source)
        )

        if np.log(np.random.random()) < log_acceptance:
            self.sources = proposal_sources
            self.current_k += 1
```

**Pros**:
- Theoretically principled
- Full joint posterior over parameters AND source count
- Asymptotically exact

**Cons**:
- Very difficult to tune acceptance ratios
- High-dimensional mixing is slow
- Birth/death proposals need careful design
- Not recommended unless you have MCMC expertise

**Complexity**: O(N_iterations × K × D) per measurement

---

## Recommendation Summary

| Approach | Difficulty | Time to Implement | Accuracy | Recommended For |
|----------|------------|-------------------|----------|-----------------|
| **PSPF** | Medium | 2-3 weeks | High | ⭐ Best overall choice |
| Parallel Multi-Layer | Medium-High | 3-4 weeks | High | Research/comparison |
| Sequential Infotaxis | Low | 1 week | Medium | Quick prototype |
| RJMCMC | Very High | 4-6 weeks | Highest | Academic research |

**My Recommendation**: Start with **PSPF** because:
1. Proven in recent literature (2018-2023)
2. Builds on your existing single-source infrastructure
3. Natural extension of particle filter you already have
4. Mean-shift clustering is well-understood (sklearn available)
5. Handles unknown source count elegantly via residual threshold

---

## Implementation Roadmap (PSPF Approach)

### Week 1: Core Infrastructure
**Files to modify**:
- `efe_igdm/estimation/particle_filter.py` - Add residual measurement support
- `efe_igdm/estimation/igdm_gas_model.py` - Add `compute_concentration_multi()`

**New methods**:
```python
# In igdm_gas_model.py
def compute_concentration_multi(self, position, sources_list):
    """
    Compute total concentration from multiple sources.

    Args:
        position: (x, y) sensor position
        sources_list: List of {'x': x, 'y': y, 'Q': Q} dicts

    Returns:
        Total concentration (superposition)
    """
    total = 0.0
    for src in sources_list:
        total += self.compute_concentration(
            position, (src['x'], src['y']), src['Q']
        )
    return total
```

### Week 2: Mean-Shift Clustering
**New file**: `efe_igdm/estimation/mode_extraction.py`

```python
from sklearn.cluster import MeanShift
import numpy as np

def extract_modes_from_particles(particles, weights, bandwidth=0.5):
    """
    Extract dominant modes from weighted particle distribution.

    Args:
        particles: (N, 3) array of [x, y, Q]
        weights: (N,) array of particle weights
        bandwidth: Mean-shift kernel bandwidth (meters)

    Returns:
        modes: List of {'x', 'y', 'Q', 'weight'} dicts
    """
    # Use only position for clustering
    positions = particles[:, :2]

    ms = MeanShift(bandwidth=bandwidth)
    labels = ms.fit_predict(positions, sample_weight=weights)

    modes = []
    for label in np.unique(labels):
        mask = labels == label
        cluster_weight = np.sum(weights[mask])

        # Weighted mean for this cluster
        mode = {
            'x': np.average(particles[mask, 0], weights=weights[mask]),
            'y': np.average(particles[mask, 1], weights=weights[mask]),
            'Q': np.average(particles[mask, 2], weights=weights[mask]),
            'weight': cluster_weight
        }
        modes.append(mode)

    # Sort by weight (strongest mode first)
    modes.sort(key=lambda m: m['weight'], reverse=True)
    return modes
```

### Week 3: PSPF Integration
**New file**: `efe_igdm/estimation/pspf.py`

Core class implementing the Peak Suppressed Particle Filter with:
- Layer management
- Residual computation
- Mode confirmation logic
- Source list management

### Week 4: Main Node Integration
**Files to modify**:
- `efe_igdm/igdm.py` - Replace single PF with PSPF wrapper
- `efe_igdm/visualization/marker_visualizer.py` - Multiple source visualization

**Changes to igdm.py**:
```python
# Replace:
self.particle_filter = ParticleFilter(...)

# With:
self.multi_source_filter = PeakSuppressedParticleFilter(
    num_particles=self.params['number_of_particles'],
    max_sources=5,
    residual_threshold=1.5,  # ppm
    ...
)
```

---

## Key Design Decisions Needed

1. **Residual Threshold**: What concentration level indicates "no more sources"?
   - Suggested: 1.0-2.0 ppm (depends on sensor noise σ_env = 1.5 ppm)
   - Too low → false positives, too high → miss weak sources

2. **Mode Convergence Criterion**: When is a source "confirmed"?
   - Suggested: σ_x, σ_y < 0.5m AND stable for 10+ steps
   - Alternative: Mode weight > 0.6 for 5+ consecutive updates

3. **Maximum Sources**: Hard limit for computational efficiency
   - Your answer: 3-5 sources ✓
   - Recommendation: Set to 5, let residual threshold handle early termination

4. **Mean-Shift Bandwidth**: Controls clustering resolution
   - Suggested: 0.5-1.0m (should be > expected localization error)
   - Too small → fragments modes, too large → merges nearby sources

5. **Planning Adaptation**: Should RRT-Infotaxis change after each source?
   - Option A: Search entire space each time (simpler)
   - Option B: Exclude confirmed source regions (more efficient)
   - Recommendation: Start with Option A, optimize later

---

## Verification Plan

### Test Scenarios
1. **Single source** - Verify backward compatibility
2. **Two sources, far apart** (>5m) - Should find both easily
3. **Two sources, close** (<2m) - Tests mode separation
4. **Three sources, varying strengths** - Tests residual handling
5. **No source** (background only) - Should converge to "no source"

### Metrics
- **Source count accuracy**: |detected - actual|
- **Position error**: Euclidean distance to true source
- **Release rate error**: |Q_est - Q_true| / Q_true
- **Total convergence time**: Steps until all sources found
- **False positive rate**: Spurious source detections

### Simulation Setup
1. GADEN with 2-3 known source locations
2. Vary source strengths (Q = 20, 50, 100 ppm·m³/s)
3. Run 10+ trials per configuration
4. Compare against single-source baseline

---

## References

### Primary Papers
- [PSPF Paper (MDPI 2018)](https://www.mdpi.com/1424-8220/18/11/3784) - Core algorithm inspiration
- [Parallel PF (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0360132323003086) - Dynamic state space
- [Regularized PF (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1738573324004194) - Unknown source count

### Review Papers
- [GSL Review (Wiley 2022)](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22109) - Comprehensive overview
- [OSL Swarm Review (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9748100/) - Multi-robot algorithms

### Additional Resources
- [Multi-Robot PSO-PF (OAEP 2023)](https://www.oaepublish.com/articles/ir.2023.38) - Swarm-inspired resampling
- [MRMSTE Framework (ResearchGate 2022)](https://www.researchgate.net/publication/365241827_Multi-gas_source_localisation_and_mapping_by_flocking_robots) - Source birth/death
