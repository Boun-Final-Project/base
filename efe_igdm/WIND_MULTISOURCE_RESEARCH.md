# Wind-Aware Multi-Source Gas Source Localization: Research Analysis

**Question**: Can we combine multiple sources + wind + unknown indoor map?
**Short Answer**: This combination does NOT exist in literature - it's a research gap.

**Author**: Claude Analysis
**Date**: February 2026

---

## Table of Contents

1. [Existing Research Landscape](#existing-research-landscape)
2. [Closest Existing Approaches](#closest-existing-approaches)
3. [The Research Gap](#the-research-gap)
4. [Potential Combined Framework](#potential-combined-framework)
5. [Key Challenges](#key-challenges)
6. [Recommended Phased Approach](#recommended-phased-approach)
7. [References](#references)

---

## Existing Research Landscape

### What Combinations Exist?

| Multi-Source | Wind/Advection | Unknown Map | Exists? | Key Paper |
|:------------:|:--------------:|:-----------:|:-------:|-----------|
| ❌ Single | ✅ Yes | ✅ Yes | ✅ **YES** | Probabilistic Mapping + Online Dispersion (IEEE T-RO 2024) |
| ✅ Multiple | ❌ No | ✅ Yes | ✅ **YES** | PSPF, IGDM-based approaches |
| ❌ Single | ✅ Yes | ❌ Known | ✅ **YES** | GW-GMRF (MDPI 2023) |
| ✅ Multiple | ✅ Yes | ❌ Known | ⚠️ **Partial** | Multi-source flocking (2022) |
| ✅ Multiple | ✅ Yes | ✅ Yes | ❌ **NO** | **RESEARCH GAP** |

### Key Observations

1. **Single source + Wind + Unknown map**: Solved by filament-based online dispersion simulation
2. **Multiple sources + No wind**: Solved by PSPF, parallel PF, sequential detection
3. **Wind estimation + Known map**: Solved by GW-GMRF and advection-diffusion models
4. **Full combination**: Does NOT exist in published literature

---

## Closest Existing Approaches

### 1. Filament Model with Online Dispersion Simulation

**Paper**: [Robotic Gas Source Localization with Probabilistic Mapping and Online Dispersion Simulation](https://arxiv.org/html/2304.08879v3)
**Published**: IEEE Transactions on Robotics, 2024

**Key Features**:
- Performs GSL in realistic indoor environments with obstacles and turbulent flow
- Robot carries gas sensor AND anemometer (wind sensor)
- Builds gas concentration map from measurements online
- Runs **real-time filament simulation** from candidate source positions
- Compares simulated vs observed concentration to compute likelihood
- Uses particle filter for Bayesian source estimation

**How Filament Model Works**:
```
Chemical release → Sequence of puffs → Each puff = n filaments
Each filament = 3D Gaussian distribution of molecules

Dispersion phenomena:
1. Advection: Filaments transported by wind
2. Turbulent diffusion: Filaments spread based on eddy size
3. Molecular diffusion: Gaussian spread over time
```

**Mathematical Model**:
```
Concentration at position r, time t:
C(r,t) = Σ_filaments G(r - r_filament(t), σ(t))

Where:
- r_filament(t) = initial_pos + ∫wind(τ)dτ  (advection)
- σ(t) = σ₀ + diffusion_rate × t  (spreading)
```

**Limitation**: **Single source only** - no mechanism for multiple sources

**Code/Implementation**: Uses GADEN simulator framework (ROS-based)

---

### 2. GW-GMRF (Gas-Wind Gaussian Markov Random Field)

**Paper**: [Information-Driven Gas Distribution Mapping for Autonomous Mobile Robots](https://www.mdpi.com/1424-8220/23/12/5387)
**Published**: MDPI Sensors, June 2023

**Key Features**:
- Models gas concentration AND wind as coupled random fields
- Environment treated as 2D lattice of cells
- Adjacent cells have probabilistic constraints (Markov property)
- Accounts for obstacles - wind cannot flow through walls
- Combines gas and wind measurements for advection-aware estimation

**Mathematical Framework**:
```
Joint distribution over gas (g) and wind (w):
P(g, w | observations) ∝ P(observations | g, w) × P(g | w) × P(w)

Where:
- P(w) encodes wind field smoothness constraints
- P(g | w) encodes advection-diffusion physics
- Obstacles create hard constraints on wind flow
```

**Wind Field Estimation**:
```python
# Simplified GW-GMRF concept
class GW_GMRF:
    def __init__(self, grid_size, obstacle_map):
        self.gas_field = np.zeros(grid_size)
        self.wind_field = np.zeros((*grid_size, 2))  # (u, v) components
        self.obstacle_map = obstacle_map

    def update(self, position, gas_measurement, wind_measurement):
        # 1. Update wind field estimate (constrained by obstacles)
        self._update_wind_field(position, wind_measurement)

        # 2. Update gas field estimate (accounting for advection)
        self._update_gas_field(position, gas_measurement, self.wind_field)

    def _update_wind_field(self, pos, wind_obs):
        # Interpolate wind observation to nearby cells
        # Enforce: wind · normal = 0 at obstacle boundaries
        # Use GMRF smoothness prior
        pass
```

**Limitation**: **Requires known obstacle map** - not compatible with SLAM exploration

---

### 3. Advection-Diffusion PDE Models

**Paper**: [Exploration and Gas Source Localization in Advection Environments](https://www.mdpi.com/1424-8220/23/22/9232)
**Published**: MDPI Sensors, November 2023

**Key Features**:
- Uses advection-diffusion equation as process model
- Time-varying wind incorporated into gas transport
- Multi-robot coordination for exploration
- PDE-based forward model for concentration prediction

**Advection-Diffusion Equation**:
```
∂C/∂t + ∇·(wC) = D∇²C + S

Where:
- C = concentration field
- w = wind velocity vector
- D = diffusion coefficient
- S = source term (emission rate at source locations)
```

**Discretized for Computation**:
```
C(r, t+Δt) = C(r, t)
           - Δt × w·∇C           (advection term)
           + Δt × D∇²C           (diffusion term)
           + Δt × S(r)           (source term)
```

**Key Finding**: "Introducing fluctuations in wind direction makes source localization a fundamentally harder problem to solve."

**Limitation**:
- Typically assumes **single source**
- Requires CFD pre-computation or simplified wind model
- Computationally expensive for real-time

---

### 4. Multi-Source with Wind (Partial Solution)

**Paper**: [Multi-gas source localisation and mapping by flocking robots](https://www.researchgate.net/publication/365241827_Multi-gas_source_localisation_and_mapping_by_flocking_robots)
**Published**: ResearchGate, 2022

**Key Features**:
- Multi-Robot Multi-Source Term Estimation (MRMSTE)
- Hybrid Bayesian inference with source birth, removal, merging
- Physics-informed state transitions
- Wind-aware coverage control (WCC)
- Superposition-based measurement model

**Framework**:
```
Multi-source belief + Local wind → Wind-aware coverage control
                                          ↓
                              Prioritize high-likelihood regions
                                          ↓
                              Account for anisotropic plume transport
```

**Limitation**:
- Requires **known environment** for coverage control
- Multi-robot system (not single robot)
- Limited details on wind field estimation method

---

## The Research Gap

### What's Missing: Multi-Source + Wind + Unknown Map

**Why this combination is hard**:

1. **Chicken-and-egg problem**:
   - Need obstacle map to estimate wind field (wind flows around obstacles)
   - Need wind field to correctly model gas dispersion
   - Need correct gas model to localize sources
   - Need source estimates to plan exploration
   - But map is unknown at start!

2. **Coupled estimation problem**:
   ```
   Unknown variables:
   - Obstacle map M (from SLAM)
   - Wind field W (from anemometer + M)
   - Number of sources K (unknown)
   - Source parameters θ₁...θₖ (positions, rates)

   All are interdependent!
   ```

3. **Computational complexity**:
   - Online SLAM: O(N) for particle filter SLAM
   - Wind field estimation: O(grid_size²) for GMRF
   - Multi-source estimation: O(K × N_particles)
   - Combined: Very expensive for real-time

4. **Indoor wind complexity**:
   - HVAC systems create complex, time-varying patterns
   - Obstacles cause turbulence and recirculation
   - Cannot assume uniform or laminar flow
   - Local measurements may not represent global field

---

## Potential Combined Framework

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Multi-Source + Wind + Unknown Map Framework                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SENSORS                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │   LiDAR     │  │ Anemometer  │  │  Gas Sensor │                      │
│  │  (mapping)  │  │   (wind)    │  │  (MOX/PID)  │                      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
│         │                │                │                              │
│         ▼                ▼                ▼                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SLAM MODULE                                   │    │
│  │  Input: LiDAR scans, odometry                                   │    │
│  │  Output: Occupancy grid M(t), robot pose                        │    │
│  │  Method: Your existing LiDAR SLAM                               │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 WIND FIELD ESTIMATION                            │    │
│  │  Input: Local wind measurements, obstacle map M(t)              │    │
│  │  Output: Estimated wind field W(x,y,t)                          │    │
│  │  Method: GW-GMRF with incremental updates                       │    │
│  │                                                                  │    │
│  │  Constraints:                                                    │    │
│  │  - W · n = 0 at obstacle boundaries (no flow through walls)    │    │
│  │  - Spatial smoothness (GMRF prior)                              │    │
│  │  - Temporal continuity                                          │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              ADVECTION-DIFFUSION DISPERSION MODEL                │    │
│  │  Input: Source hypothesis θ, wind field W, obstacle map M       │    │
│  │  Output: Predicted concentration C(r|θ,W,M)                     │    │
│  │                                                                  │    │
│  │  Model (replaces IGDM):                                         │    │
│  │  C(r) = Σᵢ Qᵢ · K(r, rᵢ, W, M)                                 │    │
│  │                                                                  │    │
│  │  Where K is advection-diffusion kernel:                         │    │
│  │  - Accounts for wind transport                                  │    │
│  │  - Respects obstacle boundaries                                 │    │
│  │  - Handles turbulent diffusion                                  │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              MULTI-SOURCE ESTIMATION (PSPF)                      │    │
│  │  Input: Gas measurements, predicted concentrations              │    │
│  │  Output: K source estimates (x₁,y₁,Q₁), ..., (xₖ,yₖ,Qₖ)        │    │
│  │                                                                  │    │
│  │  Method: Peak Suppressed Particle Filter                        │    │
│  │  - Layer 1: Find strongest source                               │    │
│  │  - Mean-shift clustering for mode extraction                    │    │
│  │  - Peak suppression (subtract confirmed sources)                │    │
│  │  - Layer 2+: Find additional sources from residual              │    │
│  │  - Terminate when residual < threshold                          │    │
│  └─────────────────────────┬───────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RRT-INFOTAXIS PLANNING                        │    │
│  │  Input: Current source beliefs, wind field, obstacle map        │    │
│  │  Output: Next robot position                                    │    │
│  │                                                                  │    │
│  │  Modifications for wind:                                        │    │
│  │  - Information gain accounts for wind direction                 │    │
│  │  - Prefer upwind exploration (sources are upwind of detections)│    │
│  │  - Frontier selection considers wind patterns                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components in Detail

#### 1. Wind Field Estimation Module

```python
class WindFieldEstimator:
    """
    Estimates global wind field from local anemometer measurements.
    Uses Gaussian Markov Random Field with obstacle constraints.
    """

    def __init__(self, grid_resolution=0.5):
        self.resolution = grid_resolution
        self.wind_field = None  # Shape: (H, W, 2) for (u, v) components
        self.confidence = None  # Shape: (H, W) measurement confidence

    def update(self, robot_pos, wind_measurement, obstacle_map):
        """
        Incrementally update wind field estimate.

        Args:
            robot_pos: (x, y) current robot position
            wind_measurement: (u, v) local wind vector from anemometer
            obstacle_map: Current SLAM occupancy grid
        """
        # 1. Initialize/resize grid if needed
        self._ensure_grid_size(obstacle_map)

        # 2. Add observation at robot position
        gx, gy = self._world_to_grid(robot_pos)
        self._add_observation(gx, gy, wind_measurement)

        # 3. Propagate using GMRF with obstacle constraints
        self._gmrf_update(obstacle_map)

    def _gmrf_update(self, obstacle_map):
        """
        GMRF update with boundary conditions.

        Constraints:
        - Smoothness: neighboring cells have similar wind
        - Obstacles: wind · normal = 0 at boundaries
        - Mass conservation: ∇ · w ≈ 0 (incompressible flow approximation)
        """
        # Iterative solver (e.g., Gauss-Seidel)
        for iteration in range(self.max_iterations):
            for i in range(self.height):
                for j in range(self.width):
                    if obstacle_map[i, j] > 0:
                        self.wind_field[i, j] = [0, 0]  # No wind inside obstacles
                        continue

                    # Average of neighbors (GMRF prior)
                    neighbors = self._get_valid_neighbors(i, j, obstacle_map)
                    if neighbors:
                        prior = np.mean([self.wind_field[n] for n in neighbors], axis=0)

                        # Blend prior with observation (if any)
                        if self.confidence[i, j] > 0:
                            obs = self.observations[i, j]
                            self.wind_field[i, j] = (
                                self.confidence[i, j] * obs +
                                (1 - self.confidence[i, j]) * prior
                            )
                        else:
                            self.wind_field[i, j] = prior

    def get_wind_at(self, position):
        """Interpolate wind vector at arbitrary position."""
        gx, gy = self._world_to_grid(position)
        # Bilinear interpolation
        return self._bilinear_interpolate(gx, gy)
```

#### 2. Advection-Diffusion Dispersion Model

```python
class AdvectionDiffusionModel:
    """
    Replaces IGDM for wind-aware concentration prediction.
    """

    def __init__(self, diffusion_coeff=0.1, sigma_m=1.5):
        self.D = diffusion_coeff  # Diffusion coefficient (m²/s)
        self.sigma_m = sigma_m    # Base dispersion (for fallback)

    def compute_concentration(self, sensor_pos, source_pos, release_rate,
                              wind_field, obstacle_map, time_horizon=10.0):
        """
        Compute concentration using advection-diffusion.

        For steady-state with uniform wind w:
        C(r) = Q / (4πD|r-r₀|) × exp(-(|r-r₀| - w·(r-r₀)/|w|)² / (4Dt))

        For complex wind fields, use numerical integration or filament model.
        """
        # Get wind at source location
        w = wind_field.get_wind_at(source_pos)
        wind_speed = np.linalg.norm(w)

        if wind_speed < 0.1:  # Low wind - use IGDM-like model
            return self._diffusion_only(sensor_pos, source_pos, release_rate)

        # Vector from source to sensor
        r_vec = np.array(sensor_pos) - np.array(source_pos)
        r_dist = np.linalg.norm(r_vec)

        if r_dist < 0.01:
            return release_rate  # At source

        # Wind direction unit vector
        w_hat = w / wind_speed

        # Downwind distance (projection onto wind direction)
        downwind = np.dot(r_vec, w_hat)

        # Crosswind distance (perpendicular to wind)
        crosswind = np.sqrt(r_dist**2 - downwind**2)

        # Gaussian plume model (simplified advection-diffusion solution)
        if downwind <= 0:
            # Sensor is upwind of source - very low concentration
            return 0.001 * release_rate * np.exp(-r_dist / self.sigma_m)

        # Dispersion parameters (grow with downwind distance)
        sigma_y = 0.1 * downwind + 0.5  # Lateral dispersion
        sigma_z = 0.1 * downwind + 0.5  # Vertical dispersion (if 3D)

        # Gaussian plume concentration
        C = (release_rate / (2 * np.pi * wind_speed * sigma_y * sigma_z)) * \
            np.exp(-0.5 * (crosswind / sigma_y)**2)

        return max(C, 0.0)

    def compute_concentrations_batch(self, sensor_pos, particle_sources,
                                      release_rates, wind_field, obstacle_map):
        """Vectorized version for particle filter."""
        concentrations = np.zeros(len(particle_sources))

        for i, (source, Q) in enumerate(zip(particle_sources, release_rates)):
            concentrations[i] = self.compute_concentration(
                sensor_pos, source, Q, wind_field, obstacle_map
            )

        return concentrations

    def _diffusion_only(self, sensor_pos, source_pos, release_rate):
        """Fallback to IGDM-like model for low wind."""
        d = np.linalg.norm(np.array(sensor_pos) - np.array(source_pos))
        return release_rate * np.exp(-d**2 / (2 * self.sigma_m**2))
```

#### 3. Modified PSPF for Wind

```python
class WindAwarePSPF:
    """
    Peak Suppressed Particle Filter with wind-aware dispersion.
    """

    def __init__(self, num_particles, max_sources=5):
        self.num_particles = num_particles
        self.max_sources = max_sources
        self.confirmed_sources = []
        self.layers = []

        # Wind-aware dispersion model
        self.dispersion_model = AdvectionDiffusionModel()
        self.wind_estimator = WindFieldEstimator()

    def update(self, measurement, position, wind_measurement, obstacle_map):
        """
        Update with gas measurement, wind measurement, and current map.
        """
        # 1. Update wind field estimate
        self.wind_estimator.update(position, wind_measurement, obstacle_map)

        # 2. Compute residual
        residual = self._compute_residual(measurement, position)

        # 3. Update active particle filter layer
        if not self.layers:
            self.layers.append(self._create_layer())

        self.layers[-1].update(
            residual, position,
            self.wind_estimator, obstacle_map
        )

        # 4. Check for mode convergence
        modes = self._mean_shift_clustering(self.layers[-1])

        if self._mode_converged(modes[0]):
            self.confirmed_sources.append(modes[0])

            # Check for more sources
            new_residual = self._compute_residual(measurement, position)
            if (new_residual > self.threshold and
                len(self.confirmed_sources) < self.max_sources):
                self.layers.append(self._create_layer())

    def _compute_residual(self, measurement, position):
        """Subtract contribution from confirmed sources."""
        predicted = 0.0
        for source in self.confirmed_sources:
            predicted += self.dispersion_model.compute_concentration(
                position,
                (source['x'], source['y']),
                source['Q'],
                self.wind_estimator,
                self.obstacle_map
            )
        return max(0, measurement - predicted)
```

---

## Key Challenges

### Technical Challenges

| Challenge | Difficulty | Description |
|-----------|------------|-------------|
| **Chicken-and-egg coupling** | Very High | Map needed for wind, wind needed for gas, gas needed for planning |
| **Indoor wind complexity** | Very High | HVAC, turbulence, recirculation zones |
| **Computational cost** | High | Real-time SLAM + wind estimation + multi-source PF |
| **Sensor fusion timing** | Medium | Synchronizing LiDAR, anemometer, gas sensor |
| **Wind field extrapolation** | High | Local measurements → global field with sparse data |
| **Turbulence modeling** | Very High | Stochastic, time-varying, hard to predict |

### Simplifying Assumptions (to make tractable)

1. **Quasi-steady wind**: Assume wind changes slowly compared to robot motion
2. **2D wind field**: Ignore vertical component (valid for ground robots)
3. **Incompressible flow**: ∇·w = 0 (mass conservation)
4. **Known HVAC locations**: If available, use as wind source priors
5. **Piecewise uniform wind**: Divide space into regions with uniform wind

---

## Recommended Phased Approach

### Phase 1: Multi-Source without Wind (Current Direction)
**Timeline**: 2-4 weeks
**Goal**: Get PSPF working with IGDM

**Deliverables**:
- Working multi-source detection (3-5 sources)
- Mean-shift mode extraction
- Residual-based termination

**Why first**: Establishes baseline, validates multi-source framework

---

### Phase 2: Add Anemometer + Local Wind
**Timeline**: 2-3 weeks
**Goal**: Use local wind measurements directly

**Changes**:
- Add anemometer sensor to robot (hardware)
- Create wind measurement subscriber (ROS)
- Simple advection: shift concentration by `wind × distance / speed`
- No global wind estimation yet

**Model**:
```python
# Simple wind-aware concentration
def concentration_with_local_wind(sensor_pos, source_pos, Q, local_wind):
    # Effective distance accounting for wind
    r = np.array(sensor_pos) - np.array(source_pos)
    wind_factor = np.dot(r, local_wind) / (np.linalg.norm(r) * np.linalg.norm(local_wind) + 0.01)

    # Reduce concentration if sensor is upwind of source
    if wind_factor < 0:  # Upwind
        adjustment = np.exp(wind_factor)  # Exponential decay
    else:  # Downwind
        adjustment = 1.0 + 0.5 * wind_factor  # Slight boost

    # Base IGDM concentration
    d = np.linalg.norm(r)
    base_conc = Q * np.exp(-d**2 / (2 * sigma_m**2))

    return base_conc * adjustment
```

**Why second**: Low implementation effort, validates wind sensor integration

---

### Phase 3: Wind Field Estimation (GW-GMRF)
**Timeline**: 3-4 weeks
**Goal**: Estimate global wind field from local measurements

**Changes**:
- Implement GW-GMRF wind estimator
- Couple with SLAM obstacle map
- Incremental updates as robot explores

**Key insight**: As robot explores and builds map, wind field estimate improves simultaneously

**Why third**: Significant research contribution, builds on Phase 1-2

---

### Phase 4: Full Advection-Diffusion Model
**Timeline**: 3-4 weeks
**Goal**: Replace IGDM with physics-based wind-aware model

**Changes**:
- Implement advection-diffusion concentration model
- Handle complex wind patterns
- Optimize for real-time performance

**Why fourth**: Most complex component, requires Phases 1-3 working

---

### Phase 5: Integration and Validation
**Timeline**: 2-3 weeks
**Goal**: Full system integration and testing

**Validation**:
- GADEN simulation with wind (already supports wind!)
- Multiple source scenarios
- Unknown environment exploration
- Compare against baselines

---

## Publication Potential

This phased approach yields **multiple publication opportunities**:

1. **Phase 1**: "Multi-Source Gas Localization in Unknown Indoor Environments using Peak Suppressed Particle Filtering" (Conference paper)

2. **Phase 2-3**: "Wind-Aware Gas Source Localization with Online Wind Field Estimation" (Conference paper)

3. **Phase 4-5**: "Simultaneous Multi-Source Localization, Mapping, and Wind Field Estimation for Indoor Gas Detection" (Journal paper - novel contribution)

The full combination (Phase 5) appears to be **unpublished** and would be a **genuine research contribution**.

---

## References

### Primary Papers (Wind-Aware GSL)

1. **Filament Model + Online Dispersion**
   - Paper: [Robotic Gas Source Localization with Probabilistic Mapping and Online Dispersion Simulation](https://arxiv.org/html/2304.08879v3)
   - Published: IEEE Transactions on Robotics, 2024
   - Key: Single source, wind-aware, unknown environment

2. **GW-GMRF**
   - Paper: [Information-Driven Gas Distribution Mapping for Autonomous Mobile Robots](https://www.mdpi.com/1424-8220/23/12/5387)
   - Published: MDPI Sensors, June 2023
   - Key: Wind field estimation, obstacle-aware

3. **Advection-Diffusion Exploration**
   - Paper: [Exploration and Gas Source Localization in Advection Environments](https://www.mdpi.com/1424-8220/23/22/9232)
   - Published: MDPI Sensors, November 2023
   - Key: PDE-based model, multi-robot

### Multi-Source Papers

4. **PSPF**
   - Paper: [Robust Radiation Sources Localization Based on Peak Suppressed Particle Filter](https://www.mdpi.com/1424-8220/18/11/3784)
   - Published: MDPI Sensors, 2018
   - Key: Multi-source, unknown count

5. **Multi-Source Flocking**
   - Paper: [Multi-gas source localisation and mapping by flocking robots](https://www.researchgate.net/publication/365241827_Multi-gas_source_localisation_and_mapping_by_flocking_robots)
   - Published: 2022
   - Key: Multi-source, wind-aware (partial)

### Review Papers

6. **Comprehensive GSL Review**
   - Paper: [Gas source localization and mapping with mobile robots: A review](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22109)
   - Published: Journal of Field Robotics, 2022

### Simulation

7. **GADEN Simulator**
   - Paper: [GADEN: A 3D Gas Dispersion Simulator for Mobile Robot Olfaction](https://www.mdpi.com/1424-8220/17/7/1479)
   - Published: MDPI Sensors, 2017
   - Key: ROS-based, supports wind, filament model

---

## Summary

**Your intuition is correct**: The combination of **multiple sources + wind + unknown indoor map** does not exist in published literature.

**This is a research opportunity**, not a limitation. A phased approach can:
1. Build on existing methods (PSPF, GW-GMRF, filament models)
2. Yield incremental publications
3. Result in a novel, publishable full system

**Recommended next step**: Implement Phase 1 (multi-source PSPF) while acquiring an anemometer sensor for Phase 2.
