# Incorporating Wind into the Gas Dispersion Model

## Current Model (No Wind)

```
R(rk|θ) = Qm · exp(-c_obs² / (2σm²))
```

- `c_obs`: obstacle-aware Dijkstra distance (isotropic, 8-connectivity)
- Gas spreads equally in all directions — wind is ignored
- ~2-5ms for 1000 particles (Numba JIT Dijkstra + vectorized lookup)

---

## Option 1: Wind-Biased Dijkstra Distance

**Idea**: Modify Dijkstra edge costs so moving downwind is cheap and upwind is expensive. The concentration equation stays the same — only the "effective distance" changes.

### Edge Cost Formula

```
cost(i → j) = base_cost × (1 - α · dot(wind_dir, step_dir))
```

Where:
- `base_cost` = resolution (straight) or resolution×√2 (diagonal)
- `wind_dir` = normalized GMRF wind vector at cell i
- `step_dir` = normalized direction from cell i to cell j
- `α ∈ [0, 0.9]` = wind influence strength (0 = no wind, 0.9 = strong bias)
- `dot()` = cos of angle between wind and step direction

**Effect on distance map**:
- Downwind from source: dot ≈ +1 → cost × (1-α) → shorter distance → higher concentration
- Upwind from source: dot ≈ -1 → cost × (1+α) → longer distance → lower concentration
- Crosswind: dot ≈ 0 → cost unchanged

### What Changes

- `_dijkstra_numba_core()`: accept wind_u, wind_v arrays, compute biased edge costs
- `compute_distance_map_from_sensor()`: pass wind arrays to Dijkstra
- Cache key: include a wind field hash (or invalidate on wind update)
- Everything else (batch computation, particle filter, sensor model) stays identical

### Pros
- Minimal code change (~20 lines in Dijkstra)
- Same equation, same caching strategy, same particle filter
- Naturally obstacle-aware (Dijkstra handles walls)
- Uses spatially varying GMRF wind at every cell
- Performance: ~same as current Dijkstra

### Cons
- Heuristic — not derived from physics
- Single parameter α needs tuning
- Doesn't capture plume shape (elongation, meandering)
- Distance is symmetric in the graph but wind makes it asymmetric — need to think about direction (source→sensor vs sensor→source)

### Important Note on Direction

The current Dijkstra runs from **sensor to all cells** (one run per measurement). Gas travels from **source to sensor**. So the wind bias should make it cheap to travel from source→sensor along the wind. Since Dijkstra runs sensor→source, we need to flip the wind direction:

```
# In Dijkstra (running from sensor toward sources):
# A step from current cell toward a neighbor is "toward the source"
# Gas travels the opposite direction (source → sensor)
# So downwind for gas = upwind in Dijkstra traversal
# → Make upwind steps (into the wind) CHEAPER in Dijkstra
cost(i → j) = base_cost × (1 + α · dot(wind_dir_i, step_dir_ij))
```

This way, Dijkstra assigns shorter distances to cells that are upwind of the sensor — exactly where gas would come from.

---

## Option 2: Advection-Diffusion Green's Function (Vergassola)

**Idea**: Replace the isotropic Gaussian with the 2D steady-state solution of the advection-diffusion equation.

### Equation

```
C(r|rs) = Q/(4πD) · exp(V·(x-xs)·cos(θw) + V·(y-ys)·sin(θw)) / (2D)) · K₀(|r-rs|/λ)
```

Where:
- `V` = local wind speed from GMRF
- `θw` = local wind direction from GMRF
- `D` = effective diffusivity (tunable, replaces σm)
- `K₀` = modified Bessel function of second kind, order 0
- `λ = √(D·τ / (1 + V²τ/(4D)))` = correlation length
- `τ` = effective gas lifetime

### Physical Meaning

- `K₀(r/λ)` gives radial decay (similar to Gaussian but with heavier tails)
- `exp(V·Δx/(2D))` creates asymmetry: concentration is higher downwind
- When V→0, reduces to isotropic: `C ∝ K₀(r/λ)` (similar to current model)

### What Changes

- Replace `Qm · exp(-c_obs²/(2σm²))` with the equation above
- Use Euclidean distance |r-rs| (not Dijkstra) for K₀ argument
- Use GMRF wind (V, θw) at the midpoint or source location
- Obstacle handling: multiply by a "visibility" factor from Dijkstra (1 if connected, 0 if blocked)
- New parameters: D (diffusivity), τ (lifetime)

### Hybrid: Vergassola + Dijkstra Obstacle Check

```
C(r|rs) = Q/(4πD) · exp(V·d_proj/(2D)) · K₀(c_obs/λ) · visibility(r, rs)
```

Where:
- `d_proj` = projection of (r-rs) onto wind direction (signed, positive = downwind)
- `c_obs` = Dijkstra distance (for K₀ radial decay, obstacle-aware)
- `visibility` = 1 if Dijkstra distance is finite, 0 otherwise

### Pros
- Physics-based (derived from advection-diffusion PDE)
- Captures asymmetric plume shape
- `K₀` has the right behavior (log singularity near source, exponential decay far away)
- Fast: closed-form, vectorizable with scipy.special.k0

### Cons
- Assumes locally uniform wind (uses wind at one point, not the full field)
- K₀ is singular at r=0 (needs clamping)
- Two new parameters (D, τ) to tune
- Mixing Dijkstra distance with Euclidean projection is an approximation

---

## Option 3: Wind-Stretched Elliptical Gaussian (Kernel DM+V/W)

**Idea**: Replace the isotropic Gaussian with an elliptical Gaussian aligned with the local wind direction. Gas disperses farther along the wind than across it.

### Equation

```
R(rk|θ) = Qm · exp(-d_along²/(2σ_along²) - d_cross²/(2σ_cross²))
```

Where:
- `d_along` = distance component along wind direction
- `d_cross` = distance component perpendicular to wind
- `σ_along = σm · (1 + β·V)` = along-wind dispersion (stretched by wind speed)
- `σ_cross = σm` = cross-wind dispersion (unchanged)
- `β` = wind stretching factor (tunable)
- `V` = local wind speed

### Distance Decomposition

Given source at rs, sensor at rk, wind direction θw at rs:
```
Δ = rk - rs
d_along = Δx·cos(θw) + Δy·sin(θw)     # projection onto wind
d_cross = -Δx·sin(θw) + Δy·cos(θw)     # perpendicular component
```

### Obstacle-Aware Variant

Use Dijkstra distance for the total radial distance, then decompose:
```
ratio = c_obs / |rk - rs|              # how much longer the obstacle path is
d_along_obs = d_along × ratio          # scale both components
d_cross_obs = d_cross × ratio
```

### Pros
- Simple to implement
- Intuitive: gas cloud is an ellipse stretched along wind
- Uses local wind vector (direction + speed)
- Can be combined with Dijkstra for obstacle awareness

### Cons
- Heuristic (not physics-based)
- Only captures elongation, not the asymmetry (concentration should be higher downwind than upwind)
- The obstacle-aware variant is an approximation

### Asymmetric Variant

Add a shift to account for downwind bias:
```
d_along_shifted = d_along - V·τ_eff    # shift the peak downwind
```
This moves the concentration maximum downwind of the source.

---

## Option 4: Filament Simulation (PMFS-style)

**Idea**: For each candidate source location, simulate gas filaments advected by the GMRF wind field. Compare simulated gas-hit map with observed detections.

### Algorithm

```
For each source hypothesis (x0, y0):
    1. Release N_fil filaments at (x0, y0)
    2. For each filament, each timestep:
       a. Advect: pos += wind(pos) · dt + noise
       b. Grow:   σ² += γ · dt
       c. Check wall collision
    3. Build predicted hit map: which cells have filaments passing through?
    4. Compare with observed gas detection map
```

### Comparison Metric (from PMFS)

```
Δ_ik = |f_measured_i - f_simulated_i|     # per-cell hit difference
P(source_k | data) ∝ Π_i (α_i·(1-Δ_ik) + (1-α_i))
```

### Pros
- Most physically accurate for indoor environments
- Naturally uses the full spatially-varying wind field
- Handles obstacles (filaments blocked by walls)
- Captures turbulent intermittency
- State-of-the-art approach (PMFS, IEEE T-RO 2024)

### Cons
- Computationally expensive (many filaments × many source candidates)
- Significant implementation effort
- Requires parameter tuning (filament release rate, growth rate γ, noise)
- Changes the likelihood model (binary hit-map comparison vs continuous concentration)

---

## Comparison

| Aspect | Option 1: Biased Dijkstra | Option 2: Vergassola | Option 3: Elliptical | Option 4: Filament |
|--------|--------------------------|---------------------|---------------------|-------------------|
| Code change | ~20 lines | ~50 lines | ~40 lines | ~300+ lines |
| Physics basis | Heuristic | Analytical PDE soln | Heuristic | Simulation |
| Wind field usage | Full (per-cell) | Local (at source) | Local (at source) | Full (per-cell) |
| Obstacle handling | Built-in (Dijkstra) | Needs hybrid | Approximation | Built-in |
| Asymmetry | Yes (via cost bias) | Yes (exponential) | Partial (with shift) | Yes (natural) |
| New parameters | α | D, τ | β, (τ_eff) | γ, σ₀, N_fil |
| Performance | Same as current | Fast (closed-form) | Fast (closed-form) | Slow (simulation) |
| PF integration | Drop-in | Drop-in | Drop-in | Needs new likelihood |
| Accuracy | Low-Medium | Medium | Medium | High |

---

## Recommended Path

### Phase 1: Wind-Biased Dijkstra (Option 1)
- Fastest to implement, minimal risk
- Validates that wind information improves localization
- If it helps → proceed to Phase 2

### Phase 2: Vergassola Hybrid (Option 2)
- Physics-based upgrade
- Keep Dijkstra for obstacle distance, add exponential wind bias
- Better plume shape modeling

### Phase 3 (optional): Filament Simulation (Option 4)
- Only if Phase 1-2 results are insufficient
- Most accurate but most complex
- Consider PMFS codebase as reference implementation

---

## References

- Vergassola et al., "Infotaxis as a strategy for searching without gradients," Nature 2007
- Monroy et al., "Online Estimation of 2D Wind Maps for Olfactory Robots," IEEE 2016
- Monroy et al., "Robotic Gas Source Localization with Probabilistic Mapping and Online Dispersion Simulation," IEEE T-RO 2024
- Lilienthal et al., "The 3D-Kernel DM+V/W Algorithm," ResearchGate 2010
- Sanchez-Garrido et al., "Application of the Gaussian Plume Model to Localization of an Indoor Gas Source," Sensors 2018
- Francis et al., "Gas source localization and mapping with mobile robots: A review," J. Field Robotics 2022
- Kim et al., "Gas Source Localization in Unknown Indoor Environments Using Dual-Mode Information-Theoretic Search," IEEE RA-L 2025
