# EFE IGDM - Implementation Documentation

ROS2 package for autonomous gas source localization using dual-mode information-theoretic search in unknown indoor environments.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [Main Node: igdm.py](#main-node-igdmpy)
4. [Estimation Module](#estimation-module)
5. [Planning Module](#planning-module)
6. [Mapping Module](#mapping-module)
7. [Visualization Module](#visualization-module)
8. [Usage Examples](#usage-examples)
9. [Performance Optimization](#performance-optimization)
10. [API Reference](#api-reference)

---

## Architecture Overview

### Package Structure

```
efe_igdm/
├── igdm.py                          # Main ROS2 node (coordinator)
├── estimation/
│   ├── igdm_gas_model.py           # Indoor Gaussian Dispersion Model
│   ├── particle_filter.py          # Bayesian inference
│   ├── sensor_model.py             # Gaussian sensor model
│   └── sensor_interface.py         # Abstract sensor interface
├── planning/
│   ├── rrt.py                      # Local planner (RRT-Infotaxis)
│   ├── global_planner.py           # Global planner (PRM + frontiers)
│   ├── dead_end_detector.py        # Mode switching logic
│   └── navigator.py                # Low-level motion control
├── mapping/
│   ├── occupancy_grid.py           # Occupancy grid data structure
│   └── lidar_mapper.py             # LiDAR-based mapping
├── visualization/
│   ├── marker_visualizer.py        # RViz markers
│   ├── text_visualizer.py          # Text information display
│   └── clear_visualization.py      # Cleanup utilities
└── utils/
    └── experiment_logger.py        # CSV data logging
```

### Design Pattern

The implementation uses a **modular coordinator pattern**:
- **Main Node** (`igdm.py`): Orchestrates the GSL loop
- **Helper Modules**: Specialized classes for estimation, planning, mapping, visualization
- **Data Flow**: Sensor → Estimation → Planning → Navigation → Repeat

---

## Core Modules

### 1. Main Node: igdm.py

**Class**: `RRTInfotaxisNode`

The central ROS2 node that coordinates all components.

#### Key Methods

```python
class RRTInfotaxisNode(Node):
    def __init__(self):
        """Initialize node with all subsystems."""

    def take_step(self):
        """Main control loop iteration. Called on every planning cycle."""
        # 1. Check if moving
        # 2. Update particle filter with sensor measurement
        # 3. Plan next move (LOCAL or GLOBAL mode)
        # 4. Visualize
        # 5. Check convergence
        # 6. Execute move
```

#### Main Loop Flow

```python
# Pseudo-code of take_step()
if sensor_not_initialized or position_none:
    return

if not initial_spin_done:
    perform_initial_spin()  # 360° calibration
    return

if is_moving:
    return  # Wait for navigation

if in_settling_mode:
    handle_settling_complete()  # Switch from GLOBAL to LOCAL
    return

# Core GSL Logic
particle_filter.update(sensor_value, position)
current_means, current_stds = particle_filter.get_estimate()

if mode == 'LOCAL':
    next_pos, debug_info, dead_end = run_local_planning()
    if dead_end and enable_global_planner:
        handle_dead_end_transition()
        mode = 'GLOBAL'

elif mode == 'GLOBAL':
    next_pos = follow_global_path()
    if high_mutual_information_at_waypoint():
        mode = 'LOCAL'

visualize(...)
log_data(...)

if converged(current_stds):
    save_summary_and_finish()
else:
    navigate_to(next_pos)
```

#### ROS2 Interfaces

**Subscribed Topics**:
- `/PioneerP3DX/ground_truth` (PoseWithCovarianceStamped): Robot pose
- `/fake_pid/Sensor_reading` (GasSensor): Gas concentration
- `/PioneerP3DX/laser_scanner` (LaserScan): LiDAR scans

**Published Topics**:
- `/PioneerP3DX/cmd_vel` (Twist): Velocity commands
- `/rrt_infotaxis/source_info_text` (MarkerArray): Status text
- `/rrt_infotaxis/slam_map` (OccupancyGrid): Built map
- `/rrt_infotaxis/particles` (MarkerArray): Particle visualization
- `/rrt_infotaxis/rrt_tree` (MarkerArray): RRT tree
- `/rrt_infotaxis/global_path` (MarkerArray): Global path

#### Parameters

```yaml
# Core parameters accessed as self.params dict
sigma_m: 1.5                      # IGDM dispersion parameter
number_of_particles: 1000         # Particle filter size
n_tn: 50                          # RRT tree vertices
delta: 0.7                        # RRT step size (m)
max_depth: 4                      # RRT branch depth
sigma_threshold: 0.5              # Convergence threshold
dead_end_epsilon: 0.6             # Dead end detector ε
enable_global_planner: true       # Dual-mode enabled
```

---

## Estimation Module

### 1. IGDM: igdm_gas_model.py

**Class**: `IndoorGaussianDispersionModel`

Computes gas concentration using obstacle-aware distances.

#### Equation

```
R(rk|θ) = Qm · exp(-c_obs(rk, r0)² / (2·σm²))
```

#### Key Methods

```python
class IndoorGaussianDispersionModel:
    def __init__(self, sigma_m: float, occupancy_grid: OccupancyGridMap):
        """
        Args:
            sigma_m: Dispersion standard deviation (default: 1.5 m)
            occupancy_grid: Map for obstacle-aware distance calculation
        """

    def compute_concentration(self,
                             position: Tuple[float, float],
                             source_location: Tuple[float, float],
                             release_rate: float) -> float:
        """Compute concentration at a single position."""

    def compute_concentrations_batch(self,
                                    sensor_position: Tuple[float, float],
                                    particle_locations: np.ndarray,  # (N, 2)
                                    release_rates: np.ndarray) -> np.ndarray:  # (N,)
        """
        VECTORIZED: Compute concentrations for ALL particles at once.

        Performance: ~2-5ms for 1000 particles

        Returns:
            concentrations: Array of shape (N,)
        """

    def compute_distance_map_from_sensor(self,
                                        sensor_position: Tuple[float, float]) -> np.ndarray:
        """
        Compute distance map using Dijkstra's algorithm (CACHED).

        Returns:
            distance_map: 2D array of shape (height, width)
        """
```

#### Numba-Accelerated Dijkstra

```python
@jit(nopython=True, cache=True)
def _dijkstra_numba_core(grid_data, start_gx, start_gy, width, height, resolution):
    """
    JIT-compiled Dijkstra for ~10-20ms performance on 100×100 grids.

    Uses 8-connectivity with diagonal cost = √2 × resolution.
    """
```

#### Performance Features

- **Caching**: Distance maps cached by sensor position (LRU, size=20)
- **Vectorization**: Batch lookup using NumPy advanced indexing
- **JIT Compilation**: Numba accelerates Dijkstra from ~500ms to ~15ms

#### Usage Example

```python
igdm = IndoorGaussianDispersionModel(sigma_m=1.5, occupancy_grid=slam_map)

# Single concentration
conc = igdm.compute_concentration(
    position=(5.0, 3.0),
    source_location=(6.0, 4.0),
    release_rate=50.0
)

# Batch (vectorized)
concentrations = igdm.compute_concentrations_batch(
    sensor_position=(5.0, 3.0),
    particle_locations=np.array([[6.0, 4.0], [5.5, 3.5], ...]),  # (N, 2)
    release_rates=np.array([50.0, 45.0, ...])  # (N,)
)

# Cache statistics
stats = igdm.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

---

### 2. Particle Filter: particle_filter.py

**Class**: `ParticleFilter`

Bayesian inference for source parameters: θ = [x₀, y₀, Q₀]

#### Initialization

```python
pf = ParticleFilter(
    num_particles=1000,
    search_bounds={
        'x': (0, 20),    # Search space x bounds
        'y': (0, 12),    # Search space y bounds
        'Q': (0, 120)    # Release rate bounds
    },
    sensor_model=ContinuousGaussianSensorModel(...),
    dispersion_model=IndoorGaussianDispersionModel(...),
    resample_threshold=0.5  # Resample when Neff < 0.5*N
)
```

#### Core Methods

```python
def update(self, measurement: float, sensor_position: Tuple[float, float],
           skip_resample: bool = False):
    """
    Update particle weights with new sensor measurement.

    Steps:
    1. Compute likelihoods: p(z|θ^i) for all particles (VECTORIZED)
    2. Update weights: w^i ← w^i · p(z|θ^i)
    3. Normalize: w^i ← w^i / Σw
    4. Resample if Neff < threshold (systematic resampling)
    5. MCMC move (diversification)

    Args:
        measurement: Continuous sensor reading (ppm)
        sensor_position: (x, y) where measurement was taken
        skip_resample: Set True for hypothetical updates (RRT)
    """

def get_estimate(self) -> Tuple[Dict, Dict]:
    """
    Get weighted mean and standard deviation.

    Returns:
        means: {'x': est_x, 'y': est_y, 'Q': est_Q}
        stds: {'x': std_x, 'y': std_y, 'Q': std_Q}
    """

def get_entropy(self) -> float:
    """
    Compute Shannon entropy: H = -Σ w^i log(w^i)

    Returns:
        entropy: in nats
    """

def compute_expected_entropy(self, sensor_position: Tuple[float, float]) -> float:
    """
    OPTIMIZED: Compute E[H_{k+1}] in one vectorized pass.

    Replaces slow loop over measurement bins.

    Performance: ~5-10ms for 1000 particles × 10 bins

    Returns:
        expected_entropy: in nats
    """
```

#### Mutual Information (for RRT)

Mutual information I = H_k - E[H_{k+1}] is computed in RRT:

```python
# In RRT.calculate_branch_information()
current_entropy = pf.get_entropy()

expected_entropy = 0.0
for bin_idx in range(num_bins):
    prob = pf.predict_measurement_probability(position, bin_idx)
    if prob > 1e-6:
        hyp_entropy = pf.compute_hypothetical_entropy(bin_idx, position)
        expected_entropy += prob * hyp_entropy

mutual_info = current_entropy - expected_entropy
```

#### Resampling Strategy

```python
def _systematic_resample(self):
    """
    Systematic resampling (O(N) complexity).

    Equivalent to placing N evenly-spaced marks on cumulative weight "ruler".
    High-weight particles get multiple copies, low-weight get none.
    """
    cumulative_sum = np.cumsum(self.weights)
    positions = (np.arange(N) + np.random.random()) / N
    indices = np.searchsorted(cumulative_sum, positions)
    return indices
```

#### MCMC Diversification

```python
def _mcmc_move(self):
    """
    Metropolis-Hastings MCMC to add diversity after resampling.

    FULLY VECTORIZED: All N particles moved in parallel.

    Steps:
    1. Generate proposals: θ' ~ N(θ, Σ_MCMC)
    2. Compute acceptance ratio: α = p(z|θ') / p(z|θ)
    3. Accept with probability min(1, α)
    """
```

---

### 3. Sensor Model: sensor_model.py

**Class**: `ContinuousGaussianSensorModel`

#### Gaussian Sensor Model

```
p(z_k | θ) = N(z_k; R(r_k|θ), σ_g²)

where σ_g = α · R(r_k|θ) + σ_env
```

#### Initialization

```python
sensor_model = ContinuousGaussianSensorModel(
    alpha=0.1,                # Proportional noise coefficient
    sigma_env=1.5,            # Environmental noise (ppm)
    num_levels=10,            # Discretization bins for RRT entropy
    max_concentration=120.0   # Upper bound for discretization
)
```

#### Core Methods

```python
def probability_continuous_vec(self,
                              measurement: float,
                              predicted_concentrations: np.ndarray) -> np.ndarray:
    """
    Vectorized Gaussian likelihood.

    Args:
        measurement: Actual sensor reading (e.g., 10.94 ppm)
        predicted_concentrations: Array of shape (N,) from dispersion model

    Returns:
        likelihoods: Array of shape (N,)
    """
    sigma_g = self.alpha * predicted_concentrations + self.sigma_env
    coefficient = 1.0 / (sigma_g * np.sqrt(2 * np.pi))
    exponent = -((measurement - predicted_concentrations) ** 2) / (2 * sigma_g ** 2)
    return coefficient * np.exp(exponent)
```

#### Discretization for RRT

For RRT entropy calculation, measurements are discretized into bins:

```python
def compute_discretized_distribution(self,
                                    predicted_concentrations: np.ndarray) -> np.ndarray:
    """
    Compute probability mass for ALL bins and ALL particles.

    Returns:
        bin_probs: Shape (num_levels, N)
                  bin_probs[i, j] = P(z ∈ bin_i | particle_j)
    """
```

**Discretization Strategy**:
- Dynamic bin edges based on particle predictions
- Bin 0: (-∞, threshold[0]) captures low concentrations
- Bin N-1: (threshold[N-2], ∞) captures high concentrations
- Uses CDF: P(a < z < b) = Φ((b-μ)/σ) - Φ((a-μ)/σ)

---

## Planning Module

### 1. RRT: rrt.py

**Class**: `RRT` - Receding-Horizon RRT-Infotaxis (Local Planner)

#### Initialization

```python
rrt = RRT(
    occupancy_grid=slam_map,
    N_tn=50,                  # Number of tree vertices
    R_range=35.0,             # Sampling radius (N_tn × delta)
    delta=0.7,                # Step size between vertices
    max_depth=4,              # Maximum branch depth
    discount_factor=0.8,      # γ for branch information
    positive_weight=0.5,      # Alias for discount_factor
    robot_radius=0.35         # Collision radius
)
```

#### Core Algorithm

```python
def get_next_move_debug(self, start_pos, particle_filter) -> dict:
    """
    Main RRT-Infotaxis algorithm with detailed debug info.

    Steps:
    1. Build RRT tree: sprawl(start_pos)
    2. Prune to get branches: paths = prune()
    3. Evaluate branches: BI = calculate_branch_information(path, pf)
    4. Select best: BI* = max(BI)
    5. Return first vertex of best branch

    Returns:
        debug_info: {
            'next_position': (x, y),
            'best_path': List[Tuple[float, float]],
            'best_BI': float,  # Branch Information
            'all_paths': List of all pruned branches,
            'all_branch_information': List[float],
            'tree_nodes': List[Node],
            'num_branches': int,
            'num_tree_nodes': int
        }
    """
```

#### Tree Building

```python
def sprawl(self, start_pos: Tuple[float, float]):
    """
    Build RRT tree with N_tn vertices.

    Algorithm:
    1. Initialize tree with root node at start_pos
    2. While len(nodes) < N_tn:
        a. Sample random point within R_range (polar coordinates)
        b. Find closest existing node
        c. Steer toward sample by delta distance
        d. Check collision
        e. Add new node if valid
    """
```

#### Branch Information Calculation

```python
def calculate_branch_information(self, path: List[Node],
                                particle_filter: ParticleFilter) -> float:
    """
    Compute Branch Information (Equation 19):

    BI(V_b) = Σ_{i=1}^m γ^{i-1} · I(v_{b,i})

    Args:
        path: List of nodes from root to leaf
        particle_filter: Current PF state

    Returns:
        BI: Branch information value
    """
    path = path[1:]  # Exclude root
    BI = 0.0

    for i, node in enumerate(path):
        if node.entropy_gain != -np.inf:
            BI += (self.discount_factor ** i) * node.entropy_gain
            continue

        # Compute mutual information at this vertex
        current_entropy = particle_filter.get_entropy()
        expected_entropy = 0.0

        for measurement_bin in range(num_levels):
            prob = particle_filter.predict_measurement_probability(
                node.position, measurement_bin
            )
            if prob > 1e-6:
                hyp_entropy = particle_filter.compute_hypothetical_entropy(
                    measurement_bin, node.position
                )
                expected_entropy += prob * hyp_entropy

        mutual_info = current_entropy - expected_entropy
        node.entropy_gain = mutual_info  # Cache
        BI += (self.discount_factor ** i) * mutual_info

    return BI
```

#### Vectorized Collision Checking

```python
def is_collision_free_vectorized(self, pos1: np.ndarray,
                                 pos2: np.ndarray) -> bool:
    """
    Check collision along line segment using vectorized sampling.

    Performance: ~0.1-0.5ms per check

    Steps:
    1. Sample points along line (spacing = resolution/2)
    2. Convert all to grid coordinates (vectorized)
    3. Batch boundary check (vectorized)
    4. Batch occupancy check (vectorized)
    """
```

---

### 2. Global Planner: global_planner.py

**Class**: `GlobalPlanner` - PRM-based Frontier Exploration

#### Initialization

```python
global_planner = GlobalPlanner(
    occupancy_grid=slam_map,
    robot_radius=0.35,
    prm_samples=300,               # Number of PRM vertices
    prm_connection_radius=5.0,     # Edge connection radius (ROS param default)
    frontier_min_size=3,           # Min cells per frontier cluster
    lambda_p=0.1,                  # Path cost weight
    lambda_s=0.05                  # Source distance weight
)
```

#### Main Planning Method

```python
def plan(self, current_position, particle_filter) -> Dict:
    """
    Execute global planning pipeline.

    Steps:
    1. detect_frontiers() - Find boundary cells
    2. cluster_frontiers() - Group into clusters
    3. build_prm_graph() - Construct roadmap
    4. evaluate_frontier_vertices() - Compute utilities
    5. Return best path

    Returns:
        result: {
            'success': bool,
            'best_frontier_vertex': PRMVertex,
            'best_global_path': List[Tuple[float, float]],
            'best_utility': float,
            'frontier_cells': List[Tuple[int, int]],
            'frontier_clusters': List[FrontierCluster],
            'prm_vertices': List[PRMVertex]
        }
    """
```

#### Frontier Detection (Vectorized)

```python
def detect_frontiers(self) -> List[Tuple[int, int]]:
    """
    Vectorized frontier detection using scipy.ndimage.

    Definition: Frontier = FREE cell (0) adjacent to UNKNOWN cell (-1)

    Algorithm:
    1. Create boolean masks: is_free, is_unknown
    2. Dilate unknown mask (8-connectivity)
    3. Intersection: frontier_mask = is_free & unknown_dilated
    4. Extract coordinates

    Performance: ~1-2ms for 100×100 grid
    """
```

#### Frontier Clustering

```python
def cluster_frontiers(self) -> List[FrontierCluster]:
    """
    Cluster frontiers using Connected Components Labeling.

    Uses scipy.ndimage.label() for efficient clustering.

    Returns:
        clusters: List of FrontierCluster objects
                 (filtered by frontier_min_size)
    """
```

#### PRM Graph Construction (KD-Tree Optimized)

```python
def build_prm_graph(self, current_position):
    """
    Build Probabilistic Roadmap using KD-Tree for O(N log N) connectivity.

    Vertices:
    - Start position (ID 0)
    - Frontier centroids
    - Random collision-free samples

    Edges:
    - KDTree.query_pairs() finds all pairs within connection_radius
    - Check collision-free for each pair

    Performance: ~50-100ms for 300 vertices

    Optimization: Replaces O(N²) brute-force with O(N log N) KD-Tree
    """
```

#### Utility Evaluation (Single-Source Dijkstra)

```python
def evaluate_frontier_vertices(self, current_pos, particle_filter) -> Dict:
    """
    Evaluate all frontiers using utility function (Equation 22).

    Utility = I(v_f) · exp(-λ_p·cost(r, v_f)) · exp(-λ_s·D(v_f, r̂_0))

    Optimization: Run Dijkstra ONCE from start to get costs to ALL frontiers.

    Steps:
    1. compute_all_paths_from_start(0) → distances, predecessors
    2. For each frontier vertex:
        a. Check reachability (dist < ∞)
        b. Compute mutual information
        c. Compute utility
    3. Sort by utility
    4. Reconstruct path for best frontier

    Performance: ~20-50ms for 10-20 frontiers
    """
```

---

### 3. Dead End Detector: dead_end_detector.py

**Class**: `DeadEndDetector`

Detects when local planner is stuck using adaptive threshold.

#### Algorithm (Equations 20-21)

```
BI* = max(BI(V_b))  for all branches                [Eq. 20]
BI_thresh_{k+1} = ε · BI_thresh_k + (1-ε) · BI*      [Eq. 21]
Dead End: BI* < BI_thresh_k
```

#### Initialization

```python
detector = DeadEndDetector(
    epsilon=0.6,                # Threshold update weight (0 < ε < 1)
    initial_threshold=0.1       # Starting threshold value
)
```

#### Core Methods

```python
def is_dead_end(self, bi_optimal: float) -> bool:
    """
    Check if current BI indicates dead end.

    Args:
        bi_optimal: BI* (best branch information from RRT)

    Returns:
        is_dead_end: True if BI* < BI_thresh

    Side Effect: Updates threshold for next iteration
    """

def update_threshold(self, bi_optimal: float):
    """
    Update adaptive threshold using exponential moving average.

    This is a low-pass filter: threshold trails the optimal BI with delay.
    When BI drops sharply, it falls below the delayed threshold.
    """
    self.bi_threshold = (
        self.epsilon * self.bi_threshold +
        (1.0 - self.epsilon) * bi_optimal
    )

def reset(self, initial_threshold: float = None):
    """Reset detector to initial state (used when switching back to LOCAL)."""
```

#### Usage in Main Node

```python
# In igdm.py
bi_optimal = debug_info.get("best_utility", 0.0)

if self.params['enable_global_planner']:
    dead_end_detected = self.dead_end_detector.is_dead_end(bi_optimal)

    if dead_end_detected:
        self._handle_dead_end_transition()
        # Switch to GLOBAL mode
```

---

### 4. Navigator: navigator.py

**Class**: `Navigator`

Handles low-level motion control and recovery behaviors.

#### Key Features

- **Initial Spin**: 360° rotation for sensor calibration
- **Goal-Based Navigation**: Move to waypoint with tolerance
- **Stuck Detection**: Monitors consecutive navigation failures (max_failures_tolerance=3)
- **Teleport Recovery**: Attempts to escape local minima

#### Methods

```python
class Navigator:
    def __init__(self, node, on_complete_callback):
        """
        Args:
            node: Parent ROS2 node
            on_complete_callback: Called when move completes
        """

    def send_goal(self, x: float, y: float, tolerance: float = 0.3):
        """Send navigation goal."""

    def perform_initial_spin(self, current_position, current_theta):
        """Execute 360° spin for sensor calibration."""

    def attempt_teleport_recovery(self, current_pos, slam_map, dead_end_detector):
        """
        Try to escape dead end by teleporting to random free position.

        Returns:
            success: True if teleport location found
        """
```

---

## Mapping Module

### 1. Occupancy Grid: occupancy_grid.py

**Class**: `OccupancyGridMap`

2D occupancy grid with world ↔ grid coordinate conversion.

#### Convention

- `0`: Free space
- `>0` (typically 1 or 100): Occupied
- `-1`: Unknown

#### Initialization

```python
# From ROS2 service (GADEN)
occupancy_map = create_occupancy_map_from_service(
    node,
    z_level=5,
    service_name='/gaden_environment/occupancyMap3D',
    timeout_sec=10.0
)

# Empty map for SLAM
slam_map = create_empty_occupancy_map(reference_map)
# All cells initialized to -1 (unknown)
```

#### Core Methods

```python
class OccupancyGridMap:
    def __init__(self, grid: np.ndarray, params: dict):
        """
        Args:
            grid: 2D array (height, width) with occupancy values
            params: dict with env_min, env_max, cell_size
        """

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    def is_valid(self, position: Tuple[float, float], radius: float = 0.2) -> bool:
        """
        Check if position is collision-free within safety radius.

        Uses squared distance comparison (no sqrt) for performance.
        """

    def is_cell_free(self, gx: int, gy: int) -> bool:
        """Fast cell-level check (for Dijkstra, A*, etc.)."""
        return 0 <= gx < self.width and 0 <= gy < self.height and self.grid[gy, gx] == 0
```

#### Properties

```python
occupancy_map.width          # Grid width (cells)
occupancy_map.height         # Grid height (cells)
occupancy_map.resolution     # Cell size (meters)
occupancy_map.origin_x       # World origin X
occupancy_map.origin_y       # World origin Y
occupancy_map.grid           # 2D numpy array (height, width)
occupancy_map.real_world_width  # Width in meters
occupancy_map.real_world_height # Height in meters
```

---

### 2. LiDAR Mapper: lidar_mapper.py

**Class**: `LidarMapper`

Updates occupancy grid from LiDAR scans using ray tracing.

#### Usage

```python
# In igdm.py laser_callback
lidar_mapper = LidarMapper(slam_map)

def laser_callback(self, msg: LaserScan):
    obstacles_found = lidar_mapper.update_from_scan(
        msg,
        robot_x,
        robot_y,
        robot_theta
    )
```

#### Algorithm

```python
def update_from_scan(self, scan: LaserScan, robot_x, robot_y, robot_theta) -> int:
    """
    Ray tracing to update occupancy grid.

    For each laser beam:
    1. Calculate endpoint in world coordinates
    2. Mark free cells along ray (0)
    3. Mark endpoint as occupied (1) if valid range

    Returns:
        num_obstacles: Number of occupied cells marked
    """
```

---

## Visualization Module

### 1. Marker Visualizer: marker_visualizer.py

**Class**: `MarkerVisualizer`

Publishes RViz markers for all visualization elements.

#### Usage

```python
marker_viz = MarkerVisualizer(node, slam_map)

# Visualize particles
marker_viz.visualize_particles(particles, weights)

# Visualize RRT tree
marker_viz.visualize_all_paths(all_paths, all_utilities)
marker_viz.visualize_best_path(best_path)

# Visualize global planner
marker_viz.visualize_frontier_cells(frontier_cells)
marker_viz.visualize_frontier_centroids(clusters)
marker_viz.visualize_prm_graph(vertices, vertex_dict)
marker_viz.visualize_global_path(path)

# Visualize estimates
marker_viz.visualize_estimated_source(est_x, est_y)
marker_viz.visualize_current_position(position)
marker_viz.visualize_planner_mode(mode)  # "LOCAL" or "GLOBAL"
```

#### Color Scheme

- **Green spheres**: Particles (sized by weight)
- **Red sphere**: Estimated source
- **Blue sphere**: Current position
- **Magenta lines**: RRT branches (thickness ∝ utility)
- **Cyan path**: Selected RRT path
- **Yellow lines**: Global path
- **Orange markers**: Frontier cells
- **Purple lines**: PRM graph edges

---

### 2. Text Visualizer: text_visualizer.py

**Class**: `TextVisualizer`

Displays numeric information as text markers in RViz.

```python
text_viz = TextVisualizer(publisher, frame_id="map")

text_viz.publish_source_info(
    timestamp=now,
    predicted_x=est_x,
    predicted_y=est_y,
    std_dev=sigma_p,
    sensor_value=sensor_reading,
    entropy=current_entropy,
    bi_optimal=bi_optimal,
    bi_threshold=bi_threshold,
    dead_end_detected=dead_end,
    ...
)
```

Displays:
- Estimated source position
- Standard deviation (convergence metric)
- Sensor reading
- Entropy
- Branch information (BI* vs threshold)
- Dead end status
- Planner mode

---

## Usage Examples

### Example 1: Running the Node

```bash
# Terminal 1: Launch simulator
ros2 launch gaden_player gaden_simulation.launch

# Terminal 2: Run GSL node
ros2 run efe_igdm rrt_infotaxis_node

# Terminal 3: Visualize in RViz
rviz2 -d ~/ros2_ws/src/base/efe_igdm/config/gsl_viz.rviz
```

### Example 2: Programmatic Usage

```python
import rclpy
from efe_igdm.igdm import RRTInfotaxisNode

rclpy.init()
node = RRTInfotaxisNode()

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
```

### Example 3: Using Components Standalone

#### Particle Filter

```python
from efe_igdm.estimation.particle_filter import ParticleFilter
from efe_igdm.estimation.igdm_gas_model import IndoorGaussianDispersionModel
from efe_igdm.estimation.sensor_model import ContinuousGaussianSensorModel

# Initialize models
igdm = IndoorGaussianDispersionModel(sigma_m=1.5, occupancy_grid=map)
sensor_model = ContinuousGaussianSensorModel(alpha=0.1, sigma_env=1.5)

# Create particle filter
pf = ParticleFilter(
    num_particles=1000,
    search_bounds={'x': (0, 20), 'y': (0, 12), 'Q': (0, 120)},
    sensor_model=sensor_model,
    dispersion_model=igdm
)

# Update loop
for measurement, position in zip(measurements, positions):
    pf.update(measurement, position)
    means, stds = pf.get_estimate()
    print(f"Estimate: ({means['x']:.2f}, {means['y']:.2f}), σ={stds['x']:.2f}")
```

#### RRT Planner

```python
from efe_igdm.planning.rrt import RRT

rrt = RRT(
    occupancy_grid=slam_map,
    N_tn=50,
    R_range=35.0,
    delta=0.7,
    max_depth=4
)

# Get next move
debug_info = rrt.get_next_move_debug(current_pos, particle_filter)
next_pos = debug_info['next_position']
best_bi = debug_info['best_BI']

print(f"Next position: {next_pos}, BI*={best_bi:.4f}")
```

#### Dead End Detector

```python
from efe_igdm.planning.dead_end_detector import DeadEndDetector

detector = DeadEndDetector(epsilon=0.6, initial_threshold=0.1)

# In planning loop
bi_optimal = rrt_result['best_BI']
is_dead_end = detector.is_dead_end(bi_optimal)

if is_dead_end:
    print("Dead end detected! Switching to global planner.")
    # Switch modes...
```

---

## Performance Optimization

### Bottlenecks and Solutions

| Component | Bottleneck | Solution | Speedup |
|-----------|-----------|----------|---------|
| IGDM Distance Calc | Dijkstra's algorithm | Numba JIT + Caching | 30×  |
| Particle Filter | Likelihood computation | Vectorized batch computation | 100× |
| RRT Collision Check | Loop over line samples | Vectorized grid sampling | 10× |
| Mutual Information | Loop over bins × particles | Single-pass vectorized | 20× |
| Global Planner | PRM connectivity | KD-Tree query_pairs | 10× |
| Frontier Detection | Pixel-by-pixel search | scipy.ndimage dilation | 50× |

### Profiling Results

Typical step timing (1000 particles, 50 RRT vertices):

```
Total step time: ~750ms

Breakdown:
- RRT tree building:           100ms
- Collision checks:             50ms
- Entropy calculation:         400ms
  - IGDM distance map:          15ms (cached)
  - Concentration batch:         5ms
  - MI computation:            380ms (10 bins × 4 branch × 50 vertices)
- Particle filter update:       30ms
- Visualization:                20ms
- Logging:                       5ms
- Other:                       145ms
```

### Optimization Tips

1. **Reduce RRT Depth**: `max_depth=3` instead of 4 saves ~30% time
2. **Fewer Bins**: `num_levels=5` instead of 10 saves ~50% entropy calc time
3. **Smaller Tree**: `N_tn=30` instead of 50 (trades off exploration)
4. **Particle Count**: 500 particles still effective, saves ~40% time
5. **Cache Strategy**: Increase `_cache_max_size` if robot revisits areas

---

## API Reference

### RRTInfotaxisNode

Main coordinator node.

**Parameters**:
```python
sigma_m: float = 1.5
number_of_particles: int = 1000
n_tn: int = 50
delta: float = 0.7
max_depth: int = 4
robot_radius: float = 0.05
sigma_threshold: float = 0.5
success_distance: float = 0.5
positive_weight: float = 0.5
dead_end_epsilon: float = 0.6
dead_end_initial_threshold: float = 0.1
enable_global_planner: bool = True
prm_samples: int = 300
prm_connection_radius: float = 5.0  # Note: GlobalPlanner default is 2.5
frontier_min_size: int = 3
lambda_p: float = 0.1
lambda_s: float = 0.05
switch_back_threshold: float = 1.5
resample_threshold: float = 0.5
true_source_x: float = 2.0
true_source_y: float = 4.5
```

**Key Methods**:
- `take_step()`: Main control loop
- `pose_callback(msg)`: Process robot pose
- `sensor_callback(msg)`: Process gas measurement
- `laser_callback(msg)`: Process LiDAR scan

---

### IndoorGaussianDispersionModel

Gas dispersion model for indoor environments.

**Constructor**:
```python
__init__(sigma_m: float = 1.0, occupancy_grid: OccupancyGridMap = None)
```

**Methods**:
- `compute_concentration(position, source_location, release_rate) -> float`
- `compute_concentrations_batch(sensor_pos, particle_locs, rates) -> np.ndarray`
- `compute_distance_map_from_sensor(sensor_position) -> np.ndarray`
- `set_occupancy_grid(occupancy_grid)`
- `clear_cache()`
- `get_cache_stats() -> dict`

---

### ParticleFilter

Bayesian source parameter estimation.

**Constructor**:
```python
__init__(num_particles, search_bounds, sensor_model, dispersion_model,
         resample_threshold=0.5, mcmc_std=None)
```

**Methods**:
- `update(measurement, sensor_position, skip_resample=False)`
- `get_estimate() -> Tuple[Dict, Dict]`
- `get_entropy() -> float`
- `compute_expected_entropy(sensor_position) -> float`
- `compute_hypothetical_entropy(measurement, sensor_position) -> float`
- `predict_measurement_probability(sensor_position, binary_value=None) -> np.ndarray`
- `get_particles() -> Tuple[np.ndarray, np.ndarray]`
- `copy() -> ParticleFilter`

---

### RRT

Local planner using RRT-Infotaxis.

**Constructor**:
```python
__init__(occupancy_grid, N_tn, R_range, delta, max_depth=4,
         discount_factor=0.8, positive_weight=0.5, robot_radius=0.35)
```

**Methods**:
- `get_next_move(start_pos, particle_filter) -> Tuple[float, float]`
- `get_next_move_debug(start_pos, particle_filter) -> dict`
- `sprawl(start_pos)`
- `prune() -> List[List[Node]]`
- `calculate_branch_information(path, particle_filter) -> float`

---

### GlobalPlanner

Frontier-based exploration using PRM.

**Constructor**:
```python
__init__(occupancy_grid, robot_radius=0.35, prm_samples=300,
         prm_connection_radius=2.5, frontier_min_size=3,
         lambda_p=0.1, lambda_s=0.05)
```

**Methods**:
- `plan(current_position, particle_filter) -> dict`
- `detect_frontiers() -> List[Tuple[int, int]]`
- `cluster_frontiers() -> List[FrontierCluster]`
- `build_prm_graph(current_position)`
- `evaluate_frontier_vertices(current_pos, particle_filter) -> dict`
- `get_next_best_frontier() -> Optional[dict]`

---

### DeadEndDetector

Adaptive dead end detection for mode switching.

**Constructor**:
```python
__init__(epsilon: float = 0.9, initial_threshold: float = 0.1)
```

**Methods**:
- `is_dead_end(bi_optimal: float) -> bool`
- `update_threshold(bi_optimal: float)`
- `get_status() -> dict`
- `reset(initial_threshold: float = None)`
- `get_history() -> dict`

---

### OccupancyGridMap

2D occupancy grid for collision checking and path planning.

**Constructor**:
```python
__init__(grid: np.ndarray, params: dict)
```

**Methods**:
- `world_to_grid(x, y) -> Tuple[int, int]`
- `grid_to_world(gx, gy) -> Tuple[float, float]`
- `is_valid(position, radius=0.2) -> bool`
- `is_cell_free(gx, gy) -> bool`
- `add_rectangle_obstacle(x_min, y_min, x_max, y_max)`
- `visualize(ax=None, show_grid=False)`

**Factory Functions**:
- `create_occupancy_map_from_service(node, z_level, service_name, timeout) -> OccupancyGridMap`
- `create_empty_occupancy_map(reference_map) -> OccupancyGridMap`

---

## Data Logging

### ExperimentLogger

Logs all step data to CSV for post-analysis.

**Log File Location**: `log/efe_igdm_YYYYMMDD_HHMMSS.csv`

**Logged Columns**:
```
step, timestamp, mode, sensor_value, position_x, position_y,
est_x, est_y, est_Q, std_x, std_y, std_Q,
entropy, bi_optimal, bi_threshold, dead_end,
num_branches, best_utility, travel_cost, num_tree_nodes,
global_path_length, global_path_index
```

**Summary File**: `summary.txt`

```
Total Steps: 82
Total Distance: 57.4 m
Total Time: 270.5 s
Avg Computation: 0.74 s/step
Estimated Source: (2.13, 4.47)
True Source: (2.0, 4.5)
Estimation Error: 0.16 m
Success: True
```

---

## Configuration

### Launch File Example

```python
# launch/gsl_simulation.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='efe_igdm',
            executable='rrt_infotaxis_node',
            name='rrt_infotaxis_node',
            output='screen',
            parameters=[{
                'sigma_m': 1.5,
                'number_of_particles': 1000,
                'n_tn': 50,
                'enable_global_planner': True,
                # ... other parameters
            }]
        )
    ])
```

### Parameter File

```yaml
# config/params.yaml
rrt_infotaxis_node:
  ros__parameters:
    # Gas Model
    sigma_m: 1.5

    # Particle Filter
    number_of_particles: 1000
    sigma_threshold: 0.5
    resample_threshold: 0.5

    # Local Planner
    n_tn: 50
    delta: 0.7
    max_depth: 4
    positive_weight: 0.5

    # Dead End Detection
    enable_global_planner: true
    dead_end_epsilon: 0.6
    dead_end_initial_threshold: 0.1
    switch_back_threshold: 1.5

    # Global Planner
    prm_samples: 300
    prm_connection_radius: 5.0
    frontier_min_size: 3
    lambda_p: 0.1
    lambda_s: 0.05

    # Robot
    robot_radius: 0.35
    xy_goal_tolerance: 0.3
```

---

## Development Notes

### Code Style

- **Type Hints**: Used throughout for clarity
- **Docstrings**: Google style for all public methods
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Performance**: NumPy vectorization preferred over loops

### Testing Strategy

```bash
# Unit tests
python3 -m pytest efe_igdm/tests/

# Integration test
ros2 launch efe_igdm test_gsl.launch.py

# Performance profiling
python3 -m cProfile -o profile.stats scripts/profile_step.py
python3 -m pstats profile.stats
```

### Common Modifications

**Change dispersion model**:
```python
# In igdm.py _init_models_and_planners()
self.dispersion_model = MyCustomDispersionModel(...)
```

**Change sensor model**:
```python
# In igdm.py _init_models_and_planners()
self.sensor_model = MyCustomSensorModel(...)
```

**Add new planner mode**:
```python
# In igdm.py take_step()
elif self.planner_mode == 'MY_MODE':
    next_pos = self._run_my_custom_planner()
```

---

## References

**Paper**:
```
Kim et al., "Gas Source Localization in Unknown Indoor Environments Using
Dual-Mode Information-Theoretic Search", IEEE RA-L, Vol. 10, No. 1, 2025
```

**Key Equations**:
- IGDM: Equation 18 (page 4)
- Branch Information: Equation 19 (page 5)
- Dead End Detection: Equations 20-21 (page 5)
- Global Utility: Equation 22 (page 5)
- Sensor Model: Equation 3 (page 3)

**Dependencies**:
- ROS2 Humble
- Python 3.8+
- NumPy, SciPy
- Numba (JIT compilation)
- rclpy
- olfaction_msgs

---

*Implementation by: Efe*
*Based on: Kim et al. IEEE RA-L 2025*
*Last Updated: February 2026*
