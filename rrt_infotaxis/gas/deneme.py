from utils.sensor_model import BinarySensorModel
from utils.gaussian_plume import GaussianPlumeModel
from utils.particle_filter import ParticleFilter
from utils.occupancy_grid import load_3d_occupancy_grid, OccupancyGridMap
import numpy as np
from rrt_infotaxis.dijkstra import dijkstra_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils.rrt import RRT

# Load occupancy grid
grid, params = load_3d_occupancy_grid("OccupancyGrid3D.csv")
occupancy_map = OccupancyGridMap(grid, params)

# Initialize models
gas_model = GaussianPlumeModel(wind_direction=90)
sensor_model = BinarySensorModel()
pf = ParticleFilter(num_particles=1000,
                    search_bounds={"x": (0, 10), "y": (0, 10), "Q": (0, 1)},
                    binary_sensor_model=sensor_model,
                    dispersion_model=gas_model)
# True source location
source_pos = (8.0, 8.0)
source_Q = 0.56
# Starting position
current_pos = (2.0, 2.0)
# Initialize RRT
rrt = RRT(occupancy_map, N_tn=30, R_range=5.0, delta=0.5, max_depth=3)

initial_measurement = gas_model.compute_concentration(current_pos, source_pos, source_Q)
sensor_model.initialize_threshold(initial_measurement)
for step in range(50):
    # Simulate sensor measurement
    measurement = gas_model.compute_concentration(current_pos, source_pos, source_Q)
    binary_measurement = sensor_model.get_binary_measurement(measurement)
    print("Measurement:", measurement, "Binary:", binary_measurement)
    sensor_model.update_threshold(measurement)
    # Update particle filter
    pf.update(binary_measurement, current_pos)

    # Estimate source location
    current_means, current_stds = pf.get_estimate()
    est_x, est_y, est_Q = current_means["x"], current_means["y"], current_means["Q"]

    debug_info = rrt.get_next_move_debug(current_pos, pf)
    next_pos = debug_info["next_position"]
    best_path = debug_info["best_path"]
    all_paths = debug_info["all_paths"]
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Add obstacles from occupancy grid
    occupancy_map.visualize(ax=ax, show_grid=False)

    # Add gas concentration contours
    xs, ys = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    concentrations = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            concentrations[i,j] = gas_model.compute_concentration((xs[i,j], ys[i,j]), source_pos, source_Q)
    ax.contourf(xs, ys, concentrations, levels=50, cmap='Reds', alpha=0.6)
    print(f"Concentration range: min={concentrations.min():.2e}, max={concentrations.max():.2e}")
    print(f"Non-zero concentrations: {(concentrations > 0).sum()} out of {concentrations.size}")

    # Plot all RRT paths
    for path in all_paths:
        positions = np.array([node.position for node in path])
        ax.plot(positions[:, 0], positions[:, 1], 'gray', alpha=0.5)

    # Plot best path
    best_path = np.array([node.position for node in best_path])
    pf_weights = pf.weights

    # Plot particles with weights
    scatter = ax.scatter(
        pf.particles[:, 0],
        pf.particles[:, 1],
        c=pf_weights,
        cmap='viridis',
        s=5,
        label='Particles'
    )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Particle Weights')

    # Plot path and positions
    ax.plot(best_path[:, 0], best_path[:, 1], 'blue', linewidth=2, label='Best Path')
    ax.scatter(current_pos[0], current_pos[1], c='green', s=100, label='Current Position', edgecolors='black', zorder=5)
    ax.scatter(source_pos[0], source_pos[1], c='red', s=100, label='True Source', edgecolors='black', zorder=5)
    ax.scatter(est_x, est_y, c='orange', s=100, label='Estimated Source', edgecolors='black', zorder=5)

    ax.legend()
    plt.savefig(f"simulation/step_{step+1:02d}.png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    # Move to next position
    current_pos = next_pos

    print(f"Step {step+1}: Moved to {current_pos}, Estimated Source: ({est_x:.2f}, {est_y:.2f}, {est_Q:.2f})")