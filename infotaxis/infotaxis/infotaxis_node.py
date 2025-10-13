#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from gaden_msgs.srv import Occupancy
from olfaction_msgs.msg import GasSensor
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String, Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from scipy.special import k0
from scipy.stats import entropy as entropy_
from typing import Tuple

from .grid_manager import GridManager
from .navigation_controller import NavigationController
from .sensor_handler import SensorHandler

"""
Parameters
dt=.1: Time step duration in seconds (0.1 s = 100 ms between samples)
src_radius=.2: Detection radius in meters (agent must get within 20 cm of source to "find" it)
w=.5: Mean wind speed in m/s (0.5 m/s, wind direction is positive x)
d=.05: Turbulent diffusivity coefficient in m²/s (controls plume spread) (cannot be adjusted in GADEN)
r=10: Source emission rate in Hz (10 particles per second)
a=.003: Searcher/sensor size in meters (3 mm diameter)
tau=100: Particle/odor lifetime in seconds (100 s before decay)
"""
# Constants for now
dt = .1 # time step duration in seconds
src_radius = .2 # detection radius in meters
w = .1 # mean wind speed in m/s
d = .02 # turbulent diffusivity coefficient in m²/s
r = 10 # source emission rate in Hz
a = .003 # searcher/sensor size in meters
tau = 100 # particle/odor lifetime in seconds

def entropy(log_p_src):
    """
    Wrapper around scipy.stats entropy function that takes in a 2D
    log probability distribution.
    :param log_p_src: 2D array of log probabilities
    """
    # convert to non-log probability distribution
    p_src = np.exp(log_p_src)

    # make sure it normalizes to 1 (only over free cells with finite log probs)
    p_src /= p_src.sum()

    # calculate entropy
    return entropy_(p_src.flatten())

def log_k0(x):
    """Logarithm of modified bessel function of the second kind of order 0.
    Infinite values may still be returned if argument is too close to
    zero."""

    y = k0(x)

    # if array
    try:
        logy = np.zeros(x.shape, dtype=float)

        # attempt to calculate bessel function for all elements in x
        logy[y!=0] = np.log(y[y!=0])

        # for zero-valued elements use approximation
        logy[y==0] = -x[y==0] - np.log(x[y==0]) + np.log(np.sqrt(np.pi/2))

        return logy

    except:
        if y == 0:
            return -x - np.log(x) + np.log(np.sqrt(np.pi/2))
        else:
            return np.log(y)
        
def get_p_src_found(pos, xs, ys, log_p_src, radius):
    """
    Return the probability that a position is within "radius"
    of the source, given the source probability distribution.
    :param pos: position to calc prob that you are close to source
    :param xs: x-coords over which source prob distribution is defined
    :param ys: y-coords over which source prob distribution is defined
    :param log_p_src: log probability distribution over source position
    :param radius: distance you must be to source to detect it
    :return: probability
    """
    # get mask containing only points within radius of pos
    xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')
    dxs = pos[0] - xs_
    dys = pos[1] - ys_

    mask = (dxs**2 + dys**2 < radius**2)

    # sum probabilities contained in mask
    p_src = np.exp(log_p_src)
    p_src /= p_src.sum()
    p_src_found = p_src[mask].sum()

    return p_src_found
        
def get_length_constant(w, d, tau):
    """
    Return the turbulence length constant: sqrt( (d*tau) / (1 + (tau * w**2)/(4d) ) )
    :param d: diffusivity coefficient (m^2/s)
    :param w: wind speed (m/s)
    :param tau: particle lifetime (s)
    :return: length constant (m)
    """
    num = d * tau
    denom = 1 + (tau * w**2) / (4 * d)

    return np.sqrt(num / denom)   

def get_hit_rate(xs_src, ys_src, pos, w, d, r, a, tau, resolution=0.000001):
    """
    Calculate hit rate at specified position for grid of possible source locations.
    This is given by Eq. 7 in the infotaxis paper supplementary materials:
        http://www.nature.com/nature/journal/v445/n7126/extref/nature05464-s1.pdf

    Note: This function calculates hit rates for ALL grid cells. Occupied cells
    (walls/obstacles) are handled by maintaining -inf log probabilities for those
    cells in update_log_p_src(), which effectively excludes them from all
    probability calculations.

    :param xs_src: 1-D array of x-positions of source (m)
    :param ys_src: 1-D array of y-positions of source (m)
    :param pos: position where hit rate is calculated (m)
    :param w: wind speed (m/s)
    :param d: diffusivity coefficient (m^2/s)
    :param r: source emission rate (Hz)
    :param a: searcher size (m)
    :param tau: particle lifetime (s)
    :return: grid of hit rates, with one value per source location
    """
    # convert xs_src & ys_src
    xs_src_, ys_src_ = np.meshgrid(xs_src, ys_src, indexing='ij')

    dx = pos[0] - xs_src_
    dy = pos[1] - ys_src_

    # round dx's and dy's less than resolution down to zero
    dx[np.abs(dx) < resolution] = 0
    dy[np.abs(dy) < resolution] = 0

    # calc lambda
    lam = get_length_constant(w=w, d=d, tau=tau)

    # calc scale factor
    scale_factor = r / np.log(lam/a)

    # calc exponential term for wind in x-direction (positive)
    exp_term = np.exp((w/(2*d))*dx)

    # calc bessel term
    abs_dist = (dx**2 + dy**2) ** 0.5
    bessel_term = np.exp(log_k0(abs_dist / lam))

    # calc final hit rate
    hit_rate = scale_factor * exp_term * bessel_term

    return hit_rate

def get_p_sample(pos, xs, ys, dt, h, w, d, r, a, tau, log_p_src):
    """
    Get the probability of sampling h at position pos.
    :param pos: position
    :param xs: x-coords over which source prob distribution is defined
    :param ys: y-coords over which source prob distribution is defined
    :param dt: sampling interval
    :param h: sample value
    :param w: wind speed (m/s)
    :param d: diffusivity coefficient (m^2/s)
    :param r: source emission rate (particles per s)
    :param log_p_src: log probability distribution over source position
    :return: probability
    """
    # poisson probability of no hit given mean hit rate
    hit_rate = get_hit_rate(xs, ys, pos, w, d, r, a, tau)
    p_no_hits = np.exp(-dt * hit_rate)

    if h == 0:
        p_samples = p_no_hits
    elif h == 1:
        p_samples = 1 - p_no_hits
    else:
        raise Exception('h must be either 0 (no hit) or 1 (hit)')

    # get source distribution
    p_src = np.exp(log_p_src)
    p_src /= p_src.sum()

    # make sure p_src being 0 wins over p_sample being nan/inf
    p_samples[p_src == 0] = 0

    # average over all source positions
    p_sample = np.sum(p_samples * p_src)

    return p_sample

def update_log_p_src(pos, xs, ys, dt, h, w, d, r, a, tau, src_radius, log_p_src):
    """
    Update the log posterior over the src given sample h at position pos.
    :param pos: position
    :param xs: x-coords over which src prob is defined
    :param ys: y-coords over which src prob is defined
    :param h: sample value (0 for no hit, 1 for hit)
    :param w: wind speed (m/s)
    :param d: diffusivity coefficient (m^2/s)
    :param r: src emission rate (Hz)
    :param a: searcher size (m)
    :param tau: particle lifetime (s)
    :param src_radius: how close agent must be to src to detect it
    :param log_p_src: previous estimate of log src posterior
    :return: new (unnormalized) log src posterior
    """
    # Store mask of occupied cells (log_p_src == -inf) to preserve them
    occupied_mask = np.isinf(log_p_src) & (log_p_src < 0)

    # first get mean number of hits at pos given different src positions
    mean_hits = dt * get_hit_rate(
        xs_src=xs, ys_src=ys, pos=pos, w=w, d=d, r=r, a=a, tau=tau)

    # calculate log-likelihood (prob of h at pos given src position [Poisson])
    if h == 0:
        log_like = -mean_hits
    else:
        log_like = np.log(1 - np.exp(-mean_hits))

    # compute the new log src posterior
    log_p_src = log_like + log_p_src

    # set log prob to -inf everywhere within src_radius of pos
    xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')
    mask = ((pos[0] - xs_)**2 + (pos[1] - ys_)**2 < src_radius**2)
    log_p_src[mask] = -np.inf

    # Restore occupied cells to -inf (preserves walls/obstacles)
    log_p_src[occupied_mask] = -np.inf

    # if we've exhausted the search space start over
    if np.all(np.isinf(log_p_src)):
        log_p_src = np.ones(log_p_src.shape)
        # But keep occupied cells as -inf even after reset
        log_p_src[occupied_mask] = -np.inf

    return log_p_src


class InfotaxisNode(Node):
    def __init__(self):
        super().__init__('infotaxis_node')

        # Declare parameters
        self.declare_parameter('z_level', 5)
        self.declare_parameter('detection_threshold', 1.0)
        self.declare_parameter('robot_namespace', '/PioneerP3DX')
        self.declare_parameter('use_ideal_plume', False)  # For testing with analytical model

        # Get parameter values
        z_level = self.get_parameter('z_level').get_parameter_value().integer_value
        detection_threshold = self.get_parameter('detection_threshold').get_parameter_value().double_value
        robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value
        self.use_ideal_plume = self.get_parameter('use_ideal_plume').get_parameter_value().bool_value

        # Initialize modules
        self.grid_manager = GridManager(self, z_level)
        self.navigation_controller = NavigationController(self, robot_namespace)
        self.sensor_handler = SensorHandler(self, detection_threshold)

        self.grid_pos : Tuple[int,int] = (0,0) # Robot's current grid position
        self.gas_detected : bool = False       # Current gas detection state

        # Throttle infotaxis updates to prevent moving too fast
        self.declare_parameter('infotaxis_update_interval', 0.1)  # seconds between moves
        self.infotaxis_update_interval = self.get_parameter('infotaxis_update_interval').get_parameter_value().double_value
        self.last_infotaxis_time = self.get_clock().now()

        # Stop condition when close to source
        self.search_active = True  # Flag to track if search is still active

        # Ideal plume parameters (only used if use_ideal_plume=True)
        self.declare_parameter('true_source_x', 2.0)
        self.declare_parameter('true_source_y', 3.0)
        self.true_source_pos = None
        if self.use_ideal_plume:
            src_x = self.get_parameter('true_source_x').get_parameter_value().double_value
            src_y = self.get_parameter('true_source_y').get_parameter_value().double_value
            self.true_source_pos = np.array([src_x, src_y])
            self.get_logger().info(f'Using ideal plume with source at ({src_x}, {src_y})')

        # Subscribe to gas sensor
        self.gas_sensor_sub = self.create_subscription(
            GasSensor,
            '/fake_pid/Sensor_reading',
            self._gas_sensor_callback,
            int(1/dt)
        )

        # Subscribe to robot ground truth
        self.ground_truth_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f'{robot_namespace}/ground_truth',
            self._ground_truth_callback,
            10
        )

        # Publisher for probability distribution visualization
        self.probability_viz_pub = self.create_publisher(
            OccupancyGrid,
            '/infotaxis/probability_map',
            10
        )

        # Timer for periodic visualization updates (5 Hz)
        self.viz_timer = self.create_timer(0.2, self._publish_probability_visualization)

        # Create service client to get occupancy grid
        self.occupancy_client = self.create_client(Occupancy, '/gaden_environment/occupancyMap3D')

        # Wait for service to be available
        self.get_logger().info('Waiting for occupancy grid service...')
        while not self.occupancy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Occupancy grid service not available, waiting...')

        # Initialize grid manager
        if self.grid_manager.initialize(self.occupancy_client):
            self.get_logger().info('Infotaxis node initialized successfully')
        else:
            self.get_logger().error('Failed to initialize infotaxis node')

    def infotaxis_step(self, detection : bool, position : Tuple[int,int]):
        """
        Perform one step of the infotaxis algorithm.

        Args:
            detection: Whether gas is currently detected
            position: Current grid position (ix, iy)
        """
        log_prob_src = self.grid_manager.get_log_probabilities()
        xs = self.grid_manager.get_xs()
        ys = self.grid_manager.get_ys()

        # Convert grid position to world coordinates
        world_pos = self.grid_manager.get_cell_coordinates(position[0], position[1])

        # Update source probability distribution based on current detection
        log_prob_src = update_log_p_src(
            pos=world_pos[:2], xs=xs, ys=ys, dt=dt, h=int(detection), w=w,
            d=d, r=r, a=a, tau=tau, src_radius=src_radius, log_p_src=log_prob_src
        )
        s = entropy(log_prob_src)

        # Check if we're close to the most probable source location
        p_src_found = get_p_src_found(
            pos=world_pos[:2], xs=xs, ys=ys,
            log_p_src=log_prob_src, radius=src_radius
        )

        if p_src_found > 0.5:  # More than 50% probability that source is within src_radius
            if self.search_active:
                self.get_logger().info(
                    f'Source found! Probability of being within {src_radius}m: {p_src_found:.2%}. '
                    f'Current entropy: {s:.4f}. Stopping search.'
                )
                self.search_active = False
                self.grid_manager.set_log_probabilities(log_prob_src)
            return  # Stop moving

        # Calculate expected entropy decrease for each legal move
        legal_moves = self.grid_manager.get_legal_moves(position)
        delta_s_expecteds = []

        # Entropy decrease if source is found (drops to zero)
        delta_s_src_found = -s

        for move in legal_moves:
            # Convert grid move to world coordinates
            move_world = self.grid_manager.get_cell_coordinates(move[0], move[1])

            # Probability of finding source at this move
            p_src_found = get_p_src_found(
                pos=move_world[:2], xs=xs, ys=ys,
                log_p_src=log_prob_src, radius=src_radius
            )
            p_src_not_found = 1 - p_src_found

            # Calculate entropy changes for each possible sample outcome (hit/no hit)
            p_samples = []
            delta_s_given_samples = []

            for h in [0, 1]:  # 0 = no hit, 1 = hit
                # Probability of this sample
                p_sample = get_p_sample(
                    pos=move_world[:2], xs=xs, ys=ys, dt=dt, h=h,
                    w=w, d=d, r=r, a=a, tau=tau, log_p_src=log_prob_src
                )

                # Entropy after this sample
                log_p_src_after = update_log_p_src(
                    pos=move_world[:2], xs=xs, ys=ys, dt=dt, 
                    h=h, w=w, d=d, r=r, a=a,
                    tau=tau, src_radius=src_radius, log_p_src=log_prob_src
                )
                s_after = entropy(log_p_src_after)

                # Change in entropy for this sample
                delta_s = s_after - s

                p_samples.append(p_sample)
                delta_s_given_samples.append(delta_s)

            # Expected entropy decrease given source not found
            p_samples = np.array(p_samples)
            delta_s_given_samples = np.array(delta_s_given_samples)
            delta_s_src_not_found = p_samples.dot(delta_s_given_samples)

            # Total expected entropy decrease
            delta_s_expected = (p_src_found * delta_s_src_found) + \
                (p_src_not_found * delta_s_src_not_found)

            delta_s_expecteds.append(delta_s_expected)

        # Choose move that decreases entropy the most (most negative delta_s)
        delta_s_expecteds = np.array(delta_s_expecteds)
        best_move_idx = np.argmin(delta_s_expecteds)
        best_move = legal_moves[best_move_idx]

        self.get_logger().info(
            f'Current position: {position}, Best move: {best_move}, '
            f'Expected entropy decrease: {delta_s_expecteds[best_move_idx]:.4f}, '
            f'Current entropy: {s:.4f}'
        )
        self.grid_manager.set_log_probabilities(log_prob_src)
        self.navigation_controller.teleport_to_cell(best_move, self.grid_manager)

    def _ground_truth_callback(self, msg):
        """Callback for robot ground truth position."""
        # Extract position
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract orientation
        orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        # Update navigation controller
        self.navigation_controller.update_position(position, orientation)

        # Convert to grid coordinates (only if grid is initialized)
        if self.grid_manager.grid_origin is not None:
            self.grid_pos = self.grid_manager.get_cell_indices(position[0], position[1])
            self.navigation_controller.update_grid_position(self.grid_pos)

    def _gas_sensor_callback(self, msg):
        """Callback for gas sensor readings."""
        # Use ideal plume if enabled, otherwise use real sensor
        if self.use_ideal_plume:
            self.gas_detected = self._sample_ideal_plume()
        else:
            self.gas_detected = self.sensor_handler.process_sensor_reading(msg)

        # Only run infotaxis if grid is initialized
        if self.grid_manager.grid_origin is None:
            return

        # Throttle infotaxis updates - only run at specified interval
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_infotaxis_time).nanoseconds / 1e9

        if time_diff < self.infotaxis_update_interval:
            return  # Skip this update, too soon since last move

        # Update timestamp and run infotaxis
        self.last_infotaxis_time = current_time
        current_pos = self.grid_pos
        self.infotaxis_step(self.gas_detected, current_pos)

    def _sample_ideal_plume(self) -> bool:
        """Sample from ideal analytical plume model (for testing)."""
        if self.true_source_pos is None or self.grid_manager.grid_origin is None:
            return False

        # Get current position
        world_pos = self.grid_manager.get_cell_coordinates(self.grid_pos[0], self.grid_pos[1])

        # Calculate hit rate at current position
        xs_src = np.array([self.true_source_pos[0]])
        ys_src = np.array([self.true_source_pos[1]])

        hit_rate = get_hit_rate(
            xs_src=xs_src,
            ys_src=ys_src,
            pos=world_pos[:2],
            w=w,
            d=d,
            r=r,
            a=a,
            tau=tau
        )[0, 0]

        mean_hits = hit_rate * dt

        # Sample from Poisson distribution
        sample = int(np.random.poisson(mean_hits) > 0)

        return bool(sample)

    def _publish_probability_visualization(self):
        """Publish probability distribution as OccupancyGrid for visualization in RViz2."""
        # Don't publish if grid not initialized
        if self.grid_manager.grid_origin is None:
            return

        # Get log probabilities (nx, ny shape)
        log_probs = self.grid_manager.get_log_probabilities()

        # Convert to regular probabilities
        # Handle -inf values (they represent impossible locations)
        finite_mask = np.isfinite(log_probs)
        probs = np.zeros_like(log_probs)

        if np.any(finite_mask):
            # Convert log probabilities to probabilities
            probs[finite_mask] = np.exp(log_probs[finite_mask])
            # Normalize to sum to 1
            probs = probs / probs.sum()

        # Scale to occupancy grid range [0, 100]
        # Higher probability = higher occupancy value (darker in RViz)
        if probs.max() > 0:
            # Scale so max probability shows as 100
            occupancy_data = (probs / probs.max() * 100).astype(np.int8)
        else:
            occupancy_data = np.zeros_like(probs, dtype=np.int8)

        # Transpose back to (ny, nx) for occupancy grid row-major ordering
        occupancy_data = occupancy_data.T

        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Set map metadata
        msg.info = MapMetaData()
        msg.info.resolution = self.grid_manager.cell_size
        msg.info.width = self.grid_manager.grid_shape[0]  # nx
        msg.info.height = self.grid_manager.grid_shape[1]  # ny
        msg.info.origin.position.x = self.grid_manager.grid_origin[0]
        msg.info.origin.position.y = self.grid_manager.grid_origin[1]
        msg.info.origin.position.z = self.grid_manager.z_height
        msg.info.origin.orientation.w = 1.0

        # Flatten and convert to list (row-major order)
        msg.data = occupancy_data.flatten().tolist()

        # Publish
        self.probability_viz_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InfotaxisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
