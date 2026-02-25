"""
Dead End Detector for Dual-Mode RRT-Infotaxis GSL

Based on the paper "Gas Source Localization in Unknown Indoor Environments
Using Dual-Mode Information-Theoretic Search" by Kim et al., 2025.

Section IV.B.2 - Dead End Detection (Equations 20-21)

The dead end detector determines when the local RRT planner has reached
a situation where no useful information can be obtained, triggering a
switch to the global planner.
"""

import numpy as np


class DeadEndDetector:
    """
    Detects dead ends in RRT-Infotaxis local planning.

    A dead end occurs when the optimal branch information falls below
    an adaptive threshold, indicating that the local planner cannot
    find informative paths anymore.

    This implements Equations 20-21 from the paper:
    - BI* = max(BI(V_b)) for all branches V_b  (Eq. 20)
    - BI_thresh_{k+1} = ε · BI_thresh_k + (1-ε) · BI*  (Eq. 21)
    - Dead end detected when: BI* < BI_thresh_k

    Attributes
    ----------
    epsilon : float
        Weight parameter for threshold update (0 < ε < 1).
        Higher values = slower threshold adaptation.
        Paper suggests ε around 0.8-0.9.

    bi_threshold : float
        Current adaptive threshold for branch information.

    bi_threshold_history : list
        History of threshold values for debugging/analysis.

    bi_optimal_history : list
        History of optimal branch information values.
    """

    def __init__(self, epsilon: float = 0.9, initial_threshold: float = 0.1):
        """
        Initialize the dead end detector.

        Parameters
        ----------
        epsilon : float, optional
            Weight for threshold update (default: 0.9).
            Must be in range (0, 1).
            Higher values make threshold change more slowly.

        initial_threshold : float, optional
            Initial value for BI_thresh (default: 0.1).
            This will be updated adaptively after first step.
        """
        if not (0 < epsilon < 1):
            raise ValueError(f"epsilon must be in range (0, 1), got {epsilon}")

        self.epsilon = epsilon
        self.bi_threshold = initial_threshold

        # History tracking for analysis
        self.bi_threshold_history = [initial_threshold]
        self.bi_optimal_history = []

        # Track step count
        self.step_count = 0

        # Flag for first update
        self.initialized = False

    def update_threshold(self, bi_optimal: float) -> None:
        """
        Update the adaptive threshold based on current optimal branch information.

        Implements Equation 21:
        BI_thresh_{k+1} = ε · BI_thresh_k + (1-ε) · BI*

        This acts like a low-pass filter, causing the threshold to trail
        the optimal branch information with a delay.

        Parameters
        ----------
        bi_optimal : float
            Current optimal branch information value (BI*).
        """
        # Store history
        self.bi_optimal_history.append(bi_optimal)

        # Update threshold using exponential moving average
        # This is essentially a low-pass filter
        self.bi_threshold = (
            self.epsilon * self.bi_threshold +
            (1.0 - self.epsilon) * bi_optimal
        )

        # Store updated threshold
        self.bi_threshold_history.append(self.bi_threshold)

        self.step_count += 1
        self.initialized = True

    def is_dead_end(self, bi_optimal: float) -> bool:
        """
        Check if current state represents a dead end.

        A dead end is detected when the optimal branch information
        falls below the adaptive threshold:
        Dead end = (BI* < BI_thresh_k)

        When BI* drops sharply (no useful information available),
        it becomes smaller than the delayed threshold, triggering
        dead end detection.

        Parameters
        ----------
        bi_optimal : float
            Current optimal branch information value (BI*).

        Returns
        -------
        bool
            True if dead end detected, False otherwise.
        """
        # Clamp to zero to prevent -inf from poisoning the threshold EMA
        bi_optimal = max(bi_optimal, 0.0)

        if not self.initialized:
            # On first step, just update threshold without detecting dead end
            self.update_threshold(bi_optimal)
            return False

        # Check dead end condition BEFORE updating threshold
        # This ensures we compare against the previous threshold
        is_dead_end = bi_optimal < self.bi_threshold

        # Update threshold for next iteration
        self.update_threshold(bi_optimal)

        return is_dead_end

    def get_status(self) -> dict:
        """
        Get current status of the dead end detector.

        Returns
        -------
        dict
            Dictionary containing:
            - 'bi_threshold': Current threshold value
            - 'step_count': Number of updates performed
            - 'epsilon': Current epsilon value
            - 'last_bi_optimal': Most recent optimal BI value (if available)
            - 'margin': Difference between last BI and threshold (if available)
        """
        status = {
            'bi_threshold': self.bi_threshold,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }

        if self.bi_optimal_history:
            last_bi = self.bi_optimal_history[-1]
            status['last_bi_optimal'] = last_bi
            status['margin'] = last_bi - self.bi_threshold

        return status

    def reset(self, initial_threshold: float = None) -> None:
        """
        Reset the detector to initial state.

        Parameters
        ----------
        initial_threshold : float, optional
            New initial threshold value.
            If None, uses the current threshold value.
        """
        if initial_threshold is not None:
            self.bi_threshold = initial_threshold

        self.bi_threshold_history = [self.bi_threshold]
        self.bi_optimal_history = []
        self.step_count = 0
        self.initialized = False

    def get_history(self) -> dict:
        """
        Get complete history for analysis and plotting.

        Returns
        -------
        dict
            Dictionary containing:
            - 'bi_threshold_history': List of threshold values
            - 'bi_optimal_history': List of optimal BI values
            - 'step_count': Total number of steps
        """
        return {
            'bi_threshold_history': self.bi_threshold_history.copy(),
            'bi_optimal_history': self.bi_optimal_history.copy(),
            'step_count': self.step_count
        }


class BranchInformation:
    """
    Helper class to compute branch information (BI) from RRT paths.

    This implements Equation 19 from the paper:
    BI(V_b) = Σ_{i=1}^{m} γ^{i-1} · I(v_{b,i})

    where:
    - V_b is a branch (path) in the RRT tree
    - v_{b,i} is the i-th vertex in the branch
    - I(v_{b,i}) is the mutual information at vertex v_{b,i}
    - γ is a discount factor
    """

    @staticmethod
    def compute(path_entropy_gains: list, discount_factor: float = 0.8) -> float:
        """
        Compute branch information from a path's entropy gains.

        Implements Equation 19:
        BI(V_b) = Σ_{i=1}^{m} γ^{i-1} · I(v_{b,i})

        Parameters
        ----------
        path_entropy_gains : list of float
            List of mutual information values along the path.
            Should NOT include the starting position.

        discount_factor : float, optional
            Discount factor γ (default: 0.8).
            Assigns higher weight to information from nearer future.

        Returns
        -------
        float
            Total branch information (BI).
        """
        bi = 0.0
        for i, entropy_gain in enumerate(path_entropy_gains):
            discounted_gain = (discount_factor ** i) * entropy_gain
            bi += discounted_gain
        return bi

    @staticmethod
    def compute_optimal(all_branch_information: list) -> float:
        """
        Compute optimal branch information (BI*) from all branches.

        Implements Equation 20:
        BI* = max_{V_b ∈ V} BI(V_b)

        Parameters
        ----------
        all_branch_information : list of float
            List of branch information values for all branches.

        Returns
        -------
        float
            Optimal branch information (BI*).
            Returns -inf if list is empty.
        """
        if not all_branch_information:
            return -np.inf
        return max(all_branch_information)


def example_usage():
    """
    Example usage of the dead end detector.
    """
    import matplotlib.pyplot as plt

    # Initialize detector
    detector = DeadEndDetector(epsilon=0.85, initial_threshold=0.5)

    # Simulate a scenario where BI gradually decreases (approaching dead end)
    bi_values = [
        1.2, 1.1, 1.0, 0.95, 0.9,  # Normal information gain
        0.85, 0.8, 0.75,            # Decreasing gain
        0.3, 0.2, 0.15,             # Sharp drop (dead end!)
        0.1, 0.12, 0.11             # Stuck at low values
    ]

    dead_end_detected = []

    print("Step | BI*   | Threshold | Dead End?")
    print("-" * 45)

    for i, bi_optimal in enumerate(bi_values):
        is_dead_end = detector.is_dead_end(bi_optimal)
        dead_end_detected.append(is_dead_end)

        status = detector.get_status()
        print(f"{i:4d} | {bi_optimal:5.2f} | {status['bi_threshold']:9.3f} | {'YES' if is_dead_end else 'No'}")

    # Plot results
    history = detector.get_history()

    plt.figure(figsize=(12, 6))

    steps = range(len(bi_values))
    plt.plot(steps, bi_values, 'b-o', label='BI* (Optimal Branch Info)', linewidth=2)
    plt.plot(steps, history['bi_threshold_history'][1:], 'r--', label='BI_threshold', linewidth=2)

    # Mark dead end points
    dead_end_steps = [i for i, de in enumerate(dead_end_detected) if de]
    if dead_end_steps:
        dead_end_bi = [bi_values[i] for i in dead_end_steps]
        plt.scatter(dead_end_steps, dead_end_bi, color='red', s=200,
                   marker='X', label='Dead End Detected', zorder=5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Branch Information', fontsize=12)
    plt.title('Dead End Detection Example\n(Red X marks indicate dead end detection)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dead_end_detection_example.png', dpi=150)
    print("\nPlot saved to: dead_end_detection_example.png")


if __name__ == '__main__':
    example_usage()
