import csv
import os
from datetime import datetime

class ExperimentLogger:
    """
    Handles CSV logging for the IGDM experiment.
    """
    def __init__(self, log_dir='~/igdm_logs'):
        self.log_dir = os.path.expanduser(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = os.path.join(self.log_dir, f'igdm_log_{timestamp}.csv')
        
        self.log_file = open(self.log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write Header
        self.csv_writer.writerow([
            'step', 'elapsed_time', 'entropy', 'std_dev_x', 'std_dev_y', 'std_dev_Q',
            'est_x', 'est_y', 'est_Q', 'sensor_value', 'continuous_measurement', 'threshold',
            'num_branches', 'best_utility', 'J1_entropy_gain', 'J2_travel_cost',
            'robot_x', 'robot_y', 'sigma_m', 'bi_optimal', 'bi_threshold', 'dead_end_detected',
            'planner_mode', 'global_path_length', 'global_waypoint_index'
        ])
        self.log_file.flush()
    
    def log_step(self, step_count, particle_filter, sensor_value, current_pos, 
                 params, debug_info, bi_optimal, dead_end_detected, 
                 planner_mode, global_path_len, global_path_index):
        """
        Logs a single step of the experiment.
        """
        means, stds = particle_filter.get_estimate()
        entropy = particle_filter.get_entropy()
        debug_info = debug_info or {}
        
        # Build the row (Logic moved from igdm.py)
        row = [
            step_count, 
            0, # elapsed time placeholder (can be filled if needed)
            f'{entropy:.4f}',
            f'{stds["x"]:.4f}', f'{stds["y"]:.4f}', f'{stds["Q"]:.4f}',
            f'{means["x"]:.4f}', f'{means["y"]:.4f}', f'{means["Q"]:.4f}',
            f'{sensor_value:.4f}', f'{sensor_value:.4f}', '0.0',
            debug_info.get("num_branches", 0),
            f'{debug_info.get("best_utility", 0.0):.4f}',
            f'{debug_info.get("best_entropy_gain", 0.0):.4f}',
            f'{debug_info.get("best_travel_cost", 0.0):.4f}',
            f'{current_pos[0]:.4f}', f'{current_pos[1]:.4f}',
            f'{params["sigma_m"]:.4f}',
            f'{bi_optimal:.4f}',
            f'{dead_end_detected}', # simplified from boolean
            1 if dead_end_detected else 0,
            planner_mode,
            global_path_len,
            global_path_index
        ]
        
        self.csv_writer.writerow(row)
        self.log_file.flush()

    def save_summary(self, step_count, total_dist, elapsed_time, avg_comp_time, 
                     est_x, est_y, est_error):
        """
        Saves the final summary text file.
        """
        summary = (
            f"ST: {step_count} steps\n"
            f"TD: {total_dist:.2f} m\n"
            f"Time: {elapsed_time:.2f} s\n"
            f"Avg Comp: {avg_comp_time:.4f} s\n"
            f"Error: {est_error:.3f} m\n"
            f"Est Source: ({est_x:.3f}, {est_y:.3f})"
        )
        
        summary_filename = self.log_filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write(summary)
        
        return summary

    def close(self):
        if self.log_file:
            self.log_file.close()