"""CSV experiment logger for efe_pmfs. Simpler than efe_igdm's, no particle-filter fields."""
import csv
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_dir='~/efe_pmfs_logs'):
        self.log_dir = os.path.expanduser(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = os.path.join(self.log_dir, f'pmfs_log_{timestamp}.csv')
        self.log_file = open(self.log_filename, 'w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow([
            'step', 'elapsed', 'mode', 'sensor_ppm', 'is_hit',
            'robot_x', 'robot_y',
            'est_x', 'est_y', 'std_x', 'std_y',
            'entropy', 'bi_optimal', 'bi_threshold', 'dead_end',
            'num_branches', 'global_path_len', 'global_path_idx',
            'source_updates', 'step_time_s',
        ])
        self.log_file.flush()

    def log_step(self, step, elapsed, mode, sensor_ppm, is_hit,
                 robot_pos, est_pos, est_std, entropy,
                 bi_optimal, bi_threshold, dead_end,
                 num_branches, global_path_len, global_path_idx,
                 source_updates, step_time_s):
        self.writer.writerow([
            step, f'{elapsed:.3f}', mode, f'{sensor_ppm:.4f}', int(bool(is_hit)),
            f'{robot_pos[0]:.4f}', f'{robot_pos[1]:.4f}',
            f'{est_pos[0]:.4f}', f'{est_pos[1]:.4f}',
            f'{est_std[0]:.4f}', f'{est_std[1]:.4f}',
            f'{entropy:.4f}', f'{bi_optimal:.4f}', f'{bi_threshold:.4f}',
            int(bool(dead_end)),
            int(num_branches), int(global_path_len), int(global_path_idx),
            int(source_updates), f'{step_time_s:.4f}',
        ])
        self.log_file.flush()

    def save_summary(self, steps, total_dist_m, elapsed_s, avg_step_s,
                     est_x, est_y, est_error_m):
        path = self.log_filename.replace('.csv', '_summary.txt')
        with open(path, 'w') as f:
            f.write(
                f"steps: {steps}\n"
                f"total_distance_m: {total_dist_m:.3f}\n"
                f"elapsed_s: {elapsed_s:.3f}\n"
                f"avg_step_s: {avg_step_s:.4f}\n"
                f"estimated_source: ({est_x:.3f}, {est_y:.3f})\n"
                f"error_m: {est_error_m:.3f}\n"
            )

    def close(self):
        if self.log_file:
            self.log_file.close()
