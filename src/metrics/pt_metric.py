import csv
import os
from datetime import datetime

import hydra
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self):
        self.errors = []
        self.timestamps = []
        self.gt_positions = []     # ground truth (x, y)
        self.est_positions = []    # estimated by Kalman (x, y)

    def log(self, error, gt_pos, est_pos):
        self.errors.append(error)
        self.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.gt_positions.append((gt_pos[0].item(), gt_pos[1].item()))
        self.est_positions.append((est_pos[0].item(), est_pos[1].item()))

    def save_data(self, path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "timestamp", "error_px", "gt_x", "gt_y", "est_x", "est_y"])
            for i, (t, e, gt, est) in enumerate(zip(self.timestamps, self.errors, self.gt_positions, self.est_positions)):
                writer.writerow([
                    i, t, f"{e:.4f}",
                    f"{gt[0]:.2f}", f"{gt[1]:.2f}",
                    f"{est[0]:.2f}", f"{est[1]:.2f}"
                ])



def save_results(metric_logger, plot_figure=None):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "error_steps.csv")
    metric_logger.save_data(csv_path)

    if plot_figure:
        plot_path = os.path.join(results_dir, "error_plot.png")
        plot_figure.savefig(plot_path)
        plt.close(plot_figure)
