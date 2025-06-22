import csv
import os
from datetime import datetime

import hydra
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self):
        self.errors = []
        self.timestamps = []
        self.gt_positions = []
        self.est_positions = []

        self.var_xs = []
        self.var_ys = []
        self.mean_vars = []
        self.entropies = []
        self.max_scores = []
        self.step_times = []
        self.convergence = []

    def log(self, error, gt_pos, est_pos, var_x, var_y, max_score, entropy, step_time, convergence):
        self.errors.append(error)
        self.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.gt_positions.append((gt_pos[0].item(), gt_pos[1].item()))
        self.est_positions.append((est_pos[0].item(), est_pos[1].item()))

        self.var_xs.append(var_x)
        self.var_ys.append(var_y)
        mean_var = (var_x + var_y) / 2.0
        self.mean_vars.append(mean_var)

        self.max_scores.append(max_score)
        self.entropies.append(entropy)
        self.step_times.append(step_time)

        self.convergence.append(convergence)

    def save_data(self, path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "timestamp", "error_px", "gt_x", "gt_y", "est_x", "est_y",
                "var_x", "var_y", "mean_var", "max_score", "entropy", "step_time_s", "converged"
            ])
            for i in range(len(self.errors)):
                writer.writerow([
                    i,
                    self.timestamps[i],
                    f"{self.errors[i]:.4f}",
                    f"{self.gt_positions[i][0]:.2f}", f"{self.gt_positions[i][1]:.2f}",
                    f"{self.est_positions[i][0]:.2f}", f"{self.est_positions[i][1]:.2f}",
                    f"{self.var_xs[i]:.4f}", f"{self.var_ys[i]:.4f}", f"{self.mean_vars[i]:.4f}",
                    f"{self.max_scores[i]:.4f}", f"{self.entropies[i]:.4f}", f"{self.step_times[i]:.4f}",
                    f"{self.convergence[i]}"
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
