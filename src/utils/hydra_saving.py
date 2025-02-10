import os
import hydra
import matplotlib.pyplot as plt

def save_results(metric, plot_figure=None):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "error_steps.csv")

    metric.save_data(csv_path)

    if plot_figure:
        plot_path = os.path.join(results_dir, "error_plot.png")
        plot_figure.savefig(plot_path)

        plt.close(plot_figure)
