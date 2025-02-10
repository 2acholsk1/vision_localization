import matplotlib.pyplot as plt
import numpy as np

from src.logger import log
from src.metrics.metric import Metric
from src.particle import Particle
from src.uav import UAV


class EuclideanDistance(Metric):
    def __init__(self):
        self.total_absolute_error = np.array([0.0, 0.0])
        self.mean_absolute_error = np.array([0.0, 0.0])
        self.error_steps_x = np.array([])
        self.error_steps_y = np.array([])
        self.steps = 0

    def compute(self, particles: list[Particle], uav: UAV):
        error = np.array([0.0, 0.0, 0.0, 0.0])

        mean_x = 0.0
        mean_y = 0.0
        for particle in particles:
            mean_x += particle.get_position()[0]
            mean_y += particle.get_position()[1]
        mean_x /= len(particles)
        mean_y /= len(particles)

        for paricle in particles:
            error[0] += abs(uav.get_position()[0] - paricle.get_position()[0])
            error[1] += abs(uav.get_position()[1] - paricle.get_position()[1])
        error[2] = abs(uav.get_position()[0] - mean_x)
        error[3] = abs(uav.get_position()[1] - mean_y)

        self.error_steps_x = np.append(self.error_steps_x, error[2])
        self.error_steps_y = np.append(self.error_steps_y, error[3])

        self.total_absolute_error[0] += error[0]
        self.total_absolute_error[1] += error[1]
        self.mean_absolute_error[0] += error[2]
        self.mean_absolute_error[1] += error[3]

        self.steps += 1

    def evaluate(self, ploting):
        log.info(
            "MEAN EUCLIDEAN DISTANCE\n"
            "Total Absolute Error (TAE): x=%s; y=%s\n"
            "Mean Absolute Error (MAE): x=%s; y=%s\n",
            self.total_absolute_error[0], self.total_absolute_error[1],
            self.mean_absolute_error[0], self.mean_absolute_error[1]
        )
        if ploting:
            fig, ax = plt.subplots()
            t = np.arange(0, self.steps, 1)
            ax.plot(t, self.error_steps_x, label="X_error")
            ax.plot(t, self.error_steps_y, label="Y_error")
            ax.legend()
            ax.set_xlabel("Step")
            ax.set_ylabel("Error [px]")
            ax.set_title("Diff between Real Position and Mean Particles Value")
            ax.grid()
            return fig
        return None

    def save_data(self, path):
        np.savetxt(
            path,
            np.column_stack((self.error_steps_x, self.error_steps_y,)),
            delimiter=",",
            header="error_steps_x,error_steps_y",
            comments="",
            fmt="%.5f"
            )

    def get_shape_params(self, particles: list[Particle]):
        maximal_x = max(particles, key=lambda obj: obj.x)
        minimal_x = min(particles, key=lambda obj: obj.x)
        maximal_y = max(particles, key=lambda obj: obj.y)
        minimal_y = min(particles, key=lambda obj: obj.y)
        mean_x = 0.0
        mean_y = 0.0
        for particle in particles:
            mean_x += particle.get_position()[0] * particle.last_weight * len(particles)
            mean_y += particle.get_position()[1] * particle.last_weight * len(particles)
        mean_x /= len(particles)
        mean_y /= len(particles)

        left_point = [mean_x - (mean_x - minimal_x.x)/2, mean_y]
        right_point = [mean_x + (maximal_x.x - mean_x)/2, mean_y]
        down_point = [mean_x, mean_y - (maximal_y.y - mean_y)/2]
        upper_point = [mean_x, mean_y + (mean_y - minimal_y.y)/2]

        major = abs(right_point[0] - left_point[0]) / 2
        minor = abs(down_point[1] - upper_point[1]) / 2

        extreme_points = np.array(
            [maximal_x.get_position(),
             maximal_y.get_position(),
             minimal_x.get_position(),
             minimal_y.get_position()],
            dtype=np.int32)
        extreme_points = extreme_points.reshape((-1, 1, 2))

        mean_points = np.array([upper_point, right_point, down_point, left_point], dtype=np.int32)
        mean_points = mean_points.reshape((-1, 1, 2))

        centroid = mean_points.mean(axis=0)

        return major, minor, centroid
