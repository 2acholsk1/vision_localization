import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class StratifiedResampler(Resampler):
    """Similar to systematic resampling, but it divides the unit interval
    into equal segments and ensures each segment selects a particle.
    This method reduces variance compared to basic systematic resampling, especially when weights are uneven.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        positions = (np.random.rand(self.number) + np.arange(self.number)) / self.number
        cumulative_weights = np.cumsum(np.array([p.weight for p in particles]))
        cumulative_weights /= cumulative_weights[-1]
        indexes = np.searchsorted(cumulative_weights, positions)
        return indexes

    def set_number(self, new_number):
        self.number = new_number
