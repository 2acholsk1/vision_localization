import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class SystematicResampler(Resampler):
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        positions = (np.random.rand() + np.arange(self.number)) / self.number
        cumulative_weights = np.cumsum(np.fromiter((p.weight for p in particles), dtype=float))
        indexes = np.searchsorted(cumulative_weights, positions, side='right')
        return indexes

    def set_number(self, new_number):
        self.number = new_number
