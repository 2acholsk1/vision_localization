import numpy as np

from particle import Particle


class ResamplerSimple:
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        cumulative_weights = np.cumsum(np.fromiter((p.weight for p in particles), dtype=float))
        positions = (np.random.rand() + np.arange(self.number)) / self.number
        indexes = np.searchsorted(cumulative_weights, positions)
        return indexes

    def set_number(self, new_number):
        self.number = new_number
