import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class SystematicResampler(Resampler):
    """An adaptation of systematic resampling, where the particle positions are selected deterministically,
    but with some adjustments to improve accuracy or efficiency.
    It usually involves adding randomness or changes to the interval division in systematic resampling.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        positions = (np.random.rand() + np.arange(self.number)) / self.number
        cumulative_weights = np.cumsum(np.fromiter((p.weight for p in particles), dtype=float))
        indexes = np.searchsorted(cumulative_weights, positions, side='right')
        return indexes

    def set_number(self, new_number):
        self.number = new_number
