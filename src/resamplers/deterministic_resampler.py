import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class DeterministicResampler(Resampler):
    """A non-random resampling technique where particles are selected based on a deterministic criterion,
    often based on their weights. It selects the particles with the highest weights
    to maximize the representation of the distribution.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        weights = np.array([p.weight for p in particles])
        weights /= np.sum(weights)

        sorted_indexes = np.argsort(weights)[::-1]
        return sorted_indexes[:self.number]

    def set_number(self, new_number):
        self.number = new_number
