import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class BootstrapResampler(Resampler):
    """A general method where particles are resampled with replacement,
    meaning some particles may appear multiple times, and others may be left out.
    This method is simple but can lead to particle duplication

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        indexes = np.random.choice(len(particles), size=self.number, replace=True)

        return indexes

    def set_number(self, new_number):
        self.number = new_number
