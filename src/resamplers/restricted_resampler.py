import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class RestrictedResampler(Resampler):
    """A variation of resampling where only a subset of particles are resampled,
    while others are kept unchanged. This method allows you to impose limits on
    how many particles can be replaced to maintain some diversity in the population.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number, max_resample=0.25):
        self.number = number
        self.max_resample = max_resample

    def resampling(self, particles: list[Particle]):
        weights = np.array([p.weight for p in particles])
        weights /= np.sum(weights)
        max_resample_count = int(self.number * self.max_resample)
        resample_indexes = np.random.choice(len(particles), size=max_resample_count, p=weights)
        remaining_indexes = [i for i in range(len(particles)) if i not in resample_indexes]
        new_indexes = list(remaining_indexes) + list(resample_indexes)
        return new_indexes

    def set_number(self, new_number):
        self.number = new_number
