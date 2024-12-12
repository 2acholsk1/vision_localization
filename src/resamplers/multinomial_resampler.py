import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class MultinomialResampler(Resampler):
    """A resampling technique where particles are chosen based on their weights,
    with each particle having a probability proportional to its weight.
    Some particles can be selected multiple times, while others may not be selected at all.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        weights = np.array([p.weight for p in particles])
        indexes = np.random.choice(range(self.number), size=self.number, p=weights/weights.sum())
        return indexes

    def set_number(self, new_number):
        self.number = new_number
