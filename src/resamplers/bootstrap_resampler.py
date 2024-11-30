import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class BootstrapResampler(Resampler):
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        indexes = np.random.choice(len(particles), size=self.number, replace=True)

        return indexes

    def set_number(self, new_number):
        self.number = new_number
