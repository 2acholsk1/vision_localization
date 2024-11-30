import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class ResidualResampler(Resampler):
    """A method that combines elements of stratified and multinomial resampling.
    It first selects particles deterministically based on the integer part of their weights,
    then uses multinomial resampling for the residual fractional part of the weights.

    Args:
        Resampler (ABC): Abstract class
    """
    def __init__(self, number):
        self.number = number

    def resampling(self, particles: list[Particle]):
        weights = np.array([p.weight for p in particles])
        weights /= np.sum(weights)

        integer_part = np.floor(weights * self.number).astype(int)
        remaining_particles = self.number - np.sum(integer_part)

        fractional_part = weights * self.number - integer_part
        residual_indexes = np.random.choice(
            len(particles),
            size=remaining_particles,
            p=fractional_part / np.sum(fractional_part)
            )

        new_particles = []
        new_indexes = []

        for i, count in enumerate(integer_part):
            new_particles.extend([particles[i]] * count)
            new_indexes.extend([i] * count)

        for idx in residual_indexes:
            new_particles.append(particles[idx])
            new_indexes.append(idx)

        return new_indexes

    def set_number(self, new_number):
        self.number = new_number
