import numpy as np

from src.particle import Particle
from src.resamplers.resampler import Resampler


class RejectionResampler(Resampler):
    def __init__(self, number, rejection_treshold):
        self.number = number
        self.rejection_threshold = rejection_treshold

    def resampling(self, particles: list[Particle]):
        weights = np.array([p.weight for p in particles])
        weights /= np.sum(weights)

        new_indexes = []

        for _ in range(self.number):
            random_choice = np.random.rand()
            chosen_index = np.random.choice(len(particles), p=weights)

            if random_choice < weights[chosen_index] / self.rejection_threshold:
                new_indexes.append(chosen_index)

        if len(new_indexes) < self.number:
            raise ValueError("Not enough particles selected after rejection step")

        return new_indexes

    def set_number(self, new_number):
        self.number = new_number
