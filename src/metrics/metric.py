from abc import ABC, abstractmethod


class Metric(ABC):

    @abstractmethod
    def compute(self, particles, uav):
        pass

    @abstractmethod
    def evaluate(self):
        pass
