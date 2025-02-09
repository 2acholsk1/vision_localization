from abc import ABC, abstractmethod


class Metric(ABC):

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
