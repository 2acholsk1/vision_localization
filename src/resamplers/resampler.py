from abc import ABC, abstractmethod


class Resampler(ABC):

    @abstractmethod
    def resampling(self, particles):
        pass

    @abstractmethod
    def set_number(self, new_number):
        pass
