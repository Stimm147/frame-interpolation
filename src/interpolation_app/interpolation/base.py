from abc import ABC, abstractmethod
import numpy as np


class Interpolator(ABC):
    @abstractmethod
    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        pass
