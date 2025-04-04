import numpy as np
from interpolation_app.interpolation.base import Interpolator
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


class NaiveInterpolator(Interpolator):
    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        return ((frame1.astype(np.float32) + frame2.astype(np.float32)) / 2).astype(np.uint8)
