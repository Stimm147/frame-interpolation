from interpolation_app.interpolation.base import Interpolator
from interpolation_app.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class WeightedAverageInterpolator(Interpolator):
    def __init__(self, alpha: float = 0.5):
        assert 0.0 <= alpha <= 1.0, "Alpha must be in range [0.0, 1.0]"
        self.alpha = alpha

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Interpolates a frame as a weighted average of two input frames.

        Args:
            frame1 (np.ndarray): first frame (e.g., at t=0)
            frame2 (np.ndarray): second frame (e.g., at t=1)

        Returns:
            np.ndarray: interpolated frame at time t=alpha
        """
        if frame1.shape != frame2.shape:
            raise ValueError("Both frames must have the same shape.")

        interpolated = (1.0 - self.alpha) * frame1.astype(
            np.float32
        ) + self.alpha * frame2.astype(np.float32)
        return np.clip(interpolated, 0, 255).astype(np.uint8)
