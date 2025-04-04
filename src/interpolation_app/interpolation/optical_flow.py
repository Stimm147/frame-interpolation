import numpy as np
import cv2
from interpolation_app.interpolation.base import Interpolator
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


class OpticalFlowInterpolator(Interpolator):
    """
    Interpolates frames using dense optical flow (DIS algorithm).
    Computes both forward and backward flow, warps both frames to t=0.5,
    and blends them. Always uses best quality preset for accuracy.
    """

    def __init__(self, alpha: float = 0.5, use_multiscale: bool = False):
        assert 0.0 <= alpha <= 1.0, "Alpha must be in range [0.0, 1.0]"
        self.alpha = alpha
        self.use_multiscale = use_multiscale

        self.dis = cv2.DISOpticalFlow_create(preset=2)
        self.dis.setUseSpatialPropagation(True)

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        if frame1.shape != frame2.shape:
            raise ValueError("Both frames must have the same shape.")

        h, w = frame1.shape[:2]
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        if self.use_multiscale:
            gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.5)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.5)

        flow_fw = self.dis.calc(gray1, gray2, None)
        flow_bw = self.dis.calc(gray2, gray1, None)

        flow_fw_half = flow_fw * self.alpha
        flow_bw_half = flow_bw * self.alpha

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        map_x_fw = grid_x + flow_fw_half[..., 0]
        map_y_fw = grid_y + flow_fw_half[..., 1]
        warped1 = cv2.remap(frame1, map_x_fw, map_y_fw, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        map_x_bw = grid_x - flow_bw_half[..., 0]
        map_y_bw = grid_y - flow_bw_half[..., 1]
        warped2 = cv2.remap(frame2, map_x_bw, map_y_bw, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        blended = (1 - self.alpha) * warped1.astype(np.float32) + self.alpha * warped2.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
