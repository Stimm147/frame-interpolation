import numpy as np
import cv2
from interpolation_app.interpolation.base import Interpolator
from interpolation_app.logger import get_logger
import torch

logger = get_logger(__name__)


class MorphingInterpolator(Interpolator):
    """
    Interpolates frames using affine morphing based on keypoint matching.
    """

    def __init__(self, alpha: float = 0.5, method: str = "ORB", max_matches: int = 20):
        assert 0.0 <= alpha <= 1.0, "Alpha must be in range [0.0, 1.0]"
        assert method in ["ORB", "SIFT"], "Method must be 'ORB' or 'SIFT'"
        assert max_matches >= 3, "At least 3 matches are required for affine transform"

        self.alpha = alpha
        self.method = method.upper()
        self.max_matches = max_matches

        logger.info(
            f"MorphingInterpolator initialized with alpha={self.alpha}, method={self.method}, max_matches={self.max_matches}"
        )

        if self.method == "ORB":
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.method == "SIFT":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        if frame1.shape != frame2.shape:
            raise ValueError("Both frames must have the same shape.")

        h, w = frame1.shape[:2]
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            logger.warning("Not enough keypoints detected, falling back to blending.")
            return (
                (1 - self.alpha) * frame1.astype(np.float32)
                + self.alpha * frame2.astype(np.float32)
            ).astype(np.uint8)

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < 3:
            logger.warning("Not enough matches found, falling back to blending.")
            return (
                (1 - self.alpha) * frame1.astype(np.float32)
                + self.alpha * frame2.astype(np.float32)
            ).astype(np.uint8)

        matches = matches[: self.max_matches]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        pts_interp = (1.0 - self.alpha) * pts1 + self.alpha * pts2

        mat1, _ = cv2.estimateAffinePartial2D(pts1, pts_interp)
        mat2, _ = cv2.estimateAffinePartial2D(pts2, pts_interp)

        if mat1 is None or mat2 is None:
            logger.warning("Affine estimation failed, falling back to blending.")
            return (
                (1 - self.alpha) * frame1.astype(np.float32)
                + self.alpha * frame2.astype(np.float32)
            ).astype(np.uint8)

        warp1 = cv2.warpAffine(frame1, mat1, (w, h))
        warp2 = cv2.warpAffine(frame2, mat2, (w, h))

        warp1_tensor = torch.from_numpy(warp1).float().to("cuda")
        warp2_tensor = torch.from_numpy(warp2).float().to("cuda")

        interpolated = (1.0 - self.alpha) * warp1_tensor + self.alpha * warp2_tensor
        interpolated = interpolated.clamp(0, 255).byte().cpu().numpy()

        return interpolated
