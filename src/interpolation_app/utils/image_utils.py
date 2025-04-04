import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


class ImageEvaluator:
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        score = ssim(img1, img2, channel_axis=-1)
        return score

    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        return psnr(img1, img2, data_range=255)
