import warnings
import numpy as np
from interpolation_app.utils.image_utils import ImageEvaluator


def test_ssim_and_psnr_identical():
    image = np.ones((64, 64, 3), dtype=np.uint8) * 128

    ssim = ImageEvaluator.compute_ssim(image, image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psnr = ImageEvaluator.compute_psnr(image, image)

    assert ssim == 1.0
    assert psnr == float("inf")
