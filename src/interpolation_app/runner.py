import cv2
from pathlib import Path
from interpolation_app.interpolation.naive import NaiveInterpolator
from interpolation_app.utils.image_utils import ImageEvaluator
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        logger.error(f"Failed to load image: {path}")
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def main():
    dataset_path = Path("data/triplet_dataset")
    frame1_path = dataset_path / "im1.png"
    frame2_path = dataset_path / "im3.png"
    gt_middle_path = dataset_path / "im2.png"  # ground truth middle frame

    frame1 = load_image(frame1_path)
    frame2 = load_image(frame2_path)
    gt_middle = load_image(gt_middle_path)

    interpolator = NaiveInterpolator()
    interpolated = interpolator.interpolate(frame1, frame2)

    # Save output
    output_path = dataset_path / "interpolated.png"
    cv2.imwrite(str(output_path), interpolated)
    logger.info(f"Saved interpolated frame to {output_path}")

    # Evaluation
    ssim_score = ImageEvaluator.compute_ssim(interpolated, gt_middle)
    psnr_score = ImageEvaluator.compute_psnr(interpolated, gt_middle)

    logger.info(f"SSIM: {ssim_score:.4f}")
    logger.info(f"PSNR: {psnr_score:.2f} dB")


if __name__ == "__main__":
    main()
