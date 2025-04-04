from pathlib import Path
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from interpolation_app.interpolation.naive import NaiveInterpolator
from interpolation_app.utils.triplet_dataset import TripletDataset
from interpolation_app.utils.image_utils import ImageEvaluator
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


def main():
    ROOT = Path.cwd()
    dataset_dir = ROOT / "data" / "triplet_dataset"
    output_dir = ROOT / "results" / "naive"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TripletDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    interpolator = NaiveInterpolator()

    ssim_total, psnr_total, count = 0.0, 0.0, 0

    logger.info(f"Starting evaluation on {len(dataset)} sequences...\n")

    for sample in tqdm(dataloader, desc="Evaluating"):
        name = sample["name"][0]
        before = sample["before"][0].numpy()
        after = sample["after"][0].numpy()
        gt = sample["ground_truth"][0].numpy()

        interpolated = interpolator.interpolate(before, after)

        ssim = ImageEvaluator.compute_ssim(interpolated, gt)
        psnr = ImageEvaluator.compute_psnr(interpolated, gt)

        out_path = output_dir / f"{name}_interp.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(interpolated, cv2.COLOR_RGB2BGR))

        ssim_total += ssim
        psnr_total += psnr
        count += 1

    avg_ssim = ssim_total / count if count else 0.0
    avg_psnr = psnr_total / count if count else 0.0

    logger.info("\n=== Final Evaluation Results ===")
    logger.info(f"Processed sequences: {count}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")


if __name__ == "__main__":
    main()
