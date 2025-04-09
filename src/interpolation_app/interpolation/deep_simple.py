import numpy as np
import torch
from interpolation_app.interpolation.base import Interpolator
from interpolation_app.models.simple_interpolator import SimpleCNNInterpolator
from interpolation_app.logger import get_logger

logger = get_logger(__name__)


class DeepCNNInterpolator(Interpolator):
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNNInterpolator.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor1 = (
                torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            tensor2 = (
                torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            x = torch.cat([tensor1, tensor2], dim=1).to(self.device)
            output = self.model(x).clamp(0, 1)
            output = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            return output
