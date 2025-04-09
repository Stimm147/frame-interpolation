from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from interpolation_app.logger import get_logger
import random

logger = get_logger(__name__)


class TripletDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform=None,
        limit: int = None,
        seed: int = 42,
        as_tensor: bool = False,
    ):
        self.root_dir = Path(root_dir)
        folders = sorted([f for f in self.root_dir.iterdir() if f.is_dir()])

        if limit is not None:
            random.seed(seed)
            folders = random.sample(folders, min(limit, len(folders)))

        self.folders = folders
        self.transform = transform
        self.as_tensor = as_tensor

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray | torch.Tensor]:
        folder = self.folders[idx]
        im1 = cv2.imread(str(folder / "im1.png"))
        im2 = cv2.imread(str(folder / "im2.png"))
        im3 = cv2.imread(str(folder / "im3.png"))

        if im1 is None or im2 is None or im3 is None:
            raise FileNotFoundError(f"Missing frames in {folder}")

        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)

        if self.as_tensor:
            im1 = torch.from_numpy(im1).permute(2, 0, 1).float() / 255.0
            im2 = torch.from_numpy(im2).permute(2, 0, 1).float() / 255.0
            im3 = torch.from_numpy(im3).permute(2, 0, 1).float() / 255.0

        sample = {"before": im1, "ground_truth": im2, "after": im3, "name": folder.name}

        if self.transform:
            sample = self.transform(sample)

        return sample
