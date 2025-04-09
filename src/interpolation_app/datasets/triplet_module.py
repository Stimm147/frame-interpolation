import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from interpolation_app.utils.triplet_dataset import TripletDataset
from pathlib import Path


class TripletDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/triplet_dataset",
        batch_size: int = 5,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = TripletDataset(root_dir=self.data_dir, as_tensor=True)
        n = len(dataset)
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset, [int(0.9 * n), n - int(0.9 * n)]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
