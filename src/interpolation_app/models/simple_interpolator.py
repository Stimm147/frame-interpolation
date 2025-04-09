import torch
from torch import nn
import pytorch_lightning as pl
from interpolation_app.losses.interpolation_loss import get_loss_fn


class SimpleCNNInterpolator(pl.LightningModule):
    def __init__(self, loss_type: str = "l1", lr: float = 1e-3, depth: str = "deep"):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        if depth == "deep":
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
        elif depth == "shallow":
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError("depth must be either 'shallow' or 'deep'")

        self.loss_fn = get_loss_fn(loss_type)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = torch.cat([batch["before"], batch["after"]], dim=1)
        y = batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch["before"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat([batch["before"], batch["after"]], dim=1)
        y = batch["ground_truth"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch["before"].size(0),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            threshold=1e-3,
            threshold_mode="rel",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
