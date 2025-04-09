import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from interpolation_app.models.simple_interpolator import SimpleCNNInterpolator
from interpolation_app.datasets.triplet_module import TripletDataModule


def train():
    model = SimpleCNNInterpolator(loss_type="ssim", lr=1e-4, depth="shallow")
    data = TripletDataModule()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="simple-cnn-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    train()
