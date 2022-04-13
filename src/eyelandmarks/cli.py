from pytorch_lightning import callbacks
from pytorch_lightning.utilities.cli import LightningCLI

from eyelandmarks.models import *
from eyelandmarks.modules import *
from eyelandmarks.util.callbacks import *


def main():
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="val/mse_epoch", dirpath="checkpoints/", save_top_k=3, mode="min")

    LightningCLI(
        LandmarkNet,
        EyeDataModule,
        trainer_defaults=dict(
            callbacks=[AllLandmarksCallback(), checkpoint_callback]
        ),
        save_config_callback=None,
    )

if __name__ == '__main__':
    main()