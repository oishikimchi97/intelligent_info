from calendar import EPOCH
from pathlib import Path
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_module import OpenEarthMapDataModule
from pl_module import OpenEarthMapLightningModule

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    IMG_SIZE = 512
    N_CLASSES = 9
    BATCH_SIZE = 8
    EPOCH = 100
    NUM_DEVICE = 1

    data_module = OpenEarthMapDataModule(
        img_size=IMG_SIZE, n_classes=N_CLASSES, batch_size=BATCH_SIZE
    )

    # Print dataset information
    data_module.setup()
    print("Total samples      :", len(data_module.train_fns) + len(data_module.val_fns))
    print("Training samples   :", len(data_module.train_fns))
    print("Validation samples :", len(data_module.val_fns))

    model_name = "UnetPlusPlus"

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="open_earth_map_intelligence_assignment",
        name=f"{model_name}-EfficientNet_B4-epoch{EPOCH}-ce_loss-w/o_aug-w_warmup",
        offline=False,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCH,
        logger=wandb_logger,
        devices=NUM_DEVICE if torch.cuda.is_available() else None,
        strategy="ddp" if NUM_DEVICE > 1 else "auto",
    )

    pl_module = OpenEarthMapLightningModule(
        model_name=model_name, n_classes=N_CLASSES, lr=1e-4, warmup_epoch=10
    )

    if trainer.global_rank == 0:

        run_dir = Path(wandb_logger.experiment.dir)
        checkpoint_dir = run_dir / "checkpoints"

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="checkpoint-{step}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )

        trainer.callbacks.append(checkpoint_callback)

    trainer.fit(pl_module, data_module)

    if trainer.global_rank == 0:
        model_save_path = str(checkpoint_dir / "final_model.pth")
        trainer.save_checkpoint(model_save_path, weights_only=True)

        print("Training completed!")
        print(f"Model saved at {model_save_path}")
