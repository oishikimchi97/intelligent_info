import os
import numpy as np
import torch
import open_earth_map
import torchvision
from pathlib import Path
import pytorch_lightning as pl


class OpenEarthMapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/home/kim/datasets/OpenEarthMap_wo_xBD",
        train_list=None,
        val_list=None,
        img_size=512,
        n_classes=9,
        batch_size=4,
        num_workers=19
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_list = train_list or os.path.join(data_dir, "train.txt")
        self.val_list = val_list or os.path.join(data_dir, "val.txt")
        self.img_size = img_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        fns = [f for f in Path(self.data_dir).rglob("*.tif") if "/images/" in str(f)]
        self.train_fns = [str(f) for f in fns if f.name in np.loadtxt(self.train_list, dtype=str)]
        self.val_fns = [str(f) for f in fns if f.name in np.loadtxt(self.val_list, dtype=str)]

        train_augm = torchvision.transforms.Compose(
            [
                open_earth_map.transforms.Rotate(),
                open_earth_map.transforms.Crop(self.img_size),
            ],
        )

        val_augm = torchvision.transforms.Compose(
            [
                open_earth_map.transforms.Resize(self.img_size),
            ],
        )

        self.train_data = open_earth_map.dataset.OpenEarthMapDataset(
            self.train_fns,
            n_classes=self.n_classes,
            augm=train_augm,
        )

        self.val_data = open_earth_map.dataset.OpenEarthMapDataset(
            self.val_fns,
            n_classes=self.n_classes,
            augm=val_augm,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )