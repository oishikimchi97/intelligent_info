from typing import List, Literal, Optional
import numpy as np
import open_earth_map
import pytorch_lightning as pl
import torch
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp

index2label = {v: k for k, v in open_earth_map.utils.class_grey_oem.items()}


class OpenEarthMapLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: Literal["Unet", "UnetFormer", "UnetPlusPlus"] = "UnetPlusPlus",
        n_classes=9,
        lr=1e-4,
        weight_decay=1e-6,
        pretrained=False,
        except_background=True,
        warmup_trainig: bool = True,
        # In default, 'bareland', 'grass', 'cropland' are set to warmup classes
        warmup_class: Optional[List[int]] = [1, 2, 7],
        warmup_epoch: int = 30,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay

        # TODO: Change the model to Unet with EfficientNet backbone using mmseg
        if model_name == "Unet":
            self.model = smp.Unet(
                encoder_name="efficientnet-b4",
                encoder_weights=None,
                in_channels=3,
                classes=n_classes,
            )
        elif model_name == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights=None,
                in_channels=3,
                classes=n_classes,
            )
        elif model_name == "UnetFormer":
            self.model = open_earth_map.networks.UNetFormer(
                in_channels=3,
                n_classes=n_classes,
                backbone_name="efficientnet_b4",
                pretrained=pretrained,
            )
        self.warmup_trainig = warmup_trainig
        self.warmup_class = warmup_class
        self.warmup_epoch = warmup_epoch

        if self.warmup_trainig and self.warmup_class:
            print("Warmup training enabled")
            self.warmup_ce_weight = np.zeros(n_classes)
            self.warmup_ce_weight[self.warmup_class] = 1
            warmup_class_labels = [index2label[i] for i in self.warmup_class]
            print(f"Using warmup classes: {warmup_class_labels}")

        else:
            self.warmup_ce_weight = None

        self.crossEntropyLoss = open_earth_map.losses.CEWithLogitsLoss(
            weight=self.warmup_ce_weight
        )
        self.focalLoss = open_earth_map.losses.FocalLoss()
        self.jacardloss = open_earth_map.losses.JaccardLoss()

        # Initialize IoU metrics
        self.train_iou = JaccardIndex(
            task="multiclass",
            num_classes=n_classes,
            average="none",
            threshold=0.0,
        )
        self.val_iou = JaccardIndex(
            task="multiclass",
            num_classes=n_classes,
            average="none",
            threshold=0.0,
        )

        if except_background:
            self.iou_start_idx = 1
        else:
            self.iou_start_idx = 0

    def cal_loss(self, logits, masks):
        ce_loss = self.crossEntropyLoss(logits, masks)
        # focal_loss = self.focalLoss(logits, masks)
        # return ce_loss + focal_loss
        # return self.jacardloss(logits, masks)
        return ce_loss

    # TODO: Use log_dict to log all metrics
    def training_step(self, batch):
        images, masks, _ = batch
        logits = self.model(images)
        loss = self.cal_loss(logits, masks.type(logits.dtype))
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            targets = torch.argmax(masks, dim=1)
            iou_per_class = self.train_iou(preds, targets)
            if self.warmup_class:
                warmup_class_mean_iou = torch.mean(iou_per_class[self.warmup_class])
                self.log(
                    "warmup_class_train_mean_iou",
                    warmup_class_mean_iou,
                    on_step=True,
                    sync_dist=True,
                    prog_bar=True,
                )

            mean_iou = torch.mean(iou_per_class[self.iou_start_idx :])

        for i, iou in enumerate(iou_per_class):
            class_name = index2label.get(i, f"Unknown_{i}")
            self.log(f"train_iou-{class_name}", iou, on_step=True, sync_dist=True)

        self.log(
            "train_mean_iou", mean_iou, on_step=True, prog_bar=True, sync_dist=True
        )

        return loss

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.warmup_trainig and self.current_epoch >= self.warmup_epoch:
            self.warmup_trainig = False
            self.crossEntropyLoss = open_earth_map.losses.CEWithLogitsLoss()
            print(f"Training without warmup from epoch {self.current_epoch}")

    def validation_step(self, batch):
        images, masks, _ = batch
        logits = self.model(images)
        loss = self.cal_loss(logits, masks.type(logits.dtype))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(masks, dim=1)
        iou_per_class = self.val_iou(preds, targets)

        if self.warmup_class:
            warmup_class_mean_iou = torch.mean(iou_per_class[self.warmup_class])
            self.log(
                "warmup_class_val_mean_iou",
                warmup_class_mean_iou,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )

        mean_iou = torch.mean(iou_per_class[self.iou_start_idx :])

        for i, iou in enumerate(iou_per_class):
            class_name = index2label.get(i, f"Unknown_{i}")
            self.log(f"val_iou-{class_name}", iou, on_epoch=True)
            if self.warmup_trainig and i in self.warmup_class:
                self.log(
                    f"train_iou-{class_name}_warmup", iou, on_step=True, sync_dist=True
                )

        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
