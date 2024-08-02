from typing import Literal, Optional
import open_earth_map
import pytorch_lightning as pl
import torch
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp

index2label = {v: k for k, v in open_earth_map.utils.class_grey_oem.items()}


class OpenEarthMapLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: Literal["Unet", "UnetFormer"] = "Unet",
        n_classes=9,
        lr=1e-4,
        weight_decay=1e-6,
        pretrained=False,
        except_background=True,
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
        elif model_name == "UnetFormer":
            self.model = open_earth_map.networks.UNetFormer(
                in_channels=3,
                n_classes=n_classes,
                backbone_name="efficientnet_b4",
                pretrained=pretrained,
            )
        self.crossEntropyLoss = open_earth_map.losses.CEWithLogitsLoss()
        self.focalLoss = open_earth_map.losses.FocalLoss()

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
        focal_loss = self.focalLoss(logits, masks)
        return ce_loss + focal_loss

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
            mean_iou = torch.mean(iou_per_class[self.iou_start_idx :])

        for i, iou in enumerate(iou_per_class):
            class_name = index2label.get(i, f"Unknown_{i}")
            self.log(f"train_iou-{class_name}", iou, on_step=True)

        self.log(
            "train_mean_iou", mean_iou, on_step=True, prog_bar=True, sync_dist=True
        )

        return loss

    def validation_step(self, batch):
        images, masks, _ = batch
        logits = self.model(images)
        loss = self.cal_loss(logits, masks.type(logits.dtype))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(masks, dim=1)
        iou_per_class = self.val_iou(preds, targets)
        mean_iou = torch.mean(iou_per_class[self.iou_start_idx :])

        for i, iou in enumerate(iou_per_class):
            class_name = index2label.get(i, f"Unknown_{i}")
            self.log(f"val_iou-{class_name}", iou, on_epoch=True)

        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
