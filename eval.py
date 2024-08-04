import re
import wandb
import torch
from tqdm import tqdm
from open_earth_map import utils

from data_module import OpenEarthMapDataModule
from pl_module import OpenEarthMapLightningModule

index2label = {v: k for k, v in utils.class_grey_oem.items()}


def eval(model, data_module, device, limit=1000):
    model.eval()
    data_loader = data_module.val_dataloader()

    table = wandb.Table(
        columns=[
            "idx",
            "input_image",
            "pred_color_map",
            "gt_color_map",
            "segmentation_result",
        ]
    )

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(
            tqdm(data_loader, desc="Evaluation")
        ):
            if batch_idx * images.shape[0] >= limit:
                break

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Convert tensors to numpy arrays
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy().argmax(axis=1)
            preds_np = preds.cpu().numpy()

            for local_idx in range(images.shape[0]):
                global_idx = batch_idx * images.shape[0] + local_idx

                if global_idx >= limit:
                    break

                # Create color maps for the predicted and ground truth masks
                image = images_np[local_idx]
                pred = preds_np[local_idx]
                mask = masks_np[local_idx]

                pred_colormap = utils.make_rgb(pred)
                mask_colormap = utils.make_rgb(mask)

                # Create wandb Image with masks
                save_img_arr = (image * 255.0).astype("uint8").transpose(1, 2, 0)

                wandb_image = wandb.Image(
                    save_img_arr,
                    masks={
                        "predictions": {
                            "mask_data": pred,
                            "class_labels": index2label,
                        },
                        "ground_truth": {
                            "mask_data": mask,
                            "class_labels": index2label,
                        },
                    },
                )

                # Add row to the table
                table.add_data(
                    f"{global_idx}",
                    wandb.Image(save_img_arr),
                    wandb.Image(pred_colormap),
                    wandb.Image(mask_colormap),
                    wandb_image,
                )

    # Log the table
    wandb.log({"validation_results": table})


if __name__ == "__main__":
    # Load the model
    checkpoint_path = (
        "wandb/run-20240802_082015-ss1rdvg4/files/checkpoints/final_model.pth"
    )
    match = re.search(r"run-[^/]+", checkpoint_path)
    run_name = match.group(0)
    model = OpenEarthMapLightningModule.load_from_checkpoint(
        checkpoint_path, weights_only=True
    ).model.cuda()

    IMG_SIZE = 512
    N_CLASSES = 9
    BATCH_SIZE = 16

    # Load the data module
    data_module = OpenEarthMapDataModule(
        img_size=IMG_SIZE, n_classes=N_CLASSES, batch_size=BATCH_SIZE
    )

    # Load the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="open_earth_map_intelligence_assignment_eval",
        name=run_name,
    )

    data_module.setup()

    # Evaluate the model
    eval(model, data_module, device, limit=1000)
