"""
Attention U-Net Training Engine for Retinal Segmentation
Supports:
  - Optic Disc & Cup Dual Segmentation
  - Full-Field Blood Vessel Segmentation
Features:
  - ComboLoss (Focal Loss + Soft Dice Loss)
  - Cosine Annealing Learning Rate Schedule
  - Validation Checkpointing & Early Stopping
"""

import os
import argparse
import yaml
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.unet.attention_unet import AttentionUNet
from models.unet.unet_standard import StandardUNet
from models.unet.losses import ComboLoss, DiceLoss, BinaryFocalLoss
from models.unet.metrics import calculate_segmentation_metrics, dice_coefficient
from data.preprocessor import RetinalPreprocessor


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for retinal segmentation.
    Loads RGB/BGR fundus images and corresponding binary or multi-channel masks.
    """
    def __init__(self, image_dir: str, mask_dir: str, target_size=(512, 512), is_dual_channel=True):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png")) + list(Path(mask_dir).glob("*.jpg")))
        self.target_size = target_size
        self.is_dual_channel = is_dual_channel
        self.preprocessor = RetinalPreprocessor(target_size=target_size)

        if len(self.image_paths) == 0:
            print(f"[Warning] No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image
        img_p = str(self.image_paths[idx])
        img = cv2.imread(img_p)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_p}")

        # Preprocess: Green channel + CLAHE + Illumination correction
        tensor_img, _ = self.preprocessor.preprocess_for_unet(img, return_tensor=True)
        # Squeeze batch dim so it's (C, H, W)
        tensor_img = tensor_img.squeeze(0)

        # Read mask
        if idx < len(self.mask_paths):
            mask_p = str(self.mask_paths[idx])
            mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.target_size, dtype=np.uint8)

        # Resize mask to match target size
        if mask.shape[:2] != self.target_size:
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Format mask tensor
        if self.is_dual_channel and len(mask.shape) == 3 and mask.shape[2] >= 2:
            # Channel 0: Optic Disc, Channel 1: Optic Cup
            mask_binary = (mask[:, :, :2] > 127).astype(np.float32)
            tensor_mask = torch.from_numpy(mask_binary).permute(2, 0, 1) # (2, H, W)
        else:
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_binary = (mask > 127).astype(np.float32)
            tensor_mask = torch.from_numpy(mask_binary).unsqueeze(0) # (1, H, W)

        return tensor_img, tensor_mask


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    count = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        dice = dice_coefficient(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        count += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    return total_loss / max(1, count), total_dice / max(1, count)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_coefficient(logits, masks)

            total_loss += loss.item()
            total_dice += dice
            count += 1

    return total_loss / max(1, count), total_dice / max(1, count)


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on Retinal Imaging")
    parser.add_argument("--config", type=str, default="configs/unet_config.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--task", type=str, choices=["od_oc", "vessels"], default="od_oc",
                        help="Segmentation task (od_oc or vessels)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/unet", help="Directory to save model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training Setup] Device: {device} | Task: {args.task}")

    # Determine paths based on task
    base_data = Path("data/processed")
    is_dual = (args.task == "od_oc")
    out_channels = 2 if is_dual else 1

    train_img_dir = base_data / "train" / "images"
    train_mask_dir = base_data / "train" / ("masks" if is_dual else "vessels")
    val_img_dir = base_data / "val" / "images"
    val_mask_dir = base_data / "val" / ("masks" if is_dual else "vessels")

    # If data does not exist yet, generate synthetic dataset
    if not train_img_dir.exists() or len(list(train_img_dir.glob("*.png"))) == 0:
        print("[Training Setup] Dataset directory not populated. Generating synthetic benchmark dataset...")
        from data.synthetic_generator import SyntheticRetinalGenerator
        gen = SyntheticRetinalGenerator(img_size=(512, 512))
        gen.generate_dataset(output_dir=str(base_data), num_train=20, num_val=5, num_test=5)

    # Dataloaders
    train_dataset = RetinalDataset(str(train_img_dir), str(train_mask_dir), is_dual_channel=is_dual)
    val_dataset = RetinalDataset(str(val_img_dir), str(val_mask_dir), is_dual_channel=is_dual)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[Dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, Optimizer, Criterion
    model = AttentionUNet(in_channels=1, out_channels=out_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = ComboLoss(focal_weight=0.5, dice_weight=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_dice = 0.0
    best_model_path = os.path.join(args.save_dir, f"best_attention_unet_{args.task}.pth")

    print("\n" + "=" * 60)
    print(" STARTING ATTENTION U-NET TRAINING")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} - Dice: {train_dice:.4f} "
              f"| Val Loss: {val_loss:.4f} - Dice: {val_dice:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Dice: {val_dice:.4f})")

    print("\n" + "=" * 60)
    print(f"[Training Complete] Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Saved weights: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
