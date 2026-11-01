"""
Training and validation pipeline for brain tumor segmentation.
Config-driven, mixed precision (CUDA), early stopping, checkpointing, and TensorBoard logging.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import BraTSSliceDataset, create_splits, preprocess_dataset
from src.losses import ComboLoss, DiceFocalLoss
from src.metrics import compute_region_dice
from src.model import build_model
from src.utils import get_device, load_config, set_seed, setup_logger


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool = False,
) -> float:
    """
    Train for a single epoch.
    """
    model.train()
    running_loss = 0.0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate validation loss and WT, TC, ET region Dice scores.
    """
    model.eval()
    running_loss = 0.0
    dice_agg = {"WT": [], "TC": [], "ET": []}

    for x, y in tqdm(dataloader, desc="Validating", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()

        preds = logits.argmax(dim=1).cpu()
        targets = y.cpu()

        # Batch-level or slice-level metrics
        for i in range(preds.shape[0]):
            scores = compute_region_dice(preds[i], targets[i])
            for k in ["WT", "TC", "ET"]:
                dice_agg[k].append(scores[k])

    val_loss = running_loss / max(1, len(dataloader))
    metrics = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in dice_agg.items()}
    metrics["mean_dice"] = float(np.mean([metrics["WT"], metrics["TC"], metrics["ET"]]))
    metrics["val_loss"] = val_loss
    return metrics


def run_training(config: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute full training pipeline governed by config dictionary.
    """
    # 1. Setup seed and environment
    seed = config.get("seed", 42)
    set_seed(seed)
    logger = setup_logger("brats_train")
    device = get_device()
    use_amp = bool(config["training"].get("mixed_precision", True) and device.type == "cuda")

    logger.info(f"Target Device: {device} | Mixed Precision (AMP): {use_amp}")

    # 2. Prepare Data and Splits
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    raw_dir = Path(data_cfg["raw_dir"])
    splits_file = Path(data_cfg.get("splits_file", "data/splits.json"))

    # Preprocess if processed folder does not exist or is empty
    if not processed_dir.exists() or len(list(processed_dir.glob("*/img.npy"))) == 0:
        logger.info(f"Preprocessed data not found. Checking raw data in {raw_dir}...")
        if not raw_dir.exists() or len(list(raw_dir.glob("*"))) == 0:
            raise FileNotFoundError(
                f"Raw data directory {raw_dir} is missing or empty. Run scripts/download_brats.py or scripts/generate_mock_brats.py first!"
            )
        preprocess_dataset(raw_dir, processed_dir, modalities=data_cfg.get("modalities", ["flair", "t1", "t1ce", "t2"]))

    # Load or generate patient-level splits
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        logger.info(f"Loaded existing splits from {splits_file}")
    else:
        splits = create_splits(
            processed_dir,
            train_ratio=data_cfg.get("train_ratio", 0.70),
            val_ratio=data_cfg.get("val_ratio", 0.15),
            test_ratio=data_cfg.get("test_ratio", 0.15),
            seed=seed,
            splits_file=splits_file,
        )

    train_cases = splits["train"]
    val_cases = splits["val"]
    logger.info(f"Dataset split: {len(train_cases)} Train | {len(val_cases)} Val | {len(splits['test'])} Test cases")

    # 3. Datasets and DataLoaders
    aug_cfg = config.get("augmentation", {})
    mode = "2.5d" if config["model"].get("name") == "unet_25d" else "2d"

    train_ds = BraTSSliceDataset(
        processed_dir,
        train_cases,
        mode=mode,
        augment=aug_cfg.get("enabled", True),
        nnunet_aug=aug_cfg.get("nnunet_style", False),
        empty_keep=data_cfg.get("empty_keep", 0.2),
        seed=seed,
    )
    val_ds = BraTSSliceDataset(
        processed_dir,
        val_cases,
        mode=mode,
        augment=False,
        nnunet_aug=False,
        empty_keep=1.0,  # evaluate on all valid slices in validation
        seed=seed,
    )

    t_cfg = config["training"]
    num_workers = t_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(f"Sampled slices: {len(train_ds)} Train slices | {len(val_ds)} Val slices")

    # 4. Model Architecture
    m_cfg = config["model"]
    model = build_model(
        name=m_cfg.get("name", "unet"),
        in_channels=m_cfg.get("in_channels", 4),
        num_classes=m_cfg.get("num_classes", 4),
        base_filters=m_cfg.get("base_filters", 16),
    ).to(device)

    # 5. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg.get("lr", 3e-4)),
        weight_decay=float(t_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(t_cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    l_cfg = config.get("loss", {})
    loss_type = l_cfg.get("type", "combo").lower()
    if loss_type == "dice_focal":
        criterion = DiceFocalLoss(
            focal_weight=float(l_cfg.get("focal_weight", 1.0)),
            dice_weight=float(l_cfg.get("dice_weight", 1.0)),
            gamma=float(l_cfg.get("focal_gamma", 2.0)),
            num_classes=m_cfg.get("num_classes", 4),
        )
    else:
        criterion = ComboLoss(
            ce_weight=float(l_cfg.get("ce_weight", 1.0)),
            dice_weight=float(l_cfg.get("dice_weight", 1.0)),
            num_classes=m_cfg.get("num_classes", 4),
        )

    scaler = torch.amp.GradScaler(enabled=use_amp)

    # 6. Logging and Checkpoint Directories
    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = t_cfg.get("checkpoint_name", "best_model.pt")
    best_ckpt_path = checkpoint_dir / ckpt_name

    log_dir = t_cfg.get("log_dir", "runs/experiment")
    writer = SummaryWriter(log_dir=log_dir)

    # 7. Training Loop with Early Stopping
    best_mean_dice = -1.0
    patience = int(t_cfg.get("early_stopping_patience", 5))
    epochs_no_improve = 0
    history = []

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp=use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        mean_dice = val_metrics["mean_dice"]
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Dice WT: {val_metrics['WT']:.4f} TC: {val_metrics['TC']:.4f} ET: {val_metrics['ET']:.4f} | "
            f"Mean Dice: {mean_dice:.4f}"
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("Dice/WT", val_metrics["WT"], epoch)
        writer.add_scalar("Dice/TC", val_metrics["TC"], epoch)
        writer.add_scalar("Dice/ET", val_metrics["ET"], epoch)
        writer.add_scalar("Dice/Mean", mean_dice, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "WT": val_metrics["WT"],
            "TC": val_metrics["TC"],
            "ET": val_metrics["ET"],
            "mean_dice": mean_dice,
        }
        history.append(epoch_record)

        # Checkpoint if best
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_dice": mean_dice,
                    "metrics": val_metrics,
                    "config": config,
                },
                best_ckpt_path,
            )
            logger.info(f"--> Saved new best checkpoint ({best_mean_dice:.4f}) to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs without improvement.")
                break

    writer.close()
    summary = {
        "best_mean_dice": best_mean_dice,
        "checkpoint_path": str(best_ckpt_path),
        "history": history,
    }

    # Save summary json next to checkpoint
    summary_path = best_ckpt_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS Model Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    run_training(config, config_path=args.config)


if __name__ == "__main__":
    main()
