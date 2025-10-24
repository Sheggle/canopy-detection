#!/usr/bin/env python3
"""
YOLO sweep script with wandb integration for hyperparameter optimization.
"""

import wandb
import json
from ultralytics import YOLO

def train_with_wandb():
    # Initialize wandb run
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Load YOLO model from config
    model = YOLO(config.model)

    # Train the model with hyperparameters from wandb
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        device='cuda',
        optimizer='AdamW',
        project='canopy_detection_sweep',
        name=f'sweep_{wandb.run.id}',

        # Hyperparameters from wandb config
        batch=config.batch,
        lr0=config.lr0,
        momentum=config.momentum,
        weight_decay=config.weight_decay,

        # Augmentation parameters
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        scale=config.scale,
        translate=config.translate,
        mosaic=config.mosaic,

        # Fixed parameters optimized for aerial imagery
        degrees=90.0,     # No rotation for aerial imagery
        flipud=0.5,      # No vertical flip for aerial imagery
        fliplr=0.5,      # Horizontal flip is OK
    )

    save_dir = results.save_dir
    best = save_dir / 'weights' / 'best.pt'
    model = YOLO(best)
    results = model.val(
        data="yolo_dataset/data.yaml",
        imgsz=640,
        device='cuda',
        split='val',
        iou=0.7,
        conf=0.001,
    )
    metrics = results.results_dict

    to_log = {
        "metrics/mAP50(M)": metrics.get("metrics/mAP50(M)", 0),
        "metrics/mAP50-95(M)": metrics.get("metrics/mAP50-95(M)", 0),
        "metrics/precision(M)": metrics.get("metrics/precision(M)", 0),
        "metrics/recall(M)": metrics.get("metrics/recall(M)", 0),
        "val/box_loss": metrics.get("val/box_loss", 0),
        "val/seg_loss": metrics.get("val/seg_loss", 0),
        "val/cls_loss": metrics.get("val/cls_loss", 0)
    }

    wandb.log(to_log)
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(to_log, f)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train_with_wandb()