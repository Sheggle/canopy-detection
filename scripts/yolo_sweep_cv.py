#!/usr/bin/env python3
"""
YOLO cross-validation sweep script with wandb integration.

Performs k-fold cross-validation training and combines predictions from all folds
into a single submission.json file.
"""

import wandb
import json
import subprocess
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def create_cv_dataset_if_needed(n_folds: int) -> Path:
    """
    Create cross-validation dataset if it doesn't exist.

    Returns:
        Path to the cv_<n_folds> directory
    """
    cv_dir = Path(f"cv_{n_folds}")

    if not cv_dir.exists():
        print(f"Creating {n_folds}-fold cross-validation dataset...")
        result = subprocess.run([
            "uv", "run", "scripts/to_yolo_cv.py", str(n_folds)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating CV dataset: {result.stderr}")
            raise RuntimeError(f"Failed to create CV dataset")

        print(f"✅ Created {n_folds}-fold CV dataset")
    else:
        print(f"✅ Using existing {n_folds}-fold CV dataset")

    return cv_dir


def train_fold(fold_idx: int, cv_dir: Path, config) -> tuple:
    """
    Train a model on a specific fold.

    Returns:
        Tuple of (model_path, metrics)
    """
    fold_dir = cv_dir / f"fold_{fold_idx}"
    data_yaml = fold_dir / "data.yaml"

    print(f"\n🔄 Training fold {fold_idx}...")

    # Load YOLO model from config
    model = YOLO(config.model)

    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        device='cuda',
        optimizer='AdamW',
        project=f'canopy_cv_sweep_fold_{fold_idx}',
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
        degrees=90.0,
        flipud=0.5,
        fliplr=0.5,

        # Training settings
        patience=10,  # Reduced for CV
        save_period=-1,
        val=True,
        verbose=False
    )

    # Get best model path
    save_dir = results.save_dir
    best_model_path = save_dir / 'weights' / 'best.pt'

    # Validate the model
    best_model = YOLO(best_model_path)
    val_results = best_model.val(
        data=str(data_yaml),
        imgsz=config.imgsz,
        device='cuda',
        split='val',
        iou=0.7,
        conf=0.001,
    )

    metrics = val_results.results_dict
    print(f"✅ Fold {fold_idx} completed - mAP50(M): {metrics.get('metrics/mAP50(M)', 0):.4f}")

    return best_model_path, metrics


def run_inference_on_fold_val(model_path: Path, fold_idx: int, cv_dir: Path, all_images: list) -> dict:
    """
    Run inference on the validation set of a specific fold.

    Returns:
        Dictionary mapping image filenames to prediction lists
    """
    # Load trained model
    model = YOLO(str(model_path))

    # Class mapping
    class_names = {0: "individual_tree", 1: "group_of_trees"}

    # Get validation images for this fold
    val_images_dir = cv_dir / f"fold_{fold_idx}" / "val" / "images"
    val_image_files = list(val_images_dir.glob("*.tif"))

    predictions = {}

    for image_path in val_image_files:
        file_name = image_path.name

        # Find the corresponding image metadata from train_annotations.json
        img_info = None
        for img in all_images:
            if img["file_name"] == file_name:
                img_info = img
                break

        if img_info is None:
            continue

        # Run inference
        results = model(str(image_path), conf=0.01, verbose=False)

        annotations = []
        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i in range(len(result.boxes)):
                    confidence = float(result.boxes.conf[i])
                    class_id = int(result.boxes.cls[i])

                    if hasattr(result.masks, 'xy') and i < len(result.masks.xy):
                        coords = result.masks.xy[i].flatten().tolist()
                        segmentation = [int(round(c)) for c in coords]

                        annotations.append({
                            "class": class_names.get(class_id, f"class_{class_id}"),
                            "confidence_score": round(confidence, 2),
                            "segmentation": segmentation
                        })

        predictions[file_name] = {
            "annotations": annotations,
            "metadata": {
                "width": img_info["width"],
                "height": img_info["height"],
                "cm_resolution": img_info["cm_resolution"],
                "scene_type": img_info["scene_type"]
            }
        }

    return predictions


def combine_fold_predictions(fold_predictions: list, all_images: list) -> dict:
    """
    Combine predictions from all folds into a complete submission covering all training images.

    Since each image appears in exactly one fold's validation set, we combine all fold predictions
    to get predictions for the entire training dataset.
    """
    # Create submission structure
    combined_submission = {"images": []}

    # Collect all predictions across folds
    all_predictions = {}
    for fold_pred in fold_predictions:
        all_predictions.update(fold_pred)

    # Create entries for all images that had predictions
    for file_name, pred_data in all_predictions.items():
        image_entry = {
            "file_name": file_name,
            "width": pred_data["metadata"]["width"],
            "height": pred_data["metadata"]["height"],
            "cm_resolution": pred_data["metadata"]["cm_resolution"],
            "scene_type": pred_data["metadata"]["scene_type"],
            "annotations": pred_data["annotations"]
        }
        combined_submission["images"].append(image_entry)

    print(f"Combined predictions for {len(combined_submission['images'])} images")
    return combined_submission


def train_with_wandb_cv():
    """
    Main function to perform cross-validation training with wandb.
    """
    # Initialize wandb run
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config
    n_folds = config.n_folds

    print(f"🚀 Starting {n_folds}-fold cross-validation sweep...")

    # Load original training data
    with open("data/train_annotations.json") as f:
        train_data = json.load(f)
    all_images = train_data["images"]

    # Create CV dataset if needed
    cv_dir = create_cv_dataset_if_needed(n_folds)

    # Store metrics from all folds
    all_fold_metrics = []
    fold_predictions = []

    # Train on each fold
    for fold_idx in range(n_folds):
        try:
            model_path, metrics = train_fold(fold_idx, cv_dir, config)
            all_fold_metrics.append(metrics)

            # Run inference on this fold's validation set
            print(f"🔍 Running inference on fold {fold_idx} validation set...")
            predictions = run_inference_on_fold_val(model_path, fold_idx, cv_dir, all_images)
            fold_predictions.append(predictions)

        except Exception as e:
            print(f"❌ Error in fold {fold_idx}: {e}")
            # Continue with other folds
            continue

    if not all_fold_metrics:
        print("❌ All folds failed!")
        wandb.finish()
        return

    # Combine predictions from all folds
    print("🔄 Combining predictions from all folds...")
    combined_submission = combine_fold_predictions(fold_predictions, all_images)

    # Save submission
    submission_path = f"submission_cv_{wandb.run.id}.json"
    with open(submission_path, "w") as f:
        json.dump(combined_submission, f, indent=2)

    print(f"✅ Cross-validation completed!")
    print(f"📁 Combined submission saved: {submission_path}")

    # Log the main metric for optimization
    wandb.log({"metrics/mAP50(M)": avg_metrics.get('cv_avg_metrics/mAP50(M)', 0)})

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    train_with_wandb_cv()