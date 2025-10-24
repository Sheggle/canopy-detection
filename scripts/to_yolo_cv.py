#!/usr/bin/env python3
"""
Create cross-validation YOLO datasets with k-fold splitting.

Usage: uv run scripts/to_yolo_cv.py <n_folds>

Creates cv_<n_folds>/ directory with fold_<n>/ subdirectories.
Each fold uses leave-one-out strategy: 1 fold for validation, (k-1) folds for training.
"""

import json
import shutil
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math


def load_annotations(json_path: Path) -> Dict[str, Any]:
    """Load annotations from JSON file."""
    print(f"Loading annotations from {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)


def polygon_to_yolo_format(segmentation: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert polygon coordinates to YOLO normalized format.
    """
    normalized_coords = []

    # Process coordinates in pairs (x, y)
    for i in range(0, len(segmentation), 2):
        if i + 1 < len(segmentation):
            x = segmentation[i]
            y = segmentation[i + 1]

            # Normalize to 0-1 range
            x_norm = x / img_width
            y_norm = y / img_height

            # Clamp to valid range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            normalized_coords.extend([x_norm, y_norm])

    return normalized_coords


def create_yolo_annotation(annotations: List[Dict], img_width: int, img_height: int, class_mapping: Dict[str, int]) -> str:
    """
    Create YOLO annotation string for an image.
    """
    yolo_lines = []

    for ann in annotations:
        class_name = ann['class']
        class_id = class_mapping[class_name]
        segmentation = ann['segmentation']

        # Convert polygon to YOLO format
        normalized_coords = polygon_to_yolo_format(segmentation, img_width, img_height)

        # Skip invalid polygons (need at least 3 points = 6 coordinates)
        if len(normalized_coords) < 6:
            continue

        # Format: class_id x1 y1 x2 y2 ... xn yn
        coords_str = ' '.join(f'{coord:.6f}' for coord in normalized_coords)
        yolo_line = f"{class_id} {coords_str}"
        yolo_lines.append(yolo_line)

    return '\n'.join(yolo_lines)


def create_k_folds(images: List[Dict], n_folds: int, seed: int = 42) -> List[List[Dict]]:
    """
    Split images into k folds using random assignment.
    """
    print(f"Creating {n_folds} folds from {len(images)} images")

    # Shuffle images with fixed seed for reproducibility
    images_copy = images.copy()
    random.seed(seed)
    random.shuffle(images_copy)

    # Calculate fold sizes
    fold_size = len(images_copy) // n_folds
    remainder = len(images_copy) % n_folds

    folds = []
    start_idx = 0

    for i in range(n_folds):
        # Distribute remainder images across first few folds
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_fold_size

        fold = images_copy[start_idx:end_idx]
        folds.append(fold)
        print(f"  Fold {i}: {len(fold)} images")

        start_idx = end_idx

    return folds


def create_fold_dataset(
    fold_idx: int,
    folds: List[List[Dict]],
    output_dir: Path,
    data_dir: Path,
    class_mapping: Dict[str, int]
) -> None:
    """
    Create dataset for a specific fold (leave-one-out validation).
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    print(f"\nCreating fold {fold_idx} dataset at {fold_dir}")

    # Create directory structure
    train_images_dir = fold_dir / "train" / "images"
    train_labels_dir = fold_dir / "train" / "labels"
    val_images_dir = fold_dir / "val" / "images"
    val_labels_dir = fold_dir / "val" / "labels"

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Validation set: current fold
    val_images = folds[fold_idx]

    # Training set: all other folds
    train_images = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train_images.extend(fold)

    print(f"  Train: {len(train_images)} images, Val: {len(val_images)} images")

    # Process validation images
    process_image_set(val_images, val_images_dir, val_labels_dir, data_dir, class_mapping, "validation")

    # Process training images
    process_image_set(train_images, train_images_dir, train_labels_dir, data_dir, class_mapping, "training")

    # Create data.yaml for this fold
    create_data_yaml(fold_dir, len(class_mapping), class_mapping)

    # Create classes.txt
    create_classes_txt(fold_dir, class_mapping)


def process_image_set(
    images: List[Dict],
    images_dir: Path,
    labels_dir: Path,
    data_dir: Path,
    class_mapping: Dict[str, int],
    set_name: str
) -> None:
    """
    Process a set of images (train or val) and copy to appropriate directories.
    """
    train_images_dir = data_dir / "train_images"

    for i, img_data in enumerate(images):
        if (i + 1) % 20 == 0:
            print(f"    Processing {set_name} image {i + 1}/{len(images)}")

        # Get image info
        filename = img_data['file_name']
        img_width = img_data['width']
        img_height = img_data['height']
        annotations = img_data['annotations']

        # Copy image file
        src_img_path = train_images_dir / filename
        if src_img_path.exists():
            dst_img_path = images_dir / filename
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"    Warning: Image file not found: {src_img_path}")
            continue

        # Create YOLO annotation file
        yolo_annotation = create_yolo_annotation(annotations, img_width, img_height, class_mapping)

        # Save annotation file (.txt with same name as image)
        annotation_filename = Path(filename).with_suffix('.txt').name
        annotation_path = labels_dir / annotation_filename

        with open(annotation_path, 'w') as f:
            f.write(yolo_annotation)


def create_data_yaml(fold_dir: Path, num_classes: int, class_mapping: Dict[str, int]) -> None:
    """
    Create data.yaml configuration file for the fold.
    """
    yaml_content = f"""# Canopy Detection Dataset - YOLO Format (Cross-Validation)
path: {fold_dir}
train: train/images
val: val/images

# Number of classes
nc: {num_classes}

# Class names
names:
"""

    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_path = fold_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)


def create_classes_txt(fold_dir: Path, class_mapping: Dict[str, int]) -> None:
    """
    Create classes.txt file for the fold.
    """
    classes_path = fold_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")


def main():
    """Main function to create cross-validation datasets."""
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/to_yolo_cv.py <n_folds>")
        print("Example: uv run scripts/to_yolo_cv.py 3")
        sys.exit(1)

    try:
        n_folds = int(sys.argv[1])
        if n_folds < 2:
            raise ValueError("Number of folds must be at least 2")
    except ValueError as e:
        print(f"Error: {e}")
        print("Number of folds must be a valid integer >= 2")
        sys.exit(1)

    # Define paths
    data_dir = Path("data")
    annotations_path = data_dir / "train_annotations.json"
    output_dir = Path(f"cv_{n_folds}")

    # Check if annotations file exists
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        sys.exit(1)

    # Load annotations
    data = load_annotations(annotations_path)
    images = data['images']

    # Define class mapping
    class_mapping = {
        'individual_tree': 0,
        'group_of_trees': 1
    }

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Creating cross-validation dataset in: {output_dir.absolute()}")

    # Create k folds
    folds = create_k_folds(images, n_folds)

    # Create dataset for each fold
    for fold_idx in range(n_folds):
        create_fold_dataset(fold_idx, folds, output_dir, data_dir, class_mapping)

    print(f"\n✅ Cross-validation dataset creation completed!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 {n_folds} folds created with leave-one-out validation")
    print(f"🏷️  Classes: {list(class_mapping.keys())}")

    # Print summary
    print(f"\n📂 Directory structure:")
    for fold_idx in range(n_folds):
        val_size = len(folds[fold_idx])
        train_size = len(images) - val_size
        print(f"   cv_{n_folds}/fold_{fold_idx}/")
        print(f"   ├── train/        ({train_size} images)")
        print(f"   ├── val/          ({val_size} images)")
        print(f"   ├── data.yaml")
        print(f"   └── classes.txt")


if __name__ == "__main__":
    main()