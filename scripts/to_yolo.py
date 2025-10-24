#!/usr/bin/env python3
"""
Convert canopy detection dataset to YOLO polygon format.

This script converts the JSON annotation format to YOLO segmentation format:
- Creates yolo_dataset/ directory structure
- Converts polygon coordinates to normalized YOLO format
- Copies images to appropriate train/val directories
- Creates class names file
"""

import json
import shutil
from pathlib import Path
import os
from typing import List, Tuple, Dict, Any


def load_annotations(json_path: Path) -> Dict[str, Any]:
    """Load annotations from JSON file."""
    print(f"Loading annotations from {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)


def polygon_to_yolo_format(segmentation: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert polygon coordinates to YOLO normalized format.

    Args:
        segmentation: List of [x1, y1, x2, y2, ..., xn, yn] coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of normalized coordinates [x1_norm, y1_norm, x2_norm, y2_norm, ...]
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

    Args:
        annotations: List of annotation dictionaries
        img_width: Image width
        img_height: Image height
        class_mapping: Mapping from class names to class IDs

    Returns:
        YOLO annotation string (one line per object)
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


def split_dataset(images: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        images: List of image dictionaries
        train_ratio: Ratio of training data (0.8 = 80% train, 20% val)

    Returns:
        Tuple of (train_images, val_images)
    """
    import random

    # Set seed for reproducible splits
    random.seed(42)

    # Shuffle the images
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)

    # Calculate split point
    split_idx = int(len(shuffled_images) * train_ratio)

    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]

    return train_images, val_images


def convert_to_yolo(
    data_dir: Path = Path("data"),
    output_dir: Path = Path("yolo_dataset"),
    train_ratio: float = 0.8
) -> None:
    """
    Convert the canopy detection dataset to YOLO format.

    Args:
        data_dir: Path to the input data directory
        output_dir: Path to the output YOLO dataset directory
        train_ratio: Ratio for train/val split
    """
    print("Starting YOLO conversion...")

    # Define paths
    annotations_path = data_dir / "train_annotations.json"
    train_images_dir = data_dir / "train_images"

    # Load annotations
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    data = load_annotations(annotations_path)
    images = data['images']

    print(f"Found {len(images)} images with annotations")

    # Define class mapping
    class_mapping = {
        'individual_tree': 0,
        'group_of_trees': 1
    }

    # Create output directory structure
    output_dir.mkdir(exist_ok=True)

    # Create train and val directories
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Split dataset
    train_images, val_images = split_dataset(images, train_ratio)

    print(f"Split: {len(train_images)} training, {len(val_images)} validation images")

    # Process train and val splits
    for split_name, split_images in [('train', train_images), ('val', val_images)]:
        print(f"\nProcessing {split_name} split...")

        images_out_dir = output_dir / split_name / 'images'
        labels_out_dir = output_dir / split_name / 'labels'

        for i, img_data in enumerate(split_images):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(split_images)} images")

            # Get image info
            filename = img_data['file_name']
            img_width = img_data['width']
            img_height = img_data['height']
            annotations = img_data['annotations']

            # Copy image file
            src_img_path = train_images_dir / filename
            if src_img_path.exists():
                dst_img_path = images_out_dir / filename
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image file not found: {src_img_path}")
                continue

            # Create YOLO annotation file
            yolo_annotation = create_yolo_annotation(annotations, img_width, img_height, class_mapping)

            # Save annotation file (.txt with same name as image)
            annotation_filename = Path(filename).with_suffix('.txt').name
            annotation_path = labels_out_dir / annotation_filename

            with open(annotation_path, 'w') as f:
                f.write(yolo_annotation)

    # Create classes.txt file
    classes_path = output_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")

    # Create data.yaml file for YOLO training
    yaml_content = f"""# Canopy Detection Dataset - YOLO Format
path: {output_dir}
train: train/images
val: val/images

# Number of classes
nc: {len(class_mapping)}

# Class names
names:
"""

    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        yaml_content += f"  {class_id}: {class_name}\n"

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ YOLO conversion completed!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Dataset split: {len(train_images)} train, {len(val_images)} val")
    print(f"🏷️  Classes: {list(class_mapping.keys())}")
    print(f"📋 Configuration file: {yaml_path}")

    # Print directory structure
    print(f"\n📂 Directory structure:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/     ({len(train_images)} .tif files)")
    print(f"   │   └── labels/     ({len(train_images)} .txt files)")
    print(f"   ├── val/")
    print(f"   │   ├── images/     ({len(val_images)} .tif files)")
    print(f"   │   └── labels/     ({len(val_images)} .txt files)")
    print(f"   ├── classes.txt")
    print(f"   └── data.yaml")


def main():
    """Main function to run the conversion."""
    try:
        convert_to_yolo()
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()