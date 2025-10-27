#!/usr/bin/env python3
"""
Segment visualization script for canopy detection.

Takes an image directory and segmentation JSON file as input, and creates
segmented images showing the detected tree canopies overlaid on the original images.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def clean_output_directory(output_dir: Path) -> None:
    """Clean the output directory if it exists, then create it."""
    if output_dir.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")


def load_segmentation_data(segmentation_file: Path) -> Dict:
    """Load and parse the segmentation JSON file."""
    print(f"Loading segmentation data from: {segmentation_file}")
    try:
        with open(segmentation_file, 'r') as f:
            data = json.load(f)

        # Validate the expected format
        if 'images' not in data:
            raise ValueError("Segmentation file must contain 'images' key")

        print(f"Loaded segmentation data for {len(data['images'])} images")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load segmentation data: {e}")


def polygon_to_coordinates(segmentation: List[float]) -> List[Tuple[int, int]]:
    """Convert flat segmentation array to list of (x, y) coordinate tuples."""
    if len(segmentation) % 2 != 0:
        raise ValueError("Segmentation array must have even number of elements")

    coordinates = []
    for i in range(0, len(segmentation), 2):
        x, y = int(segmentation[i]), int(segmentation[i + 1])
        coordinates.append((x, y))

    return coordinates


def create_segmented_image(image_path: Path, image_data: Dict, output_path: Path) -> None:
    """Create a segmented image with annotations overlaid."""
    # Load the original image
    try:
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return

    # Create a copy for drawing
    segmented_image = original_image.copy()
    draw = ImageDraw.Draw(segmented_image, 'RGBA')

    # Color scheme for different classes
    class_colors = {
        'individual_tree': (0, 255, 0, 80),  # Semi-transparent green
        'tree_cluster': (255, 255, 0, 80),   # Semi-transparent yellow
        'default': (255, 0, 0, 80)           # Semi-transparent red
    }

    annotations_count = 0

    # Draw each annotation
    for annotation in image_data.get('annotations', []):
        try:
            # Get polygon coordinates
            segmentation = annotation.get('segmentation', [])
            if not segmentation:
                continue

            coordinates = polygon_to_coordinates(segmentation)
            if len(coordinates) < 3:  # Need at least 3 points for a polygon
                continue

            # Get color for this class
            tree_class = annotation.get('class', 'default')
            color = class_colors.get(tree_class, class_colors['default'])

            # Draw filled polygon
            draw.polygon(coordinates, fill=color)

            # Draw polygon outline in a more opaque version
            outline_color = tuple(list(color[:3]) + [255])  # Fully opaque outline
            draw.polygon(coordinates, outline=outline_color)

            annotations_count += 1

        except Exception as e:
            print(f"Warning: Failed to draw annotation: {e}")
            continue

    # Save the segmented image
    try:
        segmented_image.save(output_path)
        print(f"Created segmented image: {output_path} ({annotations_count} annotations)")
    except Exception as e:
        print(f"Error: Failed to save segmented image {output_path}: {e}")


def main():
    """Main function to process images and create segmented visualizations."""
    parser = argparse.ArgumentParser(
        description="Create segmented images from image directory and segmentation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/segment.py data/train_images/ experiments/submission.json
  python scripts/segment.py /path/to/images /path/to/segmentations.json --output custom_output/
        """
    )

    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory containing input images'
    )

    parser.add_argument(
        'segmentation_file',
        type=str,
        help='JSON file containing segmentation data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='segmented_images',
        help='Output directory for segmented images (default: segmented_images)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    image_dir = Path(args.image_dir)
    segmentation_file = Path(args.segmentation_file)
    output_dir = Path(args.output)

    # Validate input paths
    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        return 1

    if not image_dir.is_dir():
        print(f"Error: Image path is not a directory: {image_dir}")
        return 1

    if not segmentation_file.exists():
        print(f"Error: Segmentation file does not exist: {segmentation_file}")
        return 1

    try:
        # Clean and create output directory
        clean_output_directory(output_dir)

        # Load segmentation data
        segmentation_data = load_segmentation_data(segmentation_file)

        # Create a mapping from filename to image data
        image_data_map = {
            image_data['file_name']: image_data
            for image_data in segmentation_data['images']
        }

        # Process each image that has segmentation data
        processed_count = 0
        skipped_count = 0

        for filename, image_data in image_data_map.items():
            image_path = image_dir / filename

            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                skipped_count += 1
                continue

            # Create output filename (convert to PNG for better compatibility)
            output_filename = Path(filename).with_suffix('.png').name
            output_path = output_dir / output_filename

            # Create segmented image
            create_segmented_image(image_path, image_data, output_path)
            processed_count += 1

        print(f"\nProcessing complete!")
        print(f"Processed: {processed_count} images")
        print(f"Skipped: {skipped_count} images")
        print(f"Output directory: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())