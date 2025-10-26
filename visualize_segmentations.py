#!/usr/bin/env python3
"""
Visualize training images with segmentation annotations overlaid.

Creates visualizations with 50% opacity overlays to inspect annotation quality
and understand why evaluation scores are low.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any


def load_annotations(json_path: Path) -> Dict[str, Any]:
    """Load annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def segmentation_to_points(segmentation: List[float]) -> np.ndarray:
    """
    Convert segmentation array to OpenCV points format.

    Args:
        segmentation: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        numpy array of points for cv2.fillPoly
    """
    if len(segmentation) < 6:  # Need at least 3 points
        return np.array([])

    # Convert flat list to pairs of coordinates
    points = []
    for i in range(0, len(segmentation), 2):
        if i + 1 < len(segmentation):
            points.append([int(segmentation[i]), int(segmentation[i + 1])])

    return np.array(points, dtype=np.int32)


def create_overlay_image(image_path: Path, annotations: List[Dict],
                        output_path: Path, opacity: float = 0.5) -> bool:
    """
    Create an image with segmentation overlays.

    Args:
        image_path: Path to the original image
        annotations: List of annotation dictionaries
        output_path: Path to save the overlaid image
        opacity: Opacity of the overlay (0.0 to 1.0)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the original image
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return False

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            return False

        h, w = image.shape[:2]

        # Create overlay for segmentations
        overlay = image.copy()

        # Color scheme for different classes
        colors = {
            "individual_tree": (0, 255, 0),    # Green
            "group_of_trees": (255, 0, 0)      # Blue
        }

        # Draw each annotation
        for ann in annotations:
            class_name = ann.get("class", "unknown")
            segmentation = ann.get("segmentation", [])

            if not segmentation:
                continue

            # Convert segmentation to points
            points = segmentation_to_points(segmentation)
            if len(points) < 3:  # Need at least 3 points for a polygon
                continue

            # Get color for this class
            color = colors.get(class_name, (128, 128, 128))  # Gray for unknown

            # Fill the polygon
            cv2.fillPoly(overlay, [points], color)

            # Draw polygon outline for better visibility
            cv2.polylines(overlay, [points], True, color, 2)

            # Add a small circle at the centroid for tiny polygons
            if len(points) > 0:
                centroid_x = int(np.mean(points[:, 0]))
                centroid_y = int(np.mean(points[:, 1]))
                cv2.circle(overlay, (centroid_x, centroid_y), 3, color, -1)

        # Blend the overlay with the original image
        result = cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)

        # Add legend
        legend_height = 60
        legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
        cv2.putText(legend, "individual_tree", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(legend, "group_of_trees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Combine result with legend
        final_result = np.vstack([result, legend])

        # Save the result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), final_result)

        if success:
            return True
        else:
            print(f"Warning: Failed to save image: {output_path}")
            return False

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def main():
    """Main function to create segmentation visualizations."""
    parser = argparse.ArgumentParser(
        description="Create visualizations of training images with segmentation overlays"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default="data/train_annotations.json",
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default="data/training_images",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="segmented_images",
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of segmentation overlay (0.0 to 1.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of images to process (for testing)"
    )

    args = parser.parse_args()

    # Load annotations
    print(f"Loading annotations from {args.annotations}")
    data = load_annotations(args.annotations)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating segmentation visualizations...")
    print(f"Input images: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Overlay opacity: {args.opacity}")
    print(f"Total images to process: {len(data['images'])}")

    if args.limit:
        print(f"Limiting to first {args.limit} images")

    # Process each image
    processed = 0
    successful = 0

    for i, image_info in enumerate(data["images"]):
        if args.limit and i >= args.limit:
            break

        file_name = image_info["file_name"]
        annotations = image_info.get("annotations", [])

        # Construct image path
        image_path = args.images_dir / file_name

        # Output path
        output_path = args.output_dir / f"segmented_{file_name.replace('.tif', '.png')}"

        # Create visualization
        if create_overlay_image(image_path, annotations, output_path, args.opacity):
            successful += 1

        processed += 1

        # Progress update
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(data['images'])} images, {successful} successful")

    print(f"\n✅ Visualization complete!")
    print(f"Processed: {processed} images")
    print(f"Successful: {successful} images")
    print(f"Output directory: {args.output_dir}")

    if successful < processed:
        print(f"⚠️  {processed - successful} images failed (likely missing image files)")


if __name__ == "__main__":
    main()