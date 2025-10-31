#!/usr/bin/env python3
"""
Visualize submission segmentations on evaluation images.
"""

import json
import cv2
import numpy as np
from pathlib import Path

# Load submission data
with open("experiments/cv_sweep_linu8cjt/submission.json") as f:
    submission = json.load(f)

# Create output directory
output_dir = Path("submission_images")
output_dir.mkdir(exist_ok=True)

# Colors for different classes
colors = {
    "individual_tree": (0, 255, 0),      # Green
    "group_of_trees": (0, 255, 255)      # Yellow
}

# Process each image
for img_info in submission["images"]:
    # Load image
    image_path = Path("data/train_images") / img_info["file_name"]
    img = cv2.imread(str(image_path))

    # Create overlay for 50% transparency
    overlay = img.copy()

    # Draw each annotation
    for ann in img_info["annotations"]:
        class_name = ann["class"]
        confidence = ann["confidence_score"]
        segmentation = ann["segmentation"]

        # Convert segmentation to polygon points
        points = np.array(segmentation).reshape(-1, 2).astype(np.int32)

        # Draw filled polygon on overlay
        color = colors.get(class_name, (255, 255, 255))
        cv2.fillPoly(overlay, [points], color)

        # Draw polygon outline
        cv2.polylines(img, [points], True, (0, 0, 0), 2)

        # Add confidence text
        center = points.mean(axis=0).astype(int)
        cv2.putText(img, f"{confidence:.2f}", tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Blend with 50% opacity
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # Save visualization
    output_path = output_dir / img_info["file_name"]
    cv2.imwrite(str(output_path), img)

print(f"✅ Visualizations saved to {output_dir}/")