#!/usr/bin/env python3
"""
Minimal YOLO inference script for canopy detection.
"""

import json
import sys
from pathlib import Path
from ultralytics import YOLO

# Load model and sample format
model = YOLO(sys.argv[1])
with open("data/sample_answer.json") as f:
    sample_data = json.load(f)

# Class mapping
class_names = {0: "individual_tree", 1: "group_of_trees"}

# Process each image
for img_info in sample_data["images"]:
    image_path = Path("data/evaluation_images") / img_info["file_name"]
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

    img_info["annotations"] = annotations

# Save submission
with open("submission.json", "w") as f:
    json.dump(sample_data, f, indent=2)

print("✅ Inference complete! Saved as submission.json")