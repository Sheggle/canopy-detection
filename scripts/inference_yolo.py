#!/usr/bin/env python3
"""
Minimal YOLO inference script for canopy detection.
"""

import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="YOLO inference for canopy detection")
    parser.add_argument("model_path", help="Path to the trained YOLO model")
    parser.add_argument("-o", "--output", default="submission.json", help="Output submission file (default: submission.json)")

    args = parser.parse_args()

    # Load model and sample format
    model = YOLO(args.model_path)
    with open("data/sample_answer.json") as f:
        sample_data = json.load(f)

    # Class mapping
    class_names = {0: "individual_tree", 1: "group_of_trees"}

    # Process each image with progress bar
    print(f"🔄 Running inference on {len(sample_data['images'])} images...")
    for img_info in tqdm(sample_data["images"], desc="Processing images", unit="img"):
        image_path = Path("data/evaluation_images") / img_info["file_name"]
        results = model(str(image_path), conf=0.01, verbose=False, iou=0.1)

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
    with open(args.output, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"✅ Inference complete! Saved as {args.output}")


if __name__ == "__main__":
    main()