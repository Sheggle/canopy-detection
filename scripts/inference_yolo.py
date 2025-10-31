#!/usr/bin/env python3
"""
Minimal YOLO inference script for canopy detection.
"""

import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def run_yolo_inference(model_path, images_source, conf=0.01, iou=0.1, verbose=False):
    """
    Run YOLO inference on a set of images.

    Args:
        model_path: Path to the trained YOLO model
        images_source: Either a list of image info dicts or a directory path
        conf: Confidence threshold
        iou: IoU threshold
        verbose: Whether to show verbose output

    Returns:
        List of image info dicts with annotations added
    """
    # Load model
    model = YOLO(model_path)

    # Class mapping
    class_names = {0: "individual_tree", 1: "group_of_trees"}

    # Handle different input types
    if isinstance(images_source, (str, Path)):
        # If it's a path, load from sample_answer.json and use that directory
        with open("data/sample_answer.json") as f:
            sample_data = json.load(f)
        images_info = sample_data["images"]
        images_dir = Path(images_source)
    else:
        # If it's a list, use it directly
        images_info = images_source
        images_dir = None

    # Process each image with progress bar
    print(f"🔄 Running inference on {len(images_info)} images...")

    results_list = []
    for img_info in tqdm(images_info, desc="Processing images", unit="img"):
        # Create a copy to avoid modifying the original
        img_result = img_info.copy()

        # Determine image path
        if images_dir:
            image_path = images_dir / img_info["file_name"]
        elif "image_path" in img_info:
            # If explicit path provided in img_info
            image_path = Path(img_info["image_path"])
        else:
            # Assume path is provided in img_info or use default
            image_path = Path("data/evaluation_images") / img_info["file_name"]

        results = model(str(image_path), conf=conf, verbose=verbose, iou=iou, max_det=3000)

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

        img_result["annotations"] = annotations
        results_list.append(img_result)

    return results_list


def main():
    parser = argparse.ArgumentParser(description="YOLO inference for canopy detection")
    parser.add_argument("model_path", help="Path to the trained YOLO model")
    parser.add_argument("-o", "--output", default="submission.json", help="Output submission file (default: submission.json)")

    args = parser.parse_args()

    # Run inference using the reusable function
    results = run_yolo_inference(args.model_path, "data/evaluation_images")

    # Create submission format
    submission_data = {"images": results}

    # Save submission
    with open(args.output, "w") as f:
        json.dump(submission_data, f, indent=2)

    print(f"✅ Inference complete! Saved as {args.output}")


if __name__ == "__main__":
    main()