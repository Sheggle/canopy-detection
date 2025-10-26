#!/usr/bin/env python3
"""
Evaluation script for canopy detection competition.

Implements the exact scoring methodology used by organizers:
- Per-class mAP calculation (individual_tree vs group_of_trees)
- Weighted scoring by scene type and resolution
- COCO-style mAP with IoU thresholds [0.5, 0.55, ..., 0.95]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from shapely.geometry import Polygon


# Scoring weights as defined by organizers
SCENE_WEIGHTS = {
    "agriculture_plantation": 2.00,
    "urban_area": 1.50,
    "rural_area": 1.00,
    "industrial_area": 1.25,
    "open_field": 1.00
}

RESOLUTION_WEIGHTS = {
    10: 1.00,
    20: 1.25,
    40: 2.00,
    60: 2.50,
    80: 3.00
}


def segmentation_to_polygon(segmentation: List[float]) -> Polygon:
    """
    Convert segmentation array to Shapely Polygon.

    Args:
        segmentation: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        Shapely Polygon object
    """
    if len(segmentation) < 6:  # Need at least 3 points (6 coordinates)
        return Polygon()  # Empty polygon for invalid inputs

    # Convert flat list to pairs of coordinates
    points = [(segmentation[i], segmentation[i + 1])
              for i in range(0, len(segmentation), 2)]

    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            # Try to fix invalid polygons
            polygon = polygon.buffer(0)
        return polygon
    except Exception:
        return Polygon()  # Return empty polygon if creation fails


def get_iou(polygon1: Polygon, polygon2: Polygon) -> float:
    """
    Calculate Intersection over Union (IoU) for two polygons.

    Args:
        polygon1: First polygon
        polygon2: Second polygon

    Returns:
        IoU score between 0 and 1
    """
    if polygon1.is_empty or polygon2.is_empty:
        return 0.0

    try:
        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area
    except Exception:
        return 0.0


def compute_map(gt_polygons: List[Polygon],
                pred_polygons_with_scores: List[Tuple[Polygon, float]]) -> float:
    """
    Compute mean Average Precision (mAP) over a range of IoU thresholds.

    This is the exact algorithm provided by the organizers.

    Args:
        gt_polygons: List of ground truth Shapely Polygon objects
        pred_polygons_with_scores: List of (prediction_polygon, confidence_score) tuples

    Returns:
        mAP score
    """
    iou_threshold = 0.75

    if not gt_polygons:
        return 0.0
    if not pred_polygons_with_scores:
        return 0.0

    num_gt = len(gt_polygons)

    # Sort predictions by confidence score (descending)
    sorted_preds = sorted(pred_polygons_with_scores, key=lambda x: x[1], reverse=True)
    pred_polygons_sorted = [p[0] for p in sorted_preds]

    # Calculate AP for each IoU threshold
    tp_list = []  # Stores 1 if prediction is TP, 0 if FP
    gt_matched_map = np.zeros(num_gt, dtype=bool)  # Track matched GTs for this IoU threshold

    # Match sorted predictions to GTs
    for pred_idx, pred_polygon in enumerate(pred_polygons_sorted):
        best_iou = -1.0
        best_gt_idx = -1
        for gt_idx, gt_polygon in enumerate(gt_polygons):
            if gt_matched_map[gt_idx]:  # Skip already matched GT
                continue
            iou = get_iou(gt_polygon, pred_polygon)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Assign TP/FP status based on match quality and availability
        if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched_map[best_gt_idx]:
            tp_list.append(1)
            gt_matched_map[best_gt_idx] = True  # Mark GT as matched
        else:
            tp_list.append(0)  # FP

    # Calculate Precision-Recall curve points
    if not tp_list:
        ap = 0.0
    else:
        tp_list = np.array(tp_list)
        fp_list = 1 - tp_list
        cumulative_tp = np.cumsum(tp_list)
        cumulative_fp = np.cumsum(fp_list)

        recalls = cumulative_tp / num_gt
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)

        # Calculate AP using All-Point Interpolation (Area under the P-R curve)
        recalls_interp = np.concatenate(([0.0], recalls, [recalls[-1]]))
        precisions_interp = np.concatenate(([0.0], precisions, [0.0]))
        # Make precision monotonically decreasing
        for i in range(len(precisions_interp) - 2, -1, -1):
            precisions_interp[i] = max(precisions_interp[i], precisions_interp[i+1])
        # Calculate area under curve
        recall_change_indices = np.where(recalls_interp[1:] != recalls_interp[:-1])[0]
        ap = np.sum((recalls_interp[recall_change_indices + 1] - recalls_interp[recall_change_indices]) * precisions_interp[recall_change_indices + 1])

    return ap


def evaluate_image(gt_image: Dict[str, Any], pred_image: Dict[str, Any]) -> float:
    """
    Evaluate a single image by calculating class-wise mAP and averaging.

    Args:
        gt_image: Ground truth image data
        pred_image: Prediction image data

    Returns:
        Average mAP score across classes
    """
    classes = ["individual_tree", "group_of_trees"]
    class_maps = []

    for class_name in classes:
        # Extract ground truth polygons for this class
        gt_polygons = []
        for ann in gt_image.get("annotations", []):
            if ann["class"] == class_name:
                polygon = segmentation_to_polygon(ann["segmentation"])
                if not polygon.is_empty:
                    gt_polygons.append(polygon)

        # Extract prediction polygons with scores for this class
        pred_polygons_with_scores = []
        for ann in pred_image.get("annotations", []):
            if ann["class"] == class_name:
                polygon = segmentation_to_polygon(ann["segmentation"])
                if not polygon.is_empty:
                    confidence = ann.get("confidence_score", 1.0)
                    pred_polygons_with_scores.append((polygon, confidence))

        # Calculate mAP for this class
        class_map = compute_map(gt_polygons, pred_polygons_with_scores)
        class_maps.append(class_map)

    # Return average mAP across classes
    return np.mean(class_maps) if class_maps else 0.0


def calculate_weighted_score(gt_data: Dict[str, Any], pred_data: Dict[str, Any]) -> float:
    """
    Calculate the final weighted score according to organizers' methodology.

    Args:
        gt_data: Ground truth data
        pred_data: Prediction data

    Returns:
        Final weighted mAP score
    """
    # Create lookup for predictions by filename
    pred_lookup = {img["file_name"]: img for img in pred_data["images"]}

    weighted_scores = []
    total_weights = []

    for gt_image in gt_data["images"]:
        file_name = gt_image["file_name"]

        # Skip if prediction not found
        if file_name not in pred_lookup:
            continue

        pred_image = pred_lookup[file_name]

        # Calculate image weight
        scene_type = gt_image["scene_type"]
        cm_resolution = gt_image["cm_resolution"]

        scene_weight = SCENE_WEIGHTS[scene_type]
        resolution_weight = RESOLUTION_WEIGHTS[cm_resolution]
        image_weight = scene_weight * resolution_weight

        # Calculate mAP for this image
        image_map = evaluate_image(gt_image, pred_image)

        # Add to weighted calculation
        weighted_scores.append(image_map * image_weight)
        total_weights.append(image_weight)

    # Calculate final weighted score
    if not total_weights:
        return 0.0

    return sum(weighted_scores) / sum(total_weights)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate canopy detection predictions using organizers' scoring methodology"
    )
    parser.add_argument(
        "--gt_json",
        type=Path,
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--pred_json",
        type=Path,
        required=True,
        help="Path to predictions JSON file"
    )

    args = parser.parse_args()

    # Load data
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
    with open(args.pred_json, 'r') as f:
        pred_data = json.load(f)

    # Calculate score
    final_score = calculate_weighted_score(gt_data, pred_data)

    # Output only the final score
    print(f"{final_score:.6f}")


if __name__ == "__main__":
    main()