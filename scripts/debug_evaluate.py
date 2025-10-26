#!/usr/bin/env python3
"""
Debug evaluation script for canopy detection competition.

Analyzes polygon-level IoU sensitivity to identify why small shifts cause large score drops.
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


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load and validate JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


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


def get_polygon_size(polygon: Polygon) -> Tuple[float, float]:
    """Get the width and height of a polygon's bounding box."""
    if polygon.is_empty:
        return 0.0, 0.0

    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return width, height


def shift_segmentation(segmentation: List[float], offset_x: int) -> List[float]:
    """Shift all x-coordinates in segmentation by offset_x."""
    shifted = segmentation.copy()
    for i in range(0, len(shifted), 2):
        shifted[i] += offset_x
    return shifted


def debug_polygon_pair(gt_seg: List[float], pred_seg: List[float], offset: int,
                      image_name: str, class_name: str, ann_idx: int) -> Dict[str, Any]:
    """
    Debug a specific polygon pair at different offsets.

    Returns detailed information about IoU sensitivity.
    """
    # Create original polygons
    gt_polygon = segmentation_to_polygon(gt_seg)
    pred_polygon = segmentation_to_polygon(pred_seg)

    # Create shifted prediction polygon
    shifted_pred_seg = shift_segmentation(pred_seg, offset)
    shifted_pred_polygon = segmentation_to_polygon(shifted_pred_seg)

    # Calculate IoUs
    iou_original = get_iou(gt_polygon, pred_polygon)
    iou_shifted = get_iou(gt_polygon, shifted_pred_polygon)

    # Get polygon size
    width, height = get_polygon_size(gt_polygon)

    return {
        'image_name': image_name,
        'class_name': class_name,
        'ann_idx': ann_idx,
        'offset': offset,
        'gt_segmentation': gt_seg,
        'pred_segmentation': pred_seg,
        'shifted_pred_segmentation': shifted_pred_seg,
        'iou_original': iou_original,
        'iou_shifted': iou_shifted,
        'polygon_width': width,
        'polygon_height': height,
        'polygon_area': gt_polygon.area,
        'passes_threshold_original': iou_original >= 0.75,
        'passes_threshold_shifted': iou_shifted >= 0.75,
        'sensitivity_issue': iou_original >= 0.75 and iou_shifted < 0.75
    }


def debug_image_sensitivity(gt_image: Dict[str, Any], pred_image: Dict[str, Any],
                          offset: int) -> List[Dict[str, Any]]:
    """
    Debug all polygon pairs in an image for IoU sensitivity.

    Returns list of problematic polygon pairs.
    """
    problematic_pairs = []

    classes = ["individual_tree", "group_of_trees"]

    for class_name in classes:
        # Extract ground truth annotations for this class
        gt_annotations = []
        for ann in gt_image.get("annotations", []):
            if ann["class"] == class_name:
                gt_annotations.append(ann)

        # Extract prediction annotations for this class
        pred_annotations = []
        for ann in pred_image.get("annotations", []):
            if ann["class"] == class_name:
                pred_annotations.append(ann)

        # Compare each GT with each prediction (simplified 1:1 comparison for debug)
        for gt_idx, gt_ann in enumerate(gt_annotations):
            if gt_idx < len(pred_annotations):  # 1:1 comparison for identical data
                pred_ann = pred_annotations[gt_idx]

                result = debug_polygon_pair(
                    gt_ann["segmentation"],
                    pred_ann["segmentation"],
                    offset,
                    gt_image["file_name"],
                    class_name,
                    gt_idx
                )

                if result['sensitivity_issue']:
                    problematic_pairs.append(result)

    return problematic_pairs


def analyze_all_polygons(gt_data: Dict[str, Any], output_file: str = "problematic_pairs.json"):
    """
    Analyze all polygons in the dataset to identify systematic issues causing low scores.

    Args:
        gt_data: Ground truth data
        output_file: Path to save detailed analysis results
    """
    print("🔍 Analyzing all polygons for scoring issues...")
    print("=" * 70)

    all_results = {
        "analysis_summary": {},
        "problematic_pairs": [],
        "statistics": {},
        "images_analyzed": []
    }

    total_pairs = 0
    problematic_pairs = 0
    failed_pairs = 0

    # Analyze all images (not just first 10)
    for img_idx, gt_image in enumerate(gt_data["images"]):
        print(f"Processing image {img_idx + 1}/{len(gt_data['images'])}: {gt_image['file_name']}")

        image_analysis = {
            "file_name": gt_image["file_name"],
            "scene_type": gt_image.get("scene_type", "unknown"),
            "cm_resolution": gt_image.get("cm_resolution", "unknown"),
            "total_annotations": len(gt_image.get("annotations", [])),
            "classes": {},
            "problematic_count": 0
        }

        # Analyze by class
        for class_name in ["individual_tree", "group_of_trees"]:
            gt_annotations = [ann for ann in gt_image.get("annotations", []) if ann["class"] == class_name]

            class_issues = []
            for ann_idx, ann in enumerate(gt_annotations):
                total_pairs += 1

                # Test IoU with identical polygon (should be 1.0)
                gt_polygon = segmentation_to_polygon(ann["segmentation"])
                if gt_polygon.is_empty:
                    failed_pairs += 1
                    continue

                iou_perfect = get_iou(gt_polygon, gt_polygon)

                # Test with small shift
                shifted_seg = shift_segmentation(ann["segmentation"], 1)
                shifted_polygon = segmentation_to_polygon(shifted_seg)
                iou_shifted = get_iou(gt_polygon, shifted_polygon)

                width, height = get_polygon_size(gt_polygon)

                # Check if this is problematic (perfect IoU != 1.0 or sensitive to small shifts)
                is_problematic = (iou_perfect < 0.99) or (iou_perfect >= 0.75 and iou_shifted < 0.75)

                if is_problematic:
                    problematic_pairs += 1
                    image_analysis["problematic_count"] += 1

                    problem_details = {
                        "image_name": gt_image["file_name"],
                        "class_name": class_name,
                        "annotation_index": ann_idx,
                        "polygon_width": width,
                        "polygon_height": height,
                        "polygon_area": gt_polygon.area,
                        "iou_perfect": iou_perfect,
                        "iou_1px_shift": iou_shifted,
                        "passes_threshold_perfect": iou_perfect >= 0.75,
                        "passes_threshold_shifted": iou_shifted >= 0.75,
                        "segmentation_points": len(ann["segmentation"]) // 2,
                        "gt_segmentation": ann["segmentation"],
                        "shifted_segmentation": shifted_seg,
                        "issue_type": "invalid_polygon" if iou_perfect < 0.99 else "shift_sensitive"
                    }

                    all_results["problematic_pairs"].append(problem_details)
                    class_issues.append(problem_details)

            image_analysis["classes"][class_name] = {
                "total_annotations": len(gt_annotations),
                "problematic_annotations": len(class_issues),
                "issues": class_issues[:3]  # Save first 3 issues per class for brevity
            }

        all_results["images_analyzed"].append(image_analysis)

        # Progress update
        if (img_idx + 1) % 10 == 0:
            print(f"  Processed {img_idx + 1} images, found {problematic_pairs} problematic pairs so far")

    # Calculate statistics
    all_results["statistics"] = {
        "total_images": len(gt_data["images"]),
        "total_polygon_pairs": total_pairs,
        "problematic_pairs": problematic_pairs,
        "failed_pairs": failed_pairs,
        "problematic_percentage": (problematic_pairs / total_pairs * 100) if total_pairs > 0 else 0,
        "issues_by_type": {
            "invalid_polygons": len([p for p in all_results["problematic_pairs"] if p["issue_type"] == "invalid_polygon"]),
            "shift_sensitive": len([p for p in all_results["problematic_pairs"] if p["issue_type"] == "shift_sensitive"])
        }
    }

    # Summary analysis
    if all_results["problematic_pairs"]:
        problematic = all_results["problematic_pairs"]
        all_results["analysis_summary"] = {
            "avg_polygon_width": np.mean([p["polygon_width"] for p in problematic]),
            "avg_polygon_height": np.mean([p["polygon_height"] for p in problematic]),
            "avg_polygon_area": np.mean([p["polygon_area"] for p in problematic]),
            "avg_iou_perfect": np.mean([p["iou_perfect"] for p in problematic]),
            "avg_iou_shifted": np.mean([p["iou_1px_shift"] for p in problematic]),
            "small_polygons_count": len([p for p in problematic if p["polygon_width"] < 20 or p["polygon_height"] < 20]),
            "invalid_polygons_count": len([p for p in problematic if p["iou_perfect"] < 0.99])
        }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(
        description="Debug evaluation scoring issues and save detailed analysis"
    )
    parser.add_argument(
        "--gt_json",
        type=Path,
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="problematic_pairs.json",
        help="Output file for detailed analysis (default: problematic_pairs.json)"
    )

    args = parser.parse_args()

    # Load ground truth data
    gt_data = load_json(args.gt_json)

    # Run comprehensive analysis
    results = analyze_all_polygons(gt_data, args.output)

    # Print summary
    stats = results["statistics"]
    summary = results["analysis_summary"]

    print("\n" + "=" * 70)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total images analyzed: {stats['total_images']}")
    print(f"Total polygon pairs tested: {stats['total_polygon_pairs']}")
    print(f"Problematic pairs found: {stats['problematic_pairs']} ({stats['problematic_percentage']:.1f}%)")
    print(f"Failed/invalid pairs: {stats['failed_pairs']}")

    if summary:
        print(f"\n🚨 Issues found:")
        print(f"  Invalid polygons (IoU != 1.0 with self): {stats['issues_by_type']['invalid_polygons']}")
        print(f"  Shift-sensitive polygons: {stats['issues_by_type']['shift_sensitive']}")
        print(f"  Small polygons (<20px): {summary['small_polygons_count']}")
        print(f"  Average perfect IoU: {summary['avg_iou_perfect']:.4f}")
        print(f"  Average 1px-shift IoU: {summary['avg_iou_shifted']:.4f}")
        print(f"  Average polygon size: {summary['avg_polygon_width']:.1f} x {summary['avg_polygon_height']:.1f} px")

    print(f"\n📁 Detailed results saved to: {args.output}")
    print("   Use this file to analyze specific polygon issues causing low scores.")


if __name__ == "__main__":
    main()