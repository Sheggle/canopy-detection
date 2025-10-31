#!/usr/bin/env python3
"""
Script to remove all annotations with class "group_of_trees" from a submission JSON file.
"""

import json
import sys
from pathlib import Path

def filter_group_of_trees(input_file, output_file):
    """
    Remove all annotations with class 'group_of_trees' from the submission file.

    Args:
        input_file (str): Path to the input submission JSON file
        output_file (str): Path to the output filtered JSON file
    """
    print(f"Loading submission from: {input_file}")

    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Count original annotations
    total_original = 0
    total_group_of_trees = 0

    # Filter out group_of_trees annotations from each image
    for image in data['images']:
        original_count = len(image['annotations'])
        total_original += original_count

        # Filter out group_of_trees annotations
        filtered_annotations = [
            ann for ann in image['annotations']
            if ann['class'] != 'group_of_trees'
        ]

        group_of_trees_count = original_count - len(filtered_annotations)
        total_group_of_trees += group_of_trees_count

        # Update the annotations
        image['annotations'] = filtered_annotations

    print(f"Original annotations: {total_original}")
    print(f"Removed group_of_trees annotations: {total_group_of_trees}")
    print(f"Remaining annotations: {total_original - total_group_of_trees}")

    # Save the filtered data
    print(f"Saving filtered submission to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("Filtering completed successfully!")

if __name__ == "__main__":
    input_file = "submissions/submission_fold_0.json"
    output_file = "submissions/submission_fold_0_no_groups.json"

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)

    filter_group_of_trees(input_file, output_file)