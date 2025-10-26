#!/usr/bin/env python3
"""
Cache original evaluation results for optimization verification.

This script generates skewed annotations with different pixel offsets and
runs the original evaluate.py script on each variant. Results are cached
to avoid re-running the slow original evaluation during optimization.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def create_skewed_annotations(offset_x: int) -> str:
    """Create skewed annotations with x-coordinates shifted by offset_x pixels."""
    # Load original annotations
    with open('data/train_annotations.json', 'r') as f:
        data = json.load(f)

    # Create skewed version
    skewed_data = json.loads(json.dumps(data))  # Deep copy

    for img in skewed_data['images']:
        for ann in img['annotations']:
            segmentation = ann['segmentation']
            # Shift every x coordinate (even indices) by offset_x
            for i in range(0, len(segmentation), 2):
                segmentation[i] += offset_x

    # Save skewed version
    skewed_path = f'skewed_train_annotations_{offset_x}.json'
    with open(skewed_path, 'w') as f:
        json.dump(skewed_data, f)

    return skewed_path


def run_evaluation(gt_file: str, pred_file: str) -> float:
    """Run evaluation script and return score."""
    print(f"  Running evaluation with offset {pred_file.split('_')[-1].split('.')[0]}px...")
    start_time = time.time()

    result = subprocess.run([
        'uv', 'run', 'scripts/evaluate.py',
        '--gt_json', gt_file,
        '--pred_json', pred_file
    ], capture_output=True, text=True)

    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f}s")

    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        print(f"Error: {result.stderr}")
        return 0.0


def main():
    """Generate cached results for original evaluation script."""
    print("🏗️  Caching original evaluation results for optimization verification")
    print("=" * 70)

    # Test with different x-coordinate offsets
    offsets = [0, 1, 2, 3, 4, 5]
    cache_file = Path("evaluation_cache.json")

    # Check if cache already exists
    if cache_file.exists():
        print(f"✅ Cache file already exists: {cache_file}")
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)

        print(f"📊 Cached results:")
        for offset, score in cached_results.items():
            print(f"  Offset {offset}px: {score:.6f}")
        return

    # Generate cache
    print(f"🚀 Generating cache for offsets: {offsets}")
    print(f"⏱️  Estimated time: ~{len(offsets) * 3} minutes")
    print()

    cached_results = {}
    total_start = time.time()

    for i, offset in enumerate(offsets, 1):
        print(f"[{i}/{len(offsets)}] Processing offset {offset}px...")

        # Create and test skewed annotations
        skewed_file = create_skewed_annotations(offset)
        score = run_evaluation('data/train_annotations.json', skewed_file)

        # Clean up temporary file
        Path(skewed_file).unlink()

        # Store result
        cached_results[str(offset)] = score
        print(f"  Result: {score:.6f}")
        print()

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f, indent=2)

    total_elapsed = time.time() - total_start
    print("=" * 70)
    print(f"✅ Caching completed in {total_elapsed/60:.1f} minutes")
    print(f"📁 Results saved to: {cache_file}")
    print()
    print("📊 Cached results:")
    for offset, score in cached_results.items():
        print(f"  Offset {offset}px: {score:.6f}")
    print()
    print("🎯 Ready for optimization verification!")


if __name__ == "__main__":
    main()