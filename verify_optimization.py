#!/usr/bin/env python3
"""
Verify mathematical equivalence between original and optimized evaluation scripts.

This script loads cached original evaluation results and compares them against
optimized versions to ensure mathematical correctness during optimization.
"""

import json
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


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


def run_evaluation(gt_file: str, pred_file: str, script_path: str) -> float:
    """Run evaluation script and return score."""
    result = subprocess.run([
        'uv', 'run', script_path,
        '--gt_json', gt_file,
        '--pred_json', pred_file
    ], capture_output=True, text=True)

    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        print(f"❌ Error running {script_path}: {result.stderr}")
        return 0.0


def load_cached_results() -> Dict[str, float]:
    """Load cached original evaluation results."""
    cache_file = Path("evaluation_cache.json")

    if not cache_file.exists():
        print("❌ No cached results found!")
        print("   Run 'uv run test_skewed_eval_cached.py' first to generate cache.")
        return {}

    with open(cache_file, 'r') as f:
        return json.load(f)


def verify_script_equivalence(optimized_script: str, tolerance: float = 1e-10) -> bool:
    """
    Verify mathematical equivalence between original and optimized scripts.

    Args:
        optimized_script: Path to optimized evaluation script
        tolerance: Floating point comparison tolerance

    Returns:
        True if all results match within tolerance, False otherwise
    """
    print(f"🔍 Verifying equivalence: scripts/evaluate.py vs {optimized_script}")
    print("=" * 70)

    # Load cached original results
    cached_results = load_cached_results()
    if not cached_results:
        return False

    # Test offsets (using cached data we have available)
    offsets = [0, 1, 2, 3]
    all_passed = True
    results = []

    print(f"📊 Testing with offsets: {offsets}")
    print(f"🎯 Tolerance: {tolerance}")
    print()

    total_start = time.time()

    for offset in offsets:
        print(f"Testing offset {offset}px...")

        # Get cached original result
        original_score = cached_results.get(str(offset))
        if original_score is None:
            print(f"  ❌ No cached result for offset {offset}px")
            all_passed = False
            continue

        # Create skewed annotations
        skewed_file = create_skewed_annotations(offset)

        # Run optimized evaluation
        start_time = time.time()
        optimized_score = run_evaluation('data/train_annotations.json', skewed_file, optimized_script)
        elapsed = time.time() - start_time

        # Clean up
        Path(skewed_file).unlink()

        # Compare results
        diff = abs(original_score - optimized_score)
        passed = diff <= tolerance

        if passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False

        print(f"  Original:  {original_score:.10f}")
        print(f"  Optimized: {optimized_score:.10f}")
        print(f"  Difference: {diff:.2e}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {status}")
        print()

        results.append({
            'offset': offset,
            'original': original_score,
            'optimized': optimized_score,
            'difference': diff,
            'passed': passed,
            'time': elapsed
        })

    total_elapsed = time.time() - total_start

    # Summary
    print("=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    print(f"Tests passed: {passed_count}/{total_count}")
    print(f"Total time: {total_elapsed:.2f}s")

    if results:
        avg_time = sum(r['time'] for r in results) / len(results)
        max_diff = max(r['difference'] for r in results)
        print(f"Average time per test: {avg_time:.2f}s")
        print(f"Maximum difference: {max_diff:.2e}")

    print()

    if all_passed:
        print("🎉 VERIFICATION PASSED: Mathematical equivalence confirmed!")
        print("   Ready to proceed with next optimization step.")
    else:
        print("🚨 VERIFICATION FAILED: Mathematical equivalence not confirmed!")
        print("   Do not proceed until equivalence is achieved.")

    return all_passed


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Verify mathematical equivalence between evaluation scripts"
    )
    parser.add_argument(
        "optimized_script",
        help="Path to optimized evaluation script (e.g., scripts/evaluate_v1.py)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Floating point comparison tolerance (default: 1e-10)"
    )

    args = parser.parse_args()

    # Check if optimized script exists
    if not Path(args.optimized_script).exists():
        print(f"❌ Script not found: {args.optimized_script}")
        return 1

    # Run verification
    success = verify_script_equivalence(args.optimized_script, args.tolerance)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())