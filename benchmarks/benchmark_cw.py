"""Benchmarks for CW complex computations"""
import numpy as np
import time
from ect import ECT, EmbeddedCW, create_example_cw
import json
from pathlib import Path


def benchmark_cw_ect(num_runs=5):
    """Benchmark ECT computation on CW complexes"""
    results = {}

    configs = [
        (8, 10),    # Small
        (36, 36),   # Medium
        (360, 360),  # Large
    ]

    for num_dir, num_thresh in configs:
        times = []
        K = create_example_cw()

        print(
            f"\nTesting ECT with {num_dir} directions, {num_thresh} thresholds")
        for _ in range(num_runs):
            start_time = time.time()

            myect = ECT(num_dirs=num_dir, num_thresh=num_thresh)
            myect.calculateECT(K)

            execution_time = time.time() - start_time
            times.append(execution_time)

        results[f'dirs_{num_dir}_thresh_{num_thresh}'] = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times))
        }

    return results


if __name__ == "__main__":
    print("Running CW complex benchmarks...")
    results = benchmark_cw_ect()

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "cw_results.json", "w") as f:
        json.dump(results, f, indent=2)
