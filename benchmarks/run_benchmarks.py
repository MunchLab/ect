"""Main benchmark runner for ECT package"""
import numpy as np
import time
from pathlib import Path
import json
from benchmark_graph import benchmark_graph_ect, benchmark_g_omega
from benchmark_cw import benchmark_cw_ect
import platform


def run_all_benchmarks(num_runs=5):
    """Run all benchmarks and collect results"""
    results = {
        'metadata': {
            'num_runs': num_runs,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'platform': platform.platform(),
            'python_version': platform.python_version()
        },
        'benchmarks': {}
    }

    print("\nRunning graph ECT benchmarks...")
    results['benchmarks']['graph_ect'] = benchmark_graph_ect(num_runs=num_runs)

    print("\nRunning CW complex benchmarks...")
    results['benchmarks']['cw_ect'] = benchmark_cw_ect(num_runs=num_runs)

    print("\nRunning g_omega benchmarks...")
    results['benchmarks']['g_omega'] = benchmark_g_omega(num_runs=num_runs)

    return results


def save_results(results, output_dir="benchmarks/results"):
    """Save benchmark results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/benchmark_results.json")


if __name__ == "__main__":
    results = run_all_benchmarks()
    save_results(results)
