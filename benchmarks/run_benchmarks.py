"""Main benchmark runner for ECT package"""
import numpy as np
import time
from ect import ECT, EmbeddedGraph
import json
from pathlib import Path


def create_test_shape(num_points=1000):
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.cos(t) + 0.5 * np.cos(3*t)
    y = np.sin(t) + 0.5 * np.sin(3*t)
    return np.column_stack([x, y])


def run_benchmarks():
    results = {}

    sizes = [100, 500, 1000]
    for size in sizes:
        shape = create_test_shape(size)
        G = EmbeddedGraph()
        G.add_cycle(shape)

        start_time = time.time()
        myect = ECT(num_dirs=360, num_thresh=360)
        myect.calculateECT(G)
        ect_time = time.time() - start_time

        start_time = time.time()
        for theta in np.linspace(0, 2*np.pi, 360):
            G.g_omega(theta)
        g_omega_time = time.time() - start_time

        results[f'shape_size_{size}'] = {
            'ect_time': ect_time,
            'g_omega_time': g_omega_time
        }

    return results


if __name__ == "__main__":

    results = run_benchmarks()

    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
