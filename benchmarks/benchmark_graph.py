"""Benchmarks for graph-based ECT computations"""
import numpy as np
import time
from ect import ECT, EmbeddedGraph


def create_test_shape(num_points=1000, complexity=1):
    """Create test shape with varying complexity"""
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.cos(t)
    y = np.sin(t)

    for i in range(2, complexity + 2):
        x += (1/i) * np.cos(i*t)
        y += (1/i) * np.sin(i*t)

    return np.column_stack([x, y])


def benchmark_graph_ect(num_runs=5):
    """Benchmark ECT computation on graphs"""
    results = {}

    # Warmup run to trigger JIT compilation
    warmup_shape = create_test_shape(100, 1)
    G_warmup = EmbeddedGraph()
    G_warmup.add_cycle(warmup_shape)
    myect = ECT(num_dirs=360, num_thresh=360)
    myect.calculate(G_warmup)  # Warmup run

    configs = [
        (100, 1),
        (1000, 1),
        (100, 3),
        (1000, 3),
        (10000, 3),
        (100000, 5),
    ]

    for points, complexity in configs:
        shape = create_test_shape(points, complexity)
        G = EmbeddedGraph()
        G.add_cycle(shape)

        times = []
        print(
            f"\nTesting shape with {points} points and complexity {complexity}")

        for _ in range(num_runs):
            start_time = time.time()
            myect = ECT(num_dirs=360, num_thresh=360)
            myect.calculate(G)
            times.append(time.time() - start_time)

        results[f'points_{points}_complexity_{complexity}'] = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times))
        }
    print(results)

    return results



