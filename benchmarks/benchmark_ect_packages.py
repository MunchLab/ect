#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from dect.ect import compute_ect as DECT_COMPUTE_ECT
from dect.ect_fn import scaled_sigmoid as DECT_SCALED_SIGMOID
from pyect import WECT as PYECT_WECT

from ect import ECT, EmbeddedComplex, Directions

N_POINTS = 500
DIM = 3
NUM_DIRS = 256
NUM_THRESH = 256
RESOLUTIONS = "32,64,128,256,512"
REPEATS = 10
SEED = 42
DECT_SCALE = 500.0
STRUCTURE = "grid2d"
TORCH_DEVICE = "cpu"


def generate_points(n_points: int, dim: int, seed: int) -> np.ndarray:
    """
    Generate random points in [0,1]^dim.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_points, dim), dtype=np.float64)


def build_point_only_complex(points: np.ndarray) -> EmbeddedComplex:
    """
    Construct an EmbeddedComplex containing only 0-cells (vertices).
    This ensures our ECT matches a constant-weight WECT computed on the same set of points.
    """
    K = EmbeddedComplex()
    nodes_with_coords = [(i, points[i]) for i in range(points.shape[0])]
    K.add_nodes_from(nodes_with_coords)
    return K


def build_grid2d_complex(grid_size: int) -> EmbeddedComplex:
    """
    Build a simple 2D triangular grid complex with vertices, edges, and triangular faces.
    Vertices are on a unit square grid of shape (grid_size x grid_size).
    """
    K = EmbeddedComplex()
    m = grid_size
    # Add vertices with coordinates
    coords = []
    for y in range(m):
        for x in range(m):
            coords.append([x / (m - 1), y / (m - 1)])
    coords = np.asarray(coords, dtype=np.float64)
    K.add_nodes_from([(i, coords[i]) for i in range(coords.shape[0])])

    def vid(x, y):
        return y * m + x

    # Add grid edges (right and down neighbors)
    edges = []
    for y in range(m):
        for x in range(m):
            if x + 1 < m:
                edges.append((vid(x, y), vid(x + 1, y)))
            if y + 1 < m:
                edges.append((vid(x, y), vid(x, y + 1)))
    if edges:
        K.add_edges_from(edges)

    # Add faces: split each cell into two triangles
    for y in range(m - 1):
        for x in range(m - 1):
            v00 = vid(x, y)
            v10 = vid(x + 1, y)
            v01 = vid(x, y + 1)
            v11 = vid(x + 1, y + 1)
            K.add_cell([v00, v10, v11], dim=2)
            K.add_cell([v00, v11, v01], dim=2)

    return K


def _shared_directions(num_dirs: int, dim: int, seed: int) -> np.ndarray:
    if dim == 2:
        angles = np.linspace(
            0.0, 2.0 * np.pi, num_dirs, endpoint=False, dtype=np.float64
        )
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((num_dirs, dim), dtype=np.float64)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return dirs / norms


def _pyect_thresholds(points: np.ndarray, num_heights: int) -> np.ndarray:
    if num_heights <= 1:
        return np.array([0.0], dtype=np.float64)
    max_height = float(np.max(np.linalg.norm(points, axis=1)))
    if max_height == 0.0:
        return np.array([0.0], dtype=np.float64)
    k = np.arange(num_heights, dtype=np.float64)
    return (2.0 * k / float(num_heights - 1) - 1.0) * max_height


def _resolve_torch_device(torch_module, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("Requested torch device 'mps' is not available.")
        return "mps"
    return "mps" if mps_available else "cpu"


def _sync_if_mps(torch_module, device: str) -> None:
    if device == "mps":
        torch_module.mps.synchronize()


def _time_callable(
    fn,
    repeats: int,
    torch_module=None,
    device: str = "cpu",
) -> float:
    _ = fn()
    if torch_module is not None:
        _sync_if_mps(torch_module, device)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = fn()
    if torch_module is not None:
        _sync_if_mps(torch_module, device)
    end = time.perf_counter()
    return (end - start) / max(1, repeats)


def _compare_outputs(baseline_out: np.ndarray, pyect_out: np.ndarray):
    equal_flag = bool(np.array_equal(baseline_out, pyect_out))
    max_diff = float(
        np.max(np.abs(baseline_out.astype(np.float64) - pyect_out.astype(np.float64)))
    )
    return equal_flag, max_diff


def _resolve_resolutions(num_dirs: int, resolutions_text: str):
    if resolutions_text.strip():
        return [int(tok) for tok in resolutions_text.split(",") if tok.strip()]
    base = max(4, int(num_dirs))
    candidates = [max(4, base // 4), max(4, base // 2), base, base * 2]
    return sorted(set(candidates))


def _build_grid2d_torch_complex(grid_size: int, torch_module, device: str):
    m = grid_size
    xs = np.linspace(0.0, 1.0, m, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, m, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)
    x = torch_module.from_numpy(coords).to(device)

    def vid(xi, yi):
        return yi * m + xi

    edges = []
    for yi in range(m):
        for xi in range(m):
            if xi + 1 < m:
                edges.append([vid(xi, yi), vid(xi + 1, yi)])
            if yi + 1 < m:
                edges.append([vid(xi, yi), vid(xi, yi + 1)])
    edges_t = (
        torch_module.tensor(edges, dtype=torch_module.long, device=device)
        if edges
        else torch_module.empty((0, 2), dtype=torch_module.long, device=device)
    )

    faces = []
    for yi in range(m - 1):
        for xi in range(m - 1):
            v00 = vid(xi, yi)
            v10 = vid(xi + 1, yi)
            v01 = vid(xi, yi + 1)
            v11 = vid(xi + 1, yi + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces_t = (
        torch_module.tensor(faces, dtype=torch_module.long, device=device)
        if faces
        else torch_module.empty((0, 3), dtype=torch_module.long, device=device)
    )
    return x, edges_t, faces_t


def _build_pyect_points_inputs(
    points: np.ndarray,
    directions_np: np.ndarray,
    num_heights: int,
    torch_module,
    WECT,
    device: str,
):
    x = torch_module.from_numpy(points.astype(np.float32, copy=False)).to(device)
    dirs = torch_module.from_numpy(directions_np.astype(np.float32, copy=False)).to(
        device
    )
    model = WECT(dirs, int(num_heights))
    v_weights = torch_module.ones(x.shape[0], device=device, dtype=torch_module.float32)
    complex_data = [(x, v_weights)]
    return model, complex_data


def _build_pyect_grid2d_inputs(
    grid_size: int,
    directions_np: np.ndarray,
    num_heights: int,
    torch_module,
    WECT,
    device: str,
):
    x, edges_t, faces_t = _build_grid2d_torch_complex(grid_size, torch_module, device)
    dirs = torch_module.from_numpy(directions_np.astype(np.float32, copy=False)).to(
        device
    )
    model = WECT(dirs, int(num_heights))
    v_weights = torch_module.ones(x.shape[0], device=device, dtype=torch_module.float32)
    e_weights = torch_module.ones(
        edges_t.shape[0], device=device, dtype=torch_module.float32
    )
    f_weights = torch_module.ones(
        faces_t.shape[0], device=device, dtype=torch_module.float32
    )
    complex_data = [
        (x, v_weights),
        (edges_t, e_weights),
        (faces_t, f_weights),
    ]
    return model, complex_data


def time_baseline_ect_shared(
    graph: EmbeddedComplex,
    directions_np: np.ndarray,
    thresholds: np.ndarray,
    repeats: int,
) -> float:
    directions = Directions.from_vectors(directions_np)
    ect = ECT(directions=directions, thresholds=thresholds)
    _ = ect.calculate(graph)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = ect.calculate(graph)
    end = time.perf_counter()
    return (end - start) / max(1, repeats)


def compute_naive_ect_output(
    graph: EmbeddedComplex,
    directions_np: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    directions = Directions.from_vectors(directions_np)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    V = directions.vectors
    X = graph.coord_matrix
    node_projections = X @ V.T if V.shape[1] == X.shape[1] else X @ V
    num_dirs = node_projections.shape[1]
    num_thresh = thresholds.shape[0]
    out = np.zeros((num_dirs, num_thresh), dtype=np.int64)

    simplex_proj = [node_projections]
    edge_indices = np.asarray(graph.edge_indices)
    if edge_indices.size:
        edge_maxes = np.maximum(
            node_projections[edge_indices[:, 0]],
            node_projections[edge_indices[:, 1]],
        )
        simplex_proj.append(edge_maxes)
    else:
        simplex_proj.append(np.empty((0, num_dirs), dtype=node_projections.dtype))

    max_dim = max(graph.cells.keys()) if graph.cells else 1
    for dim in range(2, max_dim + 1):
        cells = graph.cells.get(dim, [])
        if len(cells) == 0:
            simplex_proj.append(np.empty((0, num_dirs), dtype=node_projections.dtype))
            continue
        cell_maxes = np.array(
            [np.max(node_projections[list(cell), :], axis=0) for cell in cells],
            dtype=node_projections.dtype,
        )
        simplex_proj.append(cell_maxes)

    for k, proj in enumerate(simplex_proj):
        if proj.shape[0] == 0:
            continue
        sign = 1 if (k % 2 == 0) else -1
        sorted_proj = np.sort(proj, axis=0)
        for i in range(num_dirs):
            counts = np.searchsorted(sorted_proj[:, i], thresholds, side="right")
            out[i, :] += sign * counts
    return out.astype(np.int32)


def time_naive_ect_shared(
    graph: EmbeddedComplex,
    directions_np: np.ndarray,
    thresholds: np.ndarray,
    repeats: int,
) -> float:
    _ = compute_naive_ect_output(graph, directions_np, thresholds)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = compute_naive_ect_output(graph, directions_np, thresholds)
    end = time.perf_counter()
    return (end - start) / max(1, repeats)


def time_external_dect_shared(
    points: np.ndarray,
    directions_np: np.ndarray,
    thresholds: np.ndarray,
    repeats: int,
    scale: float = 500.0,
    torch_device: str = "auto",
) -> float:
    device = _resolve_torch_device(torch, torch_device)
    x = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    v = torch.from_numpy(directions_np.T.astype(np.float32, copy=False)).to(device)
    radius = float(max(abs(thresholds[0]), abs(thresholds[-1])))
    resolution = int(len(thresholds))

    kwargs = {
        "v": v,
        "radius": radius,
        "resolution": resolution,
        "scale": float(scale),
    }
    kwargs["ect_fn"] = DECT_SCALED_SIGMOID

    return _time_callable(
        lambda: DECT_COMPUTE_ECT(x, **kwargs),
        repeats=repeats,
        torch_module=torch,
        device=device,
    )


def compute_external_dect_output(
    points: np.ndarray,
    directions_np: np.ndarray,
    thresholds: np.ndarray,
    scale: float = 500.0,
    torch_device: str = "auto",
) -> np.ndarray:
    device = _resolve_torch_device(torch, torch_device)
    x = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    v = torch.from_numpy(directions_np.T.astype(np.float32, copy=False)).to(device)
    radius = float(max(abs(thresholds[0]), abs(thresholds[-1])))
    resolution = int(len(thresholds))
    kwargs = {
        "v": v,
        "radius": radius,
        "resolution": resolution,
        "scale": float(scale),
        "ect_fn": DECT_SCALED_SIGMOID,
    }
    out = DECT_COMPUTE_ECT(x, **kwargs)
    if hasattr(out, "detach"):
        out_np = out.detach().cpu().numpy()
    else:
        out_np = np.asarray(out)
    out_np = np.asarray(out_np)
    if out_np.ndim == 3 and out_np.shape[0] == 1:
        out_np = out_np[0]
    if out_np.ndim != 2:
        raise ValueError(f"dect output must be 2D, got shape={out_np.shape}")
    num_dirs = directions_np.shape[0]
    num_thresh = thresholds.shape[0]
    if out_np.shape == (num_dirs, num_thresh):
        return out_np
    if out_np.shape == (num_thresh, num_dirs):
        return out_np.T
    raise ValueError(
        f"dect output shape {out_np.shape} incompatible with expected "
        f"({num_dirs}, {num_thresh})"
    )


def compute_baseline_ect_output(
    graph: EmbeddedComplex, directions_np: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    directions = Directions.from_vectors(directions_np)
    baseline = ECT(directions=directions, thresholds=thresholds)
    return np.asarray(baseline.calculate(graph))


def compute_pyect_wect_points(
    points: np.ndarray,
    directions_np: np.ndarray,
    num_heights: int,
    torch_device: str = "auto",
) -> np.ndarray:
    device = _resolve_torch_device(torch, torch_device)
    model, complex_data = _build_pyect_points_inputs(
        points, directions_np, num_heights, torch, PYECT_WECT, device
    )
    return model(complex_data).detach().cpu().numpy()


def compute_pyect_wect_grid2d(
    grid_size: int,
    directions_np: np.ndarray,
    num_heights: int,
    torch_device: str = "auto",
) -> np.ndarray:
    device = _resolve_torch_device(torch, torch_device)
    model, complex_data = _build_pyect_grid2d_inputs(
        grid_size, directions_np, num_heights, torch, PYECT_WECT, device
    )
    return model(complex_data).detach().cpu().numpy()


def time_pyect_wect_shared(
    points: np.ndarray,
    directions_np: np.ndarray,
    repeats: int,
    num_heights: int,
    torch_device: str = "auto",
) -> float:
    """
    Time pyECT's WECT with a constant weight function to emulate regular ECT on vertices.
    Returns average seconds over repeats (or None if pyect is unavailable).
    """
    device = _resolve_torch_device(torch, torch_device)
    model, complex_data = _build_pyect_points_inputs(
        points, directions_np, num_heights, torch, PYECT_WECT, device
    )

    return _time_callable(
        lambda: model(complex_data),
        repeats=repeats,
        torch_module=torch,
        device=device,
    )


def time_baseline_ect_graph(
    graph: EmbeddedComplex, num_dirs: int, num_thresh: int, repeats: int
) -> float:
    """
    Time our ECT on a provided complex (with vertices/edges/faces).
    """
    directions = Directions.uniform(num_dirs, dim=graph.dim)
    ect = ECT(directions=directions, num_thresh=num_thresh)
    graph.precompute_incidence_csr()

    # Warmup
    _ = ect.calculate(graph)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = ect.calculate(graph)
    end = time.perf_counter()
    return (end - start) / max(1, repeats)


def time_pyect_wect_grid2d_shared(
    grid_size: int,
    repeats: int,
    directions_np: np.ndarray,
    num_heights: int,
    torch_device: str = "auto",
) -> float:
    """
    Time pyECT's WECT on a 2D triangular grid complex (vertices, edges, faces).
    """
    device = _resolve_torch_device(torch, torch_device)
    model, complex_data = _build_pyect_grid2d_inputs(
        grid_size, directions_np, num_heights, torch, PYECT_WECT, device
    )

    return _time_callable(
        lambda: model(complex_data),
        repeats=repeats,
        torch_module=torch,
        device=device,
    )


def validate_equivalence(
    points: np.ndarray,
    num_dirs: int,
    num_heights: int,
    torch_device: str = "auto",
) -> float:
    """
    Validate that pyECT's WECT on a vertex-only, unit-weight complex matches our ECT
    when using identical directions and mapped thresholds.
    Returns max absolute difference (or None if pyect unavailable).
    """
    device = _resolve_torch_device(torch, torch_device)
    dirs_np = _shared_directions(num_dirs, points.shape[1], seed=0)
    model, complex_data = _build_pyect_points_inputs(
        points, dirs_np, num_heights, torch, PYECT_WECT, device
    )
    theirs = model(complex_data).detach().cpu().numpy()

    thresholds = _pyect_thresholds(points, num_heights)

    K = EmbeddedComplex()
    K.add_nodes_from([(i, points[i]) for i in range(points.shape[0])])
    our_dirs = Directions.from_vectors(dirs_np)
    ect = ECT(directions=our_dirs, thresholds=thresholds)
    ours = np.asarray(ect.calculate(K))

    if ours.shape != theirs.shape:
        return float("inf")
    return float(np.max(np.abs(ours.astype(np.float64) - theirs.astype(np.float64))))


def main():
    n_points = int(N_POINTS)
    dim = int(DIM)
    num_dirs_default = int(NUM_DIRS)
    num_thresh_default = int(NUM_THRESH)
    repeats = int(REPEATS)
    seed = int(SEED)
    dect_scale = float(DECT_SCALE)
    structure = str(STRUCTURE)
    torch_device = str(TORCH_DEVICE)
    resolutions = _resolve_resolutions(num_dirs_default, str(RESOLUTIONS))

    print("Benchmark: ECT package vs naive ECT vs dect package vs pyECT WECT")
    print(f"struct={structure}, dim={dim}, n_points={n_points}, repeats={repeats}")
    print(f"Resolutions (num_dirs=num_thresh): {resolutions}")

    baseline_times = []
    naive_times = []
    dect_times = []
    their_times = []
    final_plot_resolution = None
    final_plot_baseline = None
    final_plot_naive = None
    final_plot_dect = None
    final_plot_pyect = None

    if structure == "points":
        points = generate_points(n_points, dim, seed)
        struct_desc = f"vertex-only, n_points={n_points}"
    else:
        m = max(2, int(round(np.sqrt(max(4, n_points)))))
        graph = build_grid2d_complex(m)
        struct_desc = f"grid2d, m={m}, vertices={m * m}"

    print(f"Structure: {structure} ({struct_desc})")

    for r in resolutions:
        num_dirs = int(r)
        num_thresh = int(r)
        print(f"\nResolution r={r} (num_dirs=num_thresh={r}): benchmarking {structure}")
        if structure == "points":
            K = build_point_only_complex(points)
            dirs_np = _shared_directions(num_dirs, points.shape[1], seed)
            thresholds = _pyect_thresholds(points, num_thresh)
            baseline_avg = time_baseline_ect_shared(K, dirs_np, thresholds, repeats)
            naive_avg = time_naive_ect_shared(K, dirs_np, thresholds, repeats)
            dect_avg = time_external_dect_shared(
                points,
                dirs_np,
                thresholds,
                repeats,
                scale=dect_scale,
                torch_device=torch_device,
            )
            their_avg = time_pyect_wect_shared(
                points,
                dirs_np,
                repeats,
                num_thresh,
                torch_device=torch_device,
            )
            pyect_out = compute_pyect_wect_points(
                points, dirs_np, num_thresh, torch_device=torch_device
            )
            baseline_out = compute_baseline_ect_output(K, dirs_np, thresholds)
            equal_flag, max_diff = _compare_outputs(baseline_out, pyect_out)
            if r == resolutions[-1]:
                final_plot_naive = compute_naive_ect_output(K, dirs_np, thresholds)
                final_plot_dect = compute_external_dect_output(
                    points,
                    dirs_np,
                    thresholds,
                    scale=dect_scale,
                    torch_device=torch_device,
                )
            final_plot_resolution = r
            final_plot_baseline = baseline_out
            final_plot_pyect = pyect_out
        else:
            coords = graph.coord_matrix
            dirs_np = _shared_directions(num_dirs, 2, seed)
            thresholds = _pyect_thresholds(coords, num_thresh)
            baseline_avg = time_baseline_ect_shared(graph, dirs_np, thresholds, repeats)
            naive_avg = time_naive_ect_shared(graph, dirs_np, thresholds, repeats)
            dect_avg = time_external_dect_shared(
                coords,
                dirs_np,
                thresholds,
                repeats,
                scale=dect_scale,
                torch_device=torch_device,
            )
            their_avg = time_pyect_wect_grid2d_shared(
                m,
                repeats,
                dirs_np,
                num_thresh,
                torch_device=torch_device,
            )
            pyect_out = compute_pyect_wect_grid2d(
                m, dirs_np, num_thresh, torch_device=torch_device
            )
            baseline_out = compute_baseline_ect_output(graph, dirs_np, thresholds)
            equal_flag, max_diff = _compare_outputs(baseline_out, pyect_out)
            if r == resolutions[-1]:
                final_plot_naive = compute_naive_ect_output(graph, dirs_np, thresholds)
                final_plot_dect = compute_external_dect_output(
                    coords,
                    dirs_np,
                    thresholds,
                    scale=dect_scale,
                    torch_device=torch_device,
                )
            final_plot_resolution = r
            final_plot_baseline = baseline_out
            final_plot_pyect = pyect_out

        baseline_times.append(baseline_avg)
        naive_times.append(naive_avg)
        dect_times.append(dect_avg)
        their_times.append(their_avg)

        print(f"  Baseline ECT avg time per run:   {baseline_avg:.6f} s")
        print(f"  Naive ECT avg time per run:      {naive_avg:.6f} s")
        print(f"  dect package avg time per run:   {dect_avg:.6f} s")
        print(
            f"  Baseline vs pyECT exact equal: {equal_flag} (max_abs_diff={max_diff:.6e})"
        )
        print(f"  pyECT WECT avg time per run:    {their_avg:.6f} s")

    plt.figure(figsize=(6, 4))
    plt.loglog(
        resolutions,
        baseline_times,
        marker="o",
        label="ect package",
    )
    plt.loglog(
        resolutions,
        naive_times,
        marker="^",
        label="ect (naive implementation)",
    )
    plt.loglog(
        resolutions,
        dect_times,
        marker="d",
        label="dect package",
    )
    plt.loglog(
        resolutions,
        their_times,
        marker="s",
        label="pyECT package",
    )

    plt.xlabel("Resolution r (num_dirs = num_thresh = r)")
    plt.ylabel("Average time per run (s)")
    plt.title("ECT implementations across resolutions")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_ect_packages_resolutions.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    vmin = float(
        min(
            np.min(final_plot_baseline),
            np.min(final_plot_naive),
            np.min(final_plot_dect),
            np.min(final_plot_pyect),
        )
    )
    vmax = float(
        max(
            np.max(final_plot_baseline),
            np.max(final_plot_naive),
            np.max(final_plot_dect),
            np.max(final_plot_pyect),
        )
    )
    im0 = axes[0, 0].imshow(
        final_plot_baseline, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    axes[0, 0].set_title("Baseline ECT")
    axes[0, 0].set_ylabel("Direction index")

    axes[0, 1].imshow(
        final_plot_naive, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    naive_diff = float(
        np.max(
            np.abs(
                final_plot_naive.astype(np.float64)
                - final_plot_baseline.astype(np.float64)
            )
        )
    )
    axes[0, 1].set_title(f"Naive ECT (max|diff|={naive_diff:.3e})")

    axes[1, 0].imshow(
        final_plot_dect, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    dect_diff = float(
        np.max(
            np.abs(
                final_plot_dect.astype(np.float64)
                - final_plot_baseline.astype(np.float64)
            )
        )
    )
    axes[1, 0].set_title(f"dect package (max|diff|={dect_diff:.3e})")
    axes[1, 0].set_xlabel("Threshold index")
    axes[1, 0].set_ylabel("Direction index")

    axes[1, 1].imshow(
        final_plot_pyect, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    pyect_diff = float(
        np.max(
            np.abs(
                final_plot_pyect.astype(np.float64)
                - final_plot_baseline.astype(np.float64)
            )
        )
    )
    axes[1, 1].set_title(f"pyECT (max|diff|={pyect_diff:.3e})")
    axes[1, 1].set_xlabel("Threshold index")

    fig.suptitle(f"ECT outputs at r={final_plot_resolution} (num_dirs=num_thresh)")
    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig("benchmark_ect_packages_equality.png", dpi=150)
    plt.show()

    if structure == "points":
        diff = validate_equivalence(
            generate_points(min(256, n_points), dim, seed + 1),
            num_dirs_default,
            num_thresh_default,
            torch_device=torch_device,
        )
        print(f"Equivalence max |diff| (ours vs pyECT): {diff:.6e}")
    else:
        print("Equivalence check skipped for grid2d.")


if __name__ == "__main__":
    main()
