import numpy as np
from numba import prange, njit  # type: ignore[attr-defined]
from numba.typed import List
from typing import Optional

from .embed_complex import EmbeddedComplex
from .directions import Directions
from .results import ECTResult


def _thresholds_are_uniform(thresholds: np.ndarray) -> bool:
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be a 1-dimensional array")
    n = thresholds.size
    if n <= 1:
        return True
    diffs = np.diff(thresholds)
    first = diffs[0]
    if first == 0.0:
        return bool(np.all(diffs == 0.0))
    tol = 1e-12 * max(1.0, abs(first))
    return bool(np.all(np.abs(diffs - first) <= tol))


class ECT:
    r"""
    A class to calculate the Euler Characteristic Transform (ECT) from an input :class:`ect.embed_complex.EmbeddedComplex`,
    using a set of directions to project the complex onto and thresholds to filter the projections.

    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j`,
    where :math:`a_i` is the *i*th entry in ``self.thresholds`` and :math:`\omega_j` is the *j*th direction in ``self.directions``.

    Example:
        >>> from ect import ECT, EmbeddedComplex
        >>> from ect import EmbeddedGraph
        >>> complex = EmbeddedComplex()
        >>> complex.add_node(0, [0, 0])
        >>> complex.add_node(1, [1, 0])
        >>> complex.add_edge(0, 1)
        >>> ect = ECT(num_dirs=10, num_thresh=10) # chooses 10 uniform directions and 10 thresholds
        >>> result = ect.calculate(complex)
        >>> result.plot()
    """

    def __init__(
        self,
        directions: Optional[Directions] = None,
        num_dirs: Optional[int] = None,
        num_thresh: Optional[int] = None,
        bound_radius: Optional[float] = None,
        thresholds: Optional[np.ndarray] = None,
        dtype=np.int32,
    ):
        """Initialize ECT calculator with either a Directions object or sampling parameters

        Args:
            directions: Optional pre-configured Directions object
            num_dirs: Number of directions to sample (ignored if directions provided)
            num_thresh: Number of threshold values (required if directions not provided)
            bound_radius: Optional radius for bounding circle
            thresholds: Optional array of thresholds
            dtype: Data type for output array
        """
        self.directions = directions
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.bound_radius = bound_radius
        self.thresholds = thresholds
        self.dtype = dtype
        self._thresholds_validated = False
        if self.thresholds is not None:
            self.thresholds = np.asarray(self.thresholds, dtype=float)
            if self.thresholds.ndim != 1:
                raise ValueError("thresholds must be a 1-dimensional array")
            self._thresholds_validated = True
        if num_thresh is not None:
            self.is_uniform = True
        elif self.thresholds is not None:
            self.is_uniform = False
            if not _thresholds_are_uniform(self.thresholds):
                raise ValueError(
                    "thresholds must be uniform if num_thresh is not provided"
                )
        else:
            self.is_uniform = True

    def _ensure_directions(self, graph_dim, theta=None):
        """Ensures directions is a valid Directions object of correct dimension"""
        if self.directions is None:
            if self.num_dirs is None:
                raise ValueError("Either 'directions' or 'num_dirs' must be provided.")
            self.directions = Directions.uniform(self.num_dirs, dim=graph_dim)
        elif not isinstance(self.directions, Directions):
            # convert any array-like to Directions object
            try:
                self.directions = Directions.from_vectors(np.asarray(self.directions))
            except ValueError:
                raise ValueError(
                    "Invalid directions provided. "
                    "Must be a numpy array or a Directions object."
                )

        if theta is not None and graph_dim != 2:
            raise ValueError(
                "theta is only supported for 2D graphs. "
                "Use 'directions' or 'num_dirs' for higher dimensions."
            )

        if self.directions.dim != graph_dim:
            raise ValueError(
                "Dimension mismatch: directions dimension does not match graph dimension."
            )

    def _ensure_thresholds(self, graph, override_bound_radius=None):
        """Ensures thresholds is a valid 1-dimensional ndarray."""
        if self.thresholds is None or override_bound_radius is not None:
            if self.num_thresh is None:
                raise ValueError(
                    "Either 'thresholds' or 'num_thresh' must be provided."
                )
            # priority: override > bound_radius > graph radius
            radius = (
                override_bound_radius
                or self.bound_radius
                or graph.get_bounding_radius()
            )
            self.thresholds = np.linspace(-radius, radius, self.num_thresh, dtype=float)
            self.is_uniform = True
            self._thresholds_validated = True
        else:
            if not self._thresholds_validated:
                self.thresholds = np.asarray(self.thresholds, dtype=float)
                if self.thresholds.ndim != 1:
                    raise ValueError("thresholds must be a 1-dimensional array")
                self._thresholds_validated = True

    def calculate(
        self,
        graph: EmbeddedComplex,
        theta: Optional[float] = None,
        override_bound_radius: Optional[float] = None,
    ) -> ECTResult:
        self._ensure_directions(graph.dim, theta)
        self._ensure_thresholds(graph, override_bound_radius)
        directions = (
            self.directions if theta is None else Directions.from_angles([theta])
        )
        assert self.thresholds is not None
        ect_matrix = self._compute_ect(graph, directions, self.thresholds, self.dtype)

        return ECTResult(ect_matrix, directions, self.thresholds)

    def _compute_ect(
        self, graph, directions, thresholds: np.ndarray, dtype=np.int32
    ) -> np.ndarray:
        cell_vertex_pointers, cell_vertex_indices_flat, cell_euler_signs, N = (
            graph._build_incidence_csr()
        )
        thresholds = np.asarray(thresholds, dtype=np.float64)

        V = directions.vectors
        X = graph.coord_matrix
        H = X @ V.T  # (N, m)
        H_T = np.ascontiguousarray(H.T)  # (m, N) for contiguous per-direction rows

        is_uniform = bool(self.is_uniform) and thresholds[0] != thresholds[-1]
        if is_uniform:
            out64 = _ect_all_dirs_uniform(
                H_T,
                cell_vertex_pointers,
                cell_vertex_indices_flat,
                cell_euler_signs,
                thresholds,
                N,
            )
        else:
            out64 = _ect_all_dirs_search(
                H_T,
                cell_vertex_pointers,
                cell_vertex_indices_flat,
                cell_euler_signs,
                thresholds,
                N,
            )
        if dtype == np.int32:
            return out64.astype(np.int32)
        return out64

    def _compute_simplex_projections(self, graph: EmbeddedComplex, directions):
        """Compute projections of each k-cell for all dimensions"""
        simplex_projections = List()
        node_projections = np.matmul(graph.coord_matrix, directions.vectors.T)
        num_dirs = node_projections.shape[1]

        simplex_projections.append(node_projections)

        all_cells = {
            0: [(i,) for i in range(len(graph.node_list))],
            1: [tuple(edge) for edge in graph.edge_indices]
            if graph.edge_indices.size
            else [],
            **graph.cells,
        }

        max_dim = max(all_cells.keys()) if all_cells else 0
        for dim in range(1, max_dim + 1):
            cells = all_cells.get(dim, [])
            cell_projections = (
                np.array(
                    [np.max(node_projections[list(cell), :], axis=0) for cell in cells]
                )
                if cells
                else np.empty((0, num_dirs))
            )
            simplex_projections.append(cell_projections)

        return simplex_projections


@njit(cache=True, parallel=True)
def _ect_all_dirs_uniform(
    heights_by_direction,
    cell_vertex_pointers,
    cell_vertex_indices_flat,
    cell_euler_signs,
    threshold_values,
    num_vertices,
):
    num_directions = heights_by_direction.shape[0]
    num_thresholds = threshold_values.shape[0]
    t_min = threshold_values[0] if num_thresholds > 0 else 0.0
    t_max = threshold_values[-1] if num_thresholds > 0 else 0.0
    span = t_max - t_min
    inv_span = 1.0 / span
    n_minus_1 = num_thresholds - 1

    ect_values = np.empty((num_directions, num_thresholds), dtype=np.int64)

    for dir_idx in prange(num_directions):
        heights = heights_by_direction[dir_idx]

        diff = np.zeros(num_thresholds, dtype=np.int64)
        vertex_thresh_index = np.empty(num_vertices, dtype=np.int64)

        for v in range(num_vertices):
            h = heights[v]
            u = (h - t_min) * inv_span
            idx = int(np.ceil(u * n_minus_1))
            if idx < 0:
                idx = 0
            elif idx >= num_thresholds:
                idx = num_thresholds

            vertex_thresh_index[v] = idx
            if idx < num_thresholds:
                diff[idx] += 1

        num_cells = cell_vertex_pointers.shape[0] - 1

        for cell_idx in range(num_cells):
            start = cell_vertex_pointers[cell_idx]
            end = cell_vertex_pointers[cell_idx + 1]

            entrance_idx = -1
            for k in range(start, end):
                v = cell_vertex_indices_flat[k]
                t_idx = vertex_thresh_index[v]
                if t_idx > entrance_idx:
                    entrance_idx = t_idx

            if 0 <= entrance_idx < num_thresholds:
                diff[entrance_idx] += cell_euler_signs[cell_idx]

        running = 0
        for j in range(num_thresholds):
            running += diff[j]
            ect_values[dir_idx, j] = running

    return ect_values


@njit(cache=True, parallel=True)
def _ect_all_dirs_search(
    heights_by_direction,
    cell_vertex_pointers,
    cell_vertex_indices_flat,
    cell_euler_signs,
    threshold_values,
    num_vertices,
):
    num_directions = heights_by_direction.shape[0]
    num_thresholds = threshold_values.shape[0]

    ect_values = np.empty((num_directions, num_thresholds), dtype=np.int64)

    for dir_idx in prange(num_directions):
        heights = heights_by_direction[dir_idx]

        diff = np.zeros(num_thresholds, dtype=np.int64)
        vertex_thresh_index = np.empty(num_vertices, dtype=np.int64)

        for v in range(num_vertices):
            h = heights[v]

            left = 0
            right = num_thresholds
            while left < right:
                mid = (left + right) // 2
                if threshold_values[mid] >= h:
                    right = mid
                else:
                    left = mid + 1
            idx = left

            vertex_thresh_index[v] = idx
            if idx < num_thresholds:
                diff[idx] += 1

        num_cells = cell_vertex_pointers.shape[0] - 1

        for cell_idx in range(num_cells):
            start = cell_vertex_pointers[cell_idx]
            end = cell_vertex_pointers[cell_idx + 1]

            entrance_idx = -1
            for k in range(start, end):
                v = cell_vertex_indices_flat[k]
                t_idx = vertex_thresh_index[v]
                if t_idx > entrance_idx:
                    entrance_idx = t_idx

            if 0 <= entrance_idx < num_thresholds:
                diff[entrance_idx] += cell_euler_signs[cell_idx]

        running = 0
        for j in range(num_thresholds):
            running += diff[j]
            ect_values[dir_idx, j] = running

    return ect_values
