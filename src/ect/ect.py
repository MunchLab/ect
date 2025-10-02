import numpy as np
from numba import prange, njit
from numba.typed import List
from typing import Optional

from .embed_complex import EmbeddedComplex
from .directions import Directions
from .results import ECTResult


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input :any:`EmbeddedComplex`,
    using a set of directions to project the complex onto and thresholds to filter the projections.

    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j`
    where :math:`a_i` is the ith entry in ``self.thresholds``, and :math:`\omega_j` is the jth entry in ``self.directions``.

    Attributes
    ----------
    num_dirs : int
        The number of directions to consider in the matrix.
    num_thresh : int
        The number of thresholds to consider in the matrix.
    directions : Directions
        The directions to consider for projection.
    bound_radius : float
        Either ``None``, or a positive radius of the bounding circle.

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
                "Theta must be provided for 2D graphs. "
                "Use 'directions' or 'num_dirs' to specify directions."
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
        else:
            # validate and convert existing thresholds
            self.thresholds = np.asarray(self.thresholds, dtype=float)
            if self.thresholds.ndim != 1:
                raise ValueError("thresholds must be a 1-dimensional array")

    def calculate(
        self,
        graph: EmbeddedComplex,
        theta: float = None,
        override_bound_radius: float = None,
    ):
        self._ensure_directions(graph.dim, theta)
        self._ensure_thresholds(graph, override_bound_radius)
        directions = (
            self.directions if theta is None else Directions.from_angles([theta])
        )
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
        H = X @ V if V.shape[0] == X.shape[1] else X @ V.T  # (N, m)
        H_T = np.ascontiguousarray(H.T)  # (m, N) for contiguous per-direction rows

        out64 = _ect_all_dirs(
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
def _ect_all_dirs(
    heights_by_direction,  # shape (num_directions, num_vertices)
    cell_vertex_pointers,  # shape (num_cells + 1,)
    cell_vertex_indices_flat,  # concatenated vertex indices for all cells
    cell_euler_signs,  # per-cell sign: (+1) for even-dim, (-1) for odd-dim
    threshold_values,  # shape (num_thresholds,), assumed nondecreasing
    num_vertices,
):
    """
    Calculate the Euler Characteristic Transform (ECT) for a given direction and thresholds.

    Args:
        heights_by_direction: The heights of the vertices for each direction
        cell_vertex_pointers: The pointers to the vertices for each cell
        cell_vertex_indices_flat: The indices of the vertices for each cell
        cell_euler_signs: The signs of the cells
        threshold_values: The thresholds to calculate the ECT for
        num_vertices: The number of vertices in the graph
    """
    num_directions = heights_by_direction.shape[0]
    num_thresholds = threshold_values.shape[0]
    ect_values = np.empty((num_directions, num_thresholds), dtype=np.int64)

    for dir_idx in prange(num_directions):
        heights = heights_by_direction[dir_idx]

        sort_order = np.argsort(heights)

        # calculate what position each vertex is in the sorted heights starting from 1 (the rank)
        vertex_rank_1based = np.empty(num_vertices, dtype=np.int32)
        for rnk in range(num_vertices):
            vertex_index = sort_order[rnk]
            vertex_rank_1based[vertex_index] = rnk + 1

        # euler char can only jump at each vertex value
        jump_amount = np.zeros(num_vertices + 1, dtype=np.int64)

        # 0-cells add +1 at their entrance ranks
        for v in range(num_vertices):
            rank_v = vertex_rank_1based[v]
            jump_amount[rank_v] += 1

        # each pair of pointers defines a cell, so we iterate over them
        num_cells = cell_vertex_pointers.shape[0] - 1
        for cell_idx in range(num_cells):
            start = cell_vertex_pointers[cell_idx]
            end = cell_vertex_pointers[cell_idx + 1]
            # cells come in when the highest vertex enters
            entrance_rank = 0
            for k in range(start, end):
                v = cell_vertex_indices_flat[k]
                rnk = vertex_rank_1based[v]
                if rnk > entrance_rank:
                    entrance_rank = rnk
            # record at what rank the cell enters and how much the euler char changes
            jump_amount[entrance_rank] += cell_euler_signs[cell_idx]

        # calculate euler char at the moment each vertex enters
        euler_prefix = np.empty(num_vertices + 1, dtype=np.int64)
        running_sum = 0
        for r in range(num_vertices + 1):
            running_sum += jump_amount[r]
            euler_prefix[r] = running_sum

        # now find euler char at each threshold wrt the sorted heights
        sorted_heights = heights[sort_order]
        rank_cursor = 0  # equals r(t) = # { i : h_i <= t }
        for thresh_idx in range(num_thresholds):
            t = threshold_values[thresh_idx]
            while rank_cursor < num_vertices and sorted_heights[rank_cursor] <= t:
                rank_cursor += 1
            ect_values[dir_idx, thresh_idx] = euler_prefix[rank_cursor]

    return ect_values
