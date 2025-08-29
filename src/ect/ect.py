import numpy as np
from numba import prange, njit
from numba.typed import List
from typing import Optional

from .embed_complex import EmbeddedComplex
from .directions import Directions
from .results import ECTResult


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input :any:`EmbeddedComplex`.

    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j` where :math:`a_i` is the ith entry in ``self.thresholds``, and :math:`\omega_j` is the ith entry in ``self.thetas``.

    Attributes
        num_dirs (int):
            The number of directions to consider in the matrix.
        num_thresh (int):
            The number of thresholds to consider in the matrix.
        directions (Directions):
            The directions to consider for projection.
        bound_radius (float):
            Either ``None``, or a positive radius of the bounding circle.
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
        theta: Optional[float] = None,
        override_bound_radius: Optional[float] = None,
    ):
        """Calculate Euler Characteristic Transform (ECT) for a given graph and direction theta

        Args:
            graph (EmbeddedComplex):
                The input complex to calculate the ECT for.
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to calculate the ECT.
            override_bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a positive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.
        """
        self._ensure_directions(graph.dim, theta)
        self._ensure_thresholds(graph, override_bound_radius)

        # override with theta if provided
        directions = (
            self.directions if theta is None else Directions.from_angles([theta])
        )

        simplex_projections = self._compute_simplex_projections(graph, directions)

        ect_matrix = self._compute_directional_transform(
            simplex_projections, self.thresholds, self.dtype
        )

        return ECTResult(ect_matrix, directions, self.thresholds)

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

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_directional_transform(
        simplex_projections_list, thresholds, dtype=np.int32
    ):
        """Compute ECT by counting simplices below each threshold - VECTORIZED VERSION

        Args:
            simplex_projections_list: List of arrays containing projections for each simplex type
                [vertex_projections, edge_projections, face_projections]
            thresholds: Array of threshold values to compute ECT at
            dtype: Data type for output array (default: np.int32)

        Returns:
            Array of shape (num_directions, num_thresholds) containing Euler characteristics
        """
        num_dir = simplex_projections_list[0].shape[1]
        num_thresh = thresholds.shape[0]
        result = np.empty((num_dir, num_thresh), dtype=dtype)

        sorted_projections = List()
        for proj in simplex_projections_list:
            sorted_proj = np.empty_like(proj)
            for i in prange(num_dir):
                sorted_proj[:, i] = np.sort(proj[:, i])
            sorted_projections.append(sorted_proj)

        for i in prange(num_dir):
            chi = np.zeros(num_thresh, dtype=dtype)
            for k in range(len(sorted_projections)):
                projs = sorted_projections[k][:, i]

                count = np.searchsorted(projs, thresholds, side="right")

                sign = -1 if k % 2 else 1
                chi += sign * count
            result[i] = chi
        return result
