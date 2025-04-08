import numpy as np
from numba import prange, njit
from typing import Optional, Union

from .embed_cw import EmbeddedCW
from .embed_graph import EmbeddedGraph
from .directions import Directions
from .results import ECTResult


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input :any:`EmbeddedGraph` or :any:`EmbeddedCW`.

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
                raise ValueError(
                    "Either 'directions' or 'num_dirs' must be provided.")
            self.directions = Directions.uniform(self.num_dirs, dim=graph_dim)
        elif isinstance(self.directions, list):
            # if list of vectors, convert to Directions object
            self.directions = np.array(self.directions)
            self.directions = Directions.from_vectors(self.directions)
        elif not isinstance(self.directions, Directions):
            raise TypeError(
                "directions must be a Directions object, ndarray, or list of vectors."
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

        # determine if we need to generate thresholds
        if self.thresholds is None or override_bound_radius is not None:
            if self.num_thresh is None:
                raise ValueError(
                    "Either 'thresholds' or 'num_thresh' must be provided."
                )
            # determine the radius based on priority
            if override_bound_radius is not None:
                radius = override_bound_radius
            elif self.bound_radius is not None:
                radius = self.bound_radius
            else:
                radius = graph.get_bounding_radius()

            self.thresholds = np.linspace(-radius, radius, self.num_thresh)
        else:
            # validate existing thresholds are valid
            if not isinstance(self.thresholds, np.ndarray):
                raise TypeError("thresholds must be a numpy ndarray")

            if self.thresholds.ndim != 1:
                raise ValueError("thresholds must be a 1-dimensional array")

            self.thresholds = self.thresholds.astype(float)

    def calculate(
        self,
        graph: Union[EmbeddedGraph, EmbeddedCW],
        theta: Optional[float] = None,
        override_bound_radius: Optional[float] = None,
    ):
        """Calculate Euler Characteristic Transform (ECT) for a given graph and direction theta

        Args:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph to calculate the ECT for.
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to calculate the ECT.
            override_bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a positive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.
        """
        self._ensure_directions(graph.dim, theta)
        self._ensure_thresholds(graph, override_bound_radius)

        # override with theta if provided
        directions = (
            self.directions if theta is None else Directions.from_angles([
                                                                         theta])
        )

        simplex_projections = self._compute_simplex_projections(
            graph, directions)

        ect_matrix = self._compute_directional_transform(
            simplex_projections, self.thresholds, self.shape_descriptor, self.dtype
        )

        return ECTResult(ect_matrix, directions, self.thresholds)

    def _compute_node_projections(self, coords, directions):
        """Compute inner products of coordinates with directions"""
        return np.matmul(coords, directions.vectors.T)

    def _compute_simplex_projections(
        self, graph: Union[EmbeddedGraph, EmbeddedCW], directions
    ):
        """Compute projections of each simplex (vertices, edges, faces)"""
        simplex_projections = []
        node_projections = self._compute_node_projections(
            graph.coord_matrix, directions
        )
        edge_maxes = np.maximum(
            node_projections[graph.edge_indices[:, 0]],
            node_projections[graph.edge_indices[:, 1]],
        )

        simplex_projections.append(node_projections)
        simplex_projections.append(edge_maxes)

        if isinstance(graph, EmbeddedCW) and len(graph.faces) > 0:
            node_to_index = {n: i for i, n in enumerate(graph.node_list)}
            face_indices = [[node_to_index[v] for v in face]
                            for face in graph.faces]
            face_maxes = np.array(
                [np.max(node_projections[face, :], axis=0)
                 for face in face_indices]
            )
            simplex_projections.append(face_maxes)

        return simplex_projections

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_directional_transform(
        simplex_projections_list, thresholds, shape_descriptor, dtype=np.int32
    ):
        """Compute ECT by counting simplices below each threshold

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

        sorted_projections = []
        for proj in simplex_projections_list:
            sorted_proj = np.empty_like(proj)
            for i in prange(num_dir):
                sorted_proj[:, i] = np.sort(proj[:, i])
            sorted_projections.append(sorted_proj)

        for j in prange(num_thresh):
            thresh = thresholds[j]
            for i in range(num_dir):
                simplex_counts_list = []
                for k in range(len(sorted_projections)):
                    projs = sorted_projections[k][:, i]
                    simplex_counts_list.append(
                        np.searchsorted(projs, thresh, side="right")
                    )
                result[i, j] = shape_descriptor(simplex_counts_list)
        return result

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def shape_descriptor(simplex_counts_list):
        """Calculate shape descriptor from simplex counts (Euler characteristic)"""
        chi = 0
        for k in range(len(simplex_counts_list)):
            chi += (-1) ** k * simplex_counts_list[k]
        return chi
