from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
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
        bound_radius (int):
            Either ``None``, or a positive radius of the bounding circle.
    """

    def __init__(self,
                 directions: Optional[Directions] = None,
                 *,
                 num_dirs: Optional[int] = None,
                 num_thresh: Optional[int] = None,
                 bound_radius: Optional[float] = None):
        """Initialize ECT calculator with either a Directions object or sampling parameters
        
        Args:
            directions: Optional pre-configured Directions object
            num_dirs: Number of directions to sample (ignored if directions provided)
            num_thresh: Number of threshold values (required if directions not provided)
            bound_radius: Optional radius for bounding circle
        """
        if directions is not None:
            self.directions = directions
            self.num_dirs = len(directions)
            if num_thresh is None:
                self.num_thresh = self.num_dirs
        else:
            self.directions = None
            self.num_dirs = num_dirs or 360
            self.num_thresh = num_thresh or self.num_dirs

        self.bound_radius = None
        self.thresholds = None
        if bound_radius is not None:
            self.set_bounding_radius(bound_radius)

    @staticmethod
    def _ensure_valid_directions(calculation_method):
        """
        Decorator to ensure directions match graph dimension.
        Reinitializes directions if dimensions don't match.
        """
        @wraps(calculation_method)
        def wrapper(ect_instance, graph, *args, **kwargs):
            if ect_instance.directions is None:
                ect_instance.directions = Directions.uniform(
                    ect_instance.num_dirs, dim=graph.dim)
            elif ect_instance.directions.dim != graph.dim:
                ect_instance.directions = Directions.uniform(
                    ect_instance.num_dirs, dim=graph.dim)

            return calculation_method(ect_instance, graph, *args, **kwargs)
        return wrapper

    def set_bounding_radius(self, radius: Optional[float]):
        """Sets the bounding radius and updates thresholds"""
        if radius is not None and radius <= 0:
            raise ValueError(f'Bounding radius must be positive, got {radius}')
        
        self.bound_radius = radius
        if radius is not None:
            self.thresholds = np.linspace(-radius, radius, self.num_thresh)

    def get_thresholds(self, graph: Union[EmbeddedGraph, EmbeddedCW], override_radius: Optional[float] = None):
        """Gets thresholds based on priority: override_radius > instance radius > graph radius"""
        if override_radius is not None:
            if override_radius <= 0:
                raise ValueError(f'Bounding radius must be positive, got {override_radius}')
            return override_radius, np.linspace(-override_radius, override_radius, self.num_thresh)
            
        if self.bound_radius is not None:
            return self.bound_radius, self.thresholds
            
        graph_radius = graph.get_bounding_radius()
        return graph_radius, np.linspace(-graph_radius, graph_radius, self.num_thresh)

    @_ensure_valid_directions
    def calculate(self, graph: Union[EmbeddedGraph, EmbeddedCW], theta=None, bound_radius=None):
        """Calculate Euler Characteristic Transform (ECT) for a given graph and direction theta

        Args:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph to calculate the ECT for.
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to calculate the ECT.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.
        """
        radius, thresholds = self.get_thresholds(graph, bound_radius)

        directions = self.directions if theta is None else Directions.from_angles([theta])

        simplex_projections = self._compute_simplex_projections(graph, directions)

        ect = self._compute_directional_transform(simplex_projections, thresholds)

        return ECTResult(ect, directions, thresholds)

    def _compute_node_projections(self, coords, directions):
        """Compute inner products of coordinates with directions"""
        return np.matmul(coords, directions.vectors.T)
    
    def _compute_simplex_projections(self, graph: Union[EmbeddedGraph, EmbeddedCW], directions):
        """Compute max projections of each simplex (vertices, edges, faces)"""
        simplex_projections = []
        node_projections = self._compute_node_projections(graph.coord_matrix, directions)
        edge_maxes = np.maximum(node_projections[graph.edge_indices[:, 0]], 
                               node_projections[graph.edge_indices[:, 1]])
        
        simplex_projections.append(node_projections)
        simplex_projections.append(edge_maxes)
        
        if isinstance(graph, EmbeddedCW) and len(graph.faces) > 0:
            node_to_index = {n: i for i, n in enumerate(graph.node_list)}
            face_indices = [[node_to_index[v] for v in face] for face in graph.faces]
            face_maxes = np.array([np.max(node_projections[face, :], axis=0) 
                                  for face in face_indices])
            simplex_projections.append(face_maxes)
        
        return simplex_projections


    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _compute_directional_transform(simplex_projections_list, thresholds):
        """Compute ECT by counting simplices below each threshold
        
        Args:
            simplex_projections_list: List of arrays containing projections for each simplex type
                [vertex_projections, edge_projections, face_projections]
            thresholds: Array of threshold values to compute ECT at
        
        Returns:
            Array of shape (num_directions, num_thresholds) containing Euler characteristics
        """
        num_dir = simplex_projections_list[0].shape[1]
        num_thresh = thresholds.shape[0]
        result = np.empty((num_dir, num_thresh), dtype=np.int32)

        sorted_projections = []
        for proj in simplex_projections_list:
            sorted_proj = np.empty_like(proj)
            for i in prange(num_dir):  
                sorted_proj[:, i] = np.sort(proj[:, i])
            sorted_projections.append(sorted_proj)

        def compute_shape_descriptor(simplex_counts_list):
            """Calculate shape descriptor from simplex counts (Euler characteristic)"""
            chi = 0
            for k in range(len(simplex_counts_list)):
                chi += (-1)**k * simplex_counts_list[k]
            return chi

        for j in prange(num_thresh):
            thresh = thresholds[j]
            for i in range(num_dir):
                simplex_counts_list = []
                for k in range(len(sorted_projections)):
                    projs = sorted_projections[k][:, i]
                    simplex_counts_list.append(np.searchsorted(projs, thresh, side='right'))
                result[i, j] = compute_shape_descriptor(simplex_counts_list)
        return result
    


