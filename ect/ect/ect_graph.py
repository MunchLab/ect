from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from typing import Optional, Union

from ect.embed_cw import EmbeddedCW
from ect.embed_graph import EmbeddedGraph
from ect.directions import Directions
from ect.results import ECTResult



class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input :any:`EmbeddedGraph` or :any:`EmbeddedCW`.

    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j` where :math:`a_i` is the ith entry in ``self.threshes``, and :math:`\omega_j` is the ith entry in ``self.thetas``.

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
            self.num_thresh = num_thresh or 360

        self.num_thresh = num_thresh
        self.bound_radius = None
        self.threshes = None
        if bound_radius is not None:
            self.set_bounding_radius(bound_radius)

    def _ensure_valid_directions(self, calculation_method):
        """Ensures directions match graph dimension, creating if needed"""
        @wraps(calculation_method)
        def wrapper(ect_instance, graph, *args, **kwargs):
            if ect_instance.directions is None or ect_instance.directions.dim != graph.dim:
                ect_instance.directions = Directions.uniform(ect_instance.num_dirs, dim=graph.dim)
            return calculation_method(ect_instance, graph, *args, **kwargs)
        return wrapper

    def set_bounding_radius(self, radius: Optional[float]):
        """Sets the bounding radius and updates thresholds"""
        if radius is not None and radius <= 0:
            raise ValueError(f'Bounding radius must be positive, got {radius}')
        
        self.bound_radius = radius
        if radius is not None:
            self.threshes = np.linspace(-radius, radius, self.num_thresh)

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
        coords = graph.coord_matrix
        edges = graph.edge_indices

        if theta is None:
            node_projections = self._compute_node_projections(coords, self.directions)
        else:
            node_projections = self._compute_node_projections(coords, Directions.from_angles([theta]))

        simplex_projections = self._compute_simplex_projections(node_projections, simplex)

        face_maxes = np.empty((0, self.num_dirs))
        if isinstance(graph, EmbeddedCW) and len(graph.faces) > 0:
            node_to_index = {n: i for i, n in enumerate(graph.node_list)}
            face_indices = [[node_to_index[v] for v in face] for face in graph.faces]
            face_maxes = np.array([np.max(node_projections[face, :], axis=0) for face in face_indices])

        return self._calculate_euler_chars(node_projections, edge_maxes, face_maxes, thresholds)

    def _compute_node_projections(self, coords, directions):
        """Compute inner products of coordinates with directions"""
        return np.matmul(coords, directions.vectors.T)
    
    def _compute_simplex_projections(vertex_projections, simplices):
        """For all k ≥ 0, compute max projections of each k-simplex"""
        max_projections = []
        for k in range(len(simplices)):
            if k == 0:
                max_projections.append(vertex_projections.copy())
            else:
                k_simplices = simplices[k]
                k_proj = np.array([
                    np.max(vertex_projections[s, :], axis=0)
                    for s in k_simplices
                ])
                max_projections.append(k_proj)
        return max_projections
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _sort_projections(simplex_projections_list, thresholds):
        """Sort projections and count occurrences of each threshold"""
        num_dir = simplex_projections_list[0].shape[1]
        num_thresh = thresholds.shape[0]
        result = np.empty((num_dir, num_thresh), dtype=np.int32)

        sorted_projections = [np.sort(proj, axis=0) for proj in simplex_projections_list]

        for j in prange(num_thresh):
            thresh = thresholds[j]
            for i in range(num_dir):
                simplex_counts_list = []
                for k in range(len(sorted_projections)):
                    projs = sorted_projections[k][:, i]
                    simplex_counts_list.append(np.searchsorted(projs, thresh, side='right'))
                result[i, j] = self._calculate_euler_chars(simplex_counts_list)
        return result
    
    def _calculate_euler_chars(self, simplex_counts_list):
        """Calculate Euler characteristics from sorted counts"""
        chi = 0
        for k in range(len(simplex_counts_list)):
            chi += simplex_counts_list[k]
        return chi
    

    def calculate_euler_chars(self, projections, edge_maxes, face_maxes, thresholds):
        """Calculate Euler characteristics and wrap in ECTResult"""
        result = ECT._calculate_euler_chars_numba(
            projections, edge_maxes, face_maxes, thresholds)
        return ECTResult(result, self.directions, self.thresholds)


    # @_ensure_valid_directions
    # def calculate_ecc(self, graph, theta, bound_radius=None, return_counts=False):
    #     """Calculate ECC - directions are validated by decorator"""
    #     r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)

    #     r_threshes = np.array(r_threshes)

    #     # Sort vertices and edges based on projection
    #     v_list, g = graph.sort_vertices(theta, return_g=True)
    #     g_list = np.array([g[v] for v in v_list])
    #     sorted_g_list = np.sort(g_list)

    #     e_list, g_e = graph.sort_edges(theta, return_g=True)
    #     g_e_list = np.array([g_e[e] for e in e_list])
    #     sorted_g_e_list = np.sort(g_e_list)

    #     vertex_count = np.searchsorted(sorted_g_list, r_threshes, side='right')
    #     edge_count = np.searchsorted(sorted_g_e_list, r_threshes, side='right')

    #     if isinstance(graph, EmbeddedCW):
    #         f_list, g_f = graph.sort_faces(theta, return_g=True)
    #         g_f_list = np.array([g_f[f] for f in f_list])
    #         sorted_g_f_list = np.sort(g_f_list)
    #         face_count = np.searchsorted(
    #             sorted_g_f_list, r_threshes, side='right')
    #     else:
    #         face_count = np.zeros_like(r_threshes, dtype=np.int32)

    #     ecc = vertex_count - edge_count + face_count

    #     if return_counts:
    #         return ecc, vertex_count, edge_count, face_count
    #     else:
    #         return ecc


