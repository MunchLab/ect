import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from typing import Optional, Union

from ect.embed_cw import EmbeddedCW
from ect.embed_graph import EmbeddedGraph
from ect.directions import Directions
from ect.results import ECTResult
from functools import wraps


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input :any:`EmbeddedGraph` or :any:`EmbeddedCW`.

    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j` where :math:`a_i` is the ith entry in ``self.threshes``, and :math:`\omega_j` is the ith entry in ``self.thetas``.

    Attributes
        num_dirs (int):
            The number of directions to consider in the matrix.
        num_thresh (int):
            The number of thresholds to consider in the matrix.
        bound_radius (int):
            Either ``None``, or a positive radius of the bounding circle.
        ect_matrix (np.array):
            The matrix to store the ECT.
        sect_matrix (np.array):
            The matrix to store the SECT.

    """

    def __init__(self,
                 num_dirs: Optional[int] = None,
                 num_thresh: int = 360,
                 directions: Optional[Directions] = None,
                 bound_radius: Optional[float] = None):
        """
        Initialize ECT calculator.

        Args:
            num_dirs: Number of directions for uniform sampling (ignored if directions provided)
            num_thresh: Number of threshold values
            directions: Optional Directions object for custom sampling
            bound_radius: Optional radius for bounding circle
        """
        if directions is not None:
            self.directions = directions
            self.num_dirs = len(directions)
        else:
            self.num_dirs = num_dirs or 360
            self.directions = None

        self.num_thresh = num_thresh
        self.set_bounding_radius(bound_radius)

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

    def set_bounding_radius(self, bound_radius):
        """
        Manually sets the radius of the bounding circle centered at the origin for the ECT object.

        Parameters:
            bound_radius (int): 
                Either None, or a positive radius of the bounding circle.
        """
        self.bound_radius = bound_radius

        if self.bound_radius is None:
            self.threshes = None
        else:
            self.threshes = np.linspace(-bound_radius,
                                        bound_radius, self.num_thresh)

    def get_radius_and_thresh(self, G, bound_radius):
        """
        An internally used function to get the bounding radius and thresholds for the ECT calculation.

        Parameters:
            G (EmbeddedGraph / EmbeddedCW):
                The input graph to calculate the ECT for.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.

        Returns:
            float, np.array
                The bounding radius and the thresholds for the ECT calculation.

        """
       # Either use the global radius and the set self.threshes; or use the tight bounding box and calculate
        # the thresholds from that.
        if bound_radius is None:
            # First try to get the internally stored bounding radius
            if self.bound_radius is not None:
                r = self.bound_radius
                r_threshes = self.threshes

            # If the bounding radius is not set, use the global bounding radius
            else:
                r = G.get_bounding_radius()
                r_threshes = np.linspace(-r, r, self.num_thresh)

        else:
            # The user wants to use a different bounding radius
            if bound_radius <= 0:
                raise ValueError(
                    f'Bounding radius given was {bound_radius}, but must be a positive number.')
            r = bound_radius
            r_threshes = np.linspace(-r, r, self.num_thresh)

        return r, r_threshes

    @_ensure_valid_directions
    def calculate(self, graph, theta=None, bound_radius=None, return_counts=False):
        """Calculate ECT - directions are validated by decorator"""
        # Initialize directions if needed
        if self.directions is None:
            self.directions = Directions.uniform(
                self.num_dirs, dim=graph.dim)

        r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)
        coords = graph.coord_matrix
        edges = graph.edge_indices

        if theta is None:
            vertex_projections = np.matmul(coords, self.directions.vectors.T)
        else:
            vertex_projections = np.matmul(
                coords, Directions.from_angles([theta]).vectors.T)

        edge_maxes = np.maximum(
            vertex_projections[edges[:, 0]], vertex_projections[edges[:, 1]])

        face_maxes = np.empty((0, self.num_dirs))
        if isinstance(graph, EmbeddedCW) and len(graph.faces) > 0:
            node_to_index = {n: i for i, n in enumerate(graph.node_list)}
            face_indices = [
                [node_to_index[v] for v in face]
                for face in graph.faces
            ]
            face_maxes = np.array([
                np.max(vertex_projections[face, :], axis=0)
                for face in face_indices
            ])

        return self.calculate_euler_chars(
            vertex_projections, edge_maxes, face_maxes, r_threshes
        )

    @_ensure_valid_directions
    def calculate_ecc(self, graph, theta, bound_radius=None, return_counts=False):
        """Calculate ECC - directions are validated by decorator"""
        r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)

        r_threshes = np.array(r_threshes)

        # Sort vertices and edges based on projection
        v_list, g = graph.sort_vertices(theta, return_g=True)
        g_list = np.array([g[v] for v in v_list])
        sorted_g_list = np.sort(g_list)

        e_list, g_e = graph.sort_edges(theta, return_g=True)
        g_e_list = np.array([g_e[e] for e in e_list])
        sorted_g_e_list = np.sort(g_e_list)

        vertex_count = np.searchsorted(sorted_g_list, r_threshes, side='right')
        edge_count = np.searchsorted(sorted_g_e_list, r_threshes, side='right')

        if isinstance(graph, EmbeddedCW):
            f_list, g_f = graph.sort_faces(theta, return_g=True)
            g_f_list = np.array([g_f[f] for f in f_list])
            sorted_g_f_list = np.sort(g_f_list)
            face_count = np.searchsorted(
                sorted_g_f_list, r_threshes, side='right')
        else:
            face_count = np.zeros_like(r_threshes, dtype=np.int32)

        ecc = vertex_count - edge_count + face_count

        if return_counts:
            return ecc, vertex_count, edge_count, face_count
        else:
            return ecc

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _calculate_euler_chars_numba(projections, edge_maxes, face_maxes, thresholds):
        """Pure numerical computation of Euler characteristics"""
        num_dir = projections.shape[1]
        num_thresh = thresholds.shape[0]
        result = np.empty((num_dir, num_thresh), dtype=np.int32)

        sorted_projections = np.empty_like(projections)
        sorted_edge_maxes = np.empty_like(edge_maxes)

        for i in prange(num_dir):
            sorted_projections[:, i] = np.sort(projections[:, i])
            sorted_edge_maxes[:, i] = np.sort(edge_maxes[:, i])

        for j in prange(num_thresh):
            thresh = thresholds[j]
            for i in range(num_dir):
                v = np.searchsorted(
                    sorted_projections[:, i], thresh, side='right')
                e = np.searchsorted(
                    sorted_edge_maxes[:, i], thresh, side='right')
                f = np.searchsorted(
                    face_maxes[:, i], thresh, side='right') if face_maxes.shape[0] > 0 else 0
                result[i, j] = v - e + f

        return result

    def calculate_euler_chars(self, projections, edge_maxes, face_maxes, thresholds):
        """Calculate Euler characteristics and wrap in ECTResult"""
        result = ECT._calculate_euler_chars_numba(
            projections, edge_maxes, face_maxes, thresholds)
        return ECTResult(result, self.directions, self.threshes)

    def calculate_sect(self, ect_matrix=None):
        """
        Function to calculate the Smooth Euler Characteristic Transform (SECT) from the ECT matrix. 

        Returns:
            np.array
                The matrix representing the SECT of size (num_dirs,num_thresh).
        """

        # Calculate the SECT
        if ect_matrix is None:
            M = self.calculate_ect()
        else:
            M = ect_matrix

        # Get average of each row, corresponds to each direction
        A = np.average(M, axis=1)

        # Subtract the average from each row
        M_normalized = M - A[:, np.newaxis]

        # Take the cumulative sum of each row to get the SECT
        M_sect = np.cumsum(M_normalized, axis=1)

        return M_sect

    def plot_ecc(self, graph, theta, bound_radius=None, draw_counts=False):
        """
        Function to plot the Euler Characteristic Curve (ECC) for a specific direction theta. Note that this calculates the ECC for the input graph and then plots it.

        Parameters:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph or CW complex.
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to plot the ECC.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None. 
            draw_counts (bool):
                Whether to draw the counts of vertices, edges, and faces varying across thresholds. Default is False.
        """

        r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)
        if not draw_counts:
            ECC = self.calculate_ecc(graph, theta, r)
        else:
            ECC, vertex_count, edge_count, face_count = self.calculate_ecc(
                graph, theta, r, return_counts=True)

        # if self.threshes is None:
        #     self.set_bounding_radius(graph.get_bounding_radius())

        plt.step(r_threshes, ECC, label='ECC')

        if draw_counts:
            plt.step(r_threshes, vertex_count, label='Vertices')
            plt.step(r_threshes, edge_count, label='Edges')
            plt.step(r_threshes, face_count, label='Faces')
            plt.legend()

        theta_round = str(np.round(theta, 2))
        plt.title(r'ECC for $\omega = ' + theta_round + '$')
        plt.xlabel('$a$')
        plt.ylabel(r'$\chi(K_a)$')

    def plot_sect(self):
        """
        Function to plot the Smooth Euler Characteristic Transform (SECT) matrix. Note that the SECT matrix must be calculated before calling this function.

        The resulting plot will have the angle on the x-axis and the threshold on the y-axis.
        """

        # Make meshgrid.
        # Add back the 2pi to thetas for the pcolormesh
        thetas = np.concatenate((self.directions.thetas, [2*np.pi]))
        X, Y = np.meshgrid(thetas, self.threshes)
        M = np.zeros_like(X)

        # Transpose to get the correct orientation
        M[:, :-1] = self.SECT_matrix.T
        M[:, -1] = M[:, 0]  # Add the 2pi direction to the 0 direction

        plt.pcolormesh(X, Y, M, cmap='viridis')
        plt.colorbar()

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, 2*np.pi, 9))

        labels = [r'$0$',
                  r'$\frac{\pi}{4}$',
                  r'$\frac{\pi}{2}$',
                  r'$\frac{3\pi}{4}$',
                  r'$\pi$',
                  r'$\frac{5\pi}{4}$',
                  r'$\frac{3\pi}{2}$',
                  r'$\frac{7\pi}{4}$',
                  r'$2\pi$',
                  ]

        ax.set_xticklabels(labels)

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$t$')

        plt.title(r'SECT of Input Graph')
