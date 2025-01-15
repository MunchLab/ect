import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

from ect.embed_cw import EmbeddedCW


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

    def __init__(self, num_dirs, num_thresh, bound_radius=None):
        """
        Constructs all the necessary attributes for the ECT object.

        Parameters:
            num_dirs (int):
                The number of directions to consider in the matrix.
            num_thresh (int):
                The number of thresholds to consider in the matrix.
            bound_radius (int):
                Either None, or a positive radius of the bounding circle.
        """
        self.num_dirs = num_dirs

        # Note: This version doesn't include 2pi since its the same as the 0 direction.
        self.thetas = np.linspace(0, 2*np.pi, self.num_dirs, endpoint=False)

        self.num_thresh = num_thresh
        self.set_bounding_radius(bound_radius)

        self.ECT_matrix = np.zeros((num_dirs, num_thresh))
        self.SECT_matrix = np.zeros((num_dirs, num_thresh))

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

    def get_ect(self):
        """
        Returns the ECT matrix.
        """
        return self.ect_matrix

    def get_sect(self):
        """
        Returns the SECT matrix.
        """
        return self.sect_matrix

    def calculate_ecc(self, G, theta, bound_radius=None, return_counts=False):
        """
        Function to compute the Euler Characteristic Curve (ECC) of an `EmbeddedGraph`.

        Parameters:
            G (nx.Graph): The graph to compute the ECC for.
            theta (float): The angle (in radians) for the direction function.
            bound_radius (float, optional): Radius for threshold range. Default is None.
            return_counts (bool, optional): Whether to return vertex, edge, and face counts. Default is False.

        Returns:
            numpy.ndarray: ECC values at each threshold.
            (Optional) Tuple of counts: (ecc, vertex_count, edge_count, face_count)
        """
        r, r_threshes = self.get_radius_and_thresh(G, bound_radius)

        r_threshes = np.array(r_threshes)

        # Sort vertices and edges based on projection
        v_list, g = G.sort_vertices(theta, return_g=True)
        g_list = np.array([g[v] for v in v_list])
        sorted_g_list = np.sort(g_list)

        e_list, g_e = G.sort_edges(theta, return_g=True)
        g_e_list = np.array([g_e[e] for e in e_list])
        sorted_g_e_list = np.sort(g_e_list)

        vertex_count = np.searchsorted(sorted_g_list, r_threshes, side='right')
        edge_count = np.searchsorted(sorted_g_e_list, r_threshes, side='right')

        if isinstance(G, EmbeddedCW):
            f_list, g_f = G.sort_faces(theta, return_g=True)
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
    @jit(nopython=True, parallel=True)
    def fast_threshold_comp(projections, edge_maxes, thresholds):
        """Calculate the euler characteristic for each direction in parallel

        Parameters:
            projections (np.array):
                The projections of the vertices.
            edge_maxes (np.array):
                The projections of the edges.
            thresholds (np.array):
                The thresholds to compute the ECT at.

        Returns:
            np.array:
                The ECT matrix of size (num_dirs, num_thresh).
        """
        num_vertices, num_dir = projections.shape
        num_edges = edge_maxes.shape[0]
        num_thresh = len(thresholds)
        result = np.empty((num_dir, num_thresh), dtype=np.int32)

        # parallelize over directions
        for i in prange(num_dir):
            for j in range(num_thresh):
                thresh = thresholds[j]
                vert_count = 0
                edge_count = 0

                # Use SIMD-friendly loops
                for v in range(num_vertices):
                    if projections[v, i] <= thresh:
                        vert_count += 1
                for e in range(num_edges):
                    if edge_maxes[e, i] <= thresh:
                        edge_count += 1

                result[i, j] = vert_count - edge_count

        return result

    def calculate_ect(self, graph, bound_radius=None,):
        """Vectorized ECT calculation using optimized numpy operations

        Parameters:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph or CW complex.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.

        Returns:
            np.array:
                The ECT matrix of size (num_dirs, num_thresh).
        """
        r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)

        coords = np.array([graph.coordinates[v] for v in graph.nodes()])

        # create vertex index mapping and convert edges
        vertex_to_idx = {v: i for i, v in enumerate(graph.nodes())}
        edges = np.array([[vertex_to_idx[u], vertex_to_idx[v]]
                         for u, v in graph.edges()])

        directions = np.empty((self.num_dirs, 2), order='F')
        np.stack([np.cos(self.thetas), np.sin(self.thetas)],
                 axis=1, out=directions)

        projections = np.empty((len(coords), self.num_dirs), order='F')
        np.matmul(coords, directions.T, out=projections)

        edge_maxes = np.maximum(
            projections[edges[:, 0]], projections[edges[:, 1]])

        # use numba-optimized threshold computation
        ect_matrix = self.fast_threshold_comp(
            projections, edge_maxes, r_threshes)

        self.ect_matrix = ect_matrix

        return ect_matrix

    def calculate_sect(self):
        """
        Function to calculate the Smooth Euler Characteristic Transform (SECT) from the ECT matrix. 

        Returns:
            np.array
                The matrix representing the SECT of size (num_dirs,num_thresh).
        """

        # Calculate the SECT
        M = self.ECT_matrix

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

    def plot_ect(self):
        """
        Function to plot the Euler Characteristic Transform (ECT) matrix. Note that the ECT matrix must be calculated before calling this function.

        The resulting plot will have the angle on the x-axis and the threshold on the y-axis.
        """

        # Make meshgrid.
        # Add back the 2pi to thetas for the pcolormesh
        thetas = np.concatenate((self.thetas, [2*np.pi]))
        X, Y = np.meshgrid(thetas, self.threshes)
        M = np.zeros_like(X)

        # Transpose to get the correct orientation
        M[:, :-1] = self.ECT_matrix.T
        M[:, -1] = M[:, 0]  # Add the 2pi direction to the 0 direction

        plt.pcolormesh(X, Y, M, cmap='viridis')
        plt.colorbar()

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, 2*np.pi, 9))

        labels = [
            r'$0$',
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
        plt.ylabel(r'$a$')

        plt.title(r'ECT of Input Graph')

    def plot_sect(self):
        """
        Function to plot the Smooth Euler Characteristic Transform (SECT) matrix. Note that the SECT matrix must be calculated before calling this function.

        The resulting plot will have the angle on the x-axis and the threshold on the y-axis.
        """

        # Make meshgrid.
        # Add back the 2pi to thetas for the pcolormesh
        thetas = np.concatenate((self.thetas, [2*np.pi]))
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

    def plot(self, plot_type):
        """
        Function to plot the ECT or SECT matrix. The type parameter should be either 'ECT' or 'SECT'.

        Parameters:
            plot_type : str
                The type of plot to make. Either 'ECT' or 'SECT'.
        """

        if plot_type == 'ECT':
            self.plot_ect()
        elif plot_type == 'SECT':
            self.plot_sect()
        else:
            raise ValueError('plot_type must be either "ECT" or "SECT".')
