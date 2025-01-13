import numpy as np
from itertools import compress, combinations
from numba import jit
import matplotlib.pyplot as plt
from ect.embed_cw import EmbeddedCW
import time


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
        ECT_matrix (np.array):
            The matrix to store the ECT.
        SECT_matrix (np.array):
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
        if bound_radius == None:
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

    def get_ECT(self):
        """
        Returns the ECT matrix.
        """
        return self.ECT_matrix

    def get_SECT(self):
        """
        Returns the SECT matrix.
        """
        return self.SECT_matrix

    def calculateECC(self, G, theta, bound_radius=None, return_counts=False):
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

    def calculateECT(self, graph, bound_radius=None, compute_SECT=True):
        """
        Calculates the ECT from an input either `EmbeddedGraph` or `EmbeddedCW`. The entry ``M[i,j]`` is :math:`\\chi(K_{a_j})` for the direction :math:`\omega_i` where :math:`a_j` is the jth entry in ``self.threshes``, and :math:`\omega_i` is the ith entry in ``self.thetas``.

        Parameters:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph to calculate the ECT from.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.
            compute_SECT (bool):
                Whether to compute the SECT after the ECT is computed. Default is True. Sets the SECT_matrix attribute, but doesn't return it. Can be returned with the get_SECT method.

        Returns:
            np.array
                The matrix representing the ECT of size (num_dirs,num_thresh).
        """

        r, r_threshes = self.get_radius_and_thresh(graph, bound_radius)

        # Note... this overwrites the self.threshes if it's not set.
        self.set_bounding_radius(r)

        M = np.zeros((self.num_dirs, self.num_thresh))

        for i, theta in enumerate(self.thetas):
            M[i] = self.calculateECC(graph, theta, r)

        self.ECT_matrix = M

        if compute_SECT:
            self.SECT_matrix = self.calculateSECT()

        return self.ECT_matrix

    def calculateSECT(self):
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
        M_SECT = np.cumsum(M_normalized, axis=1)

        return M_SECT

    def plotECC(self, graph, theta, bound_radius=None, draw_counts=False):
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
            ECC = self.calculateECC(graph, theta, r)
        else:
            ECC, vertex_count, edge_count, face_count = self.calculateECC(
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

    def plotECT(self):
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
        plt.ylabel(r'$a$')

        plt.title(r'ECT of Input Graph')

    def plotSECT(self):
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
            self.plotECT()
        elif plot_type == 'SECT':
            self.plotSECT()
        else:
            raise ValueError('plot_type must be either "ECT" or "SECT".')
