import numpy as np
from itertools import compress, combinations
from numba import jit
import matplotlib.pyplot as plt


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input ``embed_graph.EmbeddedGraph``.
    The result is a matrix where entry ``M[i,j]`` is :math:`\chi(K_{a_i})` for the direction :math:`\omega_j` where :math:`a_i` is the ith entry in ``self.threshes``, and :math:`\omega_j` is the ith entry in ``self.thetas``.

    ...

    Attributes
        num_dirs : int
            The number of directions to consider in the matrix.
        num_thresh : int
            The number of thresholds to consider in the matrix.
        bound_radius : int
            Either None, or a positive radius of the bounding circle.
        ECT_matrix : np.array
            The matrix to store the ECT.
        SECT_matrix : np.array
            The matrix to store the SECT.

    Methods
        __init__(num_dirs, num_thresh):
            Constructs all the necessary attributes for the ECT object.
        calculate(graph):
            Calculates the ECT from an input EmbeddedGraph.

    """

    def __init__(self, num_dirs, num_thresh, bound_radius = None):
        """
        Constructs all the necessary attributes for the ECT object.

        Parameters:
            num_dirs : int
                The number of directions to consider in the matrix.
            num_thresh : int
                The number of thresholds to consider in the matrix.
            bound_radius : int
                Either None, or a positive radius of the bounding circle.
        """
        self.num_dirs = num_dirs

        # Note: This version doesn't include 2pi since its the same as the 0 direction.
        self.thetas = np.linspace(0, 2*np.pi, self.num_dirs,endpoint=False)


        self.num_thresh = num_thresh
        self.set_bounding_radius(bound_radius)

        self.ECT_matrix = np.zeros((num_dirs, num_thresh))
        self.SECT_matrix = np.zeros((num_dirs, num_thresh))

    def set_bounding_radius(self, bound_radius):
        """
        Manually sets the radius of the bounding circle centered at the origin for the ECT object.

        Parameters:
            bound_radius (int): Either None, or a positive radius of the bounding circle.
        """
        self.bound_radius = bound_radius

        if self.bound_radius is None:
            self.threshes = None
        else:
            self.threshes = np.linspace(-bound_radius, bound_radius, self.num_thresh)

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

    def calculateECC(self, G, theta, tightbbox = False):
        """
        Function to compute the Euler Characteristic of a graph with coordinates for each vertex (pos).

        Parameters:
            G : Graph
                The graph to compute the Euler Characteristic for.
            theta : float
                The angle (in radians) to rotate the graph by before computing the Euler Characteristic.
            tightbbox : bool, optional
                If True, use the tight bounding box of the graph. If False, use the bounding circle. Default is False.
        """
        
        # Either use the global radius and the set self.threshes; or use the tight bounding box and calculate 
        # the thresholds from that. 
        if tightbbox:
            # thresholds for filtration, r should be defined from global bounding box
            r = G.get_bounding_radius()
            r_threshes = np.linspace(-r,r, self.num_thresh)
        else:
            # The user wants to use the internally determined bounding radius
            if self.bound_radius is None:
                raise ValueError('Bounding radius must be set before calculating ECC when you have tightbbox=False.')
            else:
                r = self.bound_radius
                r_threshes = self.threshes

        #--
        def num_below_threshold(func_list,thresh):
            """ 
            Returns the number of entries in func_list that are below the threshold thresh. 
            Warning: func_list must be sorted in ascending order.

            Parameters
            func_list: list of floats
            thresh: float

            Returns
                int 
            """
            func_max = func_list[-1]
            if thresh < func_max:
                return np.argmin(func_list < thresh)
            else: 
                return len(func_list)
        #--    
        
        v_list, g = G.sort_vertices(theta, return_g=True)
        g_list = [g[v] for v in v_list]

        vertex_count = np.array([num_below_threshold(g_list,thresh) for thresh in r_threshes])
        # print(vertex_count)


        e_list, g_e = G.sort_edges(np.pi/2, return_g=True)
        g_e_list = [g_e[e] for e in e_list]
        edge_count = np.array([num_below_threshold(g_e_list,thresh) for thresh in r_threshes])
        # print(edge_count)

        # print(vertex_count - edge_count)
        ecc = vertex_count - edge_count

        return ecc

    def calculateECT(self, graph, tightbbox=False, compute_SECT=True):
        """
        Calculates the ECT from an input EmbeddedGraph. The entry M[i,j] is $\chi(K_{a_j})$ for the direction $\omega_i$ where $a_j$ is the $j$th entry in self.threshes, and $\omega_i$ is the ith entry in self.thetas.

        Parameters:
            graph : EmbeddedGraph
                The input graph to calculate the ECT from.
            tightbbox : bool, optional
                Whether to use the tight bounding box (a different value in each direction) computed from the input graph. Otherwise, a bounding box needs to already be set manually with the `set_bounding_box` method.
            compute_SECT : bool, optional
                Whether to compute the SECT after the ECT is computed. Default is True. Sets the SECT_matrix attribute, but doesn't return it. Can be returned with the get_SECT method.

        Returns:
            np.array
                The matrix representing the ECT of size (num_dirs,num_thresh).
        """

        if tightbbox == False and self.bound_radius is None:
            self.set_bounding_radius(graph.get_bounding_radius())

        M = np.zeros((self.num_dirs, self.num_thresh))
        
        for i, theta in enumerate(self.thetas):
            M[i] = self.calculateECC(graph, theta, tightbbox)
        
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
        A = np.average(M, axis = 1)

        # Subtract the average from each row
        M_normalized = M - A[:, np.newaxis]

        # Take the cumulative sum of each row to get the SECT
        M_SECT = np.cumsum(M_normalized, axis = 1)

        return M_SECT

    def plotECC(self, graph, theta):
        """
        Function to plot the Euler Characteristic Curve (ECC) for a specific direction theta. Note that this calculates the ECC for the input graph and then plots it.

        Parameters:
            graph : EmbeddedGraph
                The input graph.
            theta : float
                The angle in [0,2*np.pi] for the direction to plot the ECC.
        """

        ECC = self.calculateECC(graph, theta)
        
        plt.step(self.threshes,ECC)
        theta_round = str(np.round(theta,2))
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
        thetas = np.concatenate((self.thetas,[2*np.pi]))
        X,Y = np.meshgrid(thetas,self.threshes)
        M = np.zeros_like(X)

        M[:,:-1] = self.ECT_matrix.T # Transpose to get the correct orientation
        M[:,-1] = M[:,0] # Add the 2pi direction to the 0 direction
        

        plt.pcolormesh(X,Y,M, cmap = 'viridis')
        plt.colorbar()

        ax = plt.gca()
        ax.set_xticks(np.linspace(0,2*np.pi,9))

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
        thetas = np.concatenate((self.thetas,[2*np.pi]))
        X,Y = np.meshgrid(thetas,self.threshes)
        M = np.zeros_like(X)

        M[:,:-1] = self.SECT_matrix.T # Transpose to get the correct orientation
        M[:,-1] = M[:,0] # Add the 2pi direction to the 0 direction
        

        plt.pcolormesh(X,Y,M, cmap = 'viridis')
        plt.colorbar()

        ax = plt.gca()
        ax.set_xticks(np.linspace(0,2*np.pi,9))

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

    def plot(self, type):
        """
        Function to plot the ECT or SECT matrix. The type parameter should be either 'ECT' or 'SECT'.

        Parameters:
            type : str
                The type of plot to make. Either 'ECT' or 'SECT'.
        """

        if type == 'ECT':
            self.plotECT()
        elif type == 'SECT':
            self.plotSECT()
        else:
            raise ValueError('Type must be either "ECT" or "SECT".')