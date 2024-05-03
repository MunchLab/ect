import numpy as np
from itertools import compress, combinations
from numba import jit
import matplotlib.pyplot as plt


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input ``embed_graph.EmbeddedGraph``.
    The result is a matrix where entry ``M[i,j]`` is ``chi(K_{a_i})`` for the direction $\omega_j$ where $a_i$ is the $i$th entry in ``self.threshes``, and $\omega_j$ is the ith entry in self.thetas.

    ...

    Attributes
        num_dirs : int
            The number of directions to consider in the matrix.
        num_thresh : int
            The number of thresholds to consider in the matrix.
        bound_radius : int
            Either None, or a positive radius of the bounding circle.
        matrix : np.array
            The matrix to store the ECT.

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

        self.matrix = np.zeros((num_dirs, num_thresh))

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
            self.threshes = np.linspace(bound_radius, -bound_radius, self.num_thresh)

    def calculateECC(self, G, theta, tightbbox=False):
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
            r_threshes = np.linspace(r, -r, self.num_thresh)
        else:
            # The user wants to use the internally determined bounding radius
            if self.bound_radius is None:
                raise ValueError('Bounding radius must be set before calculating ECC when you have tightbbox=False.')
            else:
                r = self.bound_radius
                r_threshes = self.threshes


        # Direction given as 2d coordinates
        omega = (np.cos(theta), np.sin(theta))
        
        # sort the vertices according to the direction
        v_list, g = G.sort_vertices(theta, return_g=True)
                   
        def count_duplicate_edges(newV):
            """
            Function to count the number of duplicate counted edges from lower_edges. These duplicate edges are added to the EC value.
            """

            # @jit <----- Liz can't figure out how to make this work in here
            def find_combos(newV):
                #res = list(combinations(newV, 2))
                res = []
                n = len(newV)
                for i in range(n):
                    for j in range(i+1, n):
                        res.append((newV[i], newV[j]))
                return res


            res = find_combos(newV)
            count=0
            for v,w in res:
                if G.has_edge(v,w) and g[v]==g[w]:
                    count+=1
            return count
        

        
        # Full ECC vector
        ecc=[]
        ecc.append(0)

        
        for i in range(self.num_thresh):

            #set of new vertices that appear in this threshold band
            if i==self.num_thresh-1:
                newV =list(compress(v_list,[r_threshes[i]>g[v] for v in v_list]))
            else:
                newV =list(compress(v_list,[r_threshes[i]>g[v]>=r_threshes[i+1] for v in v_list]))
    
            x = ecc[i]#previous value of ECC (not index i-1 becuase of extra 0 appended to begining)
            if newV: # if there's new vertices to count
                v_count=0
                e_count=0
                for v in newV:
                    k = G.lower_edges(v, omega)
                    v_count+=1 #add 1 to vertex count
                    e_count+=k #subtract the the number of lower edges
                #check for duplicate edges counted
                dupl = count_duplicate_edges(newV)
                # after counting all new vertices and edges
                ecc.append(x+v_count-e_count+dupl)
            else:
                ecc.append(x)
        ecc = ecc[1:] #Drop the initial 0 value
        #print('ECC for direction', omega, '= ', ecc)
        
        return ecc

    def calculateECT(self, graph, tightbbox=False):
        """
        Calculates the ECT from an input EmbeddedGraph. The entry M[i,j] is $\chi(K_{a_j})$ for the direction $\omega_i$ where $a_j$ is the $j$th entry in self.threshes, and $\omega_i$ is the ith entry in self.thetas.

        Parameters:
            graph : EmbeddedGraph
                The input graph to calculate the ECT from.
            tightbbox : bool, optional
                Whether to use the tight bounding box (a different value in each direction) computed from the input graph. Otherwise, a bounding box needs to already be set manually with the `set_bounding_box` method.

        Returns:
            np.array
                The matrix representing the ECT of size (num_dirs,num_thresh).
        """

        if tightbbox == False and self.bound_radius is None:
            self.set_bounding_radius(graph.get_bounding_radius())

        M = np.zeros((self.num_dirs, self.num_thresh))
        
        for i, theta in enumerate(self.thetas):
            M[i] = self.calculateECC(graph, theta, tightbbox)
        
        self.matrix = M
       
        return self.matrix



    def plotECC(self, graph, theta, tightbbox=False):
        """
        Function to plot the Euler Characteristic Curve (ECC) for a specific direction theta.

        Parameters:
            graph : EmbeddedGraph
                The input graph.
            theta : float
                The angle in [0,2*np.pi] for the direction to plot the ECC.
        """

        ECC = self.calculateECC(graph, theta, tightbbox)
        
        plt.step(self.threshes,ECC)
        plt.title(r'ECC for $\omega = \frac{3 \pi}{4}$')
        plt.xlabel('$a$')
        plt.ylabel(r'$\chi(K_a)$')


    def plotECT(self):

        """
        Function to plot the Euler Characteristic Transform (ECT) matrix.

        The resulting plot will have the angle on the x-axis and the threshold on the y-axis.
        """

        # Make meshgrid.
        # Add back the 2pi to thetas for the pcolormesh
        thetas = np.concatenate((self.thetas,[2*np.pi]))
        X,Y = np.meshgrid(thetas,self.threshes)
        M = np.zeros_like(X)

        M[:,:-1] = self.matrix.T # Transpose to get the correct orientation
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