import numpy as np
from itertools import compress, combinations
from numba import jit


class ECT:
    """
    A class to calculate the Euler Characteristic Transform (ECT) from an input `embed_graph.EmbeddedGraph`.

    ...

    Attributes
    ----------
    num_dirs : int
        The number of directions to consider in the matrix.
    num_thresh : int
        The number of thresholds to consider in the matrix.
    matrix : np.array
        The matrix to store the ECT.

    Methods
    -------
    __init__(num_dirs, num_thresh):
        Constructs all the necessary attributes for the ECT object.
    calculate(graph):
        Calculates the ECT from an input EmbeddedGraph.

    """

    def __init__(self, num_dirs, num_thresh, bound_radius = None):
        """
        Constructs all the necessary attributes for the ECT object.

        Parameters:
            num_dirs (int): The number of directions to consider in the matrix.
            num_thresh (int): The number of thresholds to consider in the matrix.
            bound_radius (int): Either None, or a positive radius of the bounding circle.
        """
        self.num_dirs = num_dirs
        self.thetas = np.linspace(0, 2*np.pi, self.num_dirs)


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

        if self. bound_radius is None:
            self.threshes = None
        else:
            self.threshes = np.linspace(bound_radius, -bound_radius, self.num_thresh)

    def calculateECC(self, G, theta, tightbbox=False):
        """
        Function to compute the Euler Characteristic of a graph with coordinates for each vertex (pos), 
        using a specified number of thresholds and bounding box defined by radius r.

        Parameters:
            G (EmbeddedGraph): The input graph.
            theta (float): The angle in [0,2*np.pi] for the direction to compute the ECC.
            tightbbox (bool): Whether to use the tight bounding box computed from the input graph. Otherwise, a bounding box needs to already be set manually with the `set_bounding_box` method.

        """
        
        # Either use the global radius and the set self.threshes; or use the tight bounding box and calculate 
        # the thresholds from that. 
        if tightbbox and self.bound_radius is None:
            raise ValueError("Bounding box needs to be set manually with the `set_bounding_radius` method when `tightbbox` is True.")
        elif tightbbox:
            # thresholds for filtration, r should be defined from global bounding box
            r = G.get_bounding_radius()
            r_threshes = np.linspace(self.bound_radius, -self.bound_radius, self.numThresh)
        else:
            # this should be using the internal version 
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

    def calculateECT(self, graph):
        """
        Calculates the ECT from an input EmbeddedGraph.

        Parameters:
            graph (EmbeddedGraph): The input graph to calculate the ECT from.

        Returns:
            np.array: The matrix representing the ECT.
        """
        
        for i, theta in enumerate(self.thetas):
            self.matrix[i] = self.calculateECC(graph, theta)
        
        return self.matrix