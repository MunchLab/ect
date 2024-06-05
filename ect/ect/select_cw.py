import numpy as np
import networkx as nx
from ect_graph import ECT
from embed_graph import EmbeddedGraph, create_example_graph
from embed_cw import EmbeddedCW, create_example_cw
from embed_threshold_cw import EmbeddedThresholdCW

class SELECT(ECT):
    """A class to calculate the Super-Level Euler Characteristic Transform (ECT) from an input :any:`EmbeddedThresholdCW`.

    The result is a matrix where entry ``M[i,j,k]`` is :math:`\chi(K_{g \geq s_k, a_i})` for the direction :math:`\omega_j` where :math:`a_i` is the ith entry in ``self.threshes``, :math:`\omega_j` is the ith entry in ``self.thetas``, and :math:`s_k` is the jth entry in ``self.level_sets``.

    Attributes
        num_dirs (int):
            The number of directions to consider in the matrix.
        num_thresh (int):
            The number of thresholds to consider in the matrix.
        num_superlevel_sets (int):
            The number of superlevel sets to consider in the matrix.
        bound_radius (int):
            Either ``None``, or a positive radius of the bounding circle.
        ECT_matrix (np.array):
            The matrix to store the ECT.
        SECT_matrix (np.array):
            The matrix to store the SECT.
        SELECT_matrix (np.array):
            The matrix to store the SELECT.

    """

    def __init__(self, num_dirs, num_thresh, min_superlevel_set, max_superlevel_set, num_superlevel_sets, bound_radius=None):
        """
        Constructs all the necessary attributes for the ECT object.

        Parameters:
            num_dirs (int):
                The number of directions to consider in the matrix.
            num_thresh (int):
                The number of thresholds to consider in the matrix.
            min_superlevel_set (int):
                The minimum threshold for superlevel sets in the CW complex.
            max_superlevel_set (int):
                The maximum threshold for superlevel sets in the CW complex.
            num_superlevel_sets (int):
                The number of superlevel sets to consider in the matrix.
            bound_radius (int):
                Either None, or a positive radius of the bounding circle.
        """
        super().__init__(num_dirs, num_thresh, bound_radius)
        self.min_superlevel_set = min_superlevel_set
        self.max_superlevel_set = max_superlevel_set
        self.num_superlevel_sets = num_superlevel_sets
        self.SELECT_matrix = np.zeros((num_dirs, num_thresh, num_superlevel_sets))

    def calculateSELECT(self, G, bound_radius=None):
        """
        Function to compute the Euler Characteristic of an `EmbeddedThresholdCW`, a CW complex with coordinates and thresholds for each n-cell.

        Parameters:
            graph (EmbeddedGraph/EmbeddedCW):
                The input graph to calculate the ECT from.
            bound_radius (float):
                If None, uses the following in order: (i) the bounding radius stored in the class; or if not available (ii) the bounding radius of the given graph. Otherwise, should be a postive float :math:`R` where the ECC will be computed at thresholds in :math:`[-R,R]`. Default is None.
            return_counts (bool):
                Whether to return the counts of vertices, edges, and faces below the threshold. Default is False.
        """
        M = np.zeros((self.num_dirs, self.num_thresh, self.num_superlevel_sets))

        superlevel_thresholds = np.linspace(self.min_superlevel_set, self.max_superlevel_set, self.num_superlevel_sets)
        for i in range(len(superlevel_thresholds)):
            H = G.thresh(superlevel_thresholds[i])
            M[:, :, i] = super().calculateECT(H, bound_radius)
        
        self.SELECT_matrix = M

    def plotSELECT(self, G, superlevel_set):
        """
        Plots the region of the SELECT matrix for a specific threshold value.

        Parameters:
            superlevel_set (int):
                The index of the threshold superlevel_thresholds to plot.
        """
        M = self.SELECT_matrix
        self.ECT_matrix = M[:, :, superlevel_set]
        super().plotECT()
