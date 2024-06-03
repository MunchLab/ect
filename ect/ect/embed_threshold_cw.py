import numpy as np
import networkx as nx
from ect_graph import ECT
from embed_graph import EmbeddedGraph, create_example_graph
from embed_cw import EmbeddedCW, create_example_cw

class EmbeddedThresholdCW(EmbeddedCW):
    """
    A class to represent a straight-line-embedded CW complex with an added filtration function to be used for SELECT.

    Attributes
        graph : nx.Graph
            a NetworkX graph object
        coordinates : dict
            a dictionary mapping vertices to their (x, y) coordinates
        faces : list
            a dictionary listing the vertices for each face
        node_thresh : dict
            a dictionary matching each vertex to its filtration function result
        edge_thresh : dict
            a dictionary matching each edge to its filtration function result
        face_thresh : dict
            a dictionary matching each face to its filtration function result

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedThresholdCW object.

        """
        super().__init__()
        self.node_thresh = {}
        self.edge_thresh = {}
        self.face_thresh = {}

    def count_new(self, a, b):
        """
        Returns the vertices, edges, and 2-cells with threshold value s between a and b s.t. a < s <= b.

        Parameters:
            a (int):
                the lower threshold value
            b (int):
                the higher threshold value

        Returns:
            dict, dict, dict
                dictionaries of the vertices, edges, and 2-cells within the threshold range
        """


    def thresh(self, a=0):
        """
        Constructs a superlevel set of the EmbeddedThresholdCW.

        Parameters:
            a (int):
                the threshold value
            level_set (string):
                if "super" return the superlevel set, if "sub" return the sublevel set

        Returns:
            EmbeddedThresholdCw
                the superlevel set of the EmbeddedThresholdCW with threshold less/greater than a
        """
        G = EmbeddedThresholdCW()

        for node in self.nodes:
            thresh_val = self.node_thresh[node]
            if thresh_val >= a:
                coordinates = self.coordinates[node]
                G.add_node(node, coordinates[0], coordinates[1])
                G.node_thresh[node] = thresh_val

        for edge in self.edges:
            thresh_val = self.edge_thresh[edge]
            if thresh_val >= a:
                G.add_edge(edge[0], edge[1])
                G.edge_thresh[edge] = thresh_val

        for face in self.faces:
            thresh_val = self.face_thresh[face]
            if thresh_val >= a:
                G.add_face(face)
                G.face_thresh[face] = thresh_val

        return G

    def plot(self, level_set="false", a=0):
        """
        Plots a sublevel set of the EmbeddedThresholdCW.

        Parameters:
            a (int):
                the threshold value
            level_set (string):
                "false" to plot everything, "super" for a superlevel set g >= a, "sub" for a sublevel set g <= a
        """
        if level_set == "false":
            super().plot()
        else:
            K = EmbeddedThresholdCW()
            K = self.thresh(a)
            K.plot()

