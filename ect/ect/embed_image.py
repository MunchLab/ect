import numpy as np
import networkx as nx
from ect_graph import ECT
from embed_graph import EmbeddedGraph, create_example_graph
from embed_cw import EmbeddedCW, create_example_cw
from embed_threshold_cw import EmbeddedThresholdCW

class EmbeddedImage(EmbeddedThresholdCW):
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
        pixel_values : 2D NumPy array
            a list of the pixel values of the image

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedImage object.

        """
        super().__init__()
        self.pixel_values = []

    def set_pixel_values(self, image_input):
        """
        Sets the list of pixel values.

        Parameters:
            image_input (2D NumPy array):
                a 2D NumPy array of pixel values for the image
        """
        self.pixel_values = image_input

    def create_embed_image(self):
        """
        Creates the embedded image once the pixel values are set.

        TODO: Optimize this and then adjust set_threshold accordingly.
        NOTE: KEEP FACES AS TUPLES so they can be hashed by nx functions
        """
        height = len(self.pixel_values) + 1
        width = len(self.pixel_values[0]) + 1
        H = nx.grid_graph(dim = (height, width))
        for node in H.nodes:
            self.add_node(f"{node}", node[0], node[1])
        for edge in H.edges:
            self.add_edge(f"{edge[0]}", f"{edge[1]}")
        for x in range(width-1):
            for y in range(height-1):
                self.add_face((f"({x}, {y})", f"({x+1}, {y})", f"({x+1}, {y+1})", f"({x}, {y+1})"))

    def set_thresholds(self):
        """
        Uses the pixel values to set thresholds for every vertex, edge, and face.

        TODO: Edge and face allocation differs for superlevel versus sublevel sets.
        """
        # assigns face threshold values
        # len(self.pixel_values) - int(face[0][4]) - 1 makes it so the image isn't flipped vertically
        for face in self.faces:
            self.face_thresh[face] = self.pixel_values[len(self.pixel_values) - int(face[0][4]) - 1][int(face[0][1])]

        # assigns node threshold values
        # each node's threshold is the minimum threshold of the four neighboring faces
        for node in self.nodes:
            x = int(node[1])
            y = int(node[4])
            possible_thresh = []

            possible_faces = [(f"({x-1}, {y-1})", f"({x}, {y-1})", f"({x}, {y})", f"({x-1}, {y})"), 
                              (f"({x}, {y-1})", f"({x+1}, {y-1})", f"({x+1}, {y})", f"({x}, {y})"), 
                              (f"({x-1}, {y})", f"({x}, {y})", f"({x}, {y+1})", f"({x-1}, {y+1})"), 
                              (f"({x}, {y})", f"({x+1}, {y})", f"({x+1}, {y+1})", f"({x}, {y+1})")]
            
            for possible_face in possible_faces:
                if possible_face in self.faces:
                    possible_thresh.append(self.face_thresh[possible_face])

            self.node_thresh[node] = max(possible_thresh)

        # assigns edge thresholds
        # each edge's threshold is the minimum threshold of the two neighboring faces
        for edge in self.edges:
            x = min(int(edge[0][1]), int(edge[1][1]))
            y = min(int(edge[0][4]), int(edge[1][4]))
            possible_thresh = []

            if edge[0][1] != edge[1][1]:     # horizontal edge
                possible_faces = [(f"({x}, {y-1})", f"({x+1}, {y-1})", f"({x+1}, {y})", f"({x}, {y})"), 
                                  (f"({x}, {y})", f"({x+1}, {y})", f"({x+1}, {y+1})", f"({x}, {y+1})")]
            else:           # vertical edge
                possible_faces = [(f"({x-1}, {y})", f"({x}, {y})", f"({x}, {y+1})", f"({x-1}, {y+1})"), 
                                  (f"({x}, {y})", f"({x+1}, {y})", f"({x+1}, {y+1})", f"({x}, {y+1})")]
                
            for possible_face in possible_faces:
                if possible_face in self.faces:
                    possible_thresh.append(self.face_thresh[possible_face])

            self.edge_thresh[edge] = max(possible_thresh)
