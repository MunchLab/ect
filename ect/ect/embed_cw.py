import numpy as np
from itertools import compress, combinations
import matplotlib.pyplot as plt
import networkx as nx
from ect.embed_graph import EmbeddedGraph, create_example_graph
from scipy.optimize import linprog


class EmbeddedCW(EmbeddedGraph):
    """
    A class to represent a straight-line-embedded CW complex. We assume that the coordinates for the embedding of the vertices are given, the 1-skeleton is in fact a graph (so not as general as a full CW complex) with straight line embeddings, and 2-Cells are the interior of the shape outlined by its boundary edges. 

    Faces should be passed in as a list of vertices, where the vertices are in order around the face. However, the ECT function will likely still work if the ordering is different. The drawing functions however might look strange. Note the class does not (yet?) check to make sure the face is valid, i.e. is a cycle in the graph, and bounds a region in the plane.

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedCW object.
        """

        # The super class initializes the graph and the coordinates dictionary.
        super().__init__()
        self.faces = []

    def add_from_embedded_graph(self, G):
        """
        Adds the edges and coordinates from an EmbeddedGraph object to the EmbeddedCW object.

        Parameters:
            embedded_graph (EmbeddedGraph):
                The EmbeddedGraph object to add from.
        """
        self.add_nodes_from(G.nodes(), G.coordinates)
        self.add_edges_from(G.edges())

    def add_face(self, face, check=True):
        """
        Adds a face to the list of faces.

        TODO: Do we want a check to make sure the face is legit? (i.e. is a cycle in the graph, and bounds a region in the plane)

        Parameters:
            face (list):
                A list of vertices that make up the face.
            check (bool):
                Whether to check that the face is a valid addition to the cw complex.
        """
        if check:

            # Make sure all edges are in the graph
            edges = list(zip(face, face[1:] + [face[0]]))
            for edge in edges:
                if edge not in self.edges:
                    raise ValueError(f"Edge {edge} not in graph.")

            # TODO: The goal here is to check that none of the other vertices are in the polygon defined by the face.
            # Problem is that the face could be concave, so we can't just check if the point is in the convex hull of the face.
            # This is a bit of a tricky problem, so I'm going to leave it for now.

            # def in_hull(points, x):
            #     # Solution for checking if a point is in a convex hull
            #     # from Nils answer here:
            #     # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
            #     n_points = len(points)
            #     n_dim = len(x)
            #     c = np.zeros(n_points)
            #     A = np.r_[points.T,np.ones((1,n_points))]
            #     b = np.r_[x, np.ones(1)]
            #     lp = linprog(c, A_eq=A, b_eq=b)
            #     return lp.success

            # points = np.array([self.coordinates[v] for v in face])
            # if not in_hull(points.T, self.coordinates[face[0]]):
            #     raise ValueError(f"Face {face} does not bound an empty region in the plane.")

        # Note: faces need to be tuples to make
        # the face hashable so it can be used as a key in a dictionary
        self.faces.append(tuple(face))

    def g_omega_faces(self, theta):
        """
        Calculates the function value of the faces of the graph by making the value equal to the max vertex value 

        Parameters:

            theta (float): 
                The direction of the function to be calculated.

        Returns:
            dict
                A dictionary of the function values of the faces.
        """
        g = super().g_omega(theta)

        g_faces = {}
        for face in self.faces:
            g_faces[tuple(face)] = max([g[v] for v in face])

        return g_faces

    def sort_faces(self, theta, return_g=False):
        """
        Function to sort the faces of the graph according to the function

        .. math ::

            g_\omega(\sigma) = \max \{ g_\omega(v) \mid  v \in \sigma \}

        in the direction of :math:`\\theta \in [0,2\pi]` .

        Parameters:
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to sort the edges.
            return_g (bool):
                Whether to return the :math:`g(v)` values along with the sorted edges.

        Returns:
            A list of edges sorted in increasing order of the :math:`g(v)` values. 
            If ``return_g`` is True, also returns the ``g`` dictionary with the function values ``g[vertex_name]=func_value``. 

        """
        g_f = self.g_omega_faces(theta)

        f_list = sorted(self.faces, key=lambda face: g_f[face])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return f_list, g_f
        else:
            return f_list

    def plot_faces(self, theta=None, ax=None, **kwargs):
        """
        Plots the faces of the graph in the direction of theta.

        Parameters:
            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to sort the edges.
            ax (matplotlib.axes.Axes):
                The axes to plot the graph on. If None, a new figure is created.
            **kwargs:
                Additional keyword arguments to pass to the ax.fill function.

        Returns:
            matplotlib.axes.Axes
                The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for face in self.faces:
            face_coords = np.array([self.coordinates[v] for v in face])
            ax.fill(face_coords[:, 0], face_coords[:, 1], **kwargs)

        return ax

    def plot(self, bounding_circle=False, color_nodes_theta=None, ax=None, **kwargs):
        """
        Plots the graph with the faces filled in.

        Returns:
            matplotlib.axes.Axes
                The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax = self.plot_faces(0, facecolor='lightblue', ax=ax)
        ax = super().plot(bounding_circle=bounding_circle,
                          color_nodes_theta=color_nodes_theta, ax=ax)
        return ax


def create_example_cw(centered=True, center_type='min_max'):
    """
    Creates an example EmbeddedCW object with a simple CW complex. If centered is True, the coordinates are centered around the center type, which could be ``mean``, ``min_max`` or ``origin``.


    Returns:
        EmbeddedCW
            The example EmbeddedCW object.
    """
    G = create_example_graph(centered=False)
    K = EmbeddedCW()
    K.add_from_embedded_graph(G)
    K.add_node('G', 2, 4)
    K.add_node('H', 1, 5)
    K.add_node('I', 5, 4)
    K.add_node('J', 2, 2)
    K.add_node('K', 2, 7)
    K.add_edges_from([('G', 'A'), ('G', 'H'), ('H', 'D'), ('I', 'E'),
                     ('I', 'C'), ('J', 'E'), ('K', 'D'), ('K', 'C')])
    K.add_face(['B', 'A', 'G', 'H', 'D'])
    K.add_face(['K', 'D', 'C'])

    if centered:
        K.set_centered_coordinates(type=center_type)

    return K
