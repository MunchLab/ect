import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .embed_graph import EmbeddedGraph
from .utils.face_check import point_in_polygon
from typing import Optional


class EmbeddedCW(EmbeddedGraph):
    """
    A class to represent a straight-line-embedded CW complex. We assume that the coordinates for the embedding of the vertices are given, the 1-skeleton is in fact a graph (so not as general as a full CW complex) with straight line embeddings, and 2-Cells are the interior of the shape outlined by its boundary edges.

    Faces should be passed in as a list of vertices, where the vertices are in order around the face. However, the ECT function will likely still work if the ordering is different. The drawing functions however might look strange. Note the class does not (yet?) check to make sure the face is valid, i.e. is a cycle in the graph, and bounds a region in the plane.

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedCW object.
        """

        super().__init__()
        self.faces = []

    def add_from_embedded_graph(self, G):
        """
        Adds the edges and coordinates from an EmbeddedGraph object to the EmbeddedCW object.

        Parameters:
            G (EmbeddedGraph):
                The EmbeddedGraph object to add from.
        """
        nodes_with_coords = [
            (node, G.coord_matrix[G.node_to_index[node]]) for node in G.nodes()
        ]
        self.add_nodes_from(nodes_with_coords)
        self.add_edges_from(G.edges())

    @EmbeddedGraph._validate_node(exists=True)
    def add_face(self, face, check=False):
        """
        Adds a face to the list of faces.

        Parameters:
            face (list):
                A list of vertices that make up the face.
            check (bool):
                Whether to check that the face is a valid addition to the cw complex.
        """
        if len(face) < 3:
            raise ValueError("Face must have at least 3 vertices")

        if check:
            edges = list(zip(face, face[1:] + [face[0]]))
            for u, v in edges:
                if not self.has_edge(u, v):
                    raise ValueError(f"Edge ({u},{v}) missing")

            polygon = np.array(
                [self.coord_matrix[self._node_to_index[v]] for v in face]
            )
            for node in self.nodes:
                if node in face:
                    continue
                if point_in_polygon(
                    self.coord_matrix[self._node_to_index[node]], polygon
                ):
                    raise ValueError(f"Node {node} inside face {face}")

        self.faces.append(tuple(face))

    def add_faces_from(self, faces):
        """
        Adds a list of faces to the graph.

        Parameters:
            faces (list):
                A list of faces to add.
        """
        for face in faces:
            self.add_face(face)

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
            face_coords = np.array(
                [self.coord_matrix[self.node_to_index[v]] for v in face]
            )

            if self.dim == 2:
                ax.fill(face_coords[:, 0], face_coords[:, 1], **kwargs)
            else:  # 3D case
                verts = [face_coords]
                collection = Poly3DCollection(verts, **kwargs)
                ax.add_collection3d(collection)

        return ax

    def plot(
        self,
        bounding_circle: bool = False,
        bounding_center_type: str = "bounding_box",
        color_nodes_theta: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
        with_labels: bool = True,
        node_size: int = 300,
        edge_color: str = "gray",
        elev: float = 25,
        azim: float = -60,
        face_color: str = "lightblue",
        face_alpha: float = 0.3,
        **kwargs,
    ) -> plt.Axes:
        """
        Visualize the embedded CW complex in 2D or 3D, including faces
        """
        # plot faces then use parent class
        ax = self._create_axes(ax, self.dim)
        self.plot_faces(ax=ax, facecolor=face_color, alpha=face_alpha)

        return super().plot(
            bounding_circle=bounding_circle,
            bounding_center_type=bounding_center_type,
            color_nodes_theta=color_nodes_theta,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            edge_color=edge_color,
            elev=elev,
            azim=azim,
            **kwargs,
        )
