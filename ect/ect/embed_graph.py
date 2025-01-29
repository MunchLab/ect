import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import next_vert_name


CENTER_TYPES = ["mean", "min_max"]
TRANSFORM_TYPES = ["pca"]


class EmbeddedGraph(nx.Graph):
    """
    A class to represent a graph with embedded coordinates for each vertex.

    Attributes
        graph : nx.Graph
            a NetworkX graph object
        coord_matrix : np.ndarray
            a matrix of embedded coordinates for each vertex
        node_list : list
            a list of node names
        node_to_index : dict
            a dictionary mapping node ids to their index in the coord_matrix  
        dim : int
            the dimension of the embedded coordinates

    """

    def __init__(self):
        super().__init__()
        self._node_list = []
        self._node_to_index = {}
        self._coord_matrix = np.empty((0, 0))
        self.dim = 0

    # ======================================
    # Core Node Management
    # ======================================
    def add_node(self, node_id, coordinates):
        """Add a vertex to the graph. 
        If the vertex name is given as None, it will be assigned via the next_vert_name method.

        Parameters:
            node_id (hashable like int or str, or None) : The name of the vertex to add.
            coordinates (array-like) : The coordinates of the vertex being added.
        """
        if node_id in self._node_to_index:
            raise ValueError(f"Node {node_id} already exists.")

        coords = np.array(coordinates, dtype=float)
        if coords.ndim != 1:
            raise ValueError("Coordinates must be a 1D array.")

        if len(self._node_list) == 0:
            self.dim = coords.size
            self._coord_matrix = np.empty((0, self.dim))

        if coords.size != self.dim:
            raise ValueError(f"Coordinates must have dimension {self.dim}.")

        self._node_list.append(node_id)
        self._node_to_index[node_id] = len(self._node_list) - 1
        self._coord_matrix = np.vstack([self._coord_matrix, coords])
        super().add_node(node_id)

    def add_nodes_from(self, coords):
        for node_id, coords in nodes_with_coords:
            self.add_node(node_id, coords)

    # ======================================
    # Coordinate Access
    # ======================================

    @property
    def coord_matrix(self):
        """Return the N x D coordinate matrix."""
        return self._coord_matrix

    def get_coordinates(self, node_id):
        """Return the coordinates of a node"""
        return self._coord_matrix[self._node_to_index[node_id]].copy()

    def set_coordinates(self, node_id, new_coords):
        """Set the coordinates of a node"""
        if node_id not in self._node_to_index:
            raise ValueError(f"Node {node_id} does not exist.")
        if new_coords.shape != (self.dim,):
            raise ValueError(f"Coordinates must be {self.dim}-dimensional")

        idx = self._node_to_index[node_id]
        self._coord_matrix[idx] = new_coords

    @property
    def node_list(self):
        """Return ordered list of node names."""
        return self._node_list

    # ======================================
    # Graph Operations
    # ======================================

    def add_cycle(self, coord_matrix):
        """Add nodes in a cyclic pattern from coordinate matrix"""
        n = coord_matrix.shape[0]
        new_names = self.next_vert_name(
            self._node_list[-1] if self._node_list else 0, n)
        self.add_nodes_from(zip(new_names, coord_matrix))
        self.add_edges_from([(new_names[i], new_names[(i+1) % n])
                            for i in range(n)])

    # ======================================
    # Geometric Calculations
    # ======================================

    def get_center(self, method: str = 'mean') -> np.ndarray:
        """Calculate center of coordinates"""

        coords = self.get_coords_array()
        if method == 'mean':
            return np.mean(coords, axis=0)
        elif method == 'min_max':
            return (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
        elif method == 'origin':
            return np.zeros(self.dim)
        raise ValueError(f"Unknown center method: {method}")

    def get_bounding_box(self):
        """Get (min, max) for each dimension"""
        return [(dim.min(), dim.max()) for dim in self._coord_matrix.T]

    def get_bounding_radius(self, center_type: str = 'mean') -> float:
        """Get radius of minimal bounding sphere"""
        center = self.get_center(center_type)
        coords = self.get_coords_array()
        return np.max(np.linalg.norm(coords - center, axis=1))

    def get_normal_angles(self, edges_only=False, decimals=6):
        """Get angles where edge order changes (2D only)"""
        if self.dim != 2:
            raise ValueError("Angle calculations require 2D coordinates")

        angles = {}
        coords = self._coord_matrix

        # vectorized angle calculation
        vecs = coords[:, None, :] - coords[None, :, :]
        norms = np.linalg.norm(vecs, axis=2)
        valid = ~np.isclose(norms, 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            angles_rad = np.arctan2(vecs[:, :, 0], -vecs[:, :, 1]) % (2*np.pi)
        angles_rad[~valid] = np.nan

        for i in range(coords.shape[0]):
            for j in range(i+1, coords.shape[0]):
                if edges_only and not self.has_edge(self._node_list[i], self._node_list[j]):
                    continue

                angle = np.round(angles_rad[i, j], decimals)
                pair = (self._node_list[i], self._node_list[j])

                if angle in angles:
                    angles[angle].append(pair)
                else:
                    angles[angle] = [pair]

        return angles

    # ============================
    # Coordinate transformations
    # ============================

    def transform_coordinates(self, center_type=None, transform_type=None):
        """Transform coordinates center and orientation"""
        if center_type is not None:
            self.center_coordinates(center_type)
        if transform_type is not None:
            self.transform_coordinates(transform_type)

    def center_coordinates(self, center_type="mean"):
        if center_type not in CENTER_TYPES:
            raise ValueError(f"Unknown center method: {center_type}")

        center = self.get_center(center_type)
        self._coord_matrix -= center

    def scale_coordinates(self, radius=1):
        """Scale coordinates to fit within given radius"""
        current_max = np.linalg.norm(self._coord_matrix, axis=1).max()
        if current_max > 0:
            self._coord_matrix *= (radius / current_max)

    def transform_coordinates(self, transform_type: str):
        """Transform coordinates using a function"""
        if transform_type == "pca":
            self.pca_transform()
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def pca_transform(self, target_dim=2):
        """Dimensionality reduction using PCA"""
        if self.dim <= target_dim:
            return

        pca = PCA(n_components=target_dim)
        self._coord_matrix = pca.fit_transform(self._coord_matrix)
        self.dim = target_dim

    def add_cycle(self, coords: np.ndarray):
        """Add nodes in a cycle from coordinates"""
        n = len(coords)
        node_ids = [str(i) for i in range(n)]

        # Add nodes
        for node_id, coord in zip(node_ids, coords):
            self.add_node(node_id, coords=coord)

        # Add edges
        edges = [(node_ids[i], node_ids[(i+1) % n]) for i in range(n)]
        self.add_edges_from(edges)

    def add_edge(self, u, v):
        """
        Adds an edge between the vertices u and v if they exist.

        Parameters:
            u (str):
                The first vertex of the edge.
            v (str):
                The second vertex of the edge.

        """
        if not self.has_node(u) or not self.has_node(v):
            raise ValueError("One or both vertices do not exist in the graph.")
        else:
            super().add_edge(u, v)

    def get_bounding_box(self):
        if self._coord_matrix.size == 0:
            return None
        return [(dim_min, dim_max) for dim_min, dim_max in zip(
            self._coord_matrix.min(axis=0),
            self._coord_matrix.max(axis=0)
        )]

    # ===================
    # Visualization
    # ===================
    def plot(self, projection=None, ax=None, **kwargs):
        """2D visualization with optional PCA projection"""
        if self.dim >= 3:
            if projection is None:
                raise ValueError(
                    "Require 2D coordinates or specify projection")
            self.apply_pca(target_dim=3)

        if ax is None:
            fig, ax = plt.subplots()

        pos = {n: self._coord_matrix[i]
               for i, n in enumerate(self._node_list)}
        nx.draw(self, pos, ax=ax, **kwargs)
        ax.set_aspect('equal')
        return ax
