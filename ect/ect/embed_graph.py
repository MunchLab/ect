import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import next_vert_name
from typing import Dict, List, Tuple


CENTER_TYPES = ["mean", "bounding_box"]
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

    @property
    def coord_matrix(self):
        """Return the N x D coordinate matrix."""
        return self._coord_matrix

    @property
    def dim(self):
        """Return the dimension of the embedded coordinates"""
        return self._coord_matrix.shape[1] if self._coord_matrix.size > 0 else 0

    @property
    def node_list(self):
        """Return ordered list of node names."""
        return self._node_list

    # ======================================
    # Node Management
    # ======================================
    def _validate_coordinates(func):
        def wrapper(self, *args, **kwargs):
            coords = next((arg for arg in args if isinstance(
                arg, (list, np.ndarray))), None)
            if coords is not None:
                coords = np.array(coords, dtype=float)
                if coords.ndim != 1:
                    raise ValueError("Coordinates must be a 1D array")

                if len(self._node_list) == 0:
                    self.dim = coords.size
                    self._coord_matrix = np.empty((0, self.dim))
                elif coords.size != self.dim:
                    raise ValueError(
                        f"Coordinates must have dimension {self.dim}")

            return func(self, *args, **kwargs)
        return wrapper

    def _validate_node(exists=True):
        def decorator(func):
            def wrapper(self, node_id, *args, **kwargs):
                node_exists = node_id in self._node_to_index
                if exists and not node_exists:
                    raise ValueError(f"Node {node_id} does not exist")
                if not exists and node_exists:
                    raise ValueError(f"Node {node_id} already exists")
                return func(self, node_id, *args, **kwargs)
            return wrapper
        return decorator

    @_validate_coordinates
    @_validate_node(exists=False)
    def add_node(self, node_id, coordinates):
        """Add a vertex to the graph. 
        If the vertex name is given as None, it will be assigned via the next_vert_name method.

        Parameters:
            node_id (hashable like int or str, or None) : The name of the vertex to add.
            coordinates (array-like) : The coordinates of the vertex being added.
        """
        self._node_list.append(node_id)
        self._node_to_index[node_id] = len(self._node_list) - 1
        self._coord_matrix = np.vstack([self._coord_matrix, coords])
        super().add_node(node_id)

    def add_nodes_from(self, nodes_with_coords):
        for node_id, coords in nodes_with_coords:
            self.add_node(node_id, coords)

    # ======================================
    # Coordinate Access
    # ======================================

    def get_coordinates(self, node_id):
        """Return the coordinates of a node"""
        return self._coord_matrix[self._node_to_index[node_id]].copy()

    @_validate_coordinates
    @_validate_node(exists=True)
    def set_coordinates(self, node_id, new_coords):
        """Set the coordinates of a node"""
        idx = self._node_to_index[node_id]
        self._coord_matrix[idx] = new_coords

    # ======================================
    # Graph Operations
    # ======================================

    def add_cycle(self, coord_matrix):
        """Add nodes in a cyclic pattern from coordinate matrix"""
        n = coord_matrix.shape[0]
        new_names = next_vert_name(
            self._node_list[-1] if self._node_list else 0, n)
        self.add_nodes_from(zip(new_names, coord_matrix))
        self.add_edges_from([(new_names[i], new_names[(i+1) % n])
                            for i in range(n)])

    # ======================================
    # Geometric Calculations
    # ======================================

    def get_center(self, method: str = 'bounding_box') -> np.ndarray:
        """Calculate center of coordinates"""

        coords = self._coord_matrix
        if method == 'mean':
            return np.mean(coords, axis=0)
        elif method == 'bounding_box':
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
        coords = self._coord_matrix
        return np.max(np.linalg.norm(coords - center, axis=1))

    def get_normal_angles(self, edges_only=False, decimals=6):
        """
        Get angles where edge order changes (2D only).

        Args:
            edges_only: Only compute angles between vertices connected by edges
            decimals: Number of decimal places to round angles to

        Returns:
            Dictionary mapping angles to lists of vertex pairs
        """
        if self.dim != 2:
            raise ValueError("Angle calculations require 2D coordinates")

        vertices = list(self.nodes())
        coords = self.get_coords_array()
        n = len(vertices)

        angles = {}

        if edges_only:
            edges = np.array(list(self.edges()))
            idx1 = np.array([vertices.index(u) for u, _ in edges])
            idx2 = np.array([vertices.index(v) for _, v in edges])

            diffs = coords[idx2] - coords[idx1]

            edge_angles = np.arctan2(diffs[:, 0], -diffs[:, 1]) % (2*np.pi)
            edge_angles = np.round(edge_angles, decimals)

            for i, angle in enumerate(edge_angles):
                pair = (vertices[idx1[i]], vertices[idx2[i]])
                if angle in angles:
                    angles[angle].append(pair)
                else:
                    angles[angle] = [pair]

        else:
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

            norms = np.linalg.norm(diffs, axis=2)
            valid = ~np.isclose(norms, 0)

            with np.errstate(divide='ignore', invalid='ignore'):
                all_angles = np.arctan2(
                    diffs[..., 0], -diffs[..., 1]) % (2*np.pi)
            all_angles[~valid] = np.nan
            all_angles = np.round(all_angles, decimals)

            for i in range(n):
                for j in range(i+1, n):
                    if valid[i, j]:
                        angle = all_angles[i, j]
                        pair = (vertices[i], vertices[j])
                        if angle in angles:
                            angles[angle].append(pair)
                        else:
                            angles[angle] = [pair]

        return angles

    def get_normal_angles_matrix(self, edges_only=False, decimals=6):
        """
        Get angles where edge order changes (2D only).
        Vectorized implementation for efficiency.
        """
        if self.dim != 2:
            raise ValueError("Angle calculations require 2D coordinates")

    # ============================
    # Coordinate transformations
    # ============================

    def transform_coordinates(self, center_type="bounding_box", projection_type="pca"):
        """Transform coordinates center and orientation"""
        if projection_type not in TRANSFORM_TYPES:
            raise ValueError(f"Unknown transform type: {projection_type}")
        self.project_coordinates(projection_type)
        if center_type not in CENTER_TYPES:
            raise ValueError(f"Unknown center method: {center_type}")
        self.center_coordinates(center_type)

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

    def project_coordinates(self, projection_type="pca"):
        """Project coordinates using a function"""
        if projection_type == "pca":
            self.pca_projection()
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def pca_projection(self, target_dim=2):
        """Dimensionality reduction using PCA"""
        if self.dim <= target_dim:
            return

        pca = PCA(n_components=target_dim)
        self._coord_matrix = pca.fit_transform(self._coord_matrix)
        self.dim = target_dim

    @_validate_node(exists=True)
    def add_edge(self, node_id1, node_id2):
        """
        Adds an edge between the vertices node_id1 and node_id2 if they exist.

        Parameters:
            node_id1 (str):
                The first vertex of the edge.
            node_id2 (str):
                The second vertex of the edge.

        """
        super().add_edge(node_id1, node_id2)

    # ===================
    # Visualization
    # ===================
    def plot(self, projection=None, ax=None, **kwargs):
        """2D visualization with optional PCA projection"""
        if self.dim >= 3:
            if projection is None:
                raise ValueError(
                    "Require 2D coordinates or specify projection")
            self.pca_projection(target_dim=3)

        if ax is None:
            fig, ax = plt.subplots()

        pos = {n: self._coord_matrix[i]
               for i, n in enumerate(self._node_list)}
        nx.draw(self, pos, ax=ax, **kwargs)
        ax.set_aspect('equal')
        return ax
