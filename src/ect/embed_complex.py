from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA

from .utils.naming import next_vert_name
from .utils.face_check import (
    point_in_polygon,
    validate_face_embedding,
    validate_edge_embedding,
)
from .validation import EmbeddingValidator, ValidationRule


CENTER_TYPES = ["mean", "bounding_box", "origin"]
TRANSFORM_TYPES = ["pca"]


class EmbeddedComplex(nx.Graph):
    """
    A unified class to represent an embedded cell complex with cells of arbitrary dimension.

    This combines the functionality of EmbeddedGraph and EmbeddedCW, supporting:
    - 0-cells (vertices) with embedded coordinates
    - 1-cells (edges)
    - k-cells for k >= 2 (faces, volumes, etc.)

    Args:
        validate_embedding (bool): If True, automatically validate embedding properties
            when adding cells. Default: False
        embedding_tol (float): Tolerance for geometric validation. Default: 1e-10

    Attributes:
        coord_matrix : np.ndarray
            A matrix of embedded coordinates for each vertex
        node_list : list
            A list of node names
        node_to_index : dict
            A dictionary mapping node ids to their index in the coord_matrix
        dim : int
            The dimension of the embedded coordinates
        cells : dict
            Dictionary mapping dimension k to list of k-cells, where each k-cell
            is represented as a tuple of vertex indices
        validate_embedding : bool
            Whether to automatically validate embedding properties
        embedding_tol : float
            Tolerance for geometric validation
    """

    def __init__(self, validate_embedding=False, embedding_tol=1e-10):
        super().__init__()
        self._node_list = []
        self._node_to_index = {}
        self._coord_matrix = None
        self.cells = defaultdict(list)

        self.validate_embedding = validate_embedding
        self.embedding_tol = embedding_tol

        def edge_checker(v1_idx: int, v2_idx: int) -> bool:
            # closure to check if edge exists by converting indices back to node names
            if v1_idx >= len(self._node_list) or v2_idx >= len(self._node_list):
                return False
            v1_name = self._node_list[v1_idx]
            v2_name = self._node_list[v2_idx]
            return self.has_edge(v1_name, v2_name)

        self._validator = EmbeddingValidator(embedding_tol, edge_checker)

    @property
    def coord_matrix(self):
        """Return the N x D coordinate matrix"""
        if self._coord_matrix is None:
            return np.empty((0, 0))
        return self._coord_matrix

    @property
    def dim(self):
        """Return the dimension of the embedded coordinates"""
        if self._coord_matrix is None:
            return 0
        return self._coord_matrix.shape[1]

    @property
    def node_list(self):
        """Return ordered list of node names."""
        return self._node_list

    @property
    def node_to_index(self):
        """Return a dictionary mapping node ids to their index in the coord_matrix"""
        return self._node_to_index

    @property
    def position_dict(self):
        """Return a dictionary mapping node ids to their coordinates"""
        return {node: self._coord_matrix[i] for i, node in enumerate(self._node_list)}

    @property
    def edge_indices(self):
        """Return edges as array of index pairs"""
        edges = np.array(
            [(self._node_to_index[u], self._node_to_index[v]) for u, v in self.edges()],
            dtype=int,
        )
        return edges if len(edges) > 0 else np.empty((0, 2), dtype=int)

    @property
    def faces(self):
        """Return list of 2-cells (faces) for backward compatibility"""
        return [
            tuple(self._node_list[i] for i in cell) for cell in self.cells.get(2, [])
        ]

    def add_node(self, node_id, coord):
        """Add a vertex to the complex.

        Args:
            node_id: Identifier for the node
            coord: Array-like coordinates for the node
        """
        # validate coordinates using validator
        expected_dim = (
            self._coord_matrix.shape[1] if self._coord_matrix is not None else None
        )
        coord_result = self._validator.validate_coordinates(coord, expected_dim)
        if not coord_result.is_valid:
            raise ValueError(coord_result.message)

        # validate node doesn't already exist
        node_result = self._validator.validate_nodes(
            [node_id], lambda n: n in self._node_to_index, expect_exists=False
        )
        if not node_result.is_valid:
            raise ValueError(node_result.message)

        coord = np.asarray(coord, dtype=float)

        if len(self._node_list) == 0:
            # initialize coordinate matrix with first node
            self._coord_matrix = coord.reshape(1, -1)
        else:
            # append new coordinate as row
            coord_reshaped = coord.reshape(1, -1)
            self._coord_matrix = np.vstack([self._coord_matrix, coord_reshaped])

        self._node_list.append(node_id)
        self._node_to_index[node_id] = len(self._node_list) - 1
        super().add_node(node_id)

    def add_nodes_from_dict(self, nodes_with_coords: Dict[Union[str, int], np.ndarray]):
        for node_id, coordinates in nodes_with_coords.items():
            self.add_node(node_id, coordinates)

    def add_nodes_from(
        self, nodes_with_coords: List[Tuple[Union[str, int], np.ndarray]]
    ):
        for node_id, coordinates in nodes_with_coords:
            self.add_node(node_id, coordinates)

    def add_edge(self, node_id1, node_id2):
        """Add an edge (1-cell) between two nodes"""
        # validate nodes exist
        node_result = self._validator.validate_nodes(
            [node_id1, node_id2], lambda n: n in self._node_to_index, expect_exists=True
        )
        if not node_result.is_valid:
            raise ValueError(node_result.message)

        super().add_edge(node_id1, node_id2)

    def add_cell(
        self,
        cell_vertices: List,
        dim: Optional[int] = None,
        check: Optional[bool] = None,
        embedding_tol: Optional[float] = None,
    ):
        """
        Add a k-cell to the complex.

        Args:
            cell_vertices: List of vertex identifiers that form the cell
            dim: Dimension of the cell. If None, inferred as len(cell_vertices) - 1
            check: Whether to validate the cell embedding. If None, uses self.validate_embedding
            embedding_tol: Tolerance for geometric validation. If None, uses self.embedding_tol
        """
        if check is None:
            check = self.validate_embedding
        if embedding_tol is None:
            embedding_tol = self.embedding_tol
        if dim is None:
            dim = len(cell_vertices) - 1

        # check vertex existence before validation (can't validate non-existent vertices)
        missing_vertices = [v for v in cell_vertices if v not in self._node_to_index]
        if missing_vertices:
            raise ValueError(f"Vertices do not exist: {missing_vertices}")

        # convert vertex names to indices for storage
        cell_indices = tuple(self._node_to_index[v] for v in cell_vertices)

        cell_coords = (
            self._coord_matrix[list(cell_indices)]
            if self._coord_matrix is not None
            else None
        )
        all_coords = self._coord_matrix
        all_indices = list(range(len(self._node_list)))

        # check structural rules (vertex count, dimension validity)
        structural_result = self._validator.validate_cell(
            cell_coords,
            all_coords,
            list(cell_indices),
            all_indices,
            dim,
            check_geometric=False,
        )
        if not structural_result.is_valid:
            raise ValueError(structural_result.message)

        # check geometric rules (embedding properties)
        if check:
            geometric_result = self._validator.validate_cell(
                cell_coords,
                all_coords,
                list(cell_indices),
                all_indices,
                dim,
                check_geometric=True,
            )
            if not geometric_result.is_valid:
                raise ValueError(geometric_result.message)

        # update graph structure
        if dim == 1:
            self.add_edge(cell_vertices[0], cell_vertices[1])

        self.cells[dim].append(cell_indices)

    def enable_embedding_validation(self, tol: float = 1e-10):
        """
        Enable automatic embedding validation for all subsequent cell additions.

        Args:
            tol: Tolerance for geometric validation
        """
        self.validate_embedding = True
        self.embedding_tol = tol
        self._validator.set_tolerance(tol)

    def disable_embedding_validation(self):
        """Disable automatic embedding validation for all subsequent cell additions."""
        self.validate_embedding = False

    def get_validator(self) -> "EmbeddingValidator":
        """
        Get the embedding validator instance for advanced configuration.

        Returns:
            The EmbeddingValidator instance used by this complex
        """
        return self._validator

    def set_validation_rules(self, rules: List["ValidationRule"]) -> "EmbeddedComplex":
        """
        Set custom validation rules.

        Args:
            rules: List of ValidationRule instances

        Returns:
            Self for method chaining
        """
        # replace validation rules
        self._validator.rules = rules
        return self

    def add_face(self, face: List, check: Optional[bool] = None):
        """Add a 2-cell (face) to the complex. Provided for backward compatibility."""
        self.add_cell(face, dim=2, check=check)

    def add_faces_from(self, faces: List[List]):
        """Add multiple 2-cells (faces) to the complex."""
        for face in faces:
            self.add_face(face)

    def get_coord(self, node_id):
        """Return the coordinates of a node"""
        # validate node exists
        node_result = self._validator.validate_nodes(
            [node_id], lambda n: n in self._node_to_index, expect_exists=True
        )
        if not node_result.is_valid:
            raise ValueError(node_result.message)

        return self._coord_matrix[self._node_to_index[node_id]].copy()

    def set_coord(self, node_id, new_coords):
        """Set the coordinates of a node"""
        # validate coordinates
        expected_dim = (
            self._coord_matrix.shape[1] if self._coord_matrix is not None else None
        )
        coord_result = self._validator.validate_coordinates(new_coords, expected_dim)
        if not coord_result.is_valid:
            raise ValueError(coord_result.message)

        # validate node exists
        node_result = self._validator.validate_nodes(
            [node_id], lambda n: n in self._node_to_index, expect_exists=True
        )
        if not node_result.is_valid:
            raise ValueError(node_result.message)

        idx = self._node_to_index[node_id]
        self._coord_matrix[idx] = new_coords

    def add_cycle(self, coord_matrix):
        """Add nodes in a cyclic pattern from coordinate matrix"""
        # generate sequential node names and add cyclic edges
        n = coord_matrix.shape[0]
        new_names = next_vert_name(self._node_list[-1] if self._node_list else 0, n)
        self.add_nodes_from(zip(new_names, coord_matrix))
        self.add_edges_from([(new_names[i], new_names[(i + 1) % n]) for i in range(n)])

    def get_center(self, method: str = "bounding_box") -> np.ndarray:
        """Calculate center of coordinates"""

        coords = self._coord_matrix
        if coords is None or coords.size == 0:
            return np.zeros(0)

        if method == "mean":
            return np.mean(coords, axis=0)
        elif method == "bounding_box":
            return (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
        elif method == "origin":
            return np.zeros(self.dim)
        raise ValueError(f"Unknown center method: {method}")

    def get_bounding_box(self):
        """Get (min, max) for each dimension"""
        return [(dim.min(), dim.max()) for dim in self._coord_matrix.T]

    def get_bounding_radius(self, center_type: str = "bounding_box") -> float:
        """Get radius of minimal bounding sphere"""
        coords = self._coord_matrix
        if coords is None or coords.size == 0:
            return 0.0

        center = self.get_center(center_type)
        return np.max(np.linalg.norm(coords - center, axis=1))

    def get_normal_angle_matrix(
        self, edges_only: bool = False, decimals: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Optimized angle matrix computation using vectorized operations.

        Args:
            edges_only: Only compute angles between connected vertices
            decimals: Round angles to specified decimal places

        Returns:
            angle_matrix: NaN-filled matrix with pair angles
            vertex_labels: Ordered node identifiers
        """
        coords = self._coord_matrix
        vertices = self._node_list
        n = len(vertices)

        angle_matrix = np.full((n, n), np.nan, dtype=np.float64)

        if edges_only:
            # compute angles only for connected vertex pairs
            edges = np.array(list(self.edges()))
            if edges.size == 0:
                return angle_matrix, vertices

            u_indices = np.vectorize(self._node_to_index.get)(edges[:, 0])
            v_indices = np.vectorize(self._node_to_index.get)(edges[:, 1])

            dx = coords[v_indices, 0] - coords[u_indices, 0]
            dy = coords[v_indices, 1] - coords[u_indices, 1]

            # angles from u to v and reverse direction
            angles = np.arctan2(dx, -dy) % (2 * np.pi)
            rev_angles = (angles + np.pi) % (2 * np.pi)

            if decimals is not None:
                angles = np.round(angles, decimals)
                rev_angles = np.round(rev_angles, decimals)

            angle_matrix[u_indices, v_indices] = angles
            angle_matrix[v_indices, u_indices] = rev_angles

        else:
            # compute angles between all vertex pairs using broadcasting
            x = coords[:, 0]
            y = coords[:, 1]

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]

            angle_matrix = np.arctan2(dx, -dy) % (2 * np.pi)
            # nan for coincident points
            angle_matrix[np.isclose(dx**2 + dy**2, 0)] = np.nan

            if decimals is not None:
                angle_matrix = np.round(angle_matrix, decimals)

            np.fill_diagonal(angle_matrix, np.nan)

        return angle_matrix, vertices

    def get_normal_angles(
        self, edges_only: bool = False, decimals: int = 6
    ) -> Dict[float, List[Tuple[str, str]]]:
        """
        Optimized angle dictionary construction using NumPy grouping.

        Args:
            edges_only: Only include edge-connected pairs
            decimals: Round angles to specified decimal places

        Returns:
            Dictionary mapping rounded angles to vertex pairs
        """
        angle_matrix, vertices = self.get_normal_angle_matrix(edges_only, decimals)
        n = len(vertices)

        # extract upper triangle to avoid duplicate pairs
        rows, cols = np.triu_indices(n, k=1)
        angles = angle_matrix[rows, cols]
        valid_mask = ~np.isnan(angles)

        if not valid_mask.any():
            return defaultdict(list)

        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_angles = angles[valid_mask]

        # group vertex pairs by their angle
        angle_dict = defaultdict(list)
        unique_angles, inverse = np.unique(valid_angles, return_inverse=True)

        for idx, angle in enumerate(unique_angles):
            mask = inverse == idx
            pairs = [
                (vertices[i], vertices[j])
                for i, j in zip(valid_rows[mask], valid_cols[mask])
            ]
            angle_dict[float(angle)].extend(pairs)

        return angle_dict

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
        # scale so largest distance from origin equals target radius
        current_max = np.linalg.norm(self._coord_matrix, axis=1).max()
        if current_max > 0:
            self._coord_matrix *= radius / current_max

    def project_coordinates(self, projection_type="pca"):
        """Project coordinates using a function"""
        if projection_type == "pca":
            self.pca_projection()
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def pca_projection(self, target_dim=2):
        """Dimensionality reduction using PCA"""
        # only reduce dimension
        if self.dim <= target_dim:
            return

        pca = PCA(n_components=target_dim)
        self._coord_matrix = pca.fit_transform(self._coord_matrix)

    def validate_plot_parameters(func):
        # decorator to check plotting requirements
        def wrapper(self, *args, **kwargs):
            bounding_center_type = kwargs.get("bounding_center_type", "bounding_box")

            if self.dim not in [2, 3]:
                raise ValueError("At least 2D or 3D coordinates required for plotting")

            if bounding_center_type not in CENTER_TYPES:
                raise ValueError(
                    f"Invalid center type: {bounding_center_type}. "
                    f"Valid options: {CENTER_TYPES}"
                )

            return func(self, *args, **kwargs)

        return wrapper

    def plot_faces(self, ax=None, **kwargs):
        """
        Plots the 2-cells (faces) of the complex.

        Parameters:
            ax (matplotlib.axes.Axes):
                The axes to plot the graph on. If None, a new figure is created.
            **kwargs:
                Additional keyword arguments to pass to the ax.fill function.

        Returns:
            matplotlib.axes.Axes
                The axes object with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()

        # render each 2-cell as filled polygon
        for cell_indices in self.cells.get(2, []):
            face_coords = self._coord_matrix[list(cell_indices)]

            if self.dim == 2:
                ax.fill(face_coords[:, 0], face_coords[:, 1], **kwargs)
            else:
                # 3d faces need polygon collection
                verts = [face_coords]
                collection = Poly3DCollection(verts, **kwargs)
                ax.add_collection3d(collection)

        return ax

    @validate_plot_parameters
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
        Visualize the embedded complex in 2D or 3D
        """
        ax = self._create_axes(ax, self.dim)

        if 2 in self.cells and len(self.cells[2]) > 0:
            self.plot_faces(ax=ax, facecolor=face_color, alpha=face_alpha)

        pos = {node: self._coord_matrix[i] for i, node in enumerate(self._node_list)}

        if self.dim == 2:
            self._draw_2d(ax, pos, with_labels, node_size, edge_color, **kwargs)
        else:
            self._draw_3d(ax, pos, node_size, edge_color, elev, azim, **kwargs)

        if color_nodes_theta is not None:
            # color nodes by projection in specified direction
            direction = np.array(
                [np.sin(color_nodes_theta), -np.cos(color_nodes_theta)]
            )
            node_colors = np.dot(self._coord_matrix, direction)
            self._add_node_coloring(ax, pos, node_colors, node_size, self.dim, **kwargs)

        if bounding_circle:
            self._add_bounding_shape(ax, bounding_center_type, self.dim)

        self._configure_axes(ax)

        return ax

    def _create_axes(self, ax, dim=None):
        """Create appropriate axes if not provided"""
        if dim is None:
            dim = self.dim

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
        elif dim == 3 and not hasattr(ax, "zaxis"):
            raise ValueError("For 3D plots, provide axes with 3D projection")
        return ax

    def _draw_2d(self, ax, pos, with_labels, node_size, edge_color, **kwargs):
        """2D visualization components"""
        nx.draw_networkx_edges(
            self, pos=pos, ax=ax, edge_color=edge_color, width=1.5, **kwargs
        )
        nx.draw_networkx_nodes(
            self,
            pos=pos,
            ax=ax,
            node_size=node_size,
            node_color="lightblue",
            edgecolors="black",
            linewidths=0.5,
            **kwargs,
        )
        if with_labels:
            nx.draw_networkx_labels(
                self, pos=pos, ax=ax, font_size=8, font_color="black", **kwargs
            )

    def _draw_3d(self, ax, pos, node_size, edge_color, elev, azim, **kwargs):
        """3D visualization components"""
        ax.view_init(elev=elev, azim=azim)

        coords = np.array(list(pos.values()))
        ax.scatter3D(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=node_size,
            c="lightblue",
            edgecolors="black",
            linewidth=0.5,
        )

        for u, v in self.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]
            ax.plot3D(x, y, z, color=edge_color, linewidth=1.5)

    def _add_node_coloring(self, ax, pos, node_colors, node_size, dim=None, **kwargs):
        """Add node coloring based on provided values"""
        if dim is None:
            dim = self.dim

        if dim == 2:
            # 2d colored nodes using networkx
            nx.draw_networkx_nodes(
                self,
                pos=pos,
                ax=ax,
                node_size=node_size,
                node_color=node_colors,
                cmap=plt.cm.viridis,
                edgecolors="black",
                linewidths=0.5,
                **kwargs,
            )
        else:
            # 3d colored scatter plot
            coords = np.array(list(pos.values()))
            ax.scatter3D(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                s=node_size,
                c=node_colors,
                cmap=plt.cm.viridis,
                edgecolors="black",
                linewidth=0.5,
                **kwargs,
            )

        norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
        cbar.set_label("Node Values")

    def _add_bounding_shape(self, ax, center_type="bounding_box", dim=None):
        """Add bounding circle/sphere visualization"""
        if dim is None:
            dim = self.dim

        center = self.get_center(center_type)
        radius = self.get_bounding_radius(center_type)

        if dim == 2:
            # draw bounding circle
            circle = plt.Circle(
                center[:2],
                radius,
                fill=False,
                linestyle="--",
                color="darkred",
                linewidth=1.2,
                alpha=0.7,
            )
            ax.add_patch(circle)
            padding = radius * 0.1
            ax.set_xlim(center[0] - radius - padding, center[0] + radius + padding)
            ax.set_ylim(center[1] - radius - padding, center[1] + radius + padding)
        else:
            # draw bounding sphere as wireframe
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

            ax.plot_wireframe(
                x, y, z, color="darkred", linewidth=0.5, alpha=0.3, rstride=2, cstride=2
            )
            padding = radius * 0.1
            ax.set_xlim3d(center[0] - radius - padding, center[0] + radius + padding)
            ax.set_ylim3d(center[1] - radius - padding, center[1] + radius + padding)
            ax.set_zlim3d(center[2] - radius - padding, center[2] + radius + padding)

    def _configure_axes(self, ax):
        """Finalize plot appearance"""
        if hasattr(ax, "zaxis"):
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            ax.set_aspect("equal")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            x_interval = self._get_nice_interval(xlim[1] - xlim[0])
            y_interval = self._get_nice_interval(ylim[1] - ylim[0])

            ax.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
            ax.yaxis.set_major_locator(plt.MultipleLocator(y_interval))

            ax.xaxis.set_minor_locator(plt.MultipleLocator(x_interval / 2))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(y_interval / 2))

        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            labelbottom=True,
            labelleft=True,
        )

    def _get_nice_interval(self, range_size):
        # calculate visually appealing tick spacing
        rough_interval = range_size / 5

        magnitude = 10 ** np.floor(np.log10(rough_interval))

        normalized = rough_interval / magnitude

        # choose from standard intervals: 1, 2, 5, 10
        if normalized < 1.5:
            nice_interval = 1
        elif normalized < 3:
            nice_interval = 2
        elif normalized < 7:
            nice_interval = 5
        else:
            nice_interval = 10

        return nice_interval * magnitude

    def _build_incidence_csr(self) -> tuple:
        """
        Build column sparse representation of the cell-to-vertex incidence excluding 0-cells. Format is (cell_vertex_pointers, cell_vertex_indices_flat, cell_euler_signs, n_vertices).
        Example: takes the complex [(1,3),(2,4),(1,2,3)] and returns [(0,2,4,7),(1,3,2,4,1,2,3),(-1,-1,1),4]

        """
        n_vertices = len(self.node_list)

        cells_by_dimension = {}

        if hasattr(self, "edge_indices") and self.edge_indices is not None:
            edge_indices_array = np.asarray(self.edge_indices)
            if edge_indices_array.size:
                cells_by_dimension[1] = [
                    tuple(map(int, row)) for row in edge_indices_array
                ]

        if hasattr(self, "cells") and self.cells:
            for dim, cells_of_dim in self.cells.items():
                if dim == 0:
                    continue
                if dim == 1 and 1 in cells_by_dimension:
                    continue
                if isinstance(cells_of_dim, np.ndarray):
                    cell_list = [tuple(map(int, row)) for row in cells_of_dim]
                else:
                    cell_list = [tuple(map(int, c)) for c in cells_of_dim]
                if len(cell_list) > 0:
                    cells_by_dimension[dim] = cell_list

        dimensions = sorted(cells_by_dimension.keys())
        n_cells = sum(len(cells_by_dimension[d]) for d in dimensions)

        cell_vertex_pointers = np.empty(n_cells + 1, dtype=np.int64)
        cell_euler_signs = np.empty(n_cells, dtype=np.int32)
        cell_vertex_indices_flat = []

        cell_vertex_pointers[0] = 0
        cell_index = 0
        for dim in dimensions:
            cells_in_dim = cells_by_dimension[dim]
            euler_sign = 1 if (dim % 2 == 0) else -1
            for cell_vertices in cells_in_dim:
                cell_vertex_indices_flat.extend(cell_vertices)
                cell_euler_signs[cell_index] = euler_sign
                cell_index += 1
                cell_vertex_pointers[cell_index] = len(cell_vertex_indices_flat)

        cell_vertex_indices_flat = np.asarray(cell_vertex_indices_flat, dtype=np.int32)
        return (
            cell_vertex_pointers,
            cell_vertex_indices_flat,
            cell_euler_signs,
            n_vertices,
        )


EmbeddedGraph = EmbeddedComplex
EmbeddedCW = EmbeddedComplex
