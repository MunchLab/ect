from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .utils.naming import next_vert_name


CENTER_TYPES = ["mean", "bounding_box", "origin"]
TRANSFORM_TYPES = ["pca"]


class EmbeddedGraph(nx.Graph):
    """
    A class to represent a graph with embedded coordinates for each vertex with simple geometric graph operations.

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
        self._coord_matrix = None

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
            [(self._node_to_index[u], self._node_to_index[v])
             for u, v in self.edges()],
            dtype=int,
        )
        return edges if len(edges) > 0 else np.empty((0, 2), dtype=int)

    # ======================================
    # Node Management
    # ======================================
    @staticmethod
    def _validate_coords(func):
        """Validates if coordinates are nonempty and have valid dimension"""

        def wrapper(self, *args, **kwargs):
            coords = next(
                (arg for arg in args if isinstance(arg, (list, np.ndarray))), None
            )
            if coords is not None:
                coords = np.asarray(coords, dtype=float)
                if coords.ndim != 1:
                    raise ValueError("Coordinates must be a 1D array")

                # Skip dimension check for first node
                if len(self._node_list) > 0:
                    if coords.size != self._coord_matrix.shape[1]:
                        raise ValueError(
                            f"Coordinates must have dimension {self._coord_matrix.shape[1]}"
                        )

            return func(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def _validate_node(exists=True):
        """Validates if nodes exist or not already"""

        def decorator(func):
            def wrapper(self, *args, **kwargs):
                # Handle both positional and keyword arguments
                if args:
                    nodes = args[0] if isinstance(
                        args[0], (list, tuple)) else [args[0]]
                else:
                    node_id = kwargs.get("node_id") or kwargs.get("node_id1")
                    nodes = [node_id] if node_id else []

                for node_id in nodes:
                    node_exists = node_id in self._node_to_index
                    if exists and not node_exists:
                        raise ValueError(f"Node {node_id} does not exist")
                    if not exists and node_exists:
                        raise ValueError(f"Node {node_id} already exists")
                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    @_validate_coords
    @_validate_node(exists=False)
    def add_node(self, node_id, coord):
        """Add a vertex to the graph.

        Args:
            node_id: Identifier for the node
            coord: Array-like coordinates for the node
        """
        coord = np.asarray(coord, dtype=float)

        if len(self._node_list) == 0:
            self._coord_matrix = coord.reshape(1, -1)
        else:
            coord_reshaped = coord.reshape(1, -1)
            self._coord_matrix = np.vstack(
                [self._coord_matrix, coord_reshaped])

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

    # ======================================
    # Coordinate Access
    # ======================================

    @_validate_node(exists=True)
    def get_coord(self, node_id):
        """Return the coordinates of a node"""
        return self._coord_matrix[self._node_to_index[node_id]].copy()

    @_validate_coords
    @_validate_node(exists=True)
    def set_coord(self, node_id, new_coords):
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
        self.add_edges_from(
            [(new_names[i], new_names[(i + 1) % n]) for i in range(n)])

    # ======================================
    # Geometric Calculations
    # ======================================

    def get_center(self, method: str = "bounding_box") -> np.ndarray:
        """Calculate center of coordinates"""

        coords = self._coord_matrix
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
        center = self.get_center(center_type)
        coords = self._coord_matrix
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
            edges = np.array(list(self.edges()))
            if edges.size == 0:
                return angle_matrix, vertices

            u_indices = np.vectorize(self._node_to_index.get)(edges[:, 0])
            v_indices = np.vectorize(self._node_to_index.get)(edges[:, 1])

            dx = coords[v_indices, 0] - coords[u_indices, 0]
            dy = coords[v_indices, 1] - coords[u_indices, 1]

            angles = np.arctan2(dx, -dy) % (2 * np.pi)
            rev_angles = (angles + np.pi) % (2 * np.pi)

            if decimals is not None:
                angles = np.round(angles, decimals)
                rev_angles = np.round(rev_angles, decimals)

            angle_matrix[u_indices, v_indices] = angles
            angle_matrix[v_indices, u_indices] = rev_angles

        else:
            x = coords[:, 0]
            y = coords[:, 1]

            # compute all pairwise differences
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]

            # Compute angles and mask invalid pairs
            angle_matrix = np.arctan2(dx, -dy) % (2 * np.pi)
            angle_matrix[np.isclose(dx**2 + dy**2, 0)] = np.nan  # Zero vectors

            if decimals is not None:
                angle_matrix = np.round(angle_matrix, decimals)

            # mask diagonal since we don't want
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
        angle_matrix, vertices = self.get_angle_matrix(edges_only, decimals)
        n = len(vertices)

        # Extract upper triangle indices
        rows, cols = np.triu_indices(n, k=1)
        angles = angle_matrix[rows, cols]
        valid_mask = ~np.isnan(angles)

        if not valid_mask.any():
            return defaultdict(list)

        # Filter valid pairs
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_angles = angles[valid_mask]

        # Group pairs by rounded angle
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
            self._coord_matrix *= radius / current_max

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
        """Add an edge between two nodes"""
        super().add_edge(node_id1, node_id2)

    # ===================
    # Visualization
    # ===================
    def validate_plot_parameters(func):
        """Decorator to validate plot method parameters"""

        def wrapper(self, *args, **kwargs):
            bounding_center_type = kwargs.get(
                "bounding_center_type", "bounding_box")

            if self.dim not in [2, 3]:
                raise ValueError(
                    "At least 2D or 3D coordinates required for plotting")

            if bounding_center_type not in CENTER_TYPES:
                raise ValueError(
                    f"Invalid center type: {bounding_center_type}. "
                    f"Valid options: {CENTER_TYPES}"
                )

            return func(self, *args, **kwargs)

        return wrapper

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
        **kwargs,
    ) -> plt.Axes:
        """
        Visualize the embedded graph in 2D or 3D
        """
        ax = self._create_axes(ax, self.dim)

        pos = {node: self._coord_matrix[i]
               for i, node in enumerate(self._node_list)}

        if self.dim == 2:
            self._draw_2d(ax, pos, with_labels,
                          node_size, edge_color, **kwargs)
        else:
            self._draw_3d(ax, pos, node_size, edge_color, elev, azim, **kwargs)

        if color_nodes_theta is not None:
            # calculate directional projection values
            direction = np.array(
                [np.sin(color_nodes_theta), -np.cos(color_nodes_theta)]
            )
            node_colors = np.dot(self._coord_matrix, direction)

            self._add_node_coloring(
                ax, pos, node_colors, node_size, self.dim, **kwargs)

        if bounding_circle:
            self._add_bounding_shape(ax, bounding_center_type, self.dim)

        self._configure_axes(ax)

        return ax

    def _create_axes(self, ax, dim=None):
        """Create appropriate axes if not provided

        Parameters:
            ax (matplotlib.axes.Axes, optional):
                The axes to use. If None, creates new axes
            dim (int, optional):
                Dimension of the plot. If None, uses self.dim

        Returns:
            matplotlib.axes.Axes:
                The configured axes object
        """
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
        """Add node coloring based on provided values

        Parameters:
            ax (matplotlib.axes.Axes):
                The axes to add coloring to
            pos (dict):
                Dictionary of node positions
            node_colors (array-like):
                Values to use for coloring nodes
            node_size (int):
                Size of nodes in visualization
            dim (int, optional):
                Dimension of the plot. If None, uses self.dim
            **kwargs:
                Additional keyword arguments for plotting
        """
        if dim is None:
            dim = self.dim

        if dim == 2:
            nodes = nx.draw_networkx_nodes(
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
            coords = np.array(list(pos.values()))
            nodes = ax.scatter3D(
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
        """Add bounding circle/sphere visualization

        Parameters:
            ax (matplotlib.axes.Axes):
                The axes to add the bounding shape to
            center_type (str, optional):
                Method to compute center ("mean", "bounding_box", or "origin")
            dim (int, optional):
                Dimension of the plot. If None, uses self.dim
        """
        if dim is None:
            dim = self.dim

        center = self.get_center(center_type)
        radius = self.get_bounding_radius(center_type)

        if dim == 2:
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
            ax.set_xlim(center[0] - radius - padding,
                        center[0] + radius + padding)
            ax.set_ylim(center[1] - radius - padding,
                        center[1] + radius + padding)
        else:
            # sphere wireframe
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

            ax.plot_wireframe(
                x, y, z, color="darkred", linewidth=0.5, alpha=0.3, rstride=2, cstride=2
            )
            padding = radius * 0.1
            ax.set_xlim3d(center[0] - radius - padding,
                          center[0] + radius + padding)
            ax.set_ylim3d(center[1] - radius - padding,
                          center[1] + radius + padding)
            ax.set_zlim3d(center[2] - radius - padding,
                          center[2] + radius + padding)

    def _configure_axes(self, ax):
        """Finalize plot appearance"""
        if hasattr(ax, "zaxis"):
            # 3D plot configuration
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            # 2D plot configuration
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
        """Calculate a nice interval for tick spacing

        Args:
            range_size: Size of the axis range

        Returns:
            float: Nice interval value for tick spacing
        """
        # Calculate rough interval size (aim for ~5 major ticks)
        rough_interval = range_size / 5

        # Get magnitude
        magnitude = 10 ** np.floor(np.log10(rough_interval))

        # Normalize rough interval to between 1 and 10
        normalized = rough_interval / magnitude

        # Choose nice interval
        if normalized < 1.5:
            nice_interval = 1
        elif normalized < 3:
            nice_interval = 2
        elif normalized < 7:
            nice_interval = 5
        else:
            nice_interval = 10

        return nice_interval * magnitude
