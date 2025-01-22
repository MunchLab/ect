import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # for PCA for normalization


class EmbeddedGraph(nx.Graph):
    """
    A class to represent a graph with 2D embedded coordinates for each vertex.

    Attributes
        graph : nx.Graph
            a NetworkX graph object
        coordinates : dict
            a dictionary mapping vertices to their (x, y) coordinates

    """

    def __init__(self):
        """
        Initializes an empty EmbeddedGraph object.

        """
        super().__init__()
        self.coordinates = {}

    def add_node(self, vertex, x, y):
        """Add a vertex to the graph. 
        If the vertex name is given as None, it will be assigned via the ``next_vert_name`` method.

        Parameters:
            vertex (hashable like int or str, or None) : The name of the vertex to add.
            x, y (floats) : The function value of the vertex being added.
            reset_pos (bool, optional) 
                If True, will reset the positions of the nodes based on the function values.
        """
        if vertex in self.nodes:
            raise ValueError(
                f'The vertex name {vertex} is already used in the graph.')

        if vertex is None:
            if len(self.nodes) == 0:
                vertex = 0
            else:
                vertex = self.next_vert_name(max(self.nodes))

        super().add_node(vertex)
        self.coordinates[vertex] = (x, y)

    def add_nodes_from(self, nodes, coordinates):
        """
        Adds multiple vertices to the graph and assigns them the given coordinates.

        Parameters:
            nodes (list):
                A list of vertices to be added.
            coordinates (dict):
                A dictionary mapping vertices to their coordinates.

        """
        super().add_nodes_from(nodes)
        self.coordinates.update(coordinates)

    def next_vert_name(self, s, num_verts=1):
        """ 
        Making a simple name generator for vertices. 
        If you're using integers, it will just up the count by one. 
        Letters will be incremented in the alphabet. If you reach ``Z``, it will return ``AA``. If you reach ``ZZ``, it will return ``AAA``, etc.

        Parameters:
            s (str or int): The name of the vertex to increment.

        Returns:
            str or int
                The next name in the sequence.
        """

        if type(s) == int:
            if num_verts > 1:
                return [s+1+i for i in range(num_verts)]
            else:
                return s+1
        elif type(s) == str and len(s) == 1:
            if not s == 'Z':
                if num_verts > 1:
                    return [chr(ord(s)+1+i) for i in range(num_verts)]
                else:
                    return chr(ord(s)+1)
            else:
                if num_verts > 1:
                    return [chr(ord('AA')+1+i) for i in range(num_verts)]
                else:
                    return 'AA'
        elif type(s) == str and len(s) > 1:
            if s[-1] == 'Z':
                if num_verts > 1:
                    return [s[:-1] + chr(ord('A')+1+i) for i in range(num_verts)]
                else:
                    return (len(s)+1) * 'A'
            else:
                if num_verts > 1:
                    return [s[:-1] + chr(ord(s[-1])+1+i) for i in range(num_verts)]
                else:
                    return len(s) * chr(ord(s[-1])+1+1)
        else:
            raise ValueError('Input must be a string or an integer')

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

    def add_cycle(self, coord_matrix):
        """
        Add nodes and edges from a cycle of coordinates. 
        Specifically, will add a node for each row and the edges connecting the nodes in the order they appear in the matrix as a closed cycle.

        Parameters:
            coord_matrix : numpy array
                An (n x 2) matrix of coordinates.
        """
        n = len(coord_matrix)
        if len(self.nodes) == 0:
            last_name = 0
        else:
            last_name = max(self.nodes)

        nodes = self.next_vert_name(last_name, num_verts=n)
        coords = {nodes[i]: coord_matrix[i] for i in range(n)}
        self.add_nodes_from(nodes, coords)
        edges = [(nodes[i], nodes[(i+1) % n]) for i in range(n)]
        self.add_edges_from(edges)

    def get_coordinates(self, vertex):
        """
        Returns the coordinates of the given vertex.

        Parameters:
            vertex (str):
                The vertex whose coordinates are to be returned.

        Returns:
            tuple: The coordinates of the vertex.

        """
        return self.coordinates.get(vertex)

    def set_coordinates(self, vertex, x, y):
        """
        Sets the coordinates of the given vertex.

        Parameters:
            vertex (str):
                The vertex whose coordinates are to be set.
            x (float):
                The new x-coordinate of the vertex.
            y (float): 
                The new y-coordinate of the vertex.

        Raises:
            ValueError: If the vertex does not exist in the graph.

        """
        if vertex in self.coordinates:
            self.coordinates[vertex] = (x, y)
        else:
            raise ValueError("Vertex does not exist in the graph.")

    def get_bounding_box(self):
        """
        Method to find a bounding box of the vertex coordinates in the graph.

        Returns:
            list: A list of tuples representing the minimum and maximum :math:`x` and :math:`y` coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        return [(min(x_coords), max(x_coords)), (min(y_coords), max(y_coords))]

    def get_center(self, type='origin'):
        """
        Calculate and return the center of the graph. This can be done by either returning the average of the coordiantes (``mean``), the center of the bounding box (``min_max``), or the origin (``origin``).

        Parameters:
            type (str): The type of center to calculate. Options are ``mean``, ``min_max``, or ``origin``.

        Returns:
            numpy.ndarray: The :math:`(x, y)` coordinates of the center.
        """
        if not self.coordinates:
            return np.array([0.0, 0.0])

        if type == 'origin':
            return np.array([0.0, 0.0])
        elif type == 'mean':
            coords = np.array(list(self.coordinates.values()))
            return np.mean(coords, axis=0)
        elif type == 'min_max':
            x_coords, y_coords = zip(*self.coordinates.values())
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            return np.array([(max_x+min_x)/2, (max_y+min_y)/2])

    def get_bounding_radius(self, type='origin'):
        """
        Method to find the radius of the bounding circle of the vertex coordinates in the graph. 

        Parameters:
            type (str): The type of center to calculate the radius relative to. Options are ``mean``, ``min_max``, or ``origin``.

        Returns:
            float: The radius of the bounding circle.

        """
        if not self.coordinates:
            return 0

        center = self.get_center(type)
        coords = np.array(list(self.coordinates.values()))
        distances = np.linalg.norm(coords - center, axis=1)
        return np.max(distances)

    # ------
    # Methods for normalizing the coordinates in various ways
    # ------

    def get_centered_coordinates(self, type='min_max'):
        """
        Method to find the centered coordinates of the vertices in the graph.

        If type is ``min_max``, the coordinates are centered at the mean of the min and max values of the :math:`x` and :math:`y` coordinates.
        If type is ``mean``, the coordinates are centered at the mean of the :math:`x` and :math:`y` coordinates.
        """

        if not self.coordinates:
            return None

        center = self.get_center(type)
        return {v: (x - center[0], y - center[1]) for v, (x, y) in self.coordinates.items()}

    def set_centered_coordinates(self, type='min_max'):
        """
        Method to set the centered coordinates of the vertices in the graph. Warning: This overwrites the original coordinates.
        """

        self.coordinates = self.get_centered_coordinates(type=type)

    def get_scaled_coordinates(self, radius=1):
        """
        Method to find the scaled coordinates of the vertices in the graph to fit in the disk centered at 0 with radius given by ``radius``.

        Parameters:
            radius (float):
                The radius of the bounding disk.

        Returns:
            dict: A dictionary mapping vertices to their scaled coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        max_norm = max(np.linalg.norm(point)
                       for point in zip(x_coords, y_coords))
        x_coords = x_coords * radius / max_norm
        y_coords = y_coords * radius / max_norm

        return {v: (x, y) for v, x, y in zip(self.coordinates.keys(), x_coords, y_coords)}

    def set_scaled_coordinates(self, radius=1):
        """
        Method to set the scaled coordinates of the vertices in the graph to fit in the disk centered at 0 with radius given by ``radius``. Warning: This overwrites the original coordinates

        """

        self.coordinates = self.get_scaled_coordinates(radius)

    def rescale_to_unit_disk(self, preserve_center=True, center_type='origin'):
        """
        Rescales the graph coordinates to fit within a radius 1 disk.

        Parameters:
            preserve_center (bool): If True, maintains the current center point of type ``center_type``.
                                    If False, centers the graph at (0, 0).

        Returns:
            self: Returns the instance for method chaining.

        Raises:
            ValueError: If the graph has no coordinates or all coordinates are identical.
        """
        if not self.coordinates:
            raise ValueError("Graph has no coordinates to rescale.")

        center = self.get_center(center_type)
        coords = np.array(list(self.coordinates.values()))

        coords_centered = coords - center

        max_distance = np.max(np.linalg.norm(coords_centered, axis=1))

        if np.isclose(max_distance, 0):
            raise ValueError("All coordinates are identical. Cannot rescale.")

        scale_factor = 1 / max_distance

        new_coords = (coords_centered * scale_factor) + \
            (center if preserve_center else 0)

        for vertex, new_coord in zip(self.coordinates.keys(), new_coords):
            self.coordinates[vertex] = tuple(new_coord)

        return self

    def get_PCA_coordinates(self):
        """
        Method to find the PCA coordinates of the vertices in the graph.

        Returns:
            dict: A dictionary mapping vertices to their PCA normalized coordinates.

        """

        if not self.coordinates:
            return None
        x_coords, y_coords = zip(*self.coordinates.values())
        M = np.array((x_coords, y_coords)).T

        pca = PCA(n_components=2)  # initiate PCA
        pca.fit_transform(M)  # fit PCA to coordinates to find longest axis
        pca_scores = pca.transform(M)  # retrieve PCA coordinates

        nodes = list(self.coordinates.keys())
        n = len(nodes)
        out = {nodes[i]: pca_scores[i] for i in range(n)}

        return out

    def set_PCA_coordinates(self, center_type=None, scale_radius=None):
        """
        Method to set the PCA coordinates of the vertices in the graph which is helpful for coarse alignment. 
        If you also want to center at zero, the options for ``center_type`` are ``mean`` or ``min_max``.
        Set ``scale_radius`` to a value to scale to a specific radius.
        Warning: This overwrites the original coordinates.
        """
        self.coordinates = self.get_PCA_coordinates()

        if center_type:
            self.set_centered_coordinates(center_type)

        if scale_radius:
            self.set_scaled_coordinates(radius=scale_radius)

    # ================
    # Functions for computing the function g(v) for vertices and edges
    # ================

    def g_omega(self, theta):
        """
        Function to compute the function :math:`g_\\omega(v)` for all vertices :math:`v` in the graph in the direction of :math:`\\theta \\in [0,2\\pi]` . This function is defined by :math:`g_\\omega(v) = \\langle \\texttt{pos}(v), \\omega \\rangle` .

        Parameters:

            theta (float):
                The angle in :math:`[0,2\\pi]` for the direction to compute the :math:`g(v)` values.

        Returns:

            dict: A dictionary mapping vertices to their :math:`g(v)` values.

        """

        omega = (np.cos(theta), np.sin(theta))

        g = {}
        for v in self.nodes:
            g[v] = np.dot(self.coordinates[v], omega)
        return g

    def g_omega_edges(self, theta):
        """
        Calculates the function value of the edges of the graph by making the value equal to the max vertex value 

        Parameters:

            theta (float): 
                The direction of the function to be calculated.

        Returns:
            dict
                A dictionary of the function values of the edges.
        """
        g = self.g_omega(theta)

        g_edges = {}
        for e in self.edges:
            g_edges[e] = max(g[e[0]], g[e[1]])

        return g_edges

    def sort_vertices(self, theta, return_g=False):
        """
        Function to sort the vertices of the graph according to the function `g_omega(v)` in the direction of :math:`\\theta \\in [0,2\\pi]`.

        TODO: eventually, do we want this to return a sorted list of g values as well? Since we're already doing the sorting work, it might be helpful.

        Parameters:
            theta (float):
                The angle in :math:`[0,2 \\pi]` for the direction to sort the vertices.
            return_g (bool):
                Whether to return the :math:`g(v)` values along with the sorted vertices.

        Returns:
            list
                A list of vertices sorted in increasing order of the :math:`g(v)` values. 
                If ``return_g`` is True, also returns the ``g`` dictionary with the function values ``g[vertex_name]=func_value``. 

        """
        g = self.g_omega(theta)

        v_list = sorted(self.nodes, key=lambda v: g[v])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return v_list, g
        else:
            return v_list

    def sort_edges(self, theta, return_g=False):
        """
        Function to sort the edges of the graph according to the function

        .. math ::

            g_\\omega(e) = \\max \\{ g_\\omega(v) \\mid  v \in e \\}

        in the direction of :math:`\\theta \\in [0,2\\pi]` .

        Parameters:
            theta (float):
                The angle in :math:`[0,2\\pi]` for the direction to sort the edges.
            return_g (bool):
                Whether to return the :math:`g(v)` values along with the sorted edges.

        Returns:
            A list of edges sorted in increasing order of the :math:`g(v)` values. 
            If ``return_g`` is True, also returns the ``g`` dictionary with the function values ``g[vertex_name]=func_value``. 

        """
        g_e = self.g_omega_edges(theta)

        e_list = sorted(self.edges, key=lambda e: g_e[e])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return e_list, g_e
        else:
            return e_list

    def lower_edges(self, v, omega):
        """
        Function to compute the number of lower edges of a vertex `v` for a specific direction (included by the use of sorted `v_list`).

        Parameters:
            v (str):
                The vertex to compute the number of lower edges for.
            omega (tuple): 
                The direction vector to consider given as an angle in [0, 2pi].

        Returns:
            int: The number of lower edges of the vertex v.

        """
        L = [n for n in self.neighbors(v)]
        gv = np.dot(self.coordinates[v], omega)
        Lg = [np.dot(self.coordinates[v], omega) for v in L]
        return sum(n >= gv for n in Lg)  # includes possible duplicate counts

    # ================
    # Functions for getting the angles where vertices switch order
    # ================

    def get_all_normals_matrix(self, num_rounding_digits=None):
        """
        Function to get all angles of normals to any line between vertices in the graph, returned as a matrix. Note this includes both adjacent vertices and non-adjacent. This function is useful for knowing the angle of the circle where two vertices switch in order. 

        Parameters:
            num_rounding_digits (int):
                The number of digits to round the angles in the matrix. If `None`, no rounding is done. 

        Returns:
            A tuple consissting of a matrix of angles, and the sorted label list for the rows/columns.
        """
        P = np.array(list(self.coordinates.values()))
        labels = list(self.coordinates.keys())

        # Make rows of repeated copies of first column of P
        X_Cols = np.tile(P[:, 0], (P.shape[0], 1)).T
        X_diff = X_Cols - X_Cols.T  # The x value of the vector from A to B

        # Make rows of repeated copies of second column of P
        Y_Cols = np.tile(P[:, 1], (P.shape[0], 1)).T
        Y_diff = Y_Cols - Y_Cols.T

        # Convert to float to allow NaN assignment
        X_diff = X_diff.astype(float)
        Y_diff = Y_diff.astype(float)

        # Set diagonals to nan
        np.fill_diagonal(X_diff, np.nan)
        np.fill_diagonal(Y_diff, np.nan)

        angle_matrix = np.arctan2(X_diff, -Y_diff)
        # Puts all entries between 0 and 2pi
        angle_matrix = angle_matrix % (2*np.pi)
        if num_rounding_digits != None:
            angle_matrix = np.round(angle_matrix, num_rounding_digits)

        return angle_matrix, labels

    def get_normals_dict(self,
                         num_rounding_digits=None,
                         edges_only=False,
                         opposites=False):
        """
        Function to get all angles of normals to any line between vertices in the graph, returned as a dictionary of angles with ``dict[theta]`` returning the list of pairs of vertices with vector normal to :math:`\\overrightarrow{AB}` at angle ``theta``. Note this includes both adjacent vertices and non-adjacent unless ``edges_only`` is set to ``True``. This function is useful for knowing the angle of the circle where two vertices switch in order, especially when we want to sort the events around the circle. All angles are rounded to the number of digits given by ``num_rounding_digits``, and are returned in the range :math:`[0, 2\\pi]` .

        Parameters:
            num_rounding_digits (int):
                The number of digits to round the angles in the dictionary. If None, no rounding is done.
            edges_only (bool):
                If True, the dictionary version only returns the angle of the normals to the lines between vertices sharing an edge. The matrix version is unchanged.
            opposites (bool):
                If True, will also include both :math:`\\theta` and :math:`\\theta + \\pi` in the dictionary keys.

        Returns:
            dict: A dictionary of angles with `dict[theta]` returning the list of pairs of vertices with vector normal to :math:`\\overrightarrow\\{AB\\}` at angle `theta`, e.g. ``dict[theta] = [('A','B'), ('C', 'D')]``.
        """

        angle_matrix, labels = self.get_all_normals_matrix(
            num_rounding_digits=num_rounding_digits)

        angle_dict = {}
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                # Skip this edge if edges_only is True and the vertices are not connected
                if edges_only and not self.has_edge(labels[i], labels[j]):
                    continue

                if angle_matrix[i, j] in angle_dict.keys():
                    angle_dict[angle_matrix[i, j]].append(
                        (labels[i], labels[j]))
                elif angle_matrix[j, i] in angle_dict.keys():
                    angle_dict[angle_matrix[j, i]].append(
                        (labels[j], labels[i]))
                else:
                    angle_dict[angle_matrix[i, j]] = [
                        (labels[i], labels[j])]

        if opposites:
            for key in list(angle_dict.keys()):
                other_key = key + np.pi % (2*np.pi)
                if num_rounding_digits != None:
                    other_key = round(other_key, num_rounding_digits)
                angle_dict[other_key] = angle_dict[key]

        # Make sure all keys are in the range [0, 2pi]
        angle_dict = {key % (2*np.pi): value for key,
                      value in angle_dict.items()}

        # Make sure all keys are rounded to the correct number of digits
        if num_rounding_digits != None:
            angle_dict = {round(key, num_rounding_digits)                          : value for key, value in angle_dict.items()}

        return angle_dict

    # ================
    # Plotting functions
    # ================
    def plot(self,
             bounding_circle=False,
             color_nodes_theta=None,
             ax=None,
             with_labels=True,
             **kwargs):
        """
        Function to plot the graph with the embedded coordinates.

        If ``bounding_circle`` is True, a bounding circle is drawn around the graph.

        If ``color_nodes_theta`` is not ``None``, it should be given as a :math:`theta` in :math:`[0,2\\pi]`. Then the nodes are colored according to the :math:`g(v)` values in the direction of :math:`\\theta`.

        If ``with_labels`` is ``True``, the nodes are labeled with their names.

        If ``ax`` is not ``None``, the plot is drawn on the given axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        pos = self.coordinates
        # center = self.get_center(type = 'min_max')
        # r = self.get_bounding_radius(type = 'min_max')

        if color_nodes_theta is None:
            nx.draw(self, pos, with_labels=with_labels, ax=ax, **kwargs)
        else:
            g = self.g_omega(color_nodes_theta)
            color_map = [g[v] for v in self.nodes]
            # Some weird plotting to make the colorbar work.
            pathcollection = nx.draw_networkx_nodes(
                self, pos, node_color=color_map, ax=ax)
            nx.draw_networkx_labels(self, pos=pos, font_color='black', ax=ax)
            nx.draw_networkx_edges(self, pos, ax=ax, width=1, **kwargs)
            fig.colorbar(pathcollection, ax=ax, **kwargs)

        plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)

        if bounding_circle:
            self.plot_bounding_circle(ax=ax)

        ax.set_aspect('equal', 'box')

        return ax

    def plot_bounding_circle(self, ax=None, bounding_center_type='origin', **kwargs):
        """
        Function to plot the bounding circle of the graph. 

        If ``ax`` is not None, the plot is drawn on the given axis.

        If ``bounding_center_type`` is 'origin', the bounding circle is centered at the origin. If it is ``min_max``, the bounding circle is centered at the mean of the min and max values of the :math:`x` and :math`y` coordinates.


        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        circle_center = self.get_center(bounding_center_type)
        r = self.get_bounding_radius(type=bounding_center_type)
        circle = plt.Circle(circle_center, r, fill=False,
                            linestyle='--', color='r')
        ax.add_patch(circle)

        # Always adjust the plot limits to show the full graph
        ax.set_xlim(circle_center[0] - r, circle_center[0] + r)
        ax.set_ylim(circle_center[1] - r, circle_center[1] + r)

    def plot_angle_circle(self, ax=None, edges_only=False):
        """
        Function to plot the circle of angles for the graph. 

        Example Usage: 

        .. code-block:: python

            fig, ax = plt.subplots()
            G.plot(ax = ax)
            G.plot_angle_circle(ax = ax)

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        circle_center = self.get_center(type='min_max')
        r = 1.3 * self.get_bounding_radius(type='min_max')
        circle = plt.Circle(circle_center, r, fill=False,
                            linestyle='--', color='black')
        ax.add_patch(circle)

        # Get all the angles with the labels to be drawn
        angles_dict = self.get_normals_dict(
            edges_only=edges_only, opposites=True)
        angles_dict_labels = {key: ', '.join(
            [f"{a[0]}{a[1]}" for a in value]) for key, value in angles_dict.items()}

        # Draw hash marks on the circle
        hash_length = 0.1*r  # Length of the hash marks
        for angle in angles_dict_labels.keys():
            x_start = circle_center[0] + \
                (r-hash_length) * np.cos(angle)
            y_start = circle_center[1] + \
                (r-hash_length) * np.sin(angle)
            x_end = circle_center[0] + \
                (r + hash_length) * np.cos(angle)
            y_end = circle_center[1] + \
                (r + hash_length) * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end], color='black')

            # Add labels near the hash marks
            scaling = 3
            label_x = circle_center[0] + \
                (r + scaling * hash_length) * np.cos(angle)
            label_y = circle_center[1] + \
                (r + scaling * hash_length) * np.sin(angle)
            text_angle = angle if angle <= np.pi/2 or angle >= 3*np.pi/2 else angle - np.pi
            ax.text(label_x, label_y,
                    angles_dict_labels[angle], ha='center', va='center', rotation=np.degrees(text_angle), fontsize=8)
        # Always adjust the plot limits to show the full graph
        scale_factor = 1.5
        ax.set_xlim(circle_center[0] - scale_factor*r,
                    circle_center[0] + scale_factor*r)
        ax.set_ylim(circle_center[1] - scale_factor*r,
                    circle_center[1] + scale_factor*r)


def create_example_graph(centered=True, center_type='min_max'):
    """
    Function to create an example ``EmbeddedGraph`` object. Helpful for testing. If ``centered`` is True, the coordinates are centered using the center type given by ``center_type``, either ``mean`` or ``min_max``.

    Returns:
        EmbeddedGraph: An example ``EmbeddedGraph`` object.

    """
    graph = EmbeddedGraph()

    graph.add_node('A', 1, 2)
    graph.add_node('B', 3, 4)
    graph.add_node('C', 5, 7)
    graph.add_node('D', 3, 6)
    graph.add_node('E', 4, 3)
    graph.add_node('F', 4, 5)

    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')
    graph.add_edge('B', 'D')
    graph.add_edge('B', 'E')
    graph.add_edge('C', 'D')
    graph.add_edge('E', 'F')

    if centered:
        graph.set_centered_coordinates(center_type)

    return graph


if __name__ == "__main__":
    # Example usage of the EmbeddedGraph class

    # Create an instance of the EmbeddedGraph class
    graph = EmbeddedGraph()

    # Add vertices with their coordinates
    graph.add_node('A', 1, 2)
    graph.add_node('B', 3, 4)
    graph.add_node('C', 5, 6)

    # Add edges between vertices
    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')

    # Get coordinates of a vertex
    coords = graph.get_coordinates('A')
    print(f'Coordinates of A: {coords}')

    # Set new coordinates for a vertex
    graph.set_coordinates('A', 7, 8)
    coords = graph.get_coordinates('A')
    print(f'New coordinates of A: {coords}')

    # Get the bounding box of the vertex coordinates
    bbox = graph.get_bounding_box()
    print(f'Bounding box: {bbox}')
