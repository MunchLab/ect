import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
        """
        Adds a vertex to the graph and assigns it the given coordinates.

        Parameters:
            vertex (str):
                The vertex to be added.
            x (float):
                The x-coordinate of the vertex.
            y (float):
                The y-coordinate of the vertex.

        """
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
            list: A list of tuples representing the minimum and maximum x and y coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        return [(min(x_coords), max(x_coords)), (min(y_coords), max(y_coords))]

    def get_center(self):
        """
        Calculate and return the center of the graph.

        Returns:
            numpy.ndarray: The (x, y) coordinates of the center.
        """
        if not self.coordinates:
            return np.array([0.0, 0.0])
        coords = np.array(list(self.coordinates.values()))
        return np.mean(coords, axis=0)

    def get_bounding_radius(self):
        """
        Method to find the radius of the bounding circle of the vertex coordinates in the graph.

        Returns:
            float: The radius of the bounding circle.

        """
        if not self.coordinates:
            return 0

        center = self.get_center()
        coords = np.array(list(self.coordinates.values()))
        distances = np.linalg.norm(coords - center, axis=1)
        return np.max(distances)

    def get_mean_centered_coordinates(self):
        """
        Method to find the mean-centered coordinates of the vertices in the graph.

        Returns:
            dict: A dictionary mapping vertices to their mean-centered coordinates.

        """
        if not self.coordinates:
            return None

        x_coords, y_coords = zip(*self.coordinates.values())
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)

        return {v: (x - mean_x, y - mean_y) for v, (x, y) in self.coordinates.items()}

    def set_mean_centered_coordinates(self):
        """
        Method to set the mean-centered coordinates of the vertices in the graph. Warning: This overwrites the original coordinates

        """

        self.coordinates = self.get_mean_centered_coordinates()

    def g_omega(self, theta):
        """
        Function to compute the function :math:`g_\omega(v)` for all vertices :math:`v` in the graph in the direction of :math:`\\theta \in [0,2\pi]` . This function is defined by :math:`g_\omega(v) = \langle \\texttt{pos}(v), \omega \\rangle` .

        Parameters:

            theta (float):
                The angle in :math:`[0,2\pi]` for the direction to compute the :math:`g(v)` values.

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
        Function to sort the vertices of the graph according to the function g_omega(v) in the direction of theta \in [0,2*np.pi].

        TODO: eventually, do we want this to return a sorted list of g values as well? Since we're already doing the sorting work, it might be helpful.

        Parameters:
            theta (float):
                The angle in [0,2*np.pi] for the direction to sort the vertices.
            return_g (bool):
                Whether to return the g(v) values along with the sorted vertices.

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

            g_\omega(e) = \max \{ g_\omega(v) \mid  v \in e \}

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
        g_e = self.g_omega_edges(theta)

        e_list = sorted(self.edges, key=lambda e: g_e[e])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return e_list, g_e
        else:
            return e_list

    def lower_edges(self, v, omega):
        """
        Function to compute the number of lower edges of a vertex v for a specific direction (included by the use of sorted v_list).

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

    def plot(self, bounding_circle=False, color_nodes_theta=None, ax=None, **kwargs):
        """
        Function to plot the graph with the embedded coordinates.

        If ``bounding_circle`` is True, a bounding circle is drawn around the graph.

        If ``color_nodes_theta`` is not None, it should be given as a theta in :math:`[0,2\pi]`. Then the nodes are colored according to the :math:`g(v)` values in the direction of theta.

        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        pos = self.coordinates
        center = self.get_center()
        r = self.get_bounding_radius()

        if color_nodes_theta is None:
            nx.draw(self, pos, with_labels=True, ax=ax, **kwargs)
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
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        if bounding_circle:
            r = self.get_bounding_radius()
            circle = plt.Circle(center, r, fill=False, linestyle='--', color='r')
            ax.add_patch(circle)

        # Always adjust the plot limits to show the full graph
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_aspect('equal', 'box')

        return ax

    def rescale_to_unit_disk(self, preserve_center=True):
        """
        Rescales the graph coordinates to fit within a radius 1 disk.

        Parameters:
            preserve_center (bool): If True, maintains the current center point.
                                    If False, centers the graph at (0, 0).

        Returns:
            self: Returns the instance for method chaining.

        Raises:
            ValueError: If the graph has no coordinates or all coordinates are identical.
        """
        if not self.coordinates:
            raise ValueError("Graph has no coordinates to rescale.")

        center = self.get_center()
        coords = np.array(list(self.coordinates.values()))
        
        coords_centered = coords - center
        
        max_distance = np.max(np.linalg.norm(coords_centered, axis=1))
        
        if np.isclose(max_distance, 0):
            raise ValueError("All coordinates are identical. Cannot rescale.")

        scale_factor = 1 / max_distance

        new_coords = (coords_centered * scale_factor) + (center if preserve_center else 0)

        for vertex, new_coord in zip(self.coordinates.keys(), new_coords):
            self.coordinates[vertex] = tuple(new_coord)

        return self

def create_example_graph(mean_centered=True):
    """
    Function to create an example ``EmbeddedGraph`` object. Helpful for testing.

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

    if mean_centered:
        graph.set_mean_centered_coordinates()

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