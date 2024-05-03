import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class EmbeddedGraph(nx.Graph):
    """
    A class to represent a graph with 2D embedded coordinates for each vertex.

    Attributes
        coordinates : dict
            a dictionary mapping vertices to their (x, y) coordinates

    Methods
        add_vertex(vertex, x, y):
            Adds a vertex to the graph and assigns it the given coordinates.
        add_edge(u, v):
            Adds an edge between the vertices u and v.
        get_coordinates(vertex):
            Returns the coordinates of the given vertex.
        set_coordinates(vertex, x, y):
            Sets the coordinates of the given vertex.

    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the EmbeddedGraph object.

        """
        super().__init__()
        self.coordinates = {}

    def add_vertex(self, vertex, x, y):
        """
        Adds a vertex to the graph and assigns it the given coordinates.

        Parameters:
            vertex: str
                The vertex to be added.
            x : float
                The x-coordinate of the vertex.
            y : float
                The y-coordinate of the vertex.

        """
        self.add_node(vertex)
        self.coordinates[vertex] = (x, y)

    def add_edge(self, u, v):
        """
        Adds an edge between the vertices u and v if they exist.

        Parameters:
            u : str
                The first vertex of the edge.
            v : str
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
            vertex : str
                The vertex whose coordinates are to be returned.

        Returns:
            tuple: The coordinates of the vertex.

        """
        return self.coordinates.get(vertex)

    def set_coordinates(self, vertex, x, y):
        """
        Sets the coordinates of the given vertex.

        Parameters:
            vertex : str
                The vertex whose coordinates are to be set.
            x : float
                The new x-coordinate of the vertex.
            y : float 
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
    
    def get_bounding_radius(self):
        """
        Method to find the radius of the bounding circle of the vertex coordinates in the graph.

        Returns:
            float: The radius of the bounding circle.

        """
        if not self.coordinates:
            return 0

        x_coords, y_coords = zip(*self.coordinates.values())
        norms = [np.linalg.norm(point) for point in zip(x_coords, y_coords)]

        return max(norms)
    
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
        Function to compute the function g_omega(v) for all vertices v in the graph in the direction of theta \in [0,2*np.pi]. This function is defined by $g_\omega(v) = < pos(v), \omega >$.

        Parameters:

            theta : float
                The angle in [0,2*np.pi] for the direction to compute the g(v) values.

        Returns:

            dict: A dictionary mapping vertices to their g(v) values.

        """
        
        omega = (np.cos(theta), np.sin(theta))

        g = {}
        for v in self.nodes:
            g[v] = np.dot(self.coordinates[v], omega)
        return g
    
    def sort_vertices(self, theta,return_g = False):
        """
        Function to sort the vertices of the graph according to the function g_omega(v) in the direction of theta \in [0,2*np.pi].

        TODO: eventually, do we want this to return a sorted list of g values as well? Since we're already doing the sorting work, it might be helpful.

        Parameters:
            theta : float
                The angle in [0,2*np.pi] for the direction to sort the vertices.
            return_g : bool
                Whether to return the g(v) values along with the sorted vertices.

        Returns:
            list: 
                A list of vertices sorted in increasing order of the g(v) values. 
            dict: 
                If return_g is True, also returns the g dictionary with the function values. 

        """
        g = self.g_omega(theta)

        v_list = sorted(self.nodes, key=lambda v: g[v])

        if return_g:
            # g_sorted = [g[v] for v in v_list]
            return  v_list, g
        else:
            return v_list
        
    
    def lower_edges(self, v, omega):
        """
        Function to compute the number of lower edges of a vertex v for a specific direction (included by the use of sorted v_list).

        Parameters:
            v : str
                The vertex to compute the number of lower edges for.
            omega : tuple 
                The direction vector to consider given as an angle in [0, 2pi].

        Returns:
            int: The number of lower edges of the vertex v.

        """
        L = [n for n in self.neighbors(v)]
        gv = np.dot(self.coordinates[v],omega)
        Lg = [np.dot(self.coordinates[v],omega) for v in L]
        return sum(n >= gv for n in Lg) # includes possible duplicate counts 

    def plot(self):
        """
        Function to plot the graph with the embedded coordinates.

        """

        fig, ax = plt.subplots()

        pos = self.coordinates
        nx.draw(self, pos, with_labels=True, font_weight='bold')
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()


def create_example_graph(mean_centered = True):
    """
    Function to create an example EmbeddedGraph object. Helpful for testing.

    Returns:
        EmbeddedGraph: An example EmbeddedGraph object.

    """
    graph = EmbeddedGraph()

    graph.add_vertex('A', 1, 2)
    graph.add_vertex('B', 3, 4)
    graph.add_vertex('C', 5, 7)
    graph.add_vertex('D', 3, 6)
    graph.add_vertex('E', 4, 3)
    graph.add_vertex('F', 4, 5)

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
    graph.add_vertex('A', 1, 2)
    graph.add_vertex('B', 3, 4)
    graph.add_vertex('C', 5, 6)

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