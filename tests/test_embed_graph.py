import unittest
from ect import embed_graph
import numpy as np


class TestEmbeddedGraph(unittest.TestCase):
    def test_example_graph(self):
        # Make sure we can build a grpah in the first place
        G = embed_graph.create_example_graph()
        self.assertEqual( len(G.nodes), 6)  # assuming my_function squares its input


    def test_add_node(self):
        # Make sure adding a vertex updates the coordiantes list 
        G = embed_graph.create_example_graph()
        G.add_node('G', 1, 2)
        self.assertEqual( len(G.nodes), 7)
        self.assertEqual( len(G.coordinates), 7)

    def test_add_edge(self):
        # Make sure adding an edge updates the edge list
        G = embed_graph.create_example_graph()
        G.add_edge('A', 'B')
        self.assertEqual( len(G.edges), 6)

    def test_get_coordinates(self):
        # Make sure we can get the coordinates of a vertex
        G = embed_graph.create_example_graph(mean_centered=False)
        coords = G.get_coordinates('A')
        self.assertEqual( coords, (1, 2))

    def test_coords_list(self):
        # Make sure the keys in the coordinates list are the same as the nodes
        G = embed_graph.create_example_graph(mean_centered=False)
        self.assertEqual( len(G.nodes), len(G.coordinates))
        self.assertEqual( set(G.nodes), set(G.coordinates.keys()))

    def test_mean_centered_coordinates(self):
        # Make sure the mean centered coordinates are correct
        G = embed_graph.create_example_graph(mean_centered=False)
        G.set_mean_centered_coordinates()
        x_coords = [x for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.average(x_coords), 0, places = 1)

    def test_min_max_centered_coordinates(self):
        # Make sure the min-max centered coordinates are correct
        G = embed_graph.create_example_graph(mean_centered=False)
        G.set_min_max_centered_coordinates()
        x_coords = [x for x, y in G.coordinates.values()]
        y_coords = [y for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.max(x_coords) + np.min(x_coords), 0, places = 1)
        self.assertAlmostEqual( np.max(y_coords) + np.min(y_coords), 0, places = 1)

    def test_PCA_coords(self):
        # Make sure the PCA coordinates are running
        # Note this doesn't check correctness
        G = embed_graph.create_example_graph(mean_centered=False)
        G.set_PCA_coordinates()
        self.assertEqual( len(G.coordinates), 6)

    def test_add_cycle(self):
        # Make sure we can add a loop of input 
        G = embed_graph.create_example_graph(mean_centered=False)
        num_verts = len(G.nodes)
        num_edges = len(G.edges)
        verts_to_add = 8
        loop_coords = 3*np.random.rand(verts_to_add, 2)

        G.add_cycle(loop_coords)
        G.plot()
        self.assertEqual( len(G.nodes), num_verts + verts_to_add)
        self.assertEqual( len(G.edges), num_edges + verts_to_add)

    

if __name__ == '__main__':
    unittest.main()