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

    def test_get_center(self):
        G = embed_graph.create_example_graph()
        center = G.get_center()
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center), 2)
        
        # Check if center is correctly calculated
        coords = np.array(list(G.coordinates.values()))
        expected_center = np.mean(coords, axis=0)
        np.testing.assert_almost_equal(center, expected_center)

    def test_rescale_to_unit_disk(self):
        G = embed_graph.create_example_graph()
        original_center = G.get_center()
        G.rescale_to_unit_disk(preserve_center=True)
        
        self.assertAlmostEqual(G.get_bounding_radius(), 1.0, places=6)
        np.testing.assert_almost_equal(G.get_center(), original_center)

        G = embed_graph.create_example_graph()
        G.rescale_to_unit_disk(preserve_center=False)
        self.assertAlmostEqual(G.get_bounding_radius(), 1.0, places=6)
        np.testing.assert_almost_equal(G.get_center(), np.array([0, 0]), decimal=6)


if __name__ == '__main__':
    unittest.main()