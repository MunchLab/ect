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
        G = embed_graph.create_example_graph(centered=False)
        coords = G.get_coordinates('A')
        self.assertEqual( coords, (1, 2))

    def test_coords_list(self):
        # Make sure the keys in the coordinates list are the same as the nodes
        G = embed_graph.create_example_graph(centered=False)
        self.assertEqual( len(G.nodes), len(G.coordinates))
        self.assertEqual( set(G.nodes), set(G.coordinates.keys()))

    def test_mean_centered_coordinates(self):
        # Make sure the mean centered coordinates are correct
        G = embed_graph.create_example_graph(centered=False)
        G.set_centered_coordinates(type = 'mean')
        x_coords = [x for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.average(x_coords), 0, places = 1)

    def test_get_center(self):
        G = embed_graph.create_example_graph()
        center = G.get_center(type = 'mean')
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

    def test_min_max_centered_coordinates(self):
        # Make sure the min-max centered coordinates are correct
        G = embed_graph.create_example_graph(centered=False)
        G.set_centered_coordinates(type = 'min_max')
        x_coords = [x for x, y in G.coordinates.values()]
        y_coords = [y for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.max(x_coords) + np.min(x_coords), 0, places = 1)
        self.assertAlmostEqual( np.max(y_coords) + np.min(y_coords), 0, places = 1)

    def test_PCA_coords(self):
        # Make sure the PCA coordinates are running
        # Note this doesn't check correctness
        G = embed_graph.create_example_graph(centered=False)
        G.set_PCA_coordinates()
        self.assertEqual( len(G.coordinates), 6)

    def test_add_cycle(self):
        # Make sure we can add a loop of input 
        G = embed_graph.create_example_graph(centered=False)
        num_verts = len(G.nodes)
        num_edges = len(G.edges)
        verts_to_add = 8
        loop_coords = 3*np.random.rand(verts_to_add, 2)

        G.add_cycle(loop_coords)
        G.plot()
        self.assertEqual( len(G.nodes), num_verts + verts_to_add)
        self.assertEqual( len(G.edges), num_edges + verts_to_add)

    def test_get_angles(self):
        # Make sure we can get the angles of the vertices
        G = embed_graph.create_example_graph(centered=False)
        M,L = G.get_all_normals_matrix()
        self.assertEqual( M.shape, (6, 6))

        # Check that all entries in the matrix are between 0 and 2pi other 
        # than the diagonal which should be nan

        self.assertTrue( np.nanmin(M) >= 0)
        self.assertTrue( np.nanmax(M) <= 2*np.pi)

        # Check that all keys in dictionary have the same property 
        M_dict = G.get_normals_dict(opposites = True)
        keys = list(M_dict.keys())
        self.assertTrue( all([0 <= x <= 2*np.pi for x in keys]))
        

        #----
        # Check that the values the dictionary show up in pairs 
        # when asking for opposites
        M_dict = G.get_normals_dict(opposites=True)
        vals = list(M_dict.values())

        # Convert to strings to make this hashable for the counter 
        vals = [','.join([''.join(x) for x in A]) for A in vals]
        vals

        # Count all the instaces of each value. 
        # These should come in pairs, so should always be 2. 
        from collections import Counter
        counter = Counter(vals)
        for k,v in counter.items():
            self.assertEqual(v, 2) 

         



if __name__ == '__main__':
    unittest.main()