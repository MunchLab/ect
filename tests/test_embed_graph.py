import unittest
from ect.utils.examples import create_example_graph
import numpy as np


class TestEmbeddedGraph(unittest.TestCase):
    def test_example_graph(self):
        G = create_example_graph()
        self.assertEqual(len(G.nodes), 6)
        self.assertEqual(G.dim, 2)

    def test_coord_matrix(self):
        G = create_example_graph()
        self.assertEqual(G.coord_matrix.shape, (6, 2))
        self.assertTrue(isinstance(G.coord_matrix, np.ndarray))

    def test_node_list(self):
        G = create_example_graph()
        self.assertEqual(len(G.node_list), 6)
        self.assertEqual(set(G.node_list), set(G.nodes))

    def test_add_node(self):
        G = create_example_graph()
        G.add_node("G", [1, 2])
        self.assertEqual(len(G.nodes), 7)
        self.assertEqual(G.coord_matrix.shape, (7, 2))
        self.assertEqual(G.get_coord("G").tolist(), [1, 2])

    def test_add_edge(self):
        G = create_example_graph()
        G.add_edge("A", "B")
        self.assertEqual(len(G.edges), 6)

    def test_get_coord(self):
        G = create_example_graph(centered=False)
        coords = G.get_coord("A")
        self.assertTrue(np.array_equal(coords, np.array([1, 2])))

    def test_invalid_node_operations(self):
        G = create_example_graph()
        with self.assertRaises(ValueError):
            G.add_node("A", [1, 2])
        with self.assertRaises(ValueError):
            G.get_coord("Z")

    def test_mean_centered_coordinates(self):
        G = create_example_graph(centered=False)
        G.center_coordinates("mean")
        x_coords = G.coord_matrix[:, 0]
        self.assertAlmostEqual(np.average(x_coords), 0, places=1)

    def test_get_center(self):
        G = create_example_graph()
        center = G.get_center("mean")
        self.assertIsInstance(center, np.ndarray)
        self.assertEqual(len(center), 2)

        coords = G.coord_matrix
        expected_center = np.mean(coords, axis=0)
        np.testing.assert_almost_equal(center, expected_center)

    def test_rescale_to_unit_disk(self):
        G = create_example_graph()
        G.scale_coordinates(1.0)

        self.assertAlmostEqual(G.get_bounding_radius(center_type="mean"), 1.0, places=6)

        coords_before = G.coord_matrix.copy()
        G.scale_coordinates(2.0)
        coords_after = G.coord_matrix
        self.assertTrue(np.allclose(coords_after / 2.0, coords_before))

    def test_bounding_box_centered_coordinates(self):
        G = create_example_graph(centered=False)
        G.center_coordinates("bounding_box")
        x_coords = G.coord_matrix[:, 0]
        y_coords = G.coord_matrix[:, 1]

        self.assertAlmostEqual(np.max(x_coords) + np.min(x_coords), 0, places=1)
        self.assertAlmostEqual(np.max(y_coords) + np.min(y_coords), 0, places=1)

    def test_PCA_coords(self):
        G = create_example_graph(centered=False)
        G.project_coordinates("pca")
        self.assertEqual(G.coord_matrix.shape, (6, 2))

    def test_add_cycle(self):
        G = create_example_graph(centered=False)
        num_verts = len(G.nodes)
        num_edges = len(G.edges)
        verts_to_add = 8
        loop_coords = 3 * np.random.rand(verts_to_add, 2)

        G.add_cycle(loop_coords)
        G.plot()
        self.assertEqual(len(G.nodes), num_verts + verts_to_add)
        self.assertEqual(len(G.edges), num_edges + verts_to_add)


if __name__ == "__main__":
    unittest.main()
