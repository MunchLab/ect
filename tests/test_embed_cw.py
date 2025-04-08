import unittest
from ect.utils.examples import create_example_cw
import numpy as np


class TestEmbeddedCW(unittest.TestCase):
    def test_example_cw(self):
        G = create_example_cw()
        self.assertEqual(len(G.nodes), 11)
        self.assertEqual(len(G.faces), 2)

    def test_get_coordinates(self):
        G = create_example_cw(centered=False)
        coords = G.get_coord("A")
        self.assertTrue(np.array_equal(coords, np.array([1, 2])))

    def test_mean_centered_coordinates(self):
        G = create_example_cw(centered=False)
        G.center_coordinates("mean")
        x_coords = G.coord_matrix[:, 0]
        self.assertAlmostEqual(np.average(x_coords), 0, places=1)

    def test_add_face(self):
        G = create_example_cw()
        face = ["A", "B", "C"]
        G.add_edges_from([(face[i], face[(i + 1) % 3]) for i in range(3)])
        G.add_face(face)
        self.assertIn(tuple(face), G.faces)

    def test_non_existent_vertex(self):
        G = create_example_cw()
        face = ["A", "B", "C"]
        G.add_edges_from([(face[i], face[(i + 1) % 3]) for i in range(3)])

        # test non-existent vertex
        with self.assertRaises(ValueError):
            G.add_face(["A", "B", "Z"])

    def test_face_with_missing_edges(self):
        G = create_example_cw()
        face = ["A", "B", "C"]
        G.add_edges_from([(face[i], face[(i + 1) % 3]) for i in range(3)])
        with self.assertRaises(ValueError):
            G.add_face(["A", "D", "E"], check=True)

    def test_too_short_face(self):
        G = create_example_cw()
        face = ["A", "B", "C"]
        G.add_edges_from([(face[i], face[(i + 1) % 3]) for i in range(3)])
        with self.assertRaises(ValueError):
            G.add_face(["A", "B"])


if __name__ == "__main__":
    unittest.main()
