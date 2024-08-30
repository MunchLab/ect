import unittest
from ect import EmbeddedCW, create_example_cw
import numpy as np


class TestEmbeddedCW(unittest.TestCase):
    def test_example_cw(self):
        # Make sure we can build a grpah in the first place
        K = create_example_cw()
        self.assertEqual( len(K.nodes), 11)  # assuming my_function squares its input

    def test_add_face(self):
        # Make sure adding a vertex updates the coordiantes list 
        G = create_example_cw()
        G.add_face(['D','B','C'])
        self.assertEqual( len(G.faces), 3)

    # TODO: Add test to check that adding an invalid face throws an error

    def test_get_coordinates(self):
        # Make sure we can get the coordinates of a vertex
        G = create_example_cw(centered=False)
        coords = G.get_coordinates('A')
        self.assertEqual( coords, (1, 2))

    def test_mean_centered_coordinates(self):
        # Make sure the mean centered coordinates are correct
        G = create_example_cw(centered=False)
        G.set_centered_coordinates(type = 'mean')
        x_coords = [x for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.average(x_coords), 0, places = 1)

    

if __name__ == '__main__':
    unittest.main()