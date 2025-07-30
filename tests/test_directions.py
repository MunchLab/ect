import unittest
import numpy as np
from ect import Directions
from ect.directions import Sampling


class TestDirections(unittest.TestCase):
    def test_uniform_2d(self):
        num_dirs = 8
        dirs = Directions.uniform(num_dirs, dim=2)
        
        self.assertEqual(len(dirs), num_dirs)
        self.assertEqual(dirs.dim, 2)
        self.assertEqual(dirs.sampling, Sampling.UNIFORM)
        
        # test vector properties
        vectors = dirs.vectors
        self.assertEqual(vectors.shape, (num_dirs, 2))
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0)

    def test_uniform_3d(self):
        num_dirs = 10
        dirs = Directions.uniform(num_dirs, dim=3)
        
        self.assertEqual(len(dirs), num_dirs)
        self.assertEqual(dirs.dim, 3)
        
        vectors = dirs.vectors
        self.assertEqual(vectors.shape, (num_dirs, 3))
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0)

    def test_random_sampling(self):
        num_dirs = 10
        seed = 42
        dirs = Directions.random(num_dirs, seed=seed)
        
        self.assertEqual(len(dirs), num_dirs)
        self.assertEqual(dirs.sampling, Sampling.RANDOM)
        
        # test reproducibility
        dirs2 = Directions.random(num_dirs, seed=seed)
        np.testing.assert_array_equal(dirs.vectors, dirs2.vectors)

    def test_custom_angles(self):
        angles = [0, np.pi/4, np.pi/2]
        dirs = Directions.from_angles(angles)
        
        self.assertEqual(len(dirs), 3)
        self.assertEqual(dirs.dim, 2)
        self.assertEqual(dirs.sampling, Sampling.CUSTOM)
        np.testing.assert_array_equal(dirs.thetas, angles)

    def test_custom_vectors(self):
        vectors = [(1,0,0), (0,1,0), (0,0,1)]
        dirs = Directions.from_vectors(vectors)
        
        self.assertEqual(len(dirs), 3)
        self.assertEqual(dirs.dim, 3)
        self.assertEqual(dirs.sampling, Sampling.CUSTOM)

    def test_invalid_vectors(self):
        with self.assertRaises(ValueError):
            Directions.from_vectors([(0,0), (0,0)])  

    def test_angle_access_3d(self):
        dirs = Directions.uniform(8, dim=3)
        with self.assertRaises(ValueError):
            _ = dirs.thetas  # angles not available in 3D

    def test_endpoint_behavior(self):
        num_dirs = 4
        dirs_with_endpoint = Directions.uniform(num_dirs, endpoint=True)
        dirs_without_endpoint = Directions.uniform(num_dirs, endpoint=False)
        
        self.assertNotEqual(dirs_with_endpoint.thetas[-1], 
                           dirs_without_endpoint.thetas[-1])

    def test_indexing(self):
        dirs = Directions.uniform(10)
        vector = dirs[0]
        self.assertEqual(vector.shape, (2,))
        self.assertAlmostEqual(np.linalg.norm(vector), 1.0)


if __name__ == '__main__':
    unittest.main() 