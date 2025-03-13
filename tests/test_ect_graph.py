import unittest
import numpy as np
from ect.utils.examples import create_example_graph, create_example_cw
from ect import ECT, Directions


class TestECT(unittest.TestCase):
    def setUp(self):
        self.graph = create_example_graph()
        self.num_dirs = 8
        self.num_thresh = 10
        self.ect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)

    def test_initialization(self):
        self.assertIsNone(self.ect.bound_radius)
        self.assertEqual(self.ect.num_dirs, self.num_dirs)
        self.assertEqual(self.ect.num_thresh, self.num_thresh)
        self.assertIsNone(self.ect.directions)

    def test_calculate_basic(self):
        result = self.ect.calculate(self.graph)
        self.assertEqual(result.shape, (self.num_dirs, self.num_thresh))
        self.assertTrue(isinstance(result.directions, Directions))
        self.assertIsNotNone(result.thresholds)

    def test_calculate_single_direction(self):
        result = self.ect.calculate(self.graph, theta=0)
        self.assertEqual(result.shape, (1, self.num_thresh))
        self.assertEqual(len(result.directions), 1)

    def test_bounding_radius(self):
        radius = 2.0
        self.ect.set_bounding_radius(radius)
        self.assertEqual(self.ect.bound_radius, radius)
        self.assertEqual(len(self.ect.thresholds), self.num_thresh)
        self.assertEqual(self.ect.thresholds[0], -radius)
        self.assertEqual(self.ect.thresholds[-1], radius)

    def test_invalid_bounding_radius(self):
        with self.assertRaises(ValueError):
            self.ect.set_bounding_radius(-1)
        with self.assertRaises(ValueError):
            self.ect.calculate(self.graph, bound_radius=-1)

    def test_threshold_priority(self):
        graph_radius = self.graph.get_bounding_radius()
        instance_radius = 2 * graph_radius
        override_radius = 3 * graph_radius

        # test graph radius is used when no other radius specified
        result1 = self.ect.calculate(self.graph)
        self.assertAlmostEqual(abs(result1.thresholds).max(), graph_radius)

        # test instance radius takes precedence over graph radius
        self.ect.set_bounding_radius(instance_radius)
        result2 = self.ect.calculate(self.graph)
        self.assertAlmostEqual(abs(result2.thresholds).max(), instance_radius)

        # test override radius takes precedence over both
        result3 = self.ect.calculate(self.graph, bound_radius=override_radius)
        self.assertAlmostEqual(abs(result3.thresholds).max(), override_radius)

    def test_different_graph_types(self):
        cw = create_example_cw()
        result_graph = self.ect.calculate(self.graph)
        result_cw = self.ect.calculate(cw)
        
        self.assertEqual(result_graph.shape, result_cw.shape)
        self.assertEqual(len(result_graph.directions), len(result_cw.directions))

    def test_directions_matching(self):
        # test that directions are reinitialized when dimensions don't match
        G2d = create_example_graph()
        directions_3d = Directions.uniform(self.num_dirs, dim=3)
        ect = ECT(directions=directions_3d)
        
        result = ect.calculate(G2d)
        self.assertEqual(result.directions.dim, 2)
        self.assertEqual(len(result.directions), self.num_dirs)

    def test_result_properties(self):
        result = self.ect.calculate(self.graph)
        
        # test smooth transform
        smooth = result.smooth()
        self.assertEqual(smooth.shape, result.shape)
        self.assertEqual(smooth.directions, result.directions)
        self.assertEqual(smooth.thresholds.tolist(), result.thresholds.tolist())

        # verify result is integer-valued
        self.assertTrue(np.issubdtype(result.dtype, np.integer))


if __name__ == '__main__':
    unittest.main()