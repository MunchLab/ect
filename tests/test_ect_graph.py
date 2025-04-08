import unittest
import numpy as np
from ect.utils.examples import create_example_graph, create_example_cw
from ect import ECT, Directions


class TestECT(unittest.TestCase):
    def setUp(self):
        self.graph = create_example_graph()
        self.num_dirs = 8
        self.num_thresh = 10
        self.bound_radius = 2.0
        self.ect = ECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )

    def test_initialization(self):
        self.assertEqual(self.ect.bound_radius, self.bound_radius)
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

    def test_threshold_priority(self):
        graph_radius = self.graph.get_bounding_radius()
        override_bound_radius = 3 * graph_radius

        no_radius_ect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)
        # test graph radius is used when no other radius specified
        result1 = no_radius_ect.calculate(self.graph)
        self.assertAlmostEqual(abs(result1.thresholds).max(), graph_radius)

        # test override radius takes precedence over both
        result2 = self.ect.calculate(
            self.graph, override_bound_radius=override_bound_radius
        )
        self.assertAlmostEqual(abs(result2.thresholds).max(), override_bound_radius)

    def test_different_graph_types(self):
        cw = create_example_cw()
        result_graph = self.ect.calculate(self.graph)
        result_cw = self.ect.calculate(cw)

        self.assertEqual(result_graph.shape, result_cw.shape)
        self.assertEqual(len(result_graph.directions), len(result_cw.directions))

    def test_directions_matching(self):
        # test that ect raises error when dimensions don't match
        G2d = create_example_graph()
        directions_3d = Directions.uniform(self.num_dirs, dim=3)
        ect = ECT(directions=directions_3d)

        with self.assertRaises(ValueError):
            ect.calculate(G2d)

    def test_result_properties(self):
        result = self.ect.calculate(self.graph)

        # test smooth transform
        smooth = result.smooth()
        self.assertEqual(smooth.shape, result.shape)
        self.assertEqual(smooth.directions, result.directions)
        self.assertEqual(smooth.thresholds.tolist(), result.thresholds.tolist())

        # verify result is integer-valued
        self.assertTrue(np.issubdtype(result.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
