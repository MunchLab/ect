import unittest
import numpy as np
from ect import SECT, ECT
from ect.utils.examples import create_example_graph
from ect.directions import Directions


class TestSECT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.graph = create_example_graph()
        self.num_dirs = 8
        self.num_thresh = 10
        self.sect = SECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)

    def test_inheritance(self):
        """Test that SECT properly inherits from ECT"""
        self.assertIsInstance(self.sect, ECT)
        self.assertTrue(hasattr(self.sect, "calculate"))

    def test_calculate_output_shape(self):
        """Test that SECT calculation returns correct shape"""
        result = self.sect.calculate(self.graph)

        self.assertEqual(result.shape[0], self.num_dirs)
        self.assertEqual(result.shape[1], self.num_thresh)
        self.assertEqual(len(result.thresholds), self.num_thresh)
        self.assertEqual(len(result.directions), self.num_dirs)

    def test_smoothing_effect(self):
        """Test that smoothing is actually applied"""
        # Calculate both ECT and SECT
        ect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)
        sect = SECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)

        ect_result = ect.calculate(self.graph)
        sect_result = sect.calculate(self.graph)

        # Verify results are different due to smoothing
        self.assertFalse(np.allclose(ect_result, sect_result))

        # Verify smoothing preserves direction count
        self.assertEqual(
            np.sum(ect_result, axis=1).shape,
            np.sum(sect_result, axis=1).shape,
        )

    def test_with_theta(self):
        """Test SECT calculation with specific theta value"""
        theta = np.pi / 4
        result = self.sect.calculate(self.graph, theta=theta)

        # Should only have one direction when theta is specified
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], self.num_thresh)

    def test_with_override_radius(self):
        """Test SECT calculation with override_bound_radius"""
        override_radius = 2.0
        result = self.sect.calculate(self.graph, override_bound_radius=override_radius)

        # Check that thresholds are within the override radius
        self.assertLessEqual(np.max(np.abs(result.thresholds)), override_radius)

    def test_smooth_matrix_properties(self):
        """Test properties of the smoothed matrix"""
        result = self.sect.calculate(self.graph)

        # Smoothed values should be finite
        self.assertTrue(np.all(np.isfinite(result)))

        # Shape should be preserved after smoothing
        self.assertEqual(result.shape, (self.num_dirs, self.num_thresh))

        # Verify result is float type after smoothing
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
