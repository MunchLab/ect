import unittest
import numpy as np
from ect.utils.examples import create_example_graph, create_example_cw
from ect import DECT, ECT, Directions, EmbeddedComplex
from ect.results import ECTResult


class TestDECTBasicFunctionality(unittest.TestCase):
    def setUp(self):
        self.graph = create_example_graph()
        self.num_dirs = 8
        self.num_thresh = 10
        self.bound_radius = 2.0

    def test_initialization_with_scale_parameter(self):
        """Test that DECT initializes correctly with scale parameter"""
        # Test with default scale
        dect1 = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )
        self.assertEqual(dect1.scale, 10.0)  # Default scale

        # Test with custom scale
        custom_scale = 25.0
        dect2 = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
            scale=custom_scale,
        )
        self.assertEqual(dect2.scale, custom_scale)

        # Verify parent class attributes are initialized
        self.assertEqual(dect2.bound_radius, self.bound_radius)
        self.assertEqual(dect2.num_dirs, self.num_dirs)
        self.assertEqual(dect2.num_thresh, self.num_thresh)

    def test_calculate_with_default_and_custom_scale(self):
        """Test calculate method with both default and custom scale values"""
        dect = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
            scale=15.0,  # Init scale
        )

        # Test with default scale from init
        result1 = dect.calculate(self.graph)
        self.assertIsNotNone(result1)

        # Test with override scale in calculate
        result2 = dect.calculate(self.graph, scale=50.0)
        self.assertIsNotNone(result2)

        # Results should be different due to different scales
        self.assertFalse(np.allclose(result1, result2))

        # Test that None scale uses init scale
        result3 = dect.calculate(self.graph, scale=None)
        self.assertTrue(np.allclose(result1, result3))

    def test_inheritance_from_ect(self):
        """Test that DECT properly inherits from ECT"""
        dect = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )

        # Check that DECT is instance of ECT
        self.assertIsInstance(dect, ECT)

        # Verify inherited methods are available
        self.assertTrue(hasattr(dect, "_ensure_directions"))
        self.assertTrue(hasattr(dect, "_ensure_thresholds"))
        self.assertTrue(hasattr(dect, "_compute_simplex_projections"))
        self.assertTrue(hasattr(dect, "calculate"))

    def test_result_shape_and_type(self):
        """Test that DECT returns correct result shape and type"""
        dect = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )

        result = dect.calculate(self.graph)

        # Check result type
        self.assertIsInstance(result, ECTResult)

        # Check shape
        self.assertEqual(result.shape, (self.num_dirs, self.num_thresh))

        # Check that directions and thresholds are included
        self.assertIsNotNone(result.directions)
        self.assertIsInstance(result.directions, Directions)
        self.assertEqual(len(result.directions), self.num_dirs)

        self.assertIsNotNone(result.thresholds)
        self.assertEqual(len(result.thresholds), self.num_thresh)

        # Check dtype - ECTResult always converts to float64 for float types
        dect_float32 = DECT(
            num_dirs=self.num_dirs, num_thresh=self.num_thresh, dtype=np.float32
        )
        result_float32 = dect_float32.calculate(self.graph)
        # ECTResult converts float types to float64
        self.assertEqual(result_float32.dtype, np.float64)

        # Test with float64
        dect_float64 = DECT(
            num_dirs=self.num_dirs, num_thresh=self.num_thresh, dtype=np.float64
        )
        result_float64 = dect_float64.calculate(self.graph)
        self.assertEqual(result_float64.dtype, np.float64)

    def test_calculate_with_single_direction(self):
        """Test DECT calculation with a single direction (theta parameter)"""
        dect = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )

        # Test with specific theta
        result = dect.calculate(self.graph, theta=np.pi / 4)

        # Should have single direction
        self.assertEqual(result.shape, (1, self.num_thresh))
        self.assertEqual(len(result.directions), 1)

    def test_with_different_graph_types(self):
        """Test DECT works with both EmbeddedGraph and EmbeddedCW"""
        dect = DECT(
            num_dirs=self.num_dirs,
            num_thresh=self.num_thresh,
            bound_radius=self.bound_radius,
        )

        # Test with graph
        graph = create_example_graph()
        result_graph = dect.calculate(graph)
        self.assertEqual(result_graph.shape, (self.num_dirs, self.num_thresh))

        # Test with CW complex
        cw = create_example_cw()
        result_cw = dect.calculate(cw)
        self.assertEqual(result_cw.shape, (self.num_dirs, self.num_thresh))


if __name__ == "__main__":
    unittest.main()
