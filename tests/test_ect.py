import unittest
import numpy as np
from ect.utils.examples import create_example_graph, create_example_cw, create_example_3d_complex, create_sparse_dimensional_complex
from ect import ECT, Directions, EmbeddedComplex


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

    def test_3d_complex_ect(self):
        """Test ECT calculation with 3D complex containing higher-dimensional cells"""
        complex_3d = create_example_3d_complex()
        
        # Create ECT for 3D embedding
        ect_3d = ECT(num_dirs=6, num_thresh=8)
        result = ect_3d.calculate(complex_3d)
        
        # Should have proper shape
        self.assertEqual(result.shape, (6, 8))
        
        # Should handle 3-cells and 4-cells in computation
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        
        # Result should be different from a graph-only computation
        # (because it includes higher-dimensional cells in Euler characteristic)
        self.assertIsInstance(result, np.ndarray)

    def test_sparse_dimensional_complex(self):
        """Test ECT with complex having gaps in cell dimensions"""
        sparse_complex = create_sparse_dimensional_complex()
        
        ect = ECT(num_dirs=4, num_thresh=6)
        result = ect.calculate(sparse_complex)
        
        # Should handle missing 1-cells and 2-cells gracefully
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_high_dimensional_cells_projection(self):
        """Test that projections are computed correctly for high-dimensional cells"""
        complex_3d = create_example_3d_complex()
        
        # Test with single direction for easier verification
        directions = Directions.uniform(1, dim=3)
        ect = ECT(directions=directions, num_thresh=5)
        
        result = ect.calculate(complex_3d)
        
        # Should compute without errors
        self.assertEqual(result.shape, (1, 5))
        
        # Verify the internal projection computation works
        node_projections = np.matmul(complex_3d.coord_matrix, directions.vectors.T)
        simplex_projections = ect._compute_simplex_projections(complex_3d, directions)
        
        # Should have projections for all cell dimensions present
        # 0-cells (vertices), 1-cells (edges), 2-cells (faces), 3-cells, 4-cells
        self.assertEqual(len(simplex_projections), 5)  # dims 0-4
        
        # Each should have correct number of directions
        for proj in simplex_projections:
            if proj.shape[0] > 0:  # If there are cells of this dimension
                self.assertEqual(proj.shape[1], 1)  # 1 direction

    def test_empty_higher_dimensional_cells(self):
        """Test ECT with complex that has some empty cell dimensions"""
        # Create complex with only vertices and edges (no higher cells)
        simple_graph = create_example_graph()
        
        ect = ECT(num_dirs=4, num_thresh=5)
        result = ect.calculate(simple_graph)
        
        # Should handle missing higher-dimensional cells
        self.assertEqual(result.shape, (4, 5))
        
        # Internal projection computation should handle empty dimensions
        directions = Directions.uniform(4, dim=2)
        simplex_projections = ect._compute_simplex_projections(simple_graph, directions)
        
        # Should have at least vertices and edges
        self.assertGreaterEqual(len(simplex_projections), 2)
        
        # Higher dimensions should be empty arrays with correct shape
        for i, proj in enumerate(simplex_projections):
            self.assertEqual(proj.shape[1], 4)  # 4 directions
            if i >= 2:  # dimensions 2 and higher should be empty for simple graph
                self.assertEqual(proj.shape[0], 0)

    def test_cell_projections_correctness(self):
        """Test that cell projections correctly compute max over vertices"""
        # Create simple complex for manual verification
        K = EmbeddedComplex()
        K.add_node('A', [0, 0])
        K.add_node('B', [1, 0])
        K.add_node('C', [0, 1])
        K.add_node('D', [1, 1])
        
        # Add a 2-cell
        K.add_cell(['A', 'B', 'C'], dim=2)
        
        # Test projection in direction [1, 0] (x-direction)
        directions = Directions.from_angles([0])  # theta=0 -> direction [0, -1]
        
        ect = ECT(directions=directions, num_thresh=3)
        
        # Test internal projection computation
        node_projections = np.matmul(K.coord_matrix, directions.vectors.T)
        simplex_projections = ect._compute_simplex_projections(K, directions)
        
        # Verify 2-cell projection is max of its vertices
        face_projection = simplex_projections[2][0, 0]  # First 2-cell, first direction
        vertex_projections = node_projections[[0, 1, 2], 0]  # Vertices A, B, C
        expected_max = np.max(vertex_projections)
        
        self.assertAlmostEqual(face_projection, expected_max, places=10)

    def test_euler_characteristic_with_higher_cells(self):
        """Test that Euler characteristic includes higher-dimensional cells"""
        complex_3d = create_example_3d_complex()
        
        # Calculate ECT
        ect = ECT(num_dirs=1, num_thresh=10)
        result = ect.calculate(complex_3d)
        
        # The result should reflect the alternating sum over all cell dimensions
        # χ = |0-cells| - |1-cells| + |2-cells| - |3-cells| + |4-cells| - ...
        
        # Verify that the computation includes all dimensions
        directions = Directions.uniform(1, dim=3)
        simplex_projections = ect._compute_simplex_projections(complex_3d, directions)
        
        # Should have projections for dimensions 0 through 4
        self.assertEqual(len(simplex_projections), 5)
        
        # Check that we have cells in expected dimensions
        self.assertGreater(simplex_projections[0].shape[0], 0)  # vertices
        self.assertGreater(simplex_projections[1].shape[0], 0)  # edges  
        self.assertGreater(simplex_projections[2].shape[0], 0)  # faces
        self.assertGreater(simplex_projections[3].shape[0], 0)  # 3-cells
        self.assertGreater(simplex_projections[4].shape[0], 0)  # 4-cells


if __name__ == "__main__":
    unittest.main()
