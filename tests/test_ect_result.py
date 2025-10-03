import unittest
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from ect import ECT
from ect.utils.examples import create_example_graph
from ect.results import ECTResult


class TestECTResult(unittest.TestCase):
    def setUp(self):
        self.graph = create_example_graph()
        self.ect = ECT(num_dirs=8, num_thresh=10)
        self.result = self.ect.calculate(self.graph)

    def test_array_behavior(self):
        # test numpy array operations
        self.assertTrue(isinstance(self.result + 1, np.ndarray))
        # original array should be integer type
        self.assertTrue(np.issubdtype(self.result.dtype, np.integer))
        # mean operation always returns float type in numpy
        self.assertTrue(np.issubdtype(self.result.mean().dtype, np.floating))
        self.assertEqual(self.result.shape, (8, 10))

    def test_metadata_preservation(self):
        # test metadata is preserved after operations
        result2 = self.result.copy()
        self.assertEqual(result2.directions, self.result.directions)
        self.assertTrue(np.array_equal(result2.thresholds, self.result.thresholds))

    def test_smooth_transform(self):
        smooth = self.result.smooth()

        # test shape preservation
        self.assertEqual(smooth.shape, self.result.shape)

        # test metadata preservation
        self.assertEqual(smooth.directions, self.result.directions)
        self.assertTrue(np.array_equal(smooth.thresholds, self.result.thresholds))

        # test each step of SECT calculation
        data = self.result.astype(np.float64)

        # 1. test that row averages are subtracted correctly
        row_avgs = np.average(data, axis=1)
        for i in range(len(row_avgs)):
            row = data[i] - row_avgs[i]
            self.assertTrue(np.allclose(np.average(row), 0))

        # 2. test that result is cumulative sum of centered data
        centered = data - row_avgs[:, np.newaxis]
        expected_smooth = np.cumsum(centered, axis=1)
        self.assertTrue(np.allclose(smooth, expected_smooth))

    def test_plotting(self):
        # test basic plotting
        ax = self.result.plot()
        self.assertTrue(isinstance(ax, plt.Axes))
        plt.close()

        # test plotting with custom axes
        fig, ax = plt.subplots()
        self.result.plot(ax=ax)
        plt.close()

    def test_single_direction_result(self):
        result = self.ect.calculate(self.graph, theta=0)

        # test shape
        self.assertEqual(result.shape, (1, self.ect.num_thresh))

        # test plotting single direction
        ax = result.plot()
        self.assertTrue(isinstance(ax, plt.Axes))
        plt.close()

    def test_array_finalize(self):
        # test metadata preservation in array operations
        sliced = self.result[2:5]
        self.assertEqual(sliced.directions, self.result.directions)
        self.assertTrue(np.array_equal(sliced.thresholds, self.result.thresholds))

    def test_dist_single_ectresult(self):
        """Test distance computation between two ECTResults"""
        # Create a second ECTResult with same shape
        result2 = self.ect.calculate(self.graph)
        # Modify it slightly
        result2_modified = result2 + 1
        result2_modified.directions = result2.directions
        result2_modified.thresholds = result2.thresholds

        # Test L1 distance (default)
        dist_l1 = self.result.dist(result2_modified)
        expected_l1 = np.abs(self.result - result2_modified).sum()
        self.assertAlmostEqual(dist_l1, expected_l1)
        self.assertIsInstance(dist_l1, (float, np.floating))

        # Test L2 distance
        dist_l2 = self.result.dist(result2_modified, metric="euclidean")
        expected_l2 = np.sqrt(((self.result - result2_modified) ** 2).sum())
        self.assertAlmostEqual(dist_l2, expected_l2)

        # Test L-inf distance
        dist_linf = self.result.dist(result2_modified, metric="chebyshev")
        expected_linf = np.abs(self.result - result2_modified).max()
        self.assertAlmostEqual(dist_linf, expected_linf)

    def test_dist_list_of_ectresults(self):
        """Test batch distance computation with list of ECTResults"""
        # Create multiple ECTResults
        result2 = self.result + 1
        result3 = self.result + 2
        result4 = self.result + 3

        # Preserve metadata
        for r, val in [(result2, 1), (result3, 2), (result4, 3)]:
            r.directions = self.result.directions
            r.thresholds = self.result.thresholds

        # Test batch distances
        distances = self.result.dist([result2, result3, result4])

        # Check return type is array
        self.assertIsInstance(distances, np.ndarray)
        self.assertEqual(len(distances), 3)

        # Check individual distances are correct
        expected_dists = [
            np.abs(self.result - result2).sum(),
            np.abs(self.result - result3).sum(),
            np.abs(self.result - result4).sum(),
        ]
        np.testing.assert_array_almost_equal(distances, expected_dists)

    def test_dist_custom_metric(self):
        """Test distance with custom metric function"""
        result2 = self.result + 1
        result2.directions = self.result.directions
        result2.thresholds = self.result.thresholds

        # Define custom metric - L0.5 norm
        def custom_metric(u, v):
            return np.sum(np.abs(u - v) ** 0.5)

        # Test with custom metric
        dist_custom = self.result.dist(result2, metric=custom_metric)
        expected = custom_metric(self.result.ravel(), result2.ravel())
        self.assertAlmostEqual(dist_custom, expected)

        # Test custom metric with batch
        result3 = self.result + 2
        result3.directions = self.result.directions
        result3.thresholds = self.result.thresholds

        distances = self.result.dist([result2, result3], metric=custom_metric)
        expected_batch = [
            custom_metric(self.result.ravel(), result2.ravel()),
            custom_metric(self.result.ravel(), result3.ravel()),
        ]
        np.testing.assert_array_almost_equal(distances, expected_batch)

    def test_dist_additional_kwargs(self):
        """Test passing additional kwargs to metric functions"""
        result2 = self.result + 1
        result2.directions = self.result.directions
        result2.thresholds = self.result.thresholds

        # Test minkowski with different p values
        dist_p3 = self.result.dist(result2, metric="minkowski", p=3)
        expected_p3 = np.sum(np.abs(self.result - result2) ** 3) ** (1 / 3)
        self.assertAlmostEqual(dist_p3, expected_p3, places=5)

        dist_p5 = self.result.dist(result2, metric="minkowski", p=5)
        expected_p5 = np.sum(np.abs(self.result - result2) ** 5) ** (1 / 5)
        self.assertAlmostEqual(dist_p5, expected_p5, places=5)

    def test_dist_empty_list(self):
        """Test that empty list returns empty array"""
        distances = self.result.dist([])
        self.assertIsInstance(distances, np.ndarray)
        self.assertEqual(len(distances), 0)

    def test_dist_shape_mismatch(self):
        """Test that shape mismatch raises ValueError"""
        # Create ECTResult with different shape
        ect_different = ECT(num_dirs=5, num_thresh=7)
        result_different = ect_different.calculate(self.graph)

        # Single ECTResult with wrong shape
        with self.assertRaises(ValueError) as cm:
            self.result.dist(result_different)
        self.assertIn("Shape mismatch", str(cm.exception))

        # List with one wrong shape
        result2 = self.result + 1
        result2.directions = self.result.directions
        result2.thresholds = self.result.thresholds

        with self.assertRaises(ValueError) as cm:
            self.result.dist([result2, result_different])
        self.assertIn("Shape mismatch at index 1", str(cm.exception))

    def test_dist_self(self):
        """Test distance to self is zero"""
        dist_self = self.result.dist(self.result)
        self.assertEqual(dist_self, 0.0)

        # Also test with different metrics
        self.assertEqual(self.result.dist(self.result, metric="euclidean"), 0.0)
        self.assertEqual(self.result.dist(self.result, metric="chebyshev"), 0.0)

    def test_has_csr_and_to_dense_semantics(self):
        self.assertFalse(self.result.has_csr)
        dense_before = np.asarray(self.result)
        dense_after = self.result.to_dense()
        np.testing.assert_array_equal(dense_before, dense_after)

    def test_from_csr_to_dense_roundtrip(self):
        ect = ECT(num_dirs=8, num_thresh=32)
        res = ect.calculate(self.graph)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ect_sparse.npz")
            res.save_npz(p)
            z = np.load(p, allow_pickle=False)
            row_ptr = z["row_ptr"]
            col_idx = z["col_idx"]
            data = z["data"]
            thresholds = z["thresholds"]

        res2 = ECTResult.from_csr(
            row_ptr, col_idx, data, res.directions, thresholds, dtype=res.dtype
        )
        self.assertTrue(res2.has_csr)
        np.testing.assert_array_equal(
            res2.to_dense(), np.asarray(res, dtype=res2.dtype)
        )
        self.assertEqual(res2.dtype, res.dtype)
        np.testing.assert_array_equal(res2.thresholds, res.thresholds)

    def test_save_load_npz_roundtrip(self):
        ect = ECT(num_dirs=16, num_thresh=64)
        res = ect.calculate(self.graph)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ect_sparse.npz")
            res.save_npz(p)
            loaded = ECTResult.load_npz(p, directions=res.directions)

            # Loaded object should match numerically and carry metadata
            np.testing.assert_array_equal(np.asarray(loaded), np.asarray(res))
            self.assertEqual(loaded.dtype, res.dtype)
            np.testing.assert_array_equal(loaded.thresholds, res.thresholds)
            self.assertEqual(loaded.directions, res.directions)


if __name__ == "__main__":
    unittest.main()
