import unittest
import numpy as np
import matplotlib.pyplot as plt
from ect import ECT, Directions
from ect.utils.examples import create_example_graph


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


if __name__ == '__main__':
    unittest.main() 