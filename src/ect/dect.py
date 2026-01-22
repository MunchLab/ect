from ect import ECT
from .embed_complex import EmbeddedGraph, EmbeddedCW
from .directions import Directions
from .results import ECTResult
from typing import Optional, Union
import numpy as np
from numba import njit


class DECT(ECT):
    """
    A class to calculate the Differentiable Euler Characteristic Transform (DECT)
    """

    def __init__(
        self,
        directions: Optional[Directions] = None,
        num_dirs: Optional[int] = None,
        num_thresh: Optional[int] = None,
        bound_radius: Optional[float] = None,
        thresholds: Optional[np.ndarray] = None,
        dtype=np.float32,
        scale: float = 10.0,
    ):
        """
        Initialize a Differentiable Euler Characteristic Transform (DECT) calculator.

        Args:
            directions (Optional[Directions]): Directions for the transform. If None, directions are generated automatically.
            num_dirs (Optional[int]): Number of directions to use. If None, determined from directions or defaults.
            num_thresh (Optional[int]): Number of thresholds to use. If None, determined from data or defaults.
            bound_radius (Optional[float]): Bounding radius for threshold generation. If None, computed from input.
            thresholds (Optional[np.ndarray]): Array of threshold values. If None, thresholds are generated automatically.
            dtype: Data type for computation (default: np.float32).
            scale (float): Slope parameter for the sigmoid function controlling smoothness (default: 10.0).

        Notes:
            - The scale parameter controls the sharpness of the sigmoid transition in the DECT calculation.
            - All other parameters are passed to the parent :class:`ECT`.
        """
        super().__init__(
            directions, num_dirs, num_thresh, bound_radius, thresholds, dtype
        )
        self.scale = scale

    @staticmethod
    @njit(fastmath=True)
    def _compute_directional_transform(
        simplex_projections_list, thresholds, dtype=np.float32, scale=10.0
    ):
        """
        Compute the Differentiable Euler Characteristic Transform (DECT) using a sigmoid function for smooth transitions.

        Args:
            simplex_projections_list (list of np.ndarray): List of arrays containing projected simplex heights for each direction.
            thresholds (np.ndarray): Array of threshold values.
            dtype: Data type for computation (default: np.float32).
            scale (float): Slope parameter for the sigmoid function controlling smoothness (default: 10.0).

        Returns:
            np.ndarray: DECT matrix of shape

                .. math::

                    (\text{num\_dir}, \text{num\_thresh})

        Notes:
            - The DECT is computed as a sum over all simplices, using a sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ to smoothly count contributions above each threshold.
            - The sign alternates for even/odd simplex dimensions to match Euler characteristic conventions.
        """
        num_dir = simplex_projections_list[0].shape[1]
        num_thresh = len(thresholds)

        output = np.zeros((num_dir, num_thresh), dtype=dtype)

        for i, simplex_heights in enumerate(simplex_projections_list):
            for d in range(num_dir):
                for t, thresh in enumerate(thresholds):
                    diff = scale * (simplex_heights[:, d] - thresh)
                    sigmoid = 1 / (1 + np.exp(-diff))
                    sign = -1 if i % 2 == 0 else 1
                    output[d, t] += sign * np.sum(sigmoid)

        return output

    def calculate(
        self,
        graph: Union[EmbeddedGraph, EmbeddedCW],
        scale: Optional[float] = None,
        theta: Optional[float] = None,
        override_bound_radius: Optional[float] = None,
    ) -> ECTResult:
        """
        Calculate the Differentiable Euler Characteristic Transform (DECT) for a given embedded complex.

        Args:
            graph (EmbeddedGraph or EmbeddedCW): The embedded complex to analyze.
            scale (Optional[float]): Slope parameter for the sigmoid function. If None, uses the instance's scale.
            theta (Optional[float]): Specific direction angle to use. If None, uses all directions.
            override_bound_radius (Optional[float]): Override for bounding radius in threshold generation.

        Returns:
            ECTResult: Result object containing the DECT matrix, directions, and thresholds.

        Notes:
            - Uses :meth:`_compute_directional_transform` for the core DECT calculation.
            - The DECT matrix is computed for all directions and thresholds specified.
        """
        self._ensure_directions(graph.dim, theta)
        self._ensure_thresholds(graph, override_bound_radius)

        directions = (
            self.directions if theta is None else Directions.from_angles([theta])
        )

        simplex_projections = self._compute_simplex_projections(graph, directions)

        scale = self.scale if scale is None else scale

        ect_matrix = self._compute_directional_transform(
            simplex_projections, self.thresholds, self.dtype, scale
        )

        return ECTResult(ect_matrix, directions, self.thresholds)
