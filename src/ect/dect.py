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
        """Initialize DECT calculator"""
        super().__init__(
            directions, num_dirs, num_thresh, bound_radius, thresholds, dtype
        )
        self.scale = scale

    @staticmethod
    @njit(fastmath=True)
    def _compute_directional_transform(
        simplex_projections_list, thresholds, dtype=np.float32, scale=10.0
    ):
        """Compute DECT using sigmoid for smooth transitions"""
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
        """Calculate Differentiable Euler Characteristic Transform (DECT)"""
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
