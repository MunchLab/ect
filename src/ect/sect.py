from ect import ECT
from .embed_graph import EmbeddedGraph
from .embed_cw import EmbeddedCW
from .directions import Directions
from .results import ECTResult
from typing import Optional, Union
import numpy as np


class SECT(ECT):
    """
    A class to calculate the Smooth Euler Characteristic Transform (SECT).
    Inherits from ECT and applies smoothing to the final result.
    """

    def __init__(
        self,
        directions: Optional[Directions] = None,
        num_dirs: Optional[int] = None,
        num_thresh: Optional[int] = None,
        bound_radius: Optional[float] = None,
        thresholds: Optional[np.ndarray] = None,
        dtype=np.float32,
    ):
        """Initialize SECT calculator with smoothing parameter

        Args:
            directions: Optional pre-configured Directions object
            num_dirs: Number of directions to sample (ignored if directions provided)
            num_thresh: Number of threshold values (required if directions not provided)
            bound_radius: Optional radius for bounding circle
            thresholds: Optional array of thresholds
            dtype: Data type for output array
        """
        super().__init__(
            directions, num_dirs, num_thresh, bound_radius, thresholds, dtype
        )

    def calculate(
        self,
        graph: Union[EmbeddedGraph, EmbeddedCW],
        theta: Optional[float] = None,
        override_bound_radius: Optional[float] = None,
    ) -> ECTResult:
        """Calculate Smooth Euler Characteristic Transform (SECT)

        Args:
            graph: The input graph to calculate the SECT for
            theta: The angle in [0,2π] for the direction to calculate the SECT
            override_bound_radius: Optional override for bounding radius

        Returns:
            ECTResult: The smoothed transform result containing the matrix,
                      directions, and thresholds
        """
        ect_result = super().calculate(graph, theta, override_bound_radius)
        return ECTResult(
            ect_result, ect_result.directions, ect_result.thresholds
        ).smooth()
