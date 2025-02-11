from dataclasses import dataclass
import numpy as np
from ect.directions import Directions


@dataclass
class ECTResult:
    matrix: np.ndarray
    directions: Directions
    thresholds: np.ndarray
    vertex_counts: np.ndarray | None = None
    edge_counts: np.ndarray | None = None
    face_counts: np.ndarray | None = None
