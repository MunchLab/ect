import numpy as np


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting algorithm for point-in-polygon test"""
    x, y = point
    n = polygon.shape[0]
    inside = False
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        if ((p1[1] > y) != (p2[1] > y)):
            xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if x <= xinters:
                inside = not inside
    return inside
