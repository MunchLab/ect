"""
Unified validation rules for embedding validation system.

This module contains all validation rules including both structural rules
(which are always checked) and geometric rules (which are optional).
"""

from typing import List, Callable, Optional
import numpy as np

from .base import ValidationRule, ValidationResult
from ..utils.face_check import (
    validate_edge_embedding,
    validate_face_embedding,
    segments_intersect,
)


# ================================
# STRUCTURAL RULES (Always checked)
# ================================

class DimensionValidityRule(ValidationRule):
    """Validates that cell dimension is non-negative."""

    @property
    def name(self) -> str:
        return "Dimension Validity"

    @property
    def is_structural(self) -> bool:
        return True

    def applies_to_dimension(self, dim: int) -> bool:
        return True

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate that dimension is non-negative."""
        if dim is not None and dim < 0:
            return ValidationResult.invalid("Cell dimension must be non-negative")
        return ValidationResult.valid()


class VertexCountRule(ValidationRule):
    """Validates that cells have the correct number of vertices for their dimension."""

    @property
    def name(self) -> str:
        return "Vertex Count Validation"

    @property
    def is_structural(self) -> bool:
        return True

    def applies_to_dimension(self, dim: int) -> bool:
        return True

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate vertex count matches cell dimension requirements."""
        if dim is None:
            return ValidationResult.valid()

        vertex_count = len(cell_indices)

        if dim == 0 and vertex_count != 1:
            return ValidationResult.invalid("0-cells must contain exactly one vertex")
        elif dim == 1 and vertex_count != 2:
            return ValidationResult.invalid("1-cells must contain exactly two vertices")
        elif dim >= 2 and vertex_count < 3:
            return ValidationResult.invalid(
                f"{dim}-cells must contain at least 3 vertices"
            )

        return ValidationResult.valid()


class CoordinateDimensionRule(ValidationRule):
    """Validates that coordinates have consistent dimensions."""

    def __init__(
        self, tolerance: float = 1e-10, dimension_checker: Optional[Callable] = None
    ):
        super().__init__(tolerance)
        self.dimension_checker = dimension_checker  # function to get expected dimension

    @property
    def name(self) -> str:
        return "Coordinate Dimension Validation"

    @property
    def is_structural(self) -> bool:
        return True

    def applies_to_dimension(self, dim: int) -> bool:
        return True  # applies to all operations involving coordinates

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate coordinate dimensions are consistent."""
        if self.dimension_checker is None or cell_coords is None:
            return ValidationResult.valid()

        expected_dim = self.dimension_checker()
        if expected_dim is None or expected_dim == 0:
            return ValidationResult.valid()

        # check that all coordinates have the expected dimension
        if cell_coords.ndim != 2:
            return ValidationResult.invalid("Coordinates must be a 2D array")

        if cell_coords.shape[1] != expected_dim:
            return ValidationResult.invalid(
                f"Coordinates must have dimension {expected_dim}, got {cell_coords.shape[1]}"
            )

        return ValidationResult.valid()


# ================================
# GEOMETRIC RULES (Optional)
# ================================

class EdgeInteriorRule(ValidationRule):
    """Validates that no vertices lie on edge interiors."""

    @property
    def name(self) -> str:
        return "Edge Interior Validation"

    def applies_to_dimension(self, dim: int) -> bool:
        return dim == 1

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate that no other vertices lie on this edge's interior."""
        is_valid, error_msg = validate_edge_embedding(
            cell_coords, all_coords, cell_indices, all_indices, self.tolerance
        )

        if is_valid:
            return ValidationResult.valid()
        else:
            return ValidationResult.invalid(error_msg)


class FaceInteriorRule(ValidationRule):
    """Validates that no vertices lie inside face interiors."""

    @property
    def name(self) -> str:
        return "Face Interior Validation"

    def applies_to_dimension(self, dim: int) -> bool:
        return dim == 2

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate that no other vertices lie inside this face."""
        is_valid, error_msg = validate_face_embedding(
            cell_coords, all_coords, cell_indices, all_indices, self.tolerance
        )

        if is_valid:
            return ValidationResult.valid()
        else:
            return ValidationResult.invalid(error_msg)


class SelfIntersectionRule(ValidationRule):
    """Validates that face edges don't self-intersect."""

    @property
    def name(self) -> str:
        return "Self-Intersection Validation"

    def applies_to_dimension(self, dim: int) -> bool:
        return dim == 2

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate that face edges don't intersect each other"""

        if cell_coords.shape[1] > 2:
            coords_2d = cell_coords[:, :2]
        else:
            coords_2d = cell_coords

        n = len(coords_2d)
        if n < 3:
            return ValidationResult.invalid("Face must have at least 3 vertices")

        for i in range(n):
            edge1_start = coords_2d[i]
            edge1_end = coords_2d[(i + 1) % n]

            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue

                edge2_start = coords_2d[j]
                edge2_end = coords_2d[(j + 1) % n]

                if segments_intersect(
                    edge1_start, edge1_end, edge2_start, edge2_end, self.tolerance
                ):
                    return ValidationResult.invalid(
                        f"Face edges intersect: edge {i}-{(i + 1) % n} intersects edge {j}-{(j + 1) % n}"
                    )

        return ValidationResult.valid()


class BoundaryEdgeRule(ValidationRule):
    """Validates that required boundary edges exist for faces."""

    def __init__(self, tolerance: float = 1e-10, edge_checker=None):
        """
        Initialize boundary edge rule.

        Args:
            tolerance: Numerical tolerance
            edge_checker: Function to check if an edge exists (injected dependency)
        """
        super().__init__(tolerance)
        self.edge_checker = edge_checker

    @property
    def name(self) -> str:
        return "Boundary Edge Validation"

    def applies_to_dimension(self, dim: int) -> bool:
        return dim == 2

    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """Validate that all boundary edges exist for this face."""
        if self.edge_checker is None:
            return ValidationResult.valid()

        n = len(cell_indices)
        if n < 3:
            return ValidationResult.invalid("Face must have at least 3 vertices")

        for i in range(n):
            v1_idx = cell_indices[i]
            v2_idx = cell_indices[(i + 1) % n]

            if not self.edge_checker(v1_idx, v2_idx):
                return ValidationResult.invalid(
                    f"Boundary edge between vertices {v1_idx} and {v2_idx} missing for face"
                )

        return ValidationResult.valid()


# ================================
# STANDALONE UTILITY FUNCTIONS
# ================================

def validate_coordinate_array(
    coords, expected_dim: Optional[int] = None
) -> ValidationResult:
    """Standalone function to validate coordinate array format."""
    if coords is None:
        return ValidationResult.valid()

    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 1:
        return ValidationResult.invalid("Coordinates must be a 1D array")

    if expected_dim is not None and coords.size != expected_dim:
        return ValidationResult.invalid(
            f"Coordinates must have dimension {expected_dim}, got {coords.size}"
        )

    return ValidationResult.valid()


def validate_node_existence(
    nodes: List, node_checker: Callable, expect_exists: bool = True
) -> ValidationResult:
    """Standalone function to validate node existence."""
    if node_checker is None:
        return ValidationResult.valid()

    for node_id in nodes:
        node_exists = node_checker(node_id)
        if expect_exists and not node_exists:
            return ValidationResult.invalid(f"Node {node_id} does not exist")
        if not expect_exists and node_exists:
            return ValidationResult.invalid(f"Node {node_id} already exists")

    return ValidationResult.valid()