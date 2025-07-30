"""
Base classes for the embedding validation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    message: str = ""
    violating_indices: Optional[List[int]] = None

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True)

    @classmethod
    def invalid(
        cls, message: str, violating_indices: Optional[List[int]] = None
    ) -> "ValidationResult":
        """Create an invalid result with error message."""
        return cls(is_valid=False, message=message, violating_indices=violating_indices)


class ValidationRule(ABC):
    """Abstract base class for embedding validation rules."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize validation rule.

        Args:
            tolerance: Numerical tolerance for geometric checks
        """
        self.tolerance = tolerance

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this validation rule."""
        pass

    @property
    def is_structural(self) -> bool:
        """
        Check if this is a structural rule that should always be validated.
        
        Structural rules check basic requirements like vertex counts and dimension validity.
        They should always be checked regardless of validation settings.
        Geometric rules check embedding properties and are optional.
        
        Returns:
            True if this rule should always be checked, False if optional
        """
        return False

    @abstractmethod
    def applies_to_dimension(self, dim: int) -> bool:
        """
        Check if this rule applies to cells of the given dimension.

        Args:
            dim: Cell dimension

        Returns:
            True if this rule should validate cells of this dimension
        """
        pass

    @abstractmethod
    def validate(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int = None,
    ) -> ValidationResult:
        """
        Validate a cell against this rule.

        Args:
            cell_coords: Coordinates of vertices forming the cell
            all_coords: Coordinates of all vertices in the complex
            cell_indices: Indices of vertices forming the cell
            all_indices: Indices of all vertices in the complex
            dim: Dimension of the cell being validated

        Returns:
            ValidationResult indicating if the cell is valid
        """
        pass

    def set_tolerance(self, tolerance: float) -> "ValidationRule":
        """
        Set the tolerance for this rule.

        Args:
            tolerance: New tolerance value

        Returns:
            Self for method chaining
        """
        self.tolerance = tolerance
        return self
