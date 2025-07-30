"""
Main embedding validator class that orchestrates validation rules.
"""

from typing import List, Callable, Optional
import numpy as np

from .base import ValidationRule, ValidationResult
from .rules import (
    EdgeInteriorRule,
    FaceInteriorRule,
    SelfIntersectionRule,
    BoundaryEdgeRule,
    DimensionValidityRule,
    VertexCountRule,
    validate_coordinate_array,
    validate_node_existence,
)


class EmbeddingValidator:
    """
    Main validator that orchestrates multiple validation rules.

    This class manages a collection of validation rules and applies them
    to cells based on their dimension and the configured rule set.
    """

    def __init__(
        self, tolerance: float = 1e-10, edge_checker: Optional[Callable] = None
    ):
        """
        Initialize the embedding validator.

        Args:
            tolerance: Default tolerance for geometric validation
            edge_checker: Function to check if an edge exists between two vertex indices
        """
        self.tolerance = tolerance
        self.edge_checker = edge_checker

        self.rules: List[ValidationRule] = [
            DimensionValidityRule(tolerance),
            VertexCountRule(tolerance),
            EdgeInteriorRule(tolerance),
            FaceInteriorRule(tolerance),
            SelfIntersectionRule(tolerance),
            BoundaryEdgeRule(tolerance, edge_checker),
        ]

    def add_rule(self, rule: ValidationRule) -> "EmbeddingValidator":
        """
        Add a custom validation rule.

        Args:
            rule: ValidationRule to add

        Returns:
            Self for method chaining
        """
        self.rules.append(rule)
        return self

    def remove_rule(self, rule_class: type) -> "EmbeddingValidator":
        """
        Remove all rules of a specific type.

        Args:
            rule_class: Class of rules to remove

        Returns:
            Self for method chaining
        """
        self.rules = [rule for rule in self.rules if not isinstance(rule, rule_class)]
        return self

    def set_tolerance(self, tolerance: float) -> "EmbeddingValidator":
        """
        Set tolerance for all rules.

        Args:
            tolerance: New tolerance value

        Returns:
            Self for method chaining
        """
        self.tolerance = tolerance
        for rule in self.rules:
            rule.set_tolerance(tolerance)
        return self

    def set_edge_checker(self, edge_checker: Callable) -> "EmbeddingValidator":
        """
        Set the edge checker function for boundary validation.

        Args:
            edge_checker: Function that takes two vertex indices and returns bool

        Returns:
            Self for method chaining
        """
        self.edge_checker = edge_checker

        for rule in self.rules:
            if isinstance(rule, BoundaryEdgeRule):
                rule.edge_checker = edge_checker

        return self

    def validate_cell(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int,
        check_geometric: bool = True,
    ) -> ValidationResult:
        """
        Validate a cell using applicable rules.

        Args:
            cell_coords: Coordinates of vertices forming the cell
            all_coords: Coordinates of all vertices in the complex
            cell_indices: Indices of vertices forming the cell
            all_indices: Indices of all vertices in the complex
            dim: Dimension of the cell
            check_geometric: Whether to check geometric rules (structural rules always checked)

        Returns:
            ValidationResult for the cell

        Raises:
            ValueError: If validation fails
        """
        # check all rules that apply to this dimension
        for rule in self.rules:
            if not rule.applies_to_dimension(dim):
                continue
            
            # Always check structural rules, only check geometric rules if requested
            if rule.is_structural or check_geometric:
                result = rule.validate(
                    cell_coords, all_coords, cell_indices, all_indices, dim
                )

                if not result.is_valid:
                    return ValidationResult.invalid(
                        f"{rule.name}: {result.message}", result.violating_indices
                    )

        return ValidationResult.valid()

    def validate_cell_safe(
        self,
        cell_coords: np.ndarray,
        all_coords: np.ndarray,
        cell_indices: List[int],
        all_indices: List[int],
        dim: int,
        check_geometric: bool = True,
    ) -> ValidationResult:
        """
        Validate a cell, catching and wrapping any exceptions.

        Same as validate_cell but returns ValidationResult instead of raising.

        Args:
            cell_coords: Coordinates of vertices forming the cell
            all_coords: Coordinates of all vertices in the complex
            cell_indices: Indices of vertices forming the cell
            all_indices: Indices of all vertices in the complex
            dim: Dimension of the cell
            check_geometric: Whether to check geometric rules (structural rules always checked)

        Returns:
            ValidationResult for the cell
        """
        try:
            return self.validate_cell(
                cell_coords, all_coords, cell_indices, all_indices, dim, check_geometric
            )
        except Exception as e:
            return ValidationResult.invalid(f"Validation error: {str(e)}")

    def get_rules_for_dimension(self, dim: int) -> List[ValidationRule]:
        """
        Get all rules that apply to a specific dimension.

        Args:
            dim: Cell dimension

        Returns:
            List of applicable ValidationRules
        """
        return [rule for rule in self.rules if rule.applies_to_dimension(dim)]

    def get_rule_names(self) -> List[str]:
        """
        Get names of all registered rules.

        Returns:
            List of rule names
        """
        return [rule.name for rule in self.rules]

    def disable_rule(self, rule_class: type) -> "EmbeddingValidator":
        """
        Temporarily disable a rule type.

        Args:
            rule_class: Class of rule to disable

        Returns:
            Self for method chaining
        """
        return self.remove_rule(rule_class)

    def enable_strict_validation(self) -> "EmbeddingValidator":
        """
        Enable strict validation with tight tolerance.

        Returns:
            Self for method chaining
        """
        return self.set_tolerance(1e-12)

    def enable_permissive_validation(self) -> "EmbeddingValidator":
        """
        Enable permissive validation with loose tolerance.

        Returns:
            Self for method chaining
        """
        return self.set_tolerance(1e-6)

    def validate_coordinates(
        self, coords, expected_dim: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate coordinate array format and dimension consistency.

        Args:
            coords: Coordinate array to validate
            expected_dim: Expected dimension, if known

        Returns:
            ValidationResult for the coordinates
        """
        return validate_coordinate_array(coords, expected_dim)

    def validate_nodes(
        self, nodes: List, node_checker: Callable, expect_exists: bool = True
    ) -> ValidationResult:
        """
        Validate node existence/non-existence.

        Args:
            nodes: List of node identifiers to check
            node_checker: Function that takes node_id and returns bool for existence
            expect_exists: Whether nodes should exist (True) or not exist (False)

        Returns:
            ValidationResult for the nodes
        """
        return validate_node_existence(nodes, node_checker, expect_exists)

    def __repr__(self) -> str:
        """String representation of the validator."""
        rule_names = [rule.name for rule in self.rules]
        return f"EmbeddingValidator(tolerance={self.tolerance}, rules={rule_names})"
