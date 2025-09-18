"""
Embedding validation system for EmbeddedComplex.

This module provides a flexible, extensible system for validating that
cell complexes represent proper embeddings in Euclidean space.
"""

from .validator import EmbeddingValidator
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

__all__ = [
    "EmbeddingValidator",
    "ValidationRule",
    "ValidationResult",
    "EdgeInteriorRule",
    "FaceInteriorRule",
    "SelfIntersectionRule",
    "BoundaryEdgeRule",
    "DimensionValidityRule",
    "VertexCountRule",
    "validate_coordinate_array",
    "validate_node_existence",
]
