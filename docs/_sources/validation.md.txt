# Validation System

The validation system provides modular, extensible validation for embedded cell complexes to ensure they represent proper embeddings in Euclidean space.

## Overview

The validation system distinguishes between two types of rules:

- **Structural Rules** (always checked): Basic requirements like vertex counts and dimension validity
- **Geometric Rules** (optional): Embedding properties like non-intersecting edges and faces

## Architecture

The validation system consists of several components:

1. **Base Classes**: Abstract interfaces for validation rules and results
2. **Validation Rules**: Concrete implementations for specific validation checks
3. **Validator**: Main orchestrator that manages and applies rules

## Validation Rules

### Structural Rules (Always Enforced)

- **DimensionValidityRule**: Ensures cell dimensions are non-negative
- **VertexCountRule**: Validates correct vertex counts for cell dimensions
  - 0-cells must have exactly 1 vertex
  - 1-cells must have exactly 2 vertices
  - k-cells (k ≥ 2) must have at least 3 vertices

### Geometric Rules (Optional)

- **EdgeInteriorRule**: Ensures no vertices lie on edge interiors
- **FaceInteriorRule**: Ensures no vertices lie inside face interiors
- **SelfIntersectionRule**: Validates that face edges don't self-intersect
- **BoundaryEdgeRule**: Ensures required boundary edges exist for faces

## Usage

```python
from ect import EmbeddedComplex

# Enable validation during construction
K = EmbeddedComplex(validate_embedding=True)

# Or enable/disable later
K.enable_embedding_validation(tol=1e-10)
K.disable_embedding_validation()

# Override per operation
K.add_cell(vertices, dim=2, check=True)
```

## Custom Validation Rules

You can create custom validation rules by inheriting from `ValidationRule`:

```python
from ect.validation import ValidationRule, ValidationResult

class MyCustomRule(ValidationRule):
    @property
    def name(self) -> str:
        return "My Custom Rule"
    
    @property
    def is_structural(self) -> bool:
        return False  # Geometric rule
    
    def applies_to_dimension(self, dim: int) -> bool:
        return dim == 2  # Only for 2-cells
    
    def validate(self, cell_coords, all_coords, cell_indices, all_indices, dim):
        # Your validation logic here
        if some_condition:
            return ValidationResult.valid()
        else:
            return ValidationResult.invalid("Validation failed")

# Add to validator
K.get_validator().add_rule(MyCustomRule())
```

## API Reference

### Main Module

```{eval-rst}
.. automodule:: ect.validation
   :members:
```

### Base Classes

```{eval-rst}
.. automodule:: ect.validation.base
   :members:
   :show-inheritance:
```

### Validation Rules

```{eval-rst}
.. automodule:: ect.validation.rules
   :members:
   :show-inheritance:
```

### Validator

```{eval-rst}
.. automodule:: ect.validation.validator
   :members:
   :show-inheritance:
```