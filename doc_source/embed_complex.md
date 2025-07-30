# Embedded Complex

The `EmbeddedComplex` class is a unified representation for embedded cell complexes supporting arbitrary dimensional cells.

## Overview

`EmbeddedComplex` combines and extends the functionality of the previous `EmbeddedGraph` and `EmbeddedCW` classes into a single, more powerful interface. It supports:

- **0-cells (vertices)**: Points embedded in Euclidean space
- **1-cells (edges)**: Line segments between vertices
- **k-cells for k ≥ 2**: Higher dimensional cells (faces, volumes, etc.)

## Key Features

- **Unified Interface**: Single class for all cell complex operations
- **Arbitrary Dimensions**: Support for k-cells of any dimension k ≥ 0
- **Modular Validation**: Pluggable validation system for embedding properties
- **Backward Compatible**: `EmbeddedGraph` and `EmbeddedCW` remain as aliases

## Basic Usage

```python
from ect import EmbeddedComplex

# Create a complex
K = EmbeddedComplex()

# Add vertices
K.add_node("A", [0, 0])
K.add_node("B", [1, 0])
K.add_node("C", [0.5, 1])

# Add edges
K.add_edge("A", "B")
K.add_edge("B", "C")
K.add_edge("C", "A")

# Add a 2-cell (face)
K.add_face(["A", "B", "C"])

# Or use the general method for any dimension
K.add_cell(["A", "B", "C"], dim=2)
```

## API Reference

```{eval-rst}
.. automodule:: ect.embed_complex
   :members:
   :show-inheritance:
   :undoc-members:
```