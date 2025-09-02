"""
ECT: A Python package for computing the Euler Characteristic Transform

Main classes:
    ECT: Calculator for Euler Characteristic Transform
    EmbeddedComplex: Unified embedded cell complex supporting arbitrary dimensional cells
    EmbeddedGraph: Alias for EmbeddedComplex (for backward compatibility)
    EmbeddedCW: Alias for EmbeddedComplex (for backward compatibility)
    Directions: Direction vector management for ECT computation
    SECT: Smooth Euler Characteristic Transform calculator
"""

from .ect import ECT
from .embed_complex import EmbeddedComplex, EmbeddedGraph, EmbeddedCW
from .directions import Directions
from .sect import SECT
from .dect import DECT
from .utils import examples

__all__ = [
    "ECT",
    "SECT",
    "DECT",
    "EmbeddedComplex",
    "EmbeddedGraph",
    "EmbeddedCW",
    "Directions",
    "examples",
]
