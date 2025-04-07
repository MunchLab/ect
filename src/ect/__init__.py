"""
ECT: A Python package for computing the Euler Characteristic Transform

Main classes:
    ECT: Calculator for Euler Characteristic Transform
    EmbeddedGraph: Graph representation for ECT computation
    EmbeddedCW: CW complex representation for ECT computation
    Directions: Direction vector management for ECT computation
"""

from .ect_graph import ECT
from .embed_graph import EmbeddedGraph
from .embed_cw import EmbeddedCW
from .directions import Directions
from .sect import SECT
from .utils import examples

__all__ = [
    "ECT",
    "SECT",
    "EmbeddedGraph",
    "EmbeddedCW",
    "Directions",
    "examples",
]
