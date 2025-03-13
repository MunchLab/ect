from ect.embed_graph import EmbeddedGraph
from ect.embed_cw import EmbeddedCW
from typing import Type
import numpy as np


def create_example_cw(centered=True, center_type='bounding_box'):
    """
    Creates an example EmbeddedCW object with a simple CW complex. If centered is True, the coordinates are centered around the center type, which could be ``mean``, ``bounding_box`` or ``origin``.


    Returns:
        EmbeddedCW
            The example EmbeddedCW object.
    """
    G = create_example_graph(centered=False)
    K = EmbeddedCW()
    K.add_from_embedded_graph(G)

    extra_coords = {
        'G': [2, 4],
        'H': [1, 5],
        'I': [5, 4],
        'J': [2, 2],
        'K': [2, 7]
    }

    for node, coord in extra_coords.items():
        K.add_node(node, coord)

    extra_edges = [
        ('G', 'A'), ('G', 'H'), ('H', 'D'), ('I', 'E'),
        ('I', 'C'), ('J', 'E'), ('K', 'D'), ('K', 'C')
    ]
    K.add_edges_from(extra_edges)

    K.add_face(['B', 'A', 'G', 'H', 'D'])
    K.add_face(['K', 'D', 'C'])

    if centered:
        K.center_coordinates(center_type)

    return K


def create_example_graph(centered=True, center_type='mean'):
    """
    Function to create an example ``EmbeddedGraph`` object. Helpful for testing. If ``centered`` is True, the coordinates are centered using the center type given by ``center_type``, either ``mean``, ``bounding_box`` or ``origin``.

    Returns:
        EmbeddedGraph: An example ``EmbeddedGraph`` object.

    """
    graph = EmbeddedGraph()

    coords = {
        'A': [1, 2],
        'B': [3, 4],
        'C': [5, 7],
        'D': [3, 6],
        'E': [4, 3],
        'F': [4, 5]
    }

    for node, coord in coords.items():
        graph.add_node(node, coord)

    edges = [
        ('A', 'B'), ('B', 'C'), ('B', 'D'),
        ('B', 'E'), ('C', 'D'), ('E', 'F')
    ]
    graph.add_edges_from(edges)

    if centered:
        graph.center_coordinates(center_type)

    return graph
