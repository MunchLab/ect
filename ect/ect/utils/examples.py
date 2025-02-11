from ect.embed_graph import EmbeddedGraph
from ect.embed_cw import EmbeddedCW
from typing import Type
import numpy as np


def create_example_cw(centered=True, center_type='min_max'):
    """
    Creates an example EmbeddedCW object with a simple CW complex. If centered is True, the coordinates are centered around the center type, which could be ``mean``, ``min_max`` or ``origin``.


    Returns:
        EmbeddedCW
            The example EmbeddedCW object.
    """
    G = create_example_graph(centered=False)
    K = EmbeddedCW()
    K.add_from_embedded_graph(G)
    K.add_node('G', 2, 4)
    K.add_node('H', 1, 5)
    K.add_node('I', 5, 4)
    K.add_node('J', 2, 2)
    K.add_node('K', 2, 7)
    K.add_edges_from([('G', 'A'), ('G', 'H'), ('H', 'D'), ('I', 'E'),
                     ('I', 'C'), ('J', 'E'), ('K', 'D'), ('K', 'C')])
    K.add_face(['B', 'A', 'G', 'H', 'D'])
    K.add_face(['K', 'D', 'C'])

    if centered:
        K.set_centered_coordinates(type=center_type)

    return K


def create_example_graph(centered=True, center_type='min_max'):
    """
    Function to create an example ``EmbeddedGraph`` object. Helpful for testing. If ``centered`` is True, the coordinates are centered using the center type given by ``center_type``, either ``mean`` or ``min_max``.

    Returns:
        EmbeddedGraph: An example ``EmbeddedGraph`` object.

    """
    graph = EmbeddedGraph()

    graph.add_node('A', 1, 2)
    graph.add_node('B', 3, 4)
    graph.add_node('C', 5, 7)
    graph.add_node('D', 3, 6)
    graph.add_node('E', 4, 3)
    graph.add_node('F', 4, 5)

    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')
    graph.add_edge('B', 'D')
    graph.add_edge('B', 'E')
    graph.add_edge('C', 'D')
    graph.add_edge('E', 'F')

    if centered:
        graph.set_centered_coordinates(center_type)

    return graph
