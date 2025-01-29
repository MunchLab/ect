from ect.embed_graph import EmbeddedGraph
from ect.embed_cw import EmbeddedCW
import inspect
from typing import Type


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


def next_vert_name(self, s, num_verts=1):
    """Generate sequential vertex names (alphabetical or numerical)."""
    if isinstance(s, int):
        return [s + i + 1 for i in range(num_verts)] if num_verts > 1 else s + 1

    def increment_char(c):
        return 'A' if c == 'Z' else chr(ord(c) + 1)

    def increment_str(s):
        chars = list(s)
        for i in reversed(range(len(chars))):
            chars[i] = increment_char(chars[i])
            if chars[i] != 'A':
                break
            elif i == 0:
                return 'A' + ''.join(chars)
        return ''.join(chars)

    # handle multiple increments
    names = [s]
    for _ in range(num_verts):
        names.append(increment_str(names[-1]))
    return names[1:] if num_verts > 1 else names[1]
