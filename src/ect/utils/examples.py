from ect.embed_complex import EmbeddedComplex
import numpy as np


def create_example_cw(centered=True, center_type="bounding_box"):
    """
    Creates an example EmbeddedComplex object with a simple CW complex. If centered is True, the coordinates are centered around the center type, which could be ``mean``, ``bounding_box`` or ``origin``.


    Returns:
        EmbeddedComplex
            The example EmbeddedComplex object.
    """
    K = create_example_graph(centered=False)

    extra_coords = {"G": [2, 4], "H": [1, 5], "I": [5, 4], "J": [2, 2], "K": [2, 7]}

    for node, coord in extra_coords.items():
        K.add_node(node, coord)

    extra_edges = [
        ("G", "A"),
        ("G", "H"),
        ("H", "D"),
        ("I", "E"),
        ("I", "C"),
        ("J", "E"),
        ("K", "D"),
        ("K", "C"),
    ]
    K.add_edges_from(extra_edges)

    K.add_face(["B", "A", "G", "H", "D"])
    K.add_face(["K", "D", "C"])

    if centered:
        K.center_coordinates(center_type)

    return K


def create_example_graph(centered=True, center_type="mean"):
    """
    Function to create an example ``EmbeddedComplex`` object with graph structure. Helpful for testing. If ``centered`` is True, the coordinates are centered using the center type given by ``center_type``, either ``mean``, ``bounding_box`` or ``origin``.

    Returns:
        EmbeddedComplex: An example ``EmbeddedComplex`` object with graph structure.

    """
    graph = EmbeddedComplex()

    coords = {
        "A": [1, 2],
        "B": [3, 4],
        "C": [5, 7],
        "D": [3, 6],
        "E": [4, 3],
        "F": [4, 5],
    }

    for node, coord in coords.items():
        graph.add_node(node, coord)

    edges = [("A", "B"), ("B", "C"), ("B", "D"), ("B", "E"), ("C", "D"), ("E", "F")]
    graph.add_edges_from(edges)

    if centered:
        graph.center_coordinates(center_type)

    return graph


def create_random_graph(n_nodes=100, n_edges=200, dim=2):
    """Creates a random graph with random node positions in [0,1]^dim

    Args:
        n_nodes: Number of nodes
        n_edges: Number of random edges to add
        dim: Dimension of embedding space
    """
    G = EmbeddedComplex()

    coords = np.random.random((n_nodes, dim))
    nodes_with_coords = [(i, coords[i]) for i in range(n_nodes)]
    G.add_nodes_from(nodes_with_coords)

    edges = set()
    while len(edges) < n_edges:
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v:
            edges.add(tuple(sorted([u, v])))

    G.add_edges_from(edges)
    return G


def create_example_3d_complex(centered=True, center_type="bounding_box"):
    """
    Creates an example 3D EmbeddedComplex with vertices, edges, 2-cells, and 3-cells.

    Args:
        centered: Whether to center the coordinates
        center_type: Method for centering ("mean", "bounding_box", or "origin")

    Returns:
        EmbeddedComplex: A complex with cells up to dimension 3
    """

    K = EmbeddedComplex()

    # Add vertices forming a cube and tetrahedron
    coords = {
        # Cube vertices
        "A": [0, 0, 0],
        "B": [1, 0, 0],
        "C": [1, 1, 0],
        "D": [0, 1, 0],
        "E": [0, 0, 1],
        "F": [1, 0, 1],
        "G": [1, 1, 1],
        "H": [0, 1, 1],
        # Additional vertices for tetrahedron
        "P": [0.5, 0.5, 0.5],  # Center point
        "Q": [2, 0, 0],  # External point
    }

    for node, coord in coords.items():
        K.add_node(node, coord)

    # Add edges (1-cells)
    edges = [
        # Cube edges
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),  # Bottom face
        ("E", "F"),
        ("F", "G"),
        ("G", "H"),
        ("H", "E"),  # Top face
        ("A", "E"),
        ("B", "F"),
        ("C", "G"),
        ("D", "H"),  # Vertical edges
        # Additional edges
        ("A", "P"),
        ("B", "P"),
        ("C", "P"),
        ("D", "P"),  # Center connections
        ("P", "Q"),
        ("A", "Q"),
        ("B", "Q"),  # External connections
    ]
    K.add_edges_from(edges)

    # Add 2-cells (faces)
    faces = [
        # Cube faces
        ["A", "B", "C", "D"],  # Bottom
        ["E", "F", "G", "H"],  # Top
        ["A", "B", "F", "E"],  # Front
        ["C", "D", "H", "G"],  # Back
        ["B", "C", "G", "F"],  # Right
        ["A", "D", "H", "E"],  # Left
        # Triangle faces
        ["A", "B", "P"],
        ["B", "C", "P"],
        ["C", "D", "P"],
        ["D", "A", "P"],
        ["A", "B", "Q"],
        ["A", "P", "Q"],
        ["B", "P", "Q"],
    ]

    for face in faces:
        K.add_cell(face, dim=2)

    # Add 3-cells (volumes)
    volumes = [
        # Tetrahedra
        ["A", "B", "C", "P"],
        ["A", "C", "D", "P"],
        ["A", "B", "P", "Q"],
        # Part of cube (pyramid with base ABCD and apex P)
        ["A", "B", "C", "D", "P"],  # 4-cell using 5 vertices
    ]

    for volume in volumes[:-1]:  # Add tetrahedra
        K.add_cell(volume, dim=3)

    # Add the 4-cell
    K.add_cell(volumes[-1], dim=4)

    if centered:
        K.center_coordinates(center_type)

    return K


def create_sparse_dimensional_complex():
    """
    Creates a complex with gaps in cell dimensions (0-cells and 3-cells only).

    Returns:
        EmbeddedComplex: A complex with only 0-cells and 3-cells
    """

    K = EmbeddedComplex()

    # Add vertices
    coords = {
        "A": [0, 0, 0],
        "B": [1, 0, 0],
        "C": [0, 1, 0],
        "D": [0, 0, 1],
        "E": [1, 1, 0],
        "F": [1, 0, 1],
    }

    for node, coord in coords.items():
        K.add_node(node, coord)

    # Add 0-cells explicitly (usually not needed as nodes are 0-cells)
    for node in ["A", "B", "C", "D"]:
        K.add_cell([node], dim=0)

    # Skip 1-cells and 2-cells - add only 3-cells
    K.add_cell(["A", "B", "C", "D"], dim=3)
    K.add_cell(["A", "B", "E", "F"], dim=3)

    return K
