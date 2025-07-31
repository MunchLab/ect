import unittest
import numpy as np
from ect import EmbeddedComplex


class TestEmbeddedComplex(unittest.TestCase):
    def setUp(self):
        self.complex = EmbeddedComplex()
        coords = {
            "A": [0, 0, 0],
            "B": [1, 0, 0],
            "C": [0, 1, 0],
            "D": [0, 0, 1],
            "E": [1, 1, 1],
        }
        for node, coord in coords.items():
            self.complex.add_node(node, coord)

    def test_add_cell_validation(self):
        """Test validation for different cell dimensions"""

        self.complex.add_cell(["A"], dim=0)
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B"], dim=0)

        self.complex.add_cell(["A", "B"], dim=1)
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A"], dim=1)
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B", "C"], dim=1)

        self.complex.add_cell(["A", "B", "C"], dim=2)
        self.complex.add_cell(["A", "B", "C", "D"], dim=3)
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B"], dim=2)
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A"], dim=3)

    def test_add_cell_dimension_inference(self):
        """Test that cell dimension is correctly inferred from vertex count"""

        self.complex.add_cell(["A", "B", "C"])
        self.assertIn((0, 1, 2), self.complex.cells[2])

        self.complex.add_cell(["A", "B", "C", "D"])
        self.assertIn((0, 1, 2, 3), self.complex.cells[3])

    def test_add_cell_nonexistent_vertex(self):
        """Test error when adding cell with nonexistent vertex"""
        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B", "Z"])

    def test_add_cell_storage(self):
        """Test that cells are properly stored by dimension"""

        self.complex.add_cell(["A", "B"], dim=1)
        self.complex.add_cell(["A", "B", "C"], dim=2)
        self.complex.add_cell(["A", "B", "C", "D"], dim=3)
        self.complex.add_cell(["A", "B", "C", "D", "E"], dim=4)

        self.assertEqual(len(self.complex.cells[1]), 1)
        self.assertEqual(len(self.complex.cells[2]), 1)
        self.assertEqual(len(self.complex.cells[3]), 1)
        self.assertEqual(len(self.complex.cells[4]), 1)

        self.assertIn((0, 1), self.complex.cells[1])
        self.assertIn((0, 1, 2), self.complex.cells[2])
        self.assertIn((0, 1, 2, 3), self.complex.cells[3])
        self.assertIn((0, 1, 2, 3, 4), self.complex.cells[4])

    def test_add_face_backward_compatibility(self):
        """Test that add_face method works for backward compatibility"""
        self.complex.add_face(["A", "B", "C"])
        self.assertIn(("A", "B", "C"), self.complex.faces)

        with self.assertRaises(ValueError):
            self.complex.add_face(["A", "B"])

    def test_faces_property_backward_compatibility(self):
        """Test that faces property returns vertex names for backward compatibility"""
        self.complex.add_cell(["A", "B", "C"], dim=2)
        self.complex.add_cell(["B", "C", "D"], dim=2)

        faces = self.complex.faces
        self.assertEqual(len(faces), 2)
        self.assertIn(("A", "B", "C"), faces)
        self.assertIn(("B", "C", "D"), faces)

    def test_add_multiple_same_dimension_cells(self):
        """Test adding multiple cells of the same dimension"""
        self.complex.add_cell(["A", "B", "C"], dim=2)
        self.complex.add_cell(["B", "C", "D"], dim=2)
        self.complex.add_cell(["A", "C", "D"], dim=2)

        self.assertEqual(len(self.complex.cells[2]), 3)

        self.complex.add_cell(["A", "B", "C", "D"], dim=3)
        self.complex.add_cell(["A", "B", "C", "E"], dim=3)

        self.assertEqual(len(self.complex.cells[3]), 2)

    def test_high_dimensional_cells(self):
        """Test cells with very high dimensions"""
        for i in range(10):
            self.complex.add_node(f"V{i}", [i, i, i])

        vertices = [f"V{i}" for i in range(8)]
        self.complex.add_cell(vertices, dim=7)

        self.assertEqual(len(self.complex.cells[7]), 1)
        self.assertEqual(len(self.complex.cells[7][0]), 8)

    def test_edge_indices_integration(self):
        """Test that edge_indices property works with cell structure"""
        self.complex.add_edge("A", "B")
        self.complex.add_edge("B", "C")

        edge_indices = self.complex.edge_indices
        self.assertEqual(edge_indices.shape[0], 2)

        self.complex.add_cell(["C", "D"], dim=1)

        edge_indices = self.complex.edge_indices
        self.assertEqual(edge_indices.shape[0], 3)

    def test_empty_complex(self):
        """Test behavior with empty complex"""
        empty_complex = EmbeddedComplex()

        self.assertEqual(len(empty_complex.cells), 0)
        self.assertEqual(len(empty_complex.faces), 0)
        self.assertEqual(empty_complex.edge_indices.shape, (0, 2))

    def test_complex_with_gaps(self):
        """Test complex with gaps in cell dimensions"""
        self.complex.add_cell(["A"], dim=0)
        self.complex.add_cell(["A", "B", "C", "D"], dim=3)

        self.assertEqual(len(self.complex.cells[0]), 1)
        self.assertEqual(len(self.complex.cells.get(1, [])), 0)
        self.assertEqual(len(self.complex.cells.get(2, [])), 0)
        self.assertEqual(len(self.complex.cells[3]), 1)

    def test_cell_validation_with_check_parameter(self):
        """Test cell validation when check=True"""

        self.complex.add_edge("A", "B")
        self.complex.add_edge("B", "C")
        self.complex.add_edge("C", "A")

        self.complex.add_cell(["A", "B", "C"], dim=2, check=True)

        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B", "D"], dim=2, check=True)


if __name__ == "__main__":
    unittest.main()
