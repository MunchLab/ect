import unittest
import numpy as np
from ect import EmbeddedComplex


class TestEmbeddingValidation(unittest.TestCase):
    def setUp(self):
        self.complex = EmbeddedComplex()

    def test_valid_triangle_embedding(self):
        """Test that a valid triangle passes embedding validation"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [1, 0])
        self.complex.add_node("C", [0.5, 1])

        self.complex.add_cell(["A", "B"], dim=1, check=True)
        self.complex.add_cell(["B", "C"], dim=1, check=True)
        self.complex.add_cell(["C", "A"], dim=1, check=True)

        self.complex.add_cell(["A", "B", "C"], dim=2, check=True)

        self.assertEqual(len(self.complex.cells[2]), 1)

    def test_vertex_inside_face_violation(self):
        """Test that a vertex inside a face is properly detected"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [2, 0])
        self.complex.add_node("C", [1, 2])
        self.complex.add_node("D", [1, 0.5])

        self.complex.add_cell(["A", "B"], dim=1)
        self.complex.add_cell(["B", "C"], dim=1)
        self.complex.add_cell(["C", "A"], dim=1)

        with self.assertRaises(ValueError) as context:
            self.complex.add_cell(["A", "B", "C"], dim=2, check=True)

        self.assertIn("inside face interior", str(context.exception))

    def test_vertex_on_edge_violation(self):
        """Test that a vertex on an edge interior is properly detected"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [2, 0])
        self.complex.add_node("C", [1, 0])

        with self.assertRaises(ValueError) as context:
            self.complex.add_cell(["A", "B"], dim=1, check=True)

        self.assertIn("lies on edge interior", str(context.exception))

    def test_self_intersecting_face_violation(self):
        """Test that self-intersecting faces are detected"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [1, 1])
        self.complex.add_node("C", [1, 0])
        self.complex.add_node("D", [0, 1])

        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
        for edge in edges:
            self.complex.add_cell(list(edge), dim=1)

        with self.assertRaises(ValueError) as context:
            self.complex.add_cell(["A", "B", "C", "D"], dim=2, check=True)

        self.assertIn("edges intersect", str(context.exception))

    def test_valid_edge_embedding(self):
        """Test that valid edges pass validation"""
        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [1, 0])
        self.complex.add_node("C", [0, 1])

        self.complex.add_cell(["A", "B"], dim=1, check=True)

        self.assertEqual(len(self.complex.cells[1]), 1)

    def test_boundary_edge_validation(self):
        """Test that faces require boundary edges to exist"""
        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [1, 0])
        self.complex.add_node("C", [0.5, 1])

        self.complex.add_cell(["A", "B"], dim=1)

        with self.assertRaises(ValueError) as context:
            self.complex.add_cell(["A", "B", "C"], dim=2, check=True)

        self.assertIn("missing for face", str(context.exception))

    def test_tolerance_sensitivity(self):
        """Test that tolerance parameter affects validation"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [2, 0])
        self.complex.add_node("C", [1, 0])

        with self.assertRaises(ValueError):
            self.complex.add_cell(["A", "B"], dim=1, check=True, embedding_tol=1e-6)

        self.complex = EmbeddedComplex()
        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [2, 0])
        self.complex.add_node("C", [1, 1e-8])

        try:
            self.complex.add_cell(["A", "B"], dim=1, check=True, embedding_tol=1e-6)

        except ValueError:
            pass

    def test_skip_validation_when_check_false(self):
        """Test that validation is skipped when check=False"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [2, 0])
        self.complex.add_node("C", [1, 2])
        self.complex.add_node("D", [1, 0.5])

        self.complex.add_cell(["A", "B"], dim=1)
        self.complex.add_cell(["B", "C"], dim=1)
        self.complex.add_cell(["C", "A"], dim=1)
        self.complex.add_cell(["A", "B", "C"], dim=2, check=False)

        self.assertEqual(len(self.complex.cells[2]), 1)

    def test_empty_complex_validation(self):
        """Test that validation works with empty complex"""
        empty_complex = EmbeddedComplex()

        empty_complex.add_node("A", [0, 0])
        empty_complex.add_node("B", [1, 0])
        empty_complex.add_cell(["A", "B"], dim=1, check=True)

        self.assertEqual(len(empty_complex.cells[1]), 1)

    def test_3d_embedding_validation(self):
        """Test validation works in 3D"""

        self.complex.add_node("A", [0, 0, 0])
        self.complex.add_node("B", [1, 0, 0])
        self.complex.add_node("C", [0.5, 1, 0])
        self.complex.add_node("D", [0.5, 0.5, 0])

        self.complex.add_cell(["A", "B"], dim=1)
        self.complex.add_cell(["B", "C"], dim=1)
        self.complex.add_cell(["C", "A"], dim=1)

        with self.assertRaises(ValueError) as context:
            self.complex.add_cell(["A", "B", "C"], dim=2, check=True)

        self.assertIn("inside face interior", str(context.exception))

    def test_vertex_outside_face_allowed(self):
        """Test that vertices outside faces are properly allowed"""

        self.complex.add_node("A", [0, 0])
        self.complex.add_node("B", [1, 0])
        self.complex.add_node("C", [0.5, 1])
        self.complex.add_node("D", [2, 2])

        self.complex.add_cell(["A", "B"], dim=1)
        self.complex.add_cell(["B", "C"], dim=1)
        self.complex.add_cell(["C", "A"], dim=1)

        self.complex.add_cell(["A", "B", "C"], dim=2, check=True)
        self.assertEqual(len(self.complex.cells[2]), 1)


if __name__ == "__main__":
    unittest.main()
