import unittest
import numpy as np
from ect import EmbeddedComplex
from ect.validation import (
    EmbeddingValidator,
    ValidationResult,
    EdgeInteriorRule,
    FaceInteriorRule,
    SelfIntersectionRule,
    BoundaryEdgeRule,
)


class TestValidationResult(unittest.TestCase):
    def test_valid_result(self):
        """Test creating valid results"""
        result = ValidationResult.valid()
        self.assertTrue(result.is_valid)
        self.assertEqual(result.message, "")
        self.assertIsNone(result.violating_indices)

    def test_invalid_result(self):
        """Test creating invalid results"""
        result = ValidationResult.invalid("Test error", [1, 2])
        self.assertFalse(result.is_valid)
        self.assertEqual(result.message, "Test error")
        self.assertEqual(result.violating_indices, [1, 2])


class TestEdgeInteriorRule(unittest.TestCase):
    def setUp(self):
        self.rule = EdgeInteriorRule()

    def test_applies_to_dimension(self):
        """Test rule applies only to 1D cells"""
        self.assertFalse(self.rule.applies_to_dimension(0))
        self.assertTrue(self.rule.applies_to_dimension(1))
        self.assertFalse(self.rule.applies_to_dimension(2))

    def test_valid_edge(self):
        """Test valid edge passes validation"""
        cell_coords = np.array([[0, 0], [1, 0]])
        all_coords = np.array([[0, 0], [1, 0], [0.5, 0.5]])  # Third point not on edge

        result = self.rule.validate(cell_coords, all_coords, [0, 1], [0, 1, 2], dim=1)
        self.assertTrue(result.is_valid)

    def test_vertex_on_edge(self):
        """Test edge with vertex on interior fails validation"""
        cell_coords = np.array([[0, 0], [2, 0]])
        all_coords = np.array([[0, 0], [2, 0], [1, 0]])  # Third point on edge

        result = self.rule.validate(cell_coords, all_coords, [0, 1], [0, 1, 2], dim=1)
        self.assertFalse(result.is_valid)
        self.assertIn("lies on edge interior", result.message)


class TestFaceInteriorRule(unittest.TestCase):
    def setUp(self):
        self.rule = FaceInteriorRule()

    def test_applies_to_dimension(self):
        """Test rule applies only to 2D cells"""
        self.assertFalse(self.rule.applies_to_dimension(1))
        self.assertTrue(self.rule.applies_to_dimension(2))
        self.assertFalse(self.rule.applies_to_dimension(3))

    def test_valid_triangle(self):
        """Test valid triangle passes validation"""
        triangle_coords = np.array([[0, 0], [1, 0], [0.5, 1]])
        all_coords = np.array([[0, 0], [1, 0], [0.5, 1], [2, 2]])

        result = self.rule.validate(
            triangle_coords, all_coords, [0, 1, 2], [0, 1, 2, 3], dim=2
        )
        self.assertTrue(result.is_valid)

    def test_vertex_inside_triangle(self):
        """Test triangle with vertex inside fails validation"""
        triangle_coords = np.array([[0, 0], [2, 0], [1, 2]])
        all_coords = np.array([[0, 0], [2, 0], [1, 2], [1, 0.5]])  # fourth point inside

        result = self.rule.validate(
            triangle_coords, all_coords, [0, 1, 2], [0, 1, 2, 3], dim=2
        )
        self.assertFalse(result.is_valid)
        self.assertIn("inside face interior", result.message)


class TestSelfIntersectionRule(unittest.TestCase):
    def setUp(self):
        self.rule = SelfIntersectionRule()

    def test_applies_to_dimension(self):
        """Test rule applies only to 2D cells"""
        self.assertTrue(self.rule.applies_to_dimension(2))
        self.assertFalse(self.rule.applies_to_dimension(1))

    def test_valid_square(self):
        """Test valid square passes validation"""
        square_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        result = self.rule.validate(
            square_coords, None, [0, 1, 2, 3], [0, 1, 2, 3], dim=2
        )
        self.assertTrue(result.is_valid)

    def test_self_intersecting_bowtie(self):
        """Test self-intersecting bowtie fails validation"""
        bowtie_coords = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])

        result = self.rule.validate(
            bowtie_coords, None, [0, 1, 2, 3], [0, 1, 2, 3], dim=2
        )
        self.assertFalse(result.is_valid)
        self.assertIn("edges intersect", result.message)


class TestBoundaryEdgeRule(unittest.TestCase):
    def setUp(self):
        self.existing_edges = {(0, 1), (1, 2), (2, 0)}

        def edge_checker(v1_idx, v2_idx):
            return (v1_idx, v2_idx) in self.existing_edges or (
                v2_idx,
                v1_idx,
            ) in self.existing_edges

        self.rule = BoundaryEdgeRule(edge_checker=edge_checker)

    def test_applies_to_dimension(self):
        """Test rule applies only to 2D cells"""
        self.assertTrue(self.rule.applies_to_dimension(2))
        self.assertFalse(self.rule.applies_to_dimension(1))

    def test_all_edges_exist(self):
        """Test face with all boundary edges passes validation"""
        triangle_coords = np.array([[0, 0], [1, 0], [0.5, 1]])

        result = self.rule.validate(triangle_coords, None, [0, 1, 2], [0, 1, 2], dim=2)
        self.assertTrue(result.is_valid)

    def test_missing_edge(self):
        """Test face with missing boundary edge fails validation"""
        triangle_coords = np.array([[0, 0], [1, 0], [0.5, 1]])

        self.existing_edges.remove((1, 2))

        result = self.rule.validate(triangle_coords, None, [0, 1, 2], [0, 1, 2], dim=2)
        self.assertFalse(result.is_valid)
        self.assertIn("missing for face", result.message)


class TestEmbeddingValidator(unittest.TestCase):
    def setUp(self):
        def edge_checker(v1_idx, v2_idx):
            return True  # All edges exist for testing

        self.validator = EmbeddingValidator(tolerance=1e-10, edge_checker=edge_checker)

    def test_initialization(self):
        """Test validator initializes with default rules"""
        self.assertEqual(
            len(self.validator.rules), 6
        )  # 6 default rules (added dimensional rules)
        self.assertEqual(self.validator.tolerance, 1e-10)

    def test_add_rule(self):
        """Test adding custom rule"""
        initial_count = len(self.validator.rules)
        custom_rule = EdgeInteriorRule()

        self.validator.add_rule(custom_rule)
        self.assertEqual(len(self.validator.rules), initial_count + 1)

    def test_remove_rule(self):
        """Test removing rule by type"""
        initial_count = len(self.validator.rules)

        self.validator.remove_rule(EdgeInteriorRule)
        # Should have one less rule
        remaining_rules = [
            r for r in self.validator.rules if not isinstance(r, EdgeInteriorRule)
        ]
        self.assertEqual(len(remaining_rules), len(self.validator.rules))

    def test_set_tolerance(self):
        """Test setting tolerance updates all rules"""
        self.validator.set_tolerance(1e-6)

        for rule in self.validator.rules:
            self.assertEqual(rule.tolerance, 1e-6)

    def test_validate_valid_cell(self):
        """Test validating a valid cell"""
        # Valid triangle
        cell_coords = np.array([[0, 0], [1, 0], [0.5, 1]])
        all_coords = np.array([[0, 0], [1, 0], [0.5, 1], [2, 2]])

        result = self.validator.validate_cell(
            cell_coords, all_coords, [0, 1, 2], [0, 1, 2, 3], 2
        )
        self.assertTrue(result.is_valid)

    def test_validate_invalid_cell(self):
        """Test validating an invalid cell"""
        # Triangle with vertex inside
        cell_coords = np.array([[0, 0], [2, 0], [1, 2]])
        all_coords = np.array([[0, 0], [2, 0], [1, 2], [1, 0.5]])

        result = self.validator.validate_cell(
            cell_coords, all_coords, [0, 1, 2], [0, 1, 2, 3], 2
        )
        self.assertFalse(result.is_valid)
        self.assertIn("Face Interior Validation", result.message)

    def test_get_rules_for_dimension(self):
        """Test getting rules for specific dimension"""
        edge_rules = self.validator.get_rules_for_dimension(1)
        face_rules = self.validator.get_rules_for_dimension(2)

        # Should have different rules for different dimensions
        self.assertTrue(len(edge_rules) > 0)
        self.assertTrue(len(face_rules) > 0)

        # Edge rules should include EdgeInteriorRule
        rule_types = [type(rule) for rule in edge_rules]
        self.assertIn(EdgeInteriorRule, rule_types)

    def test_strict_validation(self):
        """Test enabling strict validation"""
        self.validator.enable_strict_validation()
        self.assertEqual(self.validator.tolerance, 1e-12)

    def test_permissive_validation(self):
        """Test enabling permissive validation"""
        self.validator.enable_permissive_validation()
        self.assertEqual(self.validator.tolerance, 1e-6)


class TestIntegrationWithEmbeddedComplex(unittest.TestCase):
    def test_validator_integration(self):
        """Test that EmbeddedComplex uses the new validation system"""
        complex_obj = EmbeddedComplex(validate_embedding=True)

        # Should have validator
        validator = complex_obj.get_validator()
        self.assertIsInstance(validator, EmbeddingValidator)

    def test_valid_complex_construction(self):
        """Test building valid complex with new validation"""
        complex_obj = EmbeddedComplex(validate_embedding=True)

        # Add triangle
        complex_obj.add_node("A", [0, 0])
        complex_obj.add_node("B", [1, 0])
        complex_obj.add_node("C", [0.5, 1])

        complex_obj.add_cell(["A", "B"], dim=1)
        complex_obj.add_cell(["B", "C"], dim=1)
        complex_obj.add_cell(["C", "A"], dim=1)
        complex_obj.add_cell(["A", "B", "C"], dim=2)

        # Should succeed
        self.assertEqual(len(complex_obj.cells[2]), 1)

    def test_invalid_complex_validation(self):
        """Test that invalid complex fails validation"""
        complex_obj = EmbeddedComplex(validate_embedding=True)

        # Add triangle with vertex inside
        complex_obj.add_node("A", [0, 0])
        complex_obj.add_node("B", [2, 0])
        complex_obj.add_node("C", [1, 2])
        complex_obj.add_node("D", [1, 0.5])  # Inside triangle

        complex_obj.add_cell(["A", "B"], dim=1)
        complex_obj.add_cell(["B", "C"], dim=1)
        complex_obj.add_cell(["C", "A"], dim=1)

        # Should fail
        with self.assertRaises(ValueError) as context:
            complex_obj.add_cell(["A", "B", "C"], dim=2)

        self.assertIn("Face Interior Validation", str(context.exception))

    def test_custom_validation_rules(self):
        """Test setting custom validation rules"""
        complex_obj = EmbeddedComplex(validate_embedding=True)

        # Get validator and remove strict rules
        validator = complex_obj.get_validator()
        validator.remove_rule(FaceInteriorRule)

        # Now should be able to add invalid triangle
        complex_obj.add_node("A", [0, 0])
        complex_obj.add_node("B", [2, 0])
        complex_obj.add_node("C", [1, 2])
        complex_obj.add_node("D", [1, 0.5])

        complex_obj.add_cell(["A", "B"], dim=1)
        complex_obj.add_cell(["B", "C"], dim=1)
        complex_obj.add_cell(["C", "A"], dim=1)
        complex_obj.add_cell(["A", "B", "C"], dim=2)  # Should work now

        self.assertEqual(len(complex_obj.cells[2]), 1)

    def test_tolerance_updates(self):
        """Test that tolerance updates propagate to validator"""
        complex_obj = EmbeddedComplex(validate_embedding=True)

        complex_obj.enable_embedding_validation(tol=1e-8)

        validator = complex_obj.get_validator()
        self.assertEqual(validator.tolerance, 1e-8)


if __name__ == "__main__":
    unittest.main()
