"""Tests for asteval-backed calculator."""

from app.calculation.calculator import evaluate_formula, normalize_formula


def test_normalize_formula_strips_units():
    assert normalize_formula("t = 1.2 * L [mm]").strip() == "1.2 * L"


def test_normalize_formula_replaces_superscripts():
    assert normalize_formula("A = L²") == "L**2"
    assert normalize_formula("V = L³") == "L**3"


def test_evaluate_simple_formula():
    assert evaluate_formula("t = 1.2 * L", {"L": 100}) == 120.0


def test_evaluate_with_math_functions():
    result = evaluate_formula("y = sqrt(x)", {"x": 16})
    assert result == 4.0


def test_evaluate_with_pi():
    result = evaluate_formula("c = 2 * pi * r", {"r": 1})
    assert result is not None
    assert abs(result - 6.283185307179586) < 1e-9


def test_evaluate_rejects_import():
    """asteval must reject any attempt to call __import__ or os.system."""
    result = evaluate_formula("__import__('os').system('echo hacked')", {})
    assert result is None


def test_evaluate_rejects_attribute_access():
    """Attribute access on objects should not be possible."""
    result = evaluate_formula("(1).__class__.__bases__", {})
    assert result is None


def test_evaluate_unknown_variable_returns_none():
    result = evaluate_formula("y = a + b", {"a": 1})  # b missing
    assert result is None


def test_evaluate_invalid_syntax_returns_none():
    result = evaluate_formula("x = 1 +", {})
    assert result is None
