"""Tests for app.eval.metrics — deterministic, no Ollama / Supabase."""

from app.eval.metrics import (
    calc_correctness,
    citation_accuracy,
    page_recall_at_k,
    recall_at_k,
    refusal_correct,
    refusal_detected,
)


# --- citation_accuracy ---


def test_citation_accuracy_hyphenated_page_format():
    # BKI uses "page 4-2" style — both 4 and 2 extracted as candidates
    answer = "Sources:\n- Section 4: Design Loads, page 4-2"
    assert citation_accuracy(answer, "4", 2) is True
    assert citation_accuracy(answer, "4", 4) is True


def test_citation_accuracy_exact_section_and_page():
    answer = "Sources:\n- Section 6: Shell Plating, page 123"
    assert citation_accuracy(answer, "6", 123) is True


def test_citation_accuracy_subsection_match():
    answer = "Section 2.B page 45"
    assert citation_accuracy(answer, "2.B", 45) is True


def test_citation_accuracy_page_range_contains_expected():
    answer = "See Section 6, pages 100-110 for details."
    assert citation_accuracy(answer, "6", 105) is True


def test_citation_accuracy_wrong_page():
    answer = "Section 6, page 99"
    assert citation_accuracy(answer, "6", 123) is False


def test_citation_accuracy_wrong_section():
    answer = "Section 7, page 123"
    assert citation_accuracy(answer, "6", 123) is False


def test_citation_accuracy_section_only_no_page_required():
    answer = "Section 6 covers shell plating."
    assert citation_accuracy(answer, "6", None) is True


def test_citation_accuracy_no_expected_returns_true():
    assert citation_accuracy("anything", None, None) is True


# --- calc_correctness ---


def test_calc_correctness_exact_match():
    assert calc_correctness(120.0, 120.0) is True


def test_calc_correctness_within_one_percent():
    assert calc_correctness(120.5, 120.0, tolerance_pct=1.0) is True


def test_calc_correctness_outside_tolerance():
    assert calc_correctness(125.0, 120.0, tolerance_pct=1.0) is False


def test_calc_correctness_actual_none():
    assert calc_correctness(None, 120.0) is False


def test_calc_correctness_zero_expected_uses_absolute_tolerance():
    assert calc_correctness(0.005, 0.0, tolerance_pct=1.0) is True
    assert calc_correctness(0.05, 0.0, tolerance_pct=1.0) is False


# --- refusal ---


def test_refusal_detected_indonesian_chat_phrase():
    text = "Saya belum menemukan dasar yang cukup kuat..."
    assert refusal_detected(text) is True


def test_refusal_detected_english_chat_phrase():
    text = "I could not find any relevant information."
    assert refusal_detected(text) is True


def test_refusal_detected_negative():
    text = "Section 6 says the bottom plating thickness is..."
    assert refusal_detected(text) is False


def test_refusal_correct_expected_yes_actual_yes():
    assert refusal_correct("Saya belum menemukan apapun", True) is True


def test_refusal_correct_expected_no_actual_yes_is_failure():
    assert refusal_correct("Saya belum menemukan apapun", False) is False


def test_refusal_correct_expected_yes_actual_no_is_failure():
    assert refusal_correct("Section 6 page 123", True) is False


# --- recall_at_k ---


def test_recall_at_k_present_in_top_k():
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], "c", k=5) is True


def test_recall_at_k_present_only_after_k():
    assert recall_at_k(["a", "b", "c", "d", "e", "target"], "target", k=5) is False


def test_recall_at_k_no_expected_returns_true():
    assert recall_at_k(["a", "b"], None, k=5) is True


def test_recall_at_k_empty_retrieved():
    assert recall_at_k([], "target", k=5) is False


# --- page_recall_at_k ---


def test_page_recall_at_k_present_in_top_k():
    assert page_recall_at_k([100, 200, 300, 400, 500, 999], 300, k=5) is True


def test_page_recall_at_k_present_only_after_k():
    assert page_recall_at_k([100, 200, 300, 400, 500, 999], 999, k=5) is False


def test_page_recall_at_k_no_expected_returns_true():
    assert page_recall_at_k([100, 200], None, k=5) is True


def test_page_recall_at_k_empty_retrieved():
    assert page_recall_at_k([], 100, k=5) is False
