"""Unit tests for app.case.parser.

These tests stub out the LLM call so they can run offline (no Ollama).
"""

from __future__ import annotations

import json
from unittest.mock import patch

from app.case.parser import _safe_float, parse_case_text


def test_safe_float_strips_unit():
    assert _safe_float("120 m") == 120.0
    assert _safe_float("7.5 mm") == 7.5
    assert _safe_float("90 meters") == 90.0


def test_safe_float_rejects_label_with_digits():
    assert _safe_float("AH36") is None
    assert _safe_float("steel grade") is None


def test_safe_float_handles_comma_decimal():
    assert _safe_float("12,5 t") == 12.5


def test_safe_float_returns_none_for_non_numeric():
    assert _safe_float("not a number") is None
    assert _safe_float("") is None


def test_parse_case_text_happy_path():
    fake_llm_response = json.dumps(
        {
            "questions": [
                "What is the required bottom plating thickness?",
                "Calculate equipment numeral.",
            ],
            "parameters": {
                "L": 120.0,
                "B": "20 m",
                "T": 7.5,
            },
        }
    )

    with patch("app.case.parser.generate_chat", return_value=fake_llm_response):
        result = parse_case_text("dummy case text")

    assert result["questions"] == [
        "What is the required bottom plating thickness?",
        "Calculate equipment numeral.",
    ]
    assert result["parameters"] == {"L": 120.0, "B": 20.0, "T": 7.5}


def test_parse_case_text_skips_unparseable_params():
    fake_llm_response = json.dumps(
        {
            "questions": [],
            "parameters": {
                "L": 120.0,
                "material": "AH36",
                "speed_range": "12 to 14",
            },
        }
    )

    with patch("app.case.parser.generate_chat", return_value=fake_llm_response):
        result = parse_case_text("dummy")

    assert "L" in result["parameters"]
    assert result["parameters"]["L"] == 120.0
    assert "material" not in result["parameters"]
    assert "speed_range" in result["parameters"]
    assert result["parameters"]["speed_range"] == 12.0


def test_parse_case_text_returns_empty_on_invalid_json():
    with patch("app.case.parser.generate_chat", return_value="not json at all"):
        result = parse_case_text("dummy")

    assert result == {"questions": [], "parameters": {}}


def test_parse_case_text_handles_missing_keys():
    with patch("app.case.parser.generate_chat", return_value=json.dumps({})):
        result = parse_case_text("dummy")

    assert result == {"questions": [], "parameters": {}}
