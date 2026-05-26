"""LLM-based parser for case PDF text.

Extracts technical questions and ship principal particulars from raw
PDF text using the light model. The output is a structured dict ready
to feed into the chat / calculation flow.
"""

from __future__ import annotations

import json
import re

from app.config import get_settings
from app.llm.ollama_client import generate_chat


PARSE_SYSTEM_PROMPT = """You are an extractor for naval architecture case documents.
Given the raw text of a user's case PDF, extract two things:

1. "questions": a list of distinct technical questions the user is asking.
   - Keep each question as a single string in its original language.
   - If a question references "the ship" or uses parameters, keep it as-is.
   - If there are no explicit questions, return an empty list.

2. "parameters": ship principal particulars or design data, as a JSON object.
   - Keys: standard symbols when obvious (L, B, H, T, Cb, V, displacement,
     frame_spacing, speed) or the original label (lowercase, snake_case).
   - Values: pure numbers (float). Strip units like "m", "mm", "knots",
     "tons", "t" before parsing.
   - If a value has a range or is non-numeric, skip it.

Respond ONLY with a valid JSON object, no markdown, in this exact shape:
{
  "questions": ["...", "..."],
  "parameters": {"L": 120.0, "B": 20.0}
}
"""


def _safe_float(raw: str) -> float | None:
    """Strip units and parse a number. Return None if not parseable.

    Matches only when the number is at the start of the string (after
    optional whitespace), so labels like "AH36" are correctly rejected
    as non-numeric while "120 m" or "12 to 14" return their leading number.
    """
    match = re.match(r"-?\d+(?:[.,]\d+)?", raw.strip())
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_case_text(raw_text: str) -> dict:
    """Parse raw case PDF text into questions + parameters.

    Args:
        raw_text: Concatenated text from the case PDF.

    Returns:
        Dict with shape:
            {
                "questions": list[str],
                "parameters": dict[str, float],
            }
        Returns empty lists/dicts on parse failure (caller can decide).
    """
    settings = get_settings()

    user_msg = (
        "Raw case PDF text:\n"
        "------\n"
        f"{raw_text}\n"
        "------\n"
        "Extract questions and parameters as specified."
    )

    messages = [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = generate_chat(
            messages=messages,
            model=settings.light_model,
            json_format=True,
        )
        data = json.loads(response)
    except Exception:
        return {"questions": [], "parameters": {}}

    questions = data.get("questions") or []
    if not isinstance(questions, list):
        questions = []
    questions = [str(q).strip() for q in questions if str(q).strip()]

    raw_params = data.get("parameters") or {}
    if not isinstance(raw_params, dict):
        raw_params = {}

    parameters: dict[str, float] = {}
    for key, value in raw_params.items():
        if isinstance(value, (int, float)):
            parameters[str(key)] = float(value)
        elif isinstance(value, str):
            parsed = _safe_float(value)
            if parsed is not None:
                parameters[str(key)] = parsed

    return {"questions": questions, "parameters": parameters}
