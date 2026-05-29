"""LLM-as-judge for answer faithfulness.

Single function: given a question, an assistant answer, and a one-line
expected answer summary, ask Qwen2.5:7b (configurable) to score how
faithful the answer is on a 0-5 scale and return short reasoning.

The judge is meant as a *signal*, not a hard pass/fail. The deterministic
metrics in app.eval.metrics are the authoritative ones.
"""

from __future__ import annotations

import json

from app.config import get_settings
from app.llm.ollama_client import generate_chat


JUDGE_SYSTEM_PROMPT = """You are an evaluator for a domain-specific RAG system.
Score how faithful the assistant's answer is to the expected ground truth.

Rules:
- Score 0-5 inclusive.
  - 5: Fully consistent with the expected answer, no contradictions, includes the key facts.
  - 3-4: Partially correct, missing some key facts but no contradictions.
  - 1-2: Mostly wrong or unsupported claims, but not pure nonsense.
  - 0: Hallucinated, contradictory, or completely off-topic.
- Penalize answers that invent BKI rules, page numbers, or section numbers not in the expected answer.
- Do NOT penalize differences in style, language, or verbosity if the technical content matches.
- Reasoning must be ONE short sentence.

Respond ONLY with a valid JSON object:
{"score": <int 0-5>, "reasoning": "<one short sentence>"}
"""


def llm_judge_faithfulness(
    question: str,
    answer: str,
    expected_summary: str,
    judge_criteria: str | None = None,
    model: str | None = None,
) -> dict:
    """Score the answer's faithfulness on 0-5. Returns {score, reasoning}.

    On any error (network, parse, missing keys), returns
    {score: -1, reasoning: "judge_failed: <reason>"} so the caller can
    distinguish "judge said 0" from "judge could not run".
    """
    settings = get_settings()
    model_name = model or getattr(settings, "judge_model", None) or settings.main_model

    user_msg = (
        f"Question: {question}\n\n"
        f"Expected answer summary: {expected_summary}\n\n"
        + (f"Judge criteria: {judge_criteria}\n\n" if judge_criteria else "")
        + f"Assistant answer:\n---\n{answer}\n---\n\n"
        "Score this answer."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = generate_chat(messages, model=model_name, json_format=True)
    except Exception as e:
        return {"score": -1, "reasoning": f"judge_failed: {e}"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"score": -1, "reasoning": "judge_failed: invalid_json"}

    score = data.get("score")
    if not isinstance(score, int) or not 0 <= score <= 5:
        return {"score": -1, "reasoning": "judge_failed: bad_score"}

    reasoning = str(data.get("reasoning", "")).strip() or "no_reasoning_given"
    return {"score": score, "reasoning": reasoning}
