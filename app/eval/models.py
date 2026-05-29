"""Pydantic models for the eval harness."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "definition", "table", "formula", "calculation", "bilingual_id", "out_of_scope"
]


class EvalItem(BaseModel):
    """One ground-truth question in the test set."""

    id: str
    category: Category
    question: str
    language: str = "en"

    # Source grounding (optional for out_of_scope)
    expected_section: str | None = None
    expected_page: int | None = None
    expected_chunk_id: str | None = None

    # Category-specific ground truth
    expected_answer_summary: str | None = None
    judge_criteria: str | None = None
    expected_numeric_result: float | None = None
    tolerance_pct: float = 1.0
    expected_refusal: bool = False

    # Optional input variables for calculation items
    input_variables: dict[str, float] | None = None


class EvalResult(BaseModel):
    """Result of evaluating one item."""

    item_id: str
    category: Category
    question: str
    answer: str = ""
    error: str | None = None

    # Retrieved chunk metadata (for recall@5)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_pages: list[int] = Field(default_factory=list)

    # Deterministic metrics
    citation_correct: bool | None = None
    calc_correct: bool | None = None
    refusal_correct: bool | None = None
    recall_at_5: bool | None = None

    # LLM judge (optional)
    judge_score: int | None = None  # 0-5, -1 means judge failed
    judge_reasoning: str | None = None

    # Numeric debugging
    numeric_actual: float | None = None
    numeric_expected: float | None = None

    duration_seconds: float = 0.0


class CategorySummary(BaseModel):
    """Aggregated metrics for one category."""

    category: Category
    total: int = 0
    citation_correct: int = 0
    calc_correct: int = 0
    refusal_correct: int = 0
    recall_at_5: int = 0
    judge_scores: list[int] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full evaluation report for one run."""

    test_set_version: str
    test_set_size: int
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None

    # Top-level percentages, in [0.0, 1.0]
    citation_accuracy: float = 0.0
    calc_correctness: float = 0.0
    refusal_accuracy: float = 0.0
    recall_at_5: float = 0.0
    judge_avg_score: float = 0.0  # mean of valid scores, -1 ignored

    by_category: list[CategorySummary] = Field(default_factory=list)
    results: list[EvalResult] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
