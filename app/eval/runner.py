"""Eval harness runner.

Reads a test set JSON, runs every item through the same pipeline used by
chat (intent → rewrite → search → answer or calculate), then aggregates
metrics and returns an EvalReport.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from app.calculation.calculator import evaluate_formula
from app.calculation.intent import detect_intent
from app.calculation.variable_parser import extract_variables
from app.config import get_settings
from app.eval.judge import llm_judge_faithfulness
from app.eval.metrics import (
    calc_correctness,
    citation_accuracy,
    page_recall_at_k,
    refusal_correct,
)
from app.eval.models import (
    Category,
    CategorySummary,
    EvalItem,
    EvalReport,
    EvalResult,
)
from app.llm.ollama_client import generate_chat
from app.llm.prompts import QA_SYSTEM_PROMPT
from app.llm.router import rewrite_query
from app.retrieval.hybrid_search import format_context, search_bki


console = Console()


def load_test_set(path: str | Path) -> tuple[str, list[EvalItem]]:
    """Read the JSON file. Returns (version, items)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(raw.get("version", "unknown"))
    items = [EvalItem.model_validate(item) for item in raw.get("items", [])]
    return version, items


def _answer_for_item(item: EvalItem) -> tuple[str, list[str], list[int], float | None]:
    """Run one question through the pipeline. Returns (answer, retrieved_ids, retrieved_pages, numeric_actual)."""
    settings = get_settings()

    rewritten = rewrite_query(item.question)

    if item.category == "calculation":
        results = search_bki(rewritten, match_count=8)
        formula_chunk = next((r for r in results if r.content_type == "formula"), None)
        if formula_chunk is None:
            return ("", [r.chunk_id for r in results], [r.page_start for r in results], None)

        provided = dict(item.input_variables or {})
        if not provided:
            parsed = extract_variables(item.question, formula_chunk.content)
            provided = {k: float(v) for k, v in (parsed.get("provided") or {}).items()}

        numeric = evaluate_formula(formula_chunk.content, provided)
        if numeric is None:
            return ("", [r.chunk_id for r in results], [r.page_start for r in results], None)

        answer = (
            f"**Calculation Successful**\n\n"
            f"- Formula: `{formula_chunk.content}`\n"
            f"- Inputs: {provided}\n"
            f"- Result: `{numeric:.4f}`\n\n"
            f"Source: Section {formula_chunk.section_number}, "
            f"page {formula_chunk.page_start}"
        )
        return (answer, [r.chunk_id for r in results], [r.page_start for r in results], numeric)

    # General QA path
    results = search_bki(rewritten, match_count=8)
    if not results:
        return (
            "I could not find any relevant information in the BKI Rules for Hull document.",
            [],
            [],
            None,
        )

    context_str = format_context(results)
    system_content = QA_SYSTEM_PROMPT.format(context=context_str)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": item.question},
    ]
    try:
        answer = generate_chat(messages, model=settings.main_model)
    except Exception as e:
        return (f"[error] {e}", [r.chunk_id for r in results], [r.page_start for r in results], None)

    return (answer, [r.chunk_id for r in results], [r.page_start for r in results], None)


def _evaluate_item(item: EvalItem, run_judge: bool) -> EvalResult:
    """Run one item end-to-end and apply metrics."""
    started = time.perf_counter()
    result = EvalResult(item_id=item.id, category=item.category, question=item.question)

    try:
        answer, retrieved_ids, retrieved_pages, numeric_actual = _answer_for_item(item)
    except Exception as e:
        result.error = str(e)
        result.duration_seconds = time.perf_counter() - started
        return result

    result.answer = answer
    result.retrieved_chunk_ids = retrieved_ids
    result.retrieved_pages = retrieved_pages
    result.numeric_actual = numeric_actual
    result.numeric_expected = item.expected_numeric_result

    # Deterministic metrics, conditioned on category
    if item.category == "out_of_scope":
        result.refusal_correct = refusal_correct(answer, item.expected_refusal)
    else:
        result.citation_correct = citation_accuracy(
            answer, item.expected_section, item.expected_page
        )
        if item.category == "calculation" and item.expected_numeric_result is not None:
            result.calc_correct = calc_correctness(
                numeric_actual, item.expected_numeric_result, item.tolerance_pct
            )
        if item.expected_page is not None:
            result.recall_at_5 = page_recall_at_k(retrieved_pages, item.expected_page, k=5)

    # LLM judge for non-calc / non-OOS items
    if (
        run_judge
        and item.category not in ("out_of_scope", "calculation")
        and item.expected_answer_summary
        and answer
    ):
        verdict = llm_judge_faithfulness(
            item.question,
            answer,
            item.expected_answer_summary,
            item.judge_criteria,
        )
        result.judge_score = verdict.get("score")
        result.judge_reasoning = verdict.get("reasoning")

    result.duration_seconds = time.perf_counter() - started
    return result


def _summarize(results: Iterable[EvalResult]) -> tuple[list[CategorySummary], dict]:
    """Aggregate per-category and overall percentages."""
    by_cat: dict[Category, CategorySummary] = {}
    for r in results:
        cs = by_cat.setdefault(r.category, CategorySummary(category=r.category))
        cs.total += 1
        if r.citation_correct:
            cs.citation_correct += 1
        if r.calc_correct:
            cs.calc_correct += 1
        if r.refusal_correct:
            cs.refusal_correct += 1
        if r.recall_at_5:
            cs.recall_at_5 += 1
        if r.judge_score is not None and r.judge_score >= 0:
            cs.judge_scores.append(r.judge_score)

    all_results = list(results)
    total = len(all_results)

    def _ratio(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    citation_eligible = sum(1 for r in all_results if r.citation_correct is not None)
    calc_eligible = sum(1 for r in all_results if r.calc_correct is not None)
    refusal_eligible = sum(1 for r in all_results if r.refusal_correct is not None)
    recall_eligible = sum(1 for r in all_results if r.recall_at_5 is not None)
    judge_scores = [r.judge_score for r in all_results if r.judge_score is not None and r.judge_score >= 0]

    overall = {
        "citation_accuracy": _ratio(
            sum(1 for r in all_results if r.citation_correct), citation_eligible
        ),
        "calc_correctness": _ratio(
            sum(1 for r in all_results if r.calc_correct), calc_eligible
        ),
        "refusal_accuracy": _ratio(
            sum(1 for r in all_results if r.refusal_correct), refusal_eligible
        ),
        "recall_at_5": _ratio(
            sum(1 for r in all_results if r.recall_at_5), recall_eligible
        ),
        "judge_avg_score": (sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0,
    }
    overall["_total"] = total

    return list(by_cat.values()), overall


def run_eval(
    test_set_path: str | Path,
    run_judge: bool = True,
    limit: int | None = None,
) -> EvalReport:
    """Run the full eval and return an EvalReport."""
    version, items = load_test_set(test_set_path)
    if limit is not None:
        items = items[:limit]

    results: list[EvalResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(items))
        for item in items:
            progress.update(task, description=f"[cyan]{item.id}[/cyan] {item.category}")
            results.append(_evaluate_item(item, run_judge=run_judge))
            progress.advance(task)

    summaries, overall = _summarize(results)

    report = EvalReport(
        test_set_version=version,
        test_set_size=len(items),
        citation_accuracy=overall["citation_accuracy"],
        calc_correctness=overall["calc_correctness"],
        refusal_accuracy=overall["refusal_accuracy"],
        recall_at_5=overall["recall_at_5"],
        judge_avg_score=overall["judge_avg_score"],
        by_category=summaries,
        results=results,
        metadata={"judge_enabled": run_judge, "limit": limit},
    )
    report.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report
