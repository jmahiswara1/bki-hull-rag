"""One-shot generator for data/eval/test_set.json.

Reads the latest extraction JSON from data/processed/, samples chunks per
content_type, asks Qwen2.5:7b to draft Q&A pairs grounded in each chunk,
and writes 30 items to data/eval/test_set.json.

Run once, then review and edit the JSON manually before committing it.
The eval harness consumes this file as a static dataset; do NOT regenerate
on every run, otherwise metrics become non-comparable.

Usage:
    python scripts/generate_eval_set.py [--extraction PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

# When invoked as `python scripts/generate_eval_set.py`, only `scripts/` is on
# sys.path, so `app.*` imports fail. Add the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.progress import track

from app.calculation.calculator import evaluate_formula
from app.config import get_settings
from app.llm.ollama_client import generate_chat


console = Console()


# Manually curated; these never change between runs and don't need an LLM.
OUT_OF_SCOPE_QUESTIONS = [
    {
        "id": "oos-01",
        "category": "out_of_scope",
        "question": "Siapa presiden Indonesia saat ini?",
        "language": "id",
        "expected_refusal": True,
    },
    {
        "id": "oos-02",
        "category": "out_of_scope",
        "question": "What is the recipe for nasi goreng?",
        "language": "en",
        "expected_refusal": True,
    },
    {
        "id": "oos-03",
        "category": "out_of_scope",
        "question": "Bagaimana cara mengganti oli mesin mobil?",
        "language": "id",
        "expected_refusal": True,
    },
]

# Default values for common ship parameters when generating calculation cases.
# Engineering-realistic but conservative; user can tweak after review.
DEFAULT_CALC_INPUTS: dict[str, float] = {
    "L": 120.0,
    "B": 20.0,
    "H": 10.0,
    "T": 7.5,
    "h": 4.0,
    "Cb": 0.7,
    "k": 1.0,
    "f": 1.0,
    "x": 0.5,
    "v": 14.0,
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
}


QA_PROMPT = """You are a domain expert generating evaluation questions for a BKI Rules for Hull chatbot.
Given one chunk of the document, write ONE technical question a naval architect or surveyor might ask, and a one-line expected answer summary.

Rules:
- The question must be answerable from the given chunk alone.
- DO NOT copy sentences from the chunk. Paraphrase. The user must not be able to answer just by string-matching.
- Keep the question focused and specific (one fact / one rule / one number).
- The expected_answer_summary must be ONE sentence, factual, no hedging.
- judge_criteria: ONE sentence describing what facts the answer must contain.

Respond ONLY with a valid JSON object:
{
  "question": "...",
  "expected_answer_summary": "...",
  "judge_criteria": "..."
}
"""


CALC_PROMPT = """You are generating a CALCULATION question for a BKI Rules for Hull eval set.
Given one formula chunk and a dict of variable values, write ONE question of the form
"Calculate X for L=..., B=..., ..." in English. The variables in the question must EXACTLY match the keys provided.

Respond ONLY with a valid JSON object:
{
  "question": "...",
  "expected_answer_summary": "Result of <formula> with the given inputs."
}
"""


def _load_latest_extraction(path: Path | None = None) -> dict:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    candidates = sorted(Path("data/processed").glob("extraction_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "No extraction JSON in data/processed/. Run `python -m app.ingest` first."
        )
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _filter_chunks(chunks: list[dict], content_type: str, min_chars: int = 200) -> list[dict]:
    """Pick chunks of the right type with enough content to form a question.

    Note: section_number is currently None for all chunks (ingestion does
    not yet detect headings), so we ground items by page_start instead.
    """
    return [
        c for c in chunks
        if c.get("content_type") == content_type
        and len(c.get("content", "")) >= min_chars
        and c.get("page_start")
    ]


def _qa_for_chunk(chunk: dict, category: str, language: str) -> dict | None:
    """Ask the LLM for one Q&A pair grounded in this chunk."""
    settings = get_settings()
    user_msg = (
        f"Category: {category}\n"
        f"Language: {language}\n"
        f"Chunk content:\n---\n{chunk['content'][:2000]}\n---"
    )
    if language == "id":
        user_msg += "\n\nWrite the question and answer summary in Bahasa Indonesia."

    try:
        raw = generate_chat(
            messages=[
                {"role": "system", "content": QA_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=settings.main_model,
            json_format=True,
        )
        data = json.loads(raw)
    except Exception as e:
        console.print(f"[yellow]Failed to generate Q&A for chunk {chunk.get('chunk_id')}: {e}[/yellow]")
        return None

    if not (data.get("question") and data.get("expected_answer_summary")):
        return None
    return data


def _calc_for_chunk(chunk: dict, idx: int) -> dict | None:
    """Generate a calculation Q&A grounded in a formula chunk."""
    formula_text = chunk["content"]

    # Skip narrative/reference text that masquerades as a formula chunk.
    # Real numeric formulas are short and arithmetic-heavy.
    short = formula_text.strip()
    if len(short) > 200:
        return None
    if not re.search(r"[=+\-*/·×^]", short):
        return None
    # If the chunk reads like a sentence ("...as given in...", "see Section..."),
    # the LLM-extracted "formula" is actually prose. Reject.
    if re.search(r"\b(as given|see section|refer to|table)\b", short, re.IGNORECASE):
        return None

    # Detect single-letter variables, but exclude common units.
    UNIT_TOKENS = {
        "m", "s", "g", "t", "N", "P", "K", "C",  # generic units / labels
        "mm", "cm", "kg", "kN", "MPa", "Hz",
    }
    candidates = set(re.findall(r"\b([A-Za-z])\b", short))
    candidates -= UNIT_TOKENS
    chosen = {k: v for k, v in DEFAULT_CALC_INPUTS.items() if k in candidates}
    if not chosen:
        return None

    expected = evaluate_formula(formula_text, chosen)
    if expected is None or not isinstance(expected, float):
        return None
    if abs(expected) > 1e8 or (expected != 0 and abs(expected) < 1e-6):
        # Implausible magnitude → likely the LLM-applied default vars are wrong for this formula.
        return None

    settings = get_settings()
    user_msg = (
        f"Formula chunk:\n---\n{formula_text[:1500]}\n---\n"
        f"Variable values to use: {chosen}"
    )
    try:
        raw = generate_chat(
            messages=[
                {"role": "system", "content": CALC_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=settings.main_model,
            json_format=True,
        )
        data = json.loads(raw)
    except Exception as e:
        console.print(f"[yellow]Failed to generate calc Q&A for chunk {chunk.get('chunk_id')}: {e}[/yellow]")
        return None

    question = data.get("question")
    if not question:
        return None

    return {
        "id": f"calc-{idx:02d}",
        "category": "calculation",
        "question": question,
        "language": "en",
        "expected_section": chunk.get("section_number"),
        "expected_page": chunk.get("page_start"),
        "expected_chunk_id": chunk.get("chunk_id"),
        "expected_numeric_result": float(expected),
        "tolerance_pct": 1.0,
        "input_variables": chosen,
        "expected_answer_summary": data.get(
            "expected_answer_summary",
            f"Numeric result close to {expected:.4f}.",
        ),
    }


def _build_item(chunk: dict, qa: dict, item_id: str, category: str, language: str) -> dict:
    return {
        "id": item_id,
        "category": category,
        "question": qa["question"],
        "language": language,
        "expected_section": chunk.get("section_number"),
        "expected_page": chunk.get("page_start"),
        "expected_chunk_id": chunk.get("chunk_id"),
        "expected_answer_summary": qa["expected_answer_summary"],
        "judge_criteria": qa.get("judge_criteria"),
    }


def generate(extraction_path: Path | None, out_path: Path) -> None:
    extraction = _load_latest_extraction(extraction_path)
    chunks = extraction.get("chunks", [])
    if not chunks:
        raise RuntimeError("Extraction file has no chunks.")

    console.print(f"Loaded {len(chunks)} chunks from extraction.")

    text_chunks = _filter_chunks(chunks, "text")
    table_chunks = _filter_chunks(chunks, "table", min_chars=80)
    formula_chunks = _filter_chunks(chunks, "formula", min_chars=20)

    console.print(
        f"Available — text: {len(text_chunks)}, table: {len(table_chunks)}, "
        f"formula: {len(formula_chunks)}"
    )

    rng = random.Random(42)  # reproducible sampling
    items: list[dict] = []

    # 6 definition (text)
    sample = rng.sample(text_chunks, k=min(6, len(text_chunks)))
    for i, chunk in enumerate(track(sample, description="def"), 1):
        qa = _qa_for_chunk(chunk, "definition", "en")
        if qa:
            items.append(_build_item(chunk, qa, f"def-{i:02d}", "definition", "en"))

    # 6 table
    sample = rng.sample(table_chunks, k=min(6, len(table_chunks)))
    for i, chunk in enumerate(track(sample, description="table"), 1):
        qa = _qa_for_chunk(chunk, "table", "en")
        if qa:
            items.append(_build_item(chunk, qa, f"tbl-{i:02d}", "table", "en"))

    # 6 formula (description, not calculation)
    sample = rng.sample(formula_chunks, k=min(6, len(formula_chunks)))
    for i, chunk in enumerate(track(sample, description="formula"), 1):
        qa = _qa_for_chunk(chunk, "formula", "en")
        if qa:
            items.append(_build_item(chunk, qa, f"frm-{i:02d}", "formula", "en"))

    # 6 calculation (formula chunks evaluated with default inputs)
    sample = rng.sample(formula_chunks, k=min(50, len(formula_chunks)))
    calc_built = 0
    for chunk in track(sample, description="calc"):
        if calc_built >= 6:
            break
        item = _calc_for_chunk(chunk, calc_built + 1)
        if item:
            items.append(item)
            calc_built += 1

    # 3 bilingual_id (text chunks, ID question)
    sample = rng.sample(text_chunks, k=min(3, len(text_chunks)))
    for i, chunk in enumerate(track(sample, description="bilingual"), 1):
        qa = _qa_for_chunk(chunk, "definition", "id")
        if qa:
            items.append(_build_item(chunk, qa, f"id-{i:02d}", "bilingual_id", "id"))

    # 3 out_of_scope (manually curated)
    items.extend(OUT_OF_SCOPE_QUESTIONS)

    out = {
        "version": "1",
        "created_at": str(date.today()),
        "items": items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(
        f"[bold green]Wrote {len(items)} items to {out_path}[/bold green]"
    )
    console.print(
        "[bold yellow]Now: open the file, review every item, fix anything weird, "
        "and commit.[/bold yellow]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extraction",
        type=Path,
        default=None,
        help="Path to extraction JSON. Default: latest in data/processed/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/test_set.json"),
        help="Output path for the generated test set.",
    )
    args = parser.parse_args()
    generate(args.extraction, args.out)


if __name__ == "__main__":
    main()
