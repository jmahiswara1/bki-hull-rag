# Eval Test Set

This directory holds the evaluation dataset for the BKI Hull RAG chatbot.

## Files

- `test_set.json` — the static test set, **30 items** in 6 categories (PRD section 20.1).
  Treat this as version-controlled ground truth. Do not regenerate it on every
  run, otherwise metric numbers across runs become non-comparable.
- `results/` — per-run JSON detail dumps from `python -m app.eval`.

## Categories

| Category | N | What is checked |
| --- | --- | --- |
| `definition` | 6 | citation page match + LLM judge |
| `table` | 6 | citation page match + LLM judge |
| `formula` | 6 | citation page match + LLM judge |
| `calculation` | 6 | numeric result within ±1% + citation |
| `bilingual_id` | 3 | LLM judge (Indonesian question, English source) |
| `out_of_scope` | 3 | refusal phrase detected |

## How the test set is generated (one-shot)

```bash
python scripts/generate_eval_set.py
```

This script:

1. Loads the most recent `data/processed/extraction_*.json`.
2. Samples chunks per content type (text, table, formula).
3. Asks Qwen2.5:7b to draft a paraphrased question + one-line expected answer
   summary for each chunk.
4. For calculation items, plugs default ship parameters into the formula
   (`evaluate_formula()`) to compute a deterministic ground truth.
5. Adds 3 hardcoded out-of-scope questions.
6. Writes `data/eval/test_set.json`.

**The script is a draft producer, not the source of truth.** After it runs,
open the JSON and:

- Reject items where the question is too leading (just paraphrases the chunk).
- Reject items where the expected answer is wrong or ambiguous.
- For calculation items, double-check the numeric ground truth against the rule.
- Tighten `judge_criteria` so the LLM judge has a clear bar.

Then commit. From this point on, run `python -m app.eval` only — never
re-run the generator unless you intentionally want a fresh dataset version
(bump the `version` field).

## How to add or revise items manually

Open `test_set.json` and edit. Each item must have a unique `id` and a
valid `category`. See `app/eval/models.py:EvalItem` for the full schema.

Minimal `definition`/`table`/`formula` item:

```json
{
  "id": "def-07",
  "category": "definition",
  "question": "What is X?",
  "language": "en",
  "expected_section": "6",
  "expected_page": 123,
  "expected_chunk_id": "<uuid from extraction JSON>",
  "expected_answer_summary": "X is ...",
  "judge_criteria": "Answer must mention ..."
}
```

Minimal `calculation` item:

```json
{
  "id": "calc-07",
  "category": "calculation",
  "question": "Calculate t for L=120, B=20.",
  "language": "en",
  "expected_section": "6",
  "expected_page": 130,
  "expected_chunk_id": "<formula chunk uuid>",
  "expected_numeric_result": 24.0,
  "tolerance_pct": 1.0,
  "input_variables": {"L": 120, "B": 20}
}
```

Minimal `out_of_scope` item:

```json
{
  "id": "oos-04",
  "category": "out_of_scope",
  "question": "What's the weather today?",
  "language": "en",
  "expected_refusal": true
}
```

## Running the eval

```bash
# Quick smoke (5 items, no LLM judge, no DB write)
python -m app.eval --no-judge --no-db --limit 5

# Full run, deterministic only (~5s/item)
python -m app.eval --no-judge

# Full run with LLM judge (~30s/item, total ~15 min)
python -m app.eval
```

Outputs:

- Terminal table — per-category and overall percentages.
- `data/eval/results/eval_<timestamp>.json` — full per-item detail.
- One row in Supabase `eval_runs` (unless `--no-db`).

## Supabase setup

The harness writes one row per run to a table called `eval_runs`. Run this
once in the Supabase SQL Editor (already part of `app/db/schema.sql`):

```sql
create table if not exists eval_runs (
  id uuid primary key default gen_random_uuid(),
  test_set_version text not null,
  test_set_size int not null,
  citation_accuracy float,
  calc_correctness float,
  refusal_accuracy float,
  recall_at_5 float,
  judge_avg_score float,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);
```

## Demo targets

These are reasonable thresholds for the MVP demo. Below them, dig into the
JSON detail to see which items failed and why.

- `citation_accuracy` ≥ 60%
- `calc_correctness` ≥ 70%
- `refusal_accuracy` ≥ 90%
- `judge_avg_score` ≥ 3.0 / 5
