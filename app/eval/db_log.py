"""Persist EvalReport summaries to Supabase eval_runs table.

Pattern follows app.db.sessions: errors are caught and printed,
never propagated. Eval should never fail because of a logging issue.
"""

from __future__ import annotations

from app.db.supabase import get_supabase_client
from app.eval.models import EvalReport


def save_eval_run(report: EvalReport) -> str | None:
    """Insert one row into eval_runs. Returns the row UUID, or None on failure."""
    client = get_supabase_client()
    data = {
        "test_set_version": report.test_set_version,
        "test_set_size": report.test_set_size,
        "citation_accuracy": report.citation_accuracy,
        "calc_correctness": report.calc_correctness,
        "refusal_accuracy": report.refusal_accuracy,
        "recall_at_5": report.recall_at_5,
        "judge_avg_score": report.judge_avg_score,
        "metadata": {
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            **report.metadata,
        },
    }
    try:
        response = client.table("eval_runs").insert(data).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
    except Exception as e:
        print(f"[warn] failed to save eval_run: {e}")
    return None
