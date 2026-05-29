"""CLI entry point for the eval harness.

Usage:
    python -m app.eval --set data/eval/test_set.json [--no-judge] [--limit N] [--no-db]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.eval.db_log import save_eval_run
from app.eval.models import EvalReport
from app.eval.runner import run_eval


console = Console()


def _print_summary(report: EvalReport) -> None:
    table = Table(
        title="Evaluation Summary",
        border_style="blue",
        show_lines=False,
    )
    table.add_column("Category", style="bold cyan")
    table.add_column("N", justify="right")
    table.add_column("Citation", justify="right")
    table.add_column("Calc", justify="right")
    table.add_column("Refusal", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Judge avg", justify="right")

    for cs in report.by_category:
        judge_avg = (
            f"{sum(cs.judge_scores) / len(cs.judge_scores):.2f}"
            if cs.judge_scores
            else "-"
        )
        table.add_row(
            cs.category,
            str(cs.total),
            f"{cs.citation_correct}/{cs.total}",
            f"{cs.calc_correct}/{cs.total}" if cs.category == "calculation" else "-",
            f"{cs.refusal_correct}/{cs.total}" if cs.category == "out_of_scope" else "-",
            f"{cs.recall_at_5}/{cs.total}",
            judge_avg,
        )

    console.print(table)

    overall_lines = [
        f"[bold]Test set:[/bold] v{report.test_set_version}, {report.test_set_size} items",
        f"[bold]Citation accuracy:[/bold] {report.citation_accuracy:.1%}",
        f"[bold]Calc correctness:[/bold] {report.calc_correctness:.1%}",
        f"[bold]Refusal accuracy:[/bold] {report.refusal_accuracy:.1%}",
        f"[bold]Recall@5:[/bold] {report.recall_at_5:.1%}",
        f"[bold]Judge avg score:[/bold] {report.judge_avg_score:.2f} / 5.00",
    ]
    console.print(Panel("\n".join(overall_lines), title="Overall", border_style="blue"))


def _save_json(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"eval_{ts}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval",
        description="Run the BKI Hull RAG evaluation harness.",
    )
    parser.add_argument(
        "--set",
        type=str,
        default="data/eval/test_set.json",
        help="Path to test set JSON.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge step (faster, deterministic only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N items (smoke test).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Do not write summary to Supabase eval_runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval/results",
        help="Directory for per-run JSON detail files.",
    )
    args = parser.parse_args()

    test_set_path = Path(args.set)
    if not test_set_path.exists():
        console.print(f"[bold red]Test set not found:[/bold red] {test_set_path}")
        console.print(
            "Generate one with: [cyan]python scripts/generate_eval_set.py[/cyan]"
        )
        return

    console.print(
        Panel(
            f"[bold blue]BKI Hull RAG Eval[/bold blue]\n"
            f"Test set: {test_set_path}\n"
            f"Judge: {'OFF' if args.no_judge else 'ON'}\n"
            f"Limit: {args.limit or 'all'}",
            border_style="blue",
        )
    )

    report = run_eval(
        test_set_path=test_set_path,
        run_judge=not args.no_judge,
        limit=args.limit,
    )

    _print_summary(report)

    json_path = _save_json(report, Path(args.output_dir))
    console.print(f"[dim]Detail saved to:[/dim] {json_path}")

    if not args.no_db:
        run_id = save_eval_run(report)
        if run_id:
            console.print(f"[dim]Logged to Supabase eval_runs:[/dim] {run_id}")


if __name__ == "__main__":
    main()
