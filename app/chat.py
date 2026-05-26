"""Interactive CLI Chat Loop."""

from __future__ import annotations

import argparse
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.calculation.calculator import evaluate_formula
from app.calculation.db_calc_log import log_calculation
from app.calculation.intent import detect_intent
from app.calculation.variable_parser import extract_variables
from app.case.models import CaseContent
from app.case.parser import parse_case_text
from app.case.pdf_loader import load_case_pdf
from app.config import get_settings
from app.db.sessions import create_session, log_message, log_retrieval
from app.llm.ollama_client import generate_chat, stream_chat
from app.llm.prompts import QA_SYSTEM_PROMPT
from app.llm.router import rewrite_query
from app.retrieval.hybrid_search import (
    SearchResult,
    compute_confidence,
    format_context,
    search_bki,
)

console = Console()


REFUSAL_LOW_CONFIDENCE = (
    "Saya belum menemukan dasar yang cukup kuat di BKI Rules for Hull untuk "
    "menjawab pertanyaan ini. Coba sebutkan section, topik, atau parameter "
    "yang lebih spesifik."
)


class ChatState:
    """Mutable state for one chat_loop run."""

    def __init__(self, session_id: str | None, case: CaseContent | None) -> None:
        self.session_id = session_id
        self.case = case
        self.history: list[dict[str, str]] = []
        self.last_sources: list[SearchResult] = []


def _format_params_block(parameters: dict[str, float]) -> str:
    if not parameters:
        return ""
    return ", ".join(f"{k} = {v}" for k, v in parameters.items())


def _augment_query_with_params(user_input: str, parameters: dict[str, float]) -> str:
    if not parameters:
        return user_input
    params_str = _format_params_block(parameters)
    return f"{user_input}\n\n[Ship parameters from case PDF: {params_str}]"


def _load_and_parse_case(pdf_path: str) -> CaseContent | None:
    try:
        raw_text, page_count = load_case_pdf(pdf_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Failed to load case PDF:[/bold red] {exc}")
        return None

    if not raw_text.strip():
        console.print(
            "[bold red]Case PDF is empty after text extraction.[/bold red] "
            "If the PDF is a scan, OCR is not yet supported."
        )
        return None

    with console.status("[bold cyan]Parsing case PDF...[/bold cyan]"):
        parsed = parse_case_text(raw_text)

    return CaseContent(
        raw_text=raw_text,
        questions=parsed["questions"],
        parameters=parsed["parameters"],
        page_count=page_count,
    )


def _print_case_summary(case: CaseContent) -> None:
    lines = [f"[bold]Pages:[/bold] {case.page_count}"]
    if case.parameters:
        params_str = ", ".join(f"{k}={v}" for k, v in case.parameters.items())
        lines.append(f"[bold]Parameters:[/bold] {params_str}")
    else:
        lines.append("[bold]Parameters:[/bold] (none detected)")

    if case.questions:
        lines.append(f"[bold]Questions detected ({len(case.questions)}):[/bold]")
        for i, q in enumerate(case.questions, 1):
            lines.append(f"  {i}. {q}")
    else:
        lines.append("[bold]Questions:[/bold] (none detected)")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold blue]Case PDF Loaded[/bold blue]",
            border_style="blue",
        )
    )


def _print_help() -> None:
    table = Table(title="Slash Commands", show_lines=False, border_style="blue")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear screen and reset chat history")
    table.add_row("/history", "Show recent messages in this session")
    table.add_row("/sources", "Show sources cited in the last answer")
    table.add_row("/run", "Auto-answer all questions from the loaded case PDF")
    table.add_row("/exit, /quit", "Exit the chat")
    console.print(table)


def _print_history(state: ChatState) -> None:
    if not state.history:
        console.print("[dim]No messages yet in this session.[/dim]")
        return
    for msg in state.history[-10:]:
        role = msg["role"]
        marker = ">" if role == "user" else "•"
        color = "cyan" if role == "user" else "yellow"
        console.print(f"[bold {color}]{marker}[/bold {color}] {msg['content'][:200]}")


def _print_sources(state: ChatState) -> None:
    if not state.last_sources:
        console.print("[dim]No sources from the last answer (or no answer yet).[/dim]")
        return
    table = Table(title="Sources cited in the last answer", border_style="blue")
    table.add_column("#", justify="right")
    table.add_column("Section")
    table.add_column("Page", justify="right")
    table.add_column("Type")
    table.add_column("Similarity", justify="right")
    for i, s in enumerate(state.last_sources, 1):
        table.add_row(
            str(i),
            f"{s.section_number or '-'} {s.section_title or ''}".strip(),
            str(s.page_start),
            s.content_type,
            f"{s.similarity:.3f}",
        )
    console.print(table)


def _stream_render_markdown(messages: list[dict[str, str]], model: str) -> str:
    """Stream tokens from Ollama and render as Markdown live. Returns full text."""
    accumulated = ""
    with Live(Markdown(""), console=console, refresh_per_second=12) as live:
        try:
            for chunk in stream_chat(messages, model=model):
                accumulated += chunk
                live.update(Markdown(accumulated))
        except RuntimeError as e:
            live.update(Markdown(f"**Error:** {e}"))
    return accumulated


def _answer_one(user_input: str, state: ChatState) -> None:
    """Handle one user question end-to-end (intent → search → answer/calc)."""
    settings = get_settings()
    case = state.case

    log_message(state.session_id, "user", user_input)

    effective_query = _augment_query_with_params(
        user_input, case.parameters if case else {}
    )

    try:
        with console.status("[bold cyan]Detecting intent...[/bold cyan]"):
            intent = detect_intent(user_input)

        with console.status("[bold cyan]Rewriting query...[/bold cyan]"):
            rewritten = rewrite_query(user_input)

        if intent == "calculation":
            with console.status(f"[bold cyan]Searching formula:[/bold cyan] {rewritten}"):
                results = search_bki(rewritten, match_count=8)

            log_retrieval(
                state.session_id,
                query=user_input,
                rewritten_query=rewritten,
                intent=intent,
                retrieved_chunk_ids=[r.chunk_id for r in results],
                scores={"top": results[0].similarity if results else 0.0},
            )

            formula_chunk = next(
                (r for r in results if r.content_type == "formula"), None
            )
            if formula_chunk:
                state.last_sources = [formula_chunk]
                with console.status("[bold cyan]Extracting variables...[/bold cyan]"):
                    parsed_vars = extract_variables(effective_query, formula_chunk.content)
                missing = parsed_vars.get("missing", [])
                provided = parsed_vars.get("provided", {})

                if case and case.parameters:
                    for key, value in case.parameters.items():
                        provided.setdefault(key, value)
                        if key in missing:
                            missing.remove(key)

                if missing:
                    missing_str = ", ".join(missing)
                    msg = (
                        f"I found the formula:\n`{formula_chunk.content}`\n\n"
                        f"But I need values for the following variables to calculate it: "
                        f"**{missing_str}**. Please provide them in your next message."
                    )
                    console.print()
                    console.print(Markdown(msg))
                    state.history.append({"role": "user", "content": user_input})
                    state.history.append({"role": "assistant", "content": msg})
                    log_message(state.session_id, "assistant", msg)
                    return

                with console.status("[bold cyan]Computing...[/bold cyan]"):
                    calc_result = evaluate_formula(formula_chunk.content, provided)

                if calc_result is not None:
                    ans = "**Calculation Successful**\n\n"
                    ans += f"- **Formula:** `{formula_chunk.content}`\n"
                    ans += f"- **Inputs:** {provided}\n"
                    ans += f"- **Result:** `{calc_result:.4f}`\n\n"
                    ans += (
                        f"*Source: Section {formula_chunk.section_number}, "
                        f"Page {formula_chunk.page_start}*"
                    )

                    console.print()
                    console.print(Markdown(ans))

                    log_calculation(
                        state.session_id,
                        formula_chunk.chunk_id,
                        formula_chunk.content,
                        provided,
                        [],
                        {"value": calc_result},
                    )

                    state.history.append({"role": "user", "content": user_input})
                    state.history.append({"role": "assistant", "content": ans})
                    log_message(state.session_id, "assistant", ans)
                    return

        # General QA flow (or fallback when no formula chunk found)
        with console.status(f"[bold cyan]Searching knowledge base:[/bold cyan] {rewritten}"):
            results = search_bki(rewritten, match_count=8)

        confidence = compute_confidence(results)

        log_retrieval(
            state.session_id,
            query=user_input,
            rewritten_query=rewritten,
            intent=intent,
            retrieved_chunk_ids=[r.chunk_id for r in results],
            scores={
                "top": results[0].similarity if results else 0.0,
                "confidence": confidence,
            },
        )

        if not results or confidence < settings.confidence_threshold:
            state.last_sources = []
            console.print()
            console.print(Markdown(REFUSAL_LOW_CONFIDENCE))
            console.print(
                f"[dim](confidence={confidence:.3f}, threshold={settings.confidence_threshold})[/dim]"
            )
            state.history.append({"role": "user", "content": user_input})
            state.history.append({"role": "assistant", "content": REFUSAL_LOW_CONFIDENCE})
            log_message(state.session_id, "assistant", REFUSAL_LOW_CONFIDENCE)
            return

        state.last_sources = results

        context_str = format_context(results)
        system_content = QA_SYSTEM_PROMPT.format(context=context_str)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        if case and case.parameters:
            params_str = _format_params_block(case.parameters)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The user submitted a case PDF with these ship "
                        f"parameters: {params_str}. Use them when relevant."
                    ),
                }
            )

        messages.extend(state.history[-4:])
        messages.append({"role": "user", "content": user_input})

        console.print()
        answer = _stream_render_markdown(messages, model=settings.main_model)

        state.history.append({"role": "user", "content": user_input})
        state.history.append({"role": "assistant", "content": answer})
        log_message(state.session_id, "assistant", answer)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")


def chat_loop(case_pdf: str | None = None, session_title: str | None = None) -> None:
    """Run the interactive chat loop, optionally with a case PDF preloaded."""
    settings = get_settings()

    case: CaseContent | None = None
    if case_pdf:
        case = _load_and_parse_case(case_pdf)
        if case is None:
            return
        _print_case_summary(case)

    title = session_title or f"chat-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    session_id = create_session(title=title, interface="cli")

    state = ChatState(session_id=session_id, case=case)

    panel_lines = [
        "[bold blue]BKI Hull RAG Assistant[/bold blue]",
        f"Model: {settings.main_model}",
    ]
    if session_id:
        panel_lines.append(f"Session: [dim]{session_id[:8]}…[/dim] ({title})")
    else:
        panel_lines.append("Session: [yellow]not logged (Supabase unreachable)[/yellow]")
    if case:
        panel_lines.append(
            f"Case mode: [bold cyan]ON[/bold cyan] "
            f"({len(case.questions)} questions, {len(case.parameters)} params)"
        )
    panel_lines.append(
        "Type [bold red]/help[/bold red] for commands, [bold red]/exit[/bold red] to quit."
    )

    console.print(Panel("\n".join(panel_lines), border_style="blue"))

    while True:
        try:
            user_input = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/exit", "/quit"):
            break
        if cmd == "/help":
            _print_help()
            continue
        if cmd == "/clear":
            state.history.clear()
            state.last_sources = []
            console.clear()
            console.print("[dim]History cleared.[/dim]")
            continue
        if cmd == "/history":
            _print_history(state)
            continue
        if cmd == "/sources":
            _print_sources(state)
            continue
        if cmd == "/run" and case and case.questions:
            for i, question in enumerate(case.questions, 1):
                console.print(f"\n[bold cyan]>[/bold cyan] {question}")
                _answer_one(question, state)
            continue

        _answer_one(user_input, state)


def main() -> None:
    """Entry point for the chat CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m app.chat",
        description="BKI Hull RAG interactive chat. Optionally preload a case PDF.",
    )
    parser.add_argument(
        "--case-pdf",
        type=str,
        default=None,
        help="Path to a case PDF containing questions and ship parameters.",
    )
    parser.add_argument(
        "--session-title",
        type=str,
        default=None,
        help="Label for this chat session (default: timestamp).",
    )
    args = parser.parse_args()
    chat_loop(case_pdf=args.case_pdf, session_title=args.session_title)


if __name__ == "__main__":
    main()
