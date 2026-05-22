"""Interactive CLI Chat Loop."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.config import get_settings
from app.llm.ollama_client import generate_chat
from app.llm.prompts import QA_SYSTEM_PROMPT
from app.llm.router import rewrite_query
from app.retrieval.hybrid_search import format_context, search_bki

console = Console()


def chat_loop() -> None:
    """Run the interactive chat loop."""
    settings = get_settings()

    console.print(
        Panel(
            "[bold blue]BKI Hull RAG Assistant[/bold blue]\n"
            f"Model: {settings.main_model}\n"
            "Type [bold red]/exit[/bold red] to quit.",
            border_style="blue",
        )
    )

    # Simple chat history (only for the current session)
    chat_history: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            break

        with console.status("[bold cyan]Thinking...[/bold cyan]") as status:
            try:
                # 1. Rewrite Query
                status.update("[bold cyan]Rewriting query...[/bold cyan]")
                rewritten = rewrite_query(user_input)
                
                # 2. Search
                status.update(f"[bold cyan]Searching knowledge base for:[/bold cyan] {rewritten}")
                results = search_bki(rewritten, match_count=8)
                
                if not results:
                    console.print("\n[bold yellow]Assistant:[/bold yellow] I could not find any relevant information in the BKI Rules for Hull document.")
                    continue

                # 3. Format Context
                context_str = format_context(results)
                
                # 4. Prepare prompt
                system_content = QA_SYSTEM_PROMPT.format(context=context_str)
                
                messages = [
                    {"role": "system", "content": system_content},
                    *chat_history[-4:], # Keep last 4 messages for some context
                    {"role": "user", "content": user_input}
                ]
                
                # 5. Generate Answer
                status.update("[bold cyan]Generating answer...[/bold cyan]")
                answer = generate_chat(messages, model=settings.main_model)
                
                # 6. Save to history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": answer})

                # Print answer
                console.print("\n[bold yellow]Assistant:[/bold yellow]")
                console.print(Markdown(answer))

            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")


def main() -> None:
    """Entry point for the chat CLI."""
    chat_loop()


if __name__ == "__main__":
    main()
