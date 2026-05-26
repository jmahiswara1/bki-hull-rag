"""Standalone script to debug retrieval without LLM generation."""

import argparse

from rich.console import Console
from rich.table import Table

from app.retrieval.hybrid_search import search_bki

console = Console()

def run_debug(query: str, match_count: int = 10):
    console.print(f"[bold cyan]Debugging Retrieval for query:[/bold cyan] '{query}'")
    
    results = search_bki(query, match_count=match_count)
    
    if not results:
        console.print("[bold red]No results found![/bold red]")
        return
        
    table = Table(title="Retrieval Results")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Page", justify="right", style="yellow")
    table.add_column("Snippet (first 50 chars)")
    
    for i, res in enumerate(results):
        snippet = res.content.replace("\n", " ")[:50] + "..."
        table.add_row(
            str(i+1),
            res.content_type,
            f"{res.similarity:.4f}",
            str(res.page_start),
            snippet
        )
        
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Debug Retrieval Mode")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("--count", type=int, default=10, help="Number of results to show")
    
    args = parser.parse_args()
    run_debug(args.query, args.count)

if __name__ == "__main__":
    main()
