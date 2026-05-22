"""CLI entry point for document ingestion.

Usage:
    python -m app.ingest --pdf data/raw/Rules-for-Hull-2026.pdf

Extracts text from a PDF, creates page-based chunks, and saves
the results to a JSON file in the processed data directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from app.config import get_settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.db_insert import insert_chunks, insert_document
from app.ingestion.models import ExtractionResult
from app.ingestion.pdf_loader import load_pdf
from app.llm.ollama_client import get_embedding

console = Console()


def save_extraction_result(result: ExtractionResult, output_dir: Path) -> Path:
    """Save extraction result to a timestamped JSON file.

    Args:
        result: The extraction result to save.
        output_dir: Directory to save the JSON file in.

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"extraction_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    return output_path


def run_ingestion(pdf_path: str) -> None:
    """Run the full ingestion pipeline.

    Steps:
        1. Load PDF and extract text per page.
        2. Create page-based chunks.
        3. Save extraction result to JSON.
        4. Print summary.

    Args:
        pdf_path: Path to the PDF file to ingest.
    """
    settings = get_settings()

    console.print(
        Panel(
            f"[bold blue]BKI Hull RAG — Ingestion Pipeline[/bold blue]\n"
            f"PDF: {pdf_path}",
            border_style="blue",
        )
    )

    # Step 1: Extract text from PDF
    console.print("\n[bold]Step 1:[/bold] Extracting text from PDF...")
    pages = load_pdf(pdf_path)
    console.print(f"  [green]OK[/green] Extracted [green]{len(pages)}[/green] pages")

    # Step 2: Create chunks
    console.print("\n[bold]Step 2:[/bold] Creating chunks...")
    chunks = chunk_pages(pages)
    console.print(
        f"  [green]OK[/green] Created [green]{len(chunks)}[/green] chunks "
        f"(skipped {len(pages) - len(chunks)} near-empty pages)"
    )

    # Step 3: Generate embeddings
    console.print("\n[bold]Step 3:[/bold] Generating embeddings via Ollama...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(chunks))
        
        for chunk in chunks:
            try:
                chunk.embedding = get_embedding(chunk.content)
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to embed chunk {chunk.chunk_id}: {e}")
            progress.advance(task)

    console.print(f"  [green]OK[/green] Generated embeddings for {len([c for c in chunks if c.embedding])} chunks")

    # Build extraction result
    result = ExtractionResult(
        source_file=str(Path(pdf_path).name),
        total_pages=len(pages),
        total_chunks=len(chunks),
        pages=pages,
        chunks=chunks,
    )

    # Step 4: Insert to Supabase
    console.print("\n[bold]Step 4:[/bold] Inserting to Supabase...")
    with console.status("[bold green]Inserting to database...") as status:
        try:
            doc_uuid = insert_document(result)
            insert_chunks(doc_uuid, chunks)
            console.print(f"  [green]OK[/green] Inserted {len(chunks)} chunks into Supabase")
        except Exception as e:
            console.print(f"  [red]Failed:[/red] {e}")

    # Step 5: Save to JSON
    output_dir = settings.processed_dir_resolved
    console.print(f"\n[bold]Step 5:[/bold] Saving results to {output_dir}...")
    output_path = save_extraction_result(result, output_dir)
    console.print(f"  [green]OK[/green] Saved to [cyan]{output_path}[/cyan]")

    # Summary
    console.print(
        Panel(
            f"[bold green]Ingestion Complete[/bold green]\n\n"
            f"  Pages extracted : {len(pages)}\n"
            f"  Chunks created  : {len(chunks)}\n"
            f"  Embeddings      : {len([c for c in chunks if c.embedding])}\n"
            f"  Output file     : {output_path}",
            border_style="green",
        )
    )


def main() -> None:
    """Parse CLI arguments and run ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest a BKI Rules for Hull PDF document.",
        prog="python -m app.ingest",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to ingest",
    )

    args = parser.parse_args()

    try:
        run_ingestion(args.pdf)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
