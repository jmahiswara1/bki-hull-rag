"""Script to insert previously extracted and embedded chunks from JSON into Supabase.

Useful for recovering from a failed database insertion without having to
wait for Ollama to re-generate embeddings for hundreds of pages.
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from app.ingestion.db_insert import insert_chunks, insert_document
from app.ingestion.models import Chunk, ExtractionResult

console = Console()


def load_from_json(json_path: Path) -> ExtractionResult:
    """Load an ExtractionResult from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExtractionResult.model_validate(data)


def run_insert_only(json_path: str) -> None:
    """Read a JSON file and insert its contents to Supabase."""
    path = Path(json_path)
    if not path.exists():
        console.print(f"[bold red]File not found:[/bold red] {path}")
        sys.exit(1)

    console.print(f"[bold blue]Loading data from:[/bold blue] {path}")
    
    try:
        result = load_from_json(path)
        chunks = result.chunks
        
        # Count valid embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None and len(c.embedding) > 0]
        console.print(f"  Loaded {len(chunks)} chunks.")
        console.print(f"  Found {len(valid_chunks)} chunks with valid embeddings.")

        if not valid_chunks:
            console.print("[bold red]No embeddings found in the JSON file. Nothing to insert.[/bold red]")
            sys.exit(1)

        console.print("\n[bold]Inserting to Supabase...[/bold]")
        with console.status("[bold green]Inserting to database...") as status:
            doc_uuid = insert_document(result)
            insert_chunks(doc_uuid, valid_chunks)
            
        console.print(f"  [bold green]OK[/bold green] Successfully inserted {len(valid_chunks)} chunks into Supabase!")
        
    except Exception as e:
        console.print(f"\n[bold red]Database Insert Failed:[/bold red] {e}")
        console.print("\n[yellow]Hint: If you see a 'row-level security' (RLS) error, make sure to either:[/yellow]")
        console.print("  1. Disable RLS on the tables in Supabase (SQL: `alter table documents disable row level security; alter table chunks disable row level security;`)")
        console.print("  2. OR use the 'service_role' secret key in your .env instead of the 'anon' public key.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert extraction JSON to Supabase")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file")
    args = parser.parse_args()
    run_insert_only(args.json)


if __name__ == "__main__":
    main()
