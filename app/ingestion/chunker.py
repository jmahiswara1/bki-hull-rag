"""Simple page-based chunker for the ingestion pipeline.

Creates one chunk per page as a baseline strategy. Designed to be
easily replaced with section-based or token-based chunking later.
"""

from __future__ import annotations

from app.ingestion.models import Chunk, PageContent

# Default document identifier
DEFAULT_DOC_ID = "bki_rules_hull_2026"


def chunk_pages(
    pages: list[PageContent],
    doc_id: str = DEFAULT_DOC_ID,
    min_chars: int = 50,
) -> list[Chunk]:
    """Create one chunk per page from extracted page contents.

    This is the simplest chunking strategy: each page becomes one chunk.
    Pages with very little text (below min_chars) are skipped to avoid
    noise from blank or near-blank pages.

    Args:
        pages: List of extracted page contents.
        doc_id: Document identifier string.
        min_chars: Minimum character count to include a page as a chunk.
            Pages with fewer characters are skipped.

    Returns:
        List of Chunk objects ready for embedding.
    """
    chunks: list[Chunk] = []

    for page in pages:
        # Skip pages with too little content
        if len(page.text.strip()) < min_chars:
            continue

        chunk_id = f"{doc_id}_p{page.page_number:04d}"

        chunk = Chunk(
            chunk_id=chunk_id,
            content=page.text,
            content_type="text",
            doc_id=doc_id,
            page_start=page.page_number,
            page_end=page.page_number,
            metadata={
                "char_count": page.char_count,
                "source": "page_extraction",
            },
        )
        chunks.append(chunk)

    return chunks


# TODO: Add section-based chunking (Phase 2+)
# def chunk_by_sections(pages, sections, ...) -> list[Chunk]:
#     """Split pages into chunks based on detected section boundaries."""
#     ...

# TODO: Add token-based chunking with overlap (Phase 2+)
# def chunk_by_tokens(pages, max_tokens=600, overlap=100, ...) -> list[Chunk]:
#     """Split text into chunks based on token count with overlap."""
#     ...
