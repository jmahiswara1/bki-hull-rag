"""PDF loader using PyMuPDF (fitz).

Extracts raw text content from each page of a PDF document.
This is the first step in the ingestion pipeline.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.models import PageContent
from app.ingestion.formula_extractor import extract_formulas_from_text
from app.ingestion.table_extractor import extract_table_markdown


def load_pdf(pdf_path: str | Path) -> list[PageContent]:
    """Extract text from each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file to extract.

    Returns:
        List of PageContent objects, one per page, with 1-indexed page numbers.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    pages: list[PageContent] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            tables = extract_table_markdown(page)
            formulas = extract_formulas_from_text(text)

            pages.append(
                PageContent(
                    page_number=page_num + 1,  # 1-indexed
                    text=text.strip(),
                    tables=tables,
                    formulas=formulas,
                )
            )
    finally:
        doc.close()

    return pages


def get_pdf_metadata(pdf_path: str | Path) -> dict[str, str | int]:
    """Extract basic metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with PDF metadata (title, author, page_count, etc.).
    """
    pdf_path = Path(pdf_path)

    doc = fitz.open(str(pdf_path))
    metadata = doc.metadata or {}

    result = {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "page_count": len(doc),
        "source_file": str(pdf_path.name),
    }

    doc.close()
    return result
