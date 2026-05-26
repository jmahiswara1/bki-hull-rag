"""Load text from a user-supplied case PDF.

Lightweight version of app.ingestion.pdf_loader: only raw text is needed
since the LLM parser handles structure extraction.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def load_case_pdf(pdf_path: str | Path) -> tuple[str, int]:
    """Extract concatenated text from all pages of the case PDF.

    Args:
        pdf_path: Path to the case PDF file.

    Returns:
        Tuple of (full_text, page_count).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a PDF.
        RuntimeError: If the PDF cannot be opened.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"Case PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open case PDF: {path}") from exc

    try:
        page_count = len(doc)
        parts: list[str] = []
        for i in range(page_count):
            text = doc[i].get_text("text").strip()
            if text:
                parts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(parts), page_count
    finally:
        doc.close()
