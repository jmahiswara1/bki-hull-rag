"""Table extraction utilities using PyMuPDF."""

from __future__ import annotations

import fitz


def extract_table_markdown(page: fitz.Page) -> list[str]:
    """Extract tables from a PyMuPDF page and format them as Markdown.

    Args:
        page: A fitz.Page object.

    Returns:
        List of Markdown formatted table strings.
    """
    tables_md = []
    
    # find_tables is available in recent PyMuPDF versions
    if not hasattr(page, "find_tables"):
        return tables_md
        
    tables = page.find_tables()
    for tab in tables:
        data = tab.extract()
        if not data or len(data) == 0:
            continue
            
        # Convert list of lists to Markdown
        md_lines = []
        for i, row in enumerate(data):
            # Clean up newlines in cells
            clean_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
            row_str = "| " + " | ".join(clean_row) + " |"
            md_lines.append(row_str)
            
            # Add separator after header
            if i == 0:
                sep = "| " + " | ".join(["---"] * len(row)) + " |"
                md_lines.append(sep)
                
        if md_lines:
            tables_md.append("\n".join(md_lines))
            
    return tables_md
