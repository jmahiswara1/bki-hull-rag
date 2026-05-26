"""Pydantic models for the case PDF input mode."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseContent(BaseModel):
    """Parsed content of a user-submitted case PDF."""

    raw_text: str = Field(..., description="Raw text extracted from all pages")
    questions: list[str] = Field(
        default_factory=list,
        description="Technical questions extracted from the PDF",
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Ship principal particulars as {symbol: value}, units stripped",
    )
    page_count: int = Field(0, description="Total number of pages in the case PDF")
