"""Deterministic metrics for the eval harness.

All functions here are pure and side-effect free, so they can be unit
tested without Ollama or Supabase.
"""

from __future__ import annotations

import re

# Phrases that indicate a refusal. We match in lowercase, case-insensitive.
# Both the English and Indonesian refusal sentences from app.chat are covered,
# plus a few common variants a human reviewer might write.
REFUSAL_MARKERS: tuple[str, ...] = (
    "saya belum menemukan",
    "saya tidak dapat menjawab",
    "tidak ditemukan di",
    "tidak dapat saya jawab",
    "i cannot find the answer",
    "i could not find any relevant",
    "not found in the document",
    "i don't have information",
)


def _extract_pages_from_answer(answer: str) -> list[int]:
    """Pull every integer that follows the literal token 'page' (case-insensitive).

    Also handles BKI-style hyphenated page refs like 'page 4-2' by extracting
    both the chapter number (4) and the sub-page number (2) as candidates.
    """
    pages: list[int] = []
    for m in re.finditer(r"page\s+(\d+)(?:-(\d+))?", answer, re.IGNORECASE):
        pages.append(int(m.group(1)))
        if m.group(2):
            pages.append(int(m.group(2)))
    return pages


def _extract_page_ranges_from_answer(answer: str) -> list[tuple[int, int]]:
    """Pull every 'page A-B' or 'pages A-B' range from the answer."""
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"pages?\s+(\d+)\s*[-–]\s*(\d+)", answer, re.IGNORECASE):
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        ranges.append((a, b))
    return ranges


def _extract_sections_from_answer(answer: str) -> list[str]:
    """Pull every 'Section X.Y' or 'Section X' token from the answer."""
    return [m.lower() for m in re.findall(r"section\s+([A-Za-z0-9.]+)", answer, re.IGNORECASE)]


def citation_accuracy(answer: str, expected_section: str | None, expected_page: int | None) -> bool:
    """Return True if the answer cites the expected section AND page.

    Page match also accepts a `page A-B` range that contains expected_page.
    Section match is a substring check (case-insensitive) so "2.B" matches
    a citation like "Section 2.B: Bottom plating".
    """
    if expected_section is None and expected_page is None:
        return True

    section_ok = True
    if expected_section:
        cited = _extract_sections_from_answer(answer)
        target = expected_section.lower()
        section_ok = any(target == s or target in s or s in target for s in cited)

    page_ok = True
    if expected_page is not None:
        pages = _extract_pages_from_answer(answer)
        ranges = _extract_page_ranges_from_answer(answer)
        page_ok = expected_page in pages or any(
            lo <= expected_page <= hi for lo, hi in ranges
        )

    return section_ok and page_ok


def calc_correctness(actual: float | None, expected: float, tolerance_pct: float = 1.0) -> bool:
    """Return True if `actual` is within tolerance_pct of `expected`.

    Tolerance is symmetric and relative. If expected is 0, falls back to an
    absolute tolerance of tolerance_pct/100.
    """
    if actual is None:
        return False
    if expected == 0:
        return abs(actual) <= tolerance_pct / 100.0
    return abs(actual - expected) / abs(expected) <= tolerance_pct / 100.0


def refusal_detected(answer: str) -> bool:
    """Return True if the answer looks like a refusal/no-info response."""
    if not answer:
        return False
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def refusal_correct(answer: str, expected_refusal: bool) -> bool:
    """expected_refusal=True -> answer must look like a refusal; False -> must not."""
    detected = refusal_detected(answer)
    return detected == expected_refusal


def recall_at_k(retrieved_ids: list[str], expected_id: str | None, k: int = 5) -> bool:
    """Return True if expected_id appears in the first k retrieved chunks."""
    if not expected_id:
        return True  # nothing to check (e.g. out_of_scope)
    return expected_id in retrieved_ids[:k]


def page_recall_at_k(
    retrieved_pages: list[int], expected_page: int | None, k: int = 5
) -> bool:
    """Return True if expected_page appears in the first k retrieved chunks' pages.

    More forgiving than recall_at_k when chunk IDs differ across systems
    (e.g. UUID in DB vs page-based ID in extraction JSON). A retrieved chunk
    on page p counts as a hit if p == expected_page.
    """
    if expected_page is None:
        return True
    return expected_page in retrieved_pages[:k]
