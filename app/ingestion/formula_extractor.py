"""Formula extraction utilities."""

from __future__ import annotations

import re


def extract_formulas_from_text(text: str) -> list[str]:
    """Extract potential formulas from text using heuristics.

    Args:
        text: The raw text of a page.

    Returns:
        List of strings that look like formulas.
    """
    formulas = []
    lines = text.split("\n")
    
    # Basic heuristic for formulas:
    # - Contains an equals sign
    # - Is relatively short (not a huge paragraph)
    # - Contains math operators or unit brackets like [mm], [kN]
    
    formula_pattern = re.compile(r"(=|≥|≤|>|<).*(\+|-|\*|/|\[mm\]|\[kN\]|\[t\])")
    
    for line in lines:
        line = line.strip()
        if len(line) < 5 or len(line) > 150:
            continue
            
        if "=" in line and formula_pattern.search(line):
            # It's likely a formula
            formulas.append(line)
            
    return formulas
