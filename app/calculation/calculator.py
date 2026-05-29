"""Python-based mathematical calculation engine.

Uses `asteval` to evaluate formulas safely. Unlike Python's built-in
`eval`, asteval parses an AST and rejects anything outside a whitelisted
set of mathematical operations — no attribute access, no imports, no
function calls outside its sandboxed symtable.
"""

import math
import re

from asteval import Interpreter


def normalize_formula(f_str: str) -> str:
    """Clean and normalize formula string for evaluation."""
    # Remove unit brackets like [mm], [kN/m2]
    f_str = re.sub(r"\[.*?\]", "", f_str)

    # Replace common typography
    f_str = f_str.replace("²", "**2").replace("³", "**3")
    f_str = f_str.replace("^", "**")
    # Unicode multiplication signs → "*"
    f_str = f_str.replace("·", "*").replace("×", "*").replace("⋅", "*")

    # If the formula is an equation (e.g. "t = 1.2 * L"), extract the RHS
    if "=" in f_str:
        f_str = f_str.split("=", 1)[1]

    return f_str.strip()


def evaluate_formula(formula_str: str, variables: dict) -> float | None:
    """Safely evaluate a mathematical formula with given variables.

    Returns None on any parse/evaluation error. Unsafe constructs
    (imports, attribute access, builtins beyond math) are rejected by
    asteval — they will not raise, they will simply produce no result.
    """
    interp = Interpreter(minimal=True)
    interp.symtable["pi"] = math.pi
    interp.symtable["e"] = math.e
    for name in ("sqrt", "sin", "cos", "tan", "log", "exp", "log10", "asin", "acos", "atan"):
        interp.symtable[name] = getattr(math, name)

    clean_formula = normalize_formula(formula_str)

    for k, v in variables.items():
        try:
            interp.symtable[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    try:
        result = interp(clean_formula)
    except Exception as e:
        print(f"Calculation evaluation error: {e}")
        return None

    if interp.error:
        msgs = "; ".join(err.get_error()[1] for err in interp.error)
        print(f"Calculation evaluation rejected: {msgs}")
        return None

    if result is None:
        return None

    try:
        return float(result)
    except (TypeError, ValueError):
        return None
