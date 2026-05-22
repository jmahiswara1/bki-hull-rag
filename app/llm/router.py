"""Query routing and rewriting using lightweight LLM."""

import json

from app.config import get_settings
from app.llm.ollama_client import generate_chat
from app.llm.prompts import REWRITE_SYSTEM_PROMPT


def rewrite_query(user_query: str) -> str:
    """Rewrite or translate the user's query for better retrieval.

    Uses the lightweight model (e.g., qwen2.5:3b) to convert the query
    into optimal English search terms.

    Args:
        user_query: The original query from the user (can be Indonesian).

    Returns:
        The rewritten English query. Returns original query if parsing fails.
    """
    settings = get_settings()

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    try:
        response_text = generate_chat(
            messages=messages,
            model=settings.light_model,
            json_format=True
        )
        
        # Parse JSON output
        data = json.loads(response_text)
        return data.get("query", user_query)
        
    except Exception:
        # Fallback to original query if anything fails
        return user_query
