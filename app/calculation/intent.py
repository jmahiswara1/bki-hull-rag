"""Intent detection for user queries."""

from app.llm.ollama_client import generate_chat
from app.config import get_settings

def detect_intent(query: str) -> str:
    """Detect if the query is a general question or a calculation request.
    
    Returns: 
        'calculation' or 'general_qa'
    """
    settings = get_settings()
    prompt = f"""You are an intent classifier for a maritime engineering assistant.
Analyze the user's query and classify it into exactly one of these two categories:
1. "calculation": The user is asking to calculate, compute, or find a numerical value using a formula given some parameters (e.g., "calculate equipment numeral if L=120", "hitung tebal pelat", "what is the required thickness if L is 90").
2. "general_qa": The user is asking for definitions, rules, explanations, or looking up table values without asking you to compute a formula.

Query: "{query}"

Respond with ONLY ONE WORD: either "calculation" or "general_qa". Do not include any other text or explanation.
"""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = generate_chat(messages, model=settings.router_model)
        intent = response.strip().lower()
        if "calculation" in intent:
            return "calculation"
        return "general_qa"
    except Exception:
        # Fallback to general_qa on error
        return "general_qa"
