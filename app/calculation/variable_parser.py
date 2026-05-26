"""Variable parser for calculations."""

import json
from app.llm.ollama_client import generate_chat
from app.config import get_settings

def extract_variables(query: str, formula_text: str) -> dict:
    """Extract required variables from formula and provided variables from query.
    
    Returns a dict:
    {
        "required": ["L", "B", "T"],
        "provided": {"L": 120.0, "B": 20.0},
        "missing": ["T"]
    }
    """
    settings = get_settings()
    prompt = f"""You are an expert engineering assistant.
Analyze the provided formula text from the BKI Rules and the user's query.

Formula text:
{formula_text}

User Query:
{query}

Task:
1. Identify all standard mathematical variables required to calculate the main subject of the formula (e.g., L, B, T, h, k). Do not include the subject being calculated as a required variable.
2. Identify which of those variables the user has provided a numerical value for in their query. Convert the values to pure numbers (float).
3. Determine which required variables are still missing.

Respond ONLY with a valid JSON object in this exact format, with no markdown formatting or other text:
{{
    "required": ["var1", "var2"],
    "provided": {{"var1": 120.5}},
    "missing": ["var2"]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = generate_chat(messages, model=settings.main_model)
        
        # Strip markdown if any
        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        
        # Ensure standard keys exist
        if "required" not in data:
            data["required"] = []
        if "provided" not in data:
            data["provided"] = {}
        if "missing" not in data:
            data["missing"] = []
            
        return data
    except Exception as e:
        print(f"Error parsing variables: {e}")
        return {"required": [], "provided": {}, "missing": []}
