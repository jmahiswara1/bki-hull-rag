"""Database logging for calculation operations."""

from app.db.supabase import get_supabase_client

def log_calculation(
    session_id: str | None, 
    chunk_id: str | None, 
    formula_text: str, 
    inputs: dict, 
    missing: list, 
    result: dict
) -> None:
    """Log a calculation event to Supabase."""
    client = get_supabase_client()
    data = {
        "formula_text": formula_text,
        "input_variables": inputs,
        "missing_variables": missing,
        "result": result
    }
    if session_id:
        data["session_id"] = session_id
    if chunk_id:
        data["formula_source_chunk_id"] = chunk_id
        
    try:
        client.table("calculation_logs").insert(data).execute()
    except Exception as e:
        print(f"Failed to log calculation: {e}")
