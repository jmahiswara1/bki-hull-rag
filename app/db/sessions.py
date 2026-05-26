"""Database logging for chat sessions, messages, and retrieval audit trail.

Mirrors the pattern in app.calculation.db_calc_log: errors are caught,
printed as warnings, and never propagated. The chat flow must keep
working even if Supabase is unreachable.
"""

from __future__ import annotations

from app.db.supabase import get_supabase_client


def create_session(title: str | None = None, interface: str = "cli") -> str | None:
    """Create a new chat session row.

    Returns the session UUID as string, or None if the insert failed
    (e.g. Supabase unreachable). Callers should treat None as "logging
    disabled for this run" and continue without a session_id.
    """
    client = get_supabase_client()
    data: dict = {"interface": interface}
    if title:
        data["title"] = title

    try:
        response = client.table("chat_sessions").insert(data).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
    except Exception as e:
        print(f"[warn] failed to create chat session: {e}")
    return None


def log_message(
    session_id: str | None,
    role: str,
    content: str,
    language: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert one row into chat_messages. No-op if session_id is None."""
    if not session_id:
        return

    data: dict = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    if language:
        data["language"] = language

    client = get_supabase_client()
    try:
        client.table("chat_messages").insert(data).execute()
    except Exception as e:
        print(f"[warn] failed to log chat message: {e}")


def log_retrieval(
    session_id: str | None,
    query: str,
    rewritten_query: str | None = None,
    language: str | None = None,
    intent: str | None = None,
    retrieved_chunk_ids: list[str] | None = None,
    scores: dict | None = None,
) -> None:
    """Insert one row into retrieval_logs. No-op if session_id is None."""
    if not session_id:
        return

    data: dict = {
        "session_id": session_id,
        "query": query,
        "scores": scores or {},
    }
    if rewritten_query:
        data["rewritten_query"] = rewritten_query
    if language:
        data["language"] = language
    if intent:
        data["intent"] = intent
    if retrieved_chunk_ids:
        data["retrieved_chunk_ids"] = retrieved_chunk_ids

    client = get_supabase_client()
    try:
        client.table("retrieval_logs").insert(data).execute()
    except Exception as e:
        print(f"[warn] failed to log retrieval: {e}")
