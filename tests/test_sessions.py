"""Tests for app.db.sessions logger."""

from unittest.mock import MagicMock, patch

from app.db import sessions


def _mock_client_returning(insert_response):
    """Build a Supabase client mock whose .table().insert().execute() returns insert_response."""
    client = MagicMock()
    table = MagicMock()
    insert = MagicMock()
    execute = MagicMock(return_value=insert_response)
    insert.execute = execute
    table.insert = MagicMock(return_value=insert)
    client.table = MagicMock(return_value=table)
    return client


def test_create_session_returns_uuid_on_success():
    fake_response = MagicMock(data=[{"id": "abc-123"}])
    client = _mock_client_returning(fake_response)
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sid = sessions.create_session(title="test", interface="cli")
    assert sid == "abc-123"


def test_create_session_returns_none_on_exception():
    client = MagicMock()
    client.table = MagicMock(side_effect=RuntimeError("boom"))
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sid = sessions.create_session(title="test")
    assert sid is None


def test_create_session_returns_none_on_empty_response():
    fake_response = MagicMock(data=[])
    client = _mock_client_returning(fake_response)
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sid = sessions.create_session()
    assert sid is None


def test_log_message_noop_when_session_id_is_none():
    with patch("app.db.sessions.get_supabase_client") as mock_client:
        sessions.log_message(None, "user", "hi")
    mock_client.assert_not_called()


def test_log_message_inserts_when_session_id_present():
    client = _mock_client_returning(MagicMock(data=[{"id": "msg-1"}]))
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sessions.log_message("sid-1", "user", "hello world", language="id")
    client.table.assert_called_with("chat_messages")
    inserted = client.table.return_value.insert.call_args[0][0]
    assert inserted["session_id"] == "sid-1"
    assert inserted["role"] == "user"
    assert inserted["content"] == "hello world"
    assert inserted["language"] == "id"


def test_log_retrieval_noop_when_session_id_is_none():
    with patch("app.db.sessions.get_supabase_client") as mock_client:
        sessions.log_retrieval(None, query="x")
    mock_client.assert_not_called()


def test_log_retrieval_inserts_full_payload():
    client = _mock_client_returning(MagicMock(data=[{"id": "ret-1"}]))
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sessions.log_retrieval(
            session_id="sid-1",
            query="bottom plating",
            rewritten_query="bottom plating thickness",
            language="en",
            intent="general_qa",
            retrieved_chunk_ids=["c1", "c2"],
            scores={"top": 0.78},
        )
    client.table.assert_called_with("retrieval_logs")
    payload = client.table.return_value.insert.call_args[0][0]
    assert payload["session_id"] == "sid-1"
    assert payload["query"] == "bottom plating"
    assert payload["rewritten_query"] == "bottom plating thickness"
    assert payload["intent"] == "general_qa"
    assert payload["retrieved_chunk_ids"] == ["c1", "c2"]
    assert payload["scores"] == {"top": 0.78}


def test_log_retrieval_swallows_db_errors():
    client = MagicMock()
    client.table = MagicMock(side_effect=RuntimeError("db down"))
    with patch("app.db.sessions.get_supabase_client", return_value=client):
        sessions.log_retrieval("sid-1", "any query")  # must not raise
