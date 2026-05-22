"""Supabase database insertion operations for ingestion pipeline."""

from __future__ import annotations

from postgrest.exceptions import APIError

from app.db.supabase import get_supabase_client
from app.ingestion.models import Chunk, ExtractionResult


def insert_document(doc: ExtractionResult) -> str:
    """Insert or retrieve the document record in Supabase.

    Checks if a document with the same doc_key (doc_id) already exists.
    If so, returns its UUID. Otherwise, inserts it and returns the new UUID.

    Args:
        doc: The extraction result containing document metadata.

    Returns:
        The UUID of the document record in Supabase as a string.

    Raises:
        RuntimeError: If database operation fails.
    """
    client = get_supabase_client()

    # First, check if document already exists
    try:
        response = (
            client.table("documents")
            .select("id")
            .eq("doc_key", doc.doc_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
    except APIError as e:
        raise RuntimeError(f"Failed to check existing document: {e}") from e

    # Insert new document
    try:
        data = {
            "doc_key": doc.doc_id,
            "title": doc.title,
            "edition": doc.edition,
            "source_file": doc.source_file,
        }
        response = client.table("documents").insert(data).execute()
        
        if not response.data:
            raise RuntimeError("Insert successful but no data returned.")
            
        return response.data[0]["id"]
    except APIError as e:
        raise RuntimeError(f"Failed to insert document: {e}") from e


def insert_chunks(document_uuid: str, chunks: list[Chunk]) -> None:
    """Insert chunks with their embeddings into Supabase.

    Uses batch inserts to optimize network calls.
    Existing chunks with the same chunk_id/metadata are not automatically deduped
    in this simple implementation (usually you'd delete old chunks for the doc first).
    
    For MVP, we will first delete any existing chunks for this document_uuid 
    to make ingestion idempotent.

    Args:
        document_uuid: The UUID of the parent document.
        chunks: List of Chunk objects containing content and embeddings.

    Raises:
        RuntimeError: If database operation fails.
    """
    if not chunks:
        return

    client = get_supabase_client()
    
    # Optional MVP cleanup: delete old chunks for this document
    try:
        client.table("chunks").delete().eq("document_id", document_uuid).execute()
    except APIError as e:
        # Don't fail the whole process if delete fails, just log it (or ignore for MVP)
        pass

    # Batch insert configuration
    batch_size = 50
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        records = []
        for c in batch:
            # PostgreSQL cannot store null bytes (\u0000), so we must strip them
            clean_content = c.content.replace("\x00", "")
            
            record = {
                "document_id": document_uuid,
                "content": clean_content,
                "content_type": c.content_type,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "subsection": None, # Handle later if needed
                "page_start": c.page_start,
                "page_end": c.page_end,
                "metadata": {**c.metadata, "chunk_id": c.chunk_id},
                "embedding": c.embedding,
            }
            records.append(record)
            
        try:
            client.table("chunks").insert(records).execute()
        except APIError as e:
            raise RuntimeError(f"Failed to insert chunks batch {i} to {i+len(batch)}: {e}") from e
