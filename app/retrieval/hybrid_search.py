"""Vector search and retrieval pipeline."""

from __future__ import annotations

from app.db.supabase import get_supabase_client
from app.llm.ollama_client import get_embedding


class SearchResult:
    """Wrapper for a retrieved chunk from Supabase."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        content_type: str,
        section_number: str | None,
        section_title: str | None,
        page_start: int,
        similarity: float,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.content_type = content_type
        self.section_number = section_number
        self.section_title = section_title
        self.page_start = page_start
        self.similarity = similarity


def search_bki(
    query_text: str, 
    match_threshold: float = 0.3, 
    match_count: int = 5
) -> list[SearchResult]:
    """Perform a vector search against the BKI documents in Supabase.

    Args:
        query_text: The English rewritten query text.
        match_threshold: Minimum cosine similarity score.
        match_count: Max number of results to return.

    Returns:
        List of SearchResult objects.
    """
    client = get_supabase_client()

    # 1. Generate embedding for the query
    query_embedding = get_embedding(query_text)
    
    # 2. Metadata boost heuristics
    qt = query_text.lower()
    boost_table = "table" in qt or "tabel" in qt
    boost_formula = "formula" in qt or "calculate" in qt or "hitung" in qt or "rumus" in qt

    # 3. Call the Supabase pgvector RPC function (get more to rerank)
    response = client.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count * 3, 
        },
    ).execute()

    results = []
    if response.data:
        for item in response.data:
            c_type = item.get("content_type", "text")
            sim = item.get("similarity", 0.0)

            # Metadata boosting, clamped to 1.0 so the score stays interpretable.
            if boost_table and c_type == "table":
                sim = min(1.0, sim + 0.15)
            if boost_formula and c_type == "formula":
                sim = min(1.0, sim + 0.15)

            results.append(
                SearchResult(
                    chunk_id=item.get("id"),
                    content=item.get("content", ""),
                    content_type=c_type,
                    section_number=item.get("section_number"),
                    section_title=item.get("section_title"),
                    page_start=item.get("page_start", 0),
                    similarity=sim,
                )
            )

    # 4. Sort by boosted similarity and truncate
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:match_count]


def compute_confidence(results: list[SearchResult]) -> float:
    """Return mean similarity of the top-3 results.

    Simple, transparent confidence proxy: high when several chunks score
    well, low when only one weak chunk slipped past the threshold.
    Returns 0.0 for empty input.
    """
    if not results:
        return 0.0
    top = results[:3]
    return sum(r.similarity for r in top) / len(top)


def format_context(results: list[SearchResult]) -> str:
    """Format search results into a single context string for the LLM."""
    context_parts = []
    
    for i, res in enumerate(results):
        section_info = f"Section {res.section_number}: {res.section_title}" if res.section_number else "Unknown Section"
        header = f"[Source {i+1} | {section_info} | Page {res.page_start}]"
        context_parts.append(f"{header}\n{res.content}")
        
    return "\n\n".join(context_parts)
