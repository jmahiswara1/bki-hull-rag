"""Supabase client connection manager.

Provides a cached Supabase client instance configured from application settings.
"""

from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create and return a cached Supabase client.

    Returns:
        Supabase client instance configured with URL and key from settings.

    Raises:
        Exception: If connection to Supabase fails.
    """
    settings = get_settings()
    client: Client = create_client(settings.supabase_url, settings.supabase_key)
    return client
