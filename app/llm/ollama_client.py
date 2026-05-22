"""Ollama API client for generating embeddings and completions."""

from __future__ import annotations

import httpx

from app.config import get_settings


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """Generate an embedding vector for the given text using Ollama.

    Args:
        text: The input text to embed.
        model: Optional model name. Defaults to settings.embedding_model.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the Ollama API request fails.
    """
    settings = get_settings()
    model_name = model or settings.embedding_model
    url = f"{settings.ollama_base_url}/api/embeddings"

    payload = {
        "model": model_name,
        "prompt": text,
    }

    try:
        # Using a timeout of 60s as embedding generation can sometimes take a bit
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("embedding", [])
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}") from e


def generate_chat(messages: list[dict[str, str]], model: str | None = None, json_format: bool = False) -> str:
    """Generate a chat response using Ollama.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        model: Optional model name. Defaults to settings.main_model.
        json_format: If True, asks Ollama to return a JSON object.

    Returns:
        The content of the assistant's response.

    Raises:
        RuntimeError: If the Ollama API request fails.
    """
    settings = get_settings()
    model_name = model or settings.main_model
    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    
    if json_format:
        payload["format"] = "json"

    try:
        # Chat generation can take a long time, so higher timeout
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "")
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate chat response: {e}") from e
