"""Ollama API client for generating embeddings and completions."""

from __future__ import annotations

import json
from collections.abc import Iterator

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
        response = httpx.post(
            url,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 500 and "NaN" in e.response.text:
            # Robust workaround for Ollama bge-m3 NaN bug (FP16 overflow on specific token sequences)
            # We iteratively drop the last word until the embedding generation succeeds.
            words = text.split()
            while len(words) > 1:
                words.pop()
                payload["prompt"] = " ".join(words)
                try:
                    retry_response = httpx.post(
                        url,
                        json=payload,
                        timeout=60.0,
                    )
                    retry_response.raise_for_status()
                    return retry_response.json()["embedding"]
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 500 and "NaN" in exc.response.text:
                        continue
                    raise RuntimeError(f"Ollama API error: {exc.response.text}") from exc
        raise RuntimeError(f"Ollama API error: {e.response.text}") from e
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


def stream_chat(messages: list[dict[str, str]], model: str | None = None) -> Iterator[str]:
    """Stream a chat response from Ollama, yielding content chunks as they arrive.

    Each yielded string is a partial of the assistant's reply. The full
    reply is the concatenation of all yields.

    Raises:
        RuntimeError: If the Ollama API request fails before any chunk arrives.
    """
    settings = get_settings()
    model_name = model or settings.main_model
    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama streaming error: {e}") from e
