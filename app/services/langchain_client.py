"""Day 15 - a minimal LangChain wrapper around the same Gemini provider.

The project already has a raw-SDK client (``model_client.py``). This module adds
a LangChain equivalent WITHOUT deleting the raw one, so the two can be compared.
Both read the API key from the same environment variable via
``model_client._get_api_key`` - the key stays server-side and is never printed.

Public surface:
    get_chat_model(...)      -> a configured ChatGoogleGenerativeAI
    invoke_model(prompt)     -> normalized plain-text answer (string)
"""

from __future__ import annotations

import logging

import model_client  # reuse existing key loading + error hierarchy
from app.config import CHAT_MODEL

logger = logging.getLogger("langchain_client")


def get_chat_model(model: str = CHAT_MODEL, temperature: float = 0.0):
    """Return a LangChain chat model bound to our Gemini key.

    Input: model name and temperature.
    Output: a ``ChatGoogleGenerativeAI`` instance.
    Raises: ``model_client.ConfigurationError`` if the API key is missing.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = model_client._get_api_key()  # raises ConfigurationError if missing
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def invoke_model(prompt: str, model: str = CHAT_MODEL) -> str:
    """Send a single prompt through LangChain and return normalized text.

    Input: a non-empty prompt string.
    Output: the model's answer as a clean string.
    Calls: ``get_chat_model`` then ``.invoke``.
    Fails: empty prompt -> ValueError; provider errors -> ProviderRequestError;
        empty output -> EmptyModelResponseError.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty, non-whitespace string.")

    llm = get_chat_model(model=model)
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - normalise into the known hierarchy
        logger.error("LangChain model request failed.")
        raise model_client.ProviderRequestError("Provider request failed.") from exc

    text = _normalize_text(getattr(response, "content", response))
    if not text.strip():
        raise model_client.EmptyModelResponseError("Provider returned an empty response.")
    return text.strip()


def _normalize_text(content) -> str:
    """Flatten LangChain message content (str or list-of-parts) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)
