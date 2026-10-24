import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import httpx

from .config import settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    parts: List[Dict[str, Any]]  # raw model parts, appended back into the history unchanged
    input_tokens: int = 0
    output_tokens: int = 0
    function_calls: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    text: str = ""


def _call_one(model: str, payload: dict, retries: int) -> LLMResponse:
    """One model with exponential-backoff retries (1s, 2s, 4s...) on rate limits and server errors."""
    last = "unknown error"
    for attempt in range(retries + 1):
        try:
            r = httpx.post(
                URL.format(model=model),
                headers={"x-goog-api-key": settings.gemini_api_key},
                json=payload,
                timeout=60,
            )
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return _parse(r.json())
        except httpx.TransportError as exc:
            last = str(exc)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise LLMError(f"{model} is busy ({last})")


def call_model(contents: List[Dict[str, Any]], tools: List[Dict[str, Any]], system: str) -> LLMResponse:
    """Try the main model, then the fallback model if the main one stays busy."""
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.1},
    }
    if tools:  # omit tools entirely to force a plain-text answer
        payload["tools"] = [{"functionDeclarations": tools}]
    errors = []
    for model in dict.fromkeys([settings.gemini_model, settings.gemini_fallback_model]):
        try:
            return _call_one(model, payload, min(settings.max_llm_retries, 2))
        except LLMError as exc:
            if "busy" not in str(exc):
                raise
            errors.append(str(exc))
    raise LLMError("LLM unavailable: " + "; ".join(errors))


def _parse(data: Dict[str, Any]) -> LLMResponse:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise LLMError("Model returned no content")
    usage = data.get("usageMetadata", {})
    calls = [(p["functionCall"]["name"], p["functionCall"].get("args", {})) for p in parts if "functionCall" in p]
    text = "".join(p.get("text", "") for p in parts).strip()
    return LLMResponse(
        parts=parts,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        function_calls=calls,
        text=text,
    )
