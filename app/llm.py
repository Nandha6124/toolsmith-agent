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


def call_model(contents: List[Dict[str, Any]], tools: List[Dict[str, Any]], system: str) -> LLMResponse:
    """One model call with exponential-backoff retries on rate limits and server errors."""
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "tools": [{"functionDeclarations": tools}],
        "generationConfig": {"temperature": 0.1},
    }
    last = "unknown error"
    for attempt in range(settings.max_llm_retries + 1):
        try:
            r = httpx.post(
                URL.format(model=settings.gemini_model),
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
        if attempt < settings.max_llm_retries:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise LLMError(f"LLM unavailable after {settings.max_llm_retries + 1} attempts ({last})")


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
