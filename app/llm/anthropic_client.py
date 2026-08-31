"""Anthropic API client.

Prompt caching is applied to the system prompt. The interpretation system
prompt is long, static, and re-sent for every invoice line -- 18 invoices in
the golden set means it is sent dozens of times per eval run with identical
content. Marking it `ephemeral` lets the API reuse it across calls, which cuts
both cost and time-to-first-token on everything after the first call.

Retries cover the transient failures worth retrying (429, 5xx, overloaded) with
exponential backoff, and deliberately do not cover 400s: a malformed request
will be malformed again.

Swap note: this class is the only file in the project that imports `anthropic`.
A self-hosted vLLM or Ollama backend implementing `LLMClient.complete` drops in
beside it with no changes to app/core.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Sequence

from app.llm.base import LLMResponse, ToolCall

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str | None, model: str, max_tokens: int = 2000) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The anthropic backend needs the SDK: pip install anthropic"
            ) from exc

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            # A static, repeatedly-sent system prompt is the textbook case for
            # caching. cache_control marks the breakpoint.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": list(messages),
            # Deterministic-as-possible: this is a compliance-adjacent read of a
            # contract, not creative writing.
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = list(tools)

        response = self._call_with_retries(kwargs)
        return self._to_response(response)

    def _call_with_retries(self, kwargs: dict[str, Any]):
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = (
                    status in RETRYABLE_STATUS
                    or isinstance(exc, getattr(self._anthropic, "APIConnectionError", ()))
                    or isinstance(exc, getattr(self._anthropic, "RateLimitError", ()))
                )
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise
                last_error = exc
                # Exponential backoff with jitter, so parallel eval workers do
                # not retry in lockstep.
                delay = min(2 ** (attempt - 1), 8) * (0.5 + random.random())
                log.warning(
                    "Anthropic call failed (attempt %d/%d, status=%s); retrying in %.1fs",
                    attempt, MAX_ATTEMPTS, status, delay,
                )
                time.sleep(delay)

        raise RuntimeError(f"exhausted retries: {last_error}")

    def _to_response(self, response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            } if usage else {},
        )
