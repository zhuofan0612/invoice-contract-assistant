"""LLM client interface.

The model sits behind this interface for a stated reason: the sovereignty
constraint says procurement data must be able to stay inside the customer's
environment. This build calls the Anthropic API for speed, but every call goes
through `LLMClient`, so swapping in a self-hosted model means writing one new
class and changing one environment variable. Nothing in app/core imports
`anthropic`.

`StubLLMClient` is the third implementation, and it earns its place twice: it
lets the whole pipeline and the evaluation harness run with no API key, and it
acts as the deterministic baseline that the LLM's numbers get compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def get_llm_client(settings=None) -> LLMClient:
    from app.config import get_settings

    settings = settings or get_settings()
    if settings.llm_backend == "anthropic":
        from app.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
        )

    from app.llm.stub_client import StubLLMClient

    return StubLLMClient()
