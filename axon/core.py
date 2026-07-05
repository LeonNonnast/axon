"""Axon core — the framework-agnostic agent contract.

Axon is the "neuron" region of the Brain: the worker that thinks inside an Arkwen
workcell and acts ONLY through tools the cell provides. This core is deliberately
framework-free — LangGraph / CrewAI / a minimal loop are *adapters* (see
axon.adapters) that implement the same `Agent` contract. The cell sees only this
contract, never the framework (Arkwen Invariant 1: runtime-agnostic worker).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Tool:
    """A single capability the CELL exposes. Axon never runs it in-process;
    execution is routed back to the cell (Cell.invoke_tool) under policy."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """An injected capability package (Arkwen vocab: Toolkit). Changes what the
    agent KNOWS how to do (instructions/procedure), optionally bundling the tool
    names it expects. Injected into the system context — NOT a permission grant."""

    name: str
    instructions: str
    tools: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    """Generic, injected-from-outside model provider. Axon is provider-agnostic;
    the concrete API key is leased by Arkwen's secret broker and never seen here."""

    def complete(self, messages: list[Message], tools: list[Tool]) -> ProviderResponse: ...


@dataclass
class AgentSpec:
    prompt: str  # the mission
    context: str = ""  # materials / injected memory (cortex/hippocampus)
    tools: list[Tool] = field(default_factory=list)  # cell-provided
    skills: list[Skill] = field(default_factory=list)  # injected


@dataclass
class RunResult:
    status: str  # "completed" | "failed"
    output: str = ""
    steps: int = 0


class Agent(ABC):
    """The Axon contract: ``Agent(prompt, context, tools, skills).run()``.

    ``provider`` + ``cell`` are injected (keyword-only): the provider does the
    thinking, the cell executes tools + observes. Adapters (minimal / langgraph /
    crewai) subclass and implement ``run``.

    Axon is UNTRUSTED by design; the security boundary is the Arkwen cell, not this
    class. A tool list is a guardrail (the model won't *choose* a forbidden action),
    never a sandbox (arbitrary code can still touch the OS) — confinement is the
    cell's job.
    """

    def __init__(self, prompt: str, context: str = "", tools=None, skills=None,
                 *, provider: LLMProvider, cell: "object"):
        self.spec = AgentSpec(prompt, context, list(tools or []), list(skills or []))
        self.provider = provider
        self.cell = cell

    def system_prompt(self) -> str:
        """Compose the system context from the mission + injected skills."""
        parts = [self.spec.prompt]
        if self.spec.context:
            parts.append("## Context\n" + self.spec.context)
        for s in self.spec.skills:
            parts.append(f"## Skill: {s.name}\n{s.instructions}")
        return "\n\n".join(parts)

    @abstractmethod
    def run(self) -> RunResult:  # pragma: no cover - interface
        ...
