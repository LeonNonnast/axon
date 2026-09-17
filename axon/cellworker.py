"""Cell-worker entrypoint — run an Axon agent against a REAL Arkwen cell over stdio.

This is the axon side of the "real cell" bridge. Instead of the cooperative Python
``LocalCell`` executing tools in-process, the agent talks the same line-delimited
JSON channel (``axon.cell.Cell``) to whatever cell is on the other end of
stdin/stdout — in production, Arkwen's Go Cell-Shim (see arkwen ``internal/adapter/
axon.go``), which brokers every tool call through ``WorkcellAPI`` under policy.

    <cell> -m axon.cellworker            # the cell spawns this and drives the channel

Provider is chosen by ``AXON_PROVIDER``:

* ``echo`` (default) — a dependency-free scripted provider driving ``MinimalAgent``:
  it writes one artifact and finishes. Deterministic, so the Go bridge test is
  hermetic (needs only python3 + axon, no LangGraph/Ollama).
* ``ollama`` — a real ``ChatOllama`` model driving ``LangGraphAgent`` (needs the
  ``langgraph`` extra + a running Ollama; ``AXON_MODEL`` selects the model).

The agent NEVER executes a tool itself: every tool call is routed back to the cell,
which is the whole point of running inside Arkwen rather than the LocalCell.
"""
from __future__ import annotations

import os
import sys

from .cell import Cell
from .core import Message, ProviderResponse, Tool, ToolCall


class ScriptedProvider:
    """A deterministic, dependency-free provider for the echo bridge.

    It ignores the model entirely: on the first turn it calls ``write_artifact``
    (if the cell announced it) to produce a small proof artifact; on the next turn
    it returns a final message and stops. This exercises the full channel —
    session -> tool_call -> tool_result -> finish — through the real cell without
    needing a model runtime."""

    def __init__(self, mission: str):
        self.mission = mission
        self._n = 0

    def complete(self, messages: list[Message], tools: list[Tool]) -> ProviderResponse:
        self._n += 1
        has_write = any(t.name == "write_artifact" for t in tools)
        if self._n == 1 and has_write:
            return ProviderResponse(
                text="",
                tool_calls=[ToolCall(
                    id="call_1",
                    name="write_artifact",
                    args={
                        "path": "greeting.txt",
                        "data": "Hello from Axon, running inside the real Arkwen cell.\n",
                    },
                )],
            )
        return ProviderResponse(
            text="done — wrote greeting.txt via the Arkwen cell (mission: "
            + self.mission[:60] + ")",
            tool_calls=[],
        )


def _build_agent(session, cell):
    provider_kind = os.environ.get("AXON_PROVIDER", "echo").lower()
    if provider_kind == "ollama":
        from langchain_ollama import ChatOllama  # noqa: PLC0415 (optional dep)

        from .adapters import LangGraphAgent
        model = ChatOllama(model=os.environ.get("AXON_MODEL", "qwen2.5:7b-instruct"), temperature=0)
        return LangGraphAgent(session.mission, context=session.context, tools=session.tools,
                              skills=session.skills, provider=model, cell=cell)

    from .adapters import MinimalAgent
    return MinimalAgent(session.mission, context=session.context, tools=session.tools,
                        skills=session.skills, provider=ScriptedProvider(session.mission), cell=cell)


def main() -> int:
    # Unbuffered line protocol both ways.
    cell = Cell()
    session = cell.handshake()
    agent = _build_agent(session, cell)
    try:
        result = agent.run()
    except Exception as e:  # never leave the cell hanging — always finish (fail-closed)
        cell.finish("failed", f"axon worker error: {e}")
        return 1
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
