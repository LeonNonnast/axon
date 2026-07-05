"""LangGraph and CrewAI adapters — the same Axon contract, richer engines.

Both bind the CELL-PROVIDED tools into the framework's own tool abstraction, so
the framework's agent can only ever call tools the cell announced, and every call
is executed by the cell (``Cell.invoke_tool``) under policy — never in-process.
These are scaffolds; implement against the installed framework (optional extras in
pyproject: ``axon[langgraph]`` / ``axon[crewai]``)."""
from __future__ import annotations

from ..core import Agent, RunResult


class LangGraphAgent(Agent):
    """Bind ``spec.tools`` -> LangChain tools whose func calls
    ``self.cell.invoke_tool``, then drive
    ``langgraph.prebuilt.create_react_agent(provider, tools)``. ``run()`` streams the
    graph and mirrors messages/tool-calls via ``self.cell.emit``."""

    def run(self) -> RunResult:
        raise NotImplementedError(
            "LangGraph adapter (roadmap R1): install `axon[langgraph]`, bind cell "
            "tools -> LangChain tools that call cell.invoke_tool, run create_react_agent."
        )


class CrewAIAgent(Agent):
    """Bind ``spec.tools`` -> crewai ``BaseTool`` subclasses whose ``_run`` calls
    ``self.cell.invoke_tool``; wrap in a single-member Crew. Multi-agent crews are a
    future capability — the confined single worker is the first target."""

    def run(self) -> RunResult:
        raise NotImplementedError(
            "CrewAI adapter (roadmap R2): install `axon[crewai]`, bind cell tools -> "
            "BaseTool subclasses that call cell.invoke_tool, wrap in a Crew."
        )
