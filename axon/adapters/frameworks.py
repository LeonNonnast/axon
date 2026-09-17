"""LangGraph and CrewAI adapters — the same Axon contract, richer engines.

Both bind the CELL-PROVIDED tools into the framework's own tool abstraction, so the
framework's agent can only ever call tools the cell announced, and every call is
executed by the cell (``Cell.invoke_tool``) under policy — never in-process.

``LangGraphAgent`` is implemented; ``CrewAIAgent`` is a scaffold. Framework imports
are lazy (inside ``run``) so ``import axon.adapters`` needs no optional deps."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from ..core import Agent, RunResult, Tool

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _py_type(spec: dict, name: str):
    """Map one JSON-schema node to a python/pydantic type, RECURSIVELY: an object
    with properties becomes a nested pydantic model and an array-of-objects becomes
    ``list[Model]``. Rich nested types matter for tool-calling reliability — small
    models emit a nested JSON object far more faithfully than a bare ``list``/``dict``
    (or a stringified blob)."""
    spec = spec or {}
    t = spec.get("type")
    if t == "object" and spec.get("properties"):
        return _model_from_schema(spec, name.capitalize() + "_Obj")
    if t == "array":
        item = _py_type(spec.get("items", {}), name + "_item")
        return list[item]  # type: ignore[valid-type]
    return _JSON_TO_PY.get(t, Any)


def _model_from_schema(schema: dict, model_name: str):
    from pydantic import create_model

    props = (schema or {}).get("properties", {})
    required = set((schema or {}).get("required", []))
    fields: dict[str, Any] = {}
    for name, spec in props.items():
        py = _py_type(spec or {}, f"{model_name}_{name}")
        if name in required:
            fields[name] = (py, ...)
        else:
            fields[name] = (Optional[py], None)
    return create_model(model_name, **fields)


def _args_model(tool: Tool):
    """Build a pydantic model for a tool's args from its JSON input_schema, so a
    LangChain StructuredTool can validate the model's tool-call arguments."""
    return _model_from_schema(tool.input_schema or {}, f"{tool.name}_Args")


class LangGraphAgent(Agent):
    """Drives ``langgraph.prebuilt.create_react_agent`` over cell-bound tools.

    Adapter contract: ``provider`` must be a LangChain ``BaseChatModel`` (LangGraph
    is LangChain-native). Each cell tool becomes a ``StructuredTool`` whose function
    routes to ``self.cell.invoke_tool`` — so the model can only call announced tools
    and the cell executes them under policy. Messages/tool-calls are mirrored to the
    cell via ``emit`` for Arkwen's event stream."""

    def _bind_tool(self, tool: Tool):
        from langchain_core.tools import StructuredTool

        cell = self.cell

        def _run(**kwargs):
            call_id = "tc_" + uuid.uuid4().hex[:8]
            # Executed by the CELL under policy — never in this process. The cell is
            # the sole logger of the call (invoke_tool records it), so we don't emit
            # a duplicate tool_call event here.
            res = cell.invoke_tool(call_id, tool.name, kwargs)
            if not res.get("ok", True):
                return f"error: {res.get('error', 'tool failed')}"
            return res.get("result", "")

        return StructuredTool.from_function(
            func=_run,
            name=tool.name,
            description=tool.description,
            args_schema=_args_model(tool),
        )

    def run(self) -> RunResult:
        from langchain_core.messages import HumanMessage
        from langgraph.prebuilt import create_react_agent

        lc_tools = [self._bind_tool(t) for t in self.spec.tools]
        # NOTE: on LangGraph V1 this is deprecated in favour of
        # `from langchain.agents import create_agent`; migrate before V2 (roadmap).
        agent = create_react_agent(self.provider, lc_tools, prompt=self.system_prompt())

        state = agent.invoke({"messages": [HumanMessage(content=self.spec.prompt)]})
        messages = state.get("messages", [])
        final = messages[-1] if messages else None
        output = (getattr(final, "content", "") or "") if final is not None else ""

        self.cell.emit({"kind": "message", "text": output})
        self.cell.finish("completed", output)
        return RunResult("completed", output, len(messages))


class CrewAIAgent(Agent):
    """Bind ``spec.tools`` -> crewai ``BaseTool`` subclasses whose ``_run`` calls
    ``self.cell.invoke_tool``; wrap in a single-member Crew. Multi-agent crews are a
    future capability — the confined single worker is the first target. (Scaffold.)"""

    def run(self) -> RunResult:
        raise NotImplementedError(
            "CrewAI adapter (roadmap R2): install `axon[crewai]`, bind cell tools -> "
            "BaseTool subclasses that call cell.invoke_tool, wrap in a Crew."
        )
