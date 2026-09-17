"""The run_workflow cell tool must accept STRUCTURED args (goal + stages), not only
a YAML-string blob — small models emit nested JSON tool arguments reliably but choke
on a large multi-line string argument (verified: qwen2.5:7b via Ollama returns empty
content + no tool call for a YAML-blob schema, but decomposes correctly for goal +
stages). These offline tests pin that contract so the agent-driven path keeps working.
"""
from __future__ import annotations

from pathlib import Path

from axon.adapters.frameworks import _args_model
from axon.core import Tool
from axon.localcell import LocalCell, default_toolset
from axon.workflow import RUN_WORKFLOW, register_workflow_tool


def _fake_provider():
    """A gate-aware fake langchain model (no tool calls, no network)."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Fake(BaseChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            blob = " ".join(getattr(m, "content", "") or "" for m in messages).lower()
            text = "PASS ok" if "quality gate" in blob else "done"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        @property
        def _llm_type(self):
            return "fake"

    return Fake()


def test_run_workflow_tool_schema_is_structured():
    """The tool exposes goal + stages (structured), never a single YAML string arg —
    that string-blob shape is exactly what small models fail to emit."""
    props = RUN_WORKFLOW.input_schema["properties"]
    assert set(RUN_WORKFLOW.input_schema["required"]) == {"goal", "stages"}
    assert "spec" not in props  # no YAML-blob argument on the tool surface
    stage_item = props["stages"]["items"]
    agent_item = stage_item["properties"]["agents"]["items"]
    assert set(agent_item["properties"]) == {"name", "prompt"}


def test_args_model_builds_nested_pydantic():
    """The adapter must build a RECURSIVE pydantic model so `stages` is typed as a
    list of stage objects (each with typed agents), not a bare list — nested typing
    is what makes small-model tool calls faithful."""
    Model = _args_model(RUN_WORKFLOW)
    inst = Model(goal="g", stages=[{"name": "design", "agents": [{"name": "d", "prompt": "p"}]}])
    dumped = inst.model_dump()
    assert dumped["goal"] == "g"
    stage = dumped["stages"][0]
    assert stage["name"] == "design"
    assert stage["agents"][0] == {"name": "d", "prompt": "p"}


def test_registered_tool_runs_from_structured_args(tmp_path: Path):
    """Invoking run_workflow with structured goal+stages (as the model emits them)
    drives the whole loop through the cell — the agent-driven path end to end."""
    cell = LocalCell("demo", tmp_path / "wb")
    default_toolset(cell)
    register_workflow_tool(cell, _fake_provider)

    res = cell.invoke_tool(
        "tc_1",
        "run_workflow",
        {
            "goal": "tiny module",
            "stages": [
                {"name": "design", "agents": [{"name": "designer", "prompt": "design it"}]},
                {"name": "implement", "gate": "code exists",
                 "agents": [{"name": "coder", "prompt": "write it"}]},
            ],
        },
    )
    assert res["ok"] is True
    assert "workflow completed" in res["result"]
    kinds = [e.get("kind") for e in cell.events]
    assert "workflow_started" in kinds
    finished = [e for e in cell.events if e.get("kind") == "workflow_finished"]
    assert len(finished) == 1 and finished[0]["status"] == "completed"
    # both stages ran in order
    stages = [e["stage"] for e in cell.events if e.get("kind") == "stage_started"]
    assert stages == ["design", "implement"]


def test_registered_tool_still_accepts_yaml_spec(tmp_path: Path):
    """Back-compat: a direct caller may still pass a single YAML/dict `spec`."""
    cell = LocalCell("demo", tmp_path / "wb")
    default_toolset(cell)
    register_workflow_tool(cell, _fake_provider)
    res = cell.invoke_tool("tc_2", "run_workflow",
                           {"spec": "goal: g\nstages:\n  - name: s\n    agents:\n      - {name: a, prompt: p}\n"})
    assert res["ok"] is True and "completed" in res["result"]
