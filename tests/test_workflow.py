"""Offline test of the dynamic workflow engine (the elio loop) — no Ollama.

A heuristic fake model returns "done" for agent nodes and "PASS ..." when it sees a
quality-gate prompt, so the engine's orchestration is exercised deterministically:
stages run in order, agents-in-a-stage run (parallel), the gate is evaluated, the
report is written THROUGH the cell, and progress events are emitted."""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from axon.localcell import LocalCell, default_toolset
from axon.workflow import parse_spec, run_workflow


class HeuristicModel(BaseChatModel):
    """No tool calls; says PASS for gate prompts, 'done' otherwise."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        blob = " ".join(getattr(m, "content", "") or "" for m in messages).lower()
        text = "PASS looks fine" if "quality gate" in blob else "done"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self):
        return "heuristic"


SPEC = {
    "goal": "demo pipeline",
    "stages": [
        {"name": "design", "agents": [{"name": "architect", "prompt": "design it"}]},
        {"name": "build", "gate": "code compiles and matches the design",
         "agents": [{"name": "impl-a", "prompt": "build part A"},
                    {"name": "impl-b", "prompt": "build part B"}]},
    ],
}


def test_parse_spec_from_yaml():
    wf = parse_spec("goal: g\nstages:\n  - name: s\n    agents:\n      - {name: a, prompt: p}\n")
    assert wf.goal == "g" and wf.stages[0].agents[0].name == "a"


def test_workflow_runs_stages_gate_and_report(tmp_path: Path):
    cell = LocalCell("demo pipeline", tmp_path / "wb")
    default_toolset(cell)
    summary = run_workflow(cell, SPEC, provider_factory=lambda: HeuristicModel())

    assert "completed" in summary
    kinds = [e.get("kind") for e in cell.events]
    # lifecycle events present and ordered
    assert kinds[0] == "workflow_started"
    assert kinds[-1] == "workflow_finished"
    assert cell.events[-1]["status"] == "completed"
    # both stages started, in order
    stage_order = [e["stage"] for e in cell.events if e.get("kind") == "stage_started"]
    assert stage_order == ["design", "build"]
    # both parallel agents of the build stage ran
    built = {e["agent"] for e in cell.events if e.get("kind") == "agent_finished" and e["stage"] == "build"}
    assert built == {"impl-a", "impl-b"}
    # the gate was evaluated and passed
    gates = [e for e in cell.events if e.get("kind") == "gate_result"]
    assert len(gates) == 1 and gates[0]["decision"] == "PASS"
    # the report was written THROUGH the cell (exists in the workbench)
    assert (cell.workbench / "workflow-report.md").is_file()


def test_workflow_gate_failure_stops_loop(tmp_path: Path):
    class FailGate(HeuristicModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            blob = " ".join(getattr(m, "content", "") or "" for m in messages).lower()
            text = "FAIL not good enough" if "quality gate" in blob else "done"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    spec = {"goal": "g", "stages": [
        {"name": "s1", "gate": "must be perfect", "agents": [{"name": "a", "prompt": "p"}]},
        {"name": "s2", "agents": [{"name": "b", "prompt": "q"}]},
    ]}
    cell = LocalCell("g", tmp_path / "wb")
    default_toolset(cell)
    summary = run_workflow(cell, spec, provider_factory=lambda: FailGate())

    assert "FAILED" in summary
    # s2 must never have started (fail-closed at the s1 gate)
    started = [e["stage"] for e in cell.events if e.get("kind") == "stage_started"]
    assert started == ["s1"]
    assert cell.events[-1]["status"] == "failed"
