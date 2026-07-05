"""Offline tests for the cooperative reference cell's policy layer + real routing.

Uses the fake tool-calling model from test_langgraph_agent so no Ollama is needed:
the agent drives a REAL LocalCell that actually writes a file, and we assert (a) the
file lands in the workbench, (b) a path-escape attempt is refused by policy, and
(c) a secret in tool output is redacted before it hits the event log."""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from axon.adapters import LangGraphAgent
from axon.localcell import LocalCell, default_toolset

from test_langgraph_agent import FakeToolCallingModel


def _agent(mission, cell, responses):
    return LangGraphAgent(mission, tools=cell.tools(),
                          provider=FakeToolCallingModel(responses=responses), cell=cell)


def test_localcell_writes_real_file_via_agent(tmp_path: Path):
    cell = LocalCell("write greeting", tmp_path / "wb")
    default_toolset(cell)
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "write_artifact",
            "args": {"path": "greeting.txt", "data": "hi from the cell"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ]
    result = _agent("write greeting", cell, responses).run()
    assert result.status == "completed"
    written = (cell.workbench / "greeting.txt")
    assert written.is_file() and written.read_text() == "hi from the cell"


def test_localcell_refuses_path_escape(tmp_path: Path):
    cell = LocalCell("escape", tmp_path / "wb")
    default_toolset(cell)
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "write_artifact",
            "args": {"path": "../../etc/pwned", "data": "x"},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="ok"),
    ]
    _agent("escape", cell, responses).run()
    # the escaping write must have been refused by the cell's policy
    assert not (tmp_path / "etc" / "pwned").exists()
    assert any(e.get("kind") == "tool_result" and e.get("ok") is False for e in cell.events)


def test_localcell_redacts_secret_in_output(tmp_path: Path):
    secret = "sk-live-SUPERSECRET"
    cell = LocalCell("leak", tmp_path / "wb", secrets={"MODEL_API_KEY": secret})
    default_toolset(cell)
    # tool writes the secret into a file, then reads it back — the read result would
    # carry the secret, but the cell redacts tool output before logging/returning.
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "write_artifact", "args": {"path": "s.txt", "data": secret},
            "id": "c1", "type": "tool_call",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "read_material", "args": {"path": "s.txt"},
            "id": "c2", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ]
    _agent("leak", cell, responses).run()
    # the raw secret must appear on NO event in the cell's log
    blob = repr(cell.events)
    assert secret not in blob
