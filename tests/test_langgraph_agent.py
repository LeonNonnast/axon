"""Offline proof that the LangGraph adapter routes tool calls THROUGH the cell.

No network / no real model: a fake tool-calling chat model emits one tool call
then a final answer; an in-memory FakeCell stands in for the Arkwen cell. The test
asserts the agent (a) called the cell-provided tool via cell.invoke_tool with the
right args, and (b) finished via the cell — i.e. the agent never executed the tool
itself, the cell did."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from axon.adapters import LangGraphAgent
from axon.core import Tool


class FakeCell:
    """Records interactions and returns canned tool results (the cell would really
    execute the tool under policy)."""

    def __init__(self, results):
        self.results = dict(results)
        self.calls = []
        self.emits = []
        self.finished = None

    def invoke_tool(self, call_id, name, args):
        self.calls.append((name, args))
        return {"type": "tool_result", "ok": True, "result": self.results.get(name, "")}

    def emit(self, event):
        self.emits.append(event)

    def finish(self, status, output=""):
        self.finished = (status, output)


class FakeToolCallingModel(BaseChatModel):
    """Returns queued AIMessages in order (bind_tools is a no-op)."""

    responses: list

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "fake-tool-calling"


def test_langgraph_agent_routes_tool_through_cell():
    tool = Tool(
        name="write_artifact",
        description="write a file into the workbench",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "data": {"type": "string"}},
            "required": ["path", "data"],
        },
    )
    cell = FakeCell({"write_artifact": "wrote hello.txt (2 bytes)"})
    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "write_artifact",
                "args": {"path": "hello.txt", "data": "hi"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="done — wrote hello.txt"),
    ]
    model = FakeToolCallingModel(responses=responses)

    agent = LangGraphAgent(
        "write hello.txt", tools=[tool], provider=model, cell=cell
    )
    result = agent.run()

    # the agent asked the CELL to run the tool (it did not run it itself) — the cell
    # is the sole authority that records the call (via invoke_tool)
    assert cell.calls == [("write_artifact", {"path": "hello.txt", "data": "hi"})]
    # the run finished through the cell, with the final message emitted
    assert any(e.get("kind") == "message" for e in cell.emits)
    assert cell.finished is not None and cell.finished[0] == "completed"
    assert result.status == "completed"
    assert "hello.txt" in result.output
