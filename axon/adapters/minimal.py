"""Minimal, framework-free reference adapter — proves the Axon contract.

The raw tool-use loop: ask the provider, route any tool calls back through the
cell, feed the results back, repeat until the model is done. The LangGraph and
CrewAI adapters are richer implementations of THIS same contract."""
from __future__ import annotations

from ..core import Agent, Message, RunResult


class MinimalAgent(Agent):
    MAX_STEPS = 50

    def run(self) -> RunResult:
        messages = [
            Message("system", self.system_prompt()),
            Message("user", self.spec.prompt),
        ]
        for step in range(1, self.MAX_STEPS + 1):
            resp = self.provider.complete(messages, self.spec.tools)
            messages.append(Message("assistant", resp.text))
            if not resp.tool_calls:
                self.cell.emit({"kind": "message", "text": resp.text})
                self.cell.finish("completed", resp.text)
                return RunResult("completed", resp.text, step)
            for tc in resp.tool_calls:
                self.cell.emit({"kind": "tool_call", "name": tc.name, "args": tc.args})
                # Executed by the CELL under policy — never in this process.
                result = self.cell.invoke_tool(tc.id, tc.name, tc.args)
                messages.append(Message("tool", str(result.get("result", "")), tool_call_id=tc.id))
        self.cell.finish("failed", "step budget exhausted")
        return RunResult("failed", "step budget exhausted", self.MAX_STEPS)
