"""Axon cell SDK — the single channel between the agent and the Arkwen workcell.

The cell ANNOUNCES what the agent may use (mission, context, tools, skills) and
EXECUTES tool calls on the agent's behalf under policy. In a confined workcell this
JSON-over-stdio channel is the ONLY hole in the jar: the agent process has no
ambient authority, so "only cell-provided tools" becomes a real boundary — the
guarantee comes from the confinement, the tool list only makes it useful.

Wire format (MCP-like, line-delimited JSON over stdio):
    <- {"type":"session","mission":..,"context":..,"tools":[..],"skills":[..]}
    -> {"type":"tool_call","id":..,"name":..,"args":{..}}
    <- {"type":"tool_result","id":..,"ok":true,"result":..}
    -> {"type":"emit","event":{..}}          # cooperative observability
    -> {"type":"finish","status":..,"output":..}
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from .core import Skill, Tool


@dataclass
class Session:
    mission: str
    context: str
    tools: list[Tool]
    skills: list[Skill]


class Cell:
    """Client for the Arkwen cell channel. Defaults to stdio; inject other streams
    (e.g. a socket, or in-memory buffers in tests). Policy / redaction / egress all
    live on the cell (Arkwen) side — this client only speaks the protocol."""

    def __init__(self, rx=None, tx=None):
        self._rx = rx or sys.stdin
        self._tx = tx or sys.stdout

    def _send(self, obj: dict) -> None:
        self._tx.write(json.dumps(obj) + "\n")
        self._tx.flush()

    def _recv(self) -> dict:
        line = self._rx.readline()
        if not line:
            raise ConnectionError("cell channel closed")
        return json.loads(line)

    def handshake(self) -> Session:
        """Receive the mission + the tools/skills the cell chooses to provide."""
        msg = self._recv()
        if msg.get("type") != "session":
            raise ValueError(f"expected session handshake, got {msg.get('type')!r}")
        tools = [Tool(t["name"], t.get("description", ""), t.get("input_schema", {}))
                 for t in msg.get("tools", [])]
        skills = [Skill(s["name"], s.get("instructions", ""), s.get("tools", []))
                  for s in msg.get("skills", [])]
        return Session(msg.get("mission", ""), msg.get("context", ""), tools, skills)

    def invoke_tool(self, call_id: str, name: str, args: dict) -> dict:
        """Route a tool call BACK to the cell for policy-enforced execution."""
        self._send({"type": "tool_call", "id": call_id, "name": name, "args": args})
        res = self._recv()
        if res.get("type") != "tool_result":
            raise ValueError(f"expected tool_result, got {res.get('type')!r}")
        return res

    def emit(self, event: dict) -> None:
        """Cooperative observability — folded into Arkwen's event stream."""
        self._send({"type": "emit", "event": event})

    def finish(self, status: str, output: str = "") -> None:
        self._send({"type": "finish", "status": status, "output": output})
