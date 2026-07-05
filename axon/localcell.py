"""Cooperative reference CELL for local runs + tests.

⚠️  This is NOT the security boundary. It executes tools under APP-LEVEL policy
(path sandboxing, secret redaction, an append-only event log) but it does NOT
confine the agent process — the agent could still bypass it with raw I/O. The real
boundary is Arkwen's Go cell-shim (gVisor/Firecracker + seccomp + nftables egress).

Use this to exercise the tool-routing + policy layer and to run axon end-to-end
against a real model, independent of the Arkwen (Go) runtime. It implements the same
agent-facing surface the real cell does: invoke_tool / emit / finish, plus tool
provisioning (the cell decides which tools exist)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .core import Tool


class LocalCell:
    def __init__(self, mission: str, workbench: Path, context: str = "", secrets: dict | None = None):
        self.mission = mission
        self.context = context
        self.workbench = Path(workbench)
        self.workbench.mkdir(parents=True, exist_ok=True)
        self._secrets = dict(secrets or {})
        self._tools: dict[str, tuple[Tool, Callable]] = {}
        self.events: list[dict] = []  # append-only "truth"

    # ---- provisioning: the CELL decides which tools the agent gets ----
    def register(self, tool: Tool, impl: Callable) -> None:
        self._tools[tool.name] = (tool, impl)

    def tools(self) -> list[Tool]:
        return [t for (t, _) in self._tools.values()]

    # ---- policy ----
    def _redact(self, s: str) -> str:
        for v in self._secrets.values():
            if v:
                s = s.replace(v, "[REDACTED]")
        return s

    def _redact_obj(self, o):
        """Recursively redact every string in a value — applied to EVERYTHING that
        enters the persisted event log (args, results, messages, output), regardless
        of who produced it (Invariant 5: no secret in persisted logs)."""
        if isinstance(o, str):
            return self._redact(o)
        if isinstance(o, dict):
            return {k: self._redact_obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [self._redact_obj(v) for v in o]
        return o

    def _log(self, ev: dict) -> None:
        # redaction happens BEFORE persistence — the log is the durable surface
        self.events.append(self._redact_obj({"seq": len(self.events), **ev}))

    # ---- the agent-facing channel (mirrors the real cell) ----
    def invoke_tool(self, call_id: str, name: str, args: dict) -> dict:
        self._log({"kind": "tool_call", "name": name, "args": args})
        if name not in self._tools:
            # fail-closed: the cell never announced this tool
            res = {"type": "tool_result", "ok": False, "error": f"tool '{name}' not provided by cell"}
        else:
            _tool, impl = self._tools[name]
            try:
                # the agent receives the REAL result (it works in-cell on real data);
                # redaction is applied only where the result is PERSISTED (the log).
                res = {"type": "tool_result", "ok": True, "result": str(impl(self, **args))}
            except Exception as e:  # policy violation or tool error → fail-closed result
                res = {"type": "tool_result", "ok": False, "error": str(e)}
        self._log({"kind": "tool_result", "name": name, "ok": res.get("ok"),
                   "result": res.get("result"), "error": res.get("error")})
        return res

    def emit(self, event: dict) -> None:
        self._log(event)

    def finish(self, status: str, output: str = "") -> None:
        self._log({"kind": "finish", "status": status, "output": self._redact(output)})


# ---- reference tools: executed BY the cell, under policy ----

def _safe_path(cell: LocalCell, path: str) -> Path:
    """App-level sandbox: confine paths to the workbench (a stand-in for the real
    cell's OS-level rootfs confinement)."""
    root = cell.workbench.resolve()
    p = (cell.workbench / path).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"path escapes the workbench sandbox: {path!r}")
    return p


def write_artifact(cell: LocalCell, path: str, data: str) -> str:
    p = _safe_path(cell, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)
    return f"wrote {path} ({len(data)} bytes)"


def read_material(cell: LocalCell, path: str) -> str:
    p = _safe_path(cell, path)
    if not p.is_file():
        raise FileNotFoundError(f"no such material: {path}")
    return p.read_text()


def list_workbench(cell: LocalCell) -> str:
    files = sorted(str(p.relative_to(cell.workbench)) for p in cell.workbench.rglob("*") if p.is_file())
    return "\n".join(files) if files else "(empty)"


WRITE_ARTIFACT = Tool(
    "write_artifact", "Write a text file into the workbench.",
    {"type": "object", "properties": {"path": {"type": "string"}, "data": {"type": "string"}}, "required": ["path", "data"]},
)
READ_MATERIAL = Tool(
    "read_material", "Read a text file from the workbench.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)
LIST_WORKBENCH = Tool("list_workbench", "List files currently in the workbench.", {"type": "object", "properties": {}})


def default_toolset(cell: LocalCell) -> None:
    """Register a small, safe reference toolset on the cell."""
    cell.register(WRITE_ARTIFACT, write_artifact)
    cell.register(READ_MATERIAL, read_material)
    cell.register(LIST_WORKBENCH, list_workbench)
