"""Memory plugin — the cell-driven latent-memory built-in, backed by cortex.

This is the concrete :mod:`axon.plugin` that realizes the two-strategy split the
memory-injection research settled on:

- **Strategy A (pull)** — ``memory_store`` / ``memory_recall`` cell tools the agent
  may call on demand. Pure text in / text out, so they ride a plain MCP tool call.
- **Strategy C (push)** — ``before_session`` asks cortex's *router* (``memory_route``)
  whether this turn should be primed and, if so, the cell folds the recalled text
  into the agent's context. The agent never asks; the cell injects.

Only the *push* half needs the plugin: a plain MCP server cannot write into the
agent's context, and the SeaKR-style "inject when the model is uncertain" signal
comes from cortex's own instrumented model, not from anything the reasoner (Ollama /
vLLM) can expose. So the router runs cortex-side and *returns* the decision + text;
the cell performs the injection. The latent KV round-trip stays entirely inside
cortex (a serving engine can't ingest an opaque KV blob) — over this boundary,
memory is always text.

Two backends implement the same :class:`MemoryBackend` contract:

- :class:`McpMemoryBackend` — the real bridge: JSON-RPC over stdio to cortex's
  ``engram-mcp`` (``memory_store`` / ``memory_recall`` / ``memory_route``).
- :class:`LocalMemoryBackend` — a dependency-free in-process reference mirroring
  cortex's ``EmbeddingRouter`` (cosine threshold, no uncertainty), for tests and
  offline examples without a live cortex.
"""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .core import Tool


# ─────────────────────────── contract ───────────────────────────

@dataclass
class RouteDecision:
    """Cortex's pre-loop routing verdict (mirrors ``engram_agent::router::RouteDecision``)."""

    should_inject: bool
    context: str | None = None
    reason: str = ""
    uncertainty: float | None = None


@runtime_checkable
class MemoryBackend(Protocol):
    """The cortex-facing surface the plugin needs. ``store``/``recall`` are the pull
    tools; ``route`` is the cell-driven push decision."""

    def store(self, content: str, tags: list[str] | None = None) -> dict: ...
    def recall(self, query: str, top_k: int = 3) -> list[dict]: ...
    def route(self, user_input: str) -> RouteDecision: ...


# ─────────────────────────── cell tools (pull) ───────────────────────────

RECALL_TOOL = Tool(
    "memory_recall",
    "Search long-term memory (cortex) for information relevant to a query and return the best matches as text.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A question or cue describing what to recall."},
            "top_k": {"type": "integer", "description": "How many memories to return (default 3)."},
        },
        "required": ["query"],
    },
)

STORE_TOOL = Tool(
    "memory_store",
    "Save a piece of information to long-term memory (cortex) so it can be recalled in later turns or sessions.",
    {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
        },
        "required": ["content"],
    },
)


# ─────────────────────────── the plugin ───────────────────────────

@dataclass
class MemoryPlugin:
    """A cell-driven memory built-in over any :class:`MemoryBackend`.

    ``expose_tools`` provisions the pull tools (``memory_store`` / ``memory_recall``);
    ``inject`` enables the cell-driven push (routing + priming). Both default on —
    the pull tools let the agent recall explicitly, the router primes it
    automatically when cortex judges the turn needs it. Disable either to run just
    one strategy.
    """

    backend: MemoryBackend
    inject: bool = True
    expose_tools: bool = True
    header: str = "Relevant information recalled from long-term memory:"
    name: str = "memory"
    #: The last routing verdict — surfaced for logging / inspection after priming.
    last_decision: RouteDecision | None = field(default=None, init=False)

    def provide_tools(self, register: Callable[..., None]) -> None:
        if not self.expose_tools:
            return
        backend = self.backend

        def _recall(cell, query: str, top_k: int = 3) -> str:
            return json.dumps({"hits": backend.recall(query, int(top_k))})

        def _store(cell, content: str, tags: list[str] | None = None) -> str:
            return json.dumps(backend.store(content, list(tags or [])))

        register(RECALL_TOOL, _recall)
        register(STORE_TOOL, _store)

    def before_session(self, mission: str, context: str = "") -> str:
        if not self.inject:
            return context
        decision = self.backend.route(mission)
        self.last_decision = decision
        if decision.should_inject and decision.context:
            block = f"{self.header}\n{decision.context}".rstrip()
            return f"{context}\n\n{block}".strip() if context else block
        return context


# ─────────────────────────── cortex bridge (MCP over stdio) ───────────────────────────

class McpMemoryBackend:
    """Talks JSON-RPC 2.0 over stdio to cortex's ``engram-mcp`` and adapts its tools
    to :class:`MemoryBackend`. The tool payloads come back as a JSON text blob in the
    MCP ``content`` array (the only channel the protocol offers), which is exactly
    why *injection* can't live here — it can only return data; the cell does the push.

    ``command`` is the argv to launch the server, e.g.::

        McpMemoryBackend(["./target/debug/engram-mcp", "--device", "cpu"], cwd="/…/cortex")

    Requires cortex to expose ``memory_store`` / ``memory_recall`` / ``memory_route``
    (the memory tool surface; the base ``engram-mcp`` ships the Tap surface only).
    """

    def __init__(self, command: list[str], *, cwd: str | None = None, env: dict | None = None):
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        self._id = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        assert self._proc.stdin and self._proc.stdout
        self._id += 1
        rid = self._id
        self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise ConnectionError("engram-mcp closed the channel")
            msg = json.loads(line)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"engram-mcp error on {method}: {msg['error']}")
                return msg.get("result", {})

    def _call_tool(self, name: str, args: dict) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": args})
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        if result.get("isError"):
            raise RuntimeError(f"cortex tool {name!r} failed: {text}")
        return json.loads(text) if text else {}

    def store(self, content: str, tags: list[str] | None = None) -> dict:
        return self._call_tool("memory_store", {"content": content, "tags": tags or []})

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        return self._call_tool("memory_recall", {"query": query, "top_k": top_k}).get("hits", [])

    def route(self, user_input: str) -> RouteDecision:
        d = self._call_tool("memory_route", {"user_input": user_input})
        return RouteDecision(
            should_inject=bool(d.get("should_inject")),
            context=d.get("context"),
            reason=d.get("reason", ""),
            uncertainty=d.get("uncertainty"),
        )

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        finally:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self) -> "McpMemoryBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ─────────────────────────── offline reference backend ───────────────────────────

def _bow(text: str) -> Counter:
    return Counter(text.lower().split())


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(t, 0) for t, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class LocalMemoryBackend:
    """Dependency-free in-process backend mirroring cortex's ``EmbeddingRouter``:
    bag-of-words cosine over stored text, inject when the best match clears
    ``threshold``. No latent state and no uncertainty signal (recall quality is weak
    by design) — it exists so the plugin runs and tests without a live cortex."""

    def __init__(self, threshold: float = 0.25, top_k: int = 3):
        self.threshold = threshold
        self.top_k = top_k
        self._items: list[dict] = []
        self._n = 0

    def store(self, content: str, tags: list[str] | None = None) -> dict:
        self._n += 1
        mid = f"m_{self._n:03}"
        self._items.append({"memory_id": mid, "text": content, "tags": list(tags or []), "bow": _bow(content)})
        return {"memory_id": mid, "tokens_in": len(content.split()), "latent_snapshot": False}

    def _scored(self, query: str) -> list[tuple[float, dict]]:
        q = _bow(query)
        scored = [(_cosine(q, it["bow"]), it) for it in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        return [
            {"memory_id": it["memory_id"], "text": it["text"], "tags": it["tags"], "score": round(s, 4)}
            for s, it in self._scored(query)[: max(top_k, 1)]
        ]

    def route(self, user_input: str) -> RouteDecision:
        scored = self._scored(user_input)[: self.top_k]
        best = scored[0][0] if scored else 0.0
        if best < self.threshold:
            return RouteDecision(False, None, f"top cosine {best:.3f} < threshold {self.threshold:.3f}")
        block = "".join(f"- [{it['memory_id']}] {it['text']}\n" for s, it in scored if s >= self.threshold)
        return RouteDecision(
            should_inject=True,
            context=block or None,
            reason=f"top cosine {best:.3f} >= threshold {self.threshold:.3f}",
        )
