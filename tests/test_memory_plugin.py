"""The cell-driven memory plugin: pull tools, push injection, and the host wiring.

No live cortex / model — everything runs on the dependency-free LocalMemoryBackend,
so these tests pin the *plugin contract* (what the cell provisions and injects),
not recall quality.
"""
from __future__ import annotations

import json
from pathlib import Path

from axon import LocalMemoryBackend, MemoryPlugin, RouteDecision, activate_plugins
from axon.localcell import LocalCell
from axon.memory import RECALL_TOOL, STORE_TOOL


def _cell(tmp_path: Path, mission: str = "m") -> LocalCell:
    return LocalCell(mission, tmp_path / "wb")


# ---- backend (mirrors cortex's EmbeddingRouter) ----

def test_local_backend_store_recall_ranks_by_overlap():
    b = LocalMemoryBackend()
    b.store("the user's name is Leon")
    b.store("the capital of France is Paris")
    hits = b.recall("what is the user's name", top_k=2)
    assert hits[0]["text"] == "the user's name is Leon"
    assert hits[0]["score"] >= hits[1]["score"]


def test_local_backend_route_injects_on_match_and_skips_on_miss():
    b = LocalMemoryBackend(threshold=0.2)
    b.store("the user's name is Leon")
    hit = b.route("remind me what the user's name is")
    assert hit.should_inject and hit.context and "Leon" in hit.context
    miss = b.route("compute the tenth prime number")
    assert not miss.should_inject and miss.context is None


# ---- pull surface: tools provisioned on the cell ----

def test_plugin_provisions_recall_and_store_as_cell_tools(tmp_path):
    cell = _cell(tmp_path)
    plugin = MemoryPlugin(LocalMemoryBackend())
    activate_plugins(cell, [plugin], mission="anything")
    names = {t.name for t in cell.tools()}
    assert {RECALL_TOOL.name, STORE_TOOL.name} <= names

    # the agent-facing channel: store then recall through invoke_tool
    store_res = cell.invoke_tool("c1", "memory_store", {"content": "the sky is blue"})
    assert store_res["ok"] and json.loads(store_res["result"])["memory_id"] == "m_001"
    recall_res = cell.invoke_tool("c2", "memory_recall", {"query": "what color is the sky"})
    assert recall_res["ok"]
    assert json.loads(recall_res["result"])["hits"][0]["text"] == "the sky is blue"


def test_expose_tools_false_provisions_nothing(tmp_path):
    cell = _cell(tmp_path)
    activate_plugins(cell, [MemoryPlugin(LocalMemoryBackend(), expose_tools=False)], "m")
    assert cell.tools() == []


# ---- push surface: cell-driven injection into context ----

def test_before_session_injects_matching_memory_into_context(tmp_path):
    backend = LocalMemoryBackend(threshold=0.2)
    backend.store("the user's name is Leon")
    plugin = MemoryPlugin(backend)
    ctx = activate_plugins(cell := _cell(tmp_path), [plugin], mission="greet the user by name")
    assert "Leon" in ctx
    assert plugin.last_decision and plugin.last_decision.should_inject
    # injection is a cell action — it lands in the append-only log
    assert any(e.get("kind") == "plugin_inject" and e.get("plugin") == "memory" for e in cell.events)


def test_before_session_preserves_base_context_and_appends(tmp_path):
    backend = LocalMemoryBackend(threshold=0.2)
    backend.store("the user's name is Leon")
    ctx = activate_plugins(_cell(tmp_path), [MemoryPlugin(backend)],
                           mission="what is the user's name", context="## Materials\nfoo")
    assert ctx.startswith("## Materials\nfoo")
    assert "Leon" in ctx


def test_no_injection_on_miss_leaves_context_untouched(tmp_path):
    backend = LocalMemoryBackend(threshold=0.5)
    backend.store("the user's name is Leon")
    cell = _cell(tmp_path)
    ctx = activate_plugins(cell, [MemoryPlugin(backend)], mission="unrelated arithmetic", context="base")
    assert ctx == "base"
    assert not any(e.get("kind") == "plugin_inject" for e in cell.events)


def test_inject_false_disables_push_but_keeps_pull(tmp_path):
    backend = LocalMemoryBackend(threshold=0.2)
    backend.store("the user's name is Leon")
    cell = _cell(tmp_path)
    ctx = activate_plugins(cell, [MemoryPlugin(backend, inject=False)], mission="name?", context="base")
    assert ctx == "base"  # no push
    assert {t.name for t in cell.tools()} >= {"memory_recall", "memory_store"}  # pull still there


# ---- the injected context reaches the agent's system prompt ----

def test_injected_context_flows_into_agent_system_prompt(tmp_path):
    from axon.adapters import MinimalAgent

    backend = LocalMemoryBackend(threshold=0.2)
    backend.store("the user's name is Leon")
    cell = _cell(tmp_path)
    mission = "greet the user by name"
    ctx = activate_plugins(cell, [MemoryPlugin(backend)], mission=mission)

    class _NullProvider:
        def complete(self, messages, tools):
            from axon.core import ProviderResponse
            return ProviderResponse(text="done")

    agent = MinimalAgent(mission, context=ctx, tools=cell.tools(), provider=_NullProvider(), cell=cell)
    assert "Leon" in agent.system_prompt()


# ---- RouteDecision shape (contract with cortex's memory_route) ----

def test_route_decision_defaults():
    d = RouteDecision(should_inject=False)
    assert d.context is None and d.reason == "" and d.uncertainty is None
