"""Cell-driven memory injection, end-to-end.

The cell seeds a memory, then ACTIVATES the memory plugin (``activate_plugins``):
the plugin registers ``memory_store`` / ``memory_recall`` as cell tools (pull) AND
asks cortex's router whether to prime this turn — and because the mission needs the
stored fact, the cell folds it into the agent's context (push). The agent then
answers using a fact it was never told in the mission, and the injection shows up in
the cell's append-only log as a ``plugin_inject`` event.

    python examples/run_memory_agent.py [model_name]
        # offline reference backend (no cortex needed)
    python examples/run_memory_agent.py --cortex /path/to/cortex [model_name]
        # real bridge: cortex's engram-mcp over stdio (needs memory_* tools)

Requires a local Ollama (default qwen2.5:7b-instruct) for the reasoning half.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_ollama import ChatOllama

from axon import LocalMemoryBackend, McpMemoryBackend, MemoryPlugin, activate_plugins
from axon.adapters import LangGraphAgent
from axon.localcell import LocalCell, default_toolset


def _backend(argv: list[str]):
    if "--cortex" in argv:
        i = argv.index("--cortex")
        cortex = argv[i + 1]
        argv[i : i + 2] = []
        return McpMemoryBackend(["./target/debug/engram-mcp", "--device", "cpu"], cwd=cortex)
    return LocalMemoryBackend(threshold=0.2)


def main() -> int:
    argv = sys.argv[1:]
    backend = _backend(argv)
    model_name = argv[0] if argv else "qwen2.5:7b-instruct"

    # The cell seeds durable memory (in a real run, prior turns/sessions did this).
    backend.store("The user's name is Leon and he prefers concise, direct answers.")

    mission = "Greet the user by name in one short sentence."
    workbench = Path(tempfile.mkdtemp(prefix="axon-memory-"))
    cell = LocalCell(mission, workbench)
    default_toolset(cell)

    # Cell-driven activation: provisions the memory tools AND primes the context.
    plugin = MemoryPlugin(backend)
    context = activate_plugins(cell, [plugin], mission=mission)

    d = plugin.last_decision
    print(f"model={model_name}  tools={[t.name for t in cell.tools()]}")
    print(f"router: should_inject={d.should_inject}  reason={d.reason!r}")
    print(f"--- injected context ---\n{context or '(none)'}\n------------------------")

    model = ChatOllama(model=model_name, temperature=0)
    agent = LangGraphAgent(mission, context=context, tools=cell.tools(), provider=model, cell=cell)
    result = agent.run()

    print("\n=== cell event log ===")
    for e in cell.events:
        label = e.get("plugin") or e.get("name") or e.get("status") or (e.get("text", "")[:60])
        print(f"  seq {e['seq']:<2} {e.get('kind', ''):14} {label}")
    print(f"\nterminal: {result.status}\noutput: {result.output!r}")

    if isinstance(backend, McpMemoryBackend):
        backend.close()
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
