"""Run a real Axon agent against a local Ollama model, end-to-end.

The model (default qwen2.5:7b-instruct) drives the LangGraph agent; every tool call
is routed through the cell, which executes it under app-level policy and records an
append-only event log. Proves: the agent produces a real artifact WITHOUT ever
touching the filesystem itself — the cell does.

    python examples/run_ollama.py [model_name]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_ollama import ChatOllama

from axon.adapters import LangGraphAgent
from axon.localcell import LocalCell, default_toolset


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct"
    mission = (
        "Create a file named greeting.txt in the workbench whose contents are exactly:\n"
        "Hello from Axon - the neuron that speaks only through the cell.\n"
        "Use the write_artifact tool to do it, then briefly confirm you are done."
    )
    workbench = Path(tempfile.mkdtemp(prefix="axon-workbench-"))
    cell = LocalCell(mission, workbench)
    default_toolset(cell)

    model = ChatOllama(model=model_name, temperature=0)
    agent = LangGraphAgent(mission, tools=cell.tools(), provider=model, cell=cell)

    print(f"model={model_name}  tools={[t.name for t in cell.tools()]}  workbench={workbench}")
    print("--- running (real Ollama inference) ---")
    result = agent.run()

    print("\n=== cell event log (append-only truth) ===")
    for e in cell.events:
        label = e.get("name") or e.get("status") or (e.get("text", "")[:60])
        print(f"  seq {e['seq']:<2} {e.get('kind', ''):14} {label}")

    print(f"\nterminal: {result.status}  ({result.steps} messages)")
    print("\n=== artifacts the CELL produced on the agent's behalf ===")
    any_file = False
    for p in sorted(workbench.rglob("*")):
        if p.is_file():
            any_file = True
            print(f"  {p.relative_to(workbench)}:\n    {p.read_text()!r}")
    if not any_file:
        print("  (none — the model did not call write_artifact)")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
