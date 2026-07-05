"""Run a dynamic 'elio loop' workflow via axon against Ollama.

Direct mode (default): run a hand-authored design->implement->test workflow through
the engine — deterministic proof that stages / parallel sub-agents / quality gate /
report-via-cell / status all work with a real model.

Agent mode (--agent): give the MAIN agent the run_workflow tool + a skill, and let IT
generate the YAML and call the workflow (best-effort; depends on the model).

    python examples/run_elio_workflow.py [model] [--agent]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_ollama import ChatOllama

from axon.adapters import LangGraphAgent
from axon.core import Skill
from axon.localcell import LocalCell, default_toolset
from axon.workflow import register_workflow_tool, run_workflow

DESIGN_IMPL_TEST = """
goal: Produce a tiny Python function add(a,b) with a docstring and a smoke test.
stages:
  - name: design
    agents:
      - name: designer
        prompt: In 2-3 sentences, specify a Python function add(a,b) that returns a+b, with a docstring. Do not write code yet.
  - name: implement
    gate: "there is a file add.py containing a def add(a, b) that returns a + b"
    agents:
      - name: coder
        prompt: Write the file add.py with the add(a,b) function and a docstring, using the write_artifact tool. Then confirm.
  - name: test
    agents:
      - name: tester
        prompt: Describe one concrete assert that would verify add(2,3)==5. One line.
"""


def main() -> int:
    args = sys.argv[1:]
    agent_mode = "--agent" in args
    model_name = next((a for a in args if not a.startswith("--")), "qwen2.5:7b-instruct")
    workbench = Path(tempfile.mkdtemp(prefix="axon-elio-"))

    def provider():
        return ChatOllama(model=model_name, temperature=0)

    cell = LocalCell("elio-loop demo", workbench)
    default_toolset(cell)
    print(f"model={model_name}  mode={'agent-driven' if agent_mode else 'direct'}  workbench={workbench}")

    if agent_mode:
        register_workflow_tool(cell, provider)
        skill = Skill(
            name="decompose-into-workflow",
            instructions=("For any non-trivial task, DO NOT do it yourself. Instead design a "
                          "multi-stage workflow (e.g. design -> implement -> test) and run it with "
                          "the run_workflow tool, passing a YAML spec (goal + stages:[{name, "
                          "agents:[{name,prompt}], gate?}]). Each agent should have a small, focused task."),
        )
        mission = "Produce a tiny Python function add(a,b) with a docstring and a smoke test."
        agent = LangGraphAgent(mission, tools=cell.tools(), skills=[skill], provider=provider(), cell=cell)
        print("--- main agent (it should build + run its own elio loop) ---")
        result = agent.run()
        print(f"\nmain agent terminal: {result.status}")
    else:
        print("--- running workflow directly (real Ollama sub-agents) ---")
        summary = run_workflow(cell, DESIGN_IMPL_TEST, provider_factory=provider)
        print(f"\n{summary}")

    print("\n=== cell event log (append-only truth) ===")
    for e in cell.events:
        label = e.get("stage") or e.get("goal") or e.get("status") or ""
        extra = e.get("agent") or e.get("decision") or ""
        print(f"  seq {e['seq']:<2} {e.get('kind',''):18} {label:10} {extra}")

    print("\n=== files produced (via the cell) ===")
    for p in sorted(workbench.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(workbench)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
