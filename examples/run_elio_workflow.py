"""Run a dynamic 'elio loop' workflow via axon against Ollama.

Direct mode (default): run a hand-authored design->implement->test->commit workflow
through the engine — deterministic proof that stages / parallel sub-agents / quality
gate / report-via-cell / status all work with a real model.

Agent mode (--agent): give the MAIN agent ONE tool (run_workflow) + a skill, and let
IT author the YAML (design -> implement -> test -> commit) and drive the loop. The
main agent has NO file tools of its own — its only way to produce anything is to
decompose the task into a workflow and run it (that is the whole point).

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

# A design -> implement -> test -> commit pipeline (the "elio loop"). Used verbatim
# in direct mode, and offered to the main agent as the template it should author in
# agent mode.
DESIGN_IMPL_COMMIT = """
goal: Produce a tiny Python module calc.py with add(a,b) and a smoke test.
stages:
  - name: design
    agents:
      - name: designer
        prompt: In 2-3 sentences, specify a Python function add(a,b) that returns a+b, with a docstring. Do not write code yet.
  - name: implement
    gate: "the stage output describes an add(a, b) function that returns a + b"
    agents:
      - name: coder
        prompt: Write the file calc.py with the add(a,b) function and a docstring, using the write_artifact tool. Then confirm in one sentence.
  - name: test
    gate: "the stage output states a concrete assertion equivalent to add(2, 3) == 5"
    agents:
      - name: tester
        prompt: State one concrete assertion that verifies add(2,3)==5. One line, e.g. `assert add(2, 3) == 5`.
  - name: commit
    gate: "the stage output shows a valid Python test file using `from calc import add` and asserting add(2, 3) == 5"
    agents:
      - name: committer
        prompt: "Write the file test_calc.py using the write_artifact tool. Its exact contents must be:\nfrom calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\nThen confirm in one sentence."
"""

# The skill that turns the main agent into a workflow author. It is deliberately
# forceful: the agent has no other way to touch the workbench, so it must decompose.
DECOMPOSE_SKILL = Skill(
    name="decompose-into-workflow",
    instructions=(
        "You cannot write files or run code yourself. Your ONLY capability is the "
        "run_workflow tool. To accomplish the mission you MUST call run_workflow "
        "exactly once, passing a YAML spec that decomposes the work into ordered "
        "stages: design -> implement -> test -> commit.\n\n"
        "Each stage has a name, a list of agents ({name, prompt}), and optionally a "
        "gate: '<pass criteria>'. The implement and commit stage agents write files "
        "with the write_artifact tool; the design and test agents only think/describe. "
        "Keep each agent prompt small and focused.\n\n"
        "Required YAML shape (author your own prompts for the mission):\n"
        "```yaml\n" + DESIGN_IMPL_COMMIT.strip() + "\n```\n\n"
        "Do NOT answer the mission in prose. Call run_workflow with your YAML now."
    ),
)

AGENT_MISSION = (
    "Build a tiny Python module calc.py exposing add(a, b) (with a docstring) plus a "
    "pytest-style test file test_calc.py that asserts add(2, 3) == 5. Deliver both files."
)


def main() -> int:
    args = sys.argv[1:]
    agent_mode = "--agent" in args
    model_name = next((a for a in args if not a.startswith("--")), "qwen2.5:7b-instruct")
    workbench = Path(tempfile.mkdtemp(prefix="axon-elio-"))

    def provider():
        return ChatOllama(model=model_name, temperature=0)

    cell = LocalCell("elio-loop demo", workbench)
    print(f"model={model_name}  mode={'agent-driven' if agent_mode else 'direct'}  workbench={workbench}")

    if agent_mode:
        # The main agent gets ONE tool: run_workflow. No default_toolset — it cannot
        # write files directly, so decomposing into a workflow is its only path.
        register_workflow_tool(cell, provider)
        agent = LangGraphAgent(AGENT_MISSION, tools=cell.tools(), skills=[DECOMPOSE_SKILL],
                               provider=provider(), cell=cell)
        print(f"--- main agent (tools={[t.name for t in cell.tools()]}); it must author + run its own elio loop ---")
        result = agent.run()
        called = any(e.get("kind") == "workflow_started" for e in cell.events)
        print(f"\nmain agent terminal: {result.status}   run_workflow called: {called}")
        if not called:
            print("  !! the model answered directly instead of decomposing — agent-driven run did NOT happen")
    else:
        default_toolset(cell)
        print("--- running workflow directly (real Ollama sub-agents) ---")
        summary = run_workflow(cell, DESIGN_IMPL_COMMIT, provider_factory=provider)
        print(f"\n{summary}")

    print("\n=== cell event log (append-only truth) ===")
    for e in cell.events:
        label = e.get("stage") or e.get("goal") or e.get("status") or ""
        extra = e.get("agent") or e.get("decision") or ""
        print(f"  seq {e['seq']:<2} {e.get('kind',''):18} {str(label)[:44]:44} {extra}")

    print("\n=== files produced (via the cell) ===")
    produced = [p for p in sorted(workbench.rglob("*")) if p.is_file()]
    for p in produced:
        print(f"  {p.relative_to(workbench)}")
    if not produced:
        print("  (none)")
    # In agent mode success means the agent actually drove a workflow to completion.
    if agent_mode:
        return 0 if any(e.get("kind") == "workflow_finished" for e in cell.events) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
