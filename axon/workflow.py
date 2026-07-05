"""Dynamic multi-agent workflow — the "elio loop" as a callable capability.

An agent gets a task and one tool: ``run_workflow(spec)``. It generates a YAML spec
describing STAGES of small sub-tasks (e.g. design -> review -> implement -> test ->
review -> commit), each stage holding N agent nodes, optionally guarded by a quality
gate. The TRUSTED cell then runs it:

  * stages run SEQUENTIALLY (each sees prior stages' results as context),
  * agent nodes within a stage run in PARALLEL,
  * each node is itself a confined sub-axon talking to its own sub-cell,
  * a stage's quality gate (an LLM check) must PASS to proceed (fail-closed),
  * the result is written back THROUGH the cell, and progress is emitted as events.

This is the axon-native realization of the ELIO outer-loop pattern (stages / nodes /
eval gates). The production ELIO engine (TypeScript) can later back the same tool
contract. On WSL2 sub-agents are cooperative (in-process); OS confinement is Arkwen's
job — the tool being TRUSTED (run by the cell, not the agent) is the point: the agent
can only *request* a workflow, never execute scripts or reach a terminal itself.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

from .core import Tool


@dataclass
class AgentNode:
    name: str
    prompt: str


@dataclass
class Stage:
    name: str
    agents: list[AgentNode]
    gate: str | None = None


@dataclass
class WorkflowSpec:
    goal: str
    stages: list[Stage]


def parse_spec(spec) -> WorkflowSpec:
    """Accept a dict or a YAML string: {goal, stages:[{name, agents:[{name,prompt}], gate?}]}."""
    if isinstance(spec, str):
        import yaml

        spec = yaml.safe_load(spec)
    if not isinstance(spec, dict):
        raise ValueError("workflow spec must be a mapping (or a YAML string)")
    stages: list[Stage] = []
    for s in spec.get("stages", []):
        agents = [AgentNode(a.get("name", f"agent{i + 1}"), a["prompt"])
                  for i, a in enumerate(s.get("agents", []))]
        if not agents:
            raise ValueError(f"stage {s.get('name')!r} has no agents")
        stages.append(Stage(s.get("name", "stage"), agents, s.get("gate")))
    if not stages:
        raise ValueError("workflow spec has no stages")
    return WorkflowSpec(spec.get("goal", ""), stages)


def run_workflow(parent_cell, spec, provider_factory, *, max_parallel: int = 4) -> str:
    """Execute a workflow spec. Returns a short summary; full detail is in the report
    written through the cell and in the emitted event stream."""
    from .adapters import LangGraphAgent
    from .localcell import LocalCell, default_toolset

    wf = parse_spec(spec)
    parent_cell.emit({"kind": "workflow_started", "goal": wf.goal,
                      "stages": [s.name for s in wf.stages]})
    transcript = [f"# Workflow: {wf.goal}\n"]
    context = ""  # accumulated prior-stage results, injected into later stages

    def _run_node(stage_name: str, node: AgentNode) -> dict:
        parent_cell.emit({"kind": "agent_started", "stage": stage_name, "agent": node.name})
        sub_wb = parent_cell.workbench / "_stages" / stage_name / node.name
        sub_cell = LocalCell(node.prompt, sub_wb, context=context,
                             secrets=getattr(parent_cell, "_secrets", None))
        default_toolset(sub_cell)  # safe tools only; NOT run_workflow (depth-bounded to 1)
        prompt = node.prompt + (f"\n\n## Context from prior stages\n{context}" if context else "")
        try:
            res = LangGraphAgent(prompt, tools=sub_cell.tools(),
                                 provider=provider_factory(), cell=sub_cell).run()
            out, status = res.output, res.status
        except Exception as e:  # a failing node must not crash the whole workflow
            out, status = f"error: {e}", "failed"
        parent_cell.emit({"kind": "agent_finished", "stage": stage_name,
                          "agent": node.name, "status": status})
        return {"agent": node.name, "status": status, "output": out}

    for i, stage in enumerate(wf.stages, 1):
        parent_cell.emit({"kind": "stage_started", "stage": stage.name, "n_agents": len(stage.agents)})
        transcript.append(f"\n## Stage {i}: {stage.name}\n")
        if len(stage.agents) == 1:
            results = [_run_node(stage.name, stage.agents[0])]
        else:
            with cf.ThreadPoolExecutor(max_workers=min(max_parallel, len(stage.agents))) as ex:
                results = list(ex.map(lambda n: _run_node(stage.name, n), stage.agents))

        for r in results:
            transcript.append(f"\n### {r['agent']} ({r['status']})\n{r['output']}\n")
        context += f"\n### {stage.name}\n" + "\n".join(f"- {r['agent']}: {r['output']}" for r in results) + "\n"

        if stage.gate:
            verdict = _run_gate(stage, results, provider_factory)
            transcript.append(f"\n**Quality gate:** {verdict['decision']} — {verdict['reason']}\n")
            parent_cell.emit({"kind": "gate_result", "stage": stage.name, **verdict})
            if verdict["decision"] != "PASS":  # fail-closed: a failed gate stops the loop
                ref = _write_report(parent_cell, transcript, "failed")
                parent_cell.emit({"kind": "workflow_finished", "status": "failed",
                                  "failed_stage": stage.name, "report": ref})
                return f"workflow FAILED at gate '{stage.name}': {verdict['reason']} (report: {ref})"

    ref = _write_report(parent_cell, transcript, "completed")
    parent_cell.emit({"kind": "workflow_finished", "status": "completed", "report": ref})
    n_agents = sum(len(s.agents) for s in wf.stages)
    return f"workflow completed: {len(wf.stages)} stages, {n_agents} agents. report: {ref}"


def _run_gate(stage: Stage, results, provider_factory) -> dict:
    """A lightweight LLM quality gate: reads the stage output + criteria, returns
    PASS/FAIL. No tools — it only judges."""
    from langchain_core.messages import HumanMessage, SystemMessage

    summary = "\n".join(f"[{r['agent']}] {r['output']}" for r in results)
    msgs = [
        SystemMessage(content=("You are a strict quality gate. Decide PASS or FAIL for the stage. "
                               "Reply with PASS or FAIL as the first word, then a short reason.")),
        HumanMessage(content=f"Criteria for stage '{stage.name}':\n{stage.gate}\n\nStage output:\n{summary}"),
    ]
    try:
        text = (provider_factory().invoke(msgs).content or "").strip()
    except Exception as e:
        return {"decision": "FAIL", "reason": f"gate error: {e}"}
    parts = text.split(None, 1)
    decision = "PASS" if parts and parts[0].upper().startswith("PASS") else "FAIL"
    reason = (parts[1].strip() if len(parts) > 1 else text)[:200]
    return {"decision": decision, "reason": reason}


def _write_report(parent_cell, transcript, status) -> str:
    """Write the workflow report THROUGH the cell (redacted + logged), not as a raw file."""
    content = "".join(transcript) + f"\n\n---\nstatus: {status}\n"
    res = parent_cell.invoke_tool("wf_report", "write_artifact",
                                  {"path": "workflow-report.md", "data": content})
    return res.get("result", "workflow-report.md")


# ---- the cell tool the agent may request ----

RUN_WORKFLOW = Tool(
    "run_workflow",
    ("Run a multi-stage agent workflow (the elio loop) to accomplish a larger task. "
     "Stages run sequentially and see prior results; agents within a stage run in "
     "parallel; each stage may declare a quality gate that must PASS to proceed. "
     "Provide a YAML spec with: goal, and stages: a list of {name, agents: [{name, "
     "prompt}], gate?: '<pass criteria>'}. Results are written via the cell and "
     "progress is observable in the event stream."),
    {"type": "object",
     "properties": {"spec": {"type": "string", "description": "YAML workflow spec (goal + stages)"}},
     "required": ["spec"]},
)


def register_workflow_tool(cell, provider_factory) -> None:
    """Register run_workflow on the cell, wired to a model provider factory.

    This is a TRUSTED tool: the agent only *requests* a workflow; the cell (harness)
    runs the sub-agents. The agent never executes scripts or touches a terminal."""

    def _impl(cell, spec):
        return run_workflow(cell, spec, provider_factory)

    cell.register(RUN_WORKFLOW, _impl)
