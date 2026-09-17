"""Contract bridge: axon workflow spec  <->  ELIO feature.yaml (`elio/v1` Feature).

The axon ``run_workflow`` tool and the ELIO Outer-Loop engine describe the SAME
thing in two vocabularies:

    axon:  goal + stages[{name, agents[{name, prompt}], gate?}]
    elio:  apiVersion elio/v1, kind Feature, feature{autonomy, artifact{kind,
           evalGate}, io, graph{state, steps[], edges[]}}

``to_elio_feature`` compiles an axon spec into a FeaturePack that ELIO's loader
(`@elio/sdk` ``loadFeaturePack``) accepts and that its runtime can back:

  * each axon agent  -> an ELIO ``agent`` step (multi-turn inner loop; ``with.prompt``
    carries the inline instruction — the loader leaves prose verbatim, only true
    file-refs are resolved),
  * each stage gate  -> an ELIO ``validate`` step (the documented eval-gate
    primitive); the natural-language criteria ride along in ``with.criteria`` so a
    later upgrade can back the gate with an LLM judge without changing the shape,
  * stages/agents are chained by ``edges`` in order.

``from_elio_feature`` is the inverse, so the contract is bidirectional and the same
pipeline round-trips between the two engines (that is what lets TS-ELIO back the
axon ``run_workflow`` contract).

Known v0.1 reduction: axon runs the agents WITHIN a stage in parallel; this mapping
linearizes them into a sequential ELIO chain (ELIO can express intra-stage fan-out
via ``subworkflow``/``batch`` — a later refinement). The stage ORDER and gates are
preserved exactly.
"""
from __future__ import annotations

import re
from typing import Any

from .workflow import WorkflowSpec, parse_spec

API_VERSION = "elio/v1"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "axon-workflow"


def _agent_step_id(stage: str, agent: str) -> str:
    return f"{_slug(stage)}__{_slug(agent)}"


def _gate_step_id(stage: str) -> str:
    return f"{_slug(stage)}__gate"


def to_elio_feature(
    spec: Any,
    *,
    feature_id: str | None = None,
    version: str = "0.1.0",
    owner: str = "axon",
    artifact_kind: str = "workflow-artifact",
    eval_gate: str = "workflow_complete",
) -> dict:
    """Compile an axon workflow spec (dict / YAML string / WorkflowSpec) into an
    ``elio/v1`` FeaturePack dict."""
    wf: WorkflowSpec = spec if isinstance(spec, WorkflowSpec) else parse_spec(spec)

    state: dict[str, Any] = {}
    steps: list[dict] = []
    order: list[str] = []  # step ids in execution order

    for stage in wf.stages:
        for agent in stage.agents:
            sid = _agent_step_id(stage.name, agent.name)
            out_key = f"{sid}__output"
            state[out_key] = None
            steps.append({
                "id": sid,
                "type": "agent",  # ELIO built-in: multi-turn inner loop over ctx.model
                "with": {
                    "prompt": agent.prompt,
                    "system": f"You are '{agent.name}' in the '{stage.name}' stage of: {wf.goal}",
                    "maxTurns": 1,
                },
                "outputs": {"output": f"state.{out_key}"},
            })
            order.append(sid)

        if stage.gate:
            gid = _gate_step_id(stage.name)
            # Gate over the LAST agent's output in this stage (the stage's result).
            last_out = f"{_agent_step_id(stage.name, stage.agents[-1].name)}__output"
            steps.append({
                "id": gid,
                "type": "validate",  # ELIO built-in: doubles as an eval-gate (GateVerdict)
                "with": {
                    "value": f"{{{{state.{last_out}}}}}",
                    "minLength": 1,  # v0.1 structural gate: the stage produced output
                    # The natural-language criteria — preserved for an LLM-judge upgrade.
                    "criteria": stage.gate,
                },
            })
            order.append(gid)

    if not steps:
        raise ValueError("axon spec produced no steps")

    edges = [{"from": order[i], "to": order[i + 1]} for i in range(len(order) - 1)]

    return {
        "apiVersion": API_VERSION,
        "kind": "Feature",
        "metadata": {
            "id": feature_id or _slug(wf.goal),
            "version": version,
            "owner": owner,
            "lifecycle": "draft",
        },
        "feature": {
            "autonomy": "guided",  # agents guided; runtime deterministic (ELIO Inv. 9)
            "artifact": {"kind": artifact_kind, "evalGate": eval_gate},
            "io": {
                "input": {"type": "object", "properties": {"goal": {"type": "string"}}},
                "output": {"type": "object"},
            },
            "graph": {"state": state, "steps": steps, "edges": edges},
        },
    }


def from_elio_feature(pack: dict) -> dict:
    """Inverse of ``to_elio_feature``: recover an axon workflow spec (goal + stages)
    from an ``elio/v1`` FeaturePack dict. Agent steps are regrouped into stages by
    their ``<stage>__<agent>`` id prefix; a ``validate`` step becomes the stage gate.
    """
    if pack.get("apiVersion") != API_VERSION or pack.get("kind") != "Feature":
        raise ValueError("not an elio/v1 Feature pack")
    graph = (pack.get("feature") or {}).get("graph") or {}
    steps = graph.get("steps") or []

    stages: list[dict] = []
    by_name: dict[str, dict] = {}

    def stage_for(name: str) -> dict:
        if name not in by_name:
            st = {"name": name, "agents": [], "gate": None}
            by_name[name] = st
            stages.append(st)
        return by_name[name]

    for step in steps:
        sid = step.get("id", "")
        stype = step.get("type")
        stage_name, _, tail = sid.partition("__")
        if stype == "agent":
            with_ = step.get("with") or {}
            stage_for(stage_name)["agents"].append({"name": tail or "agent", "prompt": with_.get("prompt", "")})
        elif stype == "validate":
            with_ = step.get("with") or {}
            stage_for(stage_name)["gate"] = with_.get("criteria") or "stage produced output"

    # Drop the null gate key when absent, mirroring the compact axon shape.
    for st in stages:
        if st["gate"] is None:
            del st["gate"]

    goal = _degoal(pack)
    return {"goal": goal, "stages": stages}


def _degoal(pack: dict) -> str:
    """Best-effort recover the goal from a system prompt (…of: <goal>) or the id."""
    steps = ((pack.get("feature") or {}).get("graph") or {}).get("steps") or []
    for step in steps:
        sys = ((step.get("with") or {}).get("system") or "")
        m = re.search(r" of: (.+)$", sys)
        if m:
            return m.group(1)
    return (pack.get("metadata") or {}).get("id", "")


def to_yaml(pack: dict) -> str:
    """Serialize a FeaturePack dict to YAML (stable key order preserved)."""
    import yaml

    return yaml.safe_dump(pack, sort_keys=False, width=100, allow_unicode=True)
