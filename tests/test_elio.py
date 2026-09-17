"""The axon<->ELIO contract bridge (axon/elio.py).

Proves the axon workflow spec compiles into a structurally valid ``elio/v1`` Feature
(the exact shape ELIO's loader validates: metadata.id/version, feature.autonomy in
the allowed set, artifact.kind/evalGate, io.input/output objects, graph.steps with
unique ids + typed nodes, edges referencing existing ids) and round-trips back to the
axon shape. (A companion node check loads the emitted YAML with the real @elio/sdk
loader; see examples/elio/.)
"""
from __future__ import annotations

from axon.elio import from_elio_feature, to_elio_feature, to_yaml
from axon.workflow import parse_spec

SPEC = """
goal: Produce a tiny module with a test.
stages:
  - name: design
    agents:
      - {name: designer, prompt: specify add(a,b)}
  - name: implement
    gate: "code exists"
    agents:
      - {name: coder, prompt: write add.py}
      - {name: helper, prompt: write helpers}
  - name: commit
    gate: "test passes"
    agents:
      - {name: committer, prompt: write the test}
"""

VALID_AUTONOMY = {"static", "guided", "dynamic"}
VALID_SUSPEND = {"blocking", "parked", "timeout", "optional"}


def test_to_elio_feature_is_structurally_valid():
    pack = to_elio_feature(SPEC)
    assert pack["apiVersion"] == "elio/v1" and pack["kind"] == "Feature"

    md = pack["metadata"]
    assert isinstance(md["id"], str) and md["id"]
    assert isinstance(md["version"], str) and md["version"]

    feat = pack["feature"]
    assert feat["autonomy"] in VALID_AUTONOMY
    assert feat["artifact"]["kind"] and feat["artifact"]["evalGate"]
    assert isinstance(feat["io"]["input"], dict) and isinstance(feat["io"]["output"], dict)

    graph = feat["graph"]
    steps = graph["steps"]
    assert steps, "must produce steps"
    ids = [s["id"] for s in steps]
    assert len(ids) == len(set(ids)), "step ids must be unique (ELIO requirement)"
    for s in steps:
        assert s["id"] and s["type"], "each step needs a nonempty id and type"
        if "suspend" in s:
            assert s["suspend"] in VALID_SUSPEND

    # edges must reference existing step ids only (ELIO validates this).
    idset = set(ids)
    for e in graph["edges"]:
        assert e["from"] in idset and e["to"] in idset


def test_agents_become_agent_nodes_gates_become_validate_nodes():
    pack = to_elio_feature(SPEC)
    steps = pack["feature"]["graph"]["steps"]
    kinds = {s["type"] for s in steps}
    assert kinds == {"agent", "validate"}
    # every axon agent -> exactly one ELIO agent node (order preserved)
    wf = parse_spec(SPEC)
    n_agents = sum(len(s.agents) for s in wf.stages)
    n_gates = sum(1 for s in wf.stages if s.gate)
    assert sum(1 for s in steps if s["type"] == "agent") == n_agents
    assert sum(1 for s in steps if s["type"] == "validate") == n_gates
    # agent nodes carry the inline prompt (execution-ready; loader leaves prose verbatim)
    agent = next(s for s in steps if s["type"] == "agent")
    assert agent["with"]["prompt"] and " " in agent["with"]["prompt"]
    # gate criteria are preserved for a later LLM-judge backing
    gate = next(s for s in steps if s["type"] == "validate")
    assert gate["with"]["criteria"]


def test_round_trip_preserves_pipeline():
    pack = to_elio_feature(SPEC)
    back = from_elio_feature(pack)
    orig = parse_spec(SPEC)

    assert [s.name for s in orig.stages] == [s["name"] for s in back["stages"]]
    for o, b in zip(orig.stages, back["stages"]):
        assert [a.name for a in o.agents] == [a["name"] for a in b["agents"]]
        assert (o.gate is not None) == ("gate" in b)
    # and the recovered spec re-parses (it is a valid axon spec again)
    assert parse_spec(back).goal


def test_to_yaml_emits_stable_ordered_yaml():
    pack = to_elio_feature(SPEC)
    text = to_yaml(pack)
    # apiVersion/kind lead; a downstream loader reads them first.
    assert text.splitlines()[0].startswith("apiVersion:")
    assert "kind: Feature" in text
    import yaml
    assert yaml.safe_load(text) == pack  # yaml round-trips losslessly
