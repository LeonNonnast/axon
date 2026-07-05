# 🧠→⚡ Axon

**The agent (neuron) region of the [Brain](https://github.com/LeonNonnast/brain).**
Axon is the *worker that thinks inside an [Arkwen](https://github.com/LeonNonnast/arkwen)
workcell and acts only through the tools the cell provides.* In a neuron the axon is
the output fibre that carries the signal outward and makes something happen — that is
exactly this repo's job.

> An agent is not a magic constructor — it's a loop. Axon owns that loop and keeps it
> honest: `Agent(prompt, context, tools, skills).run()`.

## What Axon must be able to do (the spec)

1. **Accept** a `prompt` (the mission), `context` (materials / injected memory from
   cortex + hippocampus), `tools` (provided by the cell, dynamically), and `skills`.
2. **Run an agent loop** with a **generic LLM provider injected from outside** — Axon
   is provider-agnostic and never sees the real API key (Arkwen's secret broker leases it).
3. **Execute tools only via the cell** — the cell *provides* the tools and *executes*
   them under policy. Axon never touches the OS directly.
4. **Inject skills** — capability packages (instructions/procedures, optionally bundling
   tool names) that change what the agent *knows how to do*.
5. **Emit observable activity** (messages / tool-calls) so the cell can fold it into
   Arkwen's append-only event stream.

## Tool vs Skill

| | **Tool** | **Skill** |
|---|---|---|
| what | one executable function the cell exposes | an injected know-how package |
| effect | what the agent *may do* (and only via the cell) | what the agent *knows how to do* |
| executed | by the **cell**, under policy | n/a — it's context (± bundled tool names) |
| Arkwen vocab | Tools (MCP) | Toolkits (Skills) |

## Framework-agnostic core + adapters

The core (`axon/core.py`, `axon/cell.py`) is framework-free. Engines are **adapters**
of the same `Agent` contract, so the cell only ever sees the Axon contract — never the
framework (Arkwen Invariant 1):

- `axon.adapters.MinimalAgent` — a ~40-line raw tool-use loop (reference, zero deps).
- `axon.adapters.LangGraphAgent` — `create_react_agent` over cell-bound tools
  **(implemented + offline-tested)**: each cell tool becomes a `StructuredTool` that
  routes to `cell.invoke_tool`, so the model can only call announced tools and the
  cell executes them under policy.
- `axon.adapters.CrewAIAgent` — a single-worker CrewAI Crew *(scaffold; multi-agent later)*.

```python
from axon import Cell
from axon.adapters import MinimalAgent   # or LangGraphAgent / CrewAIAgent

cell = Cell()                            # stdio channel to the Arkwen workcell
session = cell.handshake()               # cell announces mission + allowed tools + skills
agent = MinimalAgent(
    session.mission, session.context,
    tools=session.tools, skills=session.skills,   # ← only what the cell provided
    provider=my_provider,                          # ← generic, injected from outside
    cell=cell,
)
agent.run()                              # loop; every tool call executed by the cell
```

## Security posture (read this)

Axon is **untrusted by design.** A tool list is a *guardrail* (the model won't *choose*
a forbidden action) — it is **not a sandbox** (arbitrary framework/dependency code can
still call `open()`, sockets, `os.system`). "Only cell-provided tools" becomes a real
guarantee **only** when the agent process is *confined* (no ambient authority, the cell
channel the single hole in the jar). That confinement — gVisor/Firecracker + seccomp +
default-deny egress — is **Arkwen's job**, not Axon's. Axon cooperates; Arkwen contains.

## The cell channel

Line-delimited JSON over stdio (MCP-like). The cell drives the session:

```
<- {"type":"session","mission":..,"context":..,"tools":[..],"skills":[..]}
-> {"type":"tool_call","id":..,"name":..,"args":{..}}
<- {"type":"tool_result","id":..,"ok":true,"result":..}
-> {"type":"emit","event":{..}}
-> {"type":"finish","status":"completed","output":..}
```

## Dynamic workflows — the "elio loop" as a tool

An agent can decompose a large task into a **multi-stage workflow** and run it via the
`run_workflow` cell tool (`axon/workflow.py`): stages run sequentially, agent nodes
within a stage run in parallel, each stage may declare a **quality gate** that must
PASS to proceed (fail-closed), every node is a confined sub-axon, results are written
**through the cell**, and progress is emitted as events. This is the axon-native
realization of the ELIO outer-loop pattern; the TS ELIO engine can later back the same
tool contract. Verified end-to-end against Ollama (`examples/run_elio_workflow.py`) — a
design→implement→test run produced real, gated code.

Crucially: `run_workflow` is a **trusted** tool (run by the cell, not the agent). The
agent only *requests* a workflow — it never executes scripts or reaches a terminal;
any such capability would itself be a policy-gated cell tool in a confined sub-cell.

## Status

**v0.0.1.** Core contract + cell SDK + a working `MinimalAgent` reference loop, a
**real, offline-tested `LangGraphAgent`** (tool calls provably routed through the cell),
a cooperative reference `LocalCell` (policy + redaction), and a **working dynamic
workflow engine** (the elio loop) — all verified against Ollama. The CrewAI adapter is a
stub. Next: running against a real Arkwen cell (the Go tool-broker + confinement).

## Roadmap

- **R1** — LangGraph adapter ✅ *(done; migrate `create_react_agent` → `langchain.agents.create_agent` before LangGraph V2)*.
- **R2** — CrewAI adapter (bind cell tools → `BaseTool`, single-worker Crew).
- **R3** — Provider examples (Anthropic / OpenAI-compatible) behind `LLMProvider`.
- **R4** — Skill loader (Arkwen Toolkits → injected `Skill`s).
- **R5** — End-to-end against an Arkwen cell + the adversarial confinement test.
