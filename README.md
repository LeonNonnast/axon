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
realization of the ELIO outer-loop pattern.

The tool takes **structured args** (`goal` + `stages:[{name, agents:[{name,prompt}],
gate?}]`), *not* a YAML-string blob — small local models emit nested JSON tool
arguments reliably but choke on a large multi-line string argument. That one change is
what makes the **agent-driven** path work: `examples/run_elio_workflow.py --agent`
gives the main agent *only* `run_workflow` and a skill, and qwen2.5:7b (via Ollama)
authors its own design→implement→test→commit YAML, runs it, **retries on a failed
gate**, and produces real code — end to end, no hand-authored spec.

Crucially: `run_workflow` is a **trusted** tool (run by the cell, not the agent). The
agent only *requests* a workflow — it never executes scripts or reaches a terminal;
any such capability would itself be a policy-gated cell tool in a confined sub-cell.

### ELIO feature.yaml — the same pipeline in ELIO's vocabulary

`axon/elio.py` bridges the contract both ways: `to_elio_feature` compiles an axon
workflow spec into a valid `elio/v1` **Feature** (each agent → an ELIO `agent` node,
each gate → a `validate` eval-gate, chained by edges), and `from_elio_feature` inverts
it. The emitted `feature.yaml` **loads in the real `@elio/sdk` loader** (verified:
metadata/steps/edges validated, a `contentHash` computed, prompts inline and
execution-ready), so the **TypeScript ELIO engine can back the same `run_workflow`
contract**. Convert with `examples/to_elio_feature.py`; a generated example lives in
`examples/elio/`.

## Plugins — cell-driven built-ins (memory injection)

A **plugin** (`axon/plugin.py`) is a capability the *cell* provisions and drives —
not something the agent imports. It has exactly two hooks, one per direction a
capability reaches the agent:

- `provide_tools(register)` — cell tools the agent MAY call on demand (**pull**).
- `before_session(mission, context) -> context'` — the **cell-driven** hook
  (**push**): the cell calls it before the loop and folds the result into the
  agent's context. The agent never asks; the cell injects.

`activate_plugins(cell, plugins, mission)` is the host-side entry point (cell-driven
by construction). This is why a plugin, not "just an MCP server": a standard MCP tool
can only *return* text when the agent calls it — it has no channel to proactively
write into the agent's context. The act of injecting is the host/cell's job; the
plugin is that seam.

The first plugin is **`MemoryPlugin`** (`axon/memory.py`), the built-in over
[cortex](https://github.com/LeonNonnast/cortex):

- **Pull** — `memory_store` / `memory_recall` cell tools (text in / text out, so a
  plain MCP call carries them).
- **Push** — `before_session` asks cortex's router (`memory_route`) whether to prime
  the turn; if yes, the cell injects the recalled text. The SeaKR-style
  "inject when the model is uncertain" signal comes from cortex's own instrumented
  model — the reasoner can't expose it — so the router runs cortex-side and *returns*
  the decision + text; the cell performs the injection. The latent KV round-trip
  stays inside cortex (a serving engine can't ingest an opaque KV blob) — over this
  boundary, memory is always text.

Two backends, one `MemoryBackend` contract: `McpMemoryBackend` (JSON-RPC over stdio
to cortex's `engram-mcp`) and `LocalMemoryBackend` (dependency-free, mirrors cortex's
`EmbeddingRouter`; for tests / offline runs). Verified end-to-end against Ollama
(`examples/run_memory_agent.py`): the agent greeted the user by a name it was given
only through injection, and the cell log shows `plugin_inject` (push) alongside the
agent's own `memory_recall` (pull) — both strategies live at once.

## The real Arkwen cell (Go tool-broker)

`examples`/`LocalCell` execute tools cooperatively in-process. To run against the
**real** Arkwen runtime, `python -m axon.cellworker` speaks the cell channel over
stdio to Arkwen's Go Cell-Shim (`arkwen internal/adapter/axon.go`), which **brokers
every tool call through `WorkcellAPI`** — writes flow through redaction-before-
persistence + the content-addressed store + the append-only event stream; the agent
process has no path to the world except the channel. Drive a full run with
`arkwen run create --worker axon` (set `ARKWEN_AXON_CMD` to this repo's
`<venv>/python -m axon.cellworker`). Verified end-to-end: a real qwen2.5:7b model
driving the Go cell produced `worker.tool_call → tool_result → artifact_written`
events and a `COMPLETED` termination — **no Python LocalCell in the loop**.

## Status

**v0.0.1.** Core contract + cell SDK + a working `MinimalAgent` reference loop, a
**real, offline-tested `LangGraphAgent`** (tool calls provably routed through the cell),
a cooperative reference `LocalCell` (policy + redaction), and a **working dynamic
workflow engine** (the elio loop) — all verified against Ollama. The **agent-driven**
path works (structured `run_workflow` args), a bridge runs axon against the **real
Arkwen Go cell** (`axon.cellworker` ↔ Go tool-broker), and the workflow contract maps
both ways to an **ELIO `feature.yaml`** (loads in the real `@elio/sdk`). A
**cell-driven plugin system** ships with `MemoryPlugin` — cortex-backed memory as a
built-in that the cell provisions (pull tools) and injects (push priming), verified
end-to-end against Ollama. The CrewAI adapter is a stub.

## Roadmap

- **R1** — LangGraph adapter ✅ *(done; migrate `create_react_agent` → `langchain.agents.create_agent` before LangGraph V2)*.
- **R2** — CrewAI adapter (bind cell tools → `BaseTool`, single-worker Crew).
- **R3** — Provider examples (Anthropic / OpenAI-compatible) behind `LLMProvider`.
- **R4** — Skill loader (Arkwen Toolkits → injected `Skill`s).
- **R7** — Plugins (cell-driven built-ins) ✅ *(done: `axon/plugin.py` + `MemoryPlugin` over cortex; pull tools + push injection, verified against Ollama)*. Next: wire `McpMemoryBackend` against a cortex `engram-mcp` that exposes `memory_store` / `memory_recall` / `memory_route` (the cortex-side tool surface), and provision the plugin through Arkwen's Go cell.
- **R5** — End-to-end against an Arkwen cell ✅ *(done: `python -m axon.cellworker` ↔ Arkwen's Go tool-broker; real model → real cell verified)*. Next: the adversarial confinement test (gVisor/Firecracker + seccomp) — Arkwen's isolation layer.
- **R6** — ELIO contract bridge ✅ *(done: `axon/elio.py` compiles a workflow to a valid `elio/v1` Feature and back; loads in `@elio/sdk`)*. Next: a full ELIO **run** of the generated feature (wire ELIO's `agent` node to a provider) so TS-ELIO executes the pipeline end to end.
