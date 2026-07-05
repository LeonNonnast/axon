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

## Status

**v0.0.1.** Core contract + cell SDK + a working `MinimalAgent` reference loop, and a
**real, offline-tested `LangGraphAgent`** (tool calls provably routed through the cell —
see `tests/test_langgraph_agent.py`). The CrewAI adapter is a stub with the exact binding
it needs. Next: a concrete provider example and running against a real Arkwen cell.

## Roadmap

- **R1** — LangGraph adapter ✅ *(done; migrate `create_react_agent` → `langchain.agents.create_agent` before LangGraph V2)*.
- **R2** — CrewAI adapter (bind cell tools → `BaseTool`, single-worker Crew).
- **R3** — Provider examples (Anthropic / OpenAI-compatible) behind `LLMProvider`.
- **R4** — Skill loader (Arkwen Toolkits → injected `Skill`s).
- **R5** — End-to-end against an Arkwen cell + the adversarial confinement test.
