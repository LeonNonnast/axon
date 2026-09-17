"""Axon plugins — cell-driven capability packages the CELL provisions and drives.

A plugin is a *built-in* the cell activates on the agent's behalf — NOT something
the agent imports or opts into. It exposes the two, and only two, ways a capability
reaches an agent, mirroring the two directions memory (and anything like it) flows:

- ``provide_tools`` — register cell tools the agent MAY call on demand (**pull**).
  These execute through the cell (``Cell.invoke_tool``) under policy, exactly like
  any other tool, so they are perfectly representable over a plain MCP tool call.
- ``before_session`` — the **cell-driven** hook (**push**): the cell calls it before
  the agent loop starts and folds whatever it returns into the agent's *context*.
  The agent never asks for this — the cell decides, and injects.

Why a plugin and not "just point the agent at an MCP server": a standard MCP tool
can only RETURN text when the agent chooses to call it (``tools/call`` ->
``{content:[{text}]}`` and nothing more). It has no channel to *proactively* write
into the agent's context or influence its state. The act of injecting — writing the
recalled material into the context before the turn — is inherently the host/cell's
job. A plugin is that host-side seam. See ``docs`` in the cortex repo
(memory-injection): the MCP surface covers pull; the cell covers push.

``activate_plugins`` is the host-side entry point: the cell calls it, so activation
is cell-driven by construction. It is deliberately framework-agnostic — it only
touches ``cell.register`` (tool provisioning) and ``cell.emit`` (observability) if
they exist, so it works with :class:`axon.localcell.LocalCell` today and any future
host that mirrors that surface. Tools provisioned by the real Arkwen Go cell arrive
through the session handshake instead, so ``provide_tools`` is skipped when the cell
exposes no ``register``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class Plugin(Protocol):
    """A cell-driven built-in. ``name`` labels it in the event log."""

    name: str

    def provide_tools(self, register: Callable[..., None]) -> None:
        """Register the cell tools this plugin backs (the pull surface). ``register``
        is the cell's own ``register(tool, impl)`` — tools execute in-cell, so their
        ``impl`` takes ``(cell, **args)`` like any other cell tool."""
        ...

    def before_session(self, mission: str, context: str = "") -> str:
        """Cell-driven injection (the push surface): return the context the agent
        should start with. Return ``context`` unchanged to inject nothing."""
        ...


def activate_plugins(cell, plugins: Iterable[Plugin], mission: str, context: str = "") -> str:
    """Provision + prime every plugin, host-side, and return the composed context.

    The cell calls this *before* constructing the agent, so injection is cell-driven:
    tools are registered on the cell (if it provisions tools locally), then each
    plugin's ``before_session`` folds its material into the context in order. A
    ``plugin_inject`` event is emitted for each plugin that changed the context, so
    the injection shows up in the cell's append-only log like any other cell action.
    """
    plugins = list(plugins)
    register = getattr(cell, "register", None)
    if callable(register):
        for p in plugins:
            p.provide_tools(register)
    emit = getattr(cell, "emit", None)
    for p in plugins:
        composed = p.before_session(mission, context)
        if composed != context and callable(emit):
            emit({"kind": "plugin_inject", "plugin": getattr(p, "name", "?")})
        context = composed
    return context
