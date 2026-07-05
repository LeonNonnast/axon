"""Axon adapters — each implements the `Agent` contract on a different engine.

minimal   : framework-free raw tool-use loop (reference; fewest deps)
langgraph : LangGraph create_react_agent (needs axon[langgraph])
crewai    : CrewAI single-worker Crew (needs axon[crewai])
"""
from .frameworks import CrewAIAgent, LangGraphAgent
from .minimal import MinimalAgent

__all__ = ["MinimalAgent", "LangGraphAgent", "CrewAIAgent"]
