"""
registry.py — Search Feedback Agent Registry
==============================================
Modality-keyed registry of concrete feedback instances.

Agents self-register via ``register_feedback(modality, key, class)``
when their module is imported.  The outer experiment runner selects
which agents to instantiate via ``create_feedback()``.
"""

from typing import Optional as _Optional

_registry: dict[str, dict[str, type]] = {}


def register_feedback(modality: str, agent_key: str, cls: type) -> None:
    _registry.setdefault(modality, {})[agent_key] = cls


def get_feedback_class(modality: str, agent_key: str) -> type:
    try:
        return _registry[modality][agent_key]
    except KeyError:
        available = {m: list(agents.keys())
                     for m, agents in _registry.items()}
        raise ValueError(
            f"No agent registered for modality={modality!r}, "
            f"key={agent_key!r}.  Available: {available}"
        )


def create_feedback(modality: str, agent_key: str, **kwargs):
    """Factory: instantiate a SearchFeedbackAgent by modality and key."""
    from .base import SearchFeedbackAgent
    cls = get_feedback_class(modality, agent_key)
    agent = cls(**kwargs)
    agent.agent_id = agent_key
    return agent


def list_registered(modality: _Optional[str] = None) -> dict:
    if modality:
        return dict(_registry.get(modality, {}))
    return {m: dict(agents) for m, agents in _registry.items()}
