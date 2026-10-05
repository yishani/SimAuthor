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


def load_evaluator(path: str, **kwargs):
    """Instantiate a user-written evaluator from a Python file.

    The file must define one subclass of :class:`SearchFeedbackAgent`.  Its
    ``agent_id`` (or the file name) becomes the key recorded in ``tree.json``.
    See ``examples/custom_task/my_evaluator.py`` for a template.
    """
    import importlib.util
    import inspect
    from pathlib import Path
    from .base import SearchFeedbackAgent
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"user_evaluator_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import evaluator file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, SearchFeedbackAgent) and c is not SearchFeedbackAgent
               and c.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"{path} must define exactly one SearchFeedbackAgent subclass, "
                         f"found {[c.__name__ for c in classes]}")
    agent = classes[0](**kwargs)
    agent.agent_id = classes[0].agent_id or path.stem
    return agent
