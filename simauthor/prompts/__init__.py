"""Prompt loading and rendering utilities shared across the simauthor package.

Templates use Python ``str.format()`` syntax (``{var}``).  Literal braces
in the template must be doubled (``{{`` and ``}}``).

``render_prompt()`` escapes all dynamic values before formatting, so that
Python code, JSON examples, and markdown containing braces pass through safely.
"""

import os

_PROMPTS_DIR = os.path.dirname(__file__)


def load_prompt(*path_parts: str) -> str:
    """Return the contents of *prompts/<path_parts>*."""
    with open(os.path.join(_PROMPTS_DIR, *path_parts)) as f:
        return f.read()


def render_prompt(template: str, **kwargs) -> str:
    """Render *template* with ``str.format(**kwargs)``.

    String values are brace-escaped so that Python code, JSON examples,
    and markdown containing ``{`` / ``}`` pass through safely.  Numeric
    values are passed as-is so that format specs like ``{delta:.4f}`` work.
    """
    safe = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            safe[k] = v.replace("{", "{{").replace("}", "}}")
        else:
            safe[k] = v
    return template.format(**safe)
