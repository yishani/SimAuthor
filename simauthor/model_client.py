"""
model_client.py — Thin LLM abstraction for the SimAuthor framework
=================================================================
All LLM calls go through one small interface with a single retry layer:

    client.generate_text(prompt, temperature=..., max_tokens=...,
                         enable_search=False, image_paths=None) -> str
    client.generate_code(prompt, ...) -> str   # extracted ```python block

Backends (chosen from the model name by :func:`create_model_client`):

  * ``gemini-*``      Google Gemini via ``google-genai`` (paper default,
                      ``gemini-3.1-pro-preview``).  Key: ``GEMINI_API_KEY``.
  * ``mock``          Deterministic offline backend for tests and dry runs.
  * anything else     Any OpenAI-compatible chat endpoint (DeepSeek, OpenAI,
                      vLLM, OpenRouter, ...).  Key: ``OPENAI_API_KEY`` and
                      optional ``OPENAI_BASE_URL``; ``deepseek-*`` models also
                      accept ``DEEPSEEK_API_KEY`` and default to
                      ``https://api.deepseek.com``.

Keys are read from the explicit argument, then a ``.env`` file in the
working directory or the repository root, then the environment.

Retry policy: only transient failures (408/429/5xx, timeouts, connection
errors) are retried, with jittered exponential backoff; anything else fails
immediately.  When retries are exhausted :class:`RetryExhaustedError` is
raised so the orchestrator can persist state and resume later.
"""

import base64
import os
import random
import time
from typing import Optional

DEFAULT_MODEL = "gemini-3.1-pro-preview"


class RetryExhaustedError(RuntimeError):
    """All retry attempts exhausted on a transient failure."""


_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 8
_INITIAL_DELAY = 5.0
_MAX_DELAY = 180.0
_BACKOFF = 2.0
_JITTER = 0.25
_REQUEST_TIMEOUT_S = 600


def _is_retryable(error: Exception) -> bool:
    """Return True if *error* represents a transient server/network condition."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUSES:
        return True
    msg = str(error)
    if any(str(code) in msg for code in _RETRYABLE_STATUSES):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return "timeout" in msg.lower() or "connection" in msg.lower()


def _read_env_file(name: str) -> Optional[str]:
    """Look up ``name=value`` in ``./.env`` or ``<repo root>/.env``."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(os.getcwd(), ".env"),
                 os.path.join(repo_root, ".env")):
        try:
            with open(path) as fh:
                for line in fh:
                    key, sep, value = line.strip().partition("=")
                    if sep and key.strip() == name:
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
        except OSError:
            continue
    return None


def _get_key(*names: str) -> Optional[str]:
    for name in names:
        value = _read_env_file(name) or os.environ.get(name)
        if value:
            return value
    return None


class _BaseClient:
    """Shared retry loop and code extraction; backends implement ``_request``."""

    model_name: str = ""
    supports_images: bool = True

    def generate_text(self,
                      prompt: str,
                      *,
                      temperature: float = 0.3,
                      max_tokens: int = 4000,
                      enable_search: bool = False,
                      image_paths: Optional[list] = None,
                      ) -> str:
        return self._call(prompt, temperature=temperature,
                          max_tokens=max_tokens, enable_search=enable_search,
                          code_execution=False,
                          image_paths=image_paths or None)

    def generate_code(self,
                      prompt: str,
                      *,
                      temperature: float = 0.3,
                      max_tokens: int = 65536,
                      enable_search: bool = False,
                      ) -> str:
        text = self._call(prompt, temperature=temperature,
                          max_tokens=max_tokens, enable_search=enable_search,
                          code_execution=True, image_paths=None)
        from .utils import extract_code_block
        code, _ = extract_code_block(text)
        if not code or len(code) < 200:
            raise RuntimeError(
                "generate_code: response contained no extractable "
                "```python fenced code block with >= 200 characters."
            )
        return code

    def _request(self, prompt, *, temperature, max_tokens, enable_search,
                 code_execution, image_paths) -> str:
        raise NotImplementedError

    def _call(self, prompt, **kw) -> str:
        t0 = time.time()
        last_error = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._request(prompt, **kw)
            except Exception as e:
                last_error = e
                if not _is_retryable(e):
                    raise
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                delay = min(_INITIAL_DELAY * (_BACKOFF ** attempt), _MAX_DELAY)
                wait = delay * (1.0 + random.uniform(-_JITTER, _JITTER))
                print(f"  [LLM] transient error (attempt "
                      f"{attempt+1}/{_MAX_ATTEMPTS}): {e}.  "
                      f"Retrying in {wait:.0f}s...")
                time.sleep(wait)
        raise RetryExhaustedError(
            f"LLM call failed after {_MAX_ATTEMPTS} attempts "
            f"({time.time() - t0:.0f}s total).  Last error: {last_error}"
        )


# ── Gemini ────────────────────────────────────────────────────────────────

class GeminiClient(_BaseClient):
    """Google Gemini via the ``google-genai`` SDK (the backend used in the paper)."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or _get_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Gemini API key. Set GEMINI_API_KEY (environment or .env).")
        self._client = None
        self._types = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai.types import HttpRetryOptions, HttpOptions
            http = HttpOptions(retry_options=HttpRetryOptions(attempts=1),
                               timeout=_REQUEST_TIMEOUT_S * 1000)
            self._client = genai.Client(api_key=self.api_key, http_options=http)
        return self._client

    @property
    def types(self):
        if self._types is None:
            from google.genai import types
            self._types = types
        return self._types

    def _request(self, prompt, *, temperature, max_tokens, enable_search,
                 code_execution, image_paths) -> str:
        t = self.types
        tools = []
        if code_execution:
            tools.append(t.Tool(code_execution=t.ToolCodeExecution()))
        if enable_search:
            tools.append(t.Tool(google_search=t.GoogleSearch()))
        if image_paths:
            contents = [t.Part.from_text(text=prompt)]
            for path in image_paths:
                with open(path, "rb") as f:
                    contents.append(t.Part.from_bytes(data=f.read(),
                                                      mime_type="image/png"))
        else:
            contents = prompt
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=t.GenerateContentConfig(
                tools=tools or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return self._extract_text(resp)

    @staticmethod
    def _extract_text(resp) -> str:
        try:
            return resp.text.strip()
        except (ValueError, AttributeError):
            pass
        try:
            parts = resp.candidates[0].content.parts
            texts = [p.text for p in parts if getattr(p, "text", None)]
            return "\n".join(texts).strip()
        except Exception:
            return ""


# ── OpenAI-compatible ─────────────────────────────────────────────────────

class OpenAICompatibleClient(_BaseClient):
    """Any OpenAI-compatible chat-completions endpoint.

    Gemini-only tools (code execution, Google Search grounding) are not
    available here; ``enable_search`` is ignored with a one-time warning.
    Set ``SIMAUTHOR_NO_IMAGES=1`` for text-only models; visual feedback is
    then dropped from the prompt.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model_name = model_name
        is_deepseek = model_name.startswith("deepseek")
        names = (("DEEPSEEK_API_KEY", "OPENAI_API_KEY") if is_deepseek
                 else ("OPENAI_API_KEY",))
        self.api_key = api_key or _get_key(*names)
        if not self.api_key:
            raise ValueError(f"No API key for {model_name}. Set {' or '.join(names)}.")
        self.base_url = (base_url or _get_key("OPENAI_BASE_URL")
                         or ("https://api.deepseek.com" if is_deepseek else None))
        self.supports_images = not (_get_key("SIMAUTHOR_NO_IMAGES") or is_deepseek)
        self._client = None
        self._warned_search = False

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=_REQUEST_TIMEOUT_S, max_retries=0)
        return self._client

    def _request(self, prompt, *, temperature, max_tokens, enable_search,
                 code_execution, image_paths) -> str:
        if enable_search and not self._warned_search:
            print("  [LLM] --enable-search is Gemini-only; ignored.")
            self._warned_search = True
        if image_paths and self.supports_images:
            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}})
        else:
            content = prompt
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ── Mock ──────────────────────────────────────────────────────────────────

class MockClient(_BaseClient):
    """Offline backend: returns a fixed blueprint, a tiny valid simulator and a
    mechanism JSON.  Used by the test-suite and ``simauthor run --model mock``
    to exercise the whole pipeline without network access."""

    model_name = "mock"

    def __init__(self, *_, **__):
        self.calls = 0

    def _request(self, prompt, **_) -> str:
        from . import _mock_responses as mr
        self.calls += 1
        return mr.respond(prompt, self.calls)


# ── Factory ───────────────────────────────────────────────────────────────

def create_model_client(model_name: str = DEFAULT_MODEL,
                        api_key: Optional[str] = None):
    """Return a client for *model_name* (see module docstring)."""
    if model_name == "mock":
        return MockClient()
    if model_name.startswith("gemini"):
        return GeminiClient(model_name, api_key=api_key)
    return OpenAICompatibleClient(model_name, api_key=api_key)


# Backwards-compatible name used throughout the codebase and the paper scripts.
ModelClient = create_model_client
