"""
model_client.py — Thin LLM abstraction for the SimAuthor framework
=================================================================
Single module for all LLM calls with one controlled retry layer.

Retry policy:
  - Retries only transient failures (429, 5xx, connection errors).
  - Fails immediately on non-transient errors (401, 403, 400).
  - Max 4 attempts with jittered exponential backoff, ~2 min total budget.
  - SDK internal retry is disabled via http_options.
  - A per-request HTTP timeout is set so a stalled connection raises instead of
    blocking forever (see _REQUEST_TIMEOUT_MS).
"""

import os
import random
import time
from typing import Optional


class RetryExhaustedError(RuntimeError):
    """All retry attempts exhausted on a transient failure.

    The caller should persist state and allow later resume.
    """
    pass


_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# Previously _MAX_ATTEMPTS=4/_MAX_DELAY=60 (~75s max backoff); 7 concurrent
# search runs all died when a gemini demand spike (503 UNAVAILABLE) outlasted the
# 4 quick retries (total 144-216s). Ramped to 8 attempts / 180s cap (~11min of
# backoff) so transient demand spikes ride out instead of killing the run. The
# orchestrator saves state on RetryExhaustedError, so worst case is a later resume.
_MAX_ATTEMPTS = 8
_INITIAL_DELAY = 5.0
_MAX_DELAY = 180.0
_BACKOFF = 2.0
_JITTER = 0.25

# A request timeout was previously unset, so a stalled connection
# blocked inside generate_content indefinitely and raised nothing — the retry
# layer below never fired. A prior long-running pilot stalled inside a request
# until the scheduler wall limit killed the job, losing the whole allocation.
# 10 min is ~7x the observed ~85s for a 28k-char refiner call, so it does not
# disturb normal calls; a genuine stall now raises and is retried as transient.
_REQUEST_TIMEOUT_MS = 600_000


def _is_retryable(error: Exception) -> bool:
    """Return True if *error* represents a transient server/network condition."""
    msg = str(error)
    # Google API errors embed status codes in the message.
    for code in _RETRYABLE_STATUSES:
        if str(code) in msg:
            return True
    # Connection/timeout errors
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if "timeout" in msg.lower() or "connection" in msg.lower():
        return True
    return False


def _load_env_key() -> Optional[str]:
    """Try to load ``GEMINI_API_KEY`` from ``.env`` files near the project root.

    Searches (in order):
      1. ``<module_dir>/../.env``  — SimAuthor project root
      2. ``./.env``                — current working directory
      3. ``<module_dir>/../../../.env`` — legacy simulators root

    Returns the key value or ``None``.
    """
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    _CANDIDATES = [
        os.path.join(_MODULE_DIR, "..", ".env"),             # SimAuthor/.env
        os.path.join(os.getcwd(), ".env"),                   # cwd/.env
        os.path.join(_MODULE_DIR, "..", "..", "..", ".env"), # simulators/.env
    ]
    for path in _CANDIDATES:
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") or \
                       line.startswith("GEMINI_API_KEY "):
                        value = line.split("=", 1)[-1].strip()
                        if value:
                            return value
        except (OSError, IOError):
            continue
    return None


class ModelClient:
    """Thin wrapper around the Gemini SDK."""

    def __init__(self,
                 model_name: str = "gemini-3.1-pro-preview",
                 api_key: Optional[str] = None):
        self.model_name = model_name
        api_key = (api_key
                   or _load_env_key()
                   or os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError("No API key. Set GEMINI_API_KEY or pass api_key.")
        self.api_key = api_key
        self._client = None
        self._types = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            # Disable SDK internal retry — we handle retries explicitly.
            from google.genai.types import HttpRetryOptions, HttpOptions
            retry = HttpRetryOptions(attempts=1)  # no SDK retry
            http = HttpOptions(retry_options=retry,
                               timeout=_REQUEST_TIMEOUT_MS)
            self._client = genai.Client(
                api_key=self.api_key, http_options=http,
            )
        return self._client

    @property
    def types(self):
        if self._types is None:
            from google.genai import types
            self._types = types
        return self._types

    # ── Public API ────────────────────────────────────────────────────────

    def generate_text(self,
                      prompt: str,
                      *,
                      temperature: float = 0.3,
                      max_tokens: int = 4000,
                      tools: Optional[list] = None,
                      image_paths: Optional[list] = None,
                      ) -> str:
        # Optional multimodal route. When image_paths is empty/None, contents
        # stays a plain string — identical to the historical text-only path.
        if image_paths:
            parts = [self.types.Part.from_text(text=prompt)]
            for path in image_paths:
                with open(path, "rb") as f:
                    parts.append(self.types.Part.from_bytes(
                        data=f.read(), mime_type="image/png"))
            contents = parts
        else:
            contents = prompt
        result = self._call(
            contents=contents, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        )
        return result["text"]

    def generate_code(self,
                      prompt: str,
                      *,
                      temperature: float = 0.3,
                      max_tokens: int = 65536,
                      enable_search: bool = False,
                      ) -> str:
        tools = [self.types.Tool(
            code_execution=self.types.ToolCodeExecution(),
        )]
        if enable_search:
            tools.append(self.types.Tool(
                google_search=self.types.GoogleSearch(),
            ))
        result = self._call(
            contents=prompt, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        )
        from .utils import extract_code_block
        code, _ = extract_code_block(result["text"])
        if not code or len(code) < 200:
            raise RuntimeError(
                "generate_code: response contained no extractable "
                "```python fenced code block with >= 200 characters."
            )
        return code

    # ── Internal ──────────────────────────────────────────────────────────

    def _call(self, contents, tools, temperature, max_tokens) -> dict:
        """Call the LLM with jittered exponential-backoff retry.

        Retries only transient failures (429, 5xx, timeouts).
        Fails immediately on non-transient errors (401, 403, 400).

        Returns ``{"text": str, "attempts": int, "duration_s": float}``.
        """
        t0 = time.time()
        last_error = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self.types.GenerateContentConfig(
                        tools=tools or None,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return {
                    "text": self._extract_text(resp),
                    "attempts": attempt + 1,
                    "duration_s": round(time.time() - t0, 1),
                }
            except Exception as e:
                last_error = e
                if not _is_retryable(e):
                    raise  # non-transient — fail immediately

                if attempt == _MAX_ATTEMPTS - 1:
                    break  # exhausted — raise below

                delay = min(
                    _INITIAL_DELAY * (_BACKOFF ** attempt),
                    _MAX_DELAY,
                )
                jitter = 1.0 + random.uniform(-_JITTER, _JITTER)
                wait = delay * jitter
                print(f"  [LLM] transient error (attempt "
                      f"{attempt+1}/{_MAX_ATTEMPTS}): {e}.  "
                      f"Retrying in {wait:.0f}s...")
                time.sleep(wait)

        elapsed = round(time.time() - t0, 0)
        raise RetryExhaustedError(
            f"LLM call failed after {_MAX_ATTEMPTS} attempts "
            f"({elapsed}s total).  Last error: {last_error}"
        )

    @staticmethod
    def _extract_text(resp) -> str:
        try:
            return resp.text.strip()
        except (ValueError, AttributeError):
            pass
        try:
            parts = resp.candidates[0].content.parts
            texts = [p.text for p in parts if hasattr(p, "text") and p.text]
            if texts:
                return "\n".join(texts).strip()
        except Exception:
            pass
        return ""
