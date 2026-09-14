"""Thin, rate-limited wrapper around the chat-completions endpoint.

Every model call in this codebase goes through LLMClient.complete_fact() so that:

1. Token/call accounting for evaluation/usage_report.md lives in exactly one place
   (UsageTracker below), regardless of which module asked for the completion.
2. The provider's rate limits are enforced uniformly - callers never sleep or retry
   themselves.
3. "Invalid JSON, retry once, then mark unparseable" is applied consistently instead
   of being reimplemented per call site.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It has no notion of "financial facts", events, or decisions. complete_fact() hands
back parsed JSON (a dict) or an unparseable marker - nothing here interprets what
that JSON means. Schema validation, closed-enum checks, and cross-referencing a
target_event_id against the user's own events all happen one layer up, in
enrich/messages.py. That separation is deliberate: it's what guarantees an
instruction embedded in a message's text can never reach the decision path even if
the model were tricked into echoing one back - this module only ever returns
key/value data, never something that gets executed.

RATE LIMITS (provider caps for this key/model tier)
----------------------------------------------------
- 30 requests/minute
- 8,000 tokens/minute
- 200,000 tokens/day
- a flat 2.5s floor between calls, regardless of the above

The flat floor alone caps us at 24 calls/min (< 30/min), so the per-minute request
cap is never the binding one. The per-minute TOKEN cap is: at ~400-500 tokens/call
for these prompts, 8,000/min allows only ~17-20 calls/min, which is BELOW what the
2.5s floor alone would allow - so the adaptive token-window wait in
_wait_for_token_headroom is the one that actually binds in practice, not the flat
sleep. Both are enforced; 429 handling (honoring Retry-After) is the backstop for
whatever slips through.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI, RateLimitError

REQUESTS_PER_MINUTE = 30
TOKENS_PER_MINUTE = 8_000
TOKENS_PER_DAY = 200_000
MIN_SECONDS_BETWEEN_CALLS = 2.5

# Deliberately generous stand-in for what one bill/payslip screenshot costs in vision
# tokens. Only the rate limiter reads it; the real usage from the response replaces
# it in the rolling window as soon as the call returns.
IMAGE_TOKEN_ESTIMATE = 3_000
MAX_RATE_LIMIT_RETRIES = 6
DEFAULT_RETRY_AFTER_SECONDS = 5.0

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"


def _load_dotenv(path: Path = DOTENV_PATH) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, without a dependency.

    Real environment variables always win - this only fills in gaps, so a shell that
    already exported GROQ_APIKEY (e.g. in CI) is never overridden by a stale .env.
    Secrets still come from the environment either way (AGENTS.md 6.4); this just
    saves every entry point from having to `source .env` by hand.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# USD per 1,000,000 tokens. Filled in from Groq's published rates for these models at
# the time this was written - CONFIRM against https://groq.com/pricing before trusting
# usage_report.md's cost estimate for a real submission; this file cannot look prices
# up live (no market-data calls), so it is a best-effort constant, not a fetched fact.
PRICE_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.20},
}

STATE_DIR = Path(__file__).resolve().parent / "cache"
DAILY_STATE_PATH = STATE_DIR / "_daily_usage.json"


class DailyBudgetExceeded(RuntimeError):
    """Raised when the next call would push a model over its 200K tokens/day cap."""


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str) -> float:
        price = PRICE_PER_MILLION_TOKENS.get(model)
        if price is None:
            return 0.0
        return (
            self.input_tokens * price["input"] + self.output_tokens * price["output"]
        ) / 1_000_000


@dataclass
class UsageTracker:
    """Accumulates calls/tokens per model for the lifetime of one process run.

    Cache hits must never touch this - see enrich/cache.py's module docstring for why.
    """

    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        usage = self.by_model.setdefault(model, ModelUsage())
        usage.calls += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens

    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.by_model.values())

    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_model.values())

    def total_cost_usd(self) -> float:
        return sum(u.cost_usd(model) for model, u in self.by_model.items())


@dataclass(frozen=True, slots=True)
class Completion:
    """Result of one complete_fact() call - well-formed JSON, or an explicit failure.

    parsed is None exactly when parse_error is set: the raw text was not valid JSON
    even after one retry. Callers record that as the message being unparseable rather
    than crashing or silently dropping it.
    """

    parsed: dict[str, Any] | None
    raw_text: str
    parse_error: str | None
    attempts: int


def _retry_after_seconds(error: RateLimitError) -> float:
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    try:
        return max(0.0, float(header))
    except ValueError:
        return DEFAULT_RETRY_AFTER_SECONDS


def _estimate_tokens(*texts: str) -> int:
    # Rough, deliberately generous (chars/4) pre-call estimate used only to decide
    # whether to wait for token headroom before we know the real usage. The real
    # usage from the response replaces this estimate in the rolling window
    # immediately after the call returns.
    return max(1, sum(len(t) for t in texts) // 4)


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = GROQ_BASE_URL,
        state_path: Path = DAILY_STATE_PATH,
    ) -> None:
        # Checked at the first real call, not here. Every result this client would
        # produce is cached to disk, so a fully-cached run makes no call at all and
        # must not require a key just to construct the object - otherwise the
        # zero-cost rerun is only zero-cost for whoever already has credentials.
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._base_url = base_url
        self._openai: OpenAI | None = None
        self.usage = UsageTracker()
        self._last_call_monotonic: float | None = None
        self._minute_window: deque[tuple[float, int]] = deque()
        self._state_path = state_path

    def _require_client(self) -> OpenAI:
        if self._openai is None:
            if not self._api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Secrets are read from environment "
                    "variables only (AGENTS.md 6.4) - put it in .env, never in code. "
                    "A key is needed only for a cold run; an intact enrich/cache/ "
                    "never reaches this point."
                )
            self._openai = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._openai

    # -- pacing -----------------------------------------------------------------

    def _sleep_for_flat_floor(self) -> None:
        if self._last_call_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_call_monotonic
        remaining = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _prune_window(self) -> int:
        cutoff = time.monotonic() - 60
        while self._minute_window and self._minute_window[0][0] < cutoff:
            self._minute_window.popleft()
        return sum(tokens for _, tokens in self._minute_window)

    def _wait_for_token_headroom(self, estimated_tokens: int) -> None:
        while True:
            used = self._prune_window()
            if used + estimated_tokens <= TOKENS_PER_MINUTE:
                return
            oldest_ts = self._minute_window[0][0]
            time.sleep(max(0.1, 60 - (time.monotonic() - oldest_ts)))

    # -- daily budget, persisted across process restarts on the same UTC day -----

    def _load_daily_state(self) -> dict[str, Any]:
        today = date.today().isoformat()
        if not self._state_path.exists():
            return {"date": today, "tokens": 0}
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"date": today, "tokens": 0}
        if state.get("date") != today:
            return {"date": today, "tokens": 0}
        return state

    def _today_tokens(self) -> int:
        return int(self._load_daily_state().get("tokens", 0))

    def _record_daily(self, tokens: int) -> None:
        state = self._load_daily_state()
        state["tokens"] = int(state.get("tokens", 0)) + tokens
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self._state_path)

    # -- the one call site --------------------------------------------------

    def _call_once(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int,
        reasoning_effort: str | None,
        image_data_url: str | None = None,
        extra_token_estimate: int = 0,
    ) -> str:
        # An image contributes tokens that no amount of measuring the TEXT can see,
        # so the caller passes an estimate for it; without that the rolling
        # per-minute window would under-count every vision call and pace into 429s.
        estimated = _estimate_tokens(system, user) + max_output_tokens + extra_token_estimate
        if self._today_tokens() + estimated > TOKENS_PER_DAY:
            raise DailyBudgetExceeded(
                f"{model}: today's usage plus this call's estimate would exceed the "
                f"{TOKENS_PER_DAY}-token/day cap"
            )

        self._sleep_for_flat_floor()
        self._wait_for_token_headroom(estimated)

        extra: dict[str, Any] = {}
        if reasoning_effort is not None:
            # gpt-oss-120b (and other reasoning models) spend completion tokens on a
            # hidden reasoning trace before the visible JSON. At the default effort
            # that trace alone can exceed max_output_tokens, truncating the JSON
            # mid-object and failing Groq's own json_object validation (observed:
            # error code json_validate_failed with an empty failed_generation).
            # This is a plain closed-vocabulary extraction task, so "low" is enough
            # and keeps almost the whole budget for the answer itself.
            extra["reasoning_effort"] = reasoning_effort

        user_content: Any = user
        if image_data_url is not None:
            user_content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]

        client = self._require_client()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_output_tokens,
                    **extra,
                )
            except RateLimitError as error:
                self._last_call_monotonic = time.monotonic()
                if attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(_retry_after_seconds(error))
                continue
            except BadRequestError:
                # gpt-oss's hidden reasoning trace can exhaust max_output_tokens before
                # any JSON is emitted; Groq reports that as a 400 (json_validate_failed),
                # not as truncated content. Surface it as an empty completion so the
                # normal "invalid JSON, retry once, then unparseable" path in
                # complete_fact() handles it - one bad message must not crash the batch.
                self._last_call_monotonic = time.monotonic()
                return ""
            break
        self._last_call_monotonic = time.monotonic()

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else estimated - max_output_tokens
        output_tokens = usage.completion_tokens if usage else 0
        self._minute_window.append((time.monotonic(), input_tokens + output_tokens))
        self.usage.record(model, input_tokens, output_tokens)
        self._record_daily(input_tokens + output_tokens)

        return response.choices[0].message.content or ""

    def complete_fact(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int = 300,
        reasoning_effort: str | None = "low",
    ) -> Completion:
        """Text-only structured extraction."""
        return self._complete_json(
            model=model,
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    def complete_vision(
        self,
        *,
        model: str,
        system: str,
        user: str,
        image_data_url: str,
        max_output_tokens: int = 200,
        reasoning_effort: str | None = None,
        image_token_estimate: int = IMAGE_TOKEN_ESTIMATE,
    ) -> Completion:
        """Structured extraction from one image, same JSON contract as complete_fact."""
        return self._complete_json(
            model=model,
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            image_data_url=image_data_url,
            extra_token_estimate=image_token_estimate,
        )

    def _complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int,
        reasoning_effort: str | None,
        image_data_url: str | None = None,
        extra_token_estimate: int = 0,
    ) -> Completion:
        """Call the model and return well-formed JSON, retrying invalid JSON once.

        This is the ONLY retry policy for malformed output. A RateLimitError is a
        transport failure and is retried transparently inside _call_once regardless
        of this method's attempt count; this loop is specifically about the model
        returning text that doesn't parse as JSON.
        """
        last_error = ""
        last_raw = ""
        for attempt in (1, 2):
            raw = self._call_once(
                model=model,
                system=system,
                user=user,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                image_data_url=image_data_url,
                extra_token_estimate=extra_token_estimate,
            )
            last_raw = raw
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                last_error = str(error)
                continue
            if not isinstance(parsed, dict):
                last_error = f"expected a JSON object, got {type(parsed).__name__}"
                continue
            return Completion(parsed=parsed, raw_text=raw, parse_error=None, attempts=attempt)
        return Completion(parsed=None, raw_text=last_raw, parse_error=last_error, attempts=2)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
