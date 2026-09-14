# Model Usage Report — Buy or Wait?

Scope: the final full-dataset run (`python3 code/main.py`, 250 requests, 250 output
rows) and the cold enrichment runs that populated the disk cache it reads.

---

## 1. Headline: the final run costs nothing

```
$ python3 code/main.py
wrote 250 rows to /home/starburst/hackerrank-orchestrate-september26/output.csv
llm_calls=0 llm_tokens=0 est_cost_usd=0.00000
```

Every model call in this system goes through one extraction layer whose results are
content-addressed to disk (`code/enrich/cache/`, 231 files, committed to the repo so
the shipped run needs no credentials). The key is a SHA-256 over
the prompt-contract version, the model id, the system prompt, and the item's own id
and text, so an unchanged dataset is a 100% cache hit. The final full-dataset run
makes **0 API calls and spends 0 tokens** — reproducibly, on any machine with the repo
checked out, with no API key present at all.

Both passes — messages and images — are wired into `code/main.py` and run through
`solver.enrich_dataset()`. Everything below accounts for the **cold** runs that
produced the cache.

## 2. Providers and models

| Role | Provider | Model | Where |
|---|---|---|---|
| Message fact extraction | Groq (OpenAI-compatible endpoint) | `openai/gpt-oss-120b` | `code/enrich/messages.py` |
| Blank-amount recovery from images | Groq | `qwen/qwen3.8-27b` (vision) | `code/enrich/images.py` |

Both are called through `code/enrich/llm.py` — the single call site, which also owns
rate limiting, retries, and the `UsageTracker` that produced the live counters above.
Overridable via the `MODEL` and `VISION_MODEL` environment variables.

## 3. Cold-run usage

The two passes are reported differently because the evidence available for each is
different, and saying so is the point:

- **The image pass is measured.** It was re-run from scratch on 2026-09-14 under
  prompt contract `v2` (see §5), and nothing else touched the API that day, so the
  daily budget counter attributes cleanly to exactly those 16 calls.
- **The message pass is reconstructed.** It was run on 2026-09-13 alongside a day of
  development, so no clean per-run total survives. The cache deliberately stores only
  the *outcome* of an extraction, never token counts — a cache hit must contribute
  zero tokens to this report, and the only way to guarantee that is to never record
  them on the hit path. So each cached entry's exact prompt was rebuilt from the
  dataset with the same `build_prompt` the pipeline uses, and both the prompt and the
  stored `raw_text` response were tokenized with `tiktoken` (`o200k_base`).
  Reconstruction covered 215/215 entries with 0 misses.

| Model | Calls | Input tokens | Output tokens | Total | Basis |
|---|---:|---:|---:|---:|---|
| `openai/gpt-oss-120b` (messages) | 215 | 109,726 | 9,079 | 118,805 | reconstructed |
| `qwen/qwen3.8-27b` (images) | 16 | 30,470 | 292 | 30,762 | **measured** |
| **Total** | **231** | **140,196** | **9,371** | **149,567** | mixed |

Averages:

| Metric | Messages | Images | Combined |
|---|---:|---:|---:|
| Tokens per call | 552.6 | 1,922.6 | 647.5 |
| Input per call | 510.4 | 1,904.4 | 606.9 |
| Output per call | 42.2 | 18.2 | 40.6 |

Per evaluation request (250 requests in `dataset/requests.csv`):

| Metric | Value |
|---|---|
| Model calls per request | 0.92 |
| Total tokens per request | 598 |

The LLM work is per-*message* and per-*image*, not per-request — 215 messages and 16
images are extracted once for the whole dataset. The per-request figures are that
one-time cost divided by 250, not a marginal cost of adding a request.

### What one image actually costs

The measured image pass answers a question the text tokenizer cannot. Of the 30,762
tokens the 16 vision calls really spent, the reconstructed text prompts and responses
account for 7,133. The remaining 23,629 is the images themselves: **~1,477 tokens per
PNG**, for files averaging 334 KB. `IMAGE_TOKEN_ESTIMATE` in `llm.py` is set to 3,000,
so the rate limiter is pacing about twice as conservatively as it needs to. That is
the safe direction for a limiter to be wrong in, and it is left alone.

### Estimated cost

At Groq's published rates for these models (`PRICE_PER_MILLION_TOKENS` in
`code/enrich/llm.py`):

| Model | $/M in | $/M out | Input cost | Output cost | Total |
|---|---:|---:|---:|---:|---:|
| `openai/gpt-oss-120b` | 0.15 | 0.75 | $0.01646 | $0.00681 | **$0.02327** |
| `qwen/qwen3.8-27b` | 0.20 | 0.20 | $0.00609 | $0.00006 | **$0.00615** |
| **Total** | | | $0.02255 | $0.00687 | **$0.02942** |

Per request: **$0.000118**. Final (cached) run: **$0.00**.

## 4. What the message figures do not include

The image row is a real bill. The message row is a floor, and three known gaps push
the true figure up:

1. **Hidden reasoning tokens.** `gpt-oss-120b` is a reasoning model: it spends
   completion tokens on a reasoning trace that never appears in the returned content,
   so it is absent from the cached `raw_text` and cannot be tokenized. Those tokens
   are billed as output. This is the largest gap.
2. **Failed and retried calls.** The cache records only settled outcomes. Calls that
   returned invalid JSON and were retried, 429s, and the `json_validate_failed`
   failures from before `reasoning_effort="low"` was set all cost real tokens and left
   no cache entry.
3. **Chat-template overhead.** Role scaffolding around the system/user messages is not
   in the raw prompt strings.

**The anchor.** `code/enrich/cache/_daily_usage.json` persists real, API-reported
token totals per day. It recorded **189,826 tokens** on 2026-09-13 — the day the
message pass ran, along with that day's development iterations, partial reruns and
failures. The reconstructed 118,805 sits inside it; the difference is the three gaps
above plus that day's throwaway calls, and it cannot be split further after the fact.

A safe upper bound for the whole cold enrichment pass: pricing all 189,826 tokens from
that day at the most expensive rate in play ($0.75/M output) gives **$0.14**. The true
cold-run cost is between $0.029 and $0.15. Either way this is a sub-dollar pipeline,
and the shipped run is free.

**Pricing caveat.** `PRICE_PER_MILLION_TOKENS` is a hardcoded constant transcribed
from Groq's published rates at build time. The system makes no live pricing call by
design, so these dollar figures are stale-able. The `qwen/qwen3.8-27b` entry in
particular was not re-verified; it is ~21% of the estimate.

## 5. Call outcomes

| Outcome | Count |
|---|---:|
| Messages parsed into a valid fact | 213 / 215 |
| Messages rejected as `schema_invalid` | 2 / 215 |
| Images resolved to a validated amount | 16 / 16 |

The 2 rejections are the defense working, not a failure: the model lifted a
human-readable ticket number out of the message body (`Ref payroll EMP-0001`) and
returned it as `target_event_id`. `validate_fact()` checks every returned event id
against the events actually owned by that message's user and dropped it. No
model-returned string reaches the decision path without passing a closed schema first.

**Why the image contract is at `v2`.** The `v1` pass misread `image_02`, an Indian
rent receipt whose "Balance Due: 1,00,000.00" is lakh-grouped and means 100,000. It
was returned as 1,000,000 — a 10x error that, once the amounts were wired into the
forecast, flipped a previously-correct sample row to fully wrong. The system prompt
now states the lakh/crore grouping rule explicitly, `IMAGE_CONTRACT_VERSION` was
bumped to `v2` so no `v1` result can be served from cache, and all 16 images were
re-extracted. `event_1442` now reads 100,000.00; the other 15 were unchanged by the
new prompt. The 16 `v1` files were deleted rather than migrated: a bumped contract
makes them permanently unreachable by construction (different hash), so keeping them
would only have shipped 16 known-stale results alongside the good ones.

## 6. Reproducing this report

```bash
# The cached run — 0 calls, 0 tokens, prints the live UsageTracker totals.
python3 code/main.py

# A cold run of both passes (needs GROQ_API_KEY, spends real tokens):
rm -rf code/enrich/cache && python3 code/main.py

# The image pass also has a standalone CLI, useful for inspecting one extraction:
PYTHONPATH=code python3 -m enrich.images all      # or: ... image_02
```

The token reconstruction in §3 was produced by rebuilding each cached entry's prompt
via `enrich.messages.build_prompt` / `enrich.images.build_prompt` and tokenizing with
`tiktoken`. `tiktoken` is a reporting-only dependency and is deliberately not in
`code/requirements.txt`, since the pipeline itself does not need it.

No API keys, credentials, or endpoints are recorded in this file. Secrets are read
from environment variables only.
