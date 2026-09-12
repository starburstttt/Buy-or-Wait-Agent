"""Content-addressed cache for message fact-extraction results.

One JSON file per message, named by a hash of everything that determines the model's
output for that message: the model id, a prompt-contract version (bumped whenever the
schema or prompt wording changes so stale results can never masquerade as fresh ones),
and the message's own id and text. Same content in -> same hash -> same file, so
reruns over an unchanged dataset never call the API again. This is what makes "cache
on a content hash so reruns cost zero tokens" true rather than aspirational.

The cache stores the OUTCOME of extraction (parsed fact, no_fact, unparseable, or
schema_invalid), not raw usage/token counts - a cache hit must contribute zero tokens
to the run's usage report, and the only way to guarantee that is to never even look at
token counts on the hit path.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Bump this whenever the prompt text, the fact schema, or the validation rules in
# messages.py change in a way that could change the answer for the SAME message text.
# Old cache files simply become permanently unreachable (different hash), which is
# safer than trying to migrate them.
PROMPT_CONTRACT_VERSION = "v1"


def content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def get(key: str) -> dict[str, Any] | None:
    path = _path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupted cache file must not crash the run - treat it as a miss and let
        # the caller regenerate it.
        return None


def set(key: str, value: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
