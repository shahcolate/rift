"""Shared text utilities used by multiple Rift subcommands.

The originals lived in ``discovery.py`` (the only consumer at the
time). Moving them here lets ``bisect`` and ``attribute`` reuse the
same dedup / parsing / provider-factory logic without copy-paste.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .config import resolve_model
from .providers import BaseProvider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider


# Maximum response characters we'll attempt to parse. Guards against a
# runaway proposer/mutator that fills its context.
_MAX_RESPONSE_CHARS = 200_000

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_array_response(
    text: str,
    *,
    required_str_keys: Iterable[str] = ("input",),
    required_keys: Iterable[str] = (),
) -> list[dict]:
    """Extract a list of object dicts from an LLM JSON-array response.

    Tolerates triple-backtick ``json`` fences and a small amount of
    surrounding prose. Each returned dict must:

    * contain every key in ``required_str_keys`` as a non-empty
      string; and
    * contain every key in ``required_keys`` with any non-``None``
      value (no type restriction — accepts dicts, numbers, lists).

    Items missing either constraint are dropped (not raised) so a
    single bad item doesn't poison a batch. Returns ``[]`` on any
    top-level parse failure.

    Discovery's ``parse_proposer_response`` uses
    ``required_str_keys=("input",), required_keys=("expected",)`` —
    matching the original behaviour where structured-extraction
    suites can have dict/number/list ``expected`` values. Bisect's
    mutator uses ``required_str_keys=("input",)`` with no
    ``required_keys`` because bisect mutates an existing case and
    pulls ``expected`` from the source suite, not the mutator
    response.
    """
    if not text:
        return []
    if len(text) > _MAX_RESPONSE_CHARS:
        text = text[:_MAX_RESPONSE_CHARS]
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
    str_required = tuple(required_str_keys)
    presence_required = tuple(required_keys)
    for item in obj:
        if not isinstance(item, dict):
            continue
        if any(
            k not in item or not isinstance(item[k], str) or not item[k].strip()
            for k in str_required
        ):
            continue
        if any(item.get(k) is None for k in presence_required):
            continue
        out.append(item)
    return out


def jaccard_5gram(a: str, b: str) -> float:
    """Character-5-gram Jaccard similarity. Cheap and surprisingly OK.

    Used to drop near-duplicate proposals/mutations. The consequence
    of a false positive is a slightly redundant batch, not a wrong
    drift conclusion, so the 5-gram approximation is fine.

    Inputs shorter than the 5-gram window have no 5-grams to
    intersect, so fall back to case-insensitive exact equality —
    otherwise dedup silently no-ops on every short prompt.
    """
    if len(a) < 5 or len(b) < 5:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

    def grams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 5] for i in range(len(s) - 4)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union


def default_provider_factory(model_id: str) -> BaseProvider:
    """Build a provider for a model identifier.

    Shared by ``discover``, ``bisect``, and ``attribute`` — every
    subcommand that needs to call a free-floating LLM (proposer,
    mutator, paraphraser) without a pre-built provider.
    """
    cfg = resolve_model(model_id)
    if cfg.provider == "anthropic":
        return AnthropicProvider(model=cfg.model, **cfg.params)
    if cfg.provider == "openai":
        return OpenAIProvider(model=cfg.model, **cfg.params)
    if cfg.provider == "google":
        return GoogleProvider(model=cfg.model, **cfg.params)
    raise ValueError(
        f"No provider available for provider='{cfg.provider}' "
        f"(model={model_id})"
    )
