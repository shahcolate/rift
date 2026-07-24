"""Prompt-mutation scaffolding shared by ``bisect`` and ``attribute``.

Both subcommands need to generate semantic-preserving rewrites of a
prompt: ``bisect`` to find the minimal edit that fixes a regressed
case, ``attribute`` to test template-sensitivity by re-running the
challenger across paraphrases.

The taxonomy below is deliberately small (seven families) and each
family is a named class of edit the LLM mutator targets. Family
definitions are exposed as module-level constants so the markdown
report can reference them by the same name the CLI accepts on
``--families``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import _text
from .pricing import cost_of
from .providers import BaseProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutation taxonomy
# ---------------------------------------------------------------------------

#: Ordered tuple of mutation family identifiers. The order is the
#: default tie-break order for "smallest-semantic-change first"
#: selection in :mod:`rift.bisect`. Earlier families represent
#: smaller / more local edits; later families represent larger
#: structural rewrites.
MUTATION_FAMILIES: tuple[str, ...] = (
    "typo_fix",
    "format_instruction",
    "clarify_ambiguity",
    "constraint_tighten",
    "constraint_loosen",
    "paraphrase",
    "context_strip",
)


#: Human-readable definitions surfaced into the mutator prompt. Each
#: must fit cleanly into "A {family} mutation means: {definition}."
FAMILY_DEFINITIONS: dict[str, str] = {
    "typo_fix": (
        "fix an apparent typo, misspelling, or punctuation mistake in "
        "the prompt without changing any other text"
    ),
    "format_instruction": (
        "rewrite the output-format instruction (e.g. \"Return ONLY ...\" "
        "→ \"Output: <value>\") while keeping the question itself "
        "verbatim"
    ),
    "clarify_ambiguity": (
        "add one short clarifying phrase to resolve an ambiguity in "
        "the question, without changing the correct answer"
    ),
    "constraint_tighten": (
        "tighten or add an explicit constraint that makes the desired "
        "answer more specific, again without changing what the correct "
        "answer is"
    ),
    "constraint_loosen": (
        "loosen or remove an over-restrictive constraint while keeping "
        "the question recognisably the same"
    ),
    "paraphrase": (
        "rewrite the prompt in different words while preserving its "
        "semantics — the correct answer must not change"
    ),
    "context_strip": (
        "remove distractor or filler context from the prompt, keeping "
        "only the core question and any constraints needed to answer it"
    ),
}


#: Jaccard threshold above which a candidate mutation is treated as a
#: duplicate of the original prompt or another already-accepted
#: mutation. Tighter than discovery's 0.80 because bisect wants even
#: small wording changes to count as distinct candidates.
MUTATION_DEDUP_JACCARD = 0.95


# ---------------------------------------------------------------------------
# Mutator prompt
# ---------------------------------------------------------------------------


MUTATOR_PROMPT_TEMPLATE = """\
You are debugging an LLM regression. Below is a prompt the BASELINE \
model answered correctly but the CHALLENGER model answered wrong.

Original prompt:
{original}

Correct answer (from baseline):
{expected_or_baseline}

Challenger's wrong answer:
{challenger_output}

Generate {n} minimally-edited rewrites of the prompt in the family \
"{family}". A "{family}" mutation means: {family_definition}.

Hard rules:
- Preserve the question's semantics — a rewrite that changes the \
correct answer is INVALID.
- Make the smallest edit that exemplifies the family.
- Do not include the answer in the rewrite.

Return ONLY a JSON array of objects with keys:
  "input"     — the rewritten prompt (string),
  "rationale" — one sentence on what changed (string).
"""


# Truncate the challenger's wrong output before splicing it into the
# prompt. The full output can blow up the mutator's context for long
# code-generation outputs; the first 4KB is plenty of failure signal.
_CHALLENGER_OUTPUT_BUDGET = 4096


def _build_mutator_prompt(
    *,
    original: str,
    expected_or_baseline: str,
    challenger_output: str,
    family: str,
    n: int,
) -> str:
    """Render :data:`MUTATOR_PROMPT_TEMPLATE` for one (family, n) call."""
    if family not in FAMILY_DEFINITIONS:
        raise ValueError(
            f"unknown mutation family {family!r}; valid families: "
            f"{', '.join(MUTATION_FAMILIES)}"
        )
    return MUTATOR_PROMPT_TEMPLATE.format(
        original=original,
        expected_or_baseline=expected_or_baseline,
        challenger_output=challenger_output[:_CHALLENGER_OUTPUT_BUDGET],
        family=family,
        family_definition=FAMILY_DEFINITIONS[family],
        n=n,
    )


# ---------------------------------------------------------------------------
# Levenshtein edit distance
# ---------------------------------------------------------------------------


def edit_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Pure-python DP, two-row optimisation. Mutator inputs are prompts
    (hundreds of characters typically), so the O(len(a) * len(b))
    runtime is fine without resorting to a C extension.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Make sure b is the shorter string for the row width.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,        # insertion
                prev[j - 1] + cost,    # substitution
            )
        prev, cur = cur, prev
    return prev[-1]


# ---------------------------------------------------------------------------
# Mutation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Mutation:
    """One candidate rewrite of a prompt.

    ``edit_distance`` and ``char_delta`` are computed against
    ``seed_input`` (the original prompt) at construction time so
    downstream code never has to keep the seed around.
    """

    family: str
    seed_input: str
    mutated_input: str
    rationale: str
    edit_distance: int
    char_delta: int

    @classmethod
    def build(
        cls, *, family: str, seed_input: str, mutated_input: str, rationale: str
    ) -> "Mutation":
        return cls(
            family=family,
            seed_input=seed_input,
            mutated_input=mutated_input,
            rationale=rationale,
            edit_distance=edit_distance(seed_input, mutated_input),
            char_delta=len(mutated_input) - len(seed_input),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------


_CACHE_SUBDIR = "mutations"


def _expected_repr(expected: Any) -> str:
    """Deterministic stringification of an ``expected`` value.

    The cache key has to round-trip identical for identical inputs.
    YAML-loaded structured ``expected`` values may be dicts, so we
    use ``json.dumps(..., sort_keys=True)`` to produce a canonical
    string and fall back to ``repr()`` for any non-JSON-serialisable
    object (rare — only ``llm_judge`` rubric blocks).
    """
    try:
        return json.dumps(expected, sort_keys=True, default=str)
    except TypeError:
        return repr(expected)


def _cache_key(
    *,
    original: str,
    expected: Any,
    challenger_output: str,
    family: str,
    model_id: str,
    n: int,
) -> str:
    payload = "|".join((
        original,
        _expected_repr(expected),
        challenger_output[:_CHALLENGER_OUTPUT_BUDGET],
        family,
        model_id,
        str(n),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / _CACHE_SUBDIR / f"{key}.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public API: propose_mutations
# ---------------------------------------------------------------------------


async def _propose_one_family(
    *,
    original: str,
    expected: Any,
    challenger_output: str,
    family: str,
    n: int,
    provider: BaseProvider,
    model_id: str,
    cache_dir: Path | None,
) -> tuple[list[Mutation], float]:
    """Propose mutations for one family. Returns (mutations, cost_usd)."""
    if n <= 0:
        return [], 0.0

    # Cache lookup.
    if cache_dir is not None:
        key = _cache_key(
            original=original,
            expected=expected,
            challenger_output=challenger_output,
            family=family,
            model_id=model_id,
            n=n,
        )
        path = _cache_path(cache_dir, key)
        if path.exists():
            cached = json.loads(path.read_text())
            return (
                [Mutation(**m) for m in cached["mutations"]],
                float(cached.get("cost_usd", 0.0)),
            )

    # Live mutator call.
    prompt = _build_mutator_prompt(
        original=original,
        expected_or_baseline=str(expected),
        challenger_output=challenger_output,
        family=family,
        n=n,
    )
    completion = await provider.complete(prompt)
    cost = cost_of(
        model_id,
        completion.input_tokens,
        completion.output_tokens,
    )
    parsed = _text.parse_json_array_response(
        completion.output_text,
        required_str_keys=("input",),
    )
    mutations: list[Mutation] = []
    for item in parsed:
        mutated = item["input"]
        # Discard mutations that are byte-identical to the original
        # or above the dedup threshold against the original. Dedup
        # against earlier accepted mutations happens in the caller.
        if _text.jaccard_5gram(mutated, original) >= MUTATION_DEDUP_JACCARD:
            continue
        mutations.append(
            Mutation.build(
                family=family,
                seed_input=original,
                mutated_input=mutated,
                rationale=str(item.get("rationale", "")).strip(),
            )
        )

    # Cache write (atomic).
    if cache_dir is not None:
        _atomic_write_json(
            path,
            {
                "family": family,
                "model_id": model_id,
                "cost_usd": cost,
                "mutations": [m.to_dict() for m in mutations],
            },
        )

    return mutations, cost


async def propose_mutations(
    *,
    original: str,
    expected: Any,
    challenger_output: str,
    families: Sequence[str],
    n_per_family: int,
    provider: BaseProvider,
    model_id: str,
    cache_dir: Path | None = None,
) -> tuple[list[Mutation], float]:
    """Propose mutations across multiple families, concurrently.

    Returns ``(mutations, total_cost_usd)``. Mutations are ordered
    by family in the order ``families`` was passed; within each
    family the LLM's emission order is preserved. The caller is
    responsible for further dedup against any already-accepted
    mutations.
    """
    if not families:
        return [], 0.0
    for family in families:
        if family not in FAMILY_DEFINITIONS:
            raise ValueError(
                f"unknown mutation family {family!r}; valid families: "
                f"{', '.join(MUTATION_FAMILIES)}"
            )

    tasks = [
        _propose_one_family(
            original=original,
            expected=expected,
            challenger_output=challenger_output,
            family=family,
            n=n_per_family,
            provider=provider,
            model_id=model_id,
            cache_dir=cache_dir,
        )
        for family in families
    ]
    results = await asyncio.gather(*tasks)

    mutations: list[Mutation] = []
    total_cost = 0.0
    # Cross-family dedup: a mutation in family B that already exists
    # in family A's accepted list is dropped from B. Order: same as
    # ``families``.
    seen: list[str] = []
    for fam_mutations, cost in results:
        total_cost += cost
        for m in fam_mutations:
            if any(
                _text.jaccard_5gram(m.mutated_input, prev) >= MUTATION_DEDUP_JACCARD
                for prev in seen
            ):
                continue
            seen.append(m.mutated_input)
            mutations.append(m)
    return mutations, total_cost
