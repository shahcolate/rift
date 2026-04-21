"""Context-rot suite expansion.

A context-rot suite takes a base set of reasoning cases and emits N
variants per case, each with a different amount of distractor text
wrapped around the real question. This isolates *context length* as
the independent variable — the question, the expected answer, the
scoring method, and even the token ordering of the question are held
constant. The only thing that changes is how much plausible-but-
irrelevant material the model must ignore.

Design choices:

* **Templating, not sampling.** Each (case, distractor_level) pair
  deterministically expands to the same prompt given the same seed.
  This is required for cache hits across runs and for paired
  statistical tests — challenger and baseline see *byte-identical*
  inputs.
* **Semantically-adjacent distractors.** We use mixed-domain corporate
  filler (policies, meeting minutes, glossaries) rather than random
  tokens. Random tokens are trivially ignorable; real-world
  distractors are the regime enterprise users actually face.
* **Position randomized per case, fixed across models.** The needle
  floats — prefix, middle, or suffix — because degradation is known
  to depend on position (lost-in-the-middle). The position is keyed
  off the case index so it is stable but not all identical.

Distractor levels are expressed in target *token* counts, not
character counts. We use a crude 4-chars-per-token heuristic which is
accurate enough for targeting — the actual token count is measured
post-hoc via provider usage accounting.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EvalCase, SuiteConfig


CHARS_PER_TOKEN = 4  # Anthropic/OpenAI BPE rough average for English prose


DISTRACTOR_CORPUS = [
    # Corporate policy boilerplate — syntactically rich, semantically inert.
    "Section 4.2.1 of the revised Vendor Onboarding Policy supersedes all prior "
    "guidance issued under Directive 2019-C. Procurement leads are reminded that "
    "net-30 terms remain the default unless overridden by a signed master "
    "services agreement. Vendors operating in jurisdictions subject to sanctions "
    "review must submit a completed OFAC questionnaire within five business days "
    "of preliminary selection; failure to do so will result in automatic "
    "rescission of any provisional offer.",

    "The Q3 all-hands concluded with a commitment to reduce cycle time for "
    "cross-functional deliverables by 18% year over year. Leadership identified "
    "three primary levers: (1) consolidation of redundant Jira projects, (2) "
    "migration of the internal wiki from Confluence to a federated Notion "
    "workspace, and (3) retirement of the weekly portfolio review in favor of "
    "an asynchronous Loom-based cadence. A task force reporting to the COO has "
    "been stood up to oversee execution.",

    "Glossary: 'Baseline throughput' refers to the steady-state request rate "
    "observed over a rolling 7-day window excluding scheduled maintenance "
    "intervals. 'Effective throughput' applies a correction factor of 0.82 to "
    "account for partial failures that nonetheless consume queue capacity. "
    "'Amortized latency' is the p50 of end-to-end response time weighted by "
    "request complexity class (A, B, or C) as defined in RFC-0044.",

    "From the minutes of the April 3 architecture review: the committee rejected "
    "the proposed adoption of gRPC for internal service-to-service calls, citing "
    "concerns over debuggability, the lack of mature tooling on the mobile "
    "client, and the additional operational burden of a second IDL alongside "
    "the existing OpenAPI surface. The proposal will be revisited in Q2 "
    "contingent on a successful pilot within the Observability org.",

    "The updated reimbursement schedule takes effect on the first of next month. "
    "Per-diem meal allowances have been adjusted upward to reflect regional "
    "inflation: Tier 1 cities (NYC, SF, LON, TYO) $90/day; Tier 2 cities $70/day; "
    "all other locations $55/day. Hotel caps are unchanged. Employees are "
    "reminded that itemized receipts are required for any single expense "
    "exceeding $75 and that alcohol is reimbursable only when hosting clients.",

    "Release notes for platform version 7.4.2: fixed a regression in the audit "
    "log exporter that caused events with payloads larger than 32KB to be "
    "silently truncated. Improved SAML assertion validation to reject responses "
    "missing the AudienceRestriction element. Added rate-limit headers to the "
    "bulk import API. Deprecated the legacy /v1/users endpoint; it will be "
    "removed in 8.0. No schema changes in this release.",

    "Data retention FAQ: user-generated content is retained for 180 days after "
    "account deletion unless the user has invoked the export-and-purge workflow, "
    "which reduces retention to 30 days. Audit logs are retained for 7 years in "
    "compliance with applicable regulatory frameworks. Customer support "
    "transcripts are retained for 2 years. Aggregate analytics data, once "
    "anonymized per the k-anonymity standard of k=50, may be retained "
    "indefinitely.",

    "Travel advisory memo: all business travel to designated high-risk regions "
    "requires pre-approval from the Head of Global Security and the filing of a "
    "duty-of-care form at least 72 hours in advance. Travelers should enroll in "
    "the company's roaming data plan, carry a company-issued satellite beacon, "
    "and register their itinerary with the International SOS portal. Personal "
    "travel extensions are permitted but not covered by corporate insurance.",
]


@dataclass(frozen=True)
class DistractorLevel:
    """A single distractor regime to expand cases under."""

    name: str          # e.g. "0k", "2k", "8k", "32k"
    target_tokens: int # approximate, 0 means no distractor


DEFAULT_LEVELS = (
    DistractorLevel("0k",  0),
    DistractorLevel("2k",  2_000),
    DistractorLevel("8k",  8_000),
    DistractorLevel("32k", 32_000),
)


def _seeded_rng(key: str) -> random.Random:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _fill_to_tokens(target_tokens: int, rng: random.Random) -> str:
    """Deterministically sample distractor paragraphs up to target token count."""
    if target_tokens <= 0:
        return ""
    target_chars = target_tokens * CHARS_PER_TOKEN
    pieces: list[str] = []
    acc = 0
    # Shuffle a copy so ordering varies between cases but is reproducible.
    corpus = list(DISTRACTOR_CORPUS)
    rng.shuffle(corpus)
    i = 0
    while acc < target_chars:
        piece = corpus[i % len(corpus)]
        pieces.append(piece)
        acc += len(piece) + 2
        i += 1
    return "\n\n".join(pieces)


def _wrap(question: str, distractor: str, position: str) -> str:
    """Embed the question into the distractor at prefix/middle/suffix."""
    if not distractor:
        return question
    header = (
        "--- BEGIN REFERENCE MATERIAL (may contain irrelevant information) ---"
    )
    footer = "--- END REFERENCE MATERIAL ---"
    if position == "prefix":
        return f"{header}\n{distractor}\n{footer}\n\nQUESTION:\n{question}"
    if position == "suffix":
        return f"QUESTION:\n{question}\n\n{header}\n{distractor}\n{footer}"
    # middle: split distractor in half
    mid = len(distractor) // 2
    first, second = distractor[:mid], distractor[mid:]
    return (
        f"{header}\n{first}\n{footer}\n\nQUESTION:\n{question}\n\n"
        f"{header}\n{second}\n{footer}"
    )


_POSITIONS = ("prefix", "middle", "suffix")


def expand_suite(
    base: SuiteConfig,
    levels: tuple[DistractorLevel, ...] = DEFAULT_LEVELS,
) -> SuiteConfig:
    """Expand a base suite into a context-rot suite.

    The returned suite has ``len(base.cases) * len(levels)`` cases. Each
    expanded case is tagged with ``distractor:<level>`` and
    ``origin:<base_index>`` so downstream reports can group by regime
    without reparsing inputs.
    """
    expanded: list[EvalCase] = []
    for base_idx, case in enumerate(base.cases):
        position = _POSITIONS[base_idx % len(_POSITIONS)]
        for level in levels:
            rng = _seeded_rng(f"{base.name}:{base_idx}:{level.name}")
            distractor = _fill_to_tokens(level.target_tokens, rng)
            new_input = _wrap(case.input, distractor, position)
            expanded.append(
                EvalCase(
                    input=new_input,
                    expected=case.expected,
                    tags=[
                        *case.tags,
                        f"distractor:{level.name}",
                        f"origin:{base_idx}",
                        f"position:{position}",
                    ],
                )
            )

    return SuiteConfig(
        name=f"{base.name}__context_rot",
        description=(
            f"{base.description} — expanded with distractor context at "
            f"levels: {', '.join(l.name for l in levels)}."
        ),
        scoring=base.scoring,
        model_params=base.model_params,
        cases=expanded,
    )


def load_base_and_expand(path_or_name: str) -> SuiteConfig:
    """Convenience: load any suite and return its context-rot expansion."""
    from .config import load_suite
    return expand_suite(load_suite(path_or_name))
