"""LLM-as-judge scorer for open-ended outputs.

When ``exact_match`` and ``fuzzy_match`` don't fit — summaries,
free-form explanations, code that admits many surface forms — an
LLM judge is the standard fallback. This scorer calls a separate
judge model and asks it to rate the candidate output against a
reference answer or a free-form rubric on a 0.0-1.0 scale.

Three properties keep an LLM judge defensible in a Rift report:

1. **The judge model is named in every report.** It is part of the
   methodology, not a black box. The scorer instance carries
   ``judge_model``; the runner stamps it into ``RunResult.metadata``
   so a reader can see who graded.
2. **Judge responses are cached** by ``(judge_model, judge_prompt)``
   so re-running a comparison is free and two runs of the same
   suite produce identical scores from identical outputs.
3. **The judge prompt is fixed and committed** (see
   :data:`JUDGE_PROMPT_TEMPLATE`). Wording shifts in the rubric
   invalidate the cache — a deliberate prompt change should always
   re-score, never silently reuse old judgments.

Known biases (Zheng et al. 2023; Wang et al. 2023):

* **Length bias** — judges over-reward longer outputs.
* **Family bias** — judges over-reward outputs from their own
  model family.
* **Self-preference** when a model judges its own outputs.

Mitigations baked in here:

* The judge is asked for a numeric score on a fixed 0-1 scale, not
  for a pairwise A-vs-B preference (which has a strong position
  bias).
* The judge prompt explicitly tells the judge to ignore style and
  length when content is equivalent.
* When ``judge_model`` matches one of the models being compared,
  the runner emits a warning at startup. Best practice: set
  ``RIFT_JUDGE_MODEL`` to a *third* model family different from
  both sides of the comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from ..providers import BaseProvider, Completion


# A fixed, committed prompt. Edit only with intent — every change
# invalidates every cached judgment. The prompt deliberately:
#   - names the 0-1 scale and four anchor points,
#   - asks the judge to ignore wording differences when content matches,
#   - requires a JSON-only response so parsing is robust,
#   - asks for one-sentence reasoning so a reader can audit.
JUDGE_PROMPT_TEMPLATE = """\
You are an impartial grader evaluating whether a model's answer satisfies a target.

Question / prompt the model was given:
\"\"\"
{question}
\"\"\"

{target_block}

Model's answer:
\"\"\"
{output}
\"\"\"

Grade the model's answer on a 0.0 to 1.0 scale:
- 1.0  -- Equivalent to the target. Differences in wording, formatting, or style do not matter as long as the content matches.
- 0.7  -- Mostly correct but missing a minor element, or contains a small mistake that does not change the headline conclusion.
- 0.3  -- Partially correct: captures some elements but misses or contradicts the core of the target.
- 0.0  -- Wrong, unrelated, refused, or empty.

You may use any value in [0.0, 1.0]. Be strict on factual errors; be lenient on wording.

Respond with a single JSON object and nothing else, in this exact shape:
{{"score": <float between 0 and 1>, "reasoning": "<one short sentence>"}}
"""


# Regex for extracting the JSON blob from the judge response. Some
# providers wrap JSON in ```json fences even when asked not to.
_JSON_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)


# Default judge model. Resolved at scorer-construction time, not at
# import time, so test code can mutate the env var.
DEFAULT_JUDGE_MODEL_ENV = "RIFT_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def _format_target_block(expected: Any) -> str:
    """Render the target portion of the judge prompt.

    ``expected`` can be:

    * a string reference answer — rendered as ``Reference answer:``
    * a dict with a ``rubric`` key — rendered as ``Grading rubric:``
      and the value used verbatim
    * any other dict — rendered as ``Reference answer (JSON):``
    """
    if isinstance(expected, dict) and "rubric" in expected:
        return f"Grading rubric:\n\"\"\"\n{expected['rubric']}\n\"\"\""
    if isinstance(expected, dict):
        return (
            f"Reference answer (JSON):\n\"\"\"\n"
            f"{json.dumps(expected, indent=2)}\n\"\"\""
        )
    return f"Reference answer:\n\"\"\"\n{expected}\n\"\"\""


def _build_judge_prompt(question: str, output: str, expected: Any) -> str:
    """Assemble the full judge prompt for one case."""
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        target_block=_format_target_block(expected),
        output=output,
    )


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Parse ``{"score": X, "reasoning": "..."}`` out of judge output.

    Tolerates triple-backtick ``json`` fences and trailing prose. On
    parse failure returns ``(0.0, "unparseable judge response: ...")``
    so a malformed judgment scores zero rather than crashing the run.
    The full raw judge response is preserved separately by the
    scorer for auditing.
    """
    s = text.strip()
    # Strip markdown fences if present.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # Try direct parse first.
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return 0.0, f"unparseable judge response: {text[:120]!r}"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            return 0.0, f"unparseable judge response: {exc}"
    score = obj.get("score")
    reasoning = str(obj.get("reasoning", ""))
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0, f"non-numeric score in judge response: {score!r}"
    # Clip out-of-range — some judges return 5 on a 1-5 scale despite
    # being asked for 0-1. Defensive but better than NaN.
    score_f = max(0.0, min(1.0, score_f))
    return score_f, reasoning


ProviderFactory = Callable[[str], BaseProvider]


class LLMJudgeScorer:
    """LLM-as-judge scorer.

    Async by design: judging is an HTTP call. The synchronous
    :meth:`score` is only available for environments without a
    running event loop (it raises inside one). The runner always
    calls :meth:`ascore` directly.

    Parameters
    ----------
    judge_model
        Identifier of the model to grade with. Resolved against the
        same provider table the runner uses. Defaults to
        ``$RIFT_JUDGE_MODEL`` or :data:`DEFAULT_JUDGE_MODEL`.
    provider_factory
        Function ``model_id -> BaseProvider``. Defaults to the same
        ``_get_provider`` the runner uses, so production code does
        not need to pass it. Tests inject a stub.
    cache_dir
        Where to persist judge responses. Defaults to
        ``$RIFT_CACHE_DIR`` or ``.rift/cache``. Cache keys include
        ``judge_model`` so swapping judges does not collide.
    judge_params
        Extra parameters forwarded to ``provider.complete`` (e.g.
        ``{"temperature": 0}``). Determinism on the judge is
        recommended — non-zero temperature here makes a run's scores
        depend on lottery.
    """

    def __init__(
        self,
        judge_model: str | None = None,
        provider_factory: ProviderFactory | None = None,
        cache_dir: str | Path | None = None,
        judge_params: dict | None = None,
    ) -> None:
        self.judge_model = (
            judge_model
            or os.environ.get(DEFAULT_JUDGE_MODEL_ENV)
            or DEFAULT_JUDGE_MODEL
        )
        self._provider_factory = provider_factory or _default_provider_factory
        self._provider: BaseProvider | None = None
        if cache_dir is None:
            cache_dir = os.environ.get("RIFT_CACHE_DIR") or ".rift/cache"
        self.cache_dir = Path(cache_dir)
        self.judge_params: dict = {"temperature": 0.0}
        if judge_params:
            self.judge_params.update(judge_params)
        # Per-judgment audit log; populated as ``ascore`` is called.
        # Keyed by the cache key; useful for ``--report`` to surface
        # the judge's reasoning on regressed cases.
        self.last_reasoning: dict[str, str] = {}

    # ----- sync entry point (compatibility with the Scorer protocol) -----

    def score(self, output: str, expected: Any) -> float:
        """Sync entry point. Only valid outside a running event loop.

        Exists so the :class:`Scorer` protocol stays single-shape.
        Inside the runner — which is already async — call
        :meth:`ascore` instead and avoid the asyncio.run round-trip.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                "LLMJudgeScorer.score() called inside a running event loop; "
                "use `await scorer.ascore(...)` instead."
            )
        return asyncio.run(self.ascore(output, expected))

    # ----- async entry point (the real implementation) -----

    async def ascore(
        self,
        output: str,
        expected: Any,
        context: str | None = None,
    ) -> float:
        """Grade ``output`` against ``expected``.

        ``context`` is the original prompt the model was given; it
        is included in the judge prompt so the judge can grade in
        context. Pass it from the runner — without it the judge is
        grading the answer in isolation, which is fine for short
        reference answers but poor for rubric scoring.
        """
        question = context or "(prompt not provided)"
        prompt = _build_judge_prompt(question, output, expected)
        cache_key = self._cache_key(prompt)
        cached = self._read_cache(cache_key)
        if cached is not None:
            score, reasoning = _parse_judge_response(cached.output_text)
            self.last_reasoning[cache_key] = reasoning
            return score

        provider = self._get_provider()
        completion = await provider.complete(prompt, **self.judge_params)
        self._write_cache(cache_key, completion)

        score, reasoning = _parse_judge_response(completion.output_text)
        self.last_reasoning[cache_key] = reasoning
        return score

    async def close(self) -> None:
        if self._provider is not None:
            await self._provider.close()
            self._provider = None

    # ----- internals -----

    def _get_provider(self) -> BaseProvider:
        if self._provider is None:
            self._provider = self._provider_factory(self.judge_model)
        return self._provider

    def _cache_key(self, prompt: str) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # judge_ prefix so judge entries don't collide with
        # completion entries the runner stores in the same dir.
        return f"judge_{self.judge_model}_{h}"

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> Completion | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return Completion(**json.load(f))
        except Exception:
            # Treat corruption as a miss and let the next write
            # overwrite the bad file.
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_key: str, completion: Completion) -> None:
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(completion), f, default=str)
        tmp.replace(path)


def _default_provider_factory(model_id: str) -> BaseProvider:
    """Build a provider for a judge-model identifier.

    Imported lazily to avoid a circular import on
    ``rift.runner`` / ``rift.config``. Mirrors
    ``rift.runner._get_provider`` so the same alias rules apply.
    """
    from ..config import resolve_model
    from ..providers.anthropic import AnthropicProvider
    from ..providers.openai import OpenAIProvider

    cfg = resolve_model(model_id)
    if cfg.provider == "anthropic":
        return AnthropicProvider(model=cfg.model, **cfg.params)
    if cfg.provider == "openai":
        return OpenAIProvider(model=cfg.model, **cfg.params)
    raise ValueError(
        f"LLM judge does not support provider '{cfg.provider}' (model={model_id})"
    )
