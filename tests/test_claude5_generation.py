"""Claude 5.1-generation support: catalog, provider wire shape, refusals.

Pins three things the Fable 5.1 launch needs:

1. The catalog knows the generation (pricing, aliases, provider params) —
   a missing pricing entry silently bills every case at $0, and a missing
   DEPRECATED_PARAMS entry sends ``temperature`` to a model that 400s on
   it, erroring the whole run.
2. An API-level refusal (HTTP 200, empty content, ``stop_reason=refusal``)
   travels from the provider through the cache and the runner into
   ``CaseResult.stop_reason``, so the refusal classifier and the drift
   report can disclose it instead of publishing it as capability loss.
3. Dated variants (``claude-opus-5-20261101``) inherit their family's
   parameter table, the same way they inherit its price.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from rift.config import MODEL_ALIASES, EvalCase, ModelConfig, SuiteConfig, resolve_model
from rift.pricing import PRICING, lookup
from rift.providers import Completion
from rift.providers.anthropic import (
    DEPRECATED_PARAMS,
    MIN_MAX_TOKENS,
    AnthropicProvider,
    _family,
)
from rift.refusal import API_REFUSAL_MARKER, classify_run
from rift.runner import CaseResult, RunResult, run_suite


# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,inp,out", [
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-sonnet-5", 2.00, 10.00),
    ("claude-haiku-4-5", 1.00, 5.00),
])
def test_claude5_generation_priced(model, inp, out):
    p = lookup(model)
    assert p is not None, f"{model} missing from PRICING — every case would bill $0"
    assert (p.input_per_mtok, p.output_per_mtok) == (inp, out)


def test_fable_5_1_dated_variant_inherits_price():
    assert lookup("claude-fable-5-1-20260901") == PRICING["claude-fable-5-1"]


def test_sonnet_5_is_cheaper_than_sonnet_4_6():
    # The generation cut Sonnet's list price; a copy-paste of the 4.6 row
    # would overstate every Sonnet 5 cost by 50%.
    assert PRICING["claude-sonnet-5"].cost(1, 1) < PRICING["claude-sonnet-4-6"].cost(1, 1)


@pytest.mark.parametrize("alias,canonical", [
    ("fable-5-1", "claude-fable-5-1"),
    ("fable", "claude-fable-5-1"),      # bare family name → current generation
    ("fable-5", "claude-fable-5"),      # pinned alias keeps pointing at 5
    ("opus-5", "claude-opus-5"),
    ("opus", "claude-opus-5"),
    ("sonnet-5", "claude-sonnet-5"),
    ("sonnet", "claude-sonnet-5"),
])
def test_aliases_resolve_to_anthropic(alias, canonical):
    cfg = resolve_model(alias)
    assert cfg.provider == "anthropic"
    assert cfg.model == canonical


def test_every_alias_target_is_priced():
    # An alias that resolves to an unpriced id is a $0-cost trap the
    # budget guard would have to catch at runtime; catch it here instead.
    for alias, target in MODEL_ALIASES.items():
        if target.startswith(("claude", "gpt-", "o1", "o3", "gemini")):
            assert lookup(target) is not None, f"alias {alias!r} → unpriced {target!r}"


def test_every_priced_claude5_model_strips_sampler_knobs():
    for model in ("claude-fable-5-1", "claude-fable-5", "claude-opus-5",
                  "claude-sonnet-5"):
        assert {"temperature", "top_p", "top_k"} <= DEPRECATED_PARAMS[model]


def test_thinking_default_models_have_output_floor():
    # Always-on / default-on thinking bills against max_tokens; the suite
    # default of 4096 truncates answers after the invisible thinking spend.
    for model in ("claude-fable-5-1", "claude-fable-5", "claude-opus-5"):
        assert MIN_MAX_TOKENS[model] >= 16000
    # Sonnet 5 runs without thinking unless asked — no floor.
    assert "claude-sonnet-5" not in MIN_MAX_TOKENS


def test_family_lookup_matches_dated_variants_only():
    assert _family("claude-opus-5", DEPRECATED_PARAMS) == "claude-opus-5"
    assert _family("claude-opus-5-20261101", DEPRECATED_PARAMS) == "claude-opus-5"
    # A named submodel is a different product — no inheritance.
    assert _family("claude-opus-5-fast", DEPRECATED_PARAMS) is None
    assert _family("claude-3-5-sonnet-20241022", DEPRECATED_PARAMS) is None


# ---------------------------------------------------------------------------
# 2. Provider wire shape + refusal capture
# ---------------------------------------------------------------------------

def _provider_with_response(model: str, body: dict, seen: dict) -> AnthropicProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    provider = AnthropicProvider(model=model, api_key="test-key")
    provider.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.anthropic.com",
    )
    return provider


_OK_BODY = {
    "model": "claude-fable-5-1-20260901",
    "stop_reason": "end_turn",
    "content": [{"type": "thinking", "thinking": ""},
                {"type": "text", "text": "4"}],
    "usage": {"input_tokens": 12, "output_tokens": 900},
}

_REFUSAL_BODY = {
    "model": "claude-fable-5-1-20260901",
    "stop_reason": "refusal",
    "stop_details": {"type": "refusal", "category": "cyber",
                     "explanation": "declined"},
    "content": [],
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


def test_fable_5_1_request_strips_temperature_and_floors_max_tokens():
    seen: dict = {}
    provider = _provider_with_response("claude-fable-5-1", _OK_BODY, seen)
    completion = asyncio.run(provider.complete("2+2?", temperature=0))
    assert "temperature" not in seen["json"]
    assert seen["json"]["max_tokens"] >= 16000
    assert "thinking" not in seen["json"]  # always-on: never sent explicitly
    assert completion.output_text == "4"
    assert completion.stop_reason == "end_turn"
    assert completion.provider_fingerprint == "claude-fable-5-1-20260901"


def test_dated_fable_variant_gets_same_normalization():
    seen: dict = {}
    provider = _provider_with_response("claude-fable-5-1-20260901", _OK_BODY, seen)
    asyncio.run(provider.complete("2+2?", temperature=0, max_tokens=512))
    assert "temperature" not in seen["json"]
    assert seen["json"]["max_tokens"] == 16000


def test_sonnet_5_strips_sampler_but_keeps_default_max_tokens():
    seen: dict = {}
    provider = _provider_with_response("claude-sonnet-5", _OK_BODY, seen)
    asyncio.run(provider.complete("2+2?", temperature=0))
    assert "temperature" not in seen["json"]
    assert seen["json"]["max_tokens"] == 4096


def test_api_refusal_is_captured_not_raised():
    seen: dict = {}
    provider = _provider_with_response("claude-fable-5-1", _REFUSAL_BODY, seen)
    completion = asyncio.run(provider.complete("do the thing"))
    assert completion.output_text == ""
    assert completion.stop_reason == "refusal"
    # Still a real, billed completion — fingerprint and usage recorded.
    assert completion.provider_fingerprint == "claude-fable-5-1-20260901"
    assert completion.input_tokens == 12


def test_completion_from_cache_tolerates_missing_stop_reason():
    # Cache blobs written before this field existed must still load.
    old_blob = {
        "model": "m", "input_text": "i", "output_text": "o", "latency_ms": 1.0,
        "input_tokens": 1, "output_tokens": 1, "raw_response": {},
        "provider_fingerprint": None,
    }
    c = Completion.from_cache(old_blob)
    assert c.stop_reason is None


# ---------------------------------------------------------------------------
# 3. Runner → CaseResult → refusal classifier → report
# ---------------------------------------------------------------------------

class _ScriptedProvider:
    """Answers by PROMPT, not by call order — the runner schedules cases
    concurrently, so arrival order is not case order."""

    def __init__(self, by_prompt):
        self._c = dict(by_prompt)
        self.model = "claude-fable-5-1"

    async def complete(self, prompt, **kwargs):
        return self._c[prompt]

    async def close(self):
        pass


def _completion(text: str, stop: str) -> Completion:
    return Completion(model="claude-fable-5-1", input_text="x", output_text=text,
                      latency_ms=1.0, input_tokens=5, output_tokens=5,
                      raw_response={}, stop_reason=stop)


def test_runner_persists_stop_reason_and_classifier_uses_it(tmp_path, monkeypatch):
    provider = _ScriptedProvider({
        "2+2?": _completion("4", "end_turn"),
        "3+3?": _completion("", "refusal"),
    })
    monkeypatch.setattr("rift.runner._get_provider", lambda cfg: provider)
    suite = SuiteConfig(name="s", scoring="exact_match", cases=[
        EvalCase(input="2+2?", expected="4"),
        EvalCase(input="3+3?", expected="6"),
    ])
    cfg = ModelConfig(provider="anthropic", model="claude-fable-5-1")
    result = asyncio.run(run_suite(suite, cfg, cache_dir=str(tmp_path),
                                   show_progress=False, concurrency=1))

    by_idx = {c.case_index: c for c in result.cases}
    assert by_idx[0].stop_reason == "end_turn"
    assert by_idx[1].stop_reason == "refusal"
    # The refusal is scored 0 on purpose (the model did not answer) and is
    # NOT an error — it is a behavior, and it is disclosed as one.
    assert by_idx[1].score == 0.0 and by_idx[1].error is None

    analysis = classify_run(result)
    flagged = {c.case_index: c for c in analysis.classifications}
    assert flagged[1].refused is True
    assert flagged[1].matched_pattern == API_REFUSAL_MARKER
    assert flagged[0].refused is False
    assert analysis.n_refused == 1

    # Round-trips through the saved-run JSON shape.
    payload = json.loads(json.dumps(result.to_dict()))
    reloaded = RunResult.from_dict(payload)
    assert {c.case_index: c.stop_reason for c in reloaded.cases} == {
        0: "end_turn", 1: "refusal"}


def test_drift_report_discloses_api_refusals(capsys):
    from rift.comparator import compare_runs
    from rift.reporter import generate_markdown_report, print_drift_report

    def run(model, stops):
        return RunResult(model=model, suite_name="s", scoring_method="exact_match",
                         cases=[CaseResult(case_index=i, input_text="q", expected="4",
                                           output="" if s == "refusal" else "4",
                                           score=0.0 if s == "refusal" else 1.0,
                                           latency_ms=1, input_tokens=1, output_tokens=1,
                                           stop_reason=s)
                                for i, s in enumerate(stops)])

    base = run("a", ["end_turn"] * 6)
    chal = run("b", ["end_turn"] * 3 + ["refusal"] * 3)

    def cmp(b, c):
        return compare_runs(b.scores, c.scores, b.model, c.model, "s",
                            bootstrap_n=50)

    drift = cmp(base, chal)

    print_drift_report(drift, base, chal)
    out = capsys.readouterr().out
    assert "API-level refusals" in out and "challenger 3" in out

    md = generate_markdown_report(drift, base, chal)
    assert "API-level refusals scored as 0" in md
    assert "stop_reason=refusal" in md

    # No refusals → no banner (the disclosure must not become noise).
    print_drift_report(cmp(base, base), base, base)
    assert "API-level refusals" not in capsys.readouterr().out
