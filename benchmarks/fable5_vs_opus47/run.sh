#!/usr/bin/env bash
# Fable 5 vs Opus 4.7 — paired drift benchmark driver.
#
# Mirrors the 3way_opus48 methodology: five standard suites pairwise
# (compare, not matrix — only two models) plus the context-rot
# distractor expansion with per-regime subgroups. Judge for
# open_ended_qa is pinned (to the package default, sonnet-4-6) so the
# run is reproducible regardless of RIFT_JUDGE_MODEL in the caller's
# environment. The judge is neither contestant; with both contestants
# being Anthropic models, any house-style judge bias lands on both
# sides equally, so the paired delta stays fair.
#
# Cost note: Fable 5 lists at $10/$50 per Mtok (2x Opus) AND its new
# tokenizer counts ~30% more tokens for identical prompts, AND its
# always-on thinking bills as output tokens. Expect the Fable side to
# cost ~3-5x the Opus side. Whole run should stay under ~$25.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE=opus-4-7
CHALLENGER=fable-5
JUDGE=sonnet-4-6

for SUITE in reasoning extraction summarization code_generation open_ended_qa; do
  rift compare --baseline "$BASELINE" --challenger "$CHALLENGER" \
    --suite "$SUITE" --concurrency 2 --refusal --power \
    --judge-model "$JUDGE" \
    -o "$DIR/$SUITE.json" -r "$DIR/$SUITE.md" || true  # exit 1 = significant regression, keep going
done

rift compare --baseline "$BASELINE" --challenger "$CHALLENGER" \
  --suite context_rot_reasoning --context-rot --subgroup distractor: \
  --concurrency 2 --refusal --power \
  -o "$DIR/context_rot.json" -r "$DIR/context_rot.md" || true
