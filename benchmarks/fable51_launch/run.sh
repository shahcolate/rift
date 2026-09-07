#!/usr/bin/env bash
# Fable 5.1 launch study — paired drift benchmark driver.
#
# Two questions, two pairings, one script:
#
#   1. "Did the upgrade regress?"     fable-5   → fable-5-1   (same tier, same price)
#   2. "Is the tier worth it, now?"   opus-5    → fable-5-1   (2× price; both think
#                                                              by default, so this
#                                                              is the FAIR version of
#                                                              the Fable 5 vs Opus 4.7
#                                                              study, which compared
#                                                              thinking vs no-thinking)
#
# Mirrors benchmarks/fable5_vs_opus47/run.sh: standard suites pairwise
# (compare, not matrix), the machine-verified hard_reasoning suite (the
# only one with headroom above the frontier), and the context-rot
# distractor expansion with per-regime subgroups. Judge for open_ended_qa
# is pinned to sonnet-4-6 — the judge validated at κ = 1.00 against the
# committed gold set (`rift validate-judge`); keeping it fixed keeps the
# judged numbers comparable to the Fable 5 study.
#
# Configuration disclosure (put this in the analysis): Rift sends no
# `thinking` / `output_config.effort`; Fable 5/5.1 and Opus 5 all run
# adaptive thinking at default effort, so pairing 2 is effort-matched
# by default — unlike the Fable 5 vs Opus 4.7 run.
#
# COST. Run `rift estimate` first — it is keyless:
#   rift estimate --model fable-5 --model fable-5-1 --model opus-5 \
#     --suite reasoning --suite extraction --suite summarization \
#     --suite code_generation --suite open_ended_qa --suite hard_reasoning \
#     --calibrate-from benchmarks/fable5_vs_opus47/hard_reasoning.json
# Calibrated against the committed Fable 5 run: ≈ $3.50 per Fable-tier
# model for everything below except context-rot, and ≈ $4.60 more per
# Fable-tier model for context-rot (454k input tokens of distractors).
# Whole script ≈ $20 at list price; SKIP_CONTEXT_ROT=1 brings it to ≈ $9.
# The fable-5-1 side is cached after pairing 1, so pairing 2 pays only
# for opus-5 (≈ half a Fable).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
JUDGE=sonnet-4-6
SUITES=(reasoning extraction summarization code_generation open_ended_qa hard_reasoning)

run_pair() {
  local baseline="$1" challenger="$2" tag="$3"
  mkdir -p "$DIR/$tag"
  for suite in "${SUITES[@]}"; do
    rift compare --baseline "$baseline" --challenger "$challenger" \
      --suite "$suite" --concurrency 2 --refusal --power \
      --judge-model "$JUDGE" \
      -o "$DIR/$tag/$suite.json" -r "$DIR/$tag/$suite.md" || true  # exit 1 = significant regression, keep going
  done
  if [ -z "${SKIP_CONTEXT_ROT:-}" ]; then
    rift compare --baseline "$baseline" --challenger "$challenger" \
      --suite context_rot_reasoning --context-rot --subgroup distractor: \
      --concurrency 2 --refusal --power \
      -o "$DIR/$tag/context_rot.json" -r "$DIR/$tag/context_rot.md" || true
  fi
}

run_pair fable-5 fable-5-1 upgrade      # 1. same tier: did 5.1 regress 5?
run_pair opus-5  fable-5-1 tier         # 2. is the tier worth 2× — effort-matched this time

# Executive briefs for both pairings (keyless re-render of the saved JSONs).
for tag in upgrade tier; do
  rift report "$DIR/$tag/hard_reasoning.json" --format brief -o "$DIR/$tag/brief.html"
done
