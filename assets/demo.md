# Rift demo — Opus 4.6 → 4.7

_A guided walkthrough of one real upgrade. Live in your terminal._

**Baseline:** `opus-4-6`  ·  **Challenger:** `opus-4-7`  ·  **Suite:** `context_rot_reasoning__context_rot` (32 cases)


## Headline numbers

| | |
|---|---|
| accuracy delta | **+3.12pp** |
| p value | **1.000** |
| cost per correct pct | **+39.7%** |
| baseline cpc | **$0.1882** |
| challenger cpc | **$0.2630** |
| input token ratio | **1.450×** |

## Act 1 — The upgrade

Your team upgraded **opus-4-6** → **opus-4-7**. Same prompts, same workflow. The bill goes out at month-end. Did anything actually break — and what does it cost now?

_Suite: `context_rot_reasoning__context_rot` (32 cases, paired, McNemar's exact). Replaying committed outcomes for reproducibility — re-run live with `--mode live`._

## Act 2 — What a casual eval sees

Accuracy is the number every benchmark leaderboard reports. By this measure, the upgrade looks like a win:

- Baseline mean: **0.8438**
- Challenger mean: **0.8750**
- Delta: **+0.0312 (+3.12pp)**
- p-value: 1.0000  ·  test: mcnemar_exact
- 95% CI: [-0.1562, +0.2188]

Headline reading: newer model, more correct answers. Ship it.

## Act 3 — What Rift sees

Same prompts. Same answers (mostly). But the bill tells a different story:

- Baseline spend: **$5.08**
- Challenger spend: **$7.36**
- Baseline $/correct: **$0.1882**
- Challenger $/correct: **$0.2630**  (**+39.7%**)

**The why.** For byte-identical prompts, the challenger emits **1.450× more input tokens** than the baseline (337,920 → 489,984). At list-price parity, this is a silent per-prompt cost increase on migration. Accuracy doesn't pay for it.

## Act 4 — Where the cost concentrates

Does the inflation hit you everywhere, or only on long prompts?

| Subgroup | n | Baseline | Challenger | Δ acc | Δ $/correct |
|----------|---|----------|------------|-------|-------------|
| distractor:0k | 8 | 0.875 | 0.875 | +0.000 | +0.0005 |
| distractor:2k | 8 | 0.875 | 0.875 | +0.000 | +0.0159 |
| distractor:32k | 8 | 0.875 | 0.875 | +0.000 | +0.2473 |
| distractor:8k | 8 | 0.750 | 0.875 | +0.125 | +0.0391 |

## Verdict — what to do Monday

**Accuracy ticked up (+3.12pp, not significant at α=0.05), but $/correct rose +39.7%.**

**Recommendation.** Do NOT migrate short-prompt workloads to opus-4-7 on price-parity assumptions. The quality lift on this suite does not pay for tokenizer inflation.

**Action items.**

- Pin opus-4-6 for short-prompt production paths until tokenizer parity is restored.
- Consider opus-4-7 only for long-context workloads (8k+ distractor tokens) where the robustness lift is largest.
- Re-evaluate every quarter — tokenizer changes ship without release-note announcements.
- Add `rift compare` to CI for any model-version bump touching production prompts.

_Findings replicate on the committed outcomes file (n=32, paired, McNemar's exact). Significance threshold α=0.05; cost figures use list pricing — apply your enterprise multiplier for contracted rates._

**Reproduce.** `rift demo  # or: python benchmarks/run_context_rot.py --mode record`

## Sources

- `benchmarks/context_rot_outcomes.yaml`
- `benchmarks/context_rot_opus47_analysis.md`
