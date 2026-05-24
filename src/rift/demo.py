"""`rift demo` — a guided, PM-grade walkthrough of a real drift finding.

The demo replays the committed Opus 4.6 → 4.7 context-rot benchmark as
a four-act narrative in the terminal, then optionally exports an HTML
executive memo, a markdown one-pager, and a static SVG screenshot. It
runs offline, with no API keys, in a few seconds.

Architecture: a single :class:`DemoScript` data structure is the source
of truth. The terminal renderer (:func:`run_demo`), the HTML exporter
(:func:`export_demo_html`), the markdown exporter
(:func:`export_demo_markdown`), and the SVG exporter
(:func:`export_demo_svg`) all consume the same script, so the live
demo, the shareable artifact, and the marketing screenshot cannot drift
out of sync.
"""

from __future__ import annotations

import asyncio
import html
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comparator import compare_by_subgroup, compare_runs
from .config import load_suite, resolve_model
from .context_rot import expand_suite
from .providers import Completion
from .reporter import (
    _fmt_cost,
    print_cost_panel,
    print_drift_report,
    print_subgroup_table,
)
from .runner import RunResult, _cache_key, run_suite


ROOT = Path(__file__).parent.parent.parent
RECORDED = ROOT / "benchmarks" / "context_rot_outcomes.yaml"


# ---------------------------------------------------------------------------
# Replay machinery — single source of truth, used by both this module and
# benchmarks/run_context_rot.py.
# ---------------------------------------------------------------------------


def prime_cache_from_recording(suite, model: str, outcomes: dict,
                                cache_dir: Path) -> None:
    """Write recorded completions into Rift's cache so ``run_suite`` hits.

    ``outcomes`` is keyed by ``(origin_index, distractor_level)``; each
    expanded case is resolved back to its recorded answer via its tags.
    This is decoupled from the suite's exact prompt text — rewording
    the suite does not invalidate a recording so long as the
    (origin, level) pairs are preserved.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case in suite.cases:
        origin = next(t for t in case.tags if t.startswith("origin:"))
        level = next(t for t in case.tags if t.startswith("distractor:"))
        key = f"{origin}|{level}"
        rec = outcomes.get(model, {}).get(key)
        if rec is None:
            continue
        output_text = rec["output"]
        input_tokens = rec.get("input_tokens", len(case.input) // 4)
        output_tokens = rec.get("output_tokens",
                                max(1, len(output_text) // 4))
        ck = _cache_key(model, case.input, suite.model_params)
        completion = Completion(
            model=model,
            input_text=case.input,
            output_text=output_text,
            latency_ms=float(rec.get("latency_ms", 0.0)),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            raw_response={"source": "recorded"},
        )
        (cache_dir / f"{ck}.json").write_text(
            json.dumps(asdict(completion), default=str)
        )


def replay_recorded_run(baseline: str, challenger: str,
                         cache_dir: Path | None = None,
                         subgroup_prefix: str = "distractor:"
                         ) -> tuple[RunResult, RunResult, "DriftResultLike"]:
    """Replay the published Opus 4.6 → 4.7 benchmark and return its data.

    Returns ``(baseline_run, challenger_run, drift)`` where ``drift``
    has ``.subgroups`` populated by ``subgroup_prefix``.
    """
    from .comparator import DriftResult  # local import to avoid cycle in types

    base_suite = load_suite("context_rot_reasoning")
    suite = expand_suite(base_suite)
    if not RECORDED.exists():
        raise FileNotFoundError(f"Recorded outcomes not found at {RECORDED}")
    outcomes = yaml.safe_load(RECORDED.read_text()) or {}

    if cache_dir is None:
        cache_dir = ROOT / ".rift" / "cache_demo"

    base_cfg = resolve_model(baseline)
    chal_cfg = resolve_model(challenger)

    prime_cache_from_recording(suite, base_cfg.model, outcomes, cache_dir)
    prime_cache_from_recording(suite, chal_cfg.model, outcomes, cache_dir)

    async def _both():
        b = await run_suite(suite, base_cfg, concurrency=8,
                            cache_dir=str(cache_dir))
        c = await run_suite(suite, chal_cfg, concurrency=8,
                            cache_dir=str(cache_dir))
        return b, c

    base_run, chal_run = asyncio.run(_both())

    drift: DriftResult = compare_runs(
        baseline_scores=base_run.scores,
        challenger_scores=chal_run.scores,
        baseline_model=baseline,
        challenger_model=challenger,
        suite_name=suite.name,
        baseline_costs=[c.cost_usd for c in base_run.cases],
        challenger_costs=[c.cost_usd for c in chal_run.cases],
    )
    drift.subgroups = compare_by_subgroup(
        baseline_scores=base_run.scores,
        challenger_scores=chal_run.scores,
        tags_per_case=[c.tags for c in base_run.cases],
        subgroup_prefix=subgroup_prefix,
        baseline_model=baseline,
        challenger_model=challenger,
        suite_name=suite.name,
        baseline_costs=[c.cost_usd for c in base_run.cases],
        challenger_costs=[c.cost_usd for c in chal_run.cases],
    )
    return base_run, chal_run, drift


# Forward-decl alias to keep replay_recorded_run signature readable.
DriftResultLike = "DriftResult"


# ---------------------------------------------------------------------------
# Demo data structures.
# ---------------------------------------------------------------------------


@dataclass
class DemoAct:
    """One beat in the narrative. Knows how to render itself two ways."""

    title: str
    render_fn: Callable[[Console], None]
    beat_seconds: float
    body_md: str


@dataclass
class VerdictCard:
    """The 'what to do Monday' payload — structured, not freeform."""

    headline: str
    recommendation: str
    action_items: list[str]
    confidence_note: str
    reproduce_cmd: str


@dataclass
class DemoScript:
    """The whole story. One object, three exporters."""

    title: str
    subtitle: str
    baseline_model: str
    challenger_model: str
    suite_name: str
    n_cases: int
    acts: list[DemoAct]
    verdict: VerdictCard
    headline_numbers: dict[str, str]   # {"accuracy_delta": "+6.25pp", ...}
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Narrative cards — demo-specific Rich panels.
# ---------------------------------------------------------------------------


_DISCLOSURE = (
    "Replaying recorded outcomes from "
    "[dim]benchmarks/context_rot_outcomes.yaml[/dim]. "
    "Re-run live against the real APIs with "
    "[dim]python benchmarks/run_context_rot.py --mode live[/dim]."
)


def print_setup_card(script: DemoScript, console: Console) -> None:
    body = (
        f"  [bold]The scenario[/bold]\n"
        f"  Your team just upgraded from [cyan]{script.baseline_model}[/cyan] "
        f"to [cyan]{script.challenger_model}[/cyan].\n"
        f"  Same prompts. Same workflow. The bill goes out at month-end.\n"
        f"\n"
        f"  [bold]The question[/bold]\n"
        f"  Did anything actually break? And what does it cost now?\n"
        f"\n"
        f"  [bold]The data[/bold]\n"
        f"  Suite: [yellow]{script.suite_name}[/yellow] "
        f"({script.n_cases} cases, paired)\n"
        f"  Test:  McNemar's exact (binary scoring)\n"
        f"\n"
        f"  [dim]{_DISCLOSURE}[/dim]"
    )
    console.print(Panel(body,
                        title="[bold]ACT 1 — The upgrade[/bold]",
                        border_style="cyan"))


def print_act_card(act_number: int, title: str, prelude: str,
                   console: Console, border: str = "white") -> None:
    """A transition card between acts — sets up the next beat."""
    body = f"  [bold]{prelude}[/bold]"
    console.print(Panel(body,
                        title=f"[bold]ACT {act_number} — {title}[/bold]",
                        border_style=border))


def print_verdict_panel(card: VerdictCard, console: Console) -> None:
    lines = [
        f"  [bold]{card.headline}[/bold]",
        "",
        f"  [bold]Recommendation[/bold]",
        f"  {card.recommendation}",
        "",
        f"  [bold]Action items[/bold]",
    ]
    for item in card.action_items:
        lines.append(f"   • {item}")
    lines += [
        "",
        f"  [dim]{card.confidence_note}[/dim]",
        "",
        f"  [bold]Reproduce[/bold]",
        f"  [dim]{card.reproduce_cmd}[/dim]",
    ]
    console.print(Panel("\n".join(lines),
                        title="[bold]VERDICT — what to do Monday[/bold]",
                        border_style="magenta"))


# ---------------------------------------------------------------------------
# Script factory.
# ---------------------------------------------------------------------------


def build_opus47_demo_script(base_run: RunResult, chal_run: RunResult,
                              drift) -> DemoScript:
    """Construct the four-act script for the Opus 4.6 → 4.7 scenario.

    All numbers come from ``base_run`` / ``chal_run`` / ``drift``. Only
    narration text is hardcoded.
    """
    delta_pp = drift.delta * 100.0
    base_cpc = drift.baseline_cost_per_correct
    chal_cpc = drift.challenger_cost_per_correct
    cpc_pct = ((chal_cpc - base_cpc) / base_cpc * 100.0) if base_cpc else 0.0
    in_ratio = (chal_run.total_input_tokens / base_run.total_input_tokens
                if base_run.total_input_tokens else 1.0)

    # ------------------------------------------------------------------
    # Act 2 — quality (accuracy without cost).
    # ------------------------------------------------------------------
    def _act2(c: Console) -> None:
        print_act_card(
            2, "What a casual eval sees",
            "Accuracy. The number every benchmark leaderboard reports.",
            c, border="green",
        )
        print_drift_report(drift, base_run, chal_run, cost=False, console=c)

    act2_md = (
        "## Act 2 — What a casual eval sees\n\n"
        "Accuracy is the number every benchmark leaderboard reports. By "
        "this measure, the upgrade looks like a win:\n\n"
        f"- Baseline mean: **{drift.baseline_mean:.4f}**\n"
        f"- Challenger mean: **{drift.challenger_mean:.4f}**\n"
        f"- Delta: **{drift.delta:+.4f} ({delta_pp:+.2f}pp)**\n"
        f"- p-value: {drift.p_value:.4f}  ·  test: {drift.test_used}\n"
        f"- 95% CI: [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}]\n\n"
        "Headline reading: newer model, more correct answers. Ship it.\n"
    )

    # ------------------------------------------------------------------
    # Act 3 — cost twist.
    # ------------------------------------------------------------------
    def _act3(c: Console) -> None:
        print_act_card(
            3, "What Rift sees",
            "Same prompts. Same correctness. But the bill tells a different "
            "story.",
            c, border="red",
        )
        print_cost_panel(drift, console=c)
        # Token inflation explainer — this is the WHY.
        tbl = Table(title="Token usage — same prompts, different tokenizer",
                    show_lines=False)
        tbl.add_column("Model", style="bold")
        tbl.add_column("Total input tokens", justify="right")
        tbl.add_column("Output tokens", justify="right")
        tbl.add_column("Input ratio vs baseline", justify="right")
        tbl.add_row(
            drift.baseline_model,
            f"{base_run.total_input_tokens:,}",
            f"{base_run.total_output_tokens:,}",
            "1.000×  (baseline)",
        )
        tbl.add_row(
            drift.challenger_model,
            f"{chal_run.total_input_tokens:,}",
            f"{chal_run.total_output_tokens:,}",
            f"[red]{in_ratio:.3f}×[/red]",
        )
        c.print(tbl)

    act3_md = (
        "## Act 3 — What Rift sees\n\n"
        "Same prompts. Same answers (mostly). But the bill tells a different "
        "story:\n\n"
        f"- Baseline spend: **{_fmt_cost(drift.baseline_cost_usd)}**\n"
        f"- Challenger spend: **{_fmt_cost(drift.challenger_cost_usd)}**\n"
        f"- Baseline $/correct: **{_fmt_cost(base_cpc)}**\n"
        f"- Challenger $/correct: **{_fmt_cost(chal_cpc)}**  "
        f"(**{cpc_pct:+.1f}%**)\n\n"
        "**The why.** For byte-identical prompts, the challenger emits "
        f"**{in_ratio:.2f}× more input tokens** than the baseline "
        f"({base_run.total_input_tokens:,} → "
        f"{chal_run.total_input_tokens:,}). At list-price parity, this is "
        "a silent per-prompt cost increase on migration. Accuracy doesn't "
        "pay for it.\n"
    )

    # ------------------------------------------------------------------
    # Verdict — built first so Act 4 can close over it cleanly.
    # ------------------------------------------------------------------
    verdict = VerdictCard(
        headline=(
            f"Accuracy ticked up (+{delta_pp:.2f}pp, not significant at α=0.05), "
            f"but $/correct rose {cpc_pct:+.1f}%."
        ),
        recommendation=(
            "Do NOT migrate short-prompt workloads to "
            f"{drift.challenger_model} on price-parity assumptions. The "
            "quality lift on this suite does not pay for tokenizer inflation."
        ),
        action_items=[
            f"Pin {drift.baseline_model} for short-prompt production paths "
            "until tokenizer parity is restored.",
            f"Consider {drift.challenger_model} only for long-context "
            "workloads (8k+ distractor tokens) where the robustness lift "
            "is largest.",
            "Re-evaluate every quarter — tokenizer changes ship without "
            "release-note announcements.",
            "Add `rift compare` to CI for any model-version bump touching "
            "production prompts.",
        ],
        confidence_note=(
            f"Findings replicate on the committed outcomes file (n="
            f"{drift.n_cases}, paired, McNemar's exact). Significance "
            "threshold α=0.05; cost figures use list pricing — apply "
            "your enterprise multiplier for contracted rates."
        ),
        reproduce_cmd=(
            "rift demo  # or: python benchmarks/run_context_rot.py --mode record"
        ),
    )

    # ------------------------------------------------------------------
    # Act 4 — drill-down + verdict.
    # ------------------------------------------------------------------
    def _act4(c: Console) -> None:
        print_act_card(
            4, "Where the cost concentrates",
            "By context length — does the inflation hit you everywhere, "
            "or only on long prompts?",
            c, border="yellow",
        )
        if drift.subgroups:
            print_subgroup_table(
                drift.subgroups, title="By distractor context length",
                console=c,
            )
        print_verdict_panel(verdict, c)

    subgroup_md_rows = []
    if drift.subgroups:
        for tag in sorted(drift.subgroups.keys()):
            d = drift.subgroups[tag]
            subgroup_md_rows.append(
                f"| {tag} | {d.n_cases} | {d.baseline_mean:.3f} | "
                f"{d.challenger_mean:.3f} | {d.delta:+.3f} | "
                f"{d.cost_normalized_delta_usd:+.4f} |"
            )
    act4_md = (
        "## Act 4 — Where the cost concentrates\n\n"
        "Does the inflation hit you everywhere, or only on long prompts?\n\n"
        "| Subgroup | n | Baseline | Challenger | Δ acc | Δ $/correct |\n"
        "|----------|---|----------|------------|-------|-------------|\n"
        + "\n".join(subgroup_md_rows) + "\n"
    )

    # ------------------------------------------------------------------
    # Compose script.
    # ------------------------------------------------------------------
    script = DemoScript(
        title="Rift demo — Opus 4.6 → 4.7",
        subtitle=(
            "A guided walkthrough of one real upgrade. Live in your terminal."
        ),
        baseline_model=drift.baseline_model,
        challenger_model=drift.challenger_model,
        suite_name=drift.suite_name,
        n_cases=drift.n_cases,
        acts=[
            DemoAct(
                title="The upgrade",
                render_fn=lambda c: print_setup_card(script, c),
                beat_seconds=3.5,
                body_md=(
                    "## Act 1 — The upgrade\n\n"
                    f"Your team upgraded **{drift.baseline_model}** → "
                    f"**{drift.challenger_model}**. Same prompts, same "
                    "workflow. The bill goes out at month-end. Did anything "
                    "actually break — and what does it cost now?\n\n"
                    f"_Suite: `{drift.suite_name}` ({drift.n_cases} cases, "
                    "paired, McNemar's exact). Replaying committed outcomes "
                    "for reproducibility — re-run live with `--mode live`._\n"
                ),
            ),
            DemoAct(
                title="What a casual eval sees",
                render_fn=_act2,
                beat_seconds=4.5,
                body_md=act2_md,
            ),
            DemoAct(
                title="What Rift sees",
                render_fn=_act3,
                beat_seconds=5.5,
                body_md=act3_md,
            ),
            DemoAct(
                title="Where the cost concentrates",
                render_fn=_act4,
                beat_seconds=4.5,
                body_md=act4_md,
            ),
        ],
        verdict=verdict,
        headline_numbers={
            "accuracy_delta": f"{delta_pp:+.2f}pp",
            "p_value": f"{drift.p_value:.3f}",
            "cost_per_correct_pct": f"{cpc_pct:+.1f}%",
            "baseline_cpc": _fmt_cost(base_cpc),
            "challenger_cpc": _fmt_cost(chal_cpc),
            "input_token_ratio": f"{in_ratio:.3f}×",
        },
        sources=[
            "benchmarks/context_rot_outcomes.yaml",
            "benchmarks/context_rot_opus47_analysis.md",
        ],
    )
    return script


# ---------------------------------------------------------------------------
# Terminal renderer.
# ---------------------------------------------------------------------------


def _is_recording_or_pipe(console: Console) -> bool:
    """True when output is being captured (no TTY) — skip sleeps."""
    return not console.is_terminal


def run_demo(script: DemoScript, auto: bool = True,
              beat_multiplier: float = 1.0,
              console: Console | None = None,
              no_clear: bool = False) -> None:
    """Render the script to the terminal, paced.

    ``auto=True`` (default) plays straight through with sleeps between
    acts. ``auto=False`` reads stdin between acts so a live presenter
    can drive the pacing.

    Skips sleeps when output is not a TTY (e.g. ``rift demo | tee``),
    so capture and CI runs don't waste recording length.
    """
    if console is None:
        console = Console()

    if not no_clear and console.is_terminal:
        console.clear()

    # Title card.
    console.print()
    console.rule(f"[bold magenta]{script.title}[/bold magenta]")
    console.print(f"  [dim italic]{script.subtitle}[/dim italic]")
    console.print()

    skip_sleeps = _is_recording_or_pipe(console) or beat_multiplier == 0.0

    try:
        for i, act in enumerate(script.acts):
            act.render_fn(console)
            if i == len(script.acts) - 1:
                break  # no pause after the last act
            if auto:
                if not skip_sleeps:
                    time.sleep(act.beat_seconds * beat_multiplier)
            else:
                console.print(
                    "  [dim]press Enter to continue…[/dim]",
                    end="",
                )
                try:
                    input()
                except EOFError:
                    # Non-interactive stdin (piped). Fall through.
                    pass
            console.print()
    except KeyboardInterrupt:
        console.print("\n[dim]demo aborted[/dim]")
        return

    console.print()
    console.rule("[bold magenta]end of demo[/bold magenta]")
    console.print()


# ---------------------------------------------------------------------------
# Markdown export.
# ---------------------------------------------------------------------------


def export_demo_markdown(script: DemoScript, path: str | Path) -> None:
    """Write a single markdown file: title, acts, verdict, sources."""
    parts: list[str] = []
    parts.append(f"# {script.title}\n")
    parts.append(f"_{script.subtitle}_\n")
    parts.append(
        f"**Baseline:** `{script.baseline_model}`  ·  "
        f"**Challenger:** `{script.challenger_model}`  ·  "
        f"**Suite:** `{script.suite_name}` ({script.n_cases} cases)\n"
    )
    parts.append("\n## Headline numbers\n")
    parts.append("| | |\n|---|---|")
    for k, v in script.headline_numbers.items():
        parts.append(f"| {k.replace('_', ' ')} | **{v}** |")
    parts.append("")

    for act in script.acts:
        parts.append(act.body_md.rstrip())
        parts.append("")

    v = script.verdict
    parts.append("## Verdict — what to do Monday\n")
    parts.append(f"**{v.headline}**\n")
    parts.append(f"**Recommendation.** {v.recommendation}\n")
    parts.append("**Action items.**\n")
    for item in v.action_items:
        parts.append(f"- {item}")
    parts.append("")
    parts.append(f"_{v.confidence_note}_\n")
    parts.append(f"**Reproduce.** `{v.reproduce_cmd}`\n")

    if script.sources:
        parts.append("## Sources\n")
        for s in script.sources:
            parts.append(f"- `{s}`")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# HTML export — single self-contained file, inline SVG charts.
# ---------------------------------------------------------------------------


_HTML_CSS = """
:root {
  --accent: #6c5ce7;
  --warn: #d63031;
  --good: #00b894;
  --ink: #1a1a1a;
  --mute: #6b6b6b;
  --bg: #fafafa;
  --card: #ffffff;
  --border: #e6e6e6;
}
* { box-sizing: border-box; }
body {
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        'Helvetica Neue', Arial, sans-serif;
  color: var(--ink);
  background: var(--bg);
  margin: 0;
  padding: 32px 16px 64px;
}
.wrap { max-width: 760px; margin: 0 auto; }
header { margin-bottom: 32px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
.subtitle { color: var(--mute); margin: 0 0 16px; font-size: 16px; }
.badges { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 8px; }
.badge {
  background: #f0eefb; color: var(--accent); padding: 3px 10px;
  border-radius: 999px; font-size: 12px; font-weight: 600;
}
.badge.warn { background: #fdecec; color: var(--warn); }
.badge.good { background: #e6f8f2; color: var(--good); }
section {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 22px 24px; margin: 16px 0;
}
section h2 {
  margin: 0 0 6px; font-size: 18px; letter-spacing: -0.005em;
}
section p { margin: 6px 0; }
.lede { color: var(--mute); margin-bottom: 14px; }
.kpi-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 12px; margin: 12px 0 4px;
}
.kpi {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; background: #fff;
}
.kpi .v { font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
.kpi .v.warn { color: var(--warn); }
.kpi .v.good { color: var(--good); }
.kpi .l {
  font-size: 11px; color: var(--mute);
  text-transform: uppercase; letter-spacing: 0.05em;
  margin-top: 2px;
}
table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 14px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--mute); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.04em; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
.callout {
  border-left: 3px solid var(--accent); background: #f7f6fd;
  padding: 12px 16px; border-radius: 6px; margin: 14px 0;
}
.callout.warn { border-left-color: var(--warn); background: #fdf3f3; }
.callout.good { border-left-color: var(--good); background: #effaf5; }
.callout h3 { margin: 0 0 6px; font-size: 14px; text-transform: uppercase;
               letter-spacing: 0.06em; color: var(--mute); }
.action-list { margin: 6px 0 0; padding-left: 18px; }
.action-list li { margin: 4px 0; }
code, .mono {
  font: 13px/1.5 ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  background: #f3f3f3; padding: 1px 5px; border-radius: 4px;
}
pre {
  background: #1a1a1a; color: #f3f3f3;
  border-radius: 8px; padding: 14px 16px; overflow-x: auto;
  font: 13px/1.5 ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}
footer {
  margin-top: 32px; color: var(--mute); font-size: 12px; text-align: center;
}
svg { display: block; margin: 12px auto; max-width: 100%; height: auto; }
@media print {
  body { background: #fff; padding: 12px; }
  section { break-inside: avoid; box-shadow: none; }
  .kpi-grid { break-inside: avoid; }
}
"""


def _svg_bar_chart(labels: list[str], values: list[float],
                   value_labels: list[str], colors: list[str],
                   title: str, width: int = 520, height: int = 200) -> str:
    """Hand-rolled SVG bar chart. No external deps."""
    pad_l, pad_r, pad_t, pad_b = 130, 30, 30, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0
    bar_h = inner_h / max(1, len(values))
    bar_thickness = bar_h * 0.55
    bars: list[str] = []
    for i, (lab, val, vlab, col) in enumerate(
            zip(labels, values, value_labels, colors)):
        y = pad_t + i * bar_h + (bar_h - bar_thickness) / 2
        w = (val / max_v) * inner_w
        bars.append(
            f'<text x="{pad_l - 10}" y="{y + bar_thickness/2 + 4}" '
            f'text-anchor="end" font-size="13" fill="#1a1a1a">'
            f'{html.escape(lab)}</text>'
        )
        bars.append(
            f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" '
            f'height="{bar_thickness:.1f}" fill="{col}" rx="3"/>'
        )
        # value label inside bar if wide enough, else outside
        if w > 60:
            tx = pad_l + w - 8
            anchor = "end"
            tfill = "#fff"
        else:
            tx = pad_l + w + 6
            anchor = "start"
            tfill = "#1a1a1a"
        bars.append(
            f'<text x="{tx:.1f}" y="{y + bar_thickness/2 + 4}" '
            f'text-anchor="{anchor}" font-size="12" font-weight="600" '
            f'fill="{tfill}">{html.escape(vlab)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        + "".join(bars)
        + "</svg>"
    )


def _svg_grouped_bar(group_labels: list[str], series_labels: list[str],
                     values: list[list[float]],   # values[group][series]
                     value_labels: list[list[str]],
                     colors: list[str],
                     title: str,
                     width: int = 560, height: int = 240) -> str:
    """Grouped bar — used for the subgroup drill-down."""
    pad_l, pad_r, pad_t, pad_b = 80, 24, 40, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n_groups = len(group_labels)
    n_series = len(series_labels)
    group_w = inner_w / max(1, n_groups)
    bar_w = group_w * 0.7 / n_series
    flat = [v for row in values for v in row]
    max_v = max(flat) if flat else 1.0
    if max_v <= 0:
        max_v = 1.0
    parts: list[str] = []
    # Legend
    legend_x = pad_l
    for i, sl in enumerate(series_labels):
        parts.append(
            f'<rect x="{legend_x}" y="10" width="12" height="12" '
            f'fill="{colors[i]}" rx="2"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="20" font-size="12" fill="#1a1a1a">'
            f'{html.escape(sl)}</text>'
        )
        legend_x += 22 + 8 * len(sl) + 16
    # Bars
    for gi, gl in enumerate(group_labels):
        gx = pad_l + gi * group_w + group_w * 0.15
        for si in range(n_series):
            v = values[gi][si]
            h = (v / max_v) * inner_h
            x = gx + si * bar_w
            y = pad_t + inner_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" '
                f'height="{h:.1f}" fill="{colors[si]}" rx="2"/>'
            )
            vl = value_labels[gi][si]
            parts.append(
                f'<text x="{x + (bar_w-2)/2:.1f}" y="{y - 4:.1f}" '
                f'text-anchor="middle" font-size="10" '
                f'fill="#1a1a1a">{html.escape(vl)}</text>'
            )
        parts.append(
            f'<text x="{pad_l + gi*group_w + group_w/2:.1f}" '
            f'y="{height - 12}" text-anchor="middle" font-size="12" '
            f'fill="#1a1a1a">{html.escape(gl)}</text>'
        )
    # Baseline axis
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + inner_h}" '
        f'x2="{width - pad_r}" y2="{pad_t + inner_h}" '
        f'stroke="#cccccc" stroke-width="1"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        + "".join(parts)
        + "</svg>"
    )


def _render_html(script: DemoScript, base_run: RunResult,
                  chal_run: RunResult, drift) -> str:
    """Build the full HTML one-pager from a script + the underlying runs."""
    base_acc = drift.baseline_mean
    chal_acc = drift.challenger_mean
    base_cpc = drift.baseline_cost_per_correct
    chal_cpc = drift.challenger_cost_per_correct
    cpc_pct = ((chal_cpc - base_cpc) / base_cpc * 100.0) if base_cpc else 0.0
    delta_pp = drift.delta * 100.0
    in_ratio = (chal_run.total_input_tokens / base_run.total_input_tokens
                if base_run.total_input_tokens else 1.0)

    # ---- Chart 1: accuracy bars ------------------------------------------
    acc_svg = _svg_bar_chart(
        labels=[script.baseline_model, script.challenger_model],
        values=[base_acc, chal_acc],
        value_labels=[f"{base_acc:.3f}", f"{chal_acc:.3f}"],
        colors=["#9b9b9b", "#00b894"],
        title="Accuracy by model",
        width=560, height=140,
    )

    # ---- Chart 2: cost-per-correct bars ----------------------------------
    cpc_svg = _svg_bar_chart(
        labels=[script.baseline_model, script.challenger_model],
        values=[base_cpc, chal_cpc],
        value_labels=[_fmt_cost(base_cpc), _fmt_cost(chal_cpc)],
        colors=["#9b9b9b", "#d63031"],
        title="$/correct by model",
        width=560, height=140,
    )

    # ---- Chart 3: subgroup grouped bar -----------------------------------
    subgroup_svg = ""
    if drift.subgroups:
        keys = sorted(drift.subgroups.keys())
        # Show $/correct by subgroup, both models, side-by-side.
        # values[group][series] where series = (baseline, challenger)
        values = []
        labels = []
        for k in keys:
            d = drift.subgroups[k]
            bcpc = d.baseline_cost_per_correct
            ccpc = d.challenger_cost_per_correct
            values.append([bcpc, ccpc])
            labels.append([_fmt_cost(bcpc), _fmt_cost(ccpc)])
        subgroup_svg = _svg_grouped_bar(
            group_labels=[k.replace("distractor:", "") for k in keys],
            series_labels=[script.baseline_model, script.challenger_model],
            values=values,
            value_labels=labels,
            colors=["#9b9b9b", "#d63031"],
            title="$/correct by distractor context length",
            width=560, height=220,
        )

    # ---- KPI grid --------------------------------------------------------
    kpis = (
        f'<div class="kpi"><div class="v">{delta_pp:+.2f}pp</div>'
        f'<div class="l">accuracy delta</div></div>'
        f'<div class="kpi"><div class="v warn">{cpc_pct:+.1f}%</div>'
        f'<div class="l">$/correct delta</div></div>'
        f'<div class="kpi"><div class="v warn">{in_ratio:.2f}×</div>'
        f'<div class="l">input tokens</div></div>'
    )

    # ---- Subgroup table --------------------------------------------------
    subgroup_rows = ""
    if drift.subgroups:
        for k in sorted(drift.subgroups.keys()):
            d = drift.subgroups[k]
            cpc_delta_pct = (
                (d.challenger_cost_per_correct - d.baseline_cost_per_correct)
                / d.baseline_cost_per_correct * 100.0
                if d.baseline_cost_per_correct else 0.0
            )
            subgroup_rows += (
                f"<tr><td>{html.escape(k)}</td>"
                f"<td class='num'>{d.n_cases}</td>"
                f"<td class='num'>{d.baseline_mean:.3f}</td>"
                f"<td class='num'>{d.challenger_mean:.3f}</td>"
                f"<td class='num'>{d.delta:+.3f}</td>"
                f"<td class='num'>{_fmt_cost(d.baseline_cost_per_correct)}</td>"
                f"<td class='num'>{_fmt_cost(d.challenger_cost_per_correct)}</td>"
                f"<td class='num'>{cpc_delta_pct:+.1f}%</td></tr>"
            )

    # ---- Verdict + action items -----------------------------------------
    actions_html = "".join(
        f"<li>{html.escape(a)}</li>" for a in script.verdict.action_items
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{html.escape(script.title)}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="badges">
    <span class="badge">Rift demo</span>
    <span class="badge warn">cost-per-correct {cpc_pct:+.1f}%</span>
    <span class="badge">replay · n={script.n_cases}</span>
  </div>
  <h1>{html.escape(script.title)}</h1>
  <p class="subtitle">{html.escape(script.subtitle)}</p>
</header>

<section>
  <h2>What we tested</h2>
  <p class="lede">
    Paired comparison of <code>{html.escape(script.baseline_model)}</code> and
    <code>{html.escape(script.challenger_model)}</code> on the
    <code>{html.escape(script.suite_name)}</code> suite ({script.n_cases}
    cases, McNemar's exact). Data replayed from committed outcomes for
    reproducibility.
  </p>
  <div class="kpi-grid">{kpis}</div>
</section>

<section>
  <h2>Act 1 — Quality (what a casual eval sees)</h2>
  <p class="lede">
    Accuracy is the number every benchmark leaderboard reports. By this
    measure, the upgrade looks like a win.
  </p>
  {acc_svg}
  <div class="callout good">
    <h3>Headline reading</h3>
    Accuracy <b>{delta_pp:+.2f}pp</b> (p = {drift.p_value:.3f}, not
    significant at α=0.05). 95% CI: [{drift.ci_lower:+.3f},
    {drift.ci_upper:+.3f}].
  </div>
</section>

<section>
  <h2>Act 2 — Cost (what Rift sees)</h2>
  <p class="lede">
    Same prompts. Same correctness. But the bill tells a different story.
  </p>
  {cpc_svg}
  <div class="callout warn">
    <h3>The twist</h3>
    $/correct rose <b>{cpc_pct:+.1f}%</b>
    ({_fmt_cost(base_cpc)} → {_fmt_cost(chal_cpc)}). For byte-identical
    prompts, the challenger emits <b>{in_ratio:.2f}× more input
    tokens</b> ({base_run.total_input_tokens:,} →
    {chal_run.total_input_tokens:,}). At list-price parity, this is a
    silent per-prompt cost increase on migration.
  </div>
</section>

{"<section><h2>Act 3 — Where the cost concentrates</h2><p class='lede'>By context length — does the inflation hit you everywhere, or only on long prompts?</p>" + subgroup_svg + "<table><thead><tr><th>Subgroup</th><th>n</th><th>Baseline acc</th><th>Challenger acc</th><th>Δ acc</th><th>Baseline $/correct</th><th>Challenger $/correct</th><th>Δ %</th></tr></thead><tbody>" + subgroup_rows + "</tbody></table></section>" if drift.subgroups else ""}

<section>
  <h2>Verdict — what to do Monday</h2>
  <div class="callout">
    <h3>Headline</h3>
    {html.escape(script.verdict.headline)}
  </div>
  <p><b>Recommendation.</b> {html.escape(script.verdict.recommendation)}</p>
  <p><b>Action items.</b></p>
  <ul class="action-list">{actions_html}</ul>
  <p class="lede" style="margin-top:14px">
    {html.escape(script.verdict.confidence_note)}
  </p>
</section>

<section>
  <h2>Reproduce this</h2>
  <p class="lede">
    Runs offline, no API key required. The recorded outcomes are committed
    to the repo.
  </p>
  <pre>{html.escape(script.verdict.reproduce_cmd)}</pre>
  <p class="lede">Sources:
    {", ".join(f"<code>{html.escape(s)}</code>" for s in script.sources)}
  </p>
</section>

<footer>
  Generated by <b>Rift</b> · {generated} · self-contained, no external assets
</footer>
</div>
</body>
</html>
"""


def export_demo_html(script: DemoScript, path: str | Path,
                      base_run: RunResult, chal_run: RunResult,
                      drift) -> None:
    """Write the HTML executive memo to ``path`` (single self-contained file)."""
    html_text = _render_html(script, base_run, chal_run, drift)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html_text)


# ---------------------------------------------------------------------------
# SVG export — for README embedding.
# ---------------------------------------------------------------------------


def export_demo_svg(script: DemoScript, path: str | Path) -> None:
    """Write a single SVG capturing the terminal render of the whole demo.

    Uses Rich's ``Console(record=True).export_svg(...)``. The result is
    a single self-contained SVG that GitHub renders inline in READMEs.
    """
    console = Console(record=True, width=100, force_terminal=True)
    # Render straight through, no clears, no sleeps.
    run_demo(script, auto=True, beat_multiplier=0.0,
              console=console, no_clear=True)
    svg = console.export_svg(title=script.title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(svg)


# ---------------------------------------------------------------------------
# Scenarios registry — extensible for future scenarios.
# ---------------------------------------------------------------------------


SCENARIOS: dict[str, dict] = {
    "opus-46-vs-47": {
        "baseline": "opus-4-6",
        "challenger": "opus-4-7",
        "subgroup_prefix": "distractor:",
        "build": build_opus47_demo_script,
    },
}


def load_scenario(name: str) -> tuple[DemoScript, RunResult, RunResult, object]:
    """Resolve a scenario name to a fully-populated demo + underlying runs."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown demo scenario: {name!r}. "
            f"Known: {sorted(SCENARIOS)}"
        )
    spec = SCENARIOS[name]
    base_run, chal_run, drift = replay_recorded_run(
        baseline=spec["baseline"],
        challenger=spec["challenger"],
        subgroup_prefix=spec["subgroup_prefix"],
    )
    script = spec["build"](base_run, chal_run, drift)
    return script, base_run, chal_run, drift
