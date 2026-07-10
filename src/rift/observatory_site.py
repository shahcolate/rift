"""Static site generator for the Rift Observatory.

Renders an observatory data directory (``index.jsonl`` + ``events.jsonl``
+ records) into a self-contained static site: a front page with the
drift feed and per-endpoint summary cards, and one timeline page per
endpoint with metric charts and the fingerprint history.

Same constraints as the demo's HTML export (:mod:`rift.demo`): zero
JavaScript, zero external assets, hand-rolled SVG. A page must render
from a file:// URL, an artifact viewer, and GitHub Pages identically —
the dashboard is a publication, not an app. All data values are
sanitized (NaN/inf dropped, text HTML-escaped) so a degenerate record
can never produce broken markup.
"""

from __future__ import annotations

import hashlib
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .observatory import (
    endpoint_slug,
    finite_or_none as _finite,
    load_events,
    load_index,
    load_selftest,
)

# Shared palette with rift.demo's HTML memo — one visual identity.
_SERIES_COLORS = ("#6c5ce7", "#00b894", "#d63031", "#e17055",
                  "#0984e3", "#9b9b9b")
_MARKER_COLOR = "#d63031"

_KIND_LABELS: dict[str, tuple[str, str]] = {
    "score_drift": ("score drift", "warn"),
    "silent_swap": ("silent swap", "warn"),
    "fingerprint_change": ("fingerprint change", "info"),
    "rollout": ("mid-run rollout", "warn"),
    "panel_changed": ("panel changed", "info"),
    "notice": ("notice", "info"),
}

_CSS = """
:root {
  --border: #e3e3e3; --mute: #777; --accent: #6c5ce7;
  --warn: #d63031; --good: #00b894; --bg: #fafafa;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px 16px 60px; background: var(--bg); color: #1a1a1a;
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        Helvetica, Arial, sans-serif;
}
.wrap { max-width: 880px; margin: 0 auto; }
header h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: -0.02em; }
header .subtitle { color: var(--mute); margin: 0 0 18px; }
.badges { margin-bottom: 10px; }
.badge {
  display: inline-block; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  border: 1px solid var(--border); border-radius: 12px;
  padding: 2px 10px; margin-right: 6px; background: #fff; color: var(--mute);
}
.badge.warn { color: var(--warn); border-color: var(--warn); }
section {
  background: #fff; border: 1px solid var(--border); border-radius: 10px;
  padding: 20px 22px; margin: 16px 0;
}
section h2 { margin: 0 0 10px; font-size: 17px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.card {
  border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
  background: #fff;
}
.card a { color: var(--accent); text-decoration: none; font-weight: 600; }
.card .meta { font-size: 12px; color: var(--mute); margin-top: 2px; }
.feed { list-style: none; margin: 0; padding: 0; }
.feed li {
  border-left: 3px solid var(--border); padding: 8px 14px; margin: 8px 0;
  background: #fff;
}
.feed li.warn { border-left-color: var(--warn); }
.feed li.info { border-left-color: var(--accent); }
.feed .kind {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; margin-right: 8px;
}
.feed li.warn .kind { color: var(--warn); }
.feed li.info .kind { color: var(--accent); }
.feed .date { color: var(--mute); font-size: 12px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 14px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--mute); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.04em; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
code, .mono {
  font: 12.5px/1.5 ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  background: #f3f3f3; padding: 1px 5px; border-radius: 4px;
}
svg { display: block; margin: 10px 0; max-width: 100%; height: auto; }
footer { margin-top: 28px; color: var(--mute); font-size: 12px; text-align: center; }
footer a { color: var(--mute); }
.empty { color: var(--mute); font-style: italic; }
"""


def _svg_line_chart(
    dates: list[str],
    series: dict[str, list[float | None]],
    title: str,
    y_fmt=lambda v: f"{v:.2f}",
    markers: list[int] | None = None,
    width: int = 680,
    height: int = 220,
) -> str:
    """Hand-rolled SVG timeline. No external deps.

    ``series`` maps a label to one value per date (``None`` = gap; the
    polyline breaks rather than interpolating across missing weeks).
    ``markers`` are date indices drawn as dashed verticals — fingerprint
    changes. Inputs are sanitized so the markup is always valid.
    """
    markers = markers or []
    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 34
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(dates)
    clean = {label: [_finite(v) for v in vals] for label, vals in series.items()}

    finite_vals = [v for vals in clean.values() for v in vals if v is not None]
    if not finite_vals:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{html.escape(title)}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'font-size="13" fill="#777">no data</text></svg>'
        )
    lo, hi = min(finite_vals), max(finite_vals)
    if hi - lo < 1e-9:
        pad = abs(hi) * 0.1 or 0.05
        lo, hi = lo - pad, hi + pad
    else:
        span = hi - lo
        lo, hi = lo - span * 0.1, hi + span * 0.1

    def x_at(i: int) -> float:
        return pad_l + (inner_w * i / max(1, n - 1) if n > 1 else inner_w / 2)

    def y_at(v: float) -> float:
        return pad_t + inner_h * (1.0 - (v - lo) / (hi - lo))

    parts: list[str] = []
    # Y axis: top / mid / bottom gridlines with labels.
    for frac in (0.0, 0.5, 1.0):
        v = lo + (hi - lo) * (1.0 - frac)
        y = pad_t + inner_h * frac
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
            f'y2="{y:.1f}" stroke="#eeeeee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#777">{html.escape(y_fmt(v))}</text>'
        )
    # Fingerprint-change markers.
    for mi in markers:
        if 0 <= mi < n:
            x = x_at(mi)
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                f'y2="{pad_t + inner_h}" stroke="{_MARKER_COLOR}" '
                f'stroke-width="1" stroke-dasharray="4,3"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{pad_t - 6}" text-anchor="middle" '
                f'font-size="10" fill="{_MARKER_COLOR}">fp</text>'
            )
    # Series polylines (broken at gaps) + points.
    legend_x = pad_l
    for si, (label, vals) in enumerate(clean.items()):
        color = _SERIES_COLORS[si % len(_SERIES_COLORS)]
        segment: list[str] = []
        for i, v in enumerate(vals[:n]):
            if v is None:
                if len(segment) > 1:
                    parts.append(
                        f'<polyline points="{" ".join(segment)}" fill="none" '
                        f'stroke="{color}" stroke-width="2"/>'
                    )
                segment = []
                continue
            segment.append(f"{x_at(i):.1f},{y_at(v):.1f}")
            parts.append(
                f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="2.5" '
                f'fill="{color}"/>'
            )
        if len(segment) > 1:
            parts.append(
                f'<polyline points="{" ".join(segment)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        parts.append(
            f'<rect x="{legend_x}" y="{height - 16}" width="10" height="10" '
            f'fill="{color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{legend_x + 15}" y="{height - 7}" font-size="11" '
            f'fill="#1a1a1a">{html.escape(label)}</text>'
        )
        legend_x += 22 + 7 * len(label) + 14
    # X labels: first / last (middle too when room).
    label_idx = sorted({0, n - 1} | ({n // 2} if n > 4 else set()))
    for i in label_idx:
        if 0 <= i < n:
            parts.append(
                f'<text x="{x_at(i):.1f}" y="{pad_t + inner_h + 16}" '
                f'text-anchor="middle" font-size="10" fill="#777">'
                f'{html.escape(dates[i])}</text>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        + "".join(parts)
        + "</svg>"
    )


def _page(title: str, body: str, root_prefix: str = "") -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{html.escape(title)}</title>
<link rel="alternate" type="application/rss+xml"
      title="Rift Observatory drift feed" href="{root_prefix}feed.xml"/>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
{body}
<footer>
  Generated by <b>Rift Observatory</b> · {generated} ·
  <a href="{root_prefix}data/index.jsonl">raw data</a> ·
  self-contained, no external assets
</footer>
</div>
</body>
</html>
"""


def _endpoint_dates(index: list[dict], endpoint: str) -> list[str]:
    return sorted({e["date"] for e in index if e["endpoint"] == endpoint})


def _fingerprint_union(index: list[dict], endpoint: str, date: str) -> list[str]:
    return sorted({
        fp for e in index
        if e["endpoint"] == endpoint and e["date"] == date
        for fp in e.get("fingerprints", [])
    })


def _fingerprint_change_marks(index: list[dict], endpoint: str,
                              dates: list[str]) -> list[int]:
    """Date indices where the endpoint's fingerprint union changed."""
    marks: list[int] = []
    prev: list[str] | None = None
    for i, d in enumerate(dates):
        fps = _fingerprint_union(index, endpoint, d)
        if prev is not None and fps and prev and fps != prev:
            marks.append(i)
        if fps:
            prev = fps
    return marks


def _metric_by_date(index: list[dict], endpoint: str, suite: str,
                    dates: list[str], key: str) -> list[float | None]:
    by_date = {
        e["date"]: e for e in index
        if e["endpoint"] == endpoint and e["suite"] == suite
        and not e.get("aborted")
    }
    return [_finite(by_date.get(d, {}).get(key)) for d in dates]


def _feed_html(events: list[dict]) -> str:
    if not events:
        return '<p class="empty">No drift events recorded yet.</p>'
    items: list[str] = []
    ordered = sorted(events, key=lambda e: e.get("date", ""), reverse=True)
    for ev in ordered:
        label, klass = _KIND_LABELS.get(ev.get("kind", ""), (ev.get("kind", "?"), "info"))
        items.append(
            f'<li class="{klass}"><span class="kind">{html.escape(label)}</span>'
            f'<span class="date">{html.escape(ev.get("date", ""))}</span><br>'
            f'{html.escape(ev.get("summary", ""))}</li>'
        )
    return f'<ul class="feed">{"".join(items)}</ul>'


def _feed_xml(events: list[dict], site_url: str = "") -> str:
    """RSS 2.0 drift feed — subscribe to model-behavior changes.

    Same zero-dependency, string-template idiom as the rest of the site.
    The ``guid`` is stable across re-renders (date/endpoint/suite/kind) so
    readers dedupe items when the site regenerates weekly. ``notice``
    events are included: subscribers asked for everything the panel saw,
    and the kind category lets them filter.
    """
    ordered = sorted(events, key=lambda e: e.get("date", ""), reverse=True)
    items: list[str] = []
    for ev in ordered:
        kind = ev.get("kind", "event")
        date = ev.get("date", "")
        endpoint = ev.get("endpoint", "")
        suite = ev.get("suite", "") or "-"
        # Two events can share (date, endpoint, suite, kind) — e.g. two
        # probe notices in one pass — so a summary digest disambiguates.
        # Summaries persist verbatim in events.jsonl, so the guid is still
        # stable across re-renders.
        digest = hashlib.sha256(
            ev.get("summary", "").encode()).hexdigest()[:8]
        guid = f"{date}/{endpoint}/{suite}/{kind}/{digest}"
        title = f"{endpoint}: {kind}" + (f" ({suite})" if suite != "-" else "")
        # RFC 822 date at midnight UTC; the panel records dates, not times.
        try:
            pub = datetime.strptime(date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc).strftime("%a, %d %b %Y 00:00:00 +0000")
        except ValueError:
            pub = ""
        items.append(
            "<item>"
            f"<title>{html.escape(title)}</title>"
            f"<description>{html.escape(ev.get('summary', ''))}</description>"
            f"<guid isPermaLink=\"false\">{html.escape(guid)}</guid>"
            f"<category>{html.escape(kind)}</category>"
            + (f"<pubDate>{pub}</pubDate>" if pub else "")
            + "</item>"
        )
    link = html.escape(site_url) if site_url else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        "<title>Rift Observatory — drift feed</title>"
        f"<link>{link}</link>"
        "<description>Statistically-gated behavioral changes on monitored "
        "LLM endpoints: score drift, silent model swaps, mid-run rollouts, "
        "and probe notices.</description>"
        + "".join(items) +
        "</channel></rss>\n"
    )


def _index_page(index: list[dict], events: list[dict],
                selftests: dict[str, dict]) -> str:
    endpoints = sorted({e["endpoint"] for e in index})
    dates = sorted({e["date"] for e in index})
    n_obs = len(index)

    cards: list[str] = []
    for ep in endpoints:
        ep_dates = _endpoint_dates(index, ep)
        latest = ep_dates[-1]
        # Headline mean uses the same aborted-exclusion as the sparkline:
        # a majority-errored outage pass must not headline as a collapse.
        mean_vals = [
            v for e in index
            if e["endpoint"] == ep and e["date"] == latest
            and not e.get("aborted")
            if (v := _finite(e.get("mean_score"))) is not None
        ]
        latest_mean = sum(mean_vals) / len(mean_vals) if mean_vals else None
        # Sparkline: cross-suite mean accuracy per date.
        spark_vals: list[float | None] = []
        for d in ep_dates:
            day = [_finite(e.get("mean_score")) for e in index
                   if e["endpoint"] == ep and e["date"] == d
                   and not e.get("aborted")]
            day = [v for v in day if v is not None]
            spark_vals.append(sum(day) / len(day) if day else None)
        spark = _svg_line_chart(
            ep_dates, {"mean": spark_vals}, f"{ep} mean accuracy",
            markers=_fingerprint_change_marks(index, ep, ep_dates),
            width=240, height=80,
        )
        fps = _fingerprint_union(index, ep, latest)
        cards.append(
            f'<div class="card"><a href="model/{html.escape(endpoint_slug(ep))}.html">'
            f'{html.escape(ep)}</a>'
            f'<div class="meta">latest {html.escape(latest)} · mean '
            f'{f"{latest_mean:.3f}" if latest_mean is not None else "n/a"}'
            + (f' · <span class="mono">{html.escape(fps[-1])}</span>' if fps else "")
            + f"</div>{spark}</div>"
        )

    selftest_note = ""
    rates = [
        f"{html.escape(ep)}: {st['false_regression_rate']:.1%}"
        for ep, st in sorted(selftests.items())
        if st.get("false_regression_rate") is not None
    ]
    if rates:
        selftest_note = (
            "<section><h2>Gate calibration</h2><p>Under the null "
            "(a model compared to itself), the drift gate's empirical "
            "false-regression rate per endpoint: "
            + " · ".join(rates)
            + ". A feed entry means little if the gate fires on unchanged "
            "models — this is how often it does (<code>rift selftest</code>)."
            "</p></section>"
        )

    body = f"""
<header>
  <div class="badges">
    <span class="badge">Rift Observatory</span>
    <span class="badge">{len(endpoints)} endpoints</span>
    <span class="badge">{n_obs} observations</span>
    <span class="badge">{html.escape(dates[0]) if dates else ""} → {html.escape(dates[-1]) if dates else ""}</span>
  </div>
  <h1>The public record of model behavior</h1>
  <p class="subtitle">Scheduled behavioral panel against live LLM endpoints:
  paired statistics on every change, fingerprint tracking for silent swaps,
  append-only history. "Did the model behind this endpoint change this week?"
  — answered with data.</p>
</header>

<section>
  <h2>Endpoints</h2>
  <div class="cards">{"".join(cards) if cards else '<p class="empty">No observations yet.</p>'}</div>
</section>

<section>
  <h2>Drift feed <small><a href="feed.xml">RSS</a></small></h2>
  {_feed_html(events)}
</section>
{selftest_note}
"""
    return _page("Rift Observatory", body)


def _model_page(index: list[dict], events: list[dict], endpoint: str,
                selftest: dict | None) -> str:
    dates = _endpoint_dates(index, endpoint)
    suites = sorted({e["suite"] for e in index if e["endpoint"] == endpoint})
    marks = _fingerprint_change_marks(index, endpoint, dates)

    def chart(metric: str, title: str, y_fmt) -> str:
        series = {
            s: _metric_by_date(index, endpoint, s, dates, metric)
            for s in suites
        }
        if all(v is None for vals in series.values() for v in vals):
            return ""
        return f"<h2>{html.escape(title)}</h2>" + _svg_line_chart(
            dates, series, title, y_fmt=y_fmt, markers=marks,
        )

    probe_series: dict[str, list[float | None]] = {}
    for metric, label in (("flip_rate", "sycophancy flip"),
                          ("refusal_rate", "refusal"),
                          ("ece", "ECE")):
        merged: list[float | None] = []
        for d in dates:
            vals = [
                _finite(e.get(metric)) for e in index
                if e["endpoint"] == endpoint and e["date"] == d
                and not e.get("aborted")
            ]
            vals = [v for v in vals if v is not None]
            merged.append(sum(vals) / len(vals) if vals else None)
        if any(v is not None for v in merged):
            probe_series[label] = merged
    probes_chart = ""
    if probe_series:
        probes_chart = "<h2>Behavioral probes</h2>" + _svg_line_chart(
            dates, probe_series, "Behavioral probes",
            y_fmt=lambda v: f"{v:.2f}", markers=marks,
        )

    # Fingerprint history table.
    fp_rows: list[str] = []
    prev_fps: list[str] | None = None
    for d in dates:
        fps = _fingerprint_union(index, endpoint, d)
        changed = prev_fps is not None and fps and prev_fps and fps != prev_fps
        mark = ' style="color:var(--warn);font-weight:600"' if changed else ""
        fp_rows.append(
            f"<tr><td>{html.escape(d)}</td>"
            f"<td{mark}><span class='mono'>"
            f"{html.escape(', '.join(fps) if fps else '—')}</span>"
            + ("&nbsp;(changed)" if changed else "")
            + "</td></tr>"
        )
        if fps:
            prev_fps = fps

    ep_events = [e for e in events if e.get("endpoint") == endpoint]
    selftest_note = ""
    if selftest and selftest.get("false_regression_rate") is not None:
        selftest_note = (
            f"<p class='meta' style='color:var(--mute);font-size:13px'>"
            f"Gate null calibration for this endpoint: "
            f"{selftest['false_regression_rate']:.1%} false-regression rate "
            f"over {selftest.get('reps', '?')} self-vs-self splits "
            f"(<code>rift selftest</code>).</p>"
        )

    body = f"""
<header>
  <div class="badges">
    <span class="badge">Rift Observatory</span>
    <span class="badge">{len(dates)} observations</span>
  </div>
  <h1>{html.escape(endpoint)}</h1>
  <p class="subtitle"><a href="../index.html">← all endpoints</a> ·
  dashed <span style="color:var(--warn)">fp</span> lines mark server
  fingerprint changes</p>
</header>

<section>
{chart("mean_score", "Accuracy by suite", lambda v: f"{v:.2f}")}
{probes_chart}
{chart("cost_usd", "Cost per run (USD)", lambda v: f"${v:.2f}")}
{chart("output_tokens", "Output tokens", lambda v: f"{v:,.0f}")}
</section>

<section>
  <h2>Fingerprint history</h2>
  <table><thead><tr><th>Date</th><th>Server fingerprints</th></tr></thead>
  <tbody>{"".join(fp_rows)}</tbody></table>
  {selftest_note}
</section>

<section>
  <h2>Events</h2>
  {_feed_html(ep_events)}
</section>
"""
    return _page(f"{endpoint} — Rift Observatory", body, root_prefix="../")


def render_site(data_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Render the full static site. Returns the written paths."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(data_dir)
    events = load_events(data_dir)
    endpoints = sorted({e["endpoint"] for e in index})
    selftests = {
        ep: st for ep in endpoints
        if (st := load_selftest(data_dir, ep)) is not None
    }

    written: list[Path] = []
    front = out_dir / "index.html"
    front.write_text(_index_page(index, events, selftests), encoding="utf-8")
    written.append(front)

    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    for ep in endpoints:
        page = model_dir / f"{endpoint_slug(ep)}.html"
        page.write_text(
            _model_page(index, events, ep, selftests.get(ep)),
            encoding="utf-8",
        )
        written.append(page)

    # RSS: the drift feed as a subscription, not just a page.
    feed = out_dir / "feed.xml"
    feed.write_text(_feed_xml(events), encoding="utf-8")
    written.append(feed)

    # Machine-readable passthrough — the record IS the product.
    data_out = out_dir / "data"
    data_out.mkdir(exist_ok=True)
    for name in ("index.jsonl", "events.jsonl"):
        src = data_dir / name
        if src.exists():
            shutil.copyfile(src, data_out / name)
            written.append(data_out / name)
    return written
