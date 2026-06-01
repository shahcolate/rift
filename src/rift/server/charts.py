"""Dependency-free inline-SVG charts for the dashboard.

The demo already proves Rift can render publishable charts as self-contained
SVG with no JS and no CDN (see ``rift.demo._svg_bar_chart``). The dashboard
follows the same principle so it renders identically offline, in print, and in
an email. These helpers return SVG strings embedded directly in the Jinja
templates and registered as template globals in :mod:`rift.server.app`.
"""

from __future__ import annotations

import html
import math


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def sparkline(
    values: list[float],
    *,
    width: int = 320,
    height: int = 60,
    stroke: str = "#6c5ce7",
    fill: str = "#efedfb",
    zero_line: bool = True,
) -> str:
    """A filled line chart over ``values`` (oldest→newest), auto-scaled.

    Designed for a drift time-series: when ``zero_line`` is set, y=0 is drawn
    as a reference so regressions (negative drift) sit visibly below it. Renders
    a friendly placeholder when there are <2 points.
    """
    pts = _finite(values)
    pad = 6
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad
    if len(pts) < 2:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="no data">'
            f'<text x="{width/2}" y="{height/2+4}" text-anchor="middle" '
            f'font-size="12" fill="#9b9b9b">not enough runs yet</text></svg>'
        )

    lo, hi = min(pts), max(pts)
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    n = len(pts)

    def x(i: int) -> float:
        return pad + (i / (n - 1)) * inner_w

    def y(v: float) -> float:
        return pad + (1 - (v - lo) / span) * inner_h

    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(pts))
    area = (
        f"{x(0):.1f},{y(lo):.1f} " + line + f" {x(n-1):.1f},{y(lo):.1f}"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="time series">',
        f'<polygon points="{area}" fill="{fill}"/>',
    ]
    if zero_line and lo <= 0.0 <= hi:
        zy = y(0.0)
        parts.append(
            f'<line x1="{pad}" y1="{zy:.1f}" x2="{width-pad}" y2="{zy:.1f}" '
            f'stroke="#d0d0d0" stroke-width="1" stroke-dasharray="3 3"/>'
        )
    parts.append(
        f'<polyline points="{line}" fill="none" stroke="{stroke}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    # Mark the most recent point.
    parts.append(
        f'<circle cx="{x(n-1):.1f}" cy="{y(pts[-1]):.1f}" r="3" fill="{stroke}"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def status_badge(status: str | None) -> str:
    """An inline HTML badge for a run status."""
    status = status or "—"
    cls = {
        "ok": "good",
        "regression": "warn",
        "error": "err",
        "running": "mute",
    }.get(status, "mute")
    label = {"ok": "no drift", "regression": "regression",
             "error": "error", "running": "running"}.get(status, status)
    return f'<span class="badge {cls}">{html.escape(label)}</span>'
