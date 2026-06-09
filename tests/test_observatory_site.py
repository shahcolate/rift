"""Observatory static site: SVG timelines, drift feed, fingerprint markers."""

from __future__ import annotations

from pathlib import Path

from rift.observatory import append_events, append_records, build_record, detect_drift
from rift.observatory_site import _svg_line_chart, render_site

from .test_observatory import make_run

DATES = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"]


def _seed(data_dir: Path) -> None:
    """3 endpoints × 4 weeks: one silent swap, one real drift, one stable."""
    plans = {
        # fp swap on week 3, scores hold → silent_swap
        "ep-swap": [("fp-a", [1.0, 0.0, 1.0, 1.0])] * 2
                   + [("fp-b", [1.0, 0.0, 1.0, 1.0])] * 2,
        # regression on the final week
        "ep-drift": [("fp-x", [1.0] * 12)] * 3
                    + [("fp-x", [0.0] * 8 + [1.0] * 4)],
        "ep-stable": [("fp-s", [1.0, 1.0, 0.0, 1.0])] * 4,
    }
    for ep, weekly in plans.items():
        for date, (fp, scores) in zip(DATES, weekly):
            run = make_run(scores, model=ep, fingerprint=fp)
            push = make_run(scores, model=ep, suite="panel_suite__pushback",
                            fingerprint=fp)
            rec = build_record(run, ep, date, pushback_run=push)
            append_records([rec], data_dir)
    for date in DATES:
        append_events(detect_drift(data_dir, date), data_dir)


class TestLineChart:
    def test_sanitizes_nan_inf_and_none(self):
        svg = _svg_line_chart(
            ["d1", "d2", "d3"],
            {"s": [1.0, float("nan"), float("inf")], "t": [None, 0.5, 0.7]},
            "title",
        )
        assert svg.startswith("<svg")
        assert "nan" not in svg.lower().replace("aria", "")
        assert "inf" not in svg

    def test_all_invalid_renders_no_data(self):
        svg = _svg_line_chart(["d1"], {"s": [float("nan")]}, "t")
        assert "no data" in svg

    def test_markers_drawn_dashed(self):
        svg = _svg_line_chart(["d1", "d2"], {"s": [0.1, 0.2]}, "t", markers=[1])
        assert "stroke-dasharray" in svg

    def test_escapes_labels(self):
        svg = _svg_line_chart(["<d>"], {"<s>": [1.0]}, "<t>")
        assert "<d>" not in svg and "&lt;d&gt;" in svg
        assert "&lt;s&gt;" in svg


class TestRenderSite:
    def test_full_site(self, tmp_path):
        data_dir = tmp_path / "data"
        out = tmp_path / "site"
        _seed(data_dir)
        written = render_site(data_dir, out)
        names = {p.relative_to(out).as_posix() for p in written}
        assert "index.html" in names
        assert "model/ep-swap.html" in names
        assert "data/index.jsonl" in names

        front = (out / "index.html").read_text()
        assert "Rift Observatory" in front
        assert "ep-swap" in front and "ep-drift" in front
        assert "<svg" in front
        assert "silent swap" in front       # feed entry from the swap
        assert "score drift" in front       # feed entry from the regression

        model = (out / "model" / "ep-swap.html").read_text()
        assert "fp-a" in model and "fp-b" in model
        assert "(changed)" in model         # fingerprint history highlight
        assert "stroke-dasharray" in model  # fp-change marker on the chart

    def test_feed_is_reverse_chronological(self, tmp_path):
        data_dir = tmp_path / "data"
        _seed(data_dir)
        out = tmp_path / "site"
        render_site(data_dir, out)
        front = (out / "index.html").read_text()
        feed = front[front.index("Drift feed"):]
        # The drift (01-26) must appear before the swap (01-19) in the feed.
        assert feed.index("2026-01-26") < feed.index("2026-01-19")

    def test_empty_data_dir(self, tmp_path):
        out = tmp_path / "site"
        written = render_site(tmp_path / "nothing", out)
        assert (out / "index.html").exists()
        front = (out / "index.html").read_text()
        assert "No observations yet" in front
        assert "No drift events" in front
        assert all(p.exists() for p in written)

    def test_selftest_cited_on_front_page(self, tmp_path):
        import json
        data_dir = tmp_path / "data"
        _seed(data_dir)
        st_dir = data_dir / "selftest"
        st_dir.mkdir()
        (st_dir / "ep-stable.json").write_text(
            json.dumps({"false_regression_rate": 0.02, "reps": 500})
        )
        out = tmp_path / "site"
        render_site(data_dir, out)
        front = (out / "index.html").read_text()
        assert "Gate calibration" in front
        assert "2.0%" in front
        model = (out / "model" / "ep-stable.html").read_text()
        assert "false-regression rate" in model
