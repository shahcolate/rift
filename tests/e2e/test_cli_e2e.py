"""End-to-end CLI workflow tests.

Each test drives the real ``rift`` subprocess against a seeded cache (no
network) and asserts on exit codes, stdout, and written artifacts — the
contract a user and a CI pipeline actually depend on.
"""

from __future__ import annotations

import json

EXACT_SUITE = """\
name: e2e_math
description: tiny arithmetic suite
scoring: exact_match
cases:
  - {input: "2+2?", expected: "4"}
  - {input: "3+5?", expected: "8"}
  - {input: "10-3?", expected: "7"}
"""


def _seed_pair(seeder, model, answers):
    """Seed a model's answers for the EXACT_SUITE prompts."""
    for prompt, out in answers.items():
        seeder.seed(model, prompt, out)


PROMPTS = ["2+2?", "3+5?", "10-3?"]

# A 6-case suite for the regression test. McNemar's exact test needs enough
# discordant pairs to reach significance: with N all-one-direction discordant
# pairs the two-sided p is 2*0.5^N, so 3 cases give p=0.25 (NOT significant)
# but 6 give p=0.03125 < 0.05. The regression gate (exit 1) only fires on a
# *significant* drop, so the test must use >= 6 fully-regressing cases.
REGRESSION_SUITE = """\
name: e2e_regress
scoring: exact_match
cases:
  - {input: "q1", expected: "a1"}
  - {input: "q2", expected: "a2"}
  - {input: "q3", expected: "a3"}
  - {input: "q4", expected: "a4"}
  - {input: "q5", expected: "a5"}
  - {input: "q6", expected: "a6"}
"""
REGRESSION_RIGHT = {f"q{i}": f"a{i}" for i in range(1, 7)}
REGRESSION_WRONG = {f"q{i}": f"WRONG{i}" for i in range(1, 7)}


class TestRunCommand:
    def test_run_all_correct_exit_0(self, run_rift, seed_cache, write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        s = seed_cache()
        _seed_pair(s, "opus-4-7", {"2+2?": "4", "3+5?": "8", "10-3?": "7"})
        out = workdir / "run.json"
        proc = run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                        "--output", str(out), "--cache-dir", str(s.cache_dir),
                        expect_exit=0)
        data = json.loads(out.read_text())
        assert [c["score"] for c in data["cases"]] == [1.0, 1.0, 1.0]
        assert data["model"] == "claude-opus-4-7"
        assert "Results saved" in proc.stdout

    def test_run_partial_scores(self, run_rift, seed_cache, write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        s = seed_cache()
        _seed_pair(s, "opus-4-7", {"2+2?": "4", "3+5?": "WRONG", "10-3?": "7"})
        out = workdir / "run.json"
        run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                 "--output", str(out), "--cache-dir", str(s.cache_dir),
                 expect_exit=0)
        data = json.loads(out.read_text())
        assert [c["score"] for c in data["cases"]] == [1.0, 0.0, 1.0]

    def test_run_missing_key_is_clean_error(self, run_rift, write_suite, workdir):
        # No cache + a real key preflight failure must be a clean message, not a
        # traceback. We clear the dummy key via env_extra.
        suite = write_suite("s.yaml", EXACT_SUITE)
        proc = run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                        "--output", str(workdir / "o.json"),
                        env_extra={"ANTHROPIC_API_KEY": ""})
        assert proc.returncode == 1
        assert "ANTHROPIC_API_KEY" in proc.stderr
        assert "Traceback" not in proc.stderr

    def test_strip_io_empties_text(self, run_rift, seed_cache, write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        s = seed_cache()
        _seed_pair(s, "opus-4-7", {"2+2?": "4", "3+5?": "8", "10-3?": "7"})
        out = workdir / "run.json"
        run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                 "--output", str(out), "--cache-dir", str(s.cache_dir),
                 "--strip-io", expect_exit=0)
        data = json.loads(out.read_text())
        for c in data["cases"]:
            assert c["input_text"] == ""
            assert c["output"] == ""
            # but scores/costs preserved
            assert c["score"] in (0.0, 1.0)


class TestCompareCommand:
    def _seed_both(self, seed_cache, base_ans, chal_ans):
        s = seed_cache()
        _seed_pair(s, "opus-4-7", base_ans)
        _seed_pair(s, "opus-4-8", chal_ans)
        return s

    def test_compare_no_drift_exit_0(self, run_rift, seed_cache, write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        allright = {"2+2?": "4", "3+5?": "8", "10-3?": "7"}
        s = self._seed_both(seed_cache, allright, dict(allright))
        proc = run_rift("compare", "--baseline", "opus-4-7",
                        "--challenger", "opus-4-8", "--suite", str(suite),
                        "--cache-dir", str(s.cache_dir), expect_exit=0)
        assert "Rift" in proc.stdout or "drift" in proc.stdout.lower()

    def test_compare_regression_exit_1(self, run_rift, seed_cache, write_suite, workdir):
        # Challenger gets all 6 wrong -> McNemar p=0.03125 < 0.05 -> significant
        # regression -> exit 1 (the CI gate). 6 cases is the minimum that
        # reaches significance; see REGRESSION_SUITE.
        suite = write_suite("s.yaml", REGRESSION_SUITE)
        s = self._seed_both(seed_cache, REGRESSION_RIGHT, REGRESSION_WRONG)
        proc = run_rift("compare", "--baseline", "opus-4-7",
                        "--challenger", "opus-4-8", "--suite", str(suite),
                        "--cache-dir", str(s.cache_dir))
        assert proc.returncode == 1  # regression gates CI
        # And assert it's the significance gate, not an incidental crash.
        assert "Traceback" not in proc.stderr

    def test_compare_writes_output_and_report(self, run_rift, seed_cache,
                                              write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        allright = {"2+2?": "4", "3+5?": "8", "10-3?": "7"}
        s = self._seed_both(seed_cache, allright, dict(allright))
        out = workdir / "cmp.json"
        report = workdir / "report.md"
        run_rift("compare", "--baseline", "opus-4-7", "--challenger", "opus-4-8",
                 "--suite", str(suite), "--cache-dir", str(s.cache_dir),
                 "--output", str(out), "--report", str(report), expect_exit=0)
        data = json.loads(out.read_text())
        assert "drift" in data
        # The drift records the model identifiers as passed on the CLI.
        assert data["drift"]["baseline_model"] == "opus-4-7"
        assert data["drift"]["challenger_model"] == "opus-4-8"
        assert report.read_text().strip()  # non-empty markdown

class TestDiffCommand:
    def test_diff_two_saved_runs(self, run_rift, seed_cache, write_suite, workdir):
        suite = write_suite("s.yaml", EXACT_SUITE)
        s = seed_cache()
        _seed_pair(s, "opus-4-7", {"2+2?": "4", "3+5?": "8", "10-3?": "7"})
        _seed_pair(s, "opus-4-8", {"2+2?": "4", "3+5?": "8", "10-3?": "WRONG"})
        run_a = workdir / "a.json"
        run_b = workdir / "b.json"
        run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                 "--output", str(run_a), "--cache-dir", str(s.cache_dir),
                 expect_exit=0)
        run_rift("run", "--model", "opus-4-8", "--suite", str(suite),
                 "--output", str(run_b), "--cache-dir", str(s.cache_dir),
                 expect_exit=0)
        proc = run_rift("diff", str(run_a), str(run_b))
        # diff exits 0 (no significant drift on 1/3 change) or 1; either way no crash
        assert proc.returncode in (0, 1)
        assert "Traceback" not in proc.stderr


class TestCustomScorerE2E:
    def test_custom_scorer_run(self, run_rift, seed_cache, write_suite, workdir):
        # A full subprocess run using scoring: custom with a sibling .py file.
        (workdir / "scorer.py").write_text(
            "def contains(output, expected):\n"
            "    return 1.0 if str(expected).lower() in output.lower() else 0.0\n"
        )
        suite = write_suite("cs.yaml",
            "name: cs\n"
            "scoring: custom\n"
            "custom_scorer: ./scorer.py:contains\n"
            "cases:\n"
            "  - {input: 'name a color', expected: 'blue'}\n"
            "  - {input: 'name a fruit', expected: 'apple'}\n")
        s = seed_cache()
        s.seed("opus-4-7", "name a color", "I like BLUE skies")
        s.seed("opus-4-7", "name a fruit", "a banana")
        out = workdir / "run.json"
        run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                 "--output", str(out), "--cache-dir", str(s.cache_dir),
                 expect_exit=0)
        data = json.loads(out.read_text())
        assert [c["score"] for c in data["cases"]] == [1.0, 0.0]
        assert data["metadata"]["custom_scorer"] == "./scorer.py:contains"

    def test_custom_scorer_bad_spec_fails_clean(self, run_rift, write_suite, workdir):
        # custom scoring with no custom_scorer must fail validation cleanly —
        # a readable message and exit 1, never a raw pydantic traceback.
        suite = write_suite("bad.yaml",
            "name: bad\nscoring: custom\ncases:\n  - {input: q, expected: a}\n")
        proc = run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                        "--output", str(workdir / "o.json"))
        assert proc.returncode == 1
        assert "Traceback" not in proc.stderr
        assert "custom_scorer" in proc.stderr
        assert "Invalid suite" in proc.stderr


class TestSuiteValidationE2E:
    def test_malformed_yaml_clean_error(self, run_rift, write_suite, workdir):
        # An unknown scoring method is a clean message, not a traceback.
        suite = write_suite("bad.yaml",
            "name: bad\nscoring: not_a_method\ncases:\n  - {input: q, expected: a}\n")
        proc = run_rift("run", "--model", "opus-4-7", "--suite", str(suite),
                        "--output", str(workdir / "o.json"))
        assert proc.returncode == 1
        assert "Traceback" not in proc.stderr
        assert "Invalid suite" in proc.stderr


class TestTopLevel:
    def test_version(self, run_rift):
        proc = run_rift("--version", expect_exit=0)
        assert "rift" in proc.stdout.lower()

    def test_help_lists_commands(self, run_rift):
        proc = run_rift("--help", expect_exit=0)
        for cmd in ("compare", "run", "diff"):
            assert cmd in proc.stdout

    def test_unknown_suite_clean_error(self, run_rift, workdir):
        proc = run_rift("run", "--model", "opus-4-7",
                        "--suite", "no_such_suite_xyz",
                        "--output", str(workdir / "o.json"))
        assert proc.returncode == 1
        assert "Traceback" not in proc.stderr
        assert "not found" in proc.stderr

    def test_missing_suite_path_clean_error(self, run_rift, workdir):
        # An explicit path that doesn't exist must also be a clean message.
        proc = run_rift("run", "--model", "opus-4-7",
                        "--suite", str(workdir / "nope.yaml"),
                        "--output", str(workdir / "o.json"))
        assert proc.returncode == 1
        assert "Traceback" not in proc.stderr
        assert "not found" in proc.stderr
