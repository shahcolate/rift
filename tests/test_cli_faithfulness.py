"""CLI-level smoke tests for the faithfulness command."""

from __future__ import annotations

from click.testing import CliRunner

from rift.cli import main


class TestFaithfulnessCLI:
    def test_listed_in_help(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_faithfulness_help_options(self):
        result = CliRunner().invoke(main, ["faithfulness", "--help"])
        assert result.exit_code == 0
        for opt in ["--baseline", "--challenger", "--suite", "--judge-model",
                    "--proposer-model", "--cues"]:
            assert opt in result.output

    def test_rejects_unsupported_scoring(self, tmp_path):
        """A non-sync-scorable suite (e.g. llm_judge) fails fast with a clean
        message, before any API key is needed."""
        suite = tmp_path / "s.yaml"
        suite.write_text(
            "name: s\nscoring: llm_judge\ncases:\n  - input: q\n    expected: a\n"
        )
        result = CliRunner().invoke(main, [
            "faithfulness", "--baseline", "opus-4-7", "--challenger", "opus-4-8",
            "--suite", str(suite),
        ])
        assert result.exit_code != 0
        assert "exact_match or fuzzy_match" in result.output
