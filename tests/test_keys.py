"""Tests for API-key onboarding (rift.keys)."""

from __future__ import annotations

import os
import stat

import click
import pytest
from rich.console import Console

from rift import keys
from rift.providers import MissingAPIKeyError

CONSOLE = Console()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point keys.ENV_DIR / ENV_FILE at a throwaway dir and clear key vars."""
    env_dir = tmp_path / ".rift"
    monkeypatch.setattr(keys, "ENV_DIR", env_dir)
    monkeypatch.setattr(keys, "ENV_FILE", env_dir / ".env")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Run from a dir with no ./.env so load_env only sees our fake home file.
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestEnvFile:
    def test_save_then_load_roundtrip(self, fake_home, monkeypatch):
        keys.save_key("ANTHROPIC_API_KEY", "sk-test-123")
        # Not in the environment yet — only on disk.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        keys.load_env()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"

    def test_saved_file_is_user_only(self, fake_home):
        path = keys.save_key("OPENAI_API_KEY", "sk-openai")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_real_env_var_wins_over_file(self, fake_home, monkeypatch):
        keys.save_key("GEMINI_API_KEY", "from-file")
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        keys.load_env()  # setdefault must not clobber the real env var
        assert os.environ["GEMINI_API_KEY"] == "from-env"

    def test_save_key_upserts_without_dropping_others(self, fake_home):
        keys.save_key("ANTHROPIC_API_KEY", "a1")
        keys.save_key("OPENAI_API_KEY", "o1")
        keys.save_key("ANTHROPIC_API_KEY", "a2")  # update existing
        text = keys.ENV_FILE.read_text()
        assert "ANTHROPIC_API_KEY=a2" in text
        assert "OPENAI_API_KEY=o1" in text
        assert "ANTHROPIC_API_KEY=a1" not in text

    def test_parse_env_ignores_comments_and_blanks(self):
        parsed = keys._parse_env("# comment\n\nFOO=bar\nBAZ = \"q\" \n")
        assert parsed == {"FOO": "bar", "BAZ": "q"}

    def test_parse_env_keeps_unbalanced_quote(self):
        # A key legitimately ending in a quote must not lose it.
        assert keys._parse_env('K=ab"') == {"K": 'ab"'}
        assert keys._parse_env("K=sk-'live") == {"K": "sk-'live"}

    def test_save_key_updates_spaced_entry_in_place(self, fake_home):
        keys.ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        keys.ENV_FILE.write_text("ANTHROPIC_API_KEY = sk-old\n")
        keys.save_key("ANTHROPIC_API_KEY", "sk-new")
        text = keys.ENV_FILE.read_text()
        assert text.count("ANTHROPIC_API_KEY") == 1  # updated, not duplicated
        assert "sk-new" in text and "sk-old" not in text

    def test_dir_is_user_only(self, fake_home):
        keys.save_key("ANTHROPIC_API_KEY", "sk-x")
        assert stat.S_IMODE(keys.ENV_DIR.stat().st_mode) == 0o700


    def test_quoted_value_roundtrips(self, fake_home, monkeypatch):
        # A value that itself starts and ends with a quote must survive a
        # save -> load round-trip unchanged (write-side quoting is the inverse
        # of the parser's quote-stripping).
        for raw in ('"already_quoted"', "'single'", '"'):
            keys.save_key("ANTHROPIC_API_KEY", raw)
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            keys.load_env()
            assert os.environ["ANTHROPIC_API_KEY"] == raw, raw
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_plain_key_written_unquoted(self, fake_home):
        keys.save_key("ANTHROPIC_API_KEY", "sk-ant-123")
        assert "ANTHROPIC_API_KEY=sk-ant-123" in keys.ENV_FILE.read_text()

    def test_save_leaves_no_temp_file(self, fake_home):
        keys.save_key("ANTHROPIC_API_KEY", "sk-x")
        leftovers = [p.name for p in keys.ENV_DIR.iterdir() if p.name != ".env"]
        assert leftovers == []


class TestEnsureProviderKeys:
    def test_present_key_does_not_raise(self, fake_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
        keys.ensure_provider_keys(["anthropic"], CONSOLE)  # no raise

    def test_local_provider_skipped(self, fake_home):
        keys.ensure_provider_keys(["local"], CONSOLE)  # no key needed, no raise

    def test_non_interactive_missing_raises(self, fake_home, monkeypatch):
        monkeypatch.setattr(keys, "_interactive", lambda: False)
        with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
            keys.ensure_provider_keys(["anthropic"], CONSOLE)

    def test_interactive_prompt_sets_key(self, fake_home, monkeypatch):
        monkeypatch.setattr(keys, "_interactive", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *a, **k: "sk-typed")
        monkeypatch.setattr(click, "confirm", lambda *a, **k: False)  # don't save
        keys.ensure_provider_keys(["openai"], CONSOLE)
        assert os.environ["OPENAI_API_KEY"] == "sk-typed"
        assert not keys.ENV_FILE.exists()  # confirm declined -> not persisted

    def test_interactive_blank_skip_raises(self, fake_home, monkeypatch):
        monkeypatch.setattr(keys, "_interactive", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *a, **k: "   ")  # blank
        with pytest.raises(MissingAPIKeyError):
            keys.ensure_provider_keys(["openai"], CONSOLE)

    def test_interactive_prompt_saves_when_confirmed(self, fake_home, monkeypatch):
        monkeypatch.setattr(keys, "_interactive", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *a, **k: "sk-saved")
        monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
        keys.ensure_provider_keys(["anthropic"], CONSOLE)
        assert "ANTHROPIC_API_KEY=sk-saved" in keys.ENV_FILE.read_text()
