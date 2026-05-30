"""API-key onboarding, kept stupidly simple for non-engineers.

Keys live in ``~/.rift/.env`` (auto-loaded on every run), are entered via
``rift setup`` or an on-demand prompt the first time a command needs one,
and never surface as a Python traceback.

Real environment variables always win over the saved file, so an explicit
``export`` or a CI secret overrides what's on disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from .providers import PROVIDER_KEYS, MissingAPIKeyError

ENV_DIR = Path.home() / ".rift"
ENV_FILE = ENV_DIR / ".env"


def _parse_env(text: str) -> dict[str, str]:
    """Parse simple ``KEY=VALUE`` lines. Ignores blanks and ``#`` comments."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def load_env() -> None:
    """Load keys from ``~/.rift/.env`` then ``./.env`` into ``os.environ``.

    Uses ``setdefault`` so a real environment variable (export / CI secret)
    is never clobbered by the file.
    """
    for path in (ENV_FILE, Path(".env")):
        try:
            if path.is_file():
                for key, value in _parse_env(path.read_text()).items():
                    os.environ.setdefault(key, value)
        except OSError:
            # An unreadable .env should never crash the CLI.
            pass


def save_key(env_var: str, value: str) -> Path:
    """Upsert ``env_var=value`` into ``~/.rift/.env`` (mode 0600).

    Preserves any other lines already in the file and writes atomically so
    a crash mid-write never leaves a half-written file.
    """
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    found = False
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith(f"{env_var}="):
                lines.append(f"{env_var}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{env_var}={value}")

    tmp = ENV_FILE.with_name(ENV_FILE.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(ENV_FILE)
    os.chmod(ENV_FILE, 0o600)
    return ENV_FILE


def _interactive() -> bool:
    """True when we can prompt the user (both stdin and stdout are a TTY)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_for_key(provider: str, console: Console) -> str:
    """Prompt for one provider's key (hidden input). Returns "" if skipped."""
    env_var, signup_url = PROVIDER_KEYS[provider]
    console.print(f"\n[bold]{provider}[/bold] needs an API key ([cyan]{env_var}[/cyan]).")
    if signup_url:
        console.print(f"  Get one at [link={signup_url}]{signup_url}[/link]")
    return click.prompt(
        f"  Paste your {env_var}",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()


def ensure_provider_keys(providers: list[str], console: Console) -> None:
    """Make sure each provider has a key before a live run.

    - Interactive TTY: prompt for any missing key and offer to save it.
    - Non-interactive (pipe / CI): raise :class:`MissingAPIKeyError`, which
      Click renders as a clean one-line message and exits 1.
    - ``local`` and unknown providers need no key and are skipped.

    This is intentionally only called by the commands that make live API
    calls; the demo and cached/replay paths stay keyless.
    """
    seen: set[str] = set()
    for provider in providers:
        if provider not in PROVIDER_KEYS or provider in seen:
            continue
        seen.add(provider)
        env_var = PROVIDER_KEYS[provider][0]
        if os.environ.get(env_var):
            continue
        if not _interactive():
            raise MissingAPIKeyError(provider)
        value = _prompt_for_key(provider, console)
        if not value:
            raise MissingAPIKeyError(provider)
        os.environ[env_var] = value
        if click.confirm("  Save it to ~/.rift/.env for next time?", default=True):
            path = save_key(env_var, value)
            console.print(f"  [green]Saved[/green] to {path}")


def run_setup(console: Console) -> None:
    """Interactive ``rift setup``: configure keys for every provider."""
    console.print(
        Panel(
            "Paste an API key for each provider you use. Leave a field blank "
            "to skip it.\nKeys are saved to [cyan]~/.rift/.env[/cyan] "
            "(readable only by you) and loaded automatically on every run.",
            title="rift setup",
            border_style="cyan",
        )
    )

    if not _interactive():
        console.print(
            "[yellow]rift setup needs an interactive terminal.[/yellow] "
            "Set the keys directly instead, e.g. "
            "[cyan]export ANTHROPIC_API_KEY=...[/cyan]"
        )
        return

    saved_any = False
    for provider, (env_var, signup_url) in PROVIDER_KEYS.items():
        status = "[green]already set[/green]" if os.environ.get(env_var) else "[dim]not set[/dim]"
        console.print(f"\n[bold]{provider}[/bold]  [{env_var}: {status}]")
        if signup_url:
            console.print(f"  Get a key: [link={signup_url}]{signup_url}[/link]")
        value = click.prompt(
            f"  Paste {env_var} (blank to keep/skip)",
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if value:
            os.environ[env_var] = value
            save_key(env_var, value)
            saved_any = True
            console.print("  [green]Saved.[/green]")

    if saved_any:
        console.print(
            f"\n[green]Done.[/green] Keys saved to [cyan]{ENV_FILE}[/cyan]. "
            "You're ready to run [bold]rift compare[/bold]."
        )
    else:
        configured = [v[0] for p, v in PROVIDER_KEYS.items() if os.environ.get(v[0])]
        if configured:
            console.print(
                f"\nNothing changed. Already configured: "
                f"[green]{', '.join(configured)}[/green]."
            )
        else:
            console.print("\nNothing changed — no keys configured yet.")
