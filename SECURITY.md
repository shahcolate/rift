# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/shahcolate/rift/security/advisories/new)
rather than opening a public issue. You should receive a response within a
week. Please include a minimal reproduction if you can.

## Scope and threat model

Rift is a local CLI that sends eval prompts to LLM provider APIs. Things
worth knowing when assessing or reporting issues:

- **API keys.** Keys are read from environment variables or `~/.rift/.env`
  (written by `rift setup` with mode 0600). Keys are never written to run
  artifacts, reports, caches, or Observatory records. A report that shows a
  key leaking into any saved artifact is a vulnerability; please report it
  privately.
- **Custom scorers execute code.** `scoring: custom` imports and runs the
  Python target named by the suite's `custom_scorer` field, and
  `scoring: exec_tests` executes model-generated code against test cases.
  This is by design and documented in the README: **only run suites you
  trust**, and treat third-party suite YAML like third-party code. Sandboxing
  bypasses of `exec_tests` are still in scope — the current implementation
  makes no security guarantee, but we want to know if its documented
  limitations are wrong.
- **`--strip-io` is a publishing convenience, not a privacy primitive.** It
  empties per-case `input_text`/`output` in saved JSON. Secrets embedded in
  `tags`, `expected`, or suite metadata still ship. Reports that assume
  otherwise are working as documented.
- **Completion cache.** Cached completions (`.rift/cache`) contain full
  prompts and outputs in plaintext on local disk. Treat the cache directory
  with the same sensitivity as the suites you run.

## Supported versions

Only the latest release receives security fixes.
