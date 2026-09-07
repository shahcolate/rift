#!/usr/bin/env bash
# Surface an Observatory failure as a GitHub issue instead of a red badge
# nobody looks at.
#
# The failure mode this guards against is not "a run failed" — it is
# "sixteen consecutive weekly runs failed and nobody noticed" (July–Sept
# 2026, see observatory_bootstrap.sh). One open issue, labeled
# `observatory-failure`, tracks the current outage: created on the first
# failure, appended to on each subsequent one, and closed by a human once
# a run is green again. Repeated failures therefore produce ONE
# notification thread, not one issue per week.
#
# Inputs via env (never interpolated into the script body):
#   GH_TOKEN  — the workflow token (issues: write)
#   RUN_URL   — link to the failed run
#   FAILED_JOB — which job failed (observe | selftest | pages)
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${RUN_URL:?RUN_URL is required}"
FAILED_JOB="${FAILED_JOB:-unknown}"
label="observatory-failure"

# `--force` makes label creation idempotent (updates instead of erroring
# when the label already exists).
gh label create "$label" --force \
  --description "Scheduled Observatory run failed — infrastructure, not drift" \
  --color "B60205" >/dev/null

existing=$(gh issue list --label "$label" --state open --json number --jq '.[0].number // empty')

body=$(cat <<EOF
The scheduled Observatory run failed in job **${FAILED_JOB}**.

Run: ${RUN_URL}

A failed run records nothing and spends nothing — but every missed week is
a hole in the longitudinal series the Observatory exists to build. Triage
checklist:

- [ ] Open the run log above; the first red step names the cause.
- [ ] Missing API key secret → add \`ANTHROPIC_API_KEY\` / \`OPENAI_API_KEY\` / \`GEMINI_API_KEY\` under Settings → Secrets.
- [ ] Pages deploy failed → Settings → Pages → Source must be "GitHub Actions".
- [ ] Provider outage / budget abort → these exit 0 with partial data, so a red run is never one of these.
- [ ] Re-run: Actions → Observatory → Run workflow.

Close this issue once a scheduled run is green again.
EOF
)

if [ -n "$existing" ]; then
  gh issue comment "$existing" --body "$body"
  echo "observatory: appended failure to open issue #$existing"
else
  gh issue create --title "Observatory: scheduled run failed" \
    --label "$label" --body "$body"
  echo "observatory: opened a new failure issue"
fi
