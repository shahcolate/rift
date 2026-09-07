#!/usr/bin/env bash
# Ensure the orphan `observatory-data` branch exists on origin.
#
# Why this exists: the Observatory workflow failed on EVERY scheduled run
# from July to September 2026 — not on an API call, not on a budget cap,
# but on `actions/checkout ref: observatory-data`, because the branch was a
# "one-time maintainer setup" step that never happened. Two months of
# observations were never taken. A scheduled job must not depend on a
# manual step it can perform itself; this script performs it.
#
# Runs from the main checkout (which has the workflow's push credential).
# Idempotent: if the branch exists, exits 0 without touching anything. If
# not, publishes a single root commit (no parents — a true orphan, so the
# data history never mixes with the code history) containing only a README,
# then the ordinary `actions/checkout` of the branch succeeds.
#
# Never force-pushes: if two runs race here, the loser's push is rejected,
# and the subsequent checkout simply picks up the winner's branch.
set -euo pipefail

branch="${OBSERVATORY_DATA_BRANCH:-observatory-data}"

if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
  echo "observatory: branch '$branch' exists on origin — nothing to bootstrap."
  exit 0
fi

echo "observatory: branch '$branch' is missing on origin — bootstrapping it."

readme=$(cat <<'MD'
# Rift Observatory — data branch

Append-only record of the scheduled Observatory panel: one JSON record per
(date, endpoint, suite) under `records/`, a compact `index.jsonl`, the drift
feed `events.jsonl`, and per-endpoint `selftest/` null-calibration results.

This branch is written only by `.github/workflows/observatory.yml` (via
`rift observe`) and rendered to GitHub Pages by `rift observatory-site`.
Do not edit by hand — the pairing logic assumes records are never rewritten.

Bootstrapped automatically by `.github/scripts/observatory_bootstrap.sh`.
MD
)

blob=$(printf '%s\n' "$readme" | git hash-object -w --stdin)
tree=$(printf '100644 blob %s\tREADME.md\n' "$blob" | git mktree)
commit=$(
  GIT_AUTHOR_NAME="rift-observatory[bot]" \
  GIT_AUTHOR_EMAIL="observatory@users.noreply.github.com" \
  GIT_COMMITTER_NAME="rift-observatory[bot]" \
  GIT_COMMITTER_EMAIL="observatory@users.noreply.github.com" \
  git commit-tree "$tree" -m "observatory: init data branch (bootstrapped by workflow)"
)

if git push origin "$commit:refs/heads/$branch"; then
  echo "observatory: created '$branch' at $commit."
elif git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
  echo "observatory: '$branch' appeared concurrently — using it."
else
  echo "::error::could not create branch '$branch' (push rejected and branch still absent)"
  exit 1
fi
