#!/usr/bin/env bash
# Commit and push the observatory-data checkout. Shared by the weekly
# observe job and the monthly selftest job so the retry semantics can't
# drift between them.
#
# Usage: observatory_commit.sh "<commit message>"
#
# The push retries with pull-rebase (another writer may have appended), and
# — critically — FAILS the job when every attempt fails: a green job whose
# observations never reached the branch would silently lose the week's data
# and the API spend that produced it.
set -euo pipefail

msg="${1:?usage: observatory_commit.sh <commit message>}"

cd observatory-data

if ! git status --porcelain | grep -q .; then
  echo "No changes to commit."
  exit 0
fi

git config user.name "rift-observatory[bot]"
git config user.email "observatory@users.noreply.github.com"
git add -A
git commit -m "$msg"

pushed=0
for i in 1 2 3 4; do
  if git push origin HEAD:observatory-data; then
    pushed=1
    break
  fi
  git pull --rebase origin observatory-data
  sleep $((i * 2))
done

if [ "$pushed" -ne 1 ]; then
  echo "::error::failed to push observatory-data after 4 attempts"
  exit 1
fi
