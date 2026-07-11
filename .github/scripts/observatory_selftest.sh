#!/usr/bin/env bash
# Refresh each observatory endpoint's null calibration.
#
# Runs `rift selftest` (the gate's empirical false-regression rate on an
# unchanged model) per panel endpoint and stores ONLY the SelfTestResult
# under observatory-data/selftest/<slug>.json — the dashboard cites it next
# to every drift verdict. A failing endpoint keeps its previous result; the
# script fails only when every endpoint fails (total outage).
set -euo pipefail

python - <<'PY'
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from rift.observatory import endpoint_slug

panel = yaml.safe_load(Path("observatory/panel.yaml").read_text())
suite = panel["suites"][0]
out_dir = Path("observatory-data/selftest")
out_dir.mkdir(parents=True, exist_ok=True)

failures = 0
for ep in panel["endpoints"]:
    model = ep.get("model", ep["id"])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    print(f"::group::selftest {ep['id']}")
    rc = subprocess.run([
        "rift", "selftest", "--model", model, "--suite", suite,
        "--trials", "3", "--output", str(tmp_path),
    ]).returncode
    print("::endgroup::")
    if rc != 0 or not tmp_path.exists():
        print(f"::warning::selftest failed for {ep['id']} (exit {rc}); "
              f"keeping the previous stored result")
        failures += 1
        continue
    data = json.loads(tmp_path.read_text())
    tmp_path.unlink(missing_ok=True)
    out = out_dir / f"{endpoint_slug(ep['id'])}.json"
    out.write_text(json.dumps(data["selftest"], indent=2) + "\n")
    print(f"wrote {out}")

sys.exit(1 if failures == len(panel["endpoints"]) else 0)
PY
