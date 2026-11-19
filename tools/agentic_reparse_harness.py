#!/usr/bin/env python3
"""Re-derive episode.json from the recorded raw trace of alt-harness episodes (codex/dsh) with the current
harness_runner parser. Never touches the trace; rewrites episode.json + re-runs extraction. Records the
previous status in episode.json["reparsed_from"]. Usage: .venv/bin/python tools/agentic_reparse_harness.py <batch_dir> [--apply]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ribench_agentic import harness_runner as hr
from ribench_agentic.extract import extract_all
from ribench_agentic.casefs import write_json

batch = Path(sys.argv[1]); apply = "--apply" in sys.argv
changed = 0
for ep in sorted(batch.glob("episodes/*/episode.json")):
    d = ep.parent
    old = json.loads(ep.read_text())
    ctx = json.loads((d / "context.json").read_text())
    harness = ctx.get("harness", "pi")
    if harness == "pi":
        continue
    raw = (d / "trace.jsonl").read_bytes()
    if harness == "codex":
        new = hr.parse_codex_trace(raw, old.get("returncode", 0), workspace=d / "workspace")
    else:
        events = [json.loads(l) for l in (d / "dsh_session.jsonl").read_text().splitlines() if l.strip()]
        new = hr.parse_dsh_trace(raw, events, old.get("returncode", 0), workspace=d / "workspace")
    for k in ("wall_seconds", "requested_model", "returncode", "timed_out", "attempt", "integrity_error", "harness"):
        new[k] = old.get(k)
    if old.get("integrity_error"):
        new["errors"].append(old["integrity_error"]); new["status"] = "error"
    if new["status"] != old["status"] or new["errors"] != old["errors"]:
        changed += 1
        print(f"{d.name[:12]} {old['status']} -> {new['status']} | old {old['errors']} | new {new['errors']} | warn {new['warnings']}")
        if apply:
            new["reparsed_from"] = {"status": old["status"], "errors": old["errors"]}
            write_json(ep, new); extract_all(d)
print(f"{'applied' if apply else 'would change'}: {changed}")
