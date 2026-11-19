"""Append-only per-invocation accounting, including failed calls."""

import fcntl
import json
import os
from pathlib import Path

from .casefs import sha256


def append_invocation(path, context, result, system, prompt, role, invocation_id=None):
    row = {key: context[key] for key in ("scenario", "case_id", "arm", "condition", "model", "bundle_sha256")}
    row.update(role=role, provider=result["provider"], returned_model=result["returned_model"],
               usage=result["usage"], wall_seconds=result["wall_seconds"], status=result["status"],
               tool_call_count=len(result["tools"]), files_read=result["files_read"],
               prompt_sha256=sha256(json.dumps([system, prompt], ensure_ascii=False).encode("utf-8")),
               attempt=result["attempt"], invocation_id=invocation_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if invocation_id is not None:
            stream.seek(0)
            existing = [json.loads(line) for line in stream]
            previous = next((entry for entry in existing if entry.get("invocation_id") == invocation_id), None)
            if previous is not None:
                differences = [key for key in ("status", "usage", "returned_model") if previous.get(key) != row[key]]
                if differences:
                    conflict = {"invocation_id": invocation_id, "fields": differences,
                                "previous": previous, "recorded": row}
                    with path.with_name("ledger_conflict.jsonl").open("a", encoding="utf-8") as conflicts:
                        conflicts.write(json.dumps(conflict, ensure_ascii=False, sort_keys=True) + "\n")
                        conflicts.flush()
                        os.fsync(conflicts.fileno())
                return previous
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    return row
