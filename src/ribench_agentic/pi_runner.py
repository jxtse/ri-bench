"""One pi invocation, authoritative message_end parsing, no implicit retries."""

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

from .casefs import snapshot, verify_sha256, write_json


TOKEN_FIELDS = ("input", "output", "reasoning", "cacheRead")
DEFAULT_AGENT_DIR = Path(__file__).resolve().parents[2] / "configs/agentic/pi-agent"
# pi extension that refuses any read/ls/grep path resolving outside the case workspace before the tool runs
# (added 2026-09-25; the paper's original runs had only the post-hoc trace check below).
WORKSPACE_GUARD = DEFAULT_AGENT_DIR / "workspace-guard.ts"
GUARD_MARK = "RIBENCH_WORKSPACE_GUARD"


def command(prompt, system_prompt, *, model, thinking="medium", pi_binary="pi"):
    return [str(pi_binary), "-p", "--mode", "json", "--no-session", "--no-skills",
            "--no-extensions", "-e", str(WORKSPACE_GUARD), "--no-context-files", "--no-prompt-templates",
            "--tools", "read,ls,grep", "--thinking", thinking, "--model", model,
            "--system-prompt", system_prompt, prompt]


def parse_trace(raw, returncode=0, workspace=None):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    messages, tools, errors, warnings = [], [], [], []
    pending = {}
    ended = False
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            kind = event["type"]
            if kind == "message_end" and event["message"].get("role") == "assistant":
                message = event["message"]
                messages.append({key: message.get(key) for key in ["model", "provider", "responseModel", "usage", "stopReason", "errorMessage"]} | {
                    "text": "".join(block["text"] for block in message.get("content", []) if block.get("type") == "text")})
            elif kind == "tool_execution_start":
                if event["toolName"] not in {"read", "ls", "grep"}:
                    errors.append("Unexpected tool outside read-only allowlist")
                arguments = event["args"]
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                path = arguments.get("path", ".")
                escape = False
                if not isinstance(path, str) or path.startswith("~"):
                    escape = True
                else:
                    # relative paths must not escape; absolute paths are accepted only when they resolve inside the
                    # workspace (some models echo the absolute cwd-prefixed path)
                    candidate = PurePosixPath(path)
                    if candidate.is_absolute():
                        root = os.path.realpath(str(workspace)) if workspace is not None else None
                        real = os.path.realpath(path)
                        if root is None or not (real == root or real.startswith(root + os.sep)):
                            # a mistyped absolute path that does not exist cannot read anything; record it as a
                            # tool-side failure (the model gets an error result), not as a workspace escape
                            if os.path.exists(real):
                                escape = True
                            else:
                                warnings.append("Tool path invalid (nonexistent absolute path)")
                    elif ".." in candidate.parts:
                        escape = True
                tool = {"id": event["toolCallId"], "name": event["toolName"], "args": event["args"], "isError": None, "result_size": None,
                        "escape": escape}
                tools.append(tool)
                pending[tool["id"]] = tool
            elif kind == "tool_execution_end":
                tool = pending.pop(event["toolCallId"])
                result_json = json.dumps(event["result"], ensure_ascii=False)
                tool.update(isError=event["isError"], result_size=len(result_json.encode("utf-8")))
                if tool.pop("escape"):
                    # refused by the workspace guard before execution: nothing outside the workspace reached the model
                    if event["isError"] and GUARD_MARK in result_json:
                        tool["guard_refused"] = True
                        warnings.append("Tool path outside case workspace refused by guard")
                    else:
                        errors.append("Tool path outside case workspace")
            elif kind == "agent_end":
                ended = True
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(f"Invalid JSON event at line {line_number}: {type(error).__name__}")
    final = messages[-1] if messages else {}
    text = final.get("text", "")
    if returncode != 0:
        errors.append(f"pi exit code {returncode}")
    if any(tool.get("escape") for tool in pending.values()):
        errors.append("Tool path outside case workspace")
    for tool in pending.values():
        tool.pop("escape", None)
    if not ended or pending:
        errors.append("Incomplete agent event stream")
    if not text.strip():
        errors.append("Empty final assistant text")
    for message in messages:
        if message["stopReason"] in {"error", "aborted"}:
            errors.append(message.get("errorMessage") or message["stopReason"])
    usage = {}
    for field in TOKEN_FIELDS:
        values = [(message.get("usage") or {}).get(field) for message in messages]
        usage[field] = sum(values) if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) else None
    files_read = sorted({tool["args"]["path"] for tool in tools if tool["name"] == "read" and tool["isError"] is False and isinstance(tool["args"].get("path"), str)})
    return {"status": "error" if errors else "complete", "final_text": text,
            "messages": messages, "tools": tools, "usage": usage, "files_read": files_read,
            "model": final.get("model"), "provider": final.get("provider"),
            "returned_model": final.get("responseModel") or final.get("model"),
            "stopReason": final.get("stopReason"), "errorMessage": final.get("errorMessage"), "errors": errors, "warnings": warnings}


def run_pi(run_dir, prompt, system_prompt, *, model, thinking="medium", timeout_s=900,
           pi_agent_dir=DEFAULT_AGENT_DIR, output_dir=None, pi_binary="pi"):
    workspace = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else workspace.parent
    if output == workspace or output.is_relative_to(workspace):
        raise ValueError("Trace and episode output must remain outside the agent workspace")
    if (output / "episode.json").exists():
        raise FileExistsError("Never overwrite a recorded invocation")
    if timeout_s <= 0:
        raise ValueError("Timeout must be positive")
    expected = snapshot(workspace)
    if (workspace / "MANIFEST.public.json").exists():
        verify_sha256(workspace)
    output.mkdir(parents=True, exist_ok=True)
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "FOUNDRY_API_KEY", "MINT_API_KEY", "ARK_API_KEY", "TOKENDANCE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "VLLM_API_KEY", "NODE_USE_ENV_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy") if key in os.environ}
    agent_dir = Path(pi_agent_dir).resolve()
    if not (agent_dir / "models.json").exists() and (agent_dir / "models.example.json").exists():
        # models.json is the user's local (ignored) provider config; fall back to the example, which names the routes
        # used in the paper and references credentials only through $ENV variables
        import shutil as _shutil
        _shutil.copy(agent_dir / "models.example.json", agent_dir / "models.json")
    environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    argv = command(prompt, system_prompt, model=model, thinking=thinking, pi_binary=pi_binary)
    started = time.monotonic()
    raw, returncode, timed_out, process_error = b"", -1, False, None
    try:
        process = subprocess.run(argv, cwd=workspace, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=timeout_s, check=False)
        raw, returncode = process.stdout, process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        raw = error.output or b""
        process_error = "pi timeout"
    except OSError as error:
        process_error = f"pi launch failed: {error}"
    elapsed = time.monotonic() - started
    (output / "trace.jsonl").write_bytes(raw)
    result = parse_trace(raw, returncode, workspace=workspace)
    result.update(wall_seconds=elapsed, requested_model=model, returncode=returncode,
                  timed_out=timed_out, attempt=1, integrity_error=None)
    if process_error:
        result["errors"].append(process_error)
    try:
        verify_sha256(workspace, expected)
    except ValueError as error:
        result["integrity_error"] = str(error)
        result["errors"].append(str(error))
    if result["errors"]:
        result["status"] = "error"
    write_json(output / "episode.json", result)
    return result
