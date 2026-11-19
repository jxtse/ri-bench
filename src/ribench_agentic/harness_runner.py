"""Alternative senior-agent harnesses for the harness-variance experiment: the same model under a different agent
loop, with the same confinement to the case directory as the pi harness.

Two harnesses, one adapter each, producing the same episode.json shape as pi_runner.run_pi so that
extract/score/export are unchanged:

- codex: OpenAI Codex CLI (`codex exec --json`). Filesystem confinement is Codex's own Seatbelt sandbox driven by
  the permissions profile in the isolated CODEX_HOME (read only the workspace root, deny everything else, no
  network for tools). Model: built-in OpenAI provider (ChatGPT login). The gateway the pi arm uses for
  gpt-5.6-sol strips Codex's namespaced `exec` tool, so the Codex arm shares the model name but not the serving
  endpoint (stated in the paper's appendix).
- dsh: DeepSeek Harness (`dsh --profile headless`). dsh's own file policy does not path-scope reads (verified:
  it read ../outside.txt), so tools/dsh_confined.sh wraps the whole process in macOS sandbox-exec with
  file-read* limited to the workspace + harness/runtime code. Tool surface is trimmed to read/glob/grep
  (no bash, no write/edit) in the isolated DSH_HOME profile patch.

Both adapters also run the same static trace audit as pi_runner (every observed path must resolve inside the
workspace) and the same before/after sha256 snapshot of the workspace.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath

from .casefs import snapshot, verify_sha256, write_json
from .pi_runner import TOKEN_FIELDS

REPO = Path(__file__).resolve().parents[2]
DSH_CONFINED = REPO / "tools" / "dsh_confined.sh"
DEFAULT_CODEX_HOME = Path(os.environ.get("RIBENCH_CODEX_HOME", REPO / ".local" / "codex-home"))
DEFAULT_DSH_HOME = Path(os.environ.get("RIBENCH_DSH_HOME", REPO / ".local" / "dsh-home"))
DSH_REPO = Path(os.environ.get("RIBENCH_DSH_REPO", Path.home() / "Documents/GitHub/deepseek-harness"))
HARNESSES = ("codex", "dsh")

_PATH_TOKEN = re.compile(r"""(?<![\w/.-])((?:\.\./|/)[^\s'"`;|&<>()]+)""")


def _path_check(path, workspace, errors, warnings):
    """Same rule as pi_runner: relative paths must not climb; absolute paths must resolve inside the workspace."""
    if not isinstance(path, str) or path.startswith("~"):
        errors.append("Tool path outside case workspace")
        return
    candidate = PurePosixPath(path)
    root = os.path.realpath(str(workspace))
    if candidate.is_absolute():
        real = os.path.realpath(path)
        if not (real == root or real.startswith(root + os.sep)):
            if os.path.exists(real):
                errors.append("Tool path outside case workspace")
            else:
                warnings.append("Tool path invalid (nonexistent absolute path)")
    elif ".." in candidate.parts:
        errors.append("Tool path outside case workspace")


_SYSTEM_PREFIXES = ("/bin/", "/usr/bin/", "/usr/sbin/", "/sbin/", "/dev/", "/opt/homebrew/bin/")
# sed/perl substitution fragments (`/g`, `//gs`, `/\\n/g`) are not paths: require a token to hold a word character
_PATHLIKE = re.compile(r"[A-Za-z0-9_.-]{2,}")


def _audit_shell(command, workspace, errors, warnings):
    """Codex exposes a shell; audit every path-looking token in the command text. Interpreter/utility paths
    (`/bin/zsh -lc`, `/usr/bin/sed`) are executables, not case reads; the Seatbelt profile still blocks reads
    of anything outside the workspace, so this static audit is the second layer, like pi_runner's."""
    for token in _PATH_TOKEN.findall(command or ""):
        if token.startswith(_SYSTEM_PREFIXES) or not _PATHLIKE.search(token):
            continue
        _path_check(token, workspace, errors, warnings)


# ---------------------------------------------------------------- Codex
def codex_command(prompt, system_prompt, *, model, thinking="medium", codex_binary="codex", output_last=None):
    # the isolated CODEX_HOME (RIBENCH_CODEX_HOME) carries the read-only permissions profile and feature flags;
    # --ignore-rules is deliberately NOT passed (it made Codex drop its exec tool in probes)
    argv = [codex_binary, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
            "-m", model, "-c", f'model_reasoning_effort="{thinking}"',
            "-c", f"developer_instructions={json.dumps(system_prompt)}"]
    if output_last:
        argv += ["-o", str(output_last)]
    argv.append(prompt)
    return argv


def _settle_extra_tools(tools, errors, warnings):
    """Tools the harness offers beyond pi's read/ls/grep (dsh `write`, Codex `web_search`...). The permission
    layer (dsh read-only mode / Codex ribench-ro profile, network off) denies them; a denied attempt is a
    harness-behaviour fact worth keeping (warning), a successful one breaks the read-only contract (error)."""
    for tool in tools:
        if not tool.get("extra"):
            continue
        if tool["isError"] is False:
            errors.append(f"Extra harness tool succeeded: {tool['name']}")
        else:
            warnings.append(f"Extra harness tool attempted and denied: {tool['name']}")


def parse_codex_trace(raw, returncode=0, workspace=None):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    tools, errors, warnings, texts = [], [], [], []
    usage = {field: 0 for field in TOKEN_FIELDS}
    completed = False
    returned_model = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            errors.append(f"Invalid JSON event at line {line_number}: ValueError")
            continue
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "command_execution":
            command = item.get("command", "")
            _audit_shell(command, workspace, errors, warnings)
            tools.append({"id": item.get("id"), "name": "shell", "args": {"command": command},
                          "isError": (item.get("exit_code") not in (0, None)),
                          "result_size": len((item.get("aggregated_output") or "").encode("utf-8"))})
        elif kind == "item.completed" and item.get("type") in {"file_change", "mcp_tool_call", "web_search"}:
            # Tools Codex offers beyond pi's read/ls/grep. Network is off and the profile is read-only, so these
            # are denied attempts; recorded as a harness-behaviour warning, not an episode error (see _settle_extra_tools).
            tools.append({"id": item.get("id"), "name": item.get("type"), "args": {"query": item.get("query", "")} if item.get("type") == "web_search" else {},
                          "isError": None, "result_size": None, "extra": True})
        elif kind == "item.completed" and item.get("type") == "agent_message":
            texts.append(item.get("text", ""))
        elif kind == "item.completed" and item.get("type") == "error":
            message = item.get("message", "")
            if "skills context budget" not in message:
                warnings.append(f"codex: {message[:200]}")
        elif kind == "turn.completed":
            completed = True
            u = event.get("usage") or {}
            usage["input"] += int(u.get("input_tokens") or 0) - int(u.get("cached_input_tokens") or 0)
            usage["cacheRead"] += int(u.get("cached_input_tokens") or 0)
            usage["output"] += int(u.get("output_tokens") or 0)
            usage["reasoning"] += int(u.get("reasoning_output_tokens") or 0)
        elif kind == "turn.failed" or kind == "error":
            message = (event.get("error") or {}).get("message") if isinstance(event.get("error"), dict) else event.get("message")
            errors.append(f"codex: {str(message)[:200]}")
        elif kind == "thread.started":
            returned_model = event.get("model") or returned_model
    text = texts[-1] if texts else ""
    _settle_extra_tools(tools, errors, warnings)
    if returncode != 0:
        errors.append(f"codex exit code {returncode}")
    if not completed:
        errors.append("Incomplete agent event stream")
    if not text.strip():
        errors.append("Empty final assistant text")
    files_read = sorted({m for tool in tools for m in _PATH_TOKEN.findall(tool["args"].get("command", ""))
                         if not tool["isError"] and not m.startswith(_SYSTEM_PREFIXES) and _PATHLIKE.search(m)}
                        | {tok for tool in tools if not tool["isError"] for tok in re.findall(r"(?<![\w/])([\w.-]+(?:/[\w.-]+)*\.(?:md|txt|json))", tool["args"].get("command", ""))})
    return {"status": "error" if errors else "complete", "final_text": text, "messages": [],
            "tools": tools, "usage": usage, "files_read": files_read, "model": returned_model, "provider": "codex",
            "returned_model": returned_model, "stopReason": "stop" if completed else None, "errorMessage": None,
            "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- dsh
def dsh_command(prompt, system_prompt, *, model, thinking="high"):
    # system prompt goes through the headless persona; the model/effort are fixed in the isolated DSH_HOME settings
    task = f"{system_prompt}\n\n{prompt}"
    return ["bash", str(DSH_CONFINED), task]


def _dsh_session_events(dsh_home, workspace, started_after):
    """Locate the session dsh persisted for this run (sessions/<cwd-slug>/session-*/session.jsonl[.zstd])."""
    slug = "--" + str(workspace).strip("/").replace("/", "-") + "--"
    root = Path(dsh_home) / "sessions" / slug
    if not root.exists():
        return None, []
    candidates = [p for p in root.glob("session-*") if p.stat().st_mtime >= started_after - 2]
    if not candidates:
        return None, []
    session = max(candidates, key=lambda p: p.stat().st_mtime)
    raw = None
    for name in ("session.jsonl", "session.jsonl.zstd"):
        f = session / name
        if f.exists():
            if name.endswith(".zstd"):
                raw = subprocess.run(["zstd", "-dc", str(f)], stdout=subprocess.PIPE, check=False).stdout
            else:
                raw = f.read_bytes()
            break
    events = []
    for line in (raw or b"").decode("utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    return session, events


def parse_dsh_trace(stdout, events, returncode=0, workspace=None):
    """dsh session.jsonl schema (verified 2026-09-24 against a real headless run):
    tool/call   {data:{callId,name,arguments:<json string>}}
    tool/result {data:{message:{content:[{type:'tool-result',toolCallId,content:[...],isError}]}}}
    assistant/message {data:{usage:{inputTokens,outputTokens,cacheReadTokens,reasoningTokens},message:{source:{model}}}}
    turn/end    {data:{reason:{kind:'completed'|...}}}
    inputTokens already excludes cacheReadTokens (DeepSeek prompt_cache_miss / prompt_cache_hit)."""
    tools, errors, warnings = [], [], []
    usage = {field: 0 for field in TOKEN_FIELDS}
    turn_end, saw_usage = None, False
    returned_model = None
    pending = {}
    for event in events:
        kind = event.get("type") or ""
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind == "tool/call":
            name = data.get("name")
            args = data.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"raw": str(args)}
            tool_extra = name not in {"read", "glob", "grep"}
            for key in ("file_path", "path", "directory", "dir", "cwd"):
                if key in args:
                    _path_check(args[key], workspace, errors, warnings)
            if name == "glob" and isinstance(args.get("pattern"), str) and (args["pattern"].startswith("/") or ".." in PurePosixPath(args["pattern"]).parts):
                _path_check(args["pattern"], workspace, errors, warnings)
            tool = {"id": data.get("callId"), "name": name, "args": args, "isError": None, "result_size": None, "extra": tool_extra}
            tools.append(tool)
            if tool["id"]:
                pending[tool["id"]] = tool
        elif kind == "tool/result":
            for part in ((data.get("message") or {}).get("content") or []):
                if part.get("type") != "tool-result":
                    continue
                tool = pending.pop(part.get("toolCallId"), None)
                if tool is not None:
                    tool["isError"] = bool(part.get("isError"))
                    tool["result_size"] = len(json.dumps(part.get("content") or "", ensure_ascii=False).encode("utf-8"))
        elif kind == "assistant/message":
            u = data.get("usage") if isinstance(data.get("usage"), dict) else None
            if u:
                saw_usage = True
                usage["input"] += int(u.get("inputTokens") or 0)
                usage["cacheRead"] += int(u.get("cacheReadTokens") or 0)
                usage["output"] += int(u.get("outputTokens") or 0)
                usage["reasoning"] += int(u.get("reasoningTokens") or 0)
            source = (data.get("message") or {}).get("source") or {}
            returned_model = source.get("model") or returned_model
        elif kind == "turn/end":
            turn_end = ((data.get("reason") or {}).get("kind")) or "completed"
    _settle_extra_tools(tools, errors, warnings)
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else (stdout or "")
    text = text.strip()
    if returncode != 0:
        errors.append(f"dsh exit code {returncode}")
    if turn_end is None or pending:
        errors.append("Incomplete agent event stream")
    elif turn_end != "completed":
        errors.append(f"dsh turn ended: {turn_end}")
    if not text:
        errors.append("Empty final assistant text")
    if not saw_usage:
        warnings.append("dsh: no usage in session log")
        usage = {field: None for field in TOKEN_FIELDS}
    files_read = sorted({tool["args"].get("file_path") or tool["args"].get("path")
                         for tool in tools if tool["name"] == "read" and tool["isError"] is False} - {None})
    return {"status": "error" if errors else "complete", "final_text": text, "messages": [],
            "tools": tools, "usage": usage, "files_read": files_read, "model": returned_model, "provider": "dsh",
            "returned_model": returned_model, "stopReason": turn_end, "errorMessage": None,
            "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- shared driver
def run_harness(harness, run_dir, prompt, system_prompt, *, model, thinking="medium", timeout_s=900, output_dir=None,
                codex_home=DEFAULT_CODEX_HOME, dsh_home=DEFAULT_DSH_HOME):
    if harness not in HARNESSES:
        raise ValueError(f"Unknown harness {harness}")
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
    base_env = {key: os.environ[key] for key in ("PATH", "HOME", "DEEPSEEK_API_KEY", "FOUNDRY_API_KEY",
                                                   "NODE_USE_ENV_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
                                                   "https_proxy", "http_proxy", "no_proxy") if key in os.environ}
    started_wall = time.time()
    started = time.monotonic()
    raw, stderr, returncode, timed_out, process_error = b"", b"", -1, False, None
    if harness == "codex":
        env = base_env | {"CODEX_HOME": str(codex_home)}
        argv = codex_command(prompt, system_prompt, model=model, thinking=thinking, output_last=output / "last_message.txt")
    else:
        env = base_env | {"DSH_HOME": str(dsh_home), "DSH_REPO": str(DSH_REPO), "DSH_PERMISSION_MODE": "read-only",
                          "DSH_TELEMETRY_DISABLED": "1"}
        argv = dsh_command(prompt, system_prompt, model=model, thinking=thinking)
    try:
        process = subprocess.run(argv, cwd=workspace, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=timeout_s, check=False)
        raw, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        raw = error.output or b""
        stderr = error.stderr or b""
        process_error = f"{harness} timeout"
    except OSError as error:
        process_error = f"{harness} launch failed: {error}"
    elapsed = time.monotonic() - started
    (output / "trace.jsonl").write_bytes(raw)
    (output / "stderr.txt").write_bytes(stderr[-20000:])
    if harness == "codex":
        result = parse_codex_trace(raw, returncode, workspace=workspace)
    else:
        session, events = _dsh_session_events(dsh_home, workspace, started_wall)
        if session is not None:
            (output / "dsh_session.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
        result = parse_dsh_trace(raw, events, returncode, workspace=workspace)
        if session is None:
            result["errors"].append("dsh session log not found")
    result.update(wall_seconds=elapsed, requested_model=model, returncode=returncode, timed_out=timed_out, attempt=1,
                  integrity_error=None, harness=harness)
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
