import json
from pathlib import Path

import pytest

from ribench_agentic.pi_runner import command, parse_trace, run_pi


def test_real_trace(trace):
    result = parse_trace(trace)
    assert result["status"] == "complete"
    assert [tool["name"] for tool in result["tools"]] == ["ls", "read", "read"]
    assert all(tool["result_size"] > 0 and not tool["isError"] for tool in result["tools"])
    assert result["final_text"].startswith("DECISION: Defer the proposal")
    assert result["usage"] == {"input": 1735, "output": 173, "reasoning": 65, "cacheRead": 0}
    assert result["model"] == "gpt-5.6-sol"
    assert result["provider"] == "foundry-a"
    assert result["returned_model"] == "gpt-5.6-sol-2026-06-26"
    assert len(result["messages"]) == 3
    assert result["files_read"] == ["docs/witness_a.md", "docs/witness_b.md"]


@pytest.mark.parametrize("mutation", ["error", "empty", "nonzero", "malformed", "truncated"])
def test_error_never_success(trace, mutation):
    events = [json.loads(line) for line in trace.splitlines()]
    last = [event for event in events if event["type"] == "message_end" and event["message"]["role"] == "assistant"][-1]
    if mutation == "error":
        last["message"].update(stopReason="error", errorMessage="provider failed")
    if mutation == "empty":
        last["message"]["content"] = []
    if mutation == "truncated":
        events = events[:-3]
    data = "\n".join(json.dumps(event) for event in events).encode()
    if mutation == "malformed":
        data += b"\nnot-json"
    result = parse_trace(data, returncode=1 if mutation == "nonzero" else 0)
    assert result["status"] == "error"


def test_missing_usage_is_unknown(trace):
    events = [json.loads(line) for line in trace.splitlines()]
    for event in events:
        if event["type"] == "message_end" and event["message"]["role"] == "assistant":
            event["message"].pop("usage")
    assert parse_trace("\n".join(json.dumps(event) for event in events))["usage"]["input"] is None


def test_fake_process_raw_trace_env_and_cwd(tmp_path, fake_pi, trace, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-pass")
    seen = {}
    import subprocess
    actual = subprocess.run

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return actual(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture)
    result = run_pi(workspace, "Only read listed files.", "You are Judge.", model="foundry-a/gpt-5.6-sol", output_dir=tmp_path / "output", pi_binary=fake_pi)
    assert result["status"] == "complete"
    assert (tmp_path / "output/trace.jsonl").read_bytes() == trace
    assert (tmp_path / "output/episode.json").exists()
    assert not list(workspace.iterdir())
    assert set(seen["env"]) <= {"PATH", "HOME", "FOUNDRY_API_KEY", "MINT_API_KEY", "ARK_API_KEY", "TOKENDANCE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "VLLM_API_KEY", "NODE_USE_ENV_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "PI_CODING_AGENT_DIR"}
    assert Path(seen["env"]["PI_CODING_AGENT_DIR"]).is_absolute()
    assert seen["cwd"] == workspace.resolve()
    assert seen["timeout"] == 900
    argv = command("task", "system", model="foundry-a/gpt-5.6-sol")
    assert argv[1:4] == ["-p", "--mode", "json"]
    for flag in ["--no-session", "--no-skills", "--no-extensions", "--no-context-files", "--no-prompt-templates"]:
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == "read,ls,grep"


def test_timeout_and_modified_input(tmp_path, monkeypatch, trace):
    import subprocess
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = workspace / "input.md"
    document.write_text("immutable")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=trace[:300])

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_pi(workspace, "task", "role", model="provider/model", output_dir=tmp_path / "timeout")
    assert result["status"] == "error" and result["timed_out"]
    assert (tmp_path / "timeout/trace.jsonl").read_bytes() == trace[:300]

    def modify(*args, **kwargs):
        document.write_text("mutated")
        return subprocess.CompletedProcess(args[0], 0, trace, b"")

    monkeypatch.setattr(subprocess, "run", modify)
    result = run_pi(workspace, "task", "role", model="provider/model", output_dir=tmp_path / "mutated")
    assert result["status"] == "error" and result["integrity_error"]


def test_absolute_path_inside_workspace_is_allowed(tmp_path):
    from ribench_agentic.pi_runner import parse_trace
    ws = tmp_path / "workspace"; (ws / "docs").mkdir(parents=True); (ws / "docs/a.md").write_text("x")
    ev = lambda path: json.dumps({"type": "tool_execution_start", "toolName": "read", "toolCallId": "t1", "args": {"path": path}})
    end = json.dumps({"type": "tool_execution_end", "toolCallId": "t1", "isError": False, "result": {}})
    inside = parse_trace("\n".join([ev(str(ws / "docs/a.md")), end]), workspace=ws)
    (tmp_path / "other.md").write_text("x")
    outside = parse_trace("\n".join([ev(str(tmp_path / "other.md")), end]), workspace=ws)
    typo = parse_trace("\n".join([ev(str(tmp_path / "does-not-exist" / "a.md")), end]), workspace=ws)
    assert "Tool path outside case workspace" not in typo["errors"] and typo["warnings"]
    nows = parse_trace("\n".join([ev(str(ws / "docs/a.md")), end]))
    assert "Tool path outside case workspace" not in inside["errors"]
    assert "Tool path outside case workspace" in outside["errors"]
    assert "Tool path outside case workspace" in nows["errors"]
