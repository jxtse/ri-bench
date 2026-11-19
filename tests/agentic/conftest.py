import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cases():
    return json.loads((FIXTURES / "cases.json").read_text())


@pytest.fixture
def injections():
    return json.loads((FIXTURES / "injections.json").read_text())


@pytest.fixture
def trace():
    return (FIXTURES / "pi_trace_sample.jsonl").read_bytes()


@pytest.fixture
def fake_pi(tmp_path):
    script = tmp_path / "fake-pi"
    script.write_text(f'#!/bin/sh\ncat "{FIXTURES / "pi_trace_sample.jsonl"}"\n')
    script.chmod(0o755)
    return str(script)
