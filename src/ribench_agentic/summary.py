"""Invocation and token counts only, not behavioral outcome statistics."""

import argparse
import json
from pathlib import Path

from .pi_runner import TOKEN_FIELDS


def summarize(ledger_path):
    models = {}
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result = models.setdefault(row["model"], {"invocations": 0, "statuses": {}, "usage": {key: 0 for key in TOKEN_FIELDS}, "missing_usage": {key: 0 for key in TOKEN_FIELDS}})
        result["invocations"] += 1
        result["statuses"][row["status"]] = result["statuses"].get(row["status"], 0) + 1
        for key in TOKEN_FIELDS:
            value = row["usage"].get(key)
            if value is None:
                result["missing_usage"][key] += 1
            else:
                result["usage"][key] += value
    for result in models.values():
        result["known_usage"] = dict(result["usage"])
        for key in TOKEN_FIELDS:
            if result["missing_usage"][key]:
                result["usage"][key] = None
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    print(json.dumps(summarize(parser.parse_args().ledger), indent=2))


if __name__ == "__main__":
    main()
