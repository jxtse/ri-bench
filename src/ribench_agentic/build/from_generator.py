import copy
import json
from pathlib import Path

from .common import allowed_arms, identifier, validate_injection
from .court import build_court_bundle
from .parliament import build_parliament_bundle
from ..casefs import verify_sha256


def to_injection(record):
    if record.get("status") != "ok":
        raise ValueError("Generator record must have status ok")
    scenario = record["scenario"]
    if scenario not in {"court", "parliament"}:
        raise ValueError("Unknown generator scenario")
    injection = copy.deepcopy(record["spec"])
    injection.update(scenario=scenario, case_id=identifier(record["case_id"]))
    if scenario == "court":
        if "arm_side" in record:
            injection["arm_side"] = copy.deepcopy(record["arm_side"])
    else:
        injection["design"] = record["design"]
    validate_injection(injection)
    return injection


def build_bundles(scenario, injections, cases_root="data/cases", output_root="data/agentic"):
    result = {"built": 0, "skipped_existing": 0, "skipped_records": 0, "bundles": [], "errors": []}
    directory = "court" if scenario == "court" else "parliament"
    builder = build_court_bundle if scenario == "court" else build_parliament_bundle
    with Path(injections).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") != "ok":
                    result["skipped_records"] += 1
                    continue
                if record["scenario"] != scenario:
                    raise ValueError("Generator scenario does not match --scenario")
                injection = to_injection(record)
                case_id = injection["case_id"]
                case = json.loads((Path(cases_root) / directory / f"{case_id}.json").read_text(encoding="utf-8"))
                for arm in sorted(allowed_arms(scenario)):
                    destination = Path(output_root) / scenario / f"{case_id}__{arm}"
                    if destination.exists():
                        verify_sha256(destination)
                        manifest = json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8"))
                        if manifest.get("injection") != injection:
                            raise ValueError(f"Existing bundle injection mismatch: {destination}")
                        result["skipped_existing"] += 1
                    else:
                        builder(case, injection, arm, output_root=output_root)
                        result["built"] += 1
                    result["bundles"].append(str(destination))
            except (ValueError, KeyError, TypeError, OSError) as error:
                result["errors"].append({"line": line_number, "error": str(error)})
    return result
