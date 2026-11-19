import json
from pathlib import Path
import re
import shutil
import tempfile

from jsonschema import Draft202012Validator, ValidationError

from ..casefs import sha256, safe_path, write_json
from ..prompts import render


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Unsafe identifier")
    return value


def validate_injection(injection):
    schema = json.loads((Path(__file__).parents[1] / "schemas/injection.schema.json").read_text())
    try:
        Draft202012Validator(schema).validate(injection)
        rule_sets = [witness["rules"] for witness in injection["witnesses"].values()] if injection["scenario"] == "parliament" else [injection["rules"]]
        for rules in rule_sets:
            for patterns in rules.values():
                for pattern in patterns:
                    re.compile(pattern, re.I)
    except (ValidationError, re.error) as error:
        raise ValueError(f"Invalid injection: {error}") from error


def allowed_arms(scenario):
    if scenario == "parliament":
        return {"P"}
    if scenario == "open_review_v3":  # constructed adversarial votes (PREREG_OPEN_REVIEW_V3)
        return {"C", "X"}
    if scenario.startswith("or_"):  # follow-up OpenReview designs: C/X as v3, P paper only, SP/SM 2:2 specific side +/-
        return {"C", "X", "P", "SP", "SM"}
    return {"A", "B"}


def check_injection(case_id, scenario, injection, arm):
    validate_injection(injection)
    if arm not in allowed_arms(scenario) or injection["scenario"] != scenario or injection["case_id"] != case_id:
        raise ValueError("Injection scenario, case or arm mismatch")


def forbidden_values(case, scenario):
    from ..casefs import SCENARIO_FORBIDDEN
    values = []
    pattern = re.compile(SCENARIO_FORBIDDEN[scenario] + r"|ground_truth|^decision$|^meta_review$", re.I)

    GENERIC = {"accept", "reject", "petitioner", "respondent", "a", "b", "majority", "dissent", "plurality",
               "unanimous", "affirmed", "reversed", "vacated", "remanded", "per curiam"}

    def collect(value):
        if isinstance(value, str) and value.strip():
            v = value.strip()
            # only specific values are leak signals: docket-like ids, URLs, or long distinctive strings.
            # Short generic legal words (e.g. "majority", a justice's surname) occur in ordinary facts text.
            # docket numbers look like 21-1086 (two-digit term, 1-5 digit sequence); bare vote tallies like 5-4 are
            # legitimately mentioned in facts (e.g. an earlier stay vote) and the vote field itself is never copied
            specific = bool(re.fullmatch(r"\d{2}-\d{2,5}[A-Za-z]?", v)) or v.startswith("http") or len(v) >= 40
            if v.lower() not in GENERIC and specific:
                values.append(v)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    for key, value in case.items():
        if pattern.search(key):
            collect(value)
    return sorted(set(values))


def publish_bundle(scenario, case_id, arm, documents, private, output_root="data/agentic"):
    identifier(case_id)
    if arm not in allowed_arms(scenario):
        raise ValueError("Invalid arm")
    parent = Path(output_root) / scenario
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{case_id}__{arm}"
    if destination.exists():
        raise FileExistsError(destination)
    scratch = Path(tempfile.mkdtemp(prefix=".bundle-", dir=parent))
    try:
        files = []
        for relative, description, role, text in documents:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing document content: {relative}")
            target = safe_path(scratch, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            files.append({"path": relative, "description": description, "role": role, "sha256": sha256(target.read_bytes())})
        manifest = {**private, "scenario": scenario, "case_id": case_id, "arm": arm, "files": files}
        files.append({"path": "README.md", "description": "Task, file manifest and output format", "role": "instructions", "sha256": ""})
        system, prompt = render(manifest, "vanilla")
        (scratch / "README.md").write_text(system + "\n\n" + prompt + "\n", encoding="utf-8")
        files[-1]["sha256"] = sha256((scratch / "README.md").read_bytes())
        write_json(scratch / "MANIFEST.json", manifest)
        scratch.rename(destination)
    except Exception:
        shutil.rmtree(scratch)
        raise
    return destination
