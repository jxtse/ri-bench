"""Allowlisted case copying, leakage checks and exact file-tree integrity."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from .prompts import render, TASKS


# Bare "strong"/"weak" are argument-quality labels only in open_review; in real court/parliament
# documents they are ordinary English and must not trip the scan.
COMMON_FORBIDDEN = r'\b(?:ground_truth\w*|argument_quality|reviewer_assignment)\b|["\']arm["\']\s*:|\barm\s+[AB]\b'
SCENARIO_FORBIDDEN = {
    # bare "disposition"/"docket" are ordinary legal English in oral-argument excerpts; match private keys only
    "court": r"\b(?:full_conclusion|\w+_opinion_excerpt|vote_\w*|\w+_vote|majority_author|dissent_authors|case_name|oyez_url|ground_truth_verdict|arm_side)\b|[\"'](?:disposition|docket|docket_number)[\"']\s*:",
    # Real submissions legitimately contain the English words aligned/opposed/stance/rules; only the
    # private-manifest identifiers (JSON keys and pattern-list names) are leakage signals.
    "parliament": r"\b(?:gt_\w*|publication_\w*|disputed_facts|stance_rationale|witness_index|claim_patterns|qualifier_patterns|hedge_patterns|qualified_text|anchor_text)\b|[\"'](?:aligned|opposed|rules)[\"']\s*:",
    # bare strong/weak are ordinary review prose ("strong claims", "weakening"); only label fields leak
    "open_review": r"\b(?:rebuttal_validity_ground_truth|real_decision|published_decision|argument_quality|reviewer_assignment|majority_stance)\b|[\"'](?:strong|weak)[\"']\s*[:,]|[\"'](?:truth|specific|generic)[\"']\s*:",
}
# v3 (constructed votes) shares the review key list; `truth` (ICLR real decision) is private there too
SCENARIO_FORBIDDEN["open_review_v3"] = SCENARIO_FORBIDDEN["open_review"]
for _scenario in ("or_r", "or_e1", "or_e2a", "or_e2b", "or_e3a", "or_e3b", "or_e3c"):
    SCENARIO_FORBIDDEN[_scenario] = SCENARIO_FORBIDDEN["open_review"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink()
            raise
    temporary.replace(path)


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Unsafe manifest path: {relative}")
    target = Path(root) / relative
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes case directory")
    if any(part.is_symlink() for part in [target, *target.parents] if part != Path(root).parent):
        raise ValueError("Symlinks are not case files")
    return target


def snapshot(root):
    root = Path(root)
    state = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("integrity: symlink in case directory")
        if path.is_file():
            state[path.relative_to(root).as_posix()] = sha256(path.read_bytes())
    return state


def verify_sha256(root, expected=None):
    root = Path(root)
    current = snapshot(root)
    if expected is not None:
        if current != expected:
            raise ValueError("integrity: file tree changed")
        return current
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.exists():
        manifest_path = root / "MANIFEST.public.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = set()
    for entry in manifest["files"]:
        path = safe_path(root, entry["path"])
        if entry["path"] in paths:
            raise ValueError("integrity: duplicate manifest path")
        paths.add(entry["path"])
        if not path.is_file() or sha256(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"integrity: changed or missing {entry['path']}")
    if set(current) != paths | {manifest_path.name, "README.md"}:
        raise ValueError("integrity: unmanifested or missing files")
    return current


def leakage_scan(root, scenario, forbidden_strings=()):
    pattern = re.compile(COMMON_FORBIDDEN + "|" + SCENARIO_FORBIDDEN[scenario], re.I)
    leaks = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            leaks.append(f"{path.name}: symlink")
        elif path.is_file():
            text = path.relative_to(root).as_posix() + "\n" + path.read_text(encoding="utf-8")
            scan_text = text
            if scenario == "court" and path == Path(root) / "README.md":
                scan_text = scan_text.replace(TASKS["judge"], "", 1)
            if pattern.search(scan_text) or any(value and value.casefold() in text.casefold() for value in forbidden_strings):
                leaks.append(path.relative_to(root).as_posix())
    return leaks


def build_run_dir(bundle, run_dir, condition, role=None, area_chair_text=None):
    bundle, run_dir = Path(bundle).resolve(), Path(run_dir).resolve()
    if run_dir == bundle or run_dir.is_relative_to(bundle):
        raise ValueError("Run directory must be outside immutable bundle")
    verify_sha256(bundle)
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    public = {"scenario": manifest["scenario"], "files": [
        {key: entry[key] for key in ("path", "description", "role", "sha256")}
        for entry in manifest["files"]
    ]}
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        for entry in public["files"]:
            target = safe_path(run_dir, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(bundle, entry["path"]), target)
        if area_chair_text is not None:
            relative = "docs/meta_review/area_chair.md"
            target = safe_path(run_dir, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(area_chair_text, encoding="utf-8")
            public["files"].append({"path": relative, "description": "Area Chair meta-review from this episode", "role": "meta_review", "sha256": sha256(target.read_bytes())})
        system, prompt = render(public, condition, role)
        (run_dir / "README.md").write_text(system + "\n\n" + prompt + "\n", encoding="utf-8")
        for entry in public["files"]:
            if entry["path"] == "README.md":
                entry["sha256"] = sha256((run_dir / "README.md").read_bytes())
        write_json(run_dir / "MANIFEST.public.json", public)
        leaks = leakage_scan(run_dir, manifest["scenario"], manifest.get("forbidden_strings", []))
        if leaks:
            raise ValueError("Leakage in agent copy: " + ", ".join(leaks))
        verify_sha256(run_dir)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    return run_dir
