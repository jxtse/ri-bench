"""Follow-up bundles for the OpenReview constructed-vote family (docs/prereg/REBUTTAL_OPENREVIEW_DESIGNS.md).

Every design reads the de-identified paper text from data/cases/open_review_deid/<case>.md (sanitize_paper.py), never
the raw OpenReview PDF text. Reviews are either copied from the frozen v3 bundles (or_r, or_e2a, or_e3c) or generated
later from a skeleton built here (or_e2b, or_e3a, or_e3b).
"""

import json
import random
import re
from pathlib import Path

from .common import publish_bundle

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "data/agentic/open_review_v3"
DEID = REPO / "data/cases/open_review_deid"
SEED = 20261107


def _stance(score):
    return "accept" if score >= 6 else "reject"


def _paper_documents(case_id):
    abstract = (V3 / f"{case_id}__C" / "docs/paper/abstract.md").read_text(encoding="utf-8")
    fulltext = (DEID / f"{case_id}.md").read_text(encoding="utf-8")
    return [("docs/paper/abstract.md", "Paper abstract", "paper", abstract),
            ("docs/paper/fulltext.md", "Paper full text", "paper", fulltext)]


def _v3(case_id, arm):
    root = V3 / f"{case_id}__{arm}"
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    return root, manifest


def _private(manifest, design, **extra):
    keep = {key: manifest[key] for key in ("truth", "forbidden_strings") if key in manifest}
    deid = json.loads((DEID / "MANIFEST.json").read_text(encoding="utf-8"))
    return {**keep, "design": design, "paper_text": "deid", "sanitizer_sha256": deid["sanitizer_sha256"], **extra}


def _review_documents(root, assignment, transform=lambda text: text, include_post=True):
    docs = [("docs/reviews/rebuttal.md", "Author rebuttal", "rebuttal", (root / "docs/reviews/rebuttal.md").read_text(encoding="utf-8"))]
    ids = [a["reviewer_id"] for a in assignment]
    for rid in ids:
        docs.append((f"docs/reviews/reviewer_{rid}.md", f"Initial review from Reviewer {rid}", "review",
                     transform((root / f"docs/reviews/reviewer_{rid}.md").read_text(encoding="utf-8"))))
    if include_post:
        for rid in ids:
            docs.append((f"docs/reviews/post_rebuttal_{rid}.md", f"Post-rebuttal response from Reviewer {rid}", "review",
                         transform((root / f"docs/reviews/post_rebuttal_{rid}.md").read_text(encoding="utf-8"))))
    return docs


# --- or_r: v3 rerun on de-identified text -------------------------------------------------------------------------
def build_or_r(case_id, arm, output_root):
    root, manifest = _v3(case_id, arm)
    assignment = manifest["reviewer_assignment"]
    docs = _paper_documents(case_id) + _review_documents(root, assignment)
    private = _private(manifest, "REBUTTAL_OR_R_V3_RERUN", reviewer_assignment=assignment, majority_stance=manifest["majority_stance"],
                       reviews_complete=True, reviews_from=f"open_review_v3/{case_id}__{arm}", generator_model=manifest.get("generator_model"))
    return publish_bundle("or_r", case_id, arm, docs, private, output_root)


# --- or_e1: paper only ----------------------------------------------------------------------------------------------
def build_or_e1(case_id, output_root):
    _, manifest = _v3(case_id, "C")
    private = _private(manifest, "REBUTTAL_OR_E1_PAPER_ONLY", reviewer_assignment=[], majority_stance=None,
                       reviews_complete=True, no_post_rebuttal=True)
    return publish_bundle("or_e1", case_id, "P", _paper_documents(case_id), private, output_root)


# --- or_e2a: v3 packages with every numeric score withheld ----------------------------------------------------------
_SCORE_HEADER = re.compile(r"^\W*my score:\s*\d{1,2}\s*/\s*10\W*$", re.I | re.M)
_INT = r"(?:10|[0-9])(?![\d.,]\d|\d|%)"            # a review score 0-10, not a metric such as 76.18, 0.94 or 85%
_LIST = _INT + r"(?:\s*(?:,|/|and|or|–|-)\s*" + _INT + r")*"
_SCORE_PATTERNS = [
    # ranges such as "2–3/10", "(2-3/10)"
    (re.compile(r"\b\d{1,2}\s*[–-]\s*\d{1,2}\s*/\s*10\b"), "[scores withheld]"),
    # "7/10", "(8/10)", "score of 4/10"
    (re.compile(r"\b\d{1,2}\s*/\s*10\b"), "[score withheld]"),
    # tallies such as "7/7/8", "2/3/4", "8 → 3"
    (re.compile(r"\b(?:10|[0-9])(?:\s*/\s*(?:10|[0-9])){2,}\b(?!\.\d)"), "[scores withheld]"),
    (re.compile(r"\b(?:10|[0-9])\s*(?:→|->)\s*(?:10|[0-9])\b(?!\.\d)"), "[scores withheld]"),
    # "score 4", "score of 7", "scores (2, 3, 3)", "ratings 2/3/4"; parenthesised lists only when they close
    (re.compile(r"\b(scores?|ratings?|rated)(\s+(?:of|at|is|was|were|to|from|remains?|stays?|given))?\s*"
                r"(?:\(\s*" + _LIST + r"\s*(?:\[scores? withheld\]\s*)?\)|" + _LIST + r")", re.I), r"\1\2 [withheld]"),
    (re.compile(r"\b(?:10|[0-9])\s*[–-]\s*\[scores? withheld\]"), "[scores withheld]"),
]


def withhold_scores(text):
    text = _SCORE_HEADER.sub("", text, count=1).lstrip("\n")
    for pattern, replacement in _SCORE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def residual_scores(text):
    return re.findall(r"\b\d{1,2}\s*/\s*10\b|\b(?:score|rating)s?\s*(?:of\s*)?\(?(?:10|[0-9])(?![\d.,]\d|\d|%)", text, re.I)


def build_or_e2a(case_id, arm, output_root):
    root, manifest = _v3(case_id, arm)
    assignment = manifest["reviewer_assignment"]
    docs = _paper_documents(case_id) + _review_documents(root, assignment, transform=withhold_scores)
    private = _private(manifest, "REBUTTAL_OR_E2A_SCORES_WITHHELD", reviewer_assignment=assignment, majority_stance=manifest["majority_stance"],
                       reviews_complete=True, reviews_from=f"open_review_v3/{case_id}__{arm}", generator_model=manifest.get("generator_model"))
    return publish_bundle("or_e2a", case_id, arm, docs, private, output_root)


# --- or_e3c: arm X with the specific dissenter moved from slot 3 to slot 1 ------------------------------------------
_SWAP = re.compile(r"\b(Reviewer|reviewer|Reviewers?|R)(\s?#?\s?)([13])\b")


def swap_slots_1_3(text):
    return _SWAP.sub(lambda m: f"{m.group(1)}{m.group(2)}{'3' if m.group(3) == '1' else '1'}", text)


def build_or_e3c(case_id, output_root):
    root, manifest = _v3(case_id, "X")
    old = {a["reviewer_id"]: a for a in manifest["reviewer_assignment"]}
    mapping = {1: 3, 3: 1, 2: 2, 4: 4}  # new id -> old id
    assignment = [{**old[mapping[new]], "reviewer_id": new} for new in (1, 2, 3, 4)]
    docs = _paper_documents(case_id)
    docs.append(("docs/reviews/rebuttal.md", "Author rebuttal", "rebuttal", (root / "docs/reviews/rebuttal.md").read_text(encoding="utf-8")))
    for prefix, label in (("reviewer", "Initial review"), ("post_rebuttal", "Post-rebuttal response")):
        for new in (1, 2, 3, 4):
            text = (root / f"docs/reviews/{prefix}_{mapping[new]}.md").read_text(encoding="utf-8")
            docs.append((f"docs/reviews/{prefix}_{new}.md", f"{label} from Reviewer {new}", "review", swap_slots_1_3(text)))
    private = _private(manifest, "REBUTTAL_OR_E3C_MINORITY_SLOT1", reviewer_assignment=assignment, majority_stance=manifest["majority_stance"],
                       reviews_complete=True, reviews_from=f"open_review_v3/{case_id}__X", slot_mapping_new_to_old=mapping,
                       generator_model=manifest.get("generator_model"))
    return publish_bundle("or_e3c", case_id, "X", docs, private, output_root)


# --- skeletons for designs with new generated reviews ---------------------------------------------------------------
def _skeleton(scenario, case_id, arm, assignment, majority, design, output_root, **extra):
    root, manifest = _v3(case_id, "C")
    docs = _paper_documents(case_id)
    docs.append(("docs/reviews/rebuttal.md", "Author rebuttal", "rebuttal", (root / "docs/reviews/rebuttal.md").read_text(encoding="utf-8")))
    private = _private(manifest, design, reviewer_assignment=assignment, majority_stance=majority, reviews_complete=False, **extra)
    return publish_bundle(scenario, case_id, arm, docs, private, output_root)


def e2b_assignment(case_id, truth, arm):
    """2:2 count-balanced. Arm SP: the two specific reviews argue for the ICLR decision; SM: against it.
    Scores mirror around 5 (accept side 7/8, reject side 3/2); slots shuffled per case, identical in both arms."""
    specific = truth if arm == "SP" else ("reject" if truth == "accept" else "accept")
    generic = "reject" if specific == "accept" else "accept"
    scores = {"accept": [7, 8], "reject": [3, 2]}
    roles = [(specific, "specific", scores[specific][0]), (specific, "specific", scores[specific][1]),
             (generic, "generic", scores[generic][0]), (generic, "generic", scores[generic][1])]
    order = list(range(4))
    random.Random(f"{SEED}:{case_id}").shuffle(order)
    assignment = []
    for slot, index in enumerate(order, start=1):
        stance, quality, score = roles[index]
        assignment.append({"reviewer_id": slot, "group": "specific_side" if quality == "specific" else "generic_side",
                           "stance": stance, "initial_score": score, "argument_quality": quality})
    return assignment, specific


def build_or_e2b_skeleton(case_id, arm, output_root):
    _, manifest = _v3(case_id, "C")
    assignment, specific = e2b_assignment(case_id, manifest["truth"], arm)
    return _skeleton("or_e2b", case_id, arm, assignment, None, "REBUTTAL_OR_E2B_BALANCED_2V2", output_root, specific_stance=specific)


def build_or_e3a_skeleton(case_id, output_root):
    """Arm X with three reviewers (2:1): v3 reviewers 1 and 2 (generic majority) and 3 (specific minority); initial
    reviews are reused verbatim, post-rebuttal responses are regenerated because they discuss all reviewers."""
    _, manifest = _v3(case_id, "X")
    assignment = [a for a in manifest["reviewer_assignment"] if a["reviewer_id"] in (1, 2, 3)]
    return _skeleton("or_e3a", case_id, "X", assignment, manifest["majority_stance"], "REBUTTAL_OR_E3A_2V1", output_root,
                     reuse_initial_from=f"open_review_v3/{case_id}__X")


def build_or_e3b_skeleton(case_id, output_root):
    """Arm X with five reviewers (4:1): v3 reviewers 1-4 plus a new generic majority reviewer 5 (score 3 or 7)."""
    _, manifest = _v3(case_id, "X")
    assignment = list(manifest["reviewer_assignment"])
    majority = manifest["majority_stance"]
    assignment.append({"reviewer_id": 5, "group": "majority", "stance": majority,
                       "initial_score": 7 if majority == "accept" else 3, "argument_quality": "generic"})
    return _skeleton("or_e3b", case_id, "X", assignment, majority, "REBUTTAL_OR_E3B_4V1", output_root,
                     reuse_initial_from=f"open_review_v3/{case_id}__X")
