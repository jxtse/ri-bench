"""Open Review v3: constructed adversarial votes (docs/prereg/PREREG_OPEN_REVIEW_V3_ADVERSARIAL_VOTES.md).

Votes are constructed, not taken from ICLR. Arm C (control): the three-reviewer majority points in the direction of the
paper's real outcome and argues specifically; the minority (always reviewer 3) points the other way and is generic.
Arm X (adversarial): the majority points against the real outcome with generic, skim-level reviews; the minority points
toward the real outcome and argues specifically for its score. `truth` (the ICLR decision) is private manifest data.
"""
from .common import forbidden_values, publish_bundle

SCORES = {"majority_correct": {"good": [7, 7, 8], "bad": [3, 3, 2]}, "minority_correct": {"good": 8, "bad": 2},
          "majority_wrong": {"good": [3, 3, 2], "bad": [7, 7, 8]}, "minority_wrong": {"good": 4, "bad": 6}}
MINORITY_SLOT = 3  # reviewer_id of the lone dissenter in both arms


def _stance(score):
    return "accept" if score >= 6 else "reject"


def reviewer_assignment_v3(truth, arm):
    if truth not in {"accept", "reject"}:
        raise ValueError("truth must be accept or reject")
    if arm not in {"C", "X"}:
        raise ValueError("Invalid arm")
    key = "good" if truth == "accept" else "bad"
    if arm == "C":
        majority, minority = SCORES["majority_correct"][key], SCORES["minority_wrong"][key]
        maj_q, min_q = "specific", "generic"
    else:
        majority, minority = SCORES["majority_wrong"][key], SCORES["minority_correct"][key]
        maj_q, min_q = "generic", "specific"
    out, it = [], iter(majority)
    for rid in range(1, 5):
        if rid == MINORITY_SLOT:
            score, group, quality = minority, "minority", min_q
        else:
            score, group, quality = next(it), "majority", maj_q
        out.append({"reviewer_id": rid, "group": group, "stance": _stance(score), "initial_score": score, "argument_quality": quality})
    stances = {a["stance"] for a in out if a["group"] == "majority"}
    if len(stances) != 1 or out[MINORITY_SLOT - 1]["stance"] in stances:
        raise ValueError("Constructed votes must be a 3:1 split")
    expected_majority = truth if arm == "C" else ("reject" if truth == "accept" else "accept")
    if stances != {expected_majority}:
        raise ValueError("Majority direction inconsistent with arm")
    return out


def build_open_review_v3_skeleton(case_json, arm, *, truth, output_root="data/agentic"):
    assignments = reviewer_assignment_v3(truth, arm)
    majority = next(a["stance"] for a in assignments if a["group"] == "majority")
    fulltext = case_json.get("fulltext", case_json.get("full_text"))
    if not fulltext or len(fulltext) < 1500:
        raise ValueError("Missing paper full text")
    documents = [
        ("docs/paper/abstract.md", "Paper abstract", "paper", f"# {case_json['title']}\n\n{case_json['abstract']}"),
        ("docs/paper/fulltext.md", "Paper full text", "paper", fulltext),
        ("docs/reviews/rebuttal.md", "Author rebuttal", "rebuttal", case_json["rebuttal"]),
    ]
    private = {"reviewer_assignment": assignments, "majority_stance": majority, "truth": truth, "reviews_complete": False,
               "design": "PREREG_OPEN_REVIEW_V3_ADVERSARIAL_VOTES", "forbidden_strings": forbidden_values(case_json, "open_review")}
    return publish_bundle("open_review_v3", case_json["paper_id"], arm, documents, private, output_root)
