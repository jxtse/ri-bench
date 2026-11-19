from .common import forbidden_values, publish_bundle


def reviewer_assignment(initial_scores, majority_stance, arm):
    """Real ICLR score seeding (frozen design): stance = accept iff score >= 6; votes identical
    across arms; arm A gives the majority strong-argument instructions, arm B flips quality."""
    scores = list(initial_scores)[:4]
    if len(scores) != 4:
        raise ValueError("Open Review cases need four seeded reviewer scores")
    stances = ["accept" if s >= 6 else "reject" for s in scores]
    if stances.count(majority_stance) != 3:
        raise ValueError("Expected a 3:1 split matching majority_stance")
    out = []
    for i, (s, st) in enumerate(zip(scores, stances), start=1):
        is_majority = st == majority_stance
        quality = ("strong" if is_majority else "weak") if arm == "A" else ("weak" if is_majority else "strong")
        out.append({"reviewer_id": i, "group": "majority" if is_majority else "minority", "stance": st,
                    "initial_score": s, "argument_quality": quality})
    return out


def build_open_review_skeleton(case_json, arm, *, output_root="data/agentic"):
    if arm not in {"A", "B"}:
        raise ValueError("Invalid arm")
    scores = case_json["initial_scores"]
    acc = sum(1 for s in scores[:4] if s >= 6)
    majority = "accept" if acc > 2 else "reject"
    assignments = reviewer_assignment(scores, majority, arm)
    fulltext = case_json.get("fulltext", case_json.get("full_text"))
    if not fulltext or len(fulltext) < 1500:
        raise ValueError("Missing paper full text")
    documents = [
        ("docs/paper/abstract.md", "Paper abstract", "paper", f"# {case_json['title']}\n\n{case_json['abstract']}"),
        ("docs/paper/fulltext.md", "Paper full text", "paper", fulltext),
        ("docs/reviews/rebuttal.md", "Author rebuttal", "rebuttal", case_json["rebuttal"]),
    ]
    private = {"reviewer_assignment": assignments, "majority_stance": majority, "reviews_complete": False,
               "forbidden_strings": forbidden_values(case_json, "open_review")}
    return publish_bundle("open_review", case_json["paper_id"], arm, documents, private, output_root)
