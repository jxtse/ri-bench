import re

from .common import check_injection, forbidden_values, publish_bundle


def build_court_bundle(case_json, injection, arm, *, output_root="data/agentic"):
    case_id = case_json["case_id"]
    check_injection(case_id, "court", injection, arm)
    mapping = injection.get("arm_side")
    verdict = case_json.get("ground_truth_verdict")
    winner = verdict.strip().lower() if isinstance(verdict, str) else None
    if "ground_truth_verdict" not in case_json and mapping:
        winner = mapping["A"]
    if winner not in {"petitioner", "respondent"}:
        raise ValueError("Unknown record winner")
    loser = "respondent" if winner == "petitioner" else "petitioner"
    if mapping is not None and mapping != {"A": winner, "B": loser}:
        raise ValueError("Injection target must be winner in A and loser in B")
    target_side = winner if arm == "A" else loser
    documents = [
        ("docs/case/facts.md", "Case facts", "facts", case_json["case_summary"]),
        ("docs/case/question_presented.md", "Question presented", "question", case_json["question_presented"]),
    ]
    for side, key in [("petitioner", "evidence_pro"), ("respondent", "evidence_def")]:
        excerpts = case_json[key]
        if not isinstance(excerpts, list) or not all(isinstance(text, str) for text in excerpts):
            raise ValueError("Briefs require raw excerpt lists")
        excerpts = [re.sub(r"^\s*--\s*", "", text) for text in excerpts]
        if side == target_side:
            excerpt = injection["arms"][side]
            # the generator emits stripped anchors; source bullets may carry trailing whitespace
            matches = [i for i, text in enumerate(excerpts) if text.strip() == excerpt["anchor_text"].strip()]
            if len(matches) != 1:
                raise ValueError("Injection anchor must equal exactly one excerpt bullet")
            excerpts[matches[0]] = excerpt["replacement_text"]
        text = f"Excerpts submitted by counsel for the {side}\n\n" + "\n".join("- " + excerpt for excerpt in excerpts) + "\n"
        documents.append((f"docs/briefs/{side}.md", f"Raw excerpts submitted by counsel for the {side}", "brief", text))
    return publish_bundle("court", case_id, arm, documents, {"injection": injection, "forbidden_strings": forbidden_values(case_json, "court")}, output_root)
