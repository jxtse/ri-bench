import re

from .common import check_injection, forbidden_values, publish_bundle


def build_parliament_bundle(case_json, injection, arm, *, output_root="data/agentic"):
    case_id = case_json.get("inquiry_id", case_json.get("case_id"))
    check_injection(case_id, "parliament", injection, arm)
    submissions = case_json["written_submissions"]
    replacements = {}
    witness_indices = set()
    for witness in injection["witnesses"].values():
        witness_index = witness["witness_index"]
        if witness_index >= len(case_json["witness_profiles"]) or witness_index in witness_indices:
            raise ValueError("Witness index out of range or repeated")
        witness_indices.add(witness_index)
        if "submission_id" in witness:
            targets = [index for index, submission in enumerate(submissions)
                       if str(submission.get("submission_id")) == str(witness["submission_id"])]
        else:
            name = " ".join(witness["submitter"].split()).casefold()
            targets = [index for index, submission in enumerate(submissions)
                       if " ".join(submission["submitter"].split()).casefold() == name]
        if len(targets) != 1 or targets[0] in replacements:
            raise ValueError("Target submission must be found exactly once for each witness")
        qualifiers = witness["rules"]["qualifier_patterns"] + witness["rules"]["hedge_patterns"]
        if not any(re.search(pattern, witness["qualified_text"], re.I) for pattern in qualifiers):
            raise ValueError("Qualified text must preserve a qualifier pattern")
        text = submissions[targets[0]]["text"]
        anchor = witness["anchor_text"].strip()
        pattern = r"\s+".join(re.escape(word) for word in anchor.split())
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError("Injection anchor must occur exactly once after whitespace normalization")
        start, end = matches[0].span()
        preceding = text[:start].rstrip()
        following = text[end:]
        if not anchor.endswith((".", "!", "?")) or (preceding and preceding[-1] not in ".!?") or (following and not following[0].isspace()):
            raise ValueError("Injection anchor must be a complete sentence")
        replacements[targets[0]] = text[:start] + witness["qualified_text"] + text[end:]
    documents = [
        ("docs/inquiry/background.md", "Inquiry background", "background", case_json["inquiry_background"]),
        ("docs/inquiry/session_summary.md", "Session summary", "session", case_json["session_summary"]),
    ]
    for index, submission in enumerate(submissions):
        name = submission["submitter"]
        affiliation = submission.get("affiliation", "affiliation not supplied")
        description = f"Written evidence submitted by {name}, {affiliation}"
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "witness"
        text = replacements.get(index, submission["text"])
        documents.append((f"docs/submissions/{index + 1:03d}-{slug}.md", description, "submission", description + "\n\n" + text))
    return publish_bundle("parliament", case_id, arm, documents, {"injection": injection, "forbidden_strings": forbidden_values(case_json, "parliament")}, output_root)
