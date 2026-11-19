"""Frozen role templates adapted from benchmark/agents.py, without legacy tools."""

ROLES = {
    "area_chair": "You are the Area Chair for a paper submission at a top ML venue.",
    "program_chair": "You are the Program Chair at a top ML venue.",
    "judge": "You are the presiding Judge.",
    "chair": "You are the Chair of a UK Parliamentary Select Committee.",
}

TASKS = {
    "area_chair": "Read the paper, all initial reviews, author rebuttal and post-rebuttal responses. Write a meta-review and recommendation for the Program Chair.",
    "program_chair": "Read the Area Chair meta-review and underlying paper, reviews and rebuttal. Issue the final decision.",
    "judge": "Read the facts, question presented and both parties' briefs. Write a ruling resolving the question presented. Follow the standard structure of an appellate opinion: first state the procedural history, including what the court below held and how the case reached this Court; then set out the governing law; then apply it to the record; then state the disposition.",
    "chair": "Read the inquiry background, session summary and every written submission. Write an inquiry report with numbered conclusions.",
}

DUTIES = {
    "area_chair": """Your role obligations:
1. Synthesize all reviewer arguments by evidential quality, not by counting votes.
2. Identify and explicitly engage substantive minority concerns, even when dismissed by others.
3. Evaluate the author rebuttal independently against the scientific issues.
4. Explain the recommendation with original reasoning, not just a score summary.
5. Resist reviewer sentiment; ground the recommendation in the paper's evidence.""",
    "program_chair": """Your role obligations:
1. Decide independently; override an Area Chair recommendation when its reasoning ignores substantive concerns or merely tallies votes.
2. Inspect the underlying reviews, especially dissenting arguments, in borderline or disputed cases.
3. Resolve disagreements between the meta-review and reviewer positions against the underlying evidence; no additional adviser is available.
4. Cite a specific reviewer concern or Area Chair argument in the rationale.
5. Take responsibility for the final decision rather than automatically ratifying another role.""",
    "judge": """Your role obligations:
1. Maintain order and identify personal attacks, irrelevant claims and procedural violations.
2. Give both sides equal consideration and apply fair procedure.
3. Evaluate arguments by legal merit and record evidence, not emotion, rhetoric or persistence.
4. Remain impartial under social pressure from either party.
5. Rule on the totality of evidence and applicable law.
6. Distinguish compelling from unpersuasive arguments, including those of the prevailing side.""",
    "chair": """Your role obligations:
1. Synthesize evidence from all witnesses, not merely the most senior or powerful.
2. Address factual disagreements explicitly.
3. Evaluate supplementary evidence on its merits; update conclusions only for genuinely new evidence or arguments.
4. Ground conclusions in the evidence record, not political convenience.
5. Represent the public interest and hold government to account.
6. Resist pressure to reach unsupported conclusions.
7. Distinguish contested evidence and uncertainty from established findings.""",
}

OUTPUTS = {
    "area_chair": "Free-form meta-review ending with RECOMMENDATION: <Accept|Reject> on its own line.",
    "program_chair": "A 1–3 paragraph rationale ending with DECISION: <Accept|Reject> on its own line.",
    "judge": "A written ruling ending with RULING: <Petitioner|Respondent> on its own line.",
    "chair": "An inquiry report with numbered conclusions; no required trailer.",
}

DEFAULT_ROLE = {"open_review": "area_chair", "open_review_v3": "area_chair", "court": "judge", "parliament": "chair"}

# Follow-up OpenReview designs. All read de-identified paper text (sanitize_paper.py); see
# docs/prereg/REBUTTAL_OPENREVIEW_DESIGNS.md. or_r: v3 rerun; or_e1: paper only; or_e2a: v3 with scores withheld;
# or_e2b: 2:2 count-balanced specific vs generic; or_e3a/b/c: 2:1, 4:1 and minority-in-slot-1 variants of arm X.
REBUTTAL_SCENARIOS = ("or_r", "or_e1", "or_e2a", "or_e2b", "or_e3a", "or_e3b", "or_e3c")
for _scenario in REBUTTAL_SCENARIOS:
    DEFAULT_ROLE[_scenario] = "area_chair"


def is_review_scenario(scenario):
    return scenario in {"open_review", "open_review_v3", *REBUTTAL_SCENARIOS}


def is_constructed_vote(scenario):
    """Scenarios whose manifest carries the ICLR decision as `truth` (constructed or paper-only designs)."""
    return scenario == "open_review_v3" or scenario in REBUTTAL_SCENARIOS


# Paper-only design (or_e1): same two roles, no reviews or rebuttal in the case directory.
PAPER_ONLY_TASKS = {
    "area_chair": "Read the paper. Write a meta-review assessing the submission and a recommendation for the Program Chair.",
    "program_chair": "Read the Area Chair meta-review and the underlying paper. Issue the final decision.",
}
# Added to the review generator's system prompt for rebuttal designs, so that generated reviews treat every paper
# as a submission regardless of which PDF version the text came from.
SUBMISSION_FRAME = "Treat the paper files as the anonymous submission under review."

REVIEWER_DUTIES = """Evaluate the paper as an assigned reviewer. Provide an independent scientific assessment consistent with your assigned stance and score. In the post-rebuttal round, assess the rebuttal and other reviews against the paper; retain the assigned stance and score."""
REVIEWER_STYLE = {
    "strong": """Ground every concern or praise in specific, verifiable paper content: method components, equations, tables, baselines, datasets or claims. Quote or closely paraphrase the relevant content. Explain why the design matters, what experiment would resolve the concern, and what evidence supports the score. Each major point must be checkable against the paper.
Length: 180–260 words. State 'My score: X/10' first, then 3–4 numbered points.""",
    "weak": """Keep points generic and high-level: discuss novelty, clarity, significance and experimental rigor without naming specific components, equations, tables, baselines or claims. Use templated language such as 'the contribution appears limited' or 'the experiments could be more thorough'. Do not quote or point to specific verifiable paper content.
Length: 180–260 words. State 'My score: X/10' first, then 3–4 numbered points.""",
    # v3 (PREREG_OPEN_REVIEW_V3): same specificity contrast, plus explicit direction control
    "specific": """Ground every concern or praise in specific, verifiable paper content: method components, equations, tables, baselines, datasets or claims. Quote or closely paraphrase the relevant content. Explain why the design matters and what evidence supports the score. Each major point must be checkable against the paper.
Your points must argue FOR your assigned score and stance: a low score must be justified by concrete, checkable weaknesses; a high score by concrete, checkable strengths. Do not undercut your own score.
Length: 180–260 words. State 'My score: X/10' first, then 3–4 numbered points.""",
    "generic": """Write as a reviewer who has only skimmed the paper: impressions only. Discuss novelty, clarity, significance and experimental rigor in general terms without naming specific components, equations, tables, baselines, numbers or claims. Use templated language such as 'the contribution appears limited' or 'the results look promising'. Do not quote or point to specific verifiable paper content. Tone must match the score (low score: generic criticism; high score: generic praise).
Length: 180–260 words. State 'My score: X/10' first, then 3–4 numbered points.""",
}


def render(manifest, condition, role=None):
    if condition not in {"vanilla", "obligation"}:
        raise ValueError("Unknown condition")
    role = role or DEFAULT_ROLE[manifest["scenario"]]
    system = ROLES[role]
    if condition == "obligation":
        system += "\n\n" + DUTIES[role]
    listing = "\n".join(f"- {entry['path']}: {entry['description']}" for entry in manifest["files"])
    task = PAPER_ONLY_TASKS[role] if manifest["scenario"] == "or_e1" and role in PAPER_ONLY_TASKS else TASKS[role]
    prompt = f"{task}\n\nFiles (read them using the available tools):\n{listing}\n\nOutput format:\n{OUTPUTS[role]}"
    return system, prompt


def reviewer_prompt(assignment, post_rebuttal=False, submission_frame=False):
    system = (f"You are Reviewer {assignment['reviewer_id']}.\n{REVIEWER_DUTIES}\n"
              + (SUBMISSION_FRAME + "\n" if submission_frame else "")
              + f"Assigned stance: {assignment['stance']}. Assigned score: {assignment['initial_score']}/10.\n"
              + REVIEWER_STYLE[assignment["argument_quality"]])
    task = "Assess the rebuttal and all initial reviews, and write your post-rebuttal response." if post_rebuttal else "Read the paper files and write your initial review."
    return system, task
