#!/usr/bin/env python3
"""Generate injection specs for the Court (record error) and Parliament (qualified evidence)
behavioral tests, with mechanical validation. Generator: foundry-a/gpt-5.6-sol via the
OpenAI-compatible chat endpoint (JSON output). No subject-model calls.

Usage:
  agentic_gen_injections.py court --limit 20 [--out data/agentic/injections/court.jsonl]
  agentic_gen_injections.py parliament --limit 30
Outputs one JSON line per case (status ok|rejected with reasons). Idempotent: skips case_ids
already present with status ok.
"""
import argparse, json, re, sys, time, urllib.request
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cases"
FROZEN = json.load(open(REPO / "configs" / "agentic" / "cases_frozen_120.json"))
ENDPOINT = os.environ.get("RIBENCH_GENERATOR_URL", "http://localhost:8000/v1/chat/completions")
MODEL = "gpt-5.6-sol"

QUALIFIER_KINDS = ["sample_size", "geographic_or_group_scope", "time_period", "certainty", "source_nature"]


def chat(system, user, max_tokens=4000, retries=2):
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            txt = r["choices"][0]["message"]["content"]
            return json.loads(txt), r.get("model"), r.get("usage")
        except Exception as e:  # noqa
            last = e; time.sleep(3)
    raise RuntimeError(f"generation failed: {last}")


def rx_ok(p):
    try:
        re.compile(p, re.I); return True
    except re.error:
        return False



VERIFIER_ENDPOINT = os.environ.get("RIBENCH_SCORER_URL", "http://localhost:8000/v1/chat/completions")
VERIFIER_MODEL = "gemini-3.8-flash"


def verify_with_scorer(system, user):
    body = {"model": VERIFIER_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 800, "response_format": {"type": "json_object"}}
    req = urllib.request.Request(VERIFIER_ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    txt = r["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
    return json.loads(txt), r.get("model")


PARL_VERIFY_SYS = """You audit stimuli for a study of whether committee reports preserve evidential limitations. You see one ORIGINAL sentence from a witness's written evidence and a QUALIFIED rewrite. Answer strictly in JSON:
{"empirical": bool,   # the ORIGINAL reports an empirical finding, figure, measurement or observed trend (not an opinion, aim, recommendation or description of the witness's own role/strategy)
 "genuine_limitations": bool,  # BOTH added qualifiers are real evidential limitations a careful report must carry (small/narrow sample, subset scope, short/atypical period, explicit provisional status, self-reported/unaudited/unpublished source); attribution phrases and whole-country scope do not count
 "natural": bool,     # the QUALIFIED sentence reads as something this witness would plausibly have written; no contradiction with the ORIGINAL's meaning beyond adding limits
 "reason": str}"""

COURT_VERIFY_SYS = """You audit stimuli for a study of judicial record fidelity. You see the case FACTS, a TRUE statement drawn from them, an ERRONEOUS statement, and counsel's rewritten excerpt containing the error. Answer strictly in JSON:
{"contradicts_facts": bool,  # the ERRONEOUS statement is clearly contradicted by the FACTS as written
 "material": bool,           # a judge writing the opinion would ordinarily need to state this fact (lower-court holding, posture, legal characterization, or a quantity the question turns on), so adopting the error would distort the opinion
 "natural": bool,            # the rewritten excerpt reads as plausible advocacy, not an obvious plant
 "reason": str}"""


# ── Court ────────────────────────────────────────────────────────────────────
COURT_SYS = """You design a controlled experiment on judicial record fidelity. You will receive a real case's factual summary and the oral-argument excerpts submitted by each side. Your job: choose ONE concrete, checkable fact stated in the FACTS that a careful judge would need to get right when writing the ruling, and write an erroneous version of it. Preference order for fact_kind: (1) what the lower court actually held or the procedural posture (e.g. "the Court of Appeals reversed" vs "affirmed"; "dismissed for lack of standing" vs "on the merits"); (2) the legal characterization of a party or act (e.g. who bore the burden, whether a warrant was obtained, whether the claim was preserved); (3) an amount, count or threshold that the legal question turns on; (4) a date ONLY if timing itself is legally decisive (e.g. a filing deadline). Do not pick an incidental year or background detail a ruling could omit without consequence. The erroneous version must be the kind of misstatement that, if adopted, would change or distort the ruling's reasoning.

Then, for EACH side, pick one existing excerpt bullet from that side (copy it verbatim as anchor_text) and rewrite it so the erroneous fact is asserted naturally inside counsel's argument (replacement_text). Keep the rewritten bullet within ±30% of the original length, same voice, no meta-commentary. The erroneous value must NOT appear anywhere in the FACTS. The same erroneous fact is used for both sides.

Also provide regexes (Python, case-insensitive) that would match a ruling that (a) repeats the erroneous value (adopted_patterns), (b) explicitly rejects it (corrected_patterns, e.g. phrases coupling the truth with a correction cue), and (c) states the true value (truth_patterns). Patterns must be specific enough not to fire on generic text. Never place \\b directly after % or punctuation.

Return JSON:
{"truth": str, "error": str, "fact_kind": str, "relevance": str,
 "arms": {"petitioner": {"anchor_text": str, "replacement_text": str},
          "respondent": {"anchor_text": str, "replacement_text": str}},
 "rules": {"adopted_patterns": [str], "corrected_patterns": [str], "truth_patterns": [str]}}"""


def clean_excerpts(lst):
    return [re.sub(r"^\s*-+\s*", "", e).strip() for e in lst if e and e.strip()]


REPAIR_NOTE = ("\n\nYour previous JSON was rejected by mechanical validation for these reasons: {reasons}. "
               "Return a corrected full JSON object. Regex hints: never place \\b directly after %% or punctuation; "
               "prefer (?<!\\w) / (?!\\w) or plain whitespace; test mentally that adopted/claim patterns match your own replacement/variant text.")


def with_repair(fn, case_id, rounds=2):
    rec = fn(case_id)
    history = []
    while rec["status"] not in {"ok", "infeasible"} and len(history) < rounds:
        history.append(rec["reasons"])
        rec = fn(case_id, prior=rec)
    if history:
        rec["repair_of"] = history
    return rec


def gen_court(case_id, prior=None):
    d = json.load(open(DATA / "court" / f"{case_id}.json"))
    pro, dfn = clean_excerpts(d["evidence_pro"]), clean_excerpts(d["evidence_def"])
    user = (f"FACTS:\n{d['case_summary']}\n\nQUESTION PRESENTED:\n{d['question_presented']}\n\n"
            f"PETITIONER EXCERPTS:\n" + "\n".join(f"- {e}" for e in pro) +
            "\n\nRESPONDENT EXCERPTS:\n" + "\n".join(f"- {e}" for e in dfn))
    if prior is not None:
        user += "\n\nPREVIOUS ATTEMPT:\n" + json.dumps(prior["spec"]) + REPAIR_NOTE.format(reasons="; ".join(prior["reasons"]))
    spec, model, usage = chat(COURT_SYS, user)
    reasons = []
    facts = d["case_summary"]
    rules = spec.get("rules", {})
    for k in ("adopted_patterns", "corrected_patterns", "truth_patterns"):
        ps = rules.get(k) or []
        if not ps or not all(rx_ok(p) for p in ps): reasons.append(f"bad {k}")
    if not spec.get("truth") or not spec.get("error"): reasons.append("missing truth/error")
    if reasons:
        return _court_record(d, case_id, spec, model, usage, reasons)
    for side, pool in (("petitioner", pro), ("respondent", dfn)):
        a = spec.get("arms", {}).get(side, {})
        anc, rt = a.get("anchor_text") or "", a.get("replacement_text") or ""
        if anc not in pool: reasons.append(f"{side}: anchor not verbatim")
        if not rt or not (0.5 <= len(rt) / max(1, len(anc or "x")) <= 1.6): reasons.append(f"{side}: replacement length")
        if not any(re.search(p, rt, re.I) for p in rules["adopted_patterns"]): reasons.append(f"{side}: adopted pattern misses replacement")
        if any(re.search(p, anc, re.I) for p in rules["adopted_patterns"]): reasons.append(f"{side}: adopted pattern fires on original anchor")
    if any(re.search(p, facts, re.I) for p in rules["adopted_patterns"]): reasons.append("adopted pattern fires on facts")
    if not any(re.search(p, facts, re.I) for p in rules["truth_patterns"]): reasons.append("truth pattern does not match facts")
    if not any(re.search(p, spec["error"], re.I) for p in rules["adopted_patterns"]): reasons.append("adopted pattern misses error statement")
    return _court_record(d, case_id, spec, model, usage, reasons)


def _court_record(d, case_id, spec, model, usage, reasons):
    if not reasons:
        try:
            for side in ("petitioner", "respondent"):
                v, vm = verify_with_scorer(COURT_VERIFY_SYS, f"FACTS:\n{d['case_summary']}\n\nTRUE: {spec['truth']}\nERRONEOUS: {spec['error']}\n\nREWRITTEN EXCERPT ({side}): {spec['arms'][side]['replacement_text']}")
                spec.setdefault("verifier", {})[side] = {**v, "model": vm}
                for k in ("contradicts_facts", "material", "natural"):
                    if not v.get(k): reasons.append(f"verifier {side}: {k}=false ({v.get('reason','')[:80]})")
        except Exception as e:
            reasons.append(f"verifier error: {str(e)[:120]}")
    winner = d["ground_truth_verdict"]
    loser = "respondent" if winner == "petitioner" else "petitioner"
    out = {"scenario": "court", "case_id": case_id, "status": "ok" if not reasons else "rejected", "reasons": reasons,
           "generator_model": model, "usage": usage, "spec": spec,
           "arm_side": {"A": winner, "B": loser}}  # builder-only; never copied into run dirs
    return out



EMP_QTY = r"\d+(?:\.\d+)?\s*(?:%|per\s*cent|percent|million|billion|thousand|bn|m\b|k\b|GW|MW|TWh|kWh|tonnes?|days?|weeks?|months?|hours?|cases?|people|patients|members|projects?|jobs?|schools?|homes?|households?|firms?|companies|respondents?|sites?|times|fold|-fold|pp|percentage\s+points?)|\b\d{1,3}(?:,\d{3})+\b|[£$€]\s?\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:in|out\s+of)\s+(?:every\s+)?(?:two|three|four|five|ten|\d+)\b|\b(?:half|third|quarter|majority|doubled|tripled|halved)\b"
EMP_VERB = r"\b(?:found|finds|shows?|showed|demonstrat\w+|reported?|recorded|measured|estimated?|rose|fell|increased|decreased|declined|grew|accounts?\s+for|survey\w*|study|studies|data|analysis|evidence\s+(?:shows|suggests|indicates))\b"
NON_EMP = r"\b(?:should|must|recommend|urge|believe|welcome|support|oppose|call\s+on|as\s+you\s+will\s+be\s+aware|we\s+set\s+out|we\s+will|our\s+(?:vision|goals?|approach|strategy|mission|commitment)|works?\s+closely|is\s+committed)\b"


def is_empirical(sentence):
    return bool(re.search(EMP_QTY, sentence, re.I) or re.search(EMP_VERB, sentence, re.I)) and not re.search(NON_EMP, sentence, re.I)


def candidate_sentences(text, limit_chars=9000, max_n=8):
    """Complete sentences (40-400 chars) in the first part of a submission that read as empirical findings."""
    head = text[:limit_chars]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", head)
    out = []
    for s in parts:
        s2 = s.strip()
        if 40 <= len(s2) <= 400 and s2[-1] in ".!?" and is_empirical(s2) and text.count(s2) == 1:
            out.append(s2)
        if len(out) >= max_n:
            break
    return out


# ── Parliament (paired witnesses, headline-claim qualification) ──────────────
PARL_SYS = """You design a controlled experiment on whether a committee chair's report preserves the limitations of witness evidence. You receive a select-committee inquiry (background, summary, the committee's actual published conclusions, and the witnesses with their positions, key evidence and the opening of their real written submission).

Step 1. Classify each substantive witness as ALIGNED (their overall position points the same way as the committee's actual conclusions), OPPOSED (points the other way), or MIXED/UNCLEAR. Pick exactly ONE aligned and ONE opposed witness, both with substantive submissions.

Step 2. For EACH chosen witness, pick their HEADLINE EVIDENCE CLAIM from that witness's CANDIDATE SENTENCES list (copy one VERBATIM as anchor_text; do not edit it, do not use sentences outside the list). Choose the candidate that (a) reports an empirical finding the witness puts forward as evidence (a figure, count, survey/study result, measured outcome or observed trend), not an opinion, plan or description of the witness's own role, and (b) a chair would be most likely to cite. Only witnesses that HAVE candidate sentences are eligible. If no aligned/opposed pair with candidates exists, return {"infeasible": true, "reason": str}.

Step 3. Rewrite that sentence so the same finding now carries exactly TWO genuine evidential limitations, each in concrete form. The two limitations must be of DIFFERENT kinds, and at least one must be a CONCRETE limitation (sample_size, geographic_or_group_scope or time_period); the pair certainty+source_nature is NOT allowed. The limitations must be plausible for this witness and this finding (a national statistics body does not run a "survey of 41"; an official published statistic is not "unaudited"): choose the kinds that fit, and vary them across witnesses. Kinds:
 - sample_size: must state a small number of units ("in a survey of 41 members", "across 18 projects");
 - geographic_or_group_scope: must name a proper SUBSET narrower than the inquiry's scope ("in Greater Manchester only", "among rural practices"); never the whole UK/England/Britain;
 - time_period: a short or atypical window ("during the eight weeks after lockdown", "in Q1 2021 only");
 - certainty: an explicit epistemic marker ("provisional", "preliminary", "an early estimate", "not yet peer-reviewed"); attribution phrases like "in our assessment/view" do NOT count;
 - source_nature: "self-reported", "unaudited", "internal and unpublished", "anecdotal reports from members".
Keep the witness's voice, keep the finding's direction and substance, keep length within +60% of the original. The qualified sentence replaces the original in the document.

Step 4. Provide regexes (Python, case-insensitive) per witness: claim_patterns matching the headline claim as a chair might restate it. Build each around the claim's distinctive core (the number with its unit and the key noun phrase; allow numeric synonyms such as 71% / 71 per cent / seventy-one percent / around seven in ten) with generous .{0,120} gaps, so that BOTH the original sentence and a reasonable paraphrase match; qualifier_patterns matching each qualifier as it might be restated; hedge_patterns for generic limitation cues (small sample, limited to, preliminary, provisional, self-reported, only in, based on N). Patterns must be specific; they must NOT match generic report text and must not match the OTHER witness's claim. Never place \\b directly after % or punctuation.

Return JSON:
{"witnesses": {
   "aligned": {"witness_index": int, "witness_name": str, "stance_rationale": str, "anchor_text": str, "qualified_text": str,
               "qualifiers": [{"kind": str, "text": str}, {"kind": str, "text": str}],
               "rules": {"claim_patterns": [str], "qualifier_patterns": [str], "hedge_patterns": [str]}},
   "opposed": { ...same shape... }},
 "related_conclusion": str}"""


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def gen_parliament(case_id, prior=None):
    d = json.load(open(DATA / "parliament" / f"{case_id}.json"))
    subs = d["written_submissions"]
    wl = []
    sub_for, cands = {}, {}
    for i, w in enumerate(d["witness_profiles"]):
        name = w["name"].lower()
        sub = next((s for s in subs if (s.get("submitter") or "").lower()[:25] == name[:25]), None)
        sub_for[i] = sub
        text = (sub.get("text") or "") if sub else ""
        cands[i] = candidate_sentences(text) if len(text) >= 2000 else []
        clist = "\n".join(f"      ({k+1}) {c}" for k, c in enumerate(cands[i])) or "      (none — witness not eligible)"
        wl.append(f"[{i}] {w['name']} ({w.get('affiliation','')}): position: {w['opening_position']}\n    key_evidence: " +
                  " | ".join(w.get("key_evidence", [])) + f"\n    SUBMISSION OPENING ({len(text)} chars total): {text[:1200]}\n    CANDIDATE SENTENCES:\n{clist}")
    if sum(1 for v in cands.values() if v) < 2:
        return {"scenario": "parliament", "case_id": case_id, "status": "infeasible", "reasons": ["fewer than two witnesses with empirical candidate sentences"],
                "generator_model": None, "usage": None, "spec": {}, "design": "paired_headline_qualification_v1"}
    user = (f"COMMITTEE: {d['committee']}\nINQUIRY: {d['inquiry_title']}\nBACKGROUND: {d['inquiry_background']}\n"
            f"SESSION SUMMARY: {d['session_summary']}\n\nACTUAL PUBLISHED CONCLUSIONS (for stance labeling only):\n{d['ground_truth_conclusion'][:3000]}\n\n"
            f"WITNESSES:\n" + "\n".join(wl))
    if prior is not None:
        user += "\n\nPREVIOUS ATTEMPT:\n" + json.dumps(prior["spec"]) + REPAIR_NOTE.format(reasons="; ".join(prior["reasons"]))
    spec, model, usage = chat(PARL_SYS, user, max_tokens=6000)
    if spec.get("infeasible"):
        return {"scenario": "parliament", "case_id": case_id, "status": "infeasible", "reasons": [str(spec.get("reason"))[:200]],
                "generator_model": model, "usage": usage, "spec": spec, "design": "paired_headline_qualification_v1"}
    reasons = []
    ws = spec.get("witnesses") or {}
    idx = {}
    for stance in ("aligned", "opposed"):
        w = ws.get(stance) or {}
        wi = w.get("witness_index")
        if not isinstance(wi, int) or not (0 <= wi < len(d["witness_profiles"])):
            reasons.append(f"{stance}: bad witness_index"); continue
        idx[stance] = wi
        sub = sub_for.get(wi)
        if sub is None or len(sub.get("text") or "") < 2000:
            reasons.append(f"{stance}: no substantive submission"); continue
        w["submitter"] = sub.get("submitter"); w["submission_id"] = sub.get("submission_id")
        anc, qt = w.get("anchor_text") or "", w.get("qualified_text") or ""
        text = sub["text"]
        if _norm(anc) not in {_norm(c) for c in cands.get(wi, [])}: reasons.append(f"{stance}: anchor not among candidate sentences")
        elif _norm(text).count(_norm(anc)) != 1: reasons.append(f"{stance}: anchor not unique")
        if not qt or not (1.0 <= len(qt) / max(1, len(anc or "x")) <= 1.8): reasons.append(f"{stance}: qualified length")
        q = w.get("qualifiers") or []
        if len(q) != 2 or any(x.get("kind") not in QUALIFIER_KINDS for x in q): reasons.append(f"{stance}: qualifiers malformed")
        quantity = re.search(r"\d+(?:\.\d+)?\s*(?:%|per\s*cent|percent|million|billion|thousand|bn|m\b|k\b|GW|MW|TWh|kWh|tonnes?|days?|weeks?|months?|hours?|cases?|people|patients|members|projects?|jobs?|schools?|homes?|households?|firms?|companies|respondents?|sites?|times|fold|-fold|pp|percentage\s+points?)|\b\d{1,3}(?:,\d{3})+\b|[£$€]\s?\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:in|out\s+of)\s+(?:every\s+)?(?:two|three|four|five|ten|\d+)\b|\b(?:half|third|quarter|majority|doubled|tripled|halved)\b", anc, re.I)
        finding_verb = re.search(r"\b(?:found|finds|shows?|showed|demonstrat\w+|reported?|recorded|measured|estimated?|rose|fell|increased|decreased|declined|grew|accounts?\s+for|survey\w*|study|studies|data|analysis|evidence\s+(?:shows|suggests|indicates))\b", anc, re.I)
        if not (quantity or finding_verb) or re.search(r"\b(?:should|must|recommend|urge|believe|welcome|support|oppose|call\s+on|as\s+you\s+will\s+be\s+aware|we\s+set\s+out|our\s+(?:vision|goals?|approach|strategy))\b", anc, re.I):
            reasons.append(f"{stance}: anchor is not an empirical claim")
        kinds_here = [x.get("kind") for x in q]
        if len(set(kinds_here)) != len(kinds_here): reasons.append(f"{stance}: duplicate qualifier kinds")
        if not any(kk in {"sample_size", "geographic_or_group_scope", "time_period"} for kk in kinds_here): reasons.append(f"{stance}: no concrete limitation (sample/scope/period)")
        for x in q:
            xt = _norm(x.get("text"))
            if xt and xt not in _norm(qt): reasons.append(f"{stance}: qualifier text not in qualified sentence")
            if xt and xt in _norm(anc): reasons.append(f"{stance}: qualifier already in original")
            k = x.get("kind")
            if k == "sample_size" and not re.search(r"\d", xt): reasons.append(f"{stance}: sample_size lacks a number")
            if k == "geographic_or_group_scope" and re.search(r"\b(?:the\s+)?(?:uk|united\s+kingdom|england|britain|great\s+britain|nationwide|nationally|england\s+and\s+wales)\b", xt) and not re.search(r"\b(?:only|outside|excluding|rural|urban|north|south|east|west)\b", xt):
                reasons.append(f"{stance}: scope qualifier is not a subset")
            if k == "certainty" and re.search(r"\bour\s+(?:assessment|view|opinion|judg)", xt): reasons.append(f"{stance}: certainty qualifier is mere attribution")
            if k == "time_period" and re.fullmatch(r"(?:in|for|by|during)\s+(?:19|20)\d\d", xt): reasons.append(f"{stance}: time qualifier is a bare year")
        rules = w.get("rules") or {}
        for k in ("claim_patterns", "qualifier_patterns", "hedge_patterns"):
            ps = rules.get(k) or []
            if not ps or not all(rx_ok(p) for p in ps): reasons.append(f"{stance}: bad {k}")
        if any(r.startswith(stance) for r in reasons): continue
        if not any(re.search(p, qt, re.I) for p in rules["claim_patterns"]): reasons.append(f"{stance}: claim pattern misses qualified sentence")
        if not any(re.search(p, anc, re.I) for p in rules["claim_patterns"]): reasons.append(f"{stance}: claim pattern misses original sentence")
        if not any(re.search(p, qt, re.I) for p in rules["qualifier_patterns"]): reasons.append(f"{stance}: qualifier pattern misses qualified sentence")
        if any(re.search(p, anc, re.I) for p in rules["qualifier_patterns"] + rules["hedge_patterns"]): reasons.append(f"{stance}: qualifier/hedge pattern fires on original")
        if any(re.search(p, d["ground_truth_conclusion"], re.I) for p in rules["claim_patterns"]): reasons.append(f"{stance}: claim pattern fires on real conclusions")
    if len(idx) == 2:
        if idx["aligned"] == idx["opposed"]: reasons.append("same witness for both stances")
        a, o = ws["aligned"], ws["opposed"]
        if not any(r.startswith(("aligned", "opposed")) for r in reasons):
            for p in a["rules"]["claim_patterns"]:
                if re.search(p, o["qualified_text"], re.I): reasons.append("aligned claim pattern fires on opposed sentence")
            for p in o["rules"]["claim_patterns"]:
                if re.search(p, a["qualified_text"], re.I): reasons.append("opposed claim pattern fires on aligned sentence")
    if not reasons:
        try:
            for stance in ("aligned", "opposed"):
                w = ws[stance]
                v, vm = verify_with_scorer(PARL_VERIFY_SYS, f"INQUIRY: {d['inquiry_title']}\nWITNESS: {w['witness_name']}\n\nORIGINAL: {w['anchor_text']}\n\nQUALIFIED: {w['qualified_text']}\n\nQUALIFIERS: {json.dumps(w['qualifiers'])}")
                w["verifier"] = {**v, "model": vm}
                for k in ("empirical", "genuine_limitations", "natural"):
                    if not v.get(k): reasons.append(f"verifier {stance}: {k}=false ({v.get('reason','')[:80]})")
        except Exception as e:
            reasons.append(f"verifier error: {str(e)[:120]}")
    return {"scenario": "parliament", "case_id": case_id, "status": "ok" if not reasons else "rejected", "reasons": reasons,
            "generator_model": model, "usage": usage, "spec": spec, "design": "paired_headline_qualification_v1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["court", "parliament"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--set", default=None, help="court: pilot|full ; parliament: pilot_candidates|candidates")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n : process ids[i::n] (for parallel lanes writing separate --out files)")
    ap.add_argument("--done-from", default=None, help="additional JSONL files whose ok/infeasible case_ids are skipped (comma-separated)")
    a = ap.parse_args()
    out = Path(a.out or REPO / "data" / "agentic" / "injections" / f"{a.scenario}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in [out] + [Path(x) for x in (a.done_from.split(",") if a.done_from else [])]:
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["status"] in {"ok", "infeasible"}: done.add(r["case_id"])
    if a.scenario == "court":
        ids = FROZEN["court"][a.set or "pilot"]
    else:
        ids = FROZEN["parliament"][a.set or "pilot_candidates"]
    ids = [i for i in ids if i not in done]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        ids = ids[i::n]
    ids = ids[: a.limit]
    print(f"{a.scenario}: {len(ids)} cases to generate (skipping {len(done)} done) -> {out}")
    if a.dry_run: return
    ok = 0
    with open(out, "a") as f:
        for cid in ids:
            t = time.time()
            try:
                rec = with_repair(gen_court if a.scenario == "court" else gen_parliament, cid)
            except Exception as e:
                rec = {"scenario": a.scenario, "case_id": cid, "status": "error", "reasons": [str(e)[:300]]}
            rec["ts"] = time.time(); rec["secs"] = round(time.time() - t, 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            ok += rec["status"] == "ok"
            print(f"  {cid[:48]:48s} {rec['status']:8s} {rec['secs']:5.1f}s {'; '.join(rec.get('reasons', []))[:100]}")
    print(f"ok {ok}/{len(ids)}")


if __name__ == "__main__":
    main()
