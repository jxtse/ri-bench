#!/usr/bin/env python3
"""Open Review v3 generation gates (PREREG_OPEN_REVIEW_V3 §3.3), blind-rated by Gemini 3.8 Flash (not a subject).

  1. Direction consistency: does each review's argument support its own score/stance?  specific >= 0.90, generic >= 0.80
  2. Specificity separation: 60 random reviews rated 1-5; specific - generic >= 1.5, mean length diff <= 20%
Exit 0 pass, 2 fail. Writes build/qa/agentic-launch-20260911/or_v3_gates.json. Retries transient 5xx.
"""
import json, random, re, statistics, sys, time, urllib.error, urllib.request
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/agentic/open_review_v3"
URL = os.environ.get("RIBENCH_SCORER_URL", "http://localhost:8000/v1/chat/completions")
SYS_SPEC = ("You rate the argument quality of a paper review on a 1-5 scale: 5 = every point names specific, checkable "
            "paper content (components, tables, numbers, baselines) and explains why it matters; 1 = generic, templated "
            "concerns with no specific reference. Reply with exactly one line: RATING: <int> | <one-sentence reason>")
SYS_DIR = ("You read a paper review that states a numeric score out of 10 (>=6 means accept). Judge whether the review's "
           "arguments, taken together, SUPPORT that score's direction (a low score backed mainly by criticisms; a high score "
           "backed mainly by strengths). A review that gives a high score but argues mostly against the paper, or vice versa, "
           "does NOT support its score. Reply with exactly one line: SUPPORTS: yes|no | <one-sentence reason>")


def ask(sys_prompt, text):
    body = {"model": "gemini-3.8-flash", "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]}
    for attempt in range(6):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"}), timeout=45))
            return r["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503):
                time.sleep(20 * (attempt + 1)); continue
            raise
        except (TimeoutError, OSError):  # hung socket: retry rather than wait forever
            time.sleep(5); continue
    raise RuntimeError("rater unavailable")


def main():
    items = []
    for b in sorted(OUT.glob("*__*")):
        m = json.loads((b / "MANIFEST.json").read_text())
        for x in m["reviewer_assignment"]:
            rid = x["reviewer_id"]
            items.append(dict(bundle=b.name, reviewer=rid, quality=x["argument_quality"], group=x["group"], score=x["initial_score"],
                              text=(b / f"docs/reviews/reviewer_{rid}.md").read_text()))
    print(f"{len(items)} reviews in {len(items)//4} bundles")
    # gate 1: direction consistency on ALL reviews
    for k, it in enumerate(items):
        if k % 50 == 0: print(f"direction {k}/{len(items)}", flush=True)
        c = ask(SYS_DIR, it["text"]); it["supports"] = bool(re.search(r"SUPPORTS:\s*yes", c, re.I)); it["dir_reason"] = c[:160]
    dirrate = {q: statistics.mean(it["supports"] for it in items if it["quality"] == q) for q in ("specific", "generic")}
    # gate 2: specificity separation on 60 random
    rng = random.Random(20260912); sample = rng.sample(items, 60)
    for it in sample:
        c = ask(SYS_SPEC, it["text"]); mm = re.search(r"RATING:\s*(\d)", c); it["rating"] = int(mm.group(1)) if mm else None
    sm = statistics.mean(it["rating"] for it in sample if it["quality"] == "specific" and it["rating"])
    gm = statistics.mean(it["rating"] for it in sample if it["quality"] == "generic" and it["rating"])
    sl = statistics.mean(len(it["text"].split()) for it in items if it["quality"] == "specific")
    gl = statistics.mean(len(it["text"].split()) for it in items if it["quality"] == "generic")
    ld = abs(sl - gl) / max(sl, gl)
    out = dict(n_reviews=len(items), direction=dirrate, direction_pass=dirrate["specific"] >= 0.90 and dirrate["generic"] >= 0.80,
               specific_mean=sm, generic_mean=gm, separation=sm - gm, specific_words=sl, generic_words=gl, length_rel_diff=ld,
               specificity_pass=(sm - gm) >= 1.5 and ld <= 0.20,
               rows=[{k: v for k, v in it.items() if k != "text"} for it in items])
    out["pass"] = out["direction_pass"] and out["specificity_pass"]
    (REPO / "build/qa/agentic-launch-20260911/or_v3_gates.json").write_text(json.dumps(out, indent=1))
    print(f"direction: specific={dirrate['specific']:.2f} (>=0.90) generic={dirrate['generic']:.2f} (>=0.80) -> {'PASS' if out['direction_pass'] else 'FAIL'}")
    print(f"specificity: specific={sm:.2f} generic={gm:.2f} sep={sm-gm:.2f} (>=1.5) | words {sl:.0f} vs {gl:.0f} rel {ld:.3f} (<=0.20) -> {'PASS' if out['specificity_pass'] else 'FAIL'}")
    print("GATES:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
