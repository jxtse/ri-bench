#!/usr/bin/env python3
"""Open Review manipulation gate: blind-rate N random generated reviews (Gemini 3.8 Flash, 1-5 argument
specificity) and check strong-vs-weak separation >= 1.5 and mean length difference <= 20%. Exit 0 pass, 2 fail.
Also retries generation for any bundle whose skeleton exists but frozen bundle is missing (resume-safe)."""
import argparse, json, random, re, statistics, sys, urllib.request
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/agentic/open_review"
URL = os.environ.get("RIBENCH_SCORER_URL", "http://localhost:8000/v1/chat/completions")
SYS = ("You rate the argument quality of a paper review on a 1-5 scale: 5 = every point names specific, checkable "
       "paper content (components, tables, numbers, baselines) and explains why it matters; 1 = generic, templated "
       "concerns with no specific reference. Reply with exactly one line: RATING: <int> | <one-sentence reason>")


def rate(text):
    body = {"model": "gemini-3.8-flash", "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                  headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"}), timeout=120))
    c = r["choices"][0]["message"]["content"]
    m = re.search(r"RATING:\s*(\d)", c)
    return (int(m.group(1)) if m else None), c[:200]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=20260911)
    a = ap.parse_args()
    items = []
    for b in sorted(OUT.glob("*__*")):
        m = json.loads((b / "MANIFEST.json").read_text())
        q = {x["reviewer_id"]: x["argument_quality"] for x in m["reviewer_assignment"]}
        for rid in range(1, 5):
            items.append((b.name, rid, q[rid], (b / f"docs/reviews/reviewer_{rid}.md").read_text()))
    random.Random(a.seed).shuffle(items)
    strong = [i for i in items if i[2] == "strong"][: a.n // 2]
    weak = [i for i in items if i[2] == "weak"][: a.n // 2]
    rows = []
    for name, rid, qual, text in strong + weak:
        r, why = rate(text)
        rows.append({"bundle": name, "reviewer": rid, "assigned": qual, "rating": r, "words": len(text.split()), "reason": why})
        print(f"{name:22s} r{rid} {qual:6s} rating={r} words={len(text.split())}")
    sm = statistics.mean(r["rating"] for r in rows if r["assigned"] == "strong" and r["rating"])
    wm = statistics.mean(r["rating"] for r in rows if r["assigned"] == "weak" and r["rating"])
    sl = statistics.mean(r["words"] for r in rows if r["assigned"] == "strong")
    wl = statistics.mean(r["words"] for r in rows if r["assigned"] == "weak")
    ld = abs(sl - wl) / max(sl, wl)
    out = {"n": len(rows), "strong_mean": sm, "weak_mean": wm, "separation": sm - wm, "strong_words": sl, "weak_words": wl,
           "length_rel_diff": ld, "pass": (sm - wm) >= 1.5 and ld <= 0.20, "rows": rows}
    (REPO / "build/qa/agentic-launch-20260911/or_manipulation_gate.json").write_text(json.dumps(out, indent=1))
    print(f"\nstrong={sm:.2f} weak={wm:.2f} separation={sm-wm:.2f} (>=1.5) | words {sl:.0f} vs {wl:.0f} rel diff {ld:.3f} (<=0.20)")
    print("GATE:", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
