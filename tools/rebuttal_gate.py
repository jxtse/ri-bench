#!/usr/bin/env python3
"""Generation gates for rebuttal designs with new reviews (or_e2b, or_e3a, or_e3b), same criteria as v3 (§3.3):
  1. Direction consistency on every newly generated initial review: specific >= 0.90, generic >= 0.80
  2. Specificity separation on 60 random new initial reviews: specific - generic >= 1.5, mean length diff <= 20%
  or_e2b additionally: separation computed separately for SP and SM (each must pass 2.).
Rater: Gemini 3.8 Flash via OpenRouter, temperature 0, same prompts as v3.
On pass, removes the subject-run STOP holds for that scenario. Writes build/qa/rebuttal-20261107/gates_<scenario>.json.
  rebuttal_gate.py or_e2b
"""
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from agentic_or_v3_gate import SYS_DIR, SYS_SPEC  # noqa: E402  (identical rater prompts)

SCEN = sys.argv[1]
OUT = REPO / "data/agentic" / SCEN


def ask(sys_prompt, text):
    body = {"model": "google/gemini-3.8-flash", "temperature": 0, "max_tokens": 4000,
            "provider": {"order": ["Google", "Google AI Studio"], "allow_fallbacks": False},
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, KeyError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("rater unavailable")


def main():
    items = []
    for b in sorted(OUT.glob("*__*")):
        m = json.loads((b / "MANIFEST.json").read_text())
        reused = set()
        if m.get("reuse_initial_from"):
            src = REPO / "data/agentic" / m["reuse_initial_from"] / "docs/reviews"
            reused = {a["reviewer_id"] for a in m["reviewer_assignment"] if (src / f"reviewer_{a['reviewer_id']}.md").exists()}
        for x in m["reviewer_assignment"]:
            if x["reviewer_id"] in reused:
                continue  # v3 review, already gated in September
            items.append(dict(bundle=b.name, arm=m["arm"], reviewer=x["reviewer_id"], quality=x["argument_quality"],
                              score=x["initial_score"], text=(b / f"docs/reviews/reviewer_{x['reviewer_id']}.md").read_text()))
    print(f"{SCEN}: {len(items)} new initial reviews in {len({i['bundle'] for i in items})} bundles", flush=True)

    def direction(it):
        c = ask(SYS_DIR, it["text"])
        it["supports"] = bool(re.search(r"SUPPORTS:\s*yes", c, re.I)); it["dir_reason"] = c[:200]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(direction, items))
    dirrate = {q: statistics.mean(it["supports"] for it in items if it["quality"] == q)
               for q in ("specific", "generic") if any(it["quality"] == q for it in items)}

    def spec(it):
        c = ask(SYS_SPEC, it["text"]); mm = re.search(r"RATING:\s*(\d)", c); it["rating"] = int(mm.group(1)) if mm else None

    def separation(pool_items, k, seed):
        sample = random.Random(seed).sample(pool_items, min(k, len(pool_items)))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(spec, [s for s in sample if "rating" not in s]))
        sp = [s["rating"] for s in sample if s["quality"] == "specific" and s["rating"]]
        ge = [s["rating"] for s in sample if s["quality"] == "generic" and s["rating"]]
        sl = statistics.mean(len(s["text"].split()) for s in pool_items if s["quality"] == "specific")
        gl = statistics.mean(len(s["text"].split()) for s in pool_items if s["quality"] == "generic") if any(s["quality"] == "generic" for s in pool_items) else sl
        out = {"n": len(sample), "specific_mean": statistics.mean(sp) if sp else None, "generic_mean": statistics.mean(ge) if ge else None,
               "specific_words": sl, "generic_words": gl, "length_rel_diff": abs(sl - gl) / max(sl, gl)}
        out["separation"] = (out["specific_mean"] - out["generic_mean"]) if sp and ge else None
        out["pass"] = out["separation"] is None or (out["separation"] >= 1.5 and out["length_rel_diff"] <= 0.20)
        return out

    seps = {"all": separation(items, 60, 20261107)}
    if SCEN == "or_e2b":
        for arm in ("SP", "SM"):
            seps[arm] = separation([i for i in items if i["arm"] == arm], 60, f"20261107:{arm}")
    direction_pass = dirrate.get("specific", 1) >= 0.90 and dirrate.get("generic", 1) >= 0.80
    ok = direction_pass and all(s["pass"] for s in seps.values())
    result = {"scenario": SCEN, "n_reviews": len(items), "direction": dirrate, "direction_pass": direction_pass,
              "specificity": seps, "pass": ok,
              "direction_failures": [{k: it[k] for k in ("bundle", "reviewer", "quality", "score", "dir_reason")} for it in items if not it["supports"]]}
    (REPO / "build/qa/rebuttal-20261107" / f"gates_{SCEN}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "direction_failures"}, indent=1))
    if ok:
        for stop in (REPO / "runs/agentic").glob(f"rebuttal-{SCEN.replace('_', '-')}-*-20261107/STOP"):
            if stop.read_text().startswith("hold: generated reviews"):
                stop.unlink()
        print("holds released")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
