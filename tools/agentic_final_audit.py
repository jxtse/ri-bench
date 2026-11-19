#!/usr/bin/env python3
"""Final consistency audit of all RI-Bench agentic batches. Read-only. For every batch: expected keys from the plan(s),
complete keys, duplicates, status of every role episode, extraction present, score present (court/parliament),
returned_model matches, plausibility probes, retried/archived counts. Exit 1 if any batch fails a hard check."""
import os, collections, glob, json, re, sys

R = os.environ.get("RIBENCH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECT_MODEL = {"foundry-a/claude-opus-5": r"claude-opus-5", "foundry-a/gpt-5.6-sol": r"gpt-5\.6-sol", "foundry-b/gpt-5.6-luna": r"gpt-5\.6-luna",
    "foundry-b/gpt-5.6-terra": r"gpt-5\.6-terra", "mint/glm-5.3": r"glm-5\.3$|glm-5-3$", "ark/glm-5.3-flash": r"glm-5[.-]3-flash",
    "openrouter/qwen/qwen3.5-35b-a3b": r"qwen3\.5-35b-a3b"}
SOL_DUP = "foundry-a/gpt-5.6-terra"
hard_fail = False
report = {}
plans = {}
for p in glob.glob(f"{R}/configs/agentic/plans/full_*.json"):
    pl = json.load(open(p)); plans.setdefault(pl["batch"], []).append(pl)
for br in sorted(glob.glob(f"{R}/runs/agentic/full-*")):
    batch = os.path.basename(br)
    if batch not in plans: report[batch] = {"note": "no plan file (historical lane); skipped"}; continue
    scen = plans[batch][0]["scenario"]
    expected = set()
    for pl in plans[batch]:
        for m in pl["models"]:
            if m == SOL_DUP: continue
            for c in pl["cases"]:
                for a in pl["arms"]:
                    for cd in pl["conditions"]: expected.add((m, c, a, cd))
    have = collections.Counter(); bad_status = []; no_extract = []; no_score = []; wrong_model = []; probes = []
    for ed in glob.glob(br + "/episodes/*"):
        ctx = json.load(open(ed + "/context.json")); m = ctx["model"]
        if m == SOL_DUP: continue
        k = (m, ctx["case_id"], ctx["arm"], ctx["condition"])
        roles = sorted(glob.glob(ed + "/*/episode.json")) or ([ed + "/episode.json"] if os.path.exists(ed + "/episode.json") else [])
        eps = [json.load(open(f)) for f in roles]
        need = 2 if scen.startswith("open_review") else 1
        if len(eps) < need or any(e["status"] != "complete" for e in eps):
            bad_status.append((os.path.basename(ed)[:10], [e["status"] for e in eps])); continue
        if not os.path.exists(ed + "/extraction.json"): no_extract.append(os.path.basename(ed)[:10]); continue
        have[k] += 1
        if scen in ("court", "parliament") and not os.path.exists(ed + "/score.json"): no_score.append(os.path.basename(ed)[:10])
        rm = eps[-1].get("returned_model") or ""
        if m in EXPECT_MODEL and not re.search(EXPECT_MODEL[m], rm): wrong_model.append((os.path.basename(ed)[:10], rm))
        if sum(len(e.get("tools", [])) for e in eps) == 0 or sum(len(e.get("files_read", [])) for e in eps) == 0: probes.append((os.path.basename(ed)[:10], "no tools/files"))
        if len(eps[-1].get("final_text") or "") < 300: probes.append((os.path.basename(ed)[:10], f"final_text {len(eps[-1].get('final_text') or '')}"))
    missing = expected - set(have); dup = {k: v for k, v in have.items() if v > 1}; extra = set(have) - expected
    retried = len(os.listdir(br + "/retried_episodes")) if os.path.isdir(br + "/retried_episodes") else 0
    ok = not missing and not dup and not bad_status and not no_extract and not wrong_model and not probes and not no_score
    hard_fail |= not ok
    report[batch] = dict(scenario=scen, expected=len(expected), complete=len(have), missing=sorted(missing)[:5], n_missing=len(missing), duplicates=len(dup),
                         extra_keys=len(extra), bad_status=bad_status[:5], n_bad_status=len(bad_status), no_extraction=no_extract[:5], unscored=len(no_score),
                         wrong_model=wrong_model[:5], probes=probes[:5], retried_archived=retried, OK=ok)
print(json.dumps(report, indent=1, ensure_ascii=False))
print("\nSUMMARY")
for b, r in report.items():
    if "OK" in r: print(f"{'OK ' if r['OK'] else 'BAD'} {b:44s} {r['complete']:5d}/{r['expected']:<5d} missing={r['n_missing']} dup={r['duplicates']} badstatus={r['n_bad_status']} unscored={r['unscored']} wrongmodel={len(r['wrong_model'])} probes={len(r['probes'])} archived={r['retried_archived']}")
sys.exit(1 if hard_fail else 0)
