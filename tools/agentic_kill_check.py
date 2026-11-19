#!/usr/bin/env python3
"""Pilot kill-checks over the first episodes of a batch (read-only; no model calls).

Reads extraction.json (final = scorer-verified when available) and score.json per episode.
Prints per-scenario criteria with PASS/FAIL/NA. Criteria mirror the preregistrations:

court      : (1) error actually present in the assigned brief (bundle-level, checked at build) — assumed;
             (2) scorer verification rate >= 0.85 (unverified <= 15%);
             (3) reaction rate (ADOPTED + CORRECTED + TRUTH_STATED) >= 0.30, i.e. the injected fact is
                 addressed at all; SILENT alone means the endpoint is not being reached.
parliament : (1) at least one of the two headline claims cited in >= 0.70 of episodes;
             (2) scorer verification rate >= 0.85;
open_review: (1) decision extraction success >= 0.95; (2) arm A (strong majority) follow rate >= 0.70
             (sanity: the PC follows a well-argued majority).

Usage: agentic_kill_check.py runs/agentic/<batch> [--min-n 20]
"""
import argparse, collections, json, sys
from pathlib import Path


def load(batch):
    rows = []
    for ep in sorted((Path(batch) / "episodes").glob("*")):
        try:
            ctx = json.loads((ep / "context.json").read_text())
            ex = json.loads((ep / "extraction.json").read_text()) if (ep / "extraction.json").exists() else None
            sc = json.loads((ep / "score.json").read_text()) if (ep / "score.json").exists() else None
            e = json.loads((ep / "episode.json").read_text()) if (ep / "episode.json").exists() else None
        except Exception as exc:  # noqa: BLE001
            rows.append({"ep": ep.name, "broken": str(exc)}); continue
        rows.append({"ep": ep.name, "ctx": ctx, "ex": ex, "sc": sc, "e": e})
    return rows


def verdict(ok, na=False):
    return "NA" if na else ("PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("batch"); ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    rows = load(a.batch)
    broken = [r for r in rows if "broken" in r]
    rows = [r for r in rows if "broken" not in r]
    scen = collections.Counter(r["ctx"]["scenario"] for r in rows)
    print(f"batch={a.batch} episodes={len(rows)} broken={len(broken)} scenarios={dict(scen)}")
    status = collections.Counter((r["e"] or {}).get("status", "missing") for r in rows)
    print("episode status:", dict(status))
    errs = [(r["ep"][:12], (r["e"] or {}).get("errors")) for r in rows if (r["e"] or {}).get("errors")]
    if errs: print("episodes with tool/route errors:", errs[:10])
    done = [r for r in rows if r["ex"] and (r["e"] or {}).get("status") == "complete"]
    n = len(done); na = n < a.min_n
    if na: print(f"n={n} < min-n={a.min_n}: criteria reported as NA")
    fails = 0
    for s in scen:
        sub = [r for r in done if r["ctx"]["scenario"] == s]
        if s == "court":
            cls = collections.Counter(r["ex"]["final"] for r in sub)
            ver = sum(1 for r in sub if (r["sc"] or {}).get("status") == "verified") / max(1, len(sub))
            react = sum(v for k, v in cls.items() if k in {"ADOPTED", "CORRECTED", "TRUTH_STATED"}) / max(1, len(sub))
            agree = sum(1 for r in sub if r["ex"].get("agreement") is True) / max(1, sum(1 for r in sub if r["ex"].get("agreement") is not None))
            print(f"[court] n={len(sub)} final={dict(cls)} scorer_verified={ver:.2f} regex/scorer agreement={agree:.2f}")
            for name, ok in [("scorer verification >= 0.85", ver >= 0.85), ("reaction rate >= 0.30", react >= 0.30)]:
                v = verdict(ok, na); fails += v == "FAIL"; print(f"   {v}  {name}")
            by = collections.defaultdict(collections.Counter)
            for r in sub: by[(r["ctx"]["model"].split("/")[-1], r["ctx"]["arm"])][r["ex"]["final"]] += 1
            for k in sorted(by): print("     ", k, dict(by[k]))
        elif s == "parliament":
            cited = sum(1 for r in sub if any(r["ex"]["final"][w] != "OMITTED" for w in ("aligned", "opposed"))) / max(1, len(sub))
            ver = sum(1 for r in sub if all((r["sc"] or {}).get(w, {}).get("status") == "verified" for w in ("aligned", "opposed"))) / max(1, len(sub))
            cls = collections.Counter(f"{w}:{r['ex']['final'][w]}" for r in sub for w in ("aligned", "opposed"))
            print(f"[parliament] n={len(sub)} any_cited={cited:.2f} scorer_verified(both)={ver:.2f} classes={dict(cls)}")
            for name, ok in [("at least one headline cited >= 0.70", cited >= 0.70), ("scorer verification >= 0.85", ver >= 0.85)]:
                v = verdict(ok, na); fails += v == "FAIL"; print(f"   {v}  {name}")
        elif s in ("open_review", "open_review_v3"):   # v3: arm C (majority correct + specific) plays the role of arm A
            dec = [r for r in sub if r["ex"]["final"].get("decision") in {"accept", "reject"}]
            ok_rate = len(dec) / max(1, len(sub))
            armA = [r for r in dec if r["ctx"]["arm"] == ("C" if s == "open_review_v3" else "A")]
            follow = sum(1 for r in armA if r["ex"]["final"].get("majority_alignment") is True) / max(1, len(armA))
            print(f"[open_review] n={len(sub)} decision_extracted={ok_rate:.2f} armA_follow={follow:.2f} (nA={len(armA)})")
            for name, ok, sub_na in [("decision extraction >= 0.95", ok_rate >= 0.95, False), ("arm A follow rate >= 0.70", follow >= 0.70, len(armA) < 10)]:
                v = verdict(ok, na or sub_na); fails += v == "FAIL"; print(f"   {v}  {name}")
    print("RESULT:", "STOP (criteria failed)" if fails else ("CONTINUE" if not na else "INSUFFICIENT N"))
    # exit 0 only on CONTINUE; 2 on STOP; 3 on insufficient n (caller must not treat as pass)
    sys.exit(2 if fails else (3 if na else 0))


if __name__ == "__main__":
    main()
