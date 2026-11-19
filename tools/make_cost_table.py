"""Token usage and list-price cost of the senior-role calls, from the run ledgers.

Reads every `runs/agentic/full-*/ledger.jsonl` (the eight models on the full case set) and
`runs/agentic/harness-hard-*/ledger.jsonl` (two-model harness check), sums the provider-reported usage
per model, and prices it at the public list prices below. The judge (Gemini 3.8 Flash) is priced from the
`usage` block of every `score.json`. The figures are estimates from reported usage at list price, not
invoices: the Anthropic endpoint reported no cache hits, so Claude is priced entirely at the uncached
input rate, and DeepSeek is priced at its peak rate. Writes `manuscript/sections/table_cost.tex` and
prints the totals used in the Setup paragraph.
"""
import collections
import glob
import json
import os

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# USD per 1M tokens: (uncached input, cached input, output). Public list prices retrieved 2026-09-24.
PRICE = {
    "foundry-a/claude-opus-5": (5.00, 0.50, 25.00),  # platform.claude.com/docs/en/about-claude/pricing
    "foundry-a/gpt-5.6-sol": (4.00, 0.40, 20.00),  # developers.openai.com/api/docs/pricing
    "foundry-a/gpt-5.6-terra": (2.00, 0.20, 12.00),
    "foundry-b/gpt-5.6-terra": (2.00, 0.20, 12.00),
    "foundry-b/gpt-5.6-luna": (0.20, 0.02, 1.20),
    "mint/glm-5.3": (1.40, 0.26, 4.40),  # docs.z.ai/guides/overview/pricing
    "ark/glm-5.3-flash": (0.15, 0.03, 0.50),
    "openrouter/qwen/qwen3.5-35b-a3b": (0.1643, 0.1643, 1.196),  # openrouter.ai weighted average, no cache tier
    "deepseek/deepseek-flash": (0.30, 0.006, 1.20),  # api-docs.deepseek.com, peak rate
}
JUDGE_PRICE = (0.75, 3.75)  # ai.google.dev/gemini-api/docs/pricing, gemini-3.8-flash paid tier, input / output incl. thinking
NAME = {
    "claude-opus-5": "Claude Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna", "glm-5.3-flash": "GLM-5.3-Flash", "glm-5.3": "GLM-5.3",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B", "deepseek-flash": "DeepSeek-V4.1-Flash",
}
ORDER = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3-flash", "glm-5.3", "qwen3.5-35b-a3b"]


def usage(pattern):
    out = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(R, pattern))):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                u = row.get("usage") or {}
                key = row["model"].split("/")[-1]
                out[key]["calls"] += 1
                out[key]["input"] += u.get("input") or 0
                out[key]["cached"] += u.get("cacheRead") or 0
                out[key]["output"] += u.get("output") or 0
                out[key]["cost"] += cost(row["model"], u)
    return out


def cost(model, u):
    p_in, p_cache, p_out = PRICE[model]
    return ((u.get("input") or 0) * p_in + (u.get("cacheRead") or 0) * p_cache + (u.get("output") or 0) * p_out) / 1e6


def judge():
    c = collections.Counter()
    for path in glob.glob(os.path.join(R, "runs/agentic/full-*/episodes/*/score.json")) + glob.glob(os.path.join(R, "runs/agentic/harness-hard-*/episodes/*/score.json")):
        try:
            u = json.load(open(path)).get("usage") or {}
        except (OSError, ValueError):
            continue
        if not u:
            continue
        c["calls"] += 1
        c["input"] += u.get("prompt_tokens") or 0
        c["output"] += (u.get("completion_tokens") or 0) + (u.get("reasoning_tokens") or 0)
    c["cost"] = (c["input"] * JUDGE_PRICE[0] + c["output"] * JUDGE_PRICE[1]) / 1e6
    return c


def m(x):
    return f"{x / 1e6:,.0f}"


full = usage("runs/agentic/full-*/ledger.jsonl")
hard = usage("runs/agentic/harness-hard-*/ledger.jsonl")
jd = judge()
eight = sum(full[k]["cost"] for k in ORDER + ["deepseek-flash"])
hard_total = sum(v["cost"] for v in hard.values())
total = eight + hard_total + jd["cost"]

lines = [
    r"\begin{table}[t]", r"\centering", r"\small",
    r"\caption{Senior-role token usage and list-price cost, from the provider-reported usage in the run ledgers. Input counts include cached tokens, priced at the cached rate where the provider reported cache hits; the Anthropic endpoint reported none. Public list prices as of 24 September 2026; the figures are estimates from usage, not invoices.}",
    r"\label{tab:cost}",
    r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
    r"Model & Calls & Input (M) & Output (M) & USD \\", r"\midrule",
]
EIGHT = ORDER + ["deepseek-flash"]
for k in EIGHT:
    v = full[k]
    lines.append(f"{NAME[k]} & {v['calls']:,} & {m(v['input'] + v['cached'])} & {m(v['output'])} & {v['cost']:,.0f} \\\\")
lines.append(r"\midrule")
hv = collections.Counter()
for vv in hard.values():
    for kk in ("calls", "input", "cached", "output", "cost"):
        hv[kk] += vv[kk]
lines.append(f"Harness check, hard-20 (\\S\\ref{{sec:harness}}) & {hv['calls']:,} & {m(hv['input'] + hv['cached'])} & {m(hv['output'])} & {hv['cost']:,.0f} \\\\")
lines.append(f"Judge, Gemini 3.8 Flash & {jd['calls']:,} & {m(jd['input'])} & {m(jd['output'])} & {jd['cost']:,.0f} \\\\")
lines.append(r"\midrule")
tc = sum(full[k]["calls"] for k in EIGHT) + hv["calls"] + jd["calls"]
ti = sum(full[k]["input"] + full[k]["cached"] for k in EIGHT) + hv["input"] + hv["cached"] + jd["input"]
to = sum(full[k]["output"] for k in EIGHT) + hv["output"] + jd["output"]
lines.append(f"Total & {tc:,} & {m(ti)} & {m(to)} & {total:,.0f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = os.path.join(R, "manuscript", "sections", "table_cost.tex")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", out)
print(f"eight models, full case set: {eight:,.0f} USD (Claude {full['claude-opus-5']['cost'] / eight:.0%}); harness check {hard_total:,.0f}; judge {jd['cost']:,.0f} over {jd['calls']:,} calls; total {total:,.0f}")
print(f"eight-model tokens: {m(sum(full[k]['input'] + full[k]['cached'] for k in EIGHT))}M in, {m(sum(full[k]['output'] for k in EIGHT))}M out")
print(f"claude uncached input {m(full['claude-opus-5']['input'])}M; at the cached rate on 75% of it the eight-model figure would be {eight - full['claude-opus-5']['input'] * 0.75 * (5.0 - 0.5) / 1e6:,.0f} USD")
