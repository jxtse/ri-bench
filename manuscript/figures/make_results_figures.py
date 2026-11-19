#!/usr/bin/env python3
"""Results figures for the RI-Bench manuscript. Reads results/*.jsonl (tools/export_results.py), writes
manuscript/figures/fig_results_overview.pdf and fig_process.pdf plus a JSON of every plotted number.
Deterministic; no model calls."""
import glob, json, os, statistics, collections, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, "manuscript", "figures")
ORDER = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3", "glm-5.3-flash", "qwen3.5-35b-a3b"]
NAME = {"claude-opus-5": "Claude Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-terra": "GPT-5.6 Terra", "gpt-5.6-luna": "GPT-5.6 Luna",
        "glm-5.3": "GLM-5.3", "glm-5.3-flash": "GLM-5.3-Flash", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B"}
SOL_DUP = "foundry-a/gpt-5.6-terra"
INK, VAN, OBL, MUTE = "#222222", "#012F63", "#1F7A5A", "#8a8a8a"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     # Helvetica to match Figure 1 (docs/figures/figure1-redesign-20260922/gen.py); math text too
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic",
                     "mathtext.bf": "Helvetica:bold"})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(scenario):
    """Rows from results/<scenario>.jsonl (exported by tools/export_results.py); no access to runs/ needed."""
    rows = []
    with open(os.path.join(os.environ.get("RIBENCH_RESULTS_DIR") or os.path.join(R, "results"), f"{scenario}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows.append(dict(model=r["model"], cond=r["condition"], arm=r["arm"], case=r["case_id"], final=r["final"], roles=r["roles"]))
    return rows


numbers = {}
with open(os.path.join(R, "results", "summary.json")) as stream:
    summary = json.load(stream)
# RIBENCH_RESULTS_DIR selects a self-contained eight-model view (e.g. results/revision8, the ICLR rebuttal revision);
# the overview panels then read every model, DeepSeek included, from that view's summary. Default unchanged.
VIEW = os.environ.get("RIBENCH_RESULTS_DIR")
if VIEW:
    with open(os.path.join(VIEW, "summary.json")) as stream:
        summary = json.load(stream)

# Supplementary DeepSeek-V4.1-Flash run (results/supplement/deepseek; tools/export_results_deepseek.py, then
# summarize_results.py with RIBENCH_RESULTS_DIR). Shown as an ordinary row in both figures; every seven-model number stays unchanged, and the supplement is kept out of `summary`.
SUPP = "deepseek-v4.1-flash"
SUPP_SUMMARY = os.path.join(R, "results", "supplement", "deepseek", "summary.json")
supp = json.load(open(SUPP_SUMMARY))["case_cluster"] if os.path.exists(SUPP_SUMMARY) else None
NAME[SUPP] = "DeepSeek-V4.1-Flash"
OVERVIEW = ORDER + ([SUPP] if supp else [])
def cluster(scenario, model, condition, metric):
    source = summary["case_cluster"] if VIEW or model != SUPP else supp
    return source[scenario][model]["rates"][condition][metric]

# ---------------- Figure: results overview ----------------
v3 = load("open_review_v3"); court = load("court"); parl = load("parliament")
panels = []
# (a) wrong-majority following, adversarial arm
pa = {(model, condition): cluster("open_review_v3", model, condition, "X/following")
      for model in OVERVIEW for condition in ("vanilla", "obligation")}
# (b) court adoption (both arms)
pb = {(model, condition): cluster("court", model, condition, "both/ADOPTED")
      for model in OVERVIEW for condition in ("vanilla", "obligation")}
# (c) parliament limitation loss among cited claims
pc = {(model, condition): cluster("parliament", model, condition, "both/ql")
      for model in OVERVIEW for condition in ("vanilla", "obligation")}
numbers["overview"] = {name: {f"{model}|{condition}": [value["k"], value["n"]] for (model, condition), value in panel.items() if model != SUPP}
                       for name, panel in (("follow_wrong_majority", pa), ("court_adopted", pb), ("parliament_limitation_loss", pc))}
numbers["overview_case_cluster"] = {name: {f"{model}|{condition}": value for (model, condition), value in panel.items() if model != SUPP}
                                    for name, panel in (("follow_adversarial_majority", pa), ("court_adopted", pb), ("parliament_limitation_loss", pc))}
if supp:
    numbers["overview_supplement"] = {name: {f"{model}|{condition}": value for (model, condition), value in panel.items() if model == SUPP}
                                      for name, panel in (("follow_adversarial_majority", pa), ("court_adopted", pb), ("parliament_limitation_loss", pc))}
    numbers["overview_supplement"]["source"] = os.path.relpath(SUPP_SUMMARY, R)
numbers["semantics"] = {"legacy_follow_wrong_majority_key": "adversarial-majority following",
                        "paper_coverage": "tool-volume proxy, not unique or complete reading",
                        "output_tokens": "reported output tokens; reasoning reported separately, not summed",
                        "files_read": "logged distinct read paths"}

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2 + (0.25 if supp else 0)), sharey=True)
titles = ["(a) Weigh: follows\nadversarial majority", "(b) Verify: adopts\nplanted error", "(c) Relay: drops\nlimitations"]
thresholds = [0.30, 0.20, 0.20]
xlims = [(-0.02, 0.72), (-0.005, 0.125), (-0.02, 0.80)]
xticks = [[0, 0.2, 0.4, 0.6], [0, 0.04, 0.08], [0, 0.2, 0.4, 0.6]]
ys_all = list(range(len(OVERVIEW)))[::-1]
for ax, data, title, thr, xl, xt in zip(axes, [pa, pb, pc], titles, thresholds, xlims, xticks):
    for y, m in zip(ys_all, OVERVIEW):
        for cond, color, dy, mk in [("vanilla", VAN, 0.16, "o"), ("obligation", OBL, -0.16, "s")]:
            value = data[(m, cond)]; p = value["rate"]; lo, hi = value["ci95_case_cluster"]
            if p is None:
                continue
            ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=1.1, solid_capstyle="round")
            ax.plot(p, y + dy, marker=mk, color=color, ms=4.2, mec="white", mew=0.5, zorder=3)
    if xl[0] < thr < xl[1]:
        ax.axvline(thr, color=MUTE, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(ys_all); ax.set_yticklabels([NAME[m] for m in OVERVIEW])
    ax.set_xlim(*xl); ax.set_xticks(xt); ax.set_title(title, loc="left", fontsize=8)
    ax.set_ylim(-0.6, ys_all[0] + 0.6)
    ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
axes[1].plot([], [], "o", color=VAN, label="Vanilla prompt"); axes[1].plot([], [], "s", color=OBL, label="Obligation prompt")
fig.legend(loc="lower center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
axes[1].set_xlabel("rate (95% case-cluster interval)")
L = 0.25 if supp else 0.22  # the DeepSeek label is the longest
fig.subplots_adjust(left=L, right=0.97, bottom=0.31 - (0.03 if supp else 0), top=0.83 + (0.02 if supp else 0))
gap = 0.035; w = (0.97 - L - 2 * gap) / 3
for i, ax in enumerate(axes):
    pos = ax.get_position(); ax.set_position([L + i * (w + gap), pos.y0, w, pos.height])
axes[0].set_xlim(-0.02, 0.88 if VIEW else 0.72); axes[2].set_xlim(-0.02, 0.80)  # revision view: Vanilla GLM-5.3 CI reaches 0.80
fig.savefig(os.path.join(OUT, "fig_results_overview.pdf")); plt.close(fig)

# ---------------- Figure: process ----------------
# For the v3 area chair (the reader in Open Review), per model: paper coverage, files read, output+reasoning tokens.
PROC = OVERVIEW
proc = {m: {"coverage": [], "files": [], "out": [], "tools": []} for m in PROC}
v3_supp = []
if supp and not VIEW:
    with open(os.path.join(R, "results", "supplement", "deepseek", "open_review_v3.jsonl")) as f:
        v3_supp = [json.loads(line) for line in f]
        v3_supp = [dict(model=r["model"], cond=r["condition"], arm=r["arm"], case=r["case_id"], final=r["final"], roles=r["roles"]) for r in v3_supp]
for r in v3 + v3_supp:
    m = r["model"]; ac = r["roles"]["area_chair"]
    for target, source in (("coverage", "paper_coverage"), ("files", "files_read"), ("out", "output_tokens"), ("tools", "tool_calls")):
        if ac.get(source) is not None:
            proc[m][target].append(ac[source])
follow = {m: (pa[(m, "vanilla")]["k"] + pa[(m, "obligation")]["k"]) / (pa[(m, "vanilla")]["n"] + pa[(m, "obligation")]["n"]) for m in PROC}
numbers["process"] = {m: {"paper_coverage_mean": sum(v["coverage"]) / len(v["coverage"]), "paper_coverage_frac_ge_0.9": sum(x >= 0.9 for x in v["coverage"]) / len(v["coverage"]),
                          "files_read_median": statistics.median(v["files"]), "output_tokens_median": statistics.median(v["out"]), "tool_calls_median": statistics.median(v["tools"]),
                          "follow_wrong_majority_pooled": follow[m], "n": len(v["coverage"])} for m, v in proc.items() if m != SUPP}
if supp:
    numbers["process_supplement"] = {m: {"paper_coverage_mean": sum(v["coverage"]) / len(v["coverage"]), "paper_coverage_frac_ge_0.9": sum(x >= 0.9 for x in v["coverage"]) / len(v["coverage"]),
                                         "output_tokens_median": statistics.median(v["out"]), "follow_wrong_majority_pooled": follow[m], "n": len(v["coverage"])} for m, v in proc.items() if m == SUPP}
for model, values in proc.items():
    if model == SUPP:
        continue
    planned = sum(row["model"] == model for row in v3)
    numbers["process"][model]["missingness"] = {metric: {"valid": len(values[metric]), "missing": planned - len(values[metric]), "planned": planned}
                                                     for metric in values}
    numbers["process"][model]["output_tokens_iqr"] = np.quantile(values["out"], [0.25, 0.75]).tolist()

fig = plt.figure(figsize=(5.5, 2.4 + (0.25 if supp else 0)))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)
ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1], sharey=ax_a); ax_c = fig.add_subplot(gs[2])
axes = [ax_a, ax_b, ax_c]
ys = list(range(len(PROC)))[::-1]
# (a) fraction of episodes with >=90% of the paper read
ax = ax_a
for y, m in zip(ys, PROC):
    v = proc[m]["coverage"]
    ax.barh(y, sum(x >= 0.9 for x in v) / len(v), color=VAN, height=0.6)
ax.set_yticks(ys); ax.set_yticklabels([NAME[m] for m in PROC]); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, ys[0] + 0.6)
ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
ax.set_title("(a) Tool-volume\nproxy $\\geq$0.90", loc="left", fontsize=8); ax.set_xlabel("fraction of episodes")
ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
# (b) provider-reported output tokens (median, IQR); reasoning is not added
ax = ax_b
for y, m in zip(ys, PROC):
    q1, med, q3 = np.quantile(proc[m]["out"], [0.25, 0.5, 0.75])
    ax.plot([q1, q3], [y, y], color=OBL, lw=1.6, solid_capstyle="round"); ax.plot(med, y, "o", color=OBL, ms=4, mec="white", mew=0.5, zorder=3)
ax.tick_params(axis="y", length=0, labelleft=False)
ax.set_title("(b) Reported\noutput tokens", loc="left", fontsize=8); ax.set_xlabel("per AC episode")
ax.set_xlim(0, 10500); ax.set_xticks([2000, 4000, 6000, 8000, 10000]); ax.set_xticklabels(["2k", "4k", "6k", "8k", "10k"])
ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
# (c) scatter: coverage vs wrong-majority following; labels in a column at left with thin leaders
ax = ax_c
pts = {m: (sum(proc[m]["coverage"]) / len(proc[m]["coverage"]), follow[m]) for m in PROC}
for m, (x, yv) in pts.items():
    ax.plot(x, yv, "o", color=INK, ms=4.5, zorder=3)
y0, y1 = 0.05, 0.52
ax.set_xlim(0.52, 1.08); ax.set_ylim(y0, y1)
short = lambda m: "DeepSeek" if m == SUPP else NAME[m]  # the full name does not fit beside the crowded upper-right cluster
L = 0.25 if supp else 0.22
fig.subplots_adjust(left=L, right=0.9, bottom=0.21 - (0.02 if supp else 0), top=0.81 + (0.02 if supp else 0), wspace=0.14)
pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
w = (0.9 - L - 2 * 0.018) / 3
ax_a.set_position([L, pos_a.y0, w, pos_a.height]); ax_b.set_position([L + w + 0.018, pos_b.y0, w, pos_b.height]); ax_c.set_position([L + 2 * (w + 0.018), pos_c.y0, w, pos_c.height])
# label next to each point: measure real text extents with the renderer, then pick the candidate offset that
# stays inside the axes and is farthest from other markers and placed labels (box-to-point distance)
import itertools
fig.canvas.draw()
inv = ax.transData.inverted()
def text_extent(lab, x, y, ha):
    t = ax.text(x, y, lab, fontsize=7, ha=ha, va="center"); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    (x0, yb0), (x1, yb1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1)); return x0, yb0, x1, yb1
placed = []
cands = [(dx, dy) for dx, dy in itertools.product([-0.015, 0.015, 0.0, -0.03, 0.03], [0.0, 0.03, -0.03, 0.055, -0.055]) if not (dx == 0 and dy == 0)]
XL, XR = 0.52, 1.08
def score(px, py, dx, dy, lab):
    ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
    x0, b0, x1, b1 = text_extent(lab, px + dx, py + dy, ha)
    inside = x0 >= XL + 0.01 and x1 <= XR - 0.005 and b0 >= y0 and b1 <= y1
    def box_pt(qx, qy):
        cx = min(max(qx, x0), x1); cy = min(max(qy, b0), b1); return math.hypot((cx - qx) / (XR - XL), (cy - qy) / (y1 - y0))
    d_pts = min(box_pt(qx, qy) for qm, (qx, qy) in pts.items() if (qx, qy) != (px, py))
    if d_pts < 0.035:  # text box within one marker radius of a foreign marker: hard reject
        return -20 + d_pts
    d_lab = min([max(0, max(lx0 - x1, x0 - lx1) / (XR - XL)) + max(0, max(lb0 - b1, b0 - lb1) / (y1 - y0)) for lx0, lb0, lx1, lb1 in placed] or [9])
    # prefer the nearest offset that is clear of everything: cap the clearance reward, then penalise distance
    return min(min(d_pts, d_lab), 0.12) - 0.6 * math.hypot(dx / (XR - XL), dy / (y1 - y0)) + (0 if inside else -9)
PIN = {SUPP: (-0.022, 0.018), "gpt-5.6-terra": (-0.04, -0.035), "gpt-5.6-sol": (-0.04, -0.038), "glm-5.3": (0.025, 0.0), "qwen3.5-35b-a3b": (-0.015, 0.04), "gpt-5.6-luna": (-0.025, 0.012), "glm-5.3-flash": (-0.025, 0.0), "claude-opus-5": (-0.025, 0.0)}  # full names; labels remain adjacent, Sol/Terra below
for m in sorted(PROC, key=lambda m: pts[m][1]):
    px, py = pts[m]; lab = short(m)
    dx, dy = PIN.get(m) or max(cands, key=lambda c: score(px, py, *c, lab))
    ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
    ax.annotate(lab, (px, py), (px + dx, py + dy), fontsize=7, ha=ha, va="center")
    placed.append(text_extent(lab, px + dx, py + dy, ha))
ax.set_xlabel("tool-volume proxy"); ax.set_ylabel("follows adversarial majority"); ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True)
ax.set_title("(c) Tool-volume vs.\nmajority following", loc="left", fontsize=8); ax.set_xticks([0.6, 0.8, 1.0]); ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig_process.pdf")); plt.close(fig)

json.dump(numbers, open(os.path.join(OUT, "fig_numbers.json"), "w"), indent=1)
print("wrote", OUT)
