#!/usr/bin/env python3
"""Paper-facing case-cluster tables/macros from results/pooled8/summary.json (eight models); never changes raw endpoints."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
NAMES={'claude-opus-5':'Claude Opus 5','gpt-5.6-sol':'GPT-5.6 Sol','gpt-5.6-terra':'GPT-5.6 Terra','gpt-5.6-luna':'GPT-5.6 Luna','glm-5.3':'GLM-5.3','glm-5.3-flash':'GLM-5.3-Flash','qwen3.5-35b-a3b':'Qwen3.5-35B-A3B','pooled':'Pooled'}
ORDER=list(NAMES)
# Eight-model pooled view: results/pooled8 = frozen seven-model export + the supplementary DeepSeek run
# (tools/build_pooled8.sh). The seven-model results/summary.json is left untouched; its per-model entries are
# identical in results/pooled8/summary.json. DeepSeek-V4.1-Flash valid rows are pooled like any other model.
NAMES['deepseek-v4.1-flash']='DeepSeek-V4.1-Flash'
ORDER=[m for m in NAMES if m!='pooled']+['pooled']
import os
# RIBENCH_RESULTS_DIR selects another eight-model view (e.g. results/revision8, the ICLR rebuttal revision); default unchanged.
SUMMARY=Path(os.environ['RIBENCH_RESULTS_DIR'])/'summary.json' if os.environ.get('RIBENCH_RESULTS_DIR') else R/'results/pooled8/summary.json'
def fmt(v,d=2):return '--' if v is None else f'{v:.{d}f}'
def ci(x):return '['+', '.join(fmt(v) for v in x['ci95_case_cluster'])+']'
def best_idx(vals,lower):
 """indices of the best value(s) among model rows (pooled excluded by caller); ties all bold"""
 v=[round(x,2) for x in vals if x is not None];b=(min if lower else max)(v);return {i for i,x in enumerate(vals) if x is not None and round(x,2)==b}
def bold(cell):return r'\textbf{'+cell+'}'
def make_review(cc):
 t=[r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3.5pt}',r'\caption{Constructed votes: 118 papers, eight models. Ctrl and Adv give agreement with the majority in the control and adversarial review packages. Intervals are case-cluster 95\% intervals. Bold: lowest Adv within each prompt. Shift counts reference-consistent control decisions that follow the opposing majority under adversarial reviews / the reverse. Acc/Rej gives adversarial following for historically accepted/rejected papers, pooling prompts. Historical outcomes are reference labels.'+'}',r'\label{tab:or-v3}',r'\resizebox{\linewidth}{!}{\begin{tabular}{l ccc ccc c}\toprule & \multicolumn{3}{c}{Vanilla} & \multicolumn{3}{c}{Obligation} & \\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Model & Ctrl & Adv [95\% CI] & Shift / reverse & Ctrl & Adv [95\% CI] & Shift / reverse & Acc / Rej \\\midrule']
 rows=[]
 for m in ORDER:
  x=cc['open_review_v3'][m];cells=[NAMES[m]];adv=[]
  for cond in ['vanilla','obligation']:
   z=x['rates'][cond];tr=x['effects'][f'{cond}/C_to_X/away_historical']['transitions']
   cells += [fmt(z['C/following']['rate']),fmt(z['X/following']['rate'])+' '+ci(z['X/following']),str(tr['positive'])+' / '+str(tr['negative'])];adv.append(z['X/following']['rate'])
  cells.append(fmt(x['rates']['all']['X/following/reference_accept']['rate'])+' / '+fmt(x['rates']['all']['X/following/reference_reject']['rate']))
  rows.append((m,cells,adv))
 models=[r for r in rows if r[0]!='pooled']
 for k,col in enumerate([2,5]):
  for i in best_idx([r[2][k] for r in models],lower=True):models[i][1][col]=bold(models[i][1][col])
 for m,cells,_ in rows:
  if m=='pooled':t.append(r'\midrule')
  t.append(' & '.join(cells)+r' \\')
 t += [r'\bottomrule\end{tabular}}\end{table}'];(R/'manuscript/sections/table_open_review_v3.tex').write_text('\n'.join(t)+'\n')

def main():
 s=json.loads(SUMMARY.read_text());cc=s['case_cluster']
 t=[r'\begin{table}[tb]\centering\small\setlength{\tabcolsep}{3.5pt}',r'\caption{Parliament: 120 inquiries, eight models, 120 reports per model/prompt. Aligned/opposed (a/o) refer to the stored historical reference. Cited is the fraction of reports citing each claim; QL is the fraction of cited claims losing both limitations, with case-cluster 95\% intervals, followed by a/o rates. Bold: lowest QL within each prompt.'+'}',r'\label{tab:parliament}',r'\resizebox{\linewidth}{!}{\begin{tabular}{l ccc ccc}\toprule & \multicolumn{3}{c}{Vanilla} & \multicolumn{3}{c}{Obligation} \\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Model & Cited a / o & QL [95\% CI] & QL a / o & Cited a / o & QL [95\% CI] & QL a / o \\\midrule']
 rows=[]
 for model in ORDER:
  cells=[NAMES[model]];ql=[]
  for cond in ['vanilla','obligation']:
   a=cc['parliament'][model]['rates'][cond]
   cells += [fmt(a['aligned/citation']['rate'])+' / '+fmt(a['opposed/citation']['rate']),fmt(a['both/ql']['rate'])+' '+ci(a['both/ql']),fmt(a['aligned/ql']['rate'])+' / '+fmt(a['opposed/ql']['rate'])];ql.append(a['both/ql']['rate'])
  rows.append((model,cells,ql))
 models=[r for r in rows if r[0]!='pooled']
 for k,col in enumerate([2,5]):
  for i in best_idx([r[2][k] for r in models],lower=True):models[i][1][col]=bold(models[i][1][col])
 for model,cells,_ in rows:
  if model=='pooled':t.append(r'\midrule')
  t.append(' & '.join(cells)+r' \\')
 t += [r'\bottomrule\end{tabular}}\end{table}'];(R/'manuscript/sections/table_parliament.tex').write_text('\n'.join(t)+'\n')
 t=[r'\section{Case-cluster uncertainty}\label{app:statistics}',r"All intervals in the paper resample cases. The unit of resampling is the paper, court case or inquiry, retaining every model, arm, prompt and witness attached to it. We draw 10,000 samples with replacement using NumPy's multinomial case weights, seed 20260914, and percentile 95\% intervals (linear quantiles). Models are treated as a fixed evaluated set, not sampled from a model population. Counts are pooled within a resample before ratios are computed. Zero-citation cases remain in the sampling frame; all-zero-denominator draws are undefined, excluded rather than set to zero, and counted in the machine-readable summary. Undefined decisions are missing, not negative outcomes. Between-condition conditional-rate differences can involve different cited subsets; same-claim improvements cannot be inferred from them.",r'Pooled episode-level McNemar values treat repeated papers as independent and are not used for inference. No bootstrap-tail probabilities are presented as calibrated p values. Intervals are not multiplicity-adjusted; model-level contrasts and post hoc source checks are descriptive, and the finite set of cases limits generalization.',r'\begin{table}[htbp]\centering\small',r'\caption{Pooled endpoints with case-cluster intervals. Court pools both arms and prompts; OpenReview and Parliament include prompt-specific estimates, with Parliament also pooled across prompts. The denominators are rulings, extractable decisions, or cited witness claims as indicated, not independent cases.}',r'\begin{tabular}{lrrl}\toprule Endpoint & Numerator / denominator & Cases & Rate [95\% CI] \\\midrule']
 rows=[('Court adoption (all)','court','all','both/ADOPTED'),('OpenReview following (V)','open_review_v3','vanilla','X/following'),('OpenReview following (O)','open_review_v3','obligation','X/following'),('Parliament loss (V)','parliament','vanilla','both/ql'),('Parliament loss (O)','parliament','obligation','both/ql'),('Parliament loss (all)','parliament','all','both/ql')]
 for name,scenario,cond,key in rows:
  x=cc[scenario]['pooled']['rates'][cond][key];t.append(f'{name} & {x["k"]} / {x["n"]} & {x["n_cases"]} & {fmt(x["rate"])} {ci(x)}'+r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[htbp]\centering\small',r'\caption{Obligation minus Vanilla effects and case-cluster 95\% intervals. OpenReview uses valid same-paper/model decision pairs; Court correction pairs both arms within each case; Parliament is a difference of two conditional loss rates. Negative OpenReview/Parliament effects indicate lower measured rates.}',r'\begin{tabular}{lccc}\toprule Model & OpenReview following & Court correction & Parliament loss \\\midrule'])
 for m in ORDER:
  cells=[NAMES[m]]
  for scen,key in [('open_review_v3','paired_obligation_minus_vanilla/X/following'),('court','paired_obligation_minus_vanilla/both/CORRECTED'),('parliament','obligation_minus_vanilla/both/ql')]:
   x=cc[scen][m]['effects'][key];cells.append(fmt(x['estimate'])+' '+ci(x))
  t.append(' & '.join(cells)+r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[htbp]\centering\small',r'\caption{Selected pooled paired/design contrasts. Signed changes are right minus left for decision pairs, and aligned minus opposed for witness comparisons. Different witness identity is not a randomized treatment.}\label{tab:app-contrasts}',r'\begin{tabular}{lrl}\toprule Contrast & Valid denominator & Effect [95\% CI] \\\midrule'])
 es=[('Real votes A to B, net toward minority (V)','open_review_v2','vanilla/A_to_B/minority'),('Real votes A to B, net toward minority (O)','open_review_v2','obligation/A_to_B/minority'),('Constructed control to adversarial (all)','open_review_v3','all/C_to_X/away_historical'),('Court B minus A adoption','court','all/B_minus_A/ADOPTED'),('Citation aligned minus opposed (V)','parliament','vanilla/aligned_minus_opposed/citation'),('Loss aligned minus opposed (V)','parliament','vanilla/aligned_minus_opposed/ql'),('Citation aligned minus opposed (O)','parliament','obligation/aligned_minus_opposed/citation'),('Loss aligned minus opposed (O)','parliament','obligation/aligned_minus_opposed/ql')]
 for name,scen,key in es:
  x=cc[scen]['pooled']['effects'][key];n=str(x.get('n')) if 'n' in x else str(x['left']['n'])+' / '+str(x['right']['n']);t.append(f'{name} & {n} & {fmt(x["estimate"])} {ci(x)}'+r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}'])
 (R/'manuscript/sections/app_statistics.tex').write_text('\n'.join(t)+'\n')
 make_review(cc)
 print('generated main tables and statistics appendix from canonical summary')
if __name__=='__main__':main()
