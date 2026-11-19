#!/usr/bin/env python3
"""Deterministic scorer/fallback accounting; no new model calls or label edits."""
import collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def esc(s): return s.replace('_',r'\_')
def load(f):return [json.loads(l) for l in (R/'results'/f'{f}.jsonl').read_text().splitlines()]
def main():
 c,p=load('court'),load('parliament');out={'schema_version':1,'court':{},'parliament':{},'by_model_condition':[]}
 labs=['ADOPTED','CORRECTED','TRUTH_STATED','SILENT'];cv=[r for r in c if r['scorer'].get('status')=='verified']
 raw_final=collections.Counter((r['pattern_label']['classification'],r['final']) for r in c)
 raw_semantic=collections.Counter((r['pattern_label']['classification'],r['scorer']['classification']) for r in cv)
 out['court']={'n':len(c),'verified':len(cv),'fallback':len(c)-len(cv),'verified_silent_no_evidence_required':sum(r['scorer']['classification']=='SILENT' for r in cv),'pattern_final_diagonal':sum(raw_final[x,x] for x in labs),'pattern_semantic_verified_diagonal':sum(raw_semantic[x,x] for x in labs),'matrix_labels':labs,'pattern_final_matrix':[[raw_final[x,y] for y in labs] for x in labs],'pattern_semantic_verified_matrix':[[raw_semantic[x,y] for y in labs] for x in labs]}
 pw=[r['scorer'][w] for r in p for w in ('aligned','opposed')]
 out['parliament']={'n':len(pw),'verified':sum(x.get('status')=='verified' for x in pw),'fallback':sum(x.get('status')!='verified' for x in pw),'verified_uncited_no_evidence_required':sum(x.get('status')=='verified' and not x.get('cited') for x in pw)}
 for model in sorted({r['model'] for r in c}):
  for cond in ['vanilla','obligation']:
   cc=[r for r in c if r['model']==model and r['condition']==cond];pp=[r['scorer'][w] for r in p if r['model']==model and r['condition']==cond for w in ('aligned','opposed')]
   out['by_model_condition'].append({'model':model,'condition':cond,'court_n':len(cc),'court_verified':sum(r['scorer'].get('status')=='verified' for r in cc),'court_fallback':sum(r['scorer'].get('status')!='verified' for r in cc),'parliament_witness_n':len(pp),'parliament_verified':sum(x.get('status')=='verified' for x in pp),'parliament_fallback':sum(x.get('status')!='verified' for x in pp)})
 retention={}
 for cond in ['vanilla','obligation']:
  ctr=collections.Counter()
  for row in p:
   if row['condition']!=cond:continue
   for side in ['aligned','opposed']:
    q=row['scorer'][side]
    key='unverified' if q.get('status')!='verified' else 'uncited' if not q.get('cited') else {'none':'zero','some':'one','all':'two'}[q['qualifiers_preserved']]
    ctr[key]+=1
  assert sum(ctr.values())==1680
  retention[cond]=dict(ctr)
 out['parliament']['retention_by_condition']=retention
 assert len(out['by_model_condition']) == 14
 assert all(x['court_n'] == x['parliament_witness_n'] == 240 for x in out['by_model_condition'])
 assert sum(x['court_verified'] for x in out['by_model_condition']) == len(cv)
 assert sum(x['parliament_verified'] for x in out['by_model_condition']) == out['parliament']['verified']
 (R/'results/scoring_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 t=[r'\subsection{Complete scorer accounting}\label{app:scorer-accounting}',r'The acceptance flag is a pipeline check, not human accuracy. Court has 3,339 accepted semantic labels and 21 pattern fallbacks; Parliament has 3,094 accepted witness labels and 266 pattern fallbacks. In particular, 76 accepted Court labels are SILENT and 1,566 accepted Parliament labels are uncited: these states do not require supporting quotes in the frozen checker. Quote checks therefore must not be reported as uniformly verified excerpts. No human confusion matrix is available yet.',r'\begin{table}[htbp]\centering\scriptsize',r'\caption{Complete seven-model scorer acceptance/fallback counts. Each Court row contains 240 rulings; each Parliament row contains 240 witness slots in 120 reports. V/O denote Vanilla/Obligation. Final endpoints retain the documented pattern fallback.}',r'\begin{tabular}{l c rr rr}\toprule Model & Prompt & \multicolumn{2}{c}{Court} & \multicolumn{2}{c}{Parliament witnesses} \\ & & Accepted & Fallback & Accepted & Fallback \\\midrule']
 for x in out['by_model_condition']:
  name={'claude-opus-5':'Claude Opus 5','gpt-5.6-sol':'GPT-5.6 Sol','gpt-5.6-terra':'GPT-5.6 Terra','gpt-5.6-luna':'GPT-5.6 Luna','glm-5.3':'GLM-5.3','glm-5.3-flash':'GLM-5.3-Flash','qwen3.5-35b-a3b':'Qwen3.5-35B-A3B'}[x['model']]
  t.append(' & '.join(map(str,[esc(name),x['condition'][0].upper(),x['court_verified'],x['court_fallback'],x['parliament_verified'],x['parliament_fallback']])) + r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}',r'The 4.6\% figure compares pattern labels against the final (semantic-or-fallback) endpoint. Restricting to accepted semantic labels gives 133/3,339 agreement (4.0\%). The two matrices below make this distinction explicit; the additional 21 diagonal entries in the final-endpoint matrix arise mechanically from fallback.',r'\begin{table}[htbp]\centering\small',r'\caption{Pattern versus accepted semantic labels only ($n=3,339$). The final-endpoint matrix above includes 21 additional pattern fallbacks. Neither matrix measures accuracy against humans.}',r'\begin{tabular}{lrrrr}\toprule Pattern & Adopted & Corrected & Truth stated & Silent \\\midrule'])
 for x in labs:t.append(x.replace('_',r'\_')+' & '+' & '.join(str(raw_semantic[x,y]) for y in labs)+r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}'])
 t.extend([r'\begin{table}[htbp]\centering\small',r'\caption{Parliament qualifier-retention labels over all 1,680 witness slots per prompt. Among accepted cited labels, none/some/all correspond to 0/1/2 retained constructed limitations. Unverified slots are shown separately: the frozen three-category pattern fallback cannot distinguish one from two. This semantic-only breakdown does not replace the final-endpoint denominator or constitute human validation.}',r'\begin{tabular}{lrrrrr}\toprule Prompt & Uncited & Cited: 0 & Cited: 1 & Cited: 2 & Unverified \\\midrule'])
 for cond in ['vanilla','obligation']:
  t.append(cond.title()+' & '+' & '.join(str(retention[cond].get(k,0)) for k in ['uncited','zero','one','two','unverified'])+r' \\')
 t.extend([r'\bottomrule\end{tabular}\end{table}'])
 tex='\n'.join(t)+'\n'
 tex=tex.replace('\n\\begin{table}','\n\n\\begin{table}').replace('\\end{table}\n','\\end{table}\n\n')
 (R/'build/tables/app_scoring_audit.tex').write_text(tex)
 print(json.dumps({'court':out['court'],'parliament':out['parliament'],'rows':len(out['by_model_condition'])}))
if __name__=='__main__':main()
