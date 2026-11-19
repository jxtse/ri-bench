#!/usr/bin/env python3
"""Reverify exploratory conclusion labels; fail closed until all reports have terminal dispositions."""
import argparse,collections,hashlib,importlib.util,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
import conclusion_alignment_audit_v2 as audit
from revision_statistics import cluster_ratio
NAMES={'vanilla':'Vanilla','obligation':'Obligation'}
def load(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 manifest=load(audit.OUT/'manifest.json');attempts=[json.loads(l) for l in (audit.OUT/'attempts.jsonl').read_text().splitlines()]
 planned={r['episode_id']:r for r in manifest['rows']};assert len(planned)==manifest['n_reports']==280
 groups=collections.defaultdict(list)
 for x in attempts:
  assert x['episode_id'] in planned
  groups[x['episode_id']].append(x)
 results=[];errors=collections.Counter();pending=[]
 for uid,r in planned.items():
  case=load(R/f'data/cases/parliament/{r["case_id"]}.json');reference=case['ground_truth_conclusion'];report=load(R/f'runs/agentic/{r["batch"]}/episodes/{uid}/episode.json')['final_text']
  assert audit.sha(reference)==r['reference_sha256'] and audit.sha(report)==r['report_sha256']
  good=[]
  for a in groups[uid]:
   if a['status']=='evidence_verified':
    j=audit.verify_response(a['response'],reference,report);assert j==a['judgment'];good.append(a)
   else:errors[a.get('error','unknown')]+=1
  base={k:r[k] for k in ['episode_id','case_id','model','condition','reference_sha256','report_sha256','publication_type','reference_provenance']}
  if good:
   a=sorted(good,key=lambda x:x['started_at'])[0];base.update(status='evidence_verified',judgment=a['judgment'],selected_attempt=a['attempt'],n_attempts=len(groups[uid]))
  else:
   base.update(status='unresolved',n_attempts=len(groups[uid]),errors=[x.get('error') for x in groups[uid]])
   if len(groups[uid])<2:pending.append(uid)
  results.append(base)
 if pending and not args.preview:raise SystemExit(f'{len(pending)} reports still eligible for technical-failure retry; no final artifacts written')
 summary={'design':manifest['design'],'scope':'20-case stratified-by-model/prompt exploratory sample; not the full 1680 reports','seed':manifest['seed'],'n_reports':280,'n_cases':20,'n_attempts':len(attempts),'complete_dispositions':not pending,'pending_retry':len(pending),'n_evidence_verified':sum(r['status']=='evidence_verified' for r in results),'n_unresolved':sum(r['status']=='unresolved' for r in results),'evaluator':audit.MODEL,'human_validated':False,'failure_counts':dict(errors),'ledger_sha256':hashlib.sha256((audit.OUT/'attempts.jsonl').read_bytes()).hexdigest(),'prompt_sha256':manifest['prompt_sha256'],'conditions':{},'technical_retry':'At most two attempts. Initial max_tokens=2048; technical retry budget increased to 4096; no valid judgment selectively rerun.','excluded_v1':'Initial 772-record partial/salvaged-JSON audit discarded; retained in original QA ledger, not pooled.'}
 for cond in ['vanilla','obligation','all']:
  rows=[r for r in results if cond=='all' or r['condition']==cond];good=[r for r in rows if r['status']=='evidence_verified'];cnt=collections.Counter(r['judgment']['verdict'] for r in good);records=[]
  for r in rows:
   j=r.get('judgment',{});eligible=j.get('verdict') in ['agree','disagree','mixed']
   records.append((r['case_id'],int(j.get('verdict')=='agree'),int(eligible),int(r['status']=='evidence_verified'),1))
  metric=cluster_ratio(manifest['selected_cases'],records)
  summary['conditions'][cond]={'n':len(rows),'n_verified':len(good),'n_unresolved':len(rows)-len(good),'verdict_counts':dict(cnt),'reference_kind_counts':dict(collections.Counter(r['judgment']['reference_kind'] for r in good)),'agreement_among_comparable_conclusions':metric,'denominator':'agree / (agree + disagree + mixed); excludes not_comparable, no_conclusion and unresolved; these counts are reported, never treated as disagreement.'}
 refkind=collections.defaultdict(set)
 for r in results:
  if r['status']=='evidence_verified':refkind[r['case_id']].add(r['judgment']['reference_kind'])
 summary['reference_kind_inconsistent_cases']={k:sorted(v) for k,v in refkind.items() if len(v)>1}
 eligible_cases={k for k,v in refkind.items() if v=={'committee_conclusion'}}
 eligible=[r for r in results if r['case_id'] in eligible_cases]
 agreed=sum(r.get('judgment',{}).get('verdict')=='agree' for r in eligible)
 unresolved=sum(r['status']=='unresolved' for r in eligible)
 summary['missing_label_bounds']={'scope':'sampled inquiries consistently classified as committee-reference; conditional on that unvalidated source classification','n_cases':len(eligible_cases),'n_reports':len(eligible),'agree':agreed,'unresolved':unresolved,'lower_all_unresolved_nonagree':agreed/len(eligible) if eligible else None,'upper_all_unresolved_agree':(agreed+unresolved)/len(eligible) if eligible else None}
 summary['legacy_call_accounting']='772 retained legacy result records are not complete API-call accounting; earlier pilot failures were not all preserved.'
 pop=[load(R/f'data/cases/parliament/{c}.json') for c in sorted({json.loads(l)['case_id'] for l in (R/'results/parliament.jsonl').read_text().splitlines()})]
 summary['population_reference_metadata']={'n_cases':len(pop),'publication_type':dict(collections.Counter(x.get('publication_type') for x in pop)),'provenance':dict(collections.Counter(x.get('gt_source') for x in pop))}
 summary['interpretation']='Automated exploratory classification, not human-calibrated and not a full-population agreement estimate. Within-case source-kind inconsistencies, if present, are listed explicitly; absence of inconsistency is not source validation. The result cannot establish which evidence is inconvenient for an agent.'
 dest=R/'build/conclusion-audit-preview.json' if args.preview else R/'results/parliament_conclusion_audit_summary.json';dest.write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:summary[k] for k in ['n_attempts','complete_dispositions','pending_retry','n_evidence_verified','n_unresolved','conditions','reference_kind_inconsistent_cases']},indent=2))
 if args.preview:return
 (R/'results/parliament_conclusion_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
 (R/'results/parliament_conclusion_audit_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 a=summary['conditions']['all'];v=a['agreement_among_comparable_conclusions'];rate='undefined' if v['rate'] is None else f"{100*v['rate']:.1f}\\%";ci='undefined' if v['ci95_case_cluster'][0] is None else '['+', '.join(f'{100*x:.1f}' for x in v['ci95_case_cluster'])+r']\%'
 t=[r'\section{Exploratory conclusion-alignment audit}\label{app:conclusion-audit}',r'This post hoc audit asks whether the generated report agrees with the stored reference on the central issue. It is not an experiment with new subjects. Twenty of 120 inquiries were sampled with seed 20260914, then all seven models and both prompts were included (280 existing reports). The same Gemini evaluator received complete report/reference text and returned source identity, an agreement label, and contiguous evidence spans. Responses had to finish normally, parse completely, return the requested model ID and pass both applicable span checks. Non-committee references were not compared. At most one technical-failure retry was allowed, with a larger completion budget; all attempts are retained.',f"Of 280 reports, {summary['n_evidence_verified']} have labels passing these checks and {summary['n_unresolved']} remain unresolved after their attempt budgets. Among {v['n']} comparable labeled conclusions, {v['k']} are classified as agreeing ({rate}; case-cluster interval {ci}). The denominator excludes non-comparable references, absent conclusions and unresolved responses; it is not 280 or the full 1,680-report population.",r'\begin{table}[htbp]\centering\small',r'\caption{Exploratory audit dispositions. Agreement is conditional on the first three columns, not all selected reports. No category is human-calibrated.}',r'\begin{tabular}{lrrrrrr}\toprule Prompt & Agree & Disagree & Mixed & \shortstack{No\\conclusion} & \shortstack{Not\\comparable} & Unresolved \\\midrule']
 for cond in ['vanilla','obligation','all']:
  x=summary['conditions'][cond];d=x['verdict_counts'];t.append(('Pooled' if cond=='all' else NAMES[cond])+' & '+' & '.join(str(d.get(k,0)) for k in ['agree','disagree','mixed','no_conclusion','not_comparable'])+' & '+str(x['n_unresolved'])+r' \\')
 stable='Source-kind labels were consistent within each sampled inquiry, but this is not independent validation of source identity.' if not summary['reference_kind_inconsistent_cases'] else f"Source-kind labels differed within {len(summary['reference_kind_inconsistent_cases'])} sampled inquiries, which further limits validity."
 t += [r'\bottomrule\end{tabular}\end{table}',stable+r' Human calibration is absent. The corpus contains 76 Special Reports and 80 LLM-derived reference summaries (overlapping metadata groups); some references summarize government responses rather than committee conclusions. Thus aligned/opposed denotes the stored-reference labels, not whether evidence opposed the conclusion of the agent.']
 if eligible:
  t.append(f"For the {len(eligible_cases)} inquiries classified as committee-reference, {agreed} of {len(eligible)} selected reports are labeled agreeing and {unresolved} remain unresolved. Assigning every unresolved label to non-agreement or agreement gives bounds [{100*agreed/len(eligible):.1f}, {100*(agreed+unresolved)/len(eligible):.1f}]\\%; these are missing-label bounds, not confidence limits, and remain conditional on the source classification.")
 t.append(r'An initial audit that accepted partial JSON was stopped and excluded; 772 result records are retained as invalid diagnostic history. This is not complete API-call accounting, because early pilot failures were not all preserved. The revised audit retains all 348 attempts. Technical failures are missing audit labels, not subject behavior.')
 (R/'manuscript/sections/app_conclusion_audit.tex').write_text('\n'.join(t)+'\n')
if __name__=='__main__':main()
