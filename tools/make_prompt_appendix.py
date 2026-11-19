#!/usr/bin/env python3
"""Publish exact static subject prompts; case-specific file lists remain in invocations."""
import hashlib,importlib.util,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen_prompts',R/'src/ribench_agentic/prompts.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def esc(s):
 return ''.join({'–':'--','—':'---','’':"'",'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','<':r'\textless{}','>':r'\textgreater{}'}.get(c,c) for c in s)
def main():
 out=[r'\subsection{Exact static subject prompts}\label{app:full-prompts}',r'The following strings are generated from the role-template module. Vanilla uses the role sentence alone as the system message; Obligation appends the listed duties after two newlines. The user message is the task text, then ``Files (read them using the available tools):'', the per-case manifest as \texttt{- path: description} lines, then ``Output format:'' and the trailer instruction shown below. The same format is used across conditions. File lists and dependency artifacts vary by episode and are recorded in its invocation; a manifest placeholder below is explanatory, not text shown to subjects. These are the full static subject templates, not a reproduction of the coding-agent SDK tool schema.',]
 for role in ['area_chair','program_chair','judge','chair']:
  title={'area_chair':'Area chair','program_chair':'Program chair','judge':'Judge','chair':'Committee chair'}[role]
  out += [r'\paragraph{'+title+r'.}',r'\begin{quote}\small',r'\textbf{System role:} '+esc(p.ROLES[role])+r'\par',r'\textbf{Obligation-only addition:}\par']
  for line in p.DUTIES[role].splitlines():out.append(esc(line)+r'\par')
  out += [r'\textbf{User task:} '+esc(p.TASKS[role])+r'\par',r'\textbf{Output format:} '+esc(p.OUTPUTS[role]),r'\end{quote}']
 (R/'build/tables/app_prompts.tex').write_text('\n'.join(out)+'\n')
 checked=[]
 for scenario in ['court','open_review_v2','open_review_v3','parliament']:
  rows=[json.loads(l) for l in (R/f'results/{scenario}.jsonl').read_text().splitlines()];seen=set()
  for row in rows:
   key=row['model'],row['condition']
   if key in seen:continue
   seen.add(key);ep=R/'runs/agentic'/row['batch']/'episodes'/row['episode_id']
   for f in sorted(ep.rglob('invocation.json')):
    inv=json.loads(f.read_text());role=inv['role'];expected=p.ROLES[role]+('\n\n'+p.DUTIES[role] if row['condition']=='obligation' else '')
    assert inv['system_prompt']==expected,(scenario,key,role,'system mismatch')
    assert inv['prompt'].startswith(p.TASKS[role]+'\n\nFiles (read them using the available tools):\n')
    assert inv['prompt'].endswith('Output format:\n'+p.OUTPUTS[role])
    checked.append({'scenario':scenario,'model':row['model'],'condition':row['condition'],'episode_id':row['episode_id'],'role':role,'invocation_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 evidence={'scope':'one frozen episode per scenario/model/prompt; all subject roles in selected episodes','template_sha256':hashlib.sha256((R/'src/ribench_agentic/prompts.py').read_bytes()).hexdigest(),'n':len(checked),'checked':checked}
 (R/'results/prompt_template_audit.json').write_text(json.dumps(evidence,indent=2)+'\n');print('verified frozen invocation samples',len(checked))
if __name__=='__main__':main()
