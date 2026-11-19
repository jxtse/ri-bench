"""Offline acceptance checks for review revision; never invokes an LLM."""
import importlib.util,json,sys
from pathlib import Path
import pytest
R=Path(__file__).resolve().parents[2]
def module(name):
 spec=importlib.util.spec_from_file_location(name,R/'tools'/f'{name}.py')
 assert spec is not None and spec.loader is not None
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def response(payload,finish='stop'):
 return {'model':'gemini-3.8-flash','choices':[{'finish_reason':finish,'message':{'content':json.dumps(payload)}}]}

def verdict():
 return {'reference_kind':'committee_conclusion','verdict':'agree','reference_quote':'The committee recommends stronger disclosure.','report_quote':'We recommend stronger disclosure requirements.','reason':'Same direction.'}

def test_conclusion_parser_requires_complete_matching_evidence():
 m=module('conclusion_alignment_audit_v2');v=verdict();ref=v['reference_quote'];rep=v['report_quote']
 assert m.verify_response(response(v),ref,rep)==v
 other=response(v);other['model']='another-model'
 with pytest.raises(ValueError,match='returned model'):m.verify_response(other,ref,rep)
 with pytest.raises(ValueError,match='incomplete'):m.verify_response(response(v,'length'),ref,rep)
 with pytest.raises(ValueError,match='reference quote'):m.verify_response(response(v),'Unrelated reference.',rep)
 with pytest.raises(ValueError,match='report quote'):m.verify_response(response(v),ref,'Unrelated report.')
 v['reference_kind']='government_response'
 with pytest.raises(ValueError,match='noncommittee'):m.verify_response(response(v),ref,rep)
 v['verdict']='not_comparable';v['report_quote']=''
 assert m.verify_response(response(v),ref,rep)['verdict']=='not_comparable'

def test_conclusion_plan_is_balanced_and_immutable_shape():
 f=R/'results/parliament_conclusion_audit_manifest.json'
 if not f.exists():pytest.skip('optional exploratory audit ledger is not included')
 p=json.loads(f.read_text());assert len(p['selected_cases'])==20 and len(p['rows'])==280
 from collections import Counter
 assert set(Counter(r['case_id'] for r in p['rows']).values())=={14}
 assert len({(r['case_id'],r['model'],r['condition']) for r in p['rows']})==280

def test_scoring_accounting_denominators_and_distinct_matrices():
 s=json.loads((R/'results/scoring_audit.json').read_text());a=s['court'];b=s['parliament']
 assert a['n']==b['n']==3360
 assert a['verified']+a['fallback']==a['n']
 assert b['verified']+b['fallback']==b['n']
 assert a['pattern_final_diagonal']==154 and a['pattern_semantic_verified_diagonal']==133
 assert sum(map(sum,a['pattern_final_matrix']))==3360
 assert sum(map(sum,a['pattern_semantic_verified_matrix']))==3339
 assert len(s['by_model_condition'])==14
 assert all(r['court_n']==r['parliament_witness_n']==240 for r in s['by_model_condition'])
 assert sum(r['court_verified'] for r in s['by_model_condition'])==3339
 assert sum(r['parliament_verified'] for r in s['by_model_condition'])==3094

def test_every_case_appears_once_in_printed_appendix():
 f=R/'build/tables/app_percase.tex'
 if not f.exists():pytest.skip('per-case appendix is generated on demand by tools/make_percase_appendix.py')
 cases=json.loads((R/'results/per_case.json').read_text());text=f.read_text()
 assert len(cases)==477
 assert text.count('\\begin{longtable}')==5
 for row in cases:assert text.count('\\path|'+row['case_id']+'|')==1
 assert 'undefined (0/0)' in text

def test_exact_prompt_audit_has_all_selected_roles():
 p=json.loads((R/'results/prompt_template_audit.json').read_text())
 assert p['n']==len(p['checked'])==76
 assert {x['role'] for x in p['checked']}=={'area_chair','program_chair','judge','chair'}
 assert {x['condition'] for x in p['checked']}=={'vanilla','obligation'}

def test_completed_conclusion_audit_accounts_for_every_selected_report():
 s=json.loads((R/'results/parliament_conclusion_audit_summary.json').read_text())
 rows=[json.loads(l) for l in (R/'results/parliament_conclusion_audit.jsonl').read_text().splitlines()]
 assert s['complete_dispositions'] and s['pending_retry']==0 and s['human_validated'] is False
 assert len(rows)==s['n_reports']==280
 assert len({r['episode_id'] for r in rows})==280
 assert sum(r['n_attempts'] for r in rows)==s['n_attempts']==348
 assert all(1<=r['n_attempts']<=2 for r in rows)
 assert sum(r['status']=='evidence_verified' for r in rows)==267
 assert sum(r['status']=='unresolved' for r in rows)==13
 for c in ['vanilla','obligation','all']:
  d=s['conditions'][c]
  assert sum(d['verdict_counts'].values())==d['n_verified']
  assert d['n_verified']+d['n_unresolved']==d['n']
 a=s['conditions']['all']['agreement_among_comparable_conclusions']
 assert (a['k'],a['n'])==(181,197)
 assert a['ci95_case_cluster'][0]<a['rate']<a['ci95_case_cluster'][1]
 b=s['missing_label_bounds']
 assert (b['n_cases'],b['n_reports'],b['agree'],b['unresolved'])==(15,210,181,13)
 assert b['lower_all_unresolved_nonagree']==181/210
 assert b['upper_all_unresolved_agree']==194/210
