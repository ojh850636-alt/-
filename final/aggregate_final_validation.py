from __future__ import annotations
import json
from pathlib import Path

def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def mean(rows,sp):
    x=[float(r['mean_logprob']) for r in rows if r['split']==sp];assert len(x)==64,(sp,len(x));return sum(x)/len(x)
def success_by_split(d):
    out={}
    for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):
        rr=[x for x in d['case_results'] if x['split']==sp];assert len(rr)==12,(sp,len(rr));out[sp]=sum(bool(x['success']) for x in rr)
    return out
sem=J('inputs/semantic/SEMANTIC_REPLAY.json');only=J('inputs/only/LEAN_SHARD.json');abl=J('inputs/ablation/LEAN_SHARD.json');full42=J('inputs/full42/LEAN_SHARD.json')
target=sem['target'];assert target in {'MLP_ALL','EARLY_BAND','EARLY_BAND&MLP_ALL'}
sm={};semantic_pass=True
for sp in ('CONFIRMATION','GENERATOR_HOLDOUT'):
    b=mean(sem['conditions']['BASE'],sp);f=mean(sem['conditions']['FULL'],sp);a=mean(sem['conditions']['TARGET_ABLATION'],sp);o=mean(sem['conditions']['TARGET_ONLY'],sp)
    nec=f-a;suf=o-b;ok=nec>0 and suf>0;semantic_pass &= ok
    sm[sp]={'base_mean_logprob':b,'full_mean_logprob':f,'target_ablation_mean_logprob':a,'target_only_mean_logprob':o,'necessity':nec,'sufficiency':suf,'pass':ok}
co=success_by_split(only);ca=success_by_split(abl);cf=success_by_split(full42)
gen_split={};generation_pass=True
for sp in ('CONFIRMATION','GENERATOR_HOLDOUT'):
    ok=co[sp]>0 and ca[sp]<cf[sp];generation_pass &= ok;gen_split[sp]={'target_only_success':co[sp],'target_ablation_success':ca[sp],'full42_success':cf[sp],'pass':ok}
negative_pass=(co['NEGATIVE_UNPROVABLE']==0 and ca['NEGATIVE_UNPROVABLE']==0);generation_pass &= negative_pass
terminal=bool(semantic_pass and generation_pass)
out={'schema':'LUCIA_AA_R22542_C39_FINAL_VALIDATION_V1','target':target,'target_site_count':sem['target_site_count'],'semantic_replay':sm,'semantic_replay_pass':semantic_pass,'lean_generation':{'by_split':gen_split,'target_only_all_splits':co,'target_ablation_all_splits':ca,'authoritative_full42_all_splits':cf,'negative_pass':negative_pass},'lean_generation_pass':generation_pass,'terminal_validation_pass':terminal,'verdict':'GREEN_E3_CAUSAL_LOCALIZATION' if terminal else 'RED_FINAL_CAUSAL_VALIDATION','claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD','E4_increment':0,'E5_increment':0,'external_algorithm_increment':0,'student_increment':0,'promotion_increment':0,'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'raw_tactics_saved':False,'hf_model_redownloads':0}
Path('public').mkdir(exist_ok=True);Path('public/FINAL_VALIDATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps({'target':target,'terminal_validation_pass':terminal,'verdict':out['verdict']},sort_keys=True))
