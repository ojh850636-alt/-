from __future__ import annotations
import json,re
from pathlib import Path

TARGETS=['MLP_ALL','EARLY_BAND','EARLY_BAND&MLP_ALL']
LABELS=['TARGET','RANDOM_0','RANDOM_1','RANDOM_2','ENERGY_TOP']
MODES=['ablation','only']
SPLITS=['CONFIRMATION','GENERATOR_HOLDOUT']

def load_json(p:Path):
    return json.loads(p.read_text(encoding='utf-8'))

def split_mean(rows,split):
    xs=[float(x['mean_logprob']) for x in rows if x['split']==split]
    assert len(xs)==64,(split,len(xs))
    return sum(xs)/len(xs)

files=list(Path('collected').rglob('result.json'))
rows={}
orig=0; recovery=0
for p in files:
    d=load_json(p)
    assert d['schema']=='LUCIA_AA_R22542_C39_CAUSAL_STAGE2_ATOMIC_CONFIG_V1'
    k=(d['target'],d['label'],d['mode'])
    assert k not in rows,('duplicate',k,str(p))
    rows[k]=d
    if 'recovery' in p.parts: recovery+=1
    else: orig+=1
expected={(t,l,m) for t in TARGETS for l in LABELS for m in MODES}
assert set(rows)==expected,(len(rows),sorted(expected-set(rows)),sorted(set(rows)-expected))
assert orig==29,(orig,recovery)
assert recovery==1,(orig,recovery)
assert rows[('MLP_ALL','TARGET','ablation')]['source']=='AUTHORITATIVE_ESCROW_RUN_31484392389'

base=load_json(Path('baseline/base/SHARD.json'))
full=load_json(Path('baseline/full/SHARD.json'))
assert base['condition']=='BASE' and full['condition']=='FULL'
assert base['schema']=='LUCIA_AA_R22542_C39_TEACHER_FAST_SHARD_V1'
assert full['schema']=='LUCIA_AA_R22542_C39_TEACHER_FAST_SHARD_V1'

out=Path('public');out.mkdir(exist_ok=True)
metrics=[]
for t in TARGETS:
    cfg=[]
    for l in LABELS:
        for m in MODES:
            d=rows[(t,l,m)]
            cfg.append({'label':l,'mode':m,'site_count':d['site_count'],'site_names_sha256':d['site_names_sha256'],'case_results':d['case_results']})
    first=rows[(t,'TARGET','ablation')]
    target_metrics={}
    pass_split={}
    for sp in SPLITS:
        bm=split_mean(base['case_results'],sp); fm=split_mean(full['case_results'],sp)
        ta=split_mean(rows[(t,'TARGET','ablation')]['case_results'],sp)
        to=split_mean(rows[(t,'TARGET','only')]['case_results'],sp)
        nec=fm-ta; suf=to-bm
        random=[]
        beats_all=True
        for i in range(3):
            lab=f'RANDOM_{i}'
            ra=split_mean(rows[(t,lab,'ablation')]['case_results'],sp)
            ro=split_mean(rows[(t,lab,'only')]['case_results'],sp)
            rnec=fm-ra; rsuf=ro-bm
            random.append({'label':lab,'necessity':rnec,'sufficiency':rsuf})
            beats_all = beats_all and (nec>rnec) and (suf>rsuf)
        ea=split_mean(rows[(t,'ENERGY_TOP','ablation')]['case_results'],sp)
        eo=split_mean(rows[(t,'ENERGY_TOP','only')]['case_results'],sp)
        enec=fm-ea; esuf=eo-bm
        ok=(nec>0.0 and suf>0.0 and beats_all)
        pass_split[sp]=ok
        target_metrics[sp]={
            'base_mean_logprob':bm,'full_mean_logprob':fm,
            'target_ablation_mean_logprob':ta,'target_only_mean_logprob':to,
            'necessity':nec,'sufficiency':suf,
            'random_controls':random,'beats_all_three_randoms_on_both_effects':beats_all,
            'energy_top_control':{'necessity':enec,'sufficiency':esuf,'positive_credit':False},
            'pass':ok,
        }
    passed=all(pass_split.values())
    z={
        'schema':'LUCIA_AA_R22542_C39_CAUSAL_STAGE2_RECOVERED_LOCKED_V1',
        'target':t,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389',
        'behavior_authority_run':31488582355,
        'stage1_main_run':31501795482,'stage1_o_proj_recovery_run':31503834917,
        'stage2_atomic_run':31507487963,'stage2_missing_config_recovery_run':31532536631,
        'prior_serial_stage2_run_credit':0,'failed_transport_attempts_credit':0,
        'locked_splits':SPLITS,'cases':128,'target_site_count':first['target_site_count'],
        'target_site_names_sha256':first['target_site_names_sha256'],'matched_random_controls':3,
        'energy_large_control':True,'configs':cfg,'metrics':target_metrics,'stage2_pass':passed,
        'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'raw_tactics_saved':False,
        'hf_model_redownloads':0,'claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'
    }
    (out/(re.sub(r'[^A-Za-z0-9_.-]+','__',t)+'.json')).write_text(json.dumps(z,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    metrics.append({'target':t,'site_count':first['target_site_count'],'pass':passed,'split_metrics':target_metrics})

passing=[x for x in metrics if x['pass']]
if passing:
    m=min(x['site_count'] for x in passing)
    bounded=sorted([x['target'] for x in passing if x['site_count']==m])
    assert len(bounded)==1,('ambiguous_minimal_survivor',bounded)
    survivor=bounded[0]
    status='GREEN_BOUNDED_SURVIVOR'
else:
    survivor=None;status='RED_NO_STAGE2_SURVIVOR'
aggregate={
    'schema':'LUCIA_AA_R22542_C39_STAGE2_AGGREGATE_V1','status':status,
    'targets':metrics,'passing_targets':[x['target'] for x in passing],
    'bounded_survivor':survivor,
    'selection_rule':'BOTH_LOCKED_SPLITS_REQUIRE_POSITIVE_NECESSITY_AND_SUFFICIENCY_AND_BEAT_ALL_3_MATCHED_RANDOMS_ON_BOTH_EFFECTS__THEN_MIN_SITE_COUNT',
    'energy_top_positive_credit':False,
    'behavior_authority_run':31488582355,'stage2_atomic_run':31507487963,'stage2_missing_config_recovery_run':31532536631,
    'raw_source_downloaded_by_collector':False,'hf_model_redownloads':0
}
(out/'STAGE2_AGGREGATE.json').write_text(json.dumps(aggregate,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(out/'C39_FINAL_TARGET.json').write_text(json.dumps({'schema':'LUCIA_AA_R22542_C39_FINAL_TARGET_V1','target':survivor,'stage2_status':status,'claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'},indent=2,sort_keys=True)+'\n',encoding='utf-8')
(out/'COLLECT_RECEIPT.json').write_text(json.dumps({'schema':'LUCIA_AA_R22542_C39_STAGE2_RECOVERED_COLLECT_RECEIPT_V1','original_atomic_run':31507487963,'missing_config_recovery_run':31532536631,'original_config_count':orig,'recovered_config_count':recovery,'unique_config_count':len(rows),'target_count':3,'behavior_baseline_run':31488582355,'hf_model_redownloads':0,'raw_source_downloaded':False},indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'status':status,'survivor':survivor,'passing':[x['target'] for x in passing]},sort_keys=True))
