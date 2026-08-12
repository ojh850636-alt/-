from __future__ import annotations
import json,re
from pathlib import Path
LABELS=['TARGET','RANDOM_0','RANDOM_1','RANDOM_2','ENERGY_TOP']
MODES=['ablation','only']
TARGETS=['MLP_ALL','EARLY_BAND','EARLY_BAND&MLP_ALL']
rows={}
for p in Path('collected').rglob('result.json'):
    d=json.load(open(p,encoding='utf-8'))
    assert d['schema']=='LUCIA_AA_R22542_C39_CAUSAL_STAGE2_ATOMIC_CONFIG_V1'
    k=(d['target'],d['label'],d['mode'])
    assert k not in rows,('duplicate',k,p)
    rows[k]=d
expected={(t,l,m) for t in TARGETS for l in LABELS for m in MODES}
assert set(rows)==expected,(len(rows),sorted(expected-set(rows)),sorted(set(rows)-expected))
out=Path('public');out.mkdir(exist_ok=True)
for t in TARGETS:
    cfg=[]
    for l in LABELS:
        for m in MODES:
            d=rows[(t,l,m)]
            cfg.append({'label':l,'mode':m,'site_count':d['site_count'],'site_names_sha256':d['site_names_sha256'],'case_results':d['case_results']})
    first=rows[(t,'TARGET','ablation')]
    z={'schema':'LUCIA_AA_R22542_C39_CAUSAL_STAGE2_LOCKED_V1','target':t,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','behavior_authority_run':31488582355,'stage1_main_run':31501795482,'stage1_o_proj_recovery_run':31503834917,'stage2_original_run':31507487963,'stage2_missing_config_recovery_run':31558693277,'locked_splits':['CONFIRMATION','GENERATOR_HOLDOUT'],'cases':128,'target_site_count':first['target_site_count'],'target_site_names_sha256':first['target_site_names_sha256'],'matched_random_controls':3,'energy_large_control':True,'configs':cfg,'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'raw_tactics_saved':False,'hf_model_redownloads':0,'claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'}
    (out/(re.sub(r'[^A-Za-z0-9_.-]+','__',t)+'.json')).write_text(json.dumps(z,indent=2,sort_keys=True)+'\n')
(out/'COLLECT_RECEIPT.json').write_text(json.dumps({'schema':'LUCIA_AA_R22542_C39_ATOMIC_STAGE2_SPLIT_RUN_COLLECT_RECEIPT_V1','original_run':31507487963,'recovery_run':31558693277,'atomic_config_count':30,'unique_config_count':len(rows),'target_count':3,'hf_model_redownloads':0,'raw_source_downloaded':False},indent=2,sort_keys=True)+'\n')
print('PASS 29+1 atomic configs -> 3 targets')
