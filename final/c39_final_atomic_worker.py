from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path

def load_worker():
    p=Path(__file__).with_name('c39_final_worker.py');spec=importlib.util.spec_from_file_location('c39_final_worker_bound',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
w=load_worker()

def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def semantic():
    target,h,tok,model,ctl,tg=w.load();rows=[r for r in h.build_suite() if r['positive'] and r['split'] in {'CONFIRMATION','GENERATOR_HOLDOUT'}];assert len(rows)==128;ns=set(tg)
    ctl.all(1);full=w.score(model,tok,rows)
    with model.disable_adapter():base=w.score(model,tok,rows)
    ctl.all(1)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=0.0
    ablation=w.score(model,tok,rows)
    ctl.all(0)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=x['scale']
    only=w.score(model,tok,rows)
    dump('public/SEMANTIC_REPLAY.json',{'schema':'LUCIA_AA_R22542_C39_FINAL_FRESH_SEMANTIC_REPLAY_V1','target':target,'target_site_count':len(tg),'target_site_names_sha256':hashlib.sha256('\n'.join(tg).encode()).hexdigest(),'cases':128,'conditions':{'BASE':base,'FULL':full,'TARGET_ABLATION':ablation,'TARGET_ONLY':only},'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'hf_model_redownloads':0,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389'})

def generation(mode):
    assert mode in {'TARGET_ONLY','TARGET_ABLATION'}
    target,h,tok,model,ctl,tg=w.load();rows=h.build_suite();locked=[];ns=set(tg)
    for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):
        locked += sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
    if mode=='TARGET_ONLY':
        ctl.all(0)
        for x in ctl.mods:
            if x['name'] in ns:x['m'].scaling[x['key']]=x['scale']
    else:
        ctl.all(1)
        for x in ctl.mods:
            if x['name'] in ns:x['m'].scaling[x['key']]=0.0
    rec=w.generate(h,model,tok,locked,mode,42,8,4);Path('private').mkdir(exist_ok=True);Path(f'private/{mode}.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in rec),encoding='utf-8')
    dump('public/GEN_RECEIPT.json',{'schema':'LUCIA_AA_R22542_C39_FINAL_CAUSAL_GENERATION_RECEIPT_V1','target':target,'condition':mode,'target_site_count':len(tg),'locked_cases':48,'records':len(rec),'seed':42,'k':8,'raw_tactics_exported_public':False,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','hf_model_redownloads':0})

if __name__=='__main__':
    m=os.environ['C39_ATOMIC_FINAL_MODE']; semantic() if m=='SEMANTIC' else generation(m)
