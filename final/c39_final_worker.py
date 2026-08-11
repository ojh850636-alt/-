from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def target_names(mods,n,target):
    a=max(1,n//3)
    def ok(x,c):
        if c=='MLP_ALL':return x['proj'] in {'gate_proj','up_proj','down_proj'}
        if c=='EARLY_BAND':return x['layer']<a
        raise KeyError(c)
    return sorted(x['name'] for x in mods if all(ok(x,c) for c in target.split('&')))

def score(model,tok,rows,bs=8):
    tok.padding_side='left';pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;out=[];model.eval()
    with torch.inference_mode():
        for st in range(0,len(rows),bs):
            rr=rows[st:st+bs];seq=[];cl=[]
            for r in rr:
                p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids;seq.append(p+c);cl.append(len(c))
            ml=max(map(len,seq));mc=max(cl);ii=torch.full((len(rr),ml),pad,dtype=torch.long,device=model.device);am=torch.zeros_like(ii)
            for i,s in enumerate(seq):ii[i,-len(s):]=torch.tensor(s,dtype=torch.long,device=model.device);am[i,-len(s):]=1
            lp=torch.log_softmax(model(input_ids=ii,attention_mask=am,use_cache=False,logits_to_keep=mc+1).logits.float(),dim=-1)
            for i,(r,nc) in enumerate(zip(rr,cl)):
                v=lp[i,-nc-1:-1,:].gather(-1,ii[i,-nc:].unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'token_count':nc})
    return out

def generate(h,model,tok,rows,condition,seed=42,k=8,prompt_batch=4):
    torch.manual_seed(seed);tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;model.eval();out=[]
    with torch.inference_mode():
        for st in range(0,len(rows),prompt_batch):
            rr=rows[st:st+prompt_batch];e=tok([r['prompt'] for r in rr],return_tensors='pt',padding=True,add_special_tokens=False,truncation=True,max_length=768);e={a:b.to(model.device) for a,b in e.items()};n=e['input_ids'].shape[1]
            g=model.generate(**e,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=k,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            for j,r in enumerate(rr):
                for rank in range(k):
                    raw=tok.decode(g[j*k+rank,n:],skip_special_tokens=True);t,stt=h.sanitize(raw);out.append({'condition':condition,'seed':seed,'case_id':r['case_id'],'split':r['split'],'family':r['family'],'rank':rank,'status':stt,'tactic':t,'tactic_sha256':hashlib.sha256(t.encode()).hexdigest() if t else None})
    return out

def load():
    target=json.loads(Path('final/C39_FINAL_TARGET.json').read_text(encoding='utf-8'))['target'];assert target in {'MLP_ALL','EARLY_BAND','EARLY_BAND&MLP_ALL'}
    h=loadmod('final/c39_behavior_pinned.py','c39_behavior_final_helper');tok=AutoTokenizer.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
    base=AutoModelForCausalLM.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True);model=PeftModel.from_pretrained(base,'source/adapter',local_files_only=True,is_trainable=False);ctl=h.Controller(model);assert len(ctl.mods)==96
    tg=target_names(ctl.mods,ctl.nl,target);assert tg;return target,h,tok,model,ctl,tg

def all_source_phase():
    target,h,tok,model,ctl,tg=load();ns=set(tg);rows=h.build_suite();semantic_rows=[r for r in rows if r['positive'] and r['split'] in {'CONFIRMATION','GENERATOR_HOLDOUT'}];assert len(semantic_rows)==128
    ctl.all(1);full=score(model,tok,semantic_rows)
    with model.disable_adapter():base=score(model,tok,semantic_rows)
    ctl.all(1)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=0.0
    ablation=score(model,tok,semantic_rows)
    ctl.all(0)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=x['scale']
    only=score(model,tok,semantic_rows)
    dump('public/SEMANTIC_REPLAY.json',{'schema':'LUCIA_AA_R22542_C39_FINAL_FRESH_SEMANTIC_REPLAY_V1','target':target,'target_site_count':len(tg),'target_site_names_sha256':hashlib.sha256('\n'.join(tg).encode()).hexdigest(),'cases':128,'conditions':{'BASE':base,'FULL':full,'TARGET_ABLATION':ablation,'TARGET_ONLY':only},'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'hf_model_redownloads':0,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389'})
    locked=[]
    for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):locked += sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
    Path('private').mkdir(exist_ok=True)
    ctl.all(0)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=x['scale']
    rec=generate(h,model,tok,locked,'TARGET_ONLY',42,8,4);Path('private/TARGET_ONLY.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in rec),encoding='utf-8')
    dump('public/TARGET_ONLY_GEN_RECEIPT.json',{'schema':'LUCIA_AA_R22542_C39_FINAL_CAUSAL_GENERATION_RECEIPT_V1','target':target,'condition':'TARGET_ONLY','target_site_count':len(tg),'locked_cases':48,'records':len(rec),'seed':42,'k':8,'raw_tactics_exported_public':False,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','hf_model_redownloads':0})
    ctl.all(1)
    for x in ctl.mods:
        if x['name'] in ns:x['m'].scaling[x['key']]=0.0
    rec=generate(h,model,tok,locked,'TARGET_ABLATION',42,8,4);Path('private/TARGET_ABLATION.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in rec),encoding='utf-8')
    dump('public/TARGET_ABLATION_GEN_RECEIPT.json',{'schema':'LUCIA_AA_R22542_C39_FINAL_CAUSAL_GENERATION_RECEIPT_V1','target':target,'condition':'TARGET_ABLATION','target_site_count':len(tg),'locked_cases':48,'records':len(rec),'seed':42,'k':8,'raw_tactics_exported_public':False,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','hf_model_redownloads':0})

if __name__=='__main__':
    assert os.environ.get('C39_FINAL_MODE')=='all_source_phase';all_source_phase()
