from __future__ import annotations
import argparse,contextlib,hashlib,json
from pathlib import Path

def dump(p,o): Path(p).write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def target_names(ctl):
    boundary=max(1,ctl.nl//3)
    return sorted(x['name'] for x in ctl.mods if x['layer']<boundary and x['proj'] in {'gate_proj','up_proj','down_proj'})
def apply_target(ctl,mode):
    names=target_names(ctl); ns=set(names)
    if mode=='TARGET_ABLATION':
        ctl.all(1.0)
        for x in ctl.mods:
            if x['name'] in ns: x['m'].scaling[x['key']]=0.0
    elif mode=='TARGET_ONLY':
        ctl.all(0.0)
        for x in ctl.mods:
            if x['name'] in ns: x['m'].scaling[x['key']]=x['scale']
    else: raise KeyError(mode)
    return names

def compact_score(model,tok,rows,bs=8):
    import torch
    tok.padding_side='left'; pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; out=[]; model.eval()
    with torch.inference_mode():
      for st in range(0,len(rows),bs):
        rr=rows[st:st+bs]; seq=[]; cl=[]
        for r in rr:
          p=tok(r['prompt'],add_special_tokens=False).input_ids; c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids; seq.append(p+c);cl.append(len(c))
        ml=max(map(len,seq));mc=max(cl);ii=torch.full((len(rr),ml),pad,dtype=torch.long,device=model.device);am=torch.zeros_like(ii)
        for i,s in enumerate(seq): ii[i,-len(s):]=torch.tensor(s,dtype=torch.long,device=model.device);am[i,-len(s):]=1
        lp=torch.log_softmax(model(input_ids=ii,attention_mask=am,use_cache=False,logits_to_keep=mc+1).logits.float(),dim=-1)
        for i,(r,n) in enumerate(zip(rr,cl)):
          v=lp[i,-n-1:-1,:].gather(-1,ii[i,-n:].unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'token_count':n})
    return out

def generation(model,tok,rows,condition,seed=42,k=8,prompt_batch=4):
    import torch
    import laa.r22542.c39_behavior as h
    tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;torch.manual_seed(seed);out=[];model.eval()
    with torch.inference_mode():
      for i in range(0,len(rows),prompt_batch):
        rr=rows[i:i+prompt_batch];e=tok([r['prompt'] for r in rr],return_tensors='pt',padding=True,add_special_tokens=False,truncation=True,max_length=768);e={a:b.to(model.device) for a,b in e.items()};n=e['input_ids'].shape[1]
        g=model.generate(**e,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=k,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        for j,r in enumerate(rr):
          for rank in range(k):
            raw=tok.decode(g[j*k+rank,n:],skip_special_tokens=True);t,st=h.sanitize(raw);out.append({'condition':condition,'seed':seed,'case_id':r['case_id'],'split':r['split'],'family':r['family'],'rank':rank,'status':st,'tactic':t,'tactic_sha256':hashlib.sha256(t.encode()).hexdigest() if t else None})
    return out

def load(base,adapter):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    import laa.r22542.c39_behavior as h
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
    b=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True);m=PeftModel.from_pretrained(b,adapter,local_files_only=True,is_trainable=False);ctl=h.Controller(m);assert len(ctl.mods)==96; names=target_names(ctl);assert len(names)==24;return m,tok,ctl,h,names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['semantic','generation'],required=True);ap.add_argument('--condition',choices=['TARGET_ABLATION','TARGET_ONLY']);ap.add_argument('--base',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--private');a=ap.parse_args()
    m,tok,ctl,h,names=load(a.base,a.adapter);site_sha=hashlib.sha256('\n'.join(names).encode()).hexdigest()
    if a.mode=='semantic':
      rows=[r for r in h.build_suite() if r['positive'] and r['split'] in {'CONFIRMATION','GENERATOR_HOLDOUT'}];assert len(rows)==128;results={}
      for cond in ['TARGET_ABLATION','TARGET_ONLY']:
        apply_target(ctl,cond);results[cond]=compact_score(m,tok,rows)
      dump(a.out,{'schema':'LUCIA_AA_R22542_C39_FINAL_SEMANTIC_REPLAY_V1','target':'EARLY_BAND&MLP_ALL','site_count':24,'site_names_sha256':site_sha,'locked_splits':['CONFIRMATION','GENERATOR_HOLDOUT'],'results':results,'fresh_process':True,'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'hf_model_redownloads':0})
    else:
      assert a.condition and a.private;apply_target(ctl,a.condition);rows=[]
      for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'): rows+=sorted([r for r in h.build_suite() if r['split']==sp],key=lambda x:x['case_id'])[:12]
      rec=generation(m,tok,rows,a.condition);Path(a.private).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in rec),encoding='utf-8');dump(a.out,{'schema':'LUCIA_AA_R22542_C39_FINAL_CAUSAL_GENERATION_RECEIPT_V1','target':'EARLY_BAND&MLP_ALL','site_count':24,'site_names_sha256':site_sha,'condition':a.condition,'seed':42,'locked_cases':48,'records':len(rec),'raw_tactics_exported_public':False})
if __name__=='__main__': main()
