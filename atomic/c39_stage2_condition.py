from __future__ import annotations
import hashlib,importlib.util,json,os,random,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
T=os.environ['C39_TARGET'];L=os.environ['C39_LABEL'];M=os.environ['C39_MODE'];SEED=22542
LAYER={0:3.00118365879277,1:2.9759737660180234,2:3.0227823473275715,3:5.287264024722158,4:3.177676094744209,5:3.270127721164994,6:3.325560465664014,7:5.872125550417814,8:3.4956401320951898,9:3.6159630030263963,10:3.5724111740162505,11:6.33461433964186,12:3.7572791310656433,13:3.76167513577718,14:3.718853044554014,15:6.719130669655517,16:3.6930369726854426,17:3.608289468314058,18:3.6448871993515772,19:6.717229149893191,20:3.6522505664600744,21:3.738242114197289,22:3.7363648644048584,23:6.3014394060099}
PROJ={'down_proj':19.552549083239654,'gate_proj':33.0784446149292,'k_proj':2.2858834821409126,'o_proj':4.57629806853297,'q_proj':6.673736116112564,'up_proj':31.701904517311462,'v_proj':2.1311841177332367}
def helper():
 p=Path('atomic/c39_behavior_pinned.py');s=importlib.util.spec_from_file_location('c39_behavior_pinned',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def cp(x,c,n):
 a=max(1,n//3);b=max(a+1,(2*n)//3);p=x['proj'];l=x['layer']
 if c=='MLP_ALL':return p in {'gate_proj','up_proj','down_proj'}
 if c=='EARLY_BAND':return l<a
 raise KeyError(c)
def target_names(mods,n):return sorted(x['name'] for x in mods if all(cp(x,c,n) for c in T.split('&')))
def randoms(mods,tg):
 names=sorted(x['name'] for x in mods);n=len(tg);rng=random.Random(SEED+int(hashlib.sha256(T.encode()).hexdigest()[:8],16));out=[];seen={tuple(tg)}
 while len(out)<3:
  s=tuple(sorted(rng.sample(names,n)))
  if s not in seen:seen.add(s);out.append(list(s))
 return out
def energy(mods,n,tg):
 r=sorted(mods,key=lambda x:(LAYER.get(x['layer'],0)*PROJ.get(x['proj'],0),x['name']),reverse=True);s=sorted(x['name'] for x in r[:n]);return sorted(x['name'] for x in r[1:n+1]) if s==tg else s
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
   for i,(r,n) in enumerate(zip(rr,cl)):
    v=lp[i,-n-1:-1,:].gather(-1,ii[i,-n:].unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'token_count':n})
 return out
def main():
 h=helper();rows=[r for r in h.build_suite() if r['positive'] and r['split'] in {'CONFIRMATION','GENERATOR_HOLDOUT'}];assert len(rows)==128
 tok=AutoTokenizer.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True);model=PeftModel.from_pretrained(base,'source/adapter',local_files_only=True,is_trainable=False);ctl=h.Controller(model);assert len(ctl.mods)==96
 tg=target_names(ctl.mods,ctl.nl);assert tg;rs=randoms(ctl.mods,tg);en=energy(ctl.mods,len(tg),tg);names=tg if L=='TARGET' else (rs[int(L.split('_')[1])] if L.startswith('RANDOM_') else en);ns=set(names)
 if M=='ablation':
  ctl.all(1.0)
  for x in ctl.mods:
   if x['name'] in ns:x['m'].scaling[x['key']]=0.0
 elif M=='only':
  ctl.all(0.0)
  for x in ctl.mods:
   if x['name'] in ns:x['m'].scaling[x['key']]=x['scale']
 else:raise KeyError(M)
 out={'schema':'LUCIA_AA_R22542_C39_CAUSAL_STAGE2_ATOMIC_CONFIG_V1','target':T,'target_site_count':len(tg),'target_site_names_sha256':hashlib.sha256('\n'.join(tg).encode()).hexdigest(),'label':L,'mode':M,'site_count':len(names),'site_names_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest(),'case_results':score(model,tok,rows),'locked_splits':['CONFIRMATION','GENERATOR_HOLDOUT'],'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'hf_model_redownloads':0,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389'}
 Path('public').mkdir(exist_ok=True);Path('public/result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
