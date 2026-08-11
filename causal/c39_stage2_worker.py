from __future__ import annotations
import hashlib, importlib.util, json, os, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TARGET=os.environ['C39_STAGE2_TARGET']
SEED=22542
LAYER_ENERGY={0:3.00118365879277,1:2.9759737660180234,2:3.0227823473275715,3:5.287264024722158,4:3.177676094744209,5:3.270127721164994,6:3.325560465664014,7:5.872125550417814,8:3.4956401320951898,9:3.6159630030263963,10:3.5724111740162505,11:6.33461433964186,12:3.7572791310656433,13:3.76167513577718,14:3.718853044554014,15:6.719130669655517,16:3.6930369726854426,17:3.608289468314058,18:3.6448871993515772,19:6.717229149893191,20:3.6522505664600744,21:3.738242114197289,22:3.7363648644048584,23:6.3014394060099}
PROJ_ENERGY={'down_proj':19.552549083239654,'gate_proj':33.0784446149292,'k_proj':2.2858834821409126,'o_proj':4.57629806853297,'q_proj':6.673736116112564,'up_proj':31.701904517311462,'v_proj':2.1311841177332367}

def helper():
 p=Path('causal/c39_behavior_pinned.py');s=importlib.util.spec_from_file_location('c39_behavior_pinned',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def component_pred(x,comp,n):
 a=max(1,n//3);b=max(a+1,(2*n)//3);p=x['proj'];l=x['layer']
 if comp=='ATTENTION_ALL':return p in {'q_proj','k_proj','v_proj','o_proj'}
 if comp=='MLP_ALL':return p in {'gate_proj','up_proj','down_proj'}
 if comp=='EARLY_BAND':return 0<=l<a
 if comp=='MIDDLE_BAND':return a<=l<b
 if comp=='LATE_BAND':return l>=b
 return p=={'Q_PROJ':'q_proj','K_PROJ':'k_proj','V_PROJ':'v_proj','O_PROJ':'o_proj','GATE_PROJ':'gate_proj','UP_PROJ':'up_proj','DOWN_PROJ':'down_proj'}[comp]
def target_pred(x,n):return all(component_pred(x,c,n) for c in TARGET.split('&'))

def batched_score(model,tok,rows,bs=8):
 tok.padding_side='left';out=[];model.eval();pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 with torch.inference_mode():
  for st in range(0,len(rows),bs):
   chunk=rows[st:st+bs];seqs=[];cls=[]
   for r in chunk:
    p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids;seqs.append(p+c);cls.append(len(c))
   ml=max(map(len,seqs));mc=max(cls);ids=torch.full((len(chunk),ml),pad,dtype=torch.long,device=model.device);am=torch.zeros_like(ids)
   for i,s in enumerate(seqs):ids[i,-len(s):]=torch.tensor(s,dtype=torch.long,device=model.device);am[i,-len(s):]=1
   logits=model(input_ids=ids,attention_mask=am,use_cache=False,logits_to_keep=mc+1).logits.float();lp=torch.log_softmax(logits,dim=-1)
   for i,(r,cl) in enumerate(zip(chunk,cls)):
    pred=lp[i,-cl-1:-1,:];t=ids[i,-cl:];v=pred.gather(-1,t.unsqueeze(-1)).squeeze(-1)
    out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'token_count':cl})
 return out

def set_selected(ctl,selected_names,mode):
 names=set(selected_names)
 if mode=='ablation':
  ctl.all(1.0)
  for x in ctl.mods:
   if x['name'] in names:x['m'].scaling[x['key']]=0.0
 elif mode=='only':
  ctl.all(0.0)
  for x in ctl.mods:
   if x['name'] in names:x['m'].scaling[x['key']]=x['scale']
 else:raise KeyError(mode)

def random_controls(mods,target_names,count=3):
 names=sorted(x['name'] for x in mods);n=len(target_names);rng=random.Random(SEED+int(hashlib.sha256(TARGET.encode()).hexdigest()[:8],16));out=[];seen={tuple(sorted(target_names))}
 while len(out)<count:
  s=tuple(sorted(rng.sample(names,n)))
  if s not in seen:seen.add(s);out.append(list(s))
 return out

def energy_control(mods,n,target_names):
 ranked=sorted(mods,key=lambda x:(LAYER_ENERGY.get(x['layer'],0)*PROJ_ENERGY.get(x['proj'],0),x['name']),reverse=True)
 s=[x['name'] for x in ranked[:n]]
 if set(s)==set(target_names):s=[x['name'] for x in ranked[1:n+1]]
 return sorted(s)

def main():
 h=helper();rows=[r for r in h.build_suite() if r['positive'] and r['split'] in {'CONFIRMATION','GENERATOR_HOLDOUT'}];assert len(rows)==128
 tok=AutoTokenizer.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True)
 model=PeftModel.from_pretrained(base,'source/adapter',local_files_only=True,is_trainable=False);ctl=h.Controller(model);assert len(ctl.mods)==96
 target=sorted(x['name'] for x in ctl.mods if target_pred(x,ctl.nl));assert target
 controls=random_controls(ctl.mods,target,3);energy=energy_control(ctl.mods,len(target),target)
 configs=[]
 for label,names in [('TARGET',target)]+[(f'RANDOM_{i}',s) for i,s in enumerate(controls)]+[('ENERGY_TOP',energy)]:
  for mode in ('ablation','only'):
   set_selected(ctl,names,mode);res=batched_score(model,tok,rows,bs=8);configs.append({'label':label,'mode':mode,'site_count':len(names),'site_names_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest(),'case_results':res})
 out={'schema':'LUCIA_AA_R22542_C39_CAUSAL_STAGE2_LOCKED_V1','target':TARGET,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','behavior_authority_run':31488582355,'stage1_main_run':31501795482,'locked_splits':['CONFIRMATION','GENERATOR_HOLDOUT'],'cases':128,'target_site_count':len(target),'target_site_names_sha256':hashlib.sha256('\n'.join(target).encode()).hexdigest(),'matched_random_controls':3,'energy_large_control':True,'configs':configs,'raw_weights_saved':False,'raw_logits_saved':False,'raw_activations_saved':False,'raw_tactics_saved':False,'hf_model_redownloads':0,'claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'}
 Path('public').mkdir(exist_ok=True);Path('public/result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
