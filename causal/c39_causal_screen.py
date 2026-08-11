from __future__ import annotations
import contextlib, importlib.util, json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

OBJECT=os.environ['C39_CAUSAL_OBJECT']
OBJECTS=('ATTENTION_ALL','MLP_ALL','EARLY_BAND','MIDDLE_BAND','LATE_BAND','Q_PROJ','K_PROJ','V_PROJ','O_PROJ','GATE_PROJ','UP_PROJ','DOWN_PROJ')
assert OBJECT in OBJECTS

def load_helper():
 p=Path('causal/c39_behavior_pinned.py');spec=importlib.util.spec_from_file_location('c39_behavior_pinned',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def score_compact(model,tok,rows):
 out=[];model.eval()
 with torch.inference_mode():
  for r in rows:
   p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids
   ids=torch.tensor([p+c],dtype=torch.long,device=model.device);am=torch.ones_like(ids)
   logits=model(input_ids=ids,attention_mask=am,use_cache=False,logits_to_keep=len(c)+1).logits.float();lp=torch.log_softmax(logits[:,:-1,:],dim=-1);t=ids[0,-len(c):];v=lp[0].gather(-1,t.unsqueeze(-1)).squeeze(-1)
   out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'token_count':len(c)})
 return out

def main():
 h=load_helper();root=Path('.');rows=[r for r in h.build_suite() if r['positive'] and r['split']=='DISCOVERY'];assert len(rows)==64
 tok=AutoTokenizer.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained('source/base',local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True)
 model=PeftModel.from_pretrained(base,'source/adapter',local_files_only=True,is_trainable=False);ctl=h.Controller(model);assert len(ctl.mods)==96
 n=max(1,ctl.nl);a=max(1,n//3);b=max(a+1,(2*n)//3)
 def pred(x):
  if OBJECT=='ATTENTION_ALL':return x['path']=='ATTENTION'
  if OBJECT=='MLP_ALL':return x['path']=='MLP'
  if OBJECT=='EARLY_BAND':return 0<=x['layer']<a
  if OBJECT=='MIDDLE_BAND':return a<=x['layer']<b
  if OBJECT=='LATE_BAND':return x['layer']>=b
  mp={'Q_PROJ':'q_proj','K_PROJ':'k_proj','V_PROJ':'v_proj','O_PROJ':'o_proj','GATE_PROJ':'gate_proj','UP_PROJ':'up_proj','DOWN_PROJ':'down_proj'}
  return x['projection']==mp[OBJECT]
 selected=[x['name'] for x in ctl.mods if pred(x)];assert selected
 ctl.reset();ctl.disable(pred);abl=score_compact(model,tok,rows)
 ctl.set_all(0.0)
 for x in ctl.modules:
  if pred(x):x['module'].scaling[x['key']]=x['base_scale']
 suf=score_compact(model,tok,rows)
 out={'schema':'LUCIA_AA_R22542_C39_CAUSAL_BROAD_OBJECT_V1','object':OBJECT,'screen_split':'DISCOVERY','screen_cases':64,'selected_site_count':len(selected),'selected_site_names_sha256':__import__('hashlib').sha256('\n'.join(sorted(selected)).encode()).hexdigest(),'necessity_ablation':abl,'candidate_only_sufficiency':suf,'raw_logits_saved':False,'raw_activations_saved':False,'raw_weights_saved':False,'hf_model_redownloads':0,'source':'AUTHORITATIVE_ESCROW_RUN_31484392389','claim_cap':'E3_EXPLORATORY_TOKENIZER_PROVENANCE_HOLD'}
 Path('public').mkdir(exist_ok=True);Path(f'public/{OBJECT}.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
