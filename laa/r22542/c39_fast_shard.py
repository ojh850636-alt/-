from __future__ import annotations
import argparse,contextlib,importlib.util,json,sys
from pathlib import Path

def loadmod(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def score_fast(model,tok,rows,disable=False,batch=32):
 import torch
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;out=[];ctx=model.disable_adapter() if disable else contextlib.nullcontext();model.eval()
 with ctx,torch.inference_mode():
  for i in range(0,len(rows),batch):
   rr=rows[i:i+batch];pairs=[]
   for r in rr:
    p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids;pairs.append((p,c))
   mx=max(len(p)+len(c) for p,c in pairs);ids=[];masks=[];starts=[];lens=[]
   for p,c in pairs:
    q=p+c;n=len(q);ids.append(q+[pad]*(mx-n));masks.append([1]*n+[0]*(mx-n));starts.append(len(p));lens.append(len(c))
   ii=torch.tensor(ids,dtype=torch.long,device=model.device);am=torch.tensor(masks,dtype=torch.long,device=model.device);logits=model(input_ids=ii,attention_mask=am,use_cache=False).logits.float();lp=torch.log_softmax(logits[:,:-1,:],dim=-1)
   for j,r in enumerate(rr):
    st=starts[j];ln=lens[j];t=ii[j,st:st+ln];v=lp[j,st-1:st+ln-1].gather(-1,t.unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'mean_logprob':float(v.mean().cpu()),'sum_logprob':float(v.sum().cpu()),'token_count':ln})
 return out

def gen_fast(model,tok,rows,condition,seed,k=8,prompt_batch=4):
 import torch,hashlib
 tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token;torch.manual_seed(seed);out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(rows),prompt_batch):
   rr=rows[i:i+prompt_batch];e=tok([r['prompt'] for r in rr],return_tensors='pt',padding=True,add_special_tokens=False,truncation=True,max_length=768);e={a:b.to(model.device) for a,b in e.items()};n=e['input_ids'].shape[1]
   g=model.generate(**e,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=k,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   for j,r in enumerate(rr):
    for rank in range(k):
     raw=tok.decode(g[j*k+rank,n:],skip_special_tokens=True);t,st=helper.sanitize(raw);out.append({'condition':condition,'seed':seed,'case_id':r['case_id'],'split':r['split'],'family':r['family'],'rank':rank,'status':st,'tactic':t,'tactic_sha256':hashlib.sha256(t.encode()).hexdigest() if t else None})
 return out

def main():
 global helper
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['teacher','generation'],required=True);ap.add_argument('--condition',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--base',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--private');a=ap.parse_args();helper=loadmod('laa/r22542/c39_behavior.py','c39_behavior_fast_helper')
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 torch.set_num_threads(4);rows=helper.build_suite();pos=[r for r in rows if r['positive']]
 tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True);model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True,is_trainable=False);ctl=helper.Controller(model);assert len(ctl.mods)==96
 snap=ctl.snap();cond=a.condition
 if cond=='FULL':ctl.all(1);ctx=contextlib.nullcontext()
 elif cond=='BASE':ctl.all(1);ctx=model.disable_adapter()
 elif cond.startswith('DOSE_'):ctl.all(float(cond.split('_',1)[1]));ctx=contextlib.nullcontext()
 elif cond=='RANDOM_42':ctl.all(1);ctl.restore(snap);ctl.random(42);ctx=contextlib.nullcontext()
 elif cond=='SHUFFLED':ctl.all(1);ctl.restore(snap);ctl.shuffle(22542);ctx=contextlib.nullcontext()
 else:raise ValueError(cond)
 if a.mode=='teacher':
  with ctx:r=score_fast(model,tok,pos,batch=32)
  dump(a.out,{'schema':'LUCIA_AA_R22542_C39_TEACHER_FAST_SHARD_V1','condition':cond,'case_results':r,'summary':helper.summary(r),'model_outputs_observed':len(r),'raw_tactics_exported':False,'execution_batch_size':32})
 else:
  locked=[]
  for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):locked+=sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
  with ctx:r=gen_fast(model,tok,locked,cond,a.seed,k=8,prompt_batch=4)
  Path(a.private).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in r),encoding='utf-8');dump(a.out,{'schema':'LUCIA_AA_R22542_C39_GENERATION_FAST_SHARD_RECEIPT_V1','condition':cond,'seed':a.seed,'locked_cases':len(locked),'records':len(r),'execution_prompt_batch_size':4,'private_path':a.private,'raw_tactics_exported_public':False})
if __name__=='__main__':main()
