from __future__ import annotations
import argparse,contextlib,importlib.util,json,sys
from pathlib import Path

def loadmod(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['teacher','generation'],required=True);ap.add_argument('--condition',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--base',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--private');a=ap.parse_args()
 b=loadmod('laa/r22542/c39_behavior.py','c39_behavior_mod')
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 torch.set_num_threads(4)
 rows=b.build_suite();pos=[r for r in rows if r['positive']]
 tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,use_safetensors=True)
 model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True,is_trainable=False);ctl=b.Controller(model);assert len(ctl.mods)==96
 cond=a.condition;snap=ctl.snap()
 def setcond():
  if cond=='FULL':ctl.all(1.0);return contextlib.nullcontext()
  if cond=='BASE':ctl.all(1.0);return model.disable_adapter()
  if cond.startswith('DOSE_'):ctl.all(float(cond.split('_',1)[1]));return contextlib.nullcontext()
  if cond=='RANDOM_42':ctl.all(1);ctl.restore(snap);ctl.random(42);return contextlib.nullcontext()
  if cond=='SHUFFLED':ctl.all(1);ctl.restore(snap);ctl.shuffle(22542);return contextlib.nullcontext()
  raise ValueError(cond)
 ctx=setcond()
 if a.mode=='teacher':
  with ctx:r=b.score(model,tok,pos,batch=8)
  dump(a.out,{'schema':'LUCIA_AA_R22542_C39_TEACHER_SHARD_V1','condition':cond,'case_results':r,'summary':b.summary(r),'model_outputs_observed':len(r),'raw_tactics_exported':False})
 else:
  locked=[]
  for sp in ('CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE'):locked+=sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
  with ctx:r=b.generate(model,tok,locked,cond,a.seed,k=8)
  Path(a.private).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in r),encoding='utf-8')
  dump(a.out,{'schema':'LUCIA_AA_R22542_C39_GENERATION_SHARD_RECEIPT_V1','condition':cond,'seed':a.seed,'locked_cases':len(locked),'records':len(r),'private_path':a.private,'raw_tactics_exported_public':False})
if __name__=='__main__':main()
