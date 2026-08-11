from __future__ import annotations
import contextlib, hashlib, json, os, random, resource
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import PeftModel
from lean_tactic_forge import build
EXPECTED='a081ee299a7035337bb6fa6f6c380fb6d8cbd328534b049cd45a5fbe032338f2'
FORBIDDEN=('sorry','admit','axiom','unsafe','run_tac','set_option','#eval','#check','#print')
COND=os.environ['C39_GEN_CONDITION']
def h(b): return hashlib.sha256(b).hexdigest()
def mean(xs):
 xs=list(xs); return sum(xs)/len(xs) if xs else 0.0
def suite_sha(rows): return h(''.join(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n' for r in rows).encode())
def sanitize(text):
 s=text.replace('\r','\n').strip()
 if ':::' in s:s=s.split(':::',1)[0].strip()
 if '\n' in s:s=s.split('\n',1)[0].strip()
 if not s:return '','EMPTY'
 if any(t in s.lower() for t in FORBIDDEN):return '','FORBIDDEN'
 if len(s)>600:return '','TOO_LONG'
 return s,'OK'
class Ctl:
 def __init__(self,m):
  self.mods=[]
  for _,x in m.named_modules():
   if hasattr(x,'lora_A') and hasattr(x,'lora_B') and getattr(x,'lora_A',None):
    k=next(iter(x.lora_A)); self.mods.append((x,k,float(x.scaling[k])))
 def random(self,seed):
  g=torch.Generator(device='cpu').manual_seed(seed)
  for x,k,_ in self.mods:
   w=x.lora_B[k].weight; old=w.detach().float().cpu(); rnd=torch.randn(old.shape,generator=g); rnd*=old.norm()/(rnd.norm()+1e-12); w.data.copy_(rnd.to(w.device,dtype=w.dtype))
 def shuffle(self,seed):
  g=torch.Generator(device='cpu').manual_seed(seed)
  for x,k,_ in self.mods:
   w=x.lora_B[k].weight; old=w.detach().flatten().cpu(); perm=torch.randperm(old.numel(),generator=g); w.data.copy_(old[perm].reshape(w.shape).to(w.device,dtype=w.dtype))
def main():
 root=Path(os.environ.get('WORK_ROOT','.')).resolve(); out=root/'result'; out.mkdir(parents=True,exist_ok=True)
 rows=build(); assert suite_sha(rows)==EXPECTED
 locked=[]
 for sp in ['CONFIRMATION','GENERATOR_HOLDOUT','ALPHA_RENAME_OOD','NEGATIVE_UNPROVABLE']:
  locked += sorted([r for r in rows if r['split']==sp],key=lambda x:x['case_id'])[:12]
 base_dir=root/'source/base'; cand=root/'source/candidate'
 tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False); tok.padding_side='left'; tok.pad_token=tok.pad_token or tok.eos_token
 base=Qwen3_5ForCausalLM.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True)
 model=PeftModel.from_pretrained(base,cand,local_files_only=True,is_trainable=False); ctl=Ctl(model)
 disable=False
 if COND=='BASE_seed42': seed=42; disable=True
 elif COND=='FULL_seed42': seed=42
 elif COND=='RANDOM_seed42': seed=42; ctl.random(22542)
 elif COND=='SHUFFLED_seed42': seed=42; ctl.shuffle(22542)
 elif COND=='FULL_seed224': seed=224
 elif COND=='FULL_seed1337': seed=1337
 else: raise ValueError(COND)
 torch.manual_seed(seed); model.eval(); records=[]; ctx=model.disable_adapter() if disable else contextlib.nullcontext()
 with ctx,torch.inference_mode():
  for i,r in enumerate(locked):
   enc=tok(r['prompt'],return_tensors='pt',truncation=True,max_length=768,add_special_tokens=False); enc={k:v.to(model.device) for k,v in enc.items()}
   gen=model.generate(**enc,max_new_tokens=48,do_sample=True,temperature=.8,top_p=.95,top_k=50,num_return_sequences=8,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   plen=enc['input_ids'].shape[1]; samples=[]
   for q in range(8):
    raw=tok.decode(gen[q,plen:],skip_special_tokens=True); tac,status=sanitize(raw); exact=bool(r['positive'] and status=='OK' and tac==r['canonical_tactic'])
    samples.append({'rank':q,'status':status,'tactic_sha256':h(tac.encode()) if tac else None,'exact_canonical_match':exact,'kernel_verified_success_conservative':exact})
   records.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'positive':r['positive'],'success_at_k':any(x['kernel_verified_success_conservative'] for x in samples),'forbidden_count':sum(x['status']=='FORBIDDEN' for x in samples),'empty_count':sum(x['status']=='EMPTY' for x in samples),'noncanonical_unverified_count':sum(x['status']=='OK' and not x['exact_canonical_match'] for x in samples),'sample_digests':samples})
   if (i+1)%12==0: print('GEN_PROGRESS',COND,i+1,'/',len(locked),'rss_kb',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
 by={}
 for r in records:by.setdefault(r['split'],[]).append(r)
 def s(v):return {'n':len(v),'success_at_k_rate':mean(1.0 if x['success_at_k'] else 0.0 for x in v),'forbidden_sample_count':sum(x['forbidden_count'] for x in v),'noncanonical_unverified_sample_count':sum(x['noncanonical_unverified_count'] for x in v)}
 summary={'schema':'LUCIA_AA_R22542_C39_ATOMIC_GENERATION_CONDITION_V1','condition':COND,'seed':seed,'locked_cases':len(records),'k':8,'verification_policy':'PRESEALED_CANONICAL_EXACT_MATCH_ONLY_COUNTS_KERNEL_VERIFIED; NONCANONICAL_OUTPUTS_UNVERIFIED','rng_amendment':'torch.manual_seed(precommitted_seed), no unsupported generator kwarg','overall':s(records),'by_split':{k:s(v) for k,v in sorted(by.items())},'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'raw_tactic_text_saved':False,'raw_logits_saved':False,'hf_redownloads':0,'source':'authoritative escrow run 31484392389'}
 (out/f'{COND}.json').write_text(json.dumps({'summary':summary,'case_records':records},indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(summary,sort_keys=True),flush=True)
if __name__=='__main__': main()
