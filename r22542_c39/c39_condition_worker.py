from __future__ import annotations
import contextlib,json,os,random,resource,re
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForCausalLM
from peft import PeftModel
from lean_tactic_forge import build
EXPECTED='a081ee299a7035337bb6fa6f6c380fb6d8cbd328534b049cd45a5fbe032338f2'
COND=os.environ['C39_CONDITION']
def mean(xs):xs=list(xs);return sum(xs)/len(xs) if xs else 0.0
class Ctl:
 def __init__(self,m):
  self.mods=[]
  for n,x in m.named_modules():
   if hasattr(x,'lora_A') and hasattr(x,'lora_B') and getattr(x,'lora_A',None):
    k=next(iter(x.lora_A));self.mods.append((x,k,float(x.scaling[k])))
 def setall(self,d):
  for x,k,s in self.mods:x.scaling[k]=s*d
 def random(self,seed):
  g=torch.Generator(device='cpu').manual_seed(seed)
  for x,k,_ in self.mods:
   w=x.lora_B[k].weight;old=w.detach().float().cpu();rnd=torch.randn(old.shape,generator=g);rnd*=old.norm()/(rnd.norm()+1e-12);w.data.copy_(rnd.to(w.device,dtype=w.dtype))
 def shuffle(self,seed):
  g=torch.Generator(device='cpu').manual_seed(seed)
  for x,k,_ in self.mods:
   w=x.lora_B[k].weight;old=w.detach().flatten().cpu();perm=torch.randperm(old.numel(),generator=g);w.data.copy_(old[perm].reshape(w.shape).to(w.device,dtype=w.dtype))
def score(model,tok,rows,disable=False):
 out=[];ctx=model.disable_adapter() if disable else contextlib.nullcontext();model.eval()
 with ctx,torch.inference_mode():
  for i,r in enumerate(rows):
   p=tok(r['prompt'],add_special_tokens=False).input_ids;c=tok(r['canonical_tactic'],add_special_tokens=False).input_ids;ids=torch.tensor([p+c],device=model.device);mask=torch.ones_like(ids);logits=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=len(c)+1).logits.float();lp=torch.log_softmax(logits[:,:-1,:],dim=-1);tgt=ids[0,-len(c):];vals=lp[0].gather(-1,tgt.unsqueeze(-1)).squeeze(-1);out.append({'case_id':r['case_id'],'split':r['split'],'family':r['family'],'token_count':len(c),'mean_logprob':float(vals.mean().cpu()),'sum_logprob':float(vals.sum().cpu())});del ids,mask,logits,lp,tgt,vals
   if (i+1)%32==0:print('PROGRESS',COND,i+1,'/',len(rows),'rss_kb',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
 return out
def main():
 root=Path(os.environ.get('WORK_ROOT','.')).resolve();out=root/'result';out.mkdir(parents=True,exist_ok=True);rows=[r for r in build() if r['positive']];base_dir=root/'source/base';cand=root/'source/candidate';tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False);base=Qwen3_5ForCausalLM.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,low_cpu_mem_usage=True);m=PeftModel.from_pretrained(base,cand,local_files_only=True,is_trainable=False);ctl=Ctl(m);disable=False
 if COND=='BASE':disable=True
 elif COND=='FULL':pass
 elif COND=='RANDOM_NORM_MATCHED_LORA':ctl.random(22542)
 elif COND=='SHUFFLED_LORA':ctl.shuffle(22542)
 elif COND.startswith('DOSE_'):ctl.setall(float(COND.split('_',1)[1]))
 else:raise ValueError(COND)
 vals=score(m,tok,rows,disable);p=out/f'{COND}.jsonl';p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in vals));by={}
 for x in vals:by.setdefault(x['split'],[]).append(x)
 summ={'schema':'LUCIA_AA_R22542_C39_ATOMIC_TEACHER_CONDITION_V1','condition':COND,'cases':len(vals),'overall_mean_logprob':mean(x['mean_logprob'] for x in vals),'by_split':{k:{'n':len(v),'mean_logprob':mean(x['mean_logprob'] for x in v)} for k,v in sorted(by.items())},'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'raw_logits_saved':False,'raw_text_saved':False,'hf_redownloads':0,'source':'authoritative escrow run 31484392389'};(out/f'{COND}.summary.json').write_text(json.dumps(summ,indent=2,sort_keys=True)+'\n');print(json.dumps(summ,sort_keys=True),flush=True)
if __name__=='__main__':main()
