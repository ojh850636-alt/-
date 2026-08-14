from __future__ import annotations
import argparse, hashlib, json, math, random, re
from pathlib import Path
from collections import defaultdict
SEED=22564
ADAPTER_SHA='87991c65e5c48403a8e4d8057fe339b4c296df86257cd7a6c1a6c11245ab111f'
BASE_SHA='340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9'

def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def mean(x): return sum(x)/len(x) if x else 0.0
def auc(labels,scores):
 pairs=sorted(zip(scores,labels)); rank=1; sr=0.; i=0
 while i<len(pairs):
  j=i+1
  while j<len(pairs) and pairs[j][0]==pairs[i][0]: j+=1
  av=(rank+(rank+j-i-1))/2; sr+=av*sum(y for _,y in pairs[i:j]); rank+=j-i; i=j
 n1=sum(labels); n0=len(labels)-n1
 return (sr-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else .5
def bootstrap_auc_delta(labels,a,b,n=1000):
 r=random.Random(SEED); N=len(labels); vals=[]
 for _ in range(n):
  ix=[r.randrange(N) for _ in range(N)]; ys=[labels[i] for i in ix]
  if len(set(ys))<2: continue
  vals.append(auc(ys,[a[i] for i in ix])-auc(ys,[b[i] for i in ix]))
 vals.sort(); return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]] if vals else [0,0]
def static_forensics(path):
 from safetensors import safe_open
 out={'schema':'LAA_R22564_C44_STATIC_V1','adapter_sha256':file_sha(path)}; assert out['adapter_sha256']==ADAPTER_SHA
 keys=[]; lora=aux=total=0; proj=defaultdict(int); layers=defaultdict(int)
 with safe_open(str(path),framework='pt',device='cpu') as f:
  for k in f.keys():
   sh=f.get_slice(k).get_shape(); n=math.prod(sh); total+=n; keys.append(k)
   if 'lora_A' in k or 'lora_B' in k:
    lora+=n; proj['Wqkv' if 'Wqkv' in k else ('Wo' if 'Wo' in k else 'other')]+=n
    m=re.search(r'layers\.(\d+)',k); layers[m.group(1) if m else 'na']+=n
   else: aux+=n
 out.update(total_params=total,lora_params=lora,auxiliary_params=aux,lora_fraction=lora/total,auxiliary_fraction=aux/total,key_count=len(keys),lora_params_by_projection=dict(proj),lora_params_by_layer=dict(layers),sample_keys=keys[:12]); return out

def iter_lora(model):
 for n,m in model.named_modules():
  if hasattr(m,'lora_A') and hasattr(m,'lora_B') and hasattr(m,'scaling'): yield n,m
def iter_aux(model):
 for n,m in model.named_modules():
  if m.__class__.__name__=='ModulesToSaveWrapper': yield n,m
def key(m): return next(iter(m.scaling))
class Controller:
 def __init__(self,m):
  self.m=m; self.l=list(iter_lora(m)); self.a=list(iter_aux(m)); self.scal={(n,key(x)):float(x.scaling[key(x)]) for n,x in self.l}; self.bs={(n,key(x)):x.lora_B[key(x)].weight.detach().clone() for n,x in self.l}
 def restore(self):
  for n,x in self.l:
   k=key(x); x.scaling[k]=self.scal[(n,k)]; x.lora_B[k].weight.data.copy_(self.bs[(n,k)])
  for _,x in self.a: x.disable_adapters=False
 def aux(self,on):
  for _,x in self.a: x.disable_adapters=not on
 def lora(self,s):
  for n,x in self.l: x.scaling[key(x)]=self.scal[(n,key(x))]*s
 def set(self,c):
  self.restore()
  if c=='BASE': self.lora(0); self.aux(False)
  elif c=='HEAD_ONLY': self.lora(0); self.aux(True)
  elif c=='LORA_ONLY': self.lora(1); self.aux(False)
  elif c=='FULL': pass
  elif c.startswith('DOSE_'): self.lora(float(c.split('_')[1])); self.aux(True)
  elif c=='RANDOM_SIGN_SAME_HEAD':
   for i,(n,x) in enumerate(self.l): x.scaling[key(x)]=self.scal[(n,key(x))]*(-1 if i%2 else 1)
   self.aux(True)
 def broad_zero(self,which):
  self.restore(); self.aux(True)
  for n,x in self.l:
   if which in n: x.scaling[key(x)]=0.0

def load_rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def prob_rows(model,tok,rows,batch=16):
 import torch
 out=[]
 for i in range(0,len(rows),batch):
  ch=rows[i:i+batch]; enc=tok([x['text'] for x in ch],padding=True,truncation=True,max_length=512,return_tensors='pt')
  with torch.no_grad(): logits=model(**enc).logits[:,0]; out.extend(torch.sigmoid(logits).cpu().tolist())
 return out
def self_test():
 from transformers import ModernBertConfig,ModernBertForSequenceClassification
 from peft import LoraConfig,get_peft_model
 c=ModernBertConfig(vocab_size=1000,pad_token_id=0,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_labels=17)
 b=ModernBertForSequenceClassification(c); pc=LoraConfig(r=4,lora_alpha=8,target_modules=['Wqkv','Wo'],modules_to_save=['classifier'],task_type='SEQ_CLS'); m=get_peft_model(b,pc); ctl=Controller(m)
 assert ctl.l and ctl.a; ctl.set('BASE'); ctl.set('HEAD_ONLY'); ctl.set('LORA_ONLY'); ctl.set('FULL'); print(json.dumps({'self_test':'PASS','lora_modules':len(ctl.l),'aux_wrappers':len(ctl.a)}))
def run(a):
 import torch
 from transformers import AutoTokenizer,ModernBertForSequenceClassification
 from peft import PeftModel
 root=Path(a.work); base=root/'base'; ad=root/'adapter'; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 assert file_sha(base/'model.safetensors')==BASE_SHA and file_sha(ad/'adapter_model.safetensors')==ADAPTER_SHA
 rows=load_rows(a.suite); labels=[x['label'] for x in rows]
 static=static_forensics(ad/'adapter_model.safetensors'); (out/'C44_STATIC.json').write_text(json.dumps(static,indent=2,sort_keys=True))
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
 bm=ModernBertForSequenceClassification.from_pretrained(base,num_labels=17,problem_type='multi_label_classification',ignore_mismatched_sizes=True,attn_implementation='eager',reference_compile=False,local_files_only=True)
 model=PeftModel.from_pretrained(bm,str(ad),local_files_only=True,is_trainable=False); model.eval(); ctl=Controller(model)
 conditions=['BASE','HEAD_ONLY','LORA_ONLY','FULL']; probs={}
 for c in conditions: ctl.set(c); probs[c]=prob_rows(model,tok,rows,a.batch)
 aucs={c:auc(labels,probs[c]) for c in conditions}; delta=aucs['FULL']-aucs['HEAD_ONLY']; ci=bootstrap_auc_delta(labels,probs['FULL'],probs['HEAD_ONLY'])
 controls={}
 for c in ['RANDOM_SIGN_SAME_HEAD','DOSE_0.25','DOSE_0.5','DOSE_1.5']:
  ctl.set(c); p=prob_rows(model,tok,rows,a.batch); controls[c]={'auc':auc(labels,p),'mean_pos':mean([v for v,y in zip(p,labels) if y]),'mean_neg':mean([v for v,y in zip(p,labels) if not y])}
 gate=delta>=0.03 and ci[0]>0 and aucs['FULL']>=0.75
 broad={}
 if gate:
  for g in ['Wqkv','Wo']:
   ctl.broad_zero(g); p=prob_rows(model,tok,rows,a.batch); broad[g]={'auc':auc(labels,p),'drop_from_full':aucs['FULL']-auc(labels,p)}
 summary={'schema':'LAA_R22564_C44_BEHAVIOR_V1','n':len(rows),'aucs':aucs,'conditional_lora_auc_gain':delta,'conditional_lora_auc_gain_ci95':ci,'controls':controls,'behavior_gate_green':gate,'causal_broad':broad,'claim_level':'E3_CANDIDATE' if gate and broad else ('E2_CONTROL_SEPARATED_BEHAVIOR' if gate else 'E1_BEHAVIOR_ONLY_OR_FAILURE')}
 (out/'C44_BEHAVIOR_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
 with (out/'C44_CASE_RESULTS.jsonl').open('w') as f:
  for i,r in enumerate(rows): f.write(json.dumps({'case_id':r['case_id'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'label':r['label'],**{c:probs[c][i] for c in conditions}},sort_keys=True)+'\n')
 print(json.dumps(summary,sort_keys=True))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--work'); p.add_argument('--suite'); p.add_argument('--out'); p.add_argument('--batch',type=int,default=16); a=p.parse_args()
 if a.self_test:self_test()
 elif a.run:run(a)
if __name__=='__main__':main()
