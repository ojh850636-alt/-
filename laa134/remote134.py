"""LAA R134 one-ingress, public MRPC LoRA behavioral forensic microscope.

No downloaded base/adapter/tokenizer or raw activation/logit is exported.
This is a bounded research experiment, NOT proof of transferable intelligence.
The behavioral protocol and static head/operator controls descend from R131.
"""
from __future__ import annotations
import os,re,json,hashlib,time,math,random,tempfile,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import torch
from huggingface_hub import HfApi,snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer,AutoModelForSequenceClassification

ROOT=Path(__file__).resolve().parent
LOCK=ROOT/'lock134.json';SUITE=ROOT/'suite134.jsonl'
OUT=Path(os.environ.get('LAA134_OUTPUT','result134.json')).resolve()
PAT=re.compile(r'^(?:base_model\.model\.)?(.+?)\.lora_([AB])(?:\.default)?\.weight$')
HEAD=re.compile(r'^(?:base_model\.model\.)?classifier\.modules_to_save\.[^.]+\.(.+)$')
DIRECT=re.compile(r'^(?:base_model\.model\.)?classifier\.((?:dense|out_proj)\.(?:weight|bias))$')


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def load_contract():
 d=json.loads(LOCK.read_text('utf8'));rows=[json.loads(s) for s in SUITE.read_text('utf8').splitlines() if s.strip()]
 assert sha(SUITE)==d['suite_sha256']
 assert sha(__file__)==d['remote_code_sha256']
 assert d['source_repo']=='october25kim/lora-merge-prereg-roberta-base'
 assert d['source_rev']=='8896fca8a657e47d4d144853f0123aff77292fd4'
 assert d['base_repo']=='FacebookAI/roberta-base' and d['base_rev']=='e2da8e2f811d1448a5b465c236feacd80ffbac7b'
 assert d['pre_output_model_forward_count']==0
 assert len(rows)==320 and len(set(x['id'] for x in rows))==320
 assert sum(x['split']=='NEGATIVE' for x in rows)==32
 assert all(x['label'] in (0,1) for x in rows if x['split']!='NEGATIVE')
 assert all(x['source_origin']=='AUTHOR134_SYNTHETIC' for x in rows)
 return d,rows


def download_source(d,root):
 repo=d['source_repo'];rev=d['source_rev']
 assert HfApi().model_info(repo,revision=rev).sha==rev
 want=[f'mrpc/seed{s}/{n}' for s in (0,1) for n in ('adapter_config.json','adapter_model.safetensors','train_meta.json')]
 ar=root/'adapter'; snapshot_download(repo,revision=rev,allow_patterns=want,local_dir=ar)
 for seed in (0,1):
  sd=ar/f'mrpc/seed{seed}';sp=d['adapters'][str(seed)]
  assert (sd/'adapter_model.safetensors').stat().st_size==sp['size']
  assert sha(sd/'adapter_model.safetensors')==sp['sha256']
  cfg=json.loads((sd/'adapter_config.json').read_text());tr=json.loads((sd/'train_meta.json').read_text())
  assert tr['task']=='mrpc' and tr['seed']==seed and tr['recipe']['base_revision']==d['base_rev']
  assert tr['budget']['num_labels']==2
  assert cfg['r']==8 and cfg['lora_alpha']==16 and cfg['bias']=='none' and cfg['task_type']=='SEQ_CLS'
  assert set(cfg['target_modules'])==set(('query','key','value','output.dense','intermediate.dense'))
  assert set(cfg['modules_to_save'])=={'classifier','score'}
  assert sha(sd/'adapter_config.json')==sp['config_sha256']
  assert sha(sd/'train_meta.json')==sp['meta_sha256']
 bp=root/'base';patterns=['model.safetensors','config.json','tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json']
 snapshot_download(d['base_repo'],revision=d['base_rev'],allow_patterns=patterns,local_dir=bp)
 assert (bp/'model.safetensors').stat().st_size==d['base_size'] and sha(bp/'model.safetensors')==d['base_sha256']
 return ar,bp


def read_adapter(path):
 pairs=collections.defaultdict(dict);head={};names=[]
 with safe_open(str(path),framework='pt',device='cpu') as f:
  for k in f.keys():
   t=f.get_tensor(k).float();assert bool(torch.isfinite(t).all())
   m=PAT.fullmatch(k);h=HEAD.fullmatch(k) or DIRECT.fullmatch(k)
   if m:pairs[m.group(1)][m.group(2)]=t
   elif h:head[h.group(1)]=t
   else:raise ValueError('UNSUPPORTED_COMPONENT:'+k[:120])
   names.append(k)
 assert len(pairs)==72 and set(head)=={'dense.weight','dense.bias','out_proj.weight','out_proj.bias'}
 stats={'total_tensors':len(names),'lora_sites':len(pairs),'head_tensors':len(head),'head_elements':sum(x.numel() for x in head.values()),'nonzero_sites':0,'zero_sites':0,'energy_by_site':{}}
 for k,v in pairs.items():
  assert set(v)=={'A','B'} and v['A'].shape[0]==v['B'].shape[1]==8
  e=torch.trace((v['A']@v['A'].T)@(v['B'].T@v['B'])).item()
  assert math.isfinite(e) and e>=-1e-7
  stats['nonzero_sites' if e>1e-18 else 'zero_sites']+=1
  stats['energy_by_site'][k]=round(float(e),8)
 return dict(pairs),head,stats


def rand_lora(pairs,seed):
 gen=torch.Generator().manual_seed(seed);out={}
 for n,v in pairs.items():
  a,b=v['A'],v['B'];ar=torch.randn(a.shape,generator=gen);br=torch.randn(b.shape,generator=gen)
  en=torch.trace((a@a.T)@(b.T@b)).item();er=torch.trace((ar@ar.T)@(br.T@br)).item()
  assert er>1e-12
  out[n]={'A':ar,'B':br*math.sqrt(max(0,en)/er)}
 return out


def shuffle_lora(pairs,seed):
 rng=random.Random(seed);grp=collections.defaultdict(list)
 for k,v in pairs.items():grp[(tuple(v['A'].shape),tuple(v['B'].shape))].append(k)
 out={}
 for ids in grp.values():
  keys=sorted(ids);perm=keys[:];rng.shuffle(perm)
  if len(keys)>1 and perm==keys:perm=keys[1:]+keys[:1]
  for a,b in zip(keys,perm):out[a]=pairs[b]
 return out


def set_head(model,head):
 own=model.classifier.state_dict()
 assert set(own)==set(head)
 assert all(own[k].shape==head[k].shape for k in own)
 model.classifier.load_state_dict(head,strict=True)


def attach(model,pairs,scale=1.,skip=(),only=None,skip_proj=None):
 modules=dict(model.named_modules());handles=[]
 try:
  for n,v in sorted(pairs.items()):
   m=re.search(r'\.encoder\.layer\.(\d+)\.',n)
   if m is None:raise ValueError('LAYER_UNRECOGNIZED:'+n)
   layer=int(m.group(1))
   if layer in skip or (only is not None and layer not in only):continue
   if skip_proj and skip_proj in n:continue
   if n not in modules:raise ValueError('MODULE_NOT_FOUND:'+n)
   mod=modules[n];a,b=v['A'],v['B']
   assert a.shape[1]==mod.in_features and b.shape[0]==mod.out_features
   def make(x,y):
    def hook(_m,args,out):return out+((args[0].float()@x.T)@y.T).to(out.dtype)*(2.*scale)
    return hook
   handles.append(mod.register_forward_hook(make(a,b)))
 except BaseException:
  for h in handles:h.remove()
  raise
 return handles


def predict(model,batches):
 ys=[];cs=[]
 with torch.inference_mode():
  for batch in batches:
   z=model(**batch).logits
   assert z.shape[-1]==2 and bool(torch.isfinite(z).all())
   prob=z.float().softmax(-1);ys.extend(z.argmax(-1).cpu().tolist())
   cs.extend((2 if x>=.9 else 1 if x>=.8 else 0) for x in prob.max(-1).values.cpu().tolist())
 return ys,cs


def acc(pred,rows,split):
 ids=[i for i,r in enumerate(rows) if r['split']==split and r['label'] is not None]
 return round(sum(pred[i]==rows[i]['label'] for i in ids)/len(ids),7)


def parity(base,path,model,head,pairs,tok,rows):
 from peft import PeftModel
 torch.manual_seed(4242)
 reference=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=2,local_files_only=True,use_safetensors=True,trust_remote_code=False)
 reference=PeftModel.from_pretrained(reference,str(path),local_files_only=True,is_trainable=False)
 reference.eval()
 b=tok([r['text1'] for r in rows[:6]],[r['text2'] for r in rows[:6]],padding=True,truncation=True,max_length=128,return_tensors='pt')
 set_head(model,head);hh=attach(model,pairs)
 try:
  with torch.inference_mode():l0=model(**b).logits
 finally:
  for h in hh:h.remove()
 with torch.inference_mode():l1=reference(**b).logits
 error=float((l0-l1).abs().max().item())
 del reference
 if not math.isfinite(error) or error>0.0003:raise ValueError('NATIVE_PEFT_PARITY_FAIL:'+str(error))
 return error


def compact(pred,conf,rows):
 splits=('DISCOVERY','CONFIRMATION','LAW_OOD','STYLE_OOD','CONTRAST')
 return {'accuracy':{s:acc(pred,rows,s) for s in splits},'digest':hashlib.sha256(bytes(pred)).hexdigest(),
         'negative_high08':sum(conf[i]>=1 for i,r in enumerate(rows) if r['split']=='NEGATIVE'),
         'negative_high09':sum(conf[i]>=2 for i,r in enumerate(rows) if r['split']=='NEGATIVE')}


def main():
 start=time.time();torch.set_num_threads(min(4,os.cpu_count() or 1))
 d,rows=load_contract();record={'schema':'LAA134_MRPC_RAWFREE_BEHAVIOR_V1','verdict':'INITIALIZED',
       'source':{'repo':d['source_repo'],'rev':d['source_rev'],'base_repo':d['base_repo'],'base_rev':d['base_rev'],
                 'adapter_sha256':{s:x['sha256'] for s,x in d['adapters'].items()},'base_sha256':d['base_sha256']},
       'lock_sha256':sha(LOCK),'suite_sha256':sha(SUITE),'neural_forward_executed':False,
       'raw_logits_exported':False,'raw_activations_exported':False,'model_weights_exported':False,
       'supported_e4':False,'supported_e5':False,'algorithm':False,'student':False,'lucia_import':False,
       'model_probe_scope':'CORRELATED_AUTHOR134_EXPERIMENT_NONINDEPENDENT','seeds':{},'predictions':[]}
 with tempfile.TemporaryDirectory(prefix='laa134_private_') as tmp:
  src=Path(tmp);os.environ['HF_HOME']=str(src/'cache');os.environ['HUGGINGFACE_HUB_CACHE']=str(src/'cache'/'hub')
  adapter,base=download_source(d,src)
  tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,use_fast=True)
  def batches_for(rr):
   return [tok([r['text1'] for r in rr[i:i+16]],[r['text2'] for r in rr[i:i+16]],padding=True,truncation=True,max_length=128,return_tensors='pt') for i in range(0,len(rr),16)]
  batches=batches_for(rows)
  focus=[i for i,x in enumerate(rows) if x['split'] in ('CONFIRMATION','LAW_OOD','STYLE_OOD','CONTRAST')]
  # Locked reduced causal subset: balanced 48 cases, all grouped by split.
  focus=[idx for k,idx in enumerate(focus) if (k%8) in (0,1)]
  focusrows=[rows[i] for i in focus];fbatches=batches_for(focusrows)
  for seed in (0,1):
   fold=adapter/f'mrpc/seed{seed}'
   pairs,head,struct=read_adapter(fold/'adapter_model.safetensors')
   torch.manual_seed(4242)
   model=AutoModelForSequenceClassification.from_pretrained(str(base),num_labels=2,local_files_only=True,use_safetensors=True,trust_remote_code=False)
   model.eval();basehead={k:t.detach().clone() for k,t in model.classifier.state_dict().items()}
   perr=parity(base,fold,model,head,pairs,tok,rows)
   o={'structure':struct,'peft_parity_maxlogit':perr,'main':{},'fine':{},'fine_status':None}
   conditions=['BASE','HEAD_ONLY','LORA_ONLY','FULL','RANDOM','SHUFFLED','DOSE_HALF','DOSE_HIGH','ABL_LATE','KEEP_LATE']
   for cond in conditions:
    set_head(model,basehead if cond in ('BASE','LORA_ONLY') else head)
    act=None;scale=1.;skip=();only=None
    if cond in ('FULL','LORA_ONLY','DOSE_HALF','DOSE_HIGH','ABL_LATE','KEEP_LATE'):act=pairs
    if cond=='RANDOM':act=rand_lora(pairs,134103+seed)
    if cond=='SHUFFLED':act=shuffle_lora(pairs,134203+seed)
    if cond=='DOSE_HALF':scale=.5
    if cond=='DOSE_HIGH':scale=1.5
    if cond=='ABL_LATE':skip=range(8,12)
    if cond=='KEEP_LATE':only=set(range(8,12))
    hh=[]
    try:
     if act:hh=attach(model,act,scale=scale,skip=skip,only=only)
     yy,cc=predict(model,batches)
    finally:
     for h in hh:h.remove()
    o['main'][cond]=compact(yy,cc,rows)
    record['predictions'].extend({'seed':seed,'condition':cond,'case_id':r['id'],'prediction':int(y),
         'confidence_bin':(int(c) if r['split']=='NEGATIVE' else None)} for r,y,c in zip(rows,yy,cc))
   # Both positive and failure exploration are preregistered, conditional on large effects.
   delta=o['main']['FULL']['accuracy']['CONFIRMATION']-o['main']['HEAD_ONLY']['accuracy']['CONFIRMATION']
   o['conditional_confirmation_delta']=round(delta,7)
   ctl=max(o['main']['RANDOM']['accuracy']['CONFIRMATION'],o['main']['SHUFFLED']['accuracy']['CONFIRMATION'])
   trigger=abs(delta)>=.12 and abs(o['main']['FULL']['accuracy']['CONFIRMATION']-ctl)>=.08
   o['fine_status']='EXPLORATORY_FINE_GIVEN_PRESEALED_GATE' if trigger else 'NOT_RUN_BEHAVIOR_GATE_FAIL'
   if trigger:
    rng=random.Random(134303+seed)
    fine=[('ABL_EARLY',set(range(0,4)),None,None),('ABL_MIDDLE',set(range(4,8)),None,None),
          ('KEEP_EARLY',set(),set(range(0,4)),None),('KEEP_MIDDLE',set(),set(range(4,8)),None)]
    fine += [(f'ABL_LAYER_{i}',{i},None,None) for i in range(12)]
    fine += [('ABL_QUERY',set(),None,'attention.self.query'),('ABL_KEY',set(),None,'attention.self.key'),
             ('ABL_VALUE',set(),None,'attention.self.value'),('ABL_INTER',set(),None,'intermediate.dense'),
             ('ABL_OUTPUT',set(),None,'output.dense')]
    for band in range(3):
     other=[i for i in range(12) if i//4!=band]
     for j in range(2):fine.append((f'RANDOM_LAYER_MATCHED_{band}_{j}',set(rng.sample(other,4)),None,None))
    for name,skip,only,proj in fine:
     set_head(model,head);hh=attach(model,pairs,skip=skip,only=only,skip_proj=proj)
     try:y,c=predict(model,fbatches)
     finally:
      for h in hh:h.remove()
     o['fine'][name]=compact(y,c,focusrows)
     record['predictions'].extend({'seed':seed,'condition':name,'case_id':r['id'],'prediction':int(yy),'confidence_bin':None} for r,yy in zip(focusrows,y))
   record['seeds'][str(seed)]=o
   print('FINISHED_SEED',seed,'Full',o['main']['FULL']['accuracy'],'Delta',delta,'Fine',o['fine_status'],'Elapsed',round(time.time()-start,2),flush=True)
   del model,pairs,head,basehead
  record['neural_forward_executed']=True;record['verdict']='R134_REAL_MRPC_E2_AND_CONDITIONAL_E3_EXPLORATORY'
 if src.exists():raise RuntimeError('SOURCE_LEAK')
 record['raw_isolation_exit_absence']=True;record['elapsed_seconds']=round(time.time()-start,3)
 OUT.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
 print('R134_DONE',len(record['predictions']),'RAW_ABSENT',not src.exists(),flush=True)

if __name__=='__main__':main()
