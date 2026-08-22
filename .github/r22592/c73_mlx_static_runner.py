from __future__ import annotations
import hashlib,json,os,re,shutil,struct,traceback,urllib.request
from pathlib import Path
import numpy as np
R='esc511/trip-optimizer-mutator'; V='b53c45b85e53dab2b4110d3cf564ee5c75737ea0'; W='adapters.safetensors'
SZ=58768604; SHA='0708a854e8c80ef0800f38b8361d37ddc120bbb2bf62b267b1673add5b0c23d4'; BASE='Qwen/Qwen3-4B-Instruct-2507'
Q='PLAN_STATE_TO_RFC6902_SINGLE_MUTATION_WITH_CONSTRAINT_PRESERVATION_AND_DETERMINISTIC_PATCH_APPLY'; QFP='5c7fd6bf73441b1fe817eb312d7eeef889d5c87edd64b241b83dd880ab973095'
ROOT=Path('work'); DL=ROOT/'download'; ESC=ROOT/'escrow'; OUT=ROOT/'out'
for p in (DL,ESC,OUT): p.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'LUCIA-AA-R22592-C73-one-use'}
def jw(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def get(u,lim=6000000):
 with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=90) as r:
  b=r.read(lim+1)
  if len(b)>lim: raise RuntimeError('META_TOO_LARGE')
  return b
def api(u):return json.loads(get(u).decode())
def txt(u):return get(u,2000000).decode('utf-8','replace')
def sib(i,n):
 for x in i.get('siblings') or []:
  if x.get('rfilename')==n:
   l=x.get('lfs') or {}; return {'size':l.get('size') or x.get('size'),'sha256':l.get('sha256') or l.get('oid'),'blob_id':x.get('blobId')}
 return {}
def dl(u,p):
 h=hashlib.sha256();z=0
 with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=300) as r,p.open('wb') as f:
  while 1:
   b=r.read(1<<20)
   if not b:break
   f.write(b);h.update(b);z+=len(b)
 return z,h.hexdigest()
def hdr(p):
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n).decode().rstrip())
 return 8+n,h
def arr(p,start,s):
 a,b=s['data_offsets'];
 with p.open('rb') as f:f.seek(start+a);buf=f.read(b-a)
 if s['dtype']=='F32':x=np.frombuffer(buf,dtype='<f4').astype(np.float32)
 elif s['dtype']=='F16':x=np.frombuffer(buf,dtype='<f2').astype(np.float32)
 elif s['dtype']=='BF16':
  u=np.frombuffer(buf,dtype='<u2').astype(np.uint32);x=(u<<16).view(np.float32)
 else:raise RuntimeError('UNSUPPORTED_DTYPE_'+s['dtype'])
 return x.reshape(tuple(s['shape'])).copy()
def fk(n):
 for suf,side in [('.lora_a.weight','A'),('.lora_b.weight','B'),('.lora_A.weight','A'),('.lora_B.weight','B'),('.lora_a','A'),('.lora_b','B'),('.lora_A','A'),('.lora_B','B')]:
  if n.endswith(suf):return n[:-len(suf)],side
 m=re.match(r'^(.*)\.lora\.linear([12])\.weight$',n)
 return (m.group(1), 'A' if m.group(2)=='1' else 'B') if m else (None,None)
def static(p,scale):
 start,h=hdr(p);spec={k:v for k,v in h.items() if k!='__metadata__'};A={};B={};other=[];dc={}
 for n,s in spec.items():
  dc[s['dtype']]=dc.get(s['dtype'],0)+1;k,side=fk(n)
  if side=='A':A[k]=arr(p,start,s)
  elif side=='B':B[k]=arr(p,start,s)
  else:other.append({'name':n,'shape':s['shape'],'dtype':s['dtype']})
 ks=sorted(set(A)|set(B));bad=[k for k in ks if k not in A or k not in B]
 if bad or not ks:raise RuntimeError('PAIR_DETECTION_FAIL')
 rows=[];bm={};bl={};tot=0.;zero=0;finite=True
 for k in ks:
  a,b=A[k],B[k];z=not(np.count_nonzero(a) and np.count_nonzero(b));zero+=int(z);finite&=bool(np.isfinite(a).all() and np.isfinite(b).all())
  er=min(int(np.linalg.matrix_rank(a.astype('float64'))),int(np.linalg.matrix_rank(b.astype('float64'))));na=float(np.linalg.norm(a.astype('float64')));nb=float(np.linalg.norm(b.astype('float64')));e=float((scale*na*nb)**2);tot+=e
  lm=re.search(r'\.layers\.(\d+)\.',k);ly=int(lm.group(1)) if lm else -1;mod=k.split('.')[-1]
  m=bm.setdefault(mod,{'pairs':0,'e':0.,'zero_pairs':0,'rank_min':999999,'rank_max':0});m['pairs']+=1;m['e']+=e;m['zero_pairs']+=int(z);m['rank_min']=min(m['rank_min'],er);m['rank_max']=max(m['rank_max'],er)
  l=bl.setdefault(str(ly),{'pairs':0,'e':0.,'modules':set()});l['pairs']+=1;l['e']+=e;l['modules'].add(mod)
  rows.append({'component_id':k,'layer':ly,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'effective_rank_upper_bound':er,'zero_pair':z,'a_fro_norm':na,'b_fro_norm':nb})
 for d in bm.values():d['energy_ratio']=d.pop('e')/tot
 for d in bl.values():d['energy_ratio']=d.pop('e')/tot;d['modules']=sorted(d['modules'])
 return {'schema':'R22592_C73_MLX_ADAPTER_OPERATOR_ARCHAEOLOGY_V1','operator':{'tensor_count':len(spec),'complete_pairs':len(ks),'alive_pairs':len(ks)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(other),'auxiliary_tensors':other,'dtype_counts':dc,'target_modules_actual':sorted(bm),'by_module':dict(sorted(bm.items())),'by_layer':dict(sorted(bl.items(),key=lambda x:int(x[0]))),'pair_shape_rank_inventory':rows,'scale_used_for_proxy':scale,'energy_proxy_definition':'(scale*||A||F*||B||F)^2 factor-only ranking; NOT Delta-W norm, causal importance, or capability','delta_w_reconstructed':False,'safetensors_backend':'native_header_dtype_parser'}}
def preflight():
 import tempfile
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'x.safetensors';a=np.array([[1,2],[3,4]],dtype=np.float32);b=np.array([[1,0],[0,1]],dtype=np.float16);bf=np.array([[1.5,-2.25],[3,4.5]],dtype=np.float32);u=(bf.view(np.uint32)>>16).astype('<u2');items=[('model.layers.0.q_proj.lora_a',a,'F32'),('model.layers.0.q_proj.lora_b',b,'F16'),('model.layers.1.up_proj.lora_a',u,'BF16'),('model.layers.1.up_proj.lora_b',u,'BF16')];off=0;H={};data=[]
  for n,x,d in items:
   raw=x.tobytes();H[n]={'dtype':d,'shape':[2,2],'data_offsets':[off,off+len(raw)]};off+=len(raw);data.append(raw)
  hb=json.dumps(H,separators=(',',':')).encode();hb+=b' '*((8-len(hb)%8)%8);p.write_bytes(struct.pack('<Q',len(hb))+hb+b''.join(data));s=static(p,2.5);assert s['operator']['complete_pairs']==2 and s['operator']['all_finite']
 print(json.dumps({'PRE_SOURCE_PASS':True,'F32_F16_BF16':True,'source_calls':0}))
def main():
 info=api(f'https://huggingface.co/api/models/{R}?blobs=true'); wm=sib(info,W);lic=(info.get('cardData') or {}).get('license')
 if info.get('sha')!=V or lic!='apache-2.0' or int(wm.get('size') or -1)!=SZ or str(wm.get('sha256')).replace('sha256:','')!=SHA:raise RuntimeError('ATOMIC_PIN_FAIL')
 readme=txt(f'https://huggingface.co/{R}/raw/{V}/README.md');cfg_txt=txt(f'https://huggingface.co/{R}/raw/{V}/adapter_config.json');cfg=json.loads(cfg_txt)
 if BASE not in readme:raise RuntimeError('BASE_ID_CARD_DRIFT')
 for pat in [r'r=8',r'alpha=20',r'dropout=0',r'Layers:\s*32',r'Training iters:\s*600']:
  if not re.search(pat,readme,re.I):raise RuntimeError('TRAINING_FACT_DRIFT_'+pat)
 exact=re.search(r'(?i)(?:training(?:-time)?\s+)?base(?:\s+model)?\s+(?:revision|commit|snapshot)\s*[:=` ]+\s*([0-9a-f]{40})',readme)
 if exact:raise RuntimeError('UNEXPECTED_EXACT_BASE_REQUIRES_STOP')
 scale=2.5
 lp=cfg.get('lora_parameters') if isinstance(cfg,dict) else None
 if isinstance(lp,dict) and isinstance(lp.get('scale'),(int,float)):scale=float(lp['scale'])
 pin={'schema':'R22592_C73_ATOMIC_SOURCE_PIN_V1','question':Q,'question_fingerprint':QFP,'adapter_repo':R,'adapter_revision':V,'adapter_file':W,'adapter_lfs':wm,'license':lic,'base_model':BASE,'training_base_revision':None,'training_base_revision_proven':False,'weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'historical_checkpoint_downloads_authorized':0,'publisher_metrics_credit':0,'final_equals_step600_metadata_sha':True,'immutable_readme_sha256':hashlib.sha256(readme.encode()).hexdigest(),'immutable_adapter_config_sha256':hashlib.sha256(cfg_txt.encode()).hexdigest(),'training_facts':{'rank':8,'alpha':20,'scale':scale,'dropout':0,'layers':32,'training_iters':600,'adapter_config':cfg}}
 jw('R22592_C73_ATOMIC_SOURCE_PIN.json',pin)
 p=DL/W;n,h=dl(f'https://huggingface.co/{R}/resolve/{V}/{W}?download=true',p)
 if (n,h)!=(SZ,SHA):raise RuntimeError('DOWNLOADED_WEIGHT_IDENTITY_MISMATCH')
 jw('R22592_C73_SOURCE_INGRESS_RECEIPT.json',{'schema':'R22592_C73_SOURCE_INGRESS_RECEIPT_V1','source_consumed':True,'adapter_weight_get_count':1,'adapter_bytes':n,'adapter_sha256':h,'adapter_revision':V,'adapter_file':W,'base_weight_get_count':0,'base_bytes':0,'model_forward_count':0,'historical_checkpoint_weight_get_count':0,'gguf_weight_get_count':0,'written_before_deep_static':True})
 ad=ESC/W;os.replace(p,ad);s=static(ad,scale);s['source']={'repo':R,'revision':V,'file':W,'sha256':h,'size':n};s['training_facts']=pin['training_facts'];s['claim_boundary']='E1_STATIC_MLX_ADAPTER_ONLY_EXACT_TRAINING_BASE_REVISION_UNPROVEN';jw('R22592_C73_MLX_ADAPTER_OPERATOR_ARCHAEOLOGY.json',s)
 o=s['operator'];
 if o['zero_pairs'] or not o['all_finite']:raise RuntimeError('OPERATOR_VIABILITY_FAIL')
 jw('R22592_C73_RAWFREE_BRAIN_MATERIAL.json',{'schema':'R22592_C73_RAWFREE_BRAIN_MATERIAL_V1','question':Q,'question_fingerprint':QFP,'adapter_live_static':True,'complete_pairs':o['complete_pairs'],'alive_pairs':o['alive_pairs'],'target_modules_actual':o['target_modules_actual'],'base_behavior_executed':False,'fresh_behavior_credit':0,'causal_credit':0,'e3_increment':0,'e4_plus_increment':0,'e5_increment':0,'reason':'Exact training-time Base revision unproven; static viability and source-free verifier are not Adapter capability evidence.','cleanroom_reusable_semantics':'RFC6902 single-operation state mutation with dual apply and invariant checking'})
 return {'status':'PASS','adapter_sha256':h,'adapter_size':n,'operator_pairs':o['complete_pairs'],'alive_pairs':o['alive_pairs'],'base_bytes':0,'behavior_executed':False}
if __name__=='__main__':
 import sys
 if '--preflight' in sys.argv:preflight();raise SystemExit(0)
 res=None;err=None
 try:res=main()
 except Exception as e:err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-12:])}
 finally:
  shutil.rmtree(DL,ignore_errors=True);shutil.rmtree(ESC,ignore_errors=True);raw=[str(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.gguf','.pkl','.pickle'}];consumed=(OUT/'R22592_C73_SOURCE_INGRESS_RECEIPT.json').exists();jw('R22592_C73_FINAL_DELETION_RECEIPT.json',{'schema':'R22592_C73_FINAL_DELETION_RECEIPT_V1','result':res,'error':err,'raw_remaining':raw,'raw_remaining_count':len(raw),'post_delete_pass':not raw,'source_consumed':consumed,'base_behavior_executed':False,'base_bytes':0,'model_forward_count':0})
 print(json.dumps({'PASS':err is None and res is not None and not raw,'result':res,'error':err,'deletion':not raw},indent=2))
 if err or not res or raw:raise SystemExit(1)
