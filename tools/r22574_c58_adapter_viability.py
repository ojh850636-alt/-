#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rank_of(a):
 if a.size==0:return 0
 s=np.linalg.svd(a.astype(np.float64,copy=False),compute_uv=False)
 if not len(s):return 0
 tol=max(a.shape)*float(s[0])*np.finfo(np.float64).eps
 return int(np.sum(s>tol))
def pair_base(k): return re.sub(r'\.lora_[AB](?:\.[^.]+)?\.weight$','',k)
def projection(pid): return pid.split('.')[-1]
def layer(pid):
 m=re.search(r'\.layers\.(\d+)\.',pid); return int(m.group(1)) if m else None
def main():
 if len(sys.argv)!=5: raise SystemExit('usage: script adapter.safetensors adapter_config.json out.json expected_sha')
 w,cfgp,out,expected=sys.argv[1:]; p=Path(w); cfg=json.loads(Path(cfgp).read_text()); actual=sha256(p)
 if actual!=expected: raise RuntimeError(f'SHA mismatch {actual} != {expected}')
 if p.stat().st_size!=15220968: raise RuntimeError(f'size mismatch {p.stat().st_size}')
 tens=load_file(str(p)); groups={}; unexpected=[]; nonfinite_tensors=0; zero_tensors=0
 for k,a in tens.items():
  arr=np.asarray(a)
  if not np.isfinite(arr).all(): nonfinite_tensors+=1
  if not np.any(arr): zero_tensors+=1
  if '.lora_A' in k or '.lora_B' in k:
   pid=pair_base(k); side='A' if '.lora_A' in k else 'B'; groups.setdefault(pid,{})[side]=(k,arr)
  else: unexpected.append({'key':k,'shape':list(arr.shape),'dtype':str(arr.dtype),'elements':int(arr.size)})
 alpha=float(cfg['lora_alpha']); r_cfg=int(cfg['r']); scale=alpha/r_cfg
 pairs=[]; proj_energy={}; layer_energy={}; live=0; incomplete=0; collapsed=0
 for pid in sorted(groups):
  g=groups[pid]
  if 'A' not in g or 'B' not in g:
   incomplete+=1; pairs.append({'pair_id':pid,'complete':False}); continue
  _,A=g['A']; _,B=g['B']; finite=bool(np.isfinite(A).all() and np.isfinite(B).all()); za=not bool(np.any(A)); zb=not bool(np.any(B))
  na=float(np.linalg.norm(A.astype(np.float64,copy=False))); nb=float(np.linalg.norm(B.astype(np.float64,copy=False))); eff=float(scale*na*nb)
  ra=rank_of(A); rb=rank_of(B); er=min(ra,rb); is_live=finite and eff>0 and er>0
  if is_live: live+=1
  if er<min(r_cfg,A.shape[0] if A.ndim else 0,B.shape[-1] if B.ndim else 0): collapsed+=1
  pr=projection(pid); ly=layer(pid); e=eff*eff; proj_energy[pr]=proj_energy.get(pr,0.0)+e
  if ly is not None: layer_energy[str(ly)]=layer_energy.get(str(ly),0.0)+e
  pairs.append({'pair_id':pid,'complete':True,'projection':pr,'layer':ly,'A_shape':list(A.shape),'B_shape':list(B.shape),'A_zero':za,'B_zero':zb,'finite':finite,'A_rank':ra,'B_rank':rb,'effective_rank_upper_bound':er,'factor_norm_A':na,'factor_norm_B':nb,'effective_scalar':eff,'live':is_live})
 total_e=sum(proj_energy.values()) or 1.0; proj_pct={k:v/total_e for k,v in sorted(proj_energy.items())}; layer_pct={k:v/total_e for k,v in sorted(layer_energy.items(),key=lambda kv:int(kv[0]))}
 viability=(len(groups)>0 and live>0 and incomplete==0 and nonfinite_tensors==0)
 report={'schema':'LAA.R22574.C58AdapterViability.v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':{'sha256':actual,'size':p.stat().st_size},'config':{'r':r_cfg,'alpha':alpha,'scale':scale,'target_modules':cfg.get('target_modules'),'modules_to_save':cfg.get('modules_to_save'),'base_model_name_or_path':cfg.get('base_model_name_or_path'),'revision':cfg.get('revision'),'peft_version':cfg.get('peft_version')},'tensor_count':len(tens),'pair_count':len(groups),'complete_pairs':sum(bool(x.get('complete')) for x in pairs),'live_pairs':live,'incomplete_pairs':incomplete,'zero_tensor_count':zero_tensors,'nonfinite_tensor_count':nonfinite_tensors,'rank_collapse_pair_count':collapsed,'unexpected_non_lora_tensors':unexpected,'projection_energy_fraction':proj_pct,'layer_energy_fraction':layer_pct,'pairs':pairs,'viability_gate':'GREEN' if viability else 'RED','base_download_authorized':bool(viability),'claim':'STATIC_E1_OPERATOR_VIABILITY_ONLY','raw_values_exported':False,'reconstructable_delta_exported':False}
 Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:report[k] for k in ['tensor_count','pair_count','live_pairs','zero_tensor_count','nonfinite_tensor_count','rank_collapse_pair_count','viability_gate','base_download_authorized']},indent=2)); return 0 if viability else 2
if __name__=='__main__': raise SystemExit(main())
