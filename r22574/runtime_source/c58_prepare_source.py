#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE='Qwen/Qwen3-1.7B'
REV='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
DATA='RampPublic/portallib-tasks'
DATA_REV='ffc3c0e44f529bf64a5ae62ed5db090952db97ea'
EXPECTED={
 'model-00001-of-00002.safetensors':(3441185608,'169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5'),
 'model-00002-of-00002.safetensors':(622329984,'912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'),
}
SAFE_META=['config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json']
FORBID={'.bin','.pt','.pth','.pkl','.pickle','.h5','.msgpack','.py'}

def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def verify_adapter_escrow(root:Path):
 man=json.loads((root/'ESCROW_MANIFEST.json').read_text())
 got=[]
 for x in man['files']:
  p=root/x['path']; assert p.is_file(); assert p.stat().st_size==x['bytes']; assert sha(p)==x['sha256']; got.append(x['path'])
 assert 'rte/adapter_model.safetensors' in got and 'copa/adapter_model.safetensors' in got
 return man

def metadata_authority(api:HfApi):
 info=api.model_info(BASE,revision=REV,files_metadata=True)
 assert info.sha==REV,(info.sha,REV)
 siblings={x.rfilename:x for x in info.siblings}
 out={}
 for name,(size,digest) in EXPECTED.items():
  s=siblings[name]; lfs=getattr(s,'lfs',None)
  lsize=getattr(lfs,'size',None) if lfs else getattr(s,'size',None)
  lsha=getattr(lfs,'sha256',None) if lfs else None
  assert int(lsize)==size,(name,lsize,size)
  assert str(lsha)==digest,(name,lsha,digest)
  out[name]={'bytes':size,'sha256':digest}
 for name in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'):
  assert name in siblings,name
 return {'repo':BASE,'resolved_revision':info.sha,'weights':out,'safe_metadata_present':[x for x in SAFE_META if x in siblings]}

def download_meta_and_abi(adapter_root:Path,meta_dir:Path):
 snapshot_download(repo_id=BASE,revision=REV,allow_patterns=SAFE_META,local_dir=meta_dir)
 tok=AutoTokenizer.from_pretrained(meta_dir,local_files_only=True,trust_remote_code=False)
 cfg=AutoConfig.from_pretrained(meta_dir,local_files_only=True,trust_remote_code=False)
 acfg=json.loads((adapter_root/'rte'/'adapter_config.json').read_text())
 old=torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
 try:
  with torch.device('meta'):
   model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
  missing=[]; wrong=[]
  for path in acfg['target_modules']:
   try: mod=model.get_submodule(path)
   except Exception: missing.append(path); continue
   if not isinstance(mod,torch.nn.Linear): wrong.append(path)
  assert not missing and not wrong,{'missing':missing[:4],'wrong_type':wrong[:4]}
 finally:
  torch.set_default_dtype(old)
  try: del model
  except Exception: pass
 for choice in (' True',' False'):
  ids=tok(choice,add_special_tokens=False).input_ids; assert ids,choice
 return {'tokenizer_class':tok.__class__.__name__,'model_type':cfg.model_type,'target_count':len(acfg['target_modules']),'target_paths_all_linear':True,'rte_choice_nonempty':True}

def prepare_natural(out:Path):
 ds=load_dataset(DATA,revision=DATA_REV,split='validation')
 rows=[]
 for x in ds:
  if str(x['task'])=='rte': rows.append({'task':'rte','prompt':str(x['prompt']),'choices':[str(c) for c in x['choices']],'gold_idx':int(x['gold_idx'])})
 assert rows
 assert all(len(x['choices'])==2 and x['choices']==[' True',' False'] for x in rows)
 out.write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
 return {'rows':len(rows),'ephemeral_sha256':sha(out),'raw_text_export_forbidden':True}

def download_weights(base_dir:Path):
 allow=SAFE_META+list(EXPECTED)
 snapshot_download(repo_id=BASE,revision=REV,allow_patterns=allow,local_dir=base_dir)
 files=[]
 for p in sorted(x for x in base_dir.rglob('*') if x.is_file()):
  if p.suffix.lower() in FORBID: raise RuntimeError(f'forbidden source file {p}')
  rel=p.relative_to(base_dir).as_posix(); files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
 for name,(size,digest) in EXPECTED.items():
  p=base_dir/name; assert p.stat().st_size==size; assert sha(p)==digest
  with safe_open(p,framework='pt') as f: _=list(f.keys())[:1]
 assert not list(base_dir.rglob('*.bin'))
 return files

def build_source_escrow(base_dir:Path,adapter_root:Path,escrow:Path,chunk:int):
 if escrow.exists(): shutil.rmtree(escrow)
 escrow.mkdir(parents=True)
 manifest={'schema':'R22574_C58_FULL_SOURCE_ESCROW_V1','base_repo':BASE,'base_revision':REV,'chunk_bytes':chunk,'originals':[],'parts':[],'derivative_adapters':[]}
 md=escrow/'base_meta'; md.mkdir()
 for name in SAFE_META:
  p=base_dir/name
  if p.exists(): shutil.copy2(p,md/name)
 for name in EXPECTED:
  p=base_dir/name; manifest['originals'].append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
  with p.open('rb') as f:
   i=0
   while True:
    b=f.read(chunk)
    if not b: break
    q=escrow/f'{name}.part{i:02d}'; q.write_bytes(b); manifest['parts'].append({'path':q.name,'bytes':len(b),'sha256':sha256_bytes(b),'source':name,'index':i}); i+=1
 ad=escrow/'adapters'; shutil.copytree(adapter_root,ad)
 for p in sorted(x for x in ad.rglob('*') if x.is_file()): manifest['derivative_adapters'].append({'path':p.relative_to(ad).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 (escrow/'SOURCE_ESCROW_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 return manifest

def sha256_bytes(b:bytes): return hashlib.sha256(b).hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adapter-escrow',required=True); ap.add_argument('--base-dir',required=True); ap.add_argument('--meta-dir',required=True); ap.add_argument('--natural-out',required=True); ap.add_argument('--source-escrow',required=True); ap.add_argument('--receipt',required=True); ap.add_argument('--chunk-bytes',type=int,default=450000000)
 a=ap.parse_args(); adapter=Path(a.adapter_escrow); base=Path(a.base_dir); meta=Path(a.meta_dir); natural=Path(a.natural_out); se=Path(a.source_escrow)
 api=HfApi(); rec={'schema':'R22574_C58_SOURCE_PREFLIGHT_V1','adapter_escrow':verify_adapter_escrow(adapter),'base_metadata':metadata_authority(api)}
 rec['preweight_abi']=download_meta_and_abi(adapter,meta); rec['natural_locked']=prepare_natural(natural)
 if base.exists(): shutil.rmtree(base)
 shutil.copytree(meta,base)
 rec['base_files']=download_weights(base)
 rec['source_escrow']=build_source_escrow(base,adapter,se,a.chunk_bytes)
 rec['model_outputs_observed']=0; rec['trust_remote_code']=False; rec['pickle_loaded']=False
 Path(a.receipt).write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'natural_rows':rec['natural_locked']['rows'],'base_weight_bytes':sum(x['bytes'] for x in rec['source_escrow']['originals']),'parts':len(rec['source_escrow']['parts'])},sort_keys=True))
if __name__=='__main__': main()
