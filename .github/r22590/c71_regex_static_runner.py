from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

REPO='wowsocool123/qwen3-4b-regex-lora'
REV='10a26a913fca394e04bcd63fac92ee471edb1073'
EXPECTED_SIZE=132_187_888
EXPECTED_SHA='73971c32bd0af1dcc8b167723eaabf89410af4770f671d2a0974330788ffe8b9'
EXPECTED_CONFIG_SHA='41e9744a57427846bb263c0e0ae5fb6f3d41dcdfbad98a91132ec4a52e67880e'
EXPECTED_README_SHA='856072ea5467c341bd20eb3f81b6c55cd68f0c1111f33796974be8cb558b651f'
EXPECTED_LICENSE='apache-2.0'
EXPECTED_BASE='unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit'
EXPECTED_TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
QUESTION='NL_TO_PYTHON_REGEX_LANGUAGE_WITH_HIDDEN_POSITIVE_NEGATIVE_SET_EXACT_SEMANTICS'
ROOT=Path('work');DL=ROOT/'download';ESC=ROOT/'escrow';OUT=ROOT/'out'
UA={'User-Agent':'LUCIA-AA-R22590-C71-regex-one-use'}
ITEMSIZE={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}


def jwrite(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def api_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:return json.loads(r.read().decode())

def text_url(url,limit=5_000_000):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read(limit+1)
        if len(b)>limit:raise RuntimeError('TEXT_METADATA_TOO_LARGE')
        return b.decode('utf-8','replace')

def sibling(info,name):
    for x in info.get('siblings') or []:
        if x.get('rfilename')==name:
            l=x.get('lfs') or {}
            return {'size':l.get('size') or x.get('size'),'sha256':l.get('sha256') or l.get('oid'),'blob_id':x.get('blobId')}
    return {}

def direct_download(dst):
    url=f'https://huggingface.co/{REPO}/resolve/{REV}/adapter_model.safetensors?download=true'
    req=urllib.request.Request(url,headers=UA);h=hashlib.sha256();n=0
    with urllib.request.urlopen(req,timeout=300) as r,Path(dst).open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b:break
            f.write(b);h.update(b);n+=len(b)
    return {'size':n,'sha256':h.hexdigest()}

class NativeSafeTensorNP:
    def __init__(self,path):
        self.path=Path(path);self.f=self.path.open('rb');raw=self.f.read(8)
        if len(raw)!=8:raise ValueError('TRUNCATED_PREFIX')
        self.header_len=struct.unpack('<Q',raw)[0]
        if not (2<=self.header_len<=100*1024*1024):raise ValueError('BAD_HEADER_LEN')
        hb=self.f.read(self.header_len);self.header=json.loads(hb.decode('utf-8').rstrip());self.start=8+self.header_len
        self.mm=mmap.mmap(self.f.fileno(),0,access=mmap.ACCESS_READ);self._validate()
    def _validate(self):
        size=self.path.stat().st_size;r=[]
        for name,s in self.header.items():
            if name=='__metadata__':continue
            dt=s.get('dtype');sh=s.get('shape');off=s.get('data_offsets')
            if dt not in ITEMSIZE:raise ValueError('UNSUPPORTED_DTYPE:'+str(dt))
            if not isinstance(sh,list) or not isinstance(off,list) or len(off)!=2:raise ValueError('BAD_SPEC:'+name)
            a,b=map(int,off);elems=math.prod(map(int,sh))
            if b-a!=elems*ITEMSIZE[dt]:raise ValueError('BYTE_SIZE_MISMATCH:'+name)
            if a<0 or b<a or self.start+b>size:raise ValueError('OFFSET_RANGE:'+name)
            r.append((a,b,name))
        sr=sorted(r)
        for x,y in zip(sr,sr[1:]):
            if x[1]>y[0]:raise ValueError('OVERLAP:'+x[2]+':'+y[2])
    def keys(self):return [k for k in self.header if k!='__metadata__']
    def spec(self,name):return self.header[name]
    def array_f32(self,name):
        s=self.header[name];a,b=map(int,s['data_offsets']);v=memoryview(self.mm)[self.start+a:self.start+b];dt=s['dtype'];shape=tuple(map(int,s['shape']))
        if dt=='BF16':
            u=np.frombuffer(v,dtype='<u2').astype(np.uint32);arr=(u<<16).view(np.float32)
        elif dt=='F16':arr=np.frombuffer(v,dtype='<f2').astype(np.float32)
        elif dt=='F32':arr=np.frombuffer(v,dtype='<f4').astype(np.float32,copy=True)
        elif dt=='F64':arr=np.frombuffer(v,dtype='<f8').astype(np.float32)
        else:raise ValueError('NON_FLOAT_FACTOR:'+dt)
        return np.asarray(arr).reshape(shape)
    def close(self):self.mm.close();self.f.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

def write_synth_bf16(p):
    vals=np.arange(12,dtype=np.float32).reshape(3,4);u=(vals.view(np.uint32)>>16).astype('<u2');pay=u.tobytes();hdr={'x':{'dtype':'BF16','shape':[3,4],'data_offsets':[0,len(pay)]}}
    hb=json.dumps(hdr,separators=(',',':')).encode();hb+=b' '*((-len(hb))%8);Path(p).write_bytes(struct.pack('<Q',len(hb))+hb+pay);return vals

def preflight():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.safetensors';exp=write_synth_bf16(p)
        with NativeSafeTensorNP(p) as f:got=f.array_f32('x').copy();dt=f.spec('x')['dtype']
    free=shutil.disk_usage('.').free
    ok=bool(np.array_equal(exp,got) and dt=='BF16' and free>1_000_000_000)
    return {'pass':ok,'bf16_dtype':dt,'bf16_sum':float(got.sum()),'numpy':np.__version__,'free_disk':free,'forbidden_backend':'safetensors.safe_open(framework=numpy)','model_source_bytes':0}

def factor_rank(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2:return 0
    g=x@x.T if x.shape[0]<=x.shape[1] else x.T@x
    return int(np.linalg.matrix_rank(g,hermitian=True))

def compkey(name):
    return re.sub(r'\.lora_[AB](?:\.default)?\.weight$','',name)

def analyze(path,cfg):
    with NativeSafeTensorNP(path) as f:
        names=f.keys();A={};B={};other=[];dtypes=collections.Counter();all_elements=0
        for n in names:
            s=f.spec(n);dtypes[s['dtype']]+=1;elems=math.prod(map(int,s['shape']));all_elements+=elems
            if re.search(r'\.lora_A(?:\.default)?\.weight$',n):A[compkey(n)]=n
            elif re.search(r'\.lora_B(?:\.default)?\.weight$',n):B[compkey(n)]=n
            else:other.append({'name':n,'dtype':s['dtype'],'shape':list(map(int,s['shape'])),'elements':elems})
        keys=sorted(set(A)|set(B));incomplete=[k for k in keys if k not in A or k not in B]
        if incomplete:raise RuntimeError('INCOMPLETE_LORA_PAIRS:'+str(len(incomplete)))
        scale=float(cfg['lora_alpha'])/float(cfg['r']);rows=[];bymod=collections.defaultdict(lambda:{'pairs':0,'proxy':0.0,'zero_pairs':0,'rank_min':10**9,'rank_max':0,'elements':0});bylayer=collections.defaultdict(lambda:{'pairs':0,'proxy':0.0,'modules':set(),'elements':0})
        total_proxy=0.0;zero=0;finite=True;lora_elements=0
        for k in keys:
            sa=f.spec(A[k]);sb=f.spec(B[k]);a=f.array_f32(A[k]);b=f.array_f32(B[k]);
            za=(np.count_nonzero(a)==0);zb=(np.count_nonzero(b)==0);z=bool(za or zb);zero+=int(z);fin=bool(np.isfinite(a).all() and np.isfinite(b).all());finite &= fin
            ra=factor_rank(a);rb=factor_rank(b);er=min(ra,rb);na=float(np.linalg.norm(a.astype(np.float64)));nb=float(np.linalg.norm(b.astype(np.float64)));proxy=float((scale*na*nb)**2);total_proxy+=proxy
            ae=math.prod(a.shape);be=math.prod(b.shape);lora_elements+=ae+be;mod=k.split('.')[-1];lm=re.search(r'\.layers\.(\d+)\.',k);layer=int(lm.group(1)) if lm else -1
            m=bymod[mod];m['pairs']+=1;m['proxy']+=proxy;m['zero_pairs']+=int(z);m['rank_min']=min(m['rank_min'],er);m['rank_max']=max(m['rank_max'],er);m['elements']+=ae+be
            l=bylayer[layer];l['pairs']+=1;l['proxy']+=proxy;l['modules'].add(mod);l['elements']+=ae+be
            rows.append({'component_id':k,'layer':layer,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'a_dtype':sa['dtype'],'b_dtype':sb['dtype'],'effective_rank_upper_bound':er,'a_rank':ra,'b_rank':rb,'zero_pair':z,'finite':fin,'a_fro_norm':na,'b_fro_norm':nb,'factor_energy_proxy':proxy,'factor_elements':ae+be})
            del a,b
        for v in bymod.values():v['energy_ratio']=v.pop('proxy')/total_proxy if total_proxy else 0.0
        for v in bylayer.values():v['energy_ratio']=v.pop('proxy')/total_proxy if total_proxy else 0.0;v['modules']=sorted(v['modules'])
        for r in rows:r['factor_energy_ratio']=r.pop('factor_energy_proxy')/total_proxy if total_proxy else 0.0
        actual=set(bymod)
        return {'tensor_count':len(names),'tensor_dtype_counts':dict(sorted(dtypes.items())),'all_tensor_elements':all_elements,'factor_tensor_count':len(A)+len(B),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':bool(finite),'lora_parameter_elements':lora_elements,'auxiliary_tensor_count':len(other),'auxiliary_tensor_elements':sum(x['elements'] for x in other),'auxiliary_tensor_inventory':other,'target_modules_actual':sorted(actual),'advertised_target_modules':sorted(cfg.get('target_modules') or []),'advertised_vs_actual_missing':sorted(set(cfg.get('target_modules') or [])-actual),'advertised_vs_actual_unexpected':sorted(actual-set(cfg.get('target_modules') or [])),'layers_actual':sorted(k for k in bylayer if k>=0),'layer_count':sum(1 for k in bylayer if k>=0),'by_module':dict(sorted(bymod.items())),'by_layer':{str(k):v for k,v in sorted(bylayer.items())},'pair_shape_rank_inventory':rows,'energy_proxy_definition':'(alpha/r*||A||F*||B||F)^2 ranking only; NOT Delta-W norm and NOT causal importance'}

def run():
    for p in (DL,ESC,OUT):p.mkdir(parents=True,exist_ok=True)
    pf=preflight();jwrite('R22590_C71_KNOWN_FAILURE_PREFLIGHT.json',pf)
    if not pf['pass']:raise RuntimeError('KNOWN_FAILURE_PREFLIGHT_FAIL')
    info=api_json('https://huggingface.co/api/models/'+REPO+'?blobs=true');card=info.get('cardData') or {};meta=sibling(info,'adapter_model.safetensors')
    if info.get('sha')!=REV:raise RuntimeError('ADAPTER_REVISION_DRIFT')
    if card.get('license')!=EXPECTED_LICENSE:raise RuntimeError('LICENSE_DRIFT:'+str(card.get('license')))
    if int(meta.get('size') or -1)!=EXPECTED_SIZE or str(meta.get('sha256')).replace('sha256:','')!=EXPECTED_SHA:raise RuntimeError('API_LFS_IDENTITY_DRIFT')
    cfg_txt=text_url(f'https://huggingface.co/{REPO}/raw/{REV}/adapter_config.json');readme=text_url(f'https://huggingface.co/{REPO}/raw/{REV}/README.md');cfg=json.loads(cfg_txt)
    if sha_bytes(cfg_txt.encode())!=EXPECTED_CONFIG_SHA or sha_bytes(readme.encode())!=EXPECTED_README_SHA:raise RuntimeError('TEXT_METADATA_HASH_DRIFT')
    if cfg.get('base_model_name_or_path')!=EXPECTED_BASE or cfg.get('peft_type')!='LORA':raise RuntimeError('BASE_OR_PEFT_DRIFT')
    if cfg.get('r')!=16 or cfg.get('lora_alpha')!=32 or cfg.get('modules_to_save') not in (None,[]):raise RuntimeError('LORA_OR_AUX_DRIFT')
    if set(cfg.get('target_modules') or [])!=EXPECTED_TARGETS:raise RuntimeError('TARGET_DRIFT')
    if cfg.get('revision') is not None:raise RuntimeError('UNEXPECTED_EXACT_BASE_REVISION_APPEARED')
    pin={'schema':'R22590_C71_ATOMIC_SOURCE_PIN_V1','question':QUESTION,'adapter_repo':REPO,'adapter_revision':REV,'adapter_license':card.get('license'),'adapter_lfs':meta,'adapter_config_sha256':EXPECTED_CONFIG_SHA,'readme_sha256':EXPECTED_README_SHA,'base_model_name_or_path':cfg.get('base_model_name_or_path'),'training_base_revision':None,'training_base_revision_proven':False,'r':cfg.get('r'),'lora_alpha':cfg.get('lora_alpha'),'target_modules_declared':sorted(cfg.get('target_modules') or []),'modules_to_save':cfg.get('modules_to_save'),'weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'base_authorized':False,'behavior_authorized':False}
    jwrite('R22590_C71_ATOMIC_SOURCE_PIN.json',pin)
    stage=DL/'adapter_model.safetensors';obs=direct_download(stage)
    ingress={'schema':'R22590_C71_SOURCE_INGRESS_V1','source_consumed':True,'ingress_count':1,'repo':REPO,'revision':REV,'observed_size':obs['size'],'observed_sha256':obs['sha256'],'expected_size':EXPECTED_SIZE,'expected_sha256':EXPECTED_SHA,'identity_pass':obs['size']==EXPECTED_SIZE and obs['sha256']==EXPECTED_SHA}
    jwrite('R22590_C71_SOURCE_INGRESS_RECEIPT.json',ingress)
    if not ingress['identity_pass']:raise RuntimeError('ADAPTER_IDENTITY_MISMATCH_AFTER_INGRESS')
    ad=ESC/'adapter';ad.mkdir(parents=True,exist_ok=False);esc=ad/'adapter_model.safetensors';os.replace(stage,esc);(ad/'adapter_config.json').write_text(cfg_txt);(ad/'README.md').write_text(readme)
    op=analyze(esc,cfg)
    report={'schema':'R22590_C71_REGEX_ADAPTER_OPERATOR_ARCHAEOLOGY_V1','source':{'repo':REPO,'revision':REV,'sha256':obs['sha256'],'size':obs['size'],'license':card.get('license')},'question':QUESTION,'operator':op,'training_base_revision_proven':False,'base_bytes':0,'model_forward_count':0,'behavior_status':'NOT_RUN_EXACT_TRAINING_BASE_REVISION_UNPROVEN','causal_status':'NOT_RUN_NO_BEHAVIOR_GATE','claim_boundary':'E1_ADAPTER_ONLY_OPERATOR_FORENSICS_TRAINING_BASE_REVISION_UNPROVEN'}
    jwrite('R22590_C71_REGEX_ADAPTER_OPERATOR_ARCHAEOLOGY.json',report)
    if op['zero_pairs'] or not op['all_finite']:raise RuntimeError('ADAPTER_OPERATOR_VIABILITY_FAIL')
    return {'status':'PASS','adapter_revision':REV,'adapter_sha256':obs['sha256'],'adapter_size':obs['size'],'complete_pairs':op['complete_pairs'],'alive_pairs':op['alive_pairs'],'base_bytes':0,'model_forward_count':0,'evidence_grade':'E1_ADAPTER_ONLY_OPERATOR_FORENSICS_TRAINING_BASE_REVISION_UNPROVEN'}

result=None;err=None
try:
    args=argparse.ArgumentParser();args.add_argument('--preflight',action='store_true');ns=args.parse_args()
    if ns.preflight:
        print(json.dumps(preflight(),indent=2));raise SystemExit(0)
    result=run()
except SystemExit:
    raise
except Exception as e:
    err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-14:])}
finally:
    if '--preflight' not in sys.argv:
        shutil.rmtree(DL,ignore_errors=True);shutil.rmtree(ESC,ignore_errors=True)
        raw=[]
        if ROOT.exists():
            for p in ROOT.rglob('*'):
                if p.is_file() and (p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.pkl','.pickle'} or p.name in {'tokenizer.json','vocab.json','merges.txt'}):raw.append(str(p))
        ingress_path=OUT/'R22590_C71_SOURCE_INGRESS_RECEIPT.json'
        receipt={'schema':'R22590_C71_FINAL_DELETION_RECEIPT_V1','result':result,'error':err,'source_consumed':ingress_path.exists(),'source_ingress_receipt_exists':ingress_path.exists(),'base_bytes':0,'model_forward_count':0,'raw_remaining':raw,'raw_remaining_count':len(raw),'post_delete_pass':not raw}
        jwrite('R22590_C71_FINAL_DELETION_RECEIPT.json',receipt)
if '--preflight' not in sys.argv:
    if err or result is None:
        print(json.dumps({'PASS':False,'error':err},indent=2));raise SystemExit(1)
    print(json.dumps({'PASS':True,'result':result},indent=2))
