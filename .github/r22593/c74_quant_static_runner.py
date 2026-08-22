from __future__ import annotations
import array, collections, hashlib, json, math, mmap, os, re, shutil, struct, sys, tempfile, time, traceback, urllib.request
from pathlib import Path

ROOT=Path('work'); RAW=ROOT/'raw'; OUT=ROOT/'out'
for p in (RAW,OUT): p.mkdir(parents=True,exist_ok=True)
REPO='codelion/Qwen3-0.6B-accuracy-recovery-lora'
REV='3d95d847c6fa0942db30ff5d005bfd3729220d3d'
WEIGHT='adapter_model.safetensors'
EXPECTED_SIZE=161533160
EXPECTED_SHA='431f3ba2744290b3dddebd5e53b5105b677632e9d7844c31fee8d8d45ec3b95e'
CONFIG_SHA='3beba897a165c70f109dde302a30d892ea95016a936d17d30f36b327eb7b3ccb'
README_SHA='6794acfaa4fbb1378e91230a9318556820cbee089fa46071ebad0d2ed1a0709d'
BASE='Qwen/Qwen3-0.6B'
QUESTION='INT4_QUANTIZED_BASE_PLUS_LORA_TO_FP_REFERENCE_DISTRIBUTION_RECOVERY_ON_SOURCE_FREE_TOKEN_SEQUENCE_SUITE'
QFP='babdd3b8a704db606f8ffedd580336d061044bf4e0a24c1983ac9b1c8ba9757d'
UA={'User-Agent':'LUCIA-AA-R22593-C74-source'}

def jw(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def hbytes(b): return hashlib.sha256(b).hexdigest()
def get(url,limit=5_000_000):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read(limit+1)
        if len(b)>limit: raise RuntimeError('META_TOO_LARGE')
        return b
def getj(url): return json.loads(get(url).decode())
def small_text(name): return get(f'https://huggingface.co/{REPO}/raw/{REV}/{name}').decode('utf-8','replace')
def lfs(info,name):
    for x in info.get('siblings') or []:
        if x.get('rfilename')==name:
            z=x.get('lfs') or {}
            return {'name':name,'size':z.get('size') or x.get('size'),'sha256':str(z.get('sha256') or z.get('oid') or '').replace('sha256:',''),'blob_id':x.get('blobId')}
    return None

def download_one(dst):
    url=f'https://huggingface.co/{REPO}/resolve/{REV}/{WEIGHT}?download=true'
    hh=hashlib.sha256(); n=0
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=900) as r, dst.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b); hh.update(b); n+=len(b)
    return n,hh.hexdigest()

def read_header(path):
    with path.open('rb') as f:
        b=f.read(8)
        if len(b)!=8: raise RuntimeError('SHORT_SAFETENSORS')
        hlen=struct.unpack('<Q',b)[0]
        if hlen<=0 or hlen>100_000_000: raise RuntimeError('BAD_HEADER_LEN')
        hdr=json.loads(f.read(hlen))
    return hlen,hdr

def tensor_stats(mm,data0,ent):
    dtype=ent['dtype']; shape=list(ent['shape']); a,b=ent['data_offsets']; start=data0+a; end=data0+b
    raw=mm[start:end]; cnt=1
    for d in shape: cnt*=d
    if dtype=='F32': width=4
    elif dtype in ('F16','BF16'): width=2
    else: raise RuntimeError('UNSUPPORTED_DTYPE:'+dtype)
    if len(raw)!=cnt*width: raise RuntimeError('SIZE_SHAPE_MISMATCH')
    nz=0; finite=True; ss=0.0; maxabs=0.0
    if dtype=='F32':
        ar=array.array('f'); ar.frombytes(raw)
        if sys.byteorder!='little': ar.byteswap()
        for x in ar:
            x=float(x); finite &= math.isfinite(x)
            if x!=0.0: nz+=1
            ax=abs(x); maxabs=max(maxabs,ax); ss+=x*x
    elif dtype=='F16':
        for (x,) in struct.iter_unpack('<e',raw):
            x=float(x); finite &= math.isfinite(x)
            if x!=0.0: nz+=1
            ax=abs(x); maxabs=max(maxabs,ax); ss+=x*x
    else:
        # BF16 -> float32 bits
        for (u,) in struct.iter_unpack('<H',raw):
            x=struct.unpack('<f',struct.pack('<I',u<<16))[0]
            finite &= math.isfinite(x)
            if x!=0.0: nz+=1
            ax=abs(x); maxabs=max(maxabs,ax); ss+=x*x
    return {'dtype':dtype,'shape':shape,'numel':cnt,'nonzero':nz,'finite':bool(finite),'sumsq':ss,'fro':math.sqrt(ss),'max_abs':maxabs}

def static_inspect(path,cfg):
    hlen,hdr=read_header(path); data0=8+hlen
    A={};B={};aux=[];dtype_counts=collections.Counter(); tensor_rows=[]
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for name in sorted(k for k in hdr if k!='__metadata__'):
            ent=hdr[name]; st=tensor_stats(mm,data0,ent); dtype_counts[st['dtype']]+=1
            row={'name':name,**st}; tensor_rows.append(row)
            if '.lora_A.' in name: A[name.split('.lora_A.')[0]]=row
            elif '.lora_B.' in name: B[name.split('.lora_B.')[0]]=row
            else: aux.append({'name':name,'dtype':st['dtype'],'shape':st['shape'],'numel':st['numel']})
    keys=sorted(set(A)|set(B)); incomplete=[k for k in keys if k not in A or k not in B]
    if incomplete: raise RuntimeError('INCOMPLETE_LORA_PAIR')
    scale=float(cfg['lora_alpha'])/float(cfg['r']); total=0.0; pairs=[]; bymod={}; bylayer={}; zero=0; allfinite=True
    for k in keys:
        a,b=A[k],B[k]; alive=a['nonzero']>0 and b['nonzero']>0; zero+=not alive; allfinite &= a['finite'] and b['finite']
        energy=(scale*a['fro']*b['fro'])**2; total+=energy
        mod=k.split('.')[-1]; m=re.search(r'\.layers\.(\d+)\.',k); layer=int(m.group(1)) if m else -1
        inner=min(a['shape']) if a['shape'] else None
        pairs.append({'component_id':k,'module':mod,'layer':layer,'a_shape':a['shape'],'b_shape':b['shape'],'factor_inner_dim':inner,'alive':alive,'factor_energy_proxy':energy})
        x=bymod.setdefault(mod,{'pairs':0,'alive_pairs':0,'factor_energy_proxy':0.0}); x['pairs']+=1;x['alive_pairs']+=int(alive);x['factor_energy_proxy']+=energy
        y=bylayer.setdefault(str(layer),{'pairs':0,'alive_pairs':0,'factor_energy_proxy':0.0,'modules':set()});y['pairs']+=1;y['alive_pairs']+=int(alive);y['factor_energy_proxy']+=energy;y['modules'].add(mod)
    for d in bymod.values(): d['energy_ratio']=d['factor_energy_proxy']/total if total>0 else None
    for d in bylayer.values(): d['energy_ratio']=d['factor_energy_proxy']/total if total>0 else None; d['modules']=sorted(d['modules'])
    return {'schema':'R22593_C74_OPERATOR_ARCHAEOLOGY_V1','header_length':hlen,'tensor_count':len(tensor_rows),'dtype_counts':dict(dtype_counts),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':bool(allfinite),'auxiliary_tensor_count':len(aux),'auxiliary_tensors':aux,'target_modules_actual':sorted(bymod),'by_module':dict(sorted(bymod.items())),'by_layer':dict(sorted(bylayer.items(),key=lambda kv:int(kv[0]))),'pair_inventory':pairs,'total_factor_energy_proxy':total,'energy_proxy_definition':'(alpha/r * ||A||F * ||B||F)^2; non-reconstructive factor-only static ranking, NOT causal importance or DeltaW norm','zero_proxy_mass_abstain':total<=0}

def synthetic_file(path):
    # three tiny tensors exercising F32/F16/BF16
    t1=struct.pack('<4f',1.0,-2.0,0.0,3.5)
    t2=struct.pack('<4e',1.5,-.5,0.0,2.0)
    vals=[1.0,-2.0,0.0,3.0]; t3=b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16) for v in vals)
    offs=0; hdr={}
    chunks=[]
    for n,dtype,shape,b in [('a','F32',[2,2],t1),('b','F16',[2,2],t2),('c','BF16',[2,2],t3)]:
        hdr[n]={'dtype':dtype,'shape':shape,'data_offsets':[offs,offs+len(b)]};offs+=len(b);chunks.append(b)
    hb=json.dumps(hdr,separators=(',',':')).encode(); path.write_bytes(struct.pack('<Q',len(hb))+hb+b''.join(chunks))

def preflight():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'tiny.safetensors';synthetic_file(p); h,hdr=read_header(p);data0=8+h
        with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            stats={k:tensor_stats(mm,data0,v) for k,v in hdr.items()}
        assert stats['a']['nonzero']==3 and stats['b']['nonzero']==3 and stats['c']['nonzero']==3
        assert all(v['finite'] for v in stats.values())
        assert abs(stats['a']['sumsq']-17.25)<1e-6 and abs(stats['c']['sumsq']-14.0)<1e-6
    print(json.dumps({'PRE_SOURCE_PASS':True,'dependencies':0,'supported_dtypes':['F32','F16','BF16'],'numpy_backend':False,'torch_backend':False},sort_keys=True))

def main():
    t0=time.time(); err=None; consumed=False; raw=RAW/WEIGHT
    try:
        info=getj(f'https://huggingface.co/api/models/{REPO}?blobs=true')
        if info.get('sha')!=REV: raise RuntimeError('ADAPTER_REVISION_DRIFT')
        lic=(info.get('cardData') or {}).get('license')
        if lic!='apache-2.0': raise RuntimeError('LICENSE_DRIFT')
        lf=lfs(info,WEIGHT)
        if not lf or int(lf.get('size') or -1)!=EXPECTED_SIZE or lf.get('sha256')!=EXPECTED_SHA: raise RuntimeError('LFS_IDENTITY_DRIFT')
        cfg_text=small_text('adapter_config.json'); readme=small_text('README.md')
        if hbytes(cfg_text.encode())!=CONFIG_SHA or hbytes(readme.encode())!=README_SHA: raise RuntimeError('IMMUTABLE_METADATA_HASH_DRIFT')
        cfg=json.loads(cfg_text)
        if cfg.get('peft_type')!='LORA' or cfg.get('r')!=64 or cfg.get('lora_alpha')!=128 or cfg.get('modules_to_save') not in (None,[]): raise RuntimeError('CONFIG_CONTRACT_DRIFT')
        targets=sorted(cfg.get('target_modules') or [])
        expected=sorted(['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        if targets!=expected: raise RuntimeError('TARGET_MODULE_DRIFT')
        jw('R22593_C74_ATOMIC_SOURCE_PIN.json',{'schema':'R22593_C74_ATOMIC_SOURCE_PIN_V1','question':QUESTION,'question_fingerprint':QFP,'adapter_repo':REPO,'adapter_revision':REV,'adapter_lfs':lf,'license':'apache-2.0','base_model_declared':BASE,'exact_training_base_revision_proven':False,'base_download_authorized':False,'weight_gets_before_pin':0,'model_weight_bytes_before_pin':0,'publisher_metrics_credit':0,'metadata_audit_run':32576006813})
        n,h=download_one(raw); consumed=True
        # MUST precede identity validation/deep static: network bytes have already been consumed.
        jw('R22593_C74_SOURCE_INGRESS_RECEIPT.json',{'schema':'R22593_C74_SOURCE_INGRESS_RECEIPT_V1','source_consumed':True,'adapter_weight_get_count':1,'observed_size':n,'observed_sha256':h,'expected_size':EXPECTED_SIZE,'expected_sha256':EXPECTED_SHA,'written_before_identity_validation':True,'base_weight_get_count':0,'base_bytes':0})
        if (n,h)!=(EXPECTED_SIZE,EXPECTED_SHA): raise RuntimeError('DOWNLOADED_WEIGHT_IDENTITY_MISMATCH')
        op=static_inspect(raw,cfg); jw('R22593_C74_OPERATOR_ARCHAEOLOGY.json',op)
        if op['zero_pairs'] or not op['all_finite'] or op['auxiliary_tensor_count'] or op['zero_proxy_mass_abstain']: raise RuntimeError('OPERATOR_VIABILITY_FAIL')
        material={'schema':'R22593_C74_RAWFREE_BRAIN_MATERIAL_V1','question':QUESTION,'question_fingerprint':QFP,'specimen':'QUANTIZATION_ACCURACY_RECOVERY_LORA','adapter_live_static':True,'complete_pairs':op['complete_pairs'],'alive_pairs':op['alive_pairs'],'target_modules_actual':op['target_modules_actual'],'exact_training_base_revision_proven':False,'base_behavior_executed':False,'fresh_behavior_credit':0,'causal_credit':0,'e3_increment':0,'e4_plus_increment':0,'e5_increment':0,'cleanroom_verifier':'local source-free quantization recovery metric preseal only; does not retro-credit Adapter capability','reason':'Exact training-time Base revision unproven. Preserve operator anatomy and quantization-recovery hypothesis without behavior/causal overclaim.'}
        jw('R22593_C74_RAWFREE_BRAIN_MATERIAL.json',material)
        jw('R22593_C74_RUN_RESULT.json',{'schema':'R22593_C74_RUN_RESULT_V1','status':'PASS','source_consumed':True,'operator_pairs':op['complete_pairs'],'alive_pairs':op['alive_pairs'],'base_bytes':0,'model_forward_count':0,'behavior_executed':False,'elapsed_seconds':time.time()-t0})
    except Exception as e:
        err={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}; jw('R22593_C74_RUN_ERROR.json',err); raise
    finally:
        if raw.exists(): raw.unlink()
        try: RAW.rmdir()
        except Exception: pass
        remaining=[]
        if RAW.exists(): remaining=[p.relative_to(ROOT).as_posix() for p in RAW.rglob('*') if p.is_file()]
        jw('R22593_C74_FINAL_DELETION_RECEIPT.json',{'schema':'R22593_C74_FINAL_DELETION_RECEIPT_V1','source_consumed':bool(consumed),'raw_remaining':remaining,'raw_remaining_count':len(remaining),'post_delete_pass':len(remaining)==0,'base_weight_files_in_out':0,'error':err,'elapsed_seconds':time.time()-t0})

if __name__=='__main__':
    if '--preflight' in sys.argv: preflight()
    else: main()
