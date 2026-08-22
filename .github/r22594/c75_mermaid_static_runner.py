from __future__ import annotations
import array, collections, hashlib, json, math, mmap, os, re, shutil, struct, sys, tempfile, time, urllib.request, urllib.error, base64, datetime
from pathlib import Path

REPO='Petanque/flow-forge-mermaid-th'
REV='87a0faf78acac3a7dcecf7e3cc8554261160c57a'
EXPECTED_SIZE=73911112
EXPECTED_SHA='c922197747f3863054728514cc323c02c1d72abd981c4a4172a6091b78b78a34'
BASE='Qwen/Qwen2.5-1.5B-Instruct'
QUESTION='NATURAL_LANGUAGE_TO_MERMAID_GRAPH_AST_AND_DENOTATION'
QFP=hashlib.sha256(QUESTION.encode()).hexdigest()
UA={'User-Agent':'LUCIA-AA-R22594-C75-mermaid-source'}
WORK=Path('work'); DL=WORK/'download'; ESC=WORK/'escrow'; OUT=WORK/'out'
for p in (DL,ESC,OUT): p.mkdir(parents=True,exist_ok=True)

def writej(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def api_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60) as r:return json.loads(r.read().decode())

def text(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60) as r:return r.read(5_000_001).decode('utf-8','replace')

def sibling(info,name):
    for x in info.get('siblings') or []:
        if x.get('rfilename')==name:
            l=x.get('lfs') or {}
            return {'size':l.get('size') or x.get('size'),'sha256':str(l.get('sha256') or l.get('oid') or '').replace('sha256:',''),'blob_id':x.get('blobId')}
    return None

def download_once(dst):
    h=hashlib.sha256();n=0
    url=f'https://huggingface.co/{REPO}/resolve/{REV}/adapter_model.safetensors?download=true'
    with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=900) as r,dst.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b:break
            f.write(b);h.update(b);n+=len(b)
    return n,h.hexdigest()

def gh_put_unique(path,obj):
    token=os.environ['GH_TOKEN']; repo=os.environ['GITHUB_REPOSITORY']; branch=os.environ['OUT_BRANCH']
    H={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'LUCIA-AA-R22594-C75-gate','Content-Type':'application/json'}
    b=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
    payload={'message':'Persist R22594 C75 durable source consumption truth','content':base64.b64encode(b).decode(),'branch':branch}
    req=urllib.request.Request(f'https://api.github.com/repos/{repo}/contents/{path}',data=json.dumps(payload).encode(),headers=H,method='PUT')
    with urllib.request.urlopen(req,timeout=30) as r:return json.loads(r.read().decode())

def read_header(path):
    with path.open('rb') as f:
        b=f.read(8)
        if len(b)!=8:raise ValueError('SHORT_HEADER')
        n=struct.unpack('<Q',b)[0]
        if n<=2 or n>100_000_000:raise ValueError('BAD_HEADER_LEN')
        h=json.loads(f.read(n).decode())
    return n,h

def iter_values(mm,start,end,dtype):
    b=memoryview(mm)[start:end]
    if dtype=='F32':
        a=array.array('f');a.frombytes(b.tobytes())
        if sys.byteorder!='little':a.byteswap()
        for x in a:yield float(x)
    elif dtype=='F16':
        for (x,) in struct.iter_unpack('<e',b):yield float(x)
    elif dtype=='BF16':
        for (u,) in struct.iter_unpack('<H',b):yield struct.unpack('<f',struct.pack('<I',u<<16))[0]
    else:raise ValueError('UNSUPPORTED_DTYPE:'+str(dtype))

def tensor_stats(mm,data0,meta):
    off=meta['data_offsets'];start=data0+int(off[0]);end=data0+int(off[1]);dtype=meta['dtype']
    ss=0.0;nonzero=0;finite=True;count=0
    for x in iter_values(mm,start,end,dtype):
        count+=1;finite=finite and math.isfinite(x);nonzero+=int(x!=0.0);ss+=x*x
    shape=list(meta['shape']);expected=1
    for d in shape:expected*=int(d)
    if expected!=count:raise ValueError('ELEMENT_COUNT_MISMATCH')
    return {'shape':shape,'dtype':dtype,'count':count,'nonzero':nonzero,'finite':finite,'sumsq':ss}

def inspect(path,cfg):
    hlen,h=read_header(path);data0=8+hlen
    tensors={k:v for k,v in h.items() if k!='__metadata__'}
    A={};B={};aux=[];dtype_counts=collections.Counter()
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for name,meta in tensors.items():
            st=tensor_stats(mm,data0,meta);dtype_counts[st['dtype']]+=1
            if '.lora_A.' in name:A[name.split('.lora_A.')[0]]=(name,st)
            elif '.lora_B.' in name:B[name.split('.lora_B.')[0]]=(name,st)
            else:aux.append({'name':name,'shape':st['shape'],'dtype':st['dtype'],'nonzero':st['nonzero'],'finite':st['finite']})
    keys=sorted(set(A)|set(B));rows=[];by_mod={};by_layer={};zero=0;finite=True;tot=0.0;scale=float(cfg['lora_alpha'])/float(cfg['r'])
    for k in keys:
        if k not in A or k not in B:raise ValueError('INCOMPLETE_PAIR:'+k)
        an,sa=A[k];bn,sb=B[k];alive=sa['nonzero']>0 and sb['nonzero']>0;zero+=int(not alive);finite=finite and sa['finite'] and sb['finite']
        ash,bsh=sa['shape'],sb['shape'];rank_dim=None
        if len(ash)==2 and len(bsh)==2:rank_dim=min(int(ash[0]),int(bsh[1]))
        e=(scale*math.sqrt(sa['sumsq'])*math.sqrt(sb['sumsq']))**2;tot+=e
        mod=k.rsplit('.',1)[-1];lm=re.search(r'\.layers\.(\d+)\.',k);layer=int(lm.group(1)) if lm else -1
        m=by_mod.setdefault(mod,{'pairs':0,'energy':0.0,'zero_pairs':0,'rank_dims':[]});m['pairs']+=1;m['energy']+=e;m['zero_pairs']+=int(not alive);m['rank_dims'].append(rank_dim)
        l=by_layer.setdefault(str(layer),{'pairs':0,'energy':0.0,'modules':set()});l['pairs']+=1;l['energy']+=e;l['modules'].add(mod)
        rows.append({'component_id':k,'layer':layer,'module':mod,'a_shape':ash,'b_shape':bsh,'rank_dimension':rank_dim,'alive':alive})
    for m in by_mod.values():
        m['energy_ratio']=m.pop('energy')/tot if tot>0 else None; ranks=[x for x in m.pop('rank_dims') if x is not None]; m['rank_min']=min(ranks) if ranks else None; m['rank_max']=max(ranks) if ranks else None
    for l in by_layer.values():l['energy_ratio']=l.pop('energy')/tot if tot>0 else None;l['modules']=sorted(l['modules'])
    return {'schema':'R22594_C75_MERMAID_ADAPTER_OPERATOR_ATLAS_V1','tensor_count':len(tensors),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(aux),'auxiliary_tensors':aux,'dtype_counts':dict(dtype_counts),'target_modules_actual':sorted(by_mod),'by_module':dict(sorted(by_mod.items())),'by_layer':dict(sorted(by_layer.items(),key=lambda kv:int(kv[0]))),'pair_inventory':rows,'energy_proxy_definition':'(alpha/r * ||A||F * ||B||F)^2; static factor proxy only, not causal importance','energy_total_positive':tot>0}

def synthetic_preflight():
    entries={};data=b''
    def add(name,dtype,shape,raw):
        nonlocal data;start=len(data);data+=raw;entries[name]={'dtype':dtype,'shape':shape,'data_offsets':[start,len(data)]}
    add('x.lora_A.weight','F32',[1,2],struct.pack('<ff',1.5,-2.0))
    add('x.lora_B.weight','F16',[2,1],struct.pack('<ee',0.5,1.0))
    add('bf','BF16',[1],struct.pack('<H',(struct.unpack('<I',struct.pack('<f',3.0))[0]>>16)))
    hb=json.dumps(entries,separators=(',',':')).encode();p=Path(tempfile.mkstemp(suffix='.safetensors')[1]);p.write_bytes(struct.pack('<Q',len(hb))+hb+data)
    n,h=read_header(p);d0=8+n
    with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        assert tensor_stats(mm,d0,h['x.lora_A.weight'])['nonzero']==2
        assert tensor_stats(mm,d0,h['x.lora_B.weight'])['count']==2
        assert abs(next(iter_values(mm,d0+h['bf']['data_offsets'][0],d0+h['bf']['data_offsets'][1],'BF16'))-3.0)<1e-6
    p.unlink()
    print(json.dumps({'PRE_SOURCE_PASS':True,'external_python_dependencies':0,'supported_dtypes':['F32','F16','BF16']}))

def main():
    consumed=False;t0=time.time()
    try:
        info=api_json('https://huggingface.co/api/models/'+REPO+'?blobs=true');w=sibling(info,'adapter_model.safetensors')
        cfg_txt=text(f'https://huggingface.co/{REPO}/raw/{REV}/adapter_config.json');readme=text(f'https://huggingface.co/{REPO}/raw/{REV}/README.md');cfg=json.loads(cfg_txt)
        pin={'schema':'R22594_C75_ATOMIC_SOURCE_PIN_V1','question':QUESTION,'question_fingerprint':QFP,'repo':REPO,'immutable_revision':REV,'license':(info.get('cardData') or {}).get('license'),'adapter_lfs':w,'adapter_config_sha256':hashlib.sha256(cfg_txt.encode()).hexdigest(),'readme_sha256':hashlib.sha256(readme.encode()).hexdigest(),'declared_base':cfg.get('base_model_name_or_path'),'exact_training_base_revision_proven':False,'base_weight_bytes':0,'model_forward_count':0}
        writej('R22594_C75_ATOMIC_SOURCE_PIN.json',pin)
        if info.get('sha')!=REV or pin['license']!='apache-2.0' or not w or int(w['size'])!=EXPECTED_SIZE or w['sha256']!=EXPECTED_SHA or 'mermaid' not in readme.lower():raise RuntimeError('IMMUTABLE_SOURCE_PIN_FAIL')
        if cfg.get('peft_type')!='LORA' or cfg.get('base_model_name_or_path')!=BASE or cfg.get('r')!=16 or cfg.get('lora_alpha')!=32 or abs(float(cfg.get('lora_dropout'))-0.05)>1e-12 or cfg.get('modules_to_save') not in (None,[]) or sorted(cfg.get('target_modules') or [])!=sorted(['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']):raise RuntimeError('CONFIG_CONTRACT_FAIL')
        ap=DL/'adapter_model.safetensors';n,h=download_once(ap);consumed=True
        gate={'schema':'R22594_C75_DURABLE_SOURCE_CONSUMED_GATE_V1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':int(os.environ['GITHUB_RUN_ID']),'head_sha':os.environ['GITHUB_SHA'],'repo':REPO,'revision':REV,'observed_size':n,'observed_sha256':h,'consumed':True}
        gh_put_unique('recovery/R22594_C75_SOURCE_CONSUMED.json',gate);writej('R22594_C75_SOURCE_INGRESS_RECEIPT.json',gate)
        if (n,h)!=(EXPECTED_SIZE,EXPECTED_SHA):raise RuntimeError('DOWNLOADED_WEIGHT_IDENTITY_MISMATCH')
        ep=ESC/'adapter_model.safetensors';os.replace(ap,ep)
        atlas=inspect(ep,cfg);writej('R22594_C75_MERMAID_ADAPTER_OPERATOR_ATLAS.json',atlas)
        if atlas['zero_pairs'] or not atlas['all_finite'] or atlas['auxiliary_tensor_count']:raise RuntimeError('OPERATOR_VIABILITY_FAIL')
        writej('R22594_C75_RAWFREE_BRAIN_MATERIAL.json',{'schema':'R22594_C75_RAWFREE_BRAIN_MATERIAL_V1','question':QUESTION,'question_fingerprint':QFP,'operator_live_static':True,'complete_pairs':atlas['complete_pairs'],'alive_pairs':atlas['alive_pairs'],'target_modules_actual':atlas['target_modules_actual'],'base_behavior_executed':False,'fresh_behavior_credit':0,'causal_credit':0,'e3_increment':0,'e4_plus_increment':0,'e5_increment':0,'sourcefree_mermaid_verifier_is_posthoc_adapter_capability_evidence':False,'reason':'Exact training-time Base revision unproven; static operator viability is not Mermaid capability evidence.'})
        result={'PASS':True,'status':'PASS','source_consumed':True,'adapter_bytes':n,'operator_pairs':atlas['complete_pairs'],'alive_pairs':atlas['alive_pairs'],'base_bytes':0,'behavior':False,'elapsed_seconds':time.time()-t0}
        writej('R22594_C75_RUN_SUMMARY.json',result);print(json.dumps(result,indent=2))
    except Exception as e:
        writej('R22594_C75_FAILURE.json',{'schema':'R22594_C75_FAILURE_V1','type':type(e).__name__,'message':str(e),'source_consumed':consumed,'base_bytes':0,'model_forward_count':0})
        raise
    finally:
        for p in [DL,ESC]:
            if p.exists():shutil.rmtree(p,ignore_errors=True)
        remaining=[]
        for root,dirs,files in os.walk(WORK):
            for fn in files:
                if fn.endswith('.safetensors'):remaining.append(str(Path(root)/fn))
        writej('R22594_C75_FINAL_DELETION_RECEIPT.json',{'schema':'R22594_C75_FINAL_DELETION_RECEIPT_V1','source_consumed':consumed,'post_delete_pass':not remaining,'raw_remaining':remaining,'base_bytes':0})

if __name__=='__main__':
    if '--preflight' in sys.argv:synthetic_preflight()
    else:main()
