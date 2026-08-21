from __future__ import annotations
import base64, gzip, hashlib, json, math, os, re, shutil, struct, sys, traceback, urllib.request
from pathlib import Path
import numpy as np

REPO='OpenxAILabs/nix-reviewer-1.5b'
TAGS=['v0.1','v0.2','v0.2a']
BASE='Qwen/Qwen2.5-Coder-1.5B-Instruct'
QUESTION='NIX_REVIEWER_CONTROLLED_VERSION_TRAJECTORY_LOW_RANK_SUBSPACE_AND_ENERGY_DRIFT_UNDER_DATA_AND_EPOCH_CHANGES'
ROOT=Path('work'); DL=ROOT/'download'; ESC=ROOT/'escrow'; OUT=ROOT/'out'
for p in (DL,ESC,OUT): p.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'LUCIA-AA-R22588-C69-one-use-family'}

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read().decode())
def get_text(url,limit=3_000_000):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        b=r.read(limit+1)
        if len(b)>limit: raise RuntimeError('TEXT_TOO_LARGE')
        return b.decode('utf-8','replace')
def direct_download(url,dst):
    req=urllib.request.Request(url,headers=UA); h=hashlib.sha256(); n=0
    with urllib.request.urlopen(req,timeout=180) as r, dst.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            h.update(b); f.write(b); n+=len(b)
    return n,h.hexdigest()
def jwrite(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def rankdata(x):
    x=np.asarray(x,float); order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),float); ranks[order]=np.arange(len(x),dtype=float)
    vals={}
    for i,v in enumerate(x): vals.setdefault(float(v),[]).append(i)
    for ids in vals.values():
        if len(ids)>1:
            m=float(np.mean([ranks[i] for i in ids]));
            for i in ids:ranks[i]=m
    return ranks
def spearman(a,b):
    ra,rb=rankdata(a),rankdata(b)
    if np.std(ra)==0 or np.std(rb)==0:return None
    return float(np.corrcoef(ra,rb)[0,1])
def jsd(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float); p=p/p.sum();q=q/q.sum();m=(p+q)/2
    def kl(a,b):
        mask=a>0; return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
    return 0.5*kl(p,m)+0.5*kl(q,m)
def qbasis_cols(x):
    q,_=np.linalg.qr(x.astype(np.float64),mode='reduced'); return q
def subspace_sim(q1,q2):
    s=np.linalg.svd(q1.T@q2,compute_uv=False); s=np.clip(s,0,1)
    return {'mean_cos2':float(np.mean(s*s)),'min_cos':float(np.min(s)),'mean_cos':float(np.mean(s)),'max_angle_deg':float(np.degrees(np.arccos(np.min(s))))}
def load_safetensors(path):
    with path.open('rb') as f:
        hl=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(hl).decode('utf-8').rstrip())
    start=8+hl; specs={k:v for k,v in hdr.items() if k!='__metadata__'}; mm=np.memmap(path,mode='r',dtype=np.uint8)
    def arr(s):
        a,b=s['data_offsets']; buf=mm[start+a:start+b]; dt=s['dtype']; sh=tuple(s['shape'])
        if dt=='BF16':
            u=np.frombuffer(buf,dtype='<u2').astype(np.uint32); x=(u<<16).view(np.float32)
        elif dt=='F16': x=np.frombuffer(buf,dtype='<f2').astype(np.float32)
        elif dt=='F32': x=np.frombuffer(buf,dtype='<f4').astype(np.float32)
        else: raise RuntimeError('UNSUPPORTED_DTYPE_'+dt)
        return x.reshape(sh).copy()
    A={};B={};other=[]
    for n,s in specs.items():
        k=None
        for suf in ('.lora_A.default.weight','.lora_B.default.weight','.lora_A.weight','.lora_B.weight'):
            if n.endswith(suf): k=n[:-len(suf)]; break
        if k is None: other.append({'name':n,'shape':s['shape'],'dtype':s['dtype']});continue
        if '.lora_A' in n:A[k]=arr(s)
        else:B[k]=arr(s)
    del mm
    return specs,A,B,other

def static_version(tag,rev,license_,cfg,weight_path,actual_sha,actual_size,api_meta,readme):
    specs,A,B,other=load_safetensors(weight_path)
    keys=sorted(set(A)|set(B)); inc=[k for k in keys if k not in A or k not in B]
    if inc: raise RuntimeError('INCOMPLETE_PAIRS_'+tag)
    scale=float(cfg['lora_alpha'])/float(cfg['r']); rows=[]; proxy=[]; bases={}; zero=0; finite=True; bymod={}; bylayer={}
    for k in keys:
        a,b=A[k],B[k]; z=not(np.count_nonzero(a) and np.count_nonzero(b)); zero+=int(z); finite &= bool(np.isfinite(a).all() and np.isfinite(b).all())
        ra=int(np.linalg.matrix_rank(a.astype(np.float64))); rb=int(np.linalg.matrix_rank(b.astype(np.float64))); er=min(ra,rb)
        na=float(np.linalg.norm(a.astype(np.float64))); nb=float(np.linalg.norm(b.astype(np.float64))); e=float((scale*na*nb)**2); proxy.append(e)
        mod=k.split('.')[-1]; lm=re.search(r'\.layers\.(\d+)\.',k); layer=int(lm.group(1)) if lm else -1
        m=bymod.setdefault(mod,{'pairs':0,'energy':0.0,'zero_pairs':0,'rank_min':999999,'rank_max':0});m['pairs']+=1;m['energy']+=e;m['zero_pairs']+=int(z);m['rank_min']=min(m['rank_min'],er);m['rank_max']=max(m['rank_max'],er)
        l=bylayer.setdefault(str(layer),{'pairs':0,'energy':0.0});l['pairs']+=1;l['energy']+=e
        bases[k]={'in':qbasis_cols(a.T),'out':qbasis_cols(b)}
        rows.append({'component_id':k,'layer':layer,'module':mod,'a_shape':list(a.shape),'b_shape':list(b.shape),'effective_rank_upper_bound':er,'zero_pair':z,'finite':bool(np.isfinite(a).all() and np.isfinite(b).all())})
    total=sum(proxy)
    for d in (bymod,bylayer):
        for v in d.values(): v['energy_ratio']=v.pop('energy')/total if total else 0.0
    report={'tag':tag,'immutable_revision':rev,'license':license_,'base_model':cfg.get('base_model_name_or_path'),'training_base_revision':cfg.get('revision'),'r':cfg.get('r'),'lora_alpha':cfg.get('lora_alpha'),'lora_dropout':cfg.get('lora_dropout'),'target_modules':sorted(cfg.get('target_modules') or []),'modules_to_save':cfg.get('modules_to_save'),'weight_sha256':actual_sha,'weight_size':actual_size,'api_weight_metadata':api_meta,'tensor_count':len(specs),'complete_pairs':len(keys),'alive_pairs':len(keys)-zero,'zero_pairs':zero,'all_finite':finite,'auxiliary_tensor_count':len(other),'by_module':dict(sorted(bymod.items())),'by_layer':dict(sorted(bylayer.items(),key=lambda kv:int(kv[0]))),'pair_shape_rank_inventory':rows,'readme_sha256':hashlib.sha256(readme.encode()).hexdigest()}
    intern={'keys':keys,'energy':np.asarray(proxy,float),'bases':bases}
    return report,intern

def transition(a,b,ia,ib):
    common=sorted(set(ia['keys'])&set(ib['keys'])); sims=[]
    for k in common:
        si=subspace_sim(ia['bases'][k]['in'],ib['bases'][k]['in']); so=subspace_sim(ia['bases'][k]['out'],ib['bases'][k]['out'])
        sims.append({'component_id':k,'input_mean_cos2':si['mean_cos2'],'input_max_angle_deg':si['max_angle_deg'],'output_mean_cos2':so['mean_cos2'],'output_max_angle_deg':so['max_angle_deg']})
    amap={k:e for k,e in zip(ia['keys'],ia['energy'])}; bmap={k:e for k,e in zip(ib['keys'],ib['energy'])}; ea=np.array([amap[k] for k in common]); eb=np.array([bmap[k] for k in common]); pa=ea/ea.sum();pb=eb/eb.sum()
    hot=sorted(({'component_id':k,'energy_ratio_delta':float(pb[i]-pa[i]),'abs_delta':float(abs(pb[i]-pa[i]))} for i,k in enumerate(common)),key=lambda x:x['abs_delta'],reverse=True)[:20]
    bym={}
    for r in sims:
        mod=r['component_id'].split('.')[-1]; d=bym.setdefault(mod,{'n':0,'in':[],'out':[],'in_angle':[],'out_angle':[]});d['n']+=1;d['in'].append(r['input_mean_cos2']);d['out'].append(r['output_mean_cos2']);d['in_angle'].append(r['input_max_angle_deg']);d['out_angle'].append(r['output_max_angle_deg'])
    for d in bym.values():
        d['input_subspace_mean_cos2']=float(np.mean(d.pop('in')));d['output_subspace_mean_cos2']=float(np.mean(d.pop('out')));d['input_max_angle_deg_mean']=float(np.mean(d.pop('in_angle')));d['output_max_angle_deg_mean']=float(np.mean(d.pop('out_angle')))
    return {'from':a,'to':b,'common_pairs':len(common),'pair_energy_jsd_bits':jsd(pa,pb),'pair_energy_spearman':spearman(ea,eb),'input_subspace_mean_cos2':float(np.mean([x['input_mean_cos2'] for x in sims])),'output_subspace_mean_cos2':float(np.mean([x['output_mean_cos2'] for x in sims])),'input_max_angle_deg_mean':float(np.mean([x['input_max_angle_deg'] for x in sims])),'output_max_angle_deg_mean':float(np.mean([x['output_max_angle_deg'] for x in sims])),'by_module':dict(sorted(bym.items())),'top20_energy_ratio_shifts':hot,'per_pair_subspace_summary':sims}

result={'schema':'R22588_C69_NIX_REVIEWER_VERSION_FAMILY_ARCHAEOLOGY_V1','scientific_question':QUESTION,'source_repo':REPO,'behavior_executed':False,'base_downloaded':False,'publisher_metrics_laa_credit':0,'versions':{},'transitions':[],'claim_boundary':'E1_MULTI_ADAPTER_STATIC_TRAJECTORY_ONLY_EXACT_TRAINING_BASE_REVISION_UNPROVEN'}
err=None; source_downloads=[]
try:
    intern={}
    for tag in TAGS:
        info=get_json(f'https://huggingface.co/api/models/{REPO}/revision/{tag}?blobs=true')
        rev=info.get('sha'); card=info.get('cardData') or {}; license_=card.get('license')
        if not re.fullmatch(r'[0-9a-f]{40}',str(rev)): raise RuntimeError('BAD_REV_'+tag)
        sib={x.get('rfilename'):x for x in info.get('siblings') or [] if isinstance(x,dict)}; wm=sib.get('adapter_model.safetensors') or {}; lfs=wm.get('lfs') or {}
        api_meta={'size':lfs.get('size') or wm.get('size'),'sha256':lfs.get('sha256') or lfs.get('oid'),'blob_id':wm.get('blobId')}
        cfg_txt=get_text(f'https://huggingface.co/{REPO}/raw/{rev}/adapter_config.json'); readme=get_text(f'https://huggingface.co/{REPO}/raw/{rev}/README.md'); cfg=json.loads(cfg_txt)
        if cfg.get('base_model_name_or_path')!=BASE or cfg.get('peft_type')!='LORA' or cfg.get('r')!=16 or cfg.get('lora_alpha')!=32: raise RuntimeError('CONFIG_DRIFT_'+tag)
        if sorted(cfg.get('target_modules') or [])!=['k_proj','o_proj','q_proj','v_proj'] or cfg.get('modules_to_save') not in (None,[]): raise RuntimeError('TARGET_DRIFT_'+tag)
        td=DL/tag; ed=ESC/tag; td.mkdir(parents=True);ed.mkdir(parents=True)
        p=td/'adapter_model.safetensors'; n,h=direct_download(f'https://huggingface.co/{REPO}/resolve/{rev}/adapter_model.safetensors?download=true',p);source_downloads.append({'tag':tag,'revision':rev,'size':n,'sha256':h})
        if api_meta['size'] and int(api_meta['size'])!=n: raise RuntimeError('SIZE_MISMATCH_'+tag)
        if api_meta['sha256'] and str(api_meta['sha256']).replace('sha256:','')!=h: raise RuntimeError('SHA_MISMATCH_'+tag)
        ep=ed/'adapter_model.safetensors';os.replace(p,ep);(ed/'adapter_config.json').write_text(cfg_txt);(ed/'README.md').write_text(readme)
        rep,inte=static_version(tag,rev,license_,cfg,ep,h,n,api_meta,readme); result['versions'][tag]=rep;intern[tag]=inte
    result['transitions']=[transition('v0.1','v0.2',intern['v0.1'],intern['v0.2']),transition('v0.2','v0.2a',intern['v0.2'],intern['v0.2a']),transition('v0.1','v0.2a',intern['v0.1'],intern['v0.2a'])]
    result['controlled_interpretation']={'v0.1_to_v0.2':'publisher card: dataset expanded 445 -> 1187 and 3 epochs retained; use only as provenance label, not LAA behavioral evidence','v0.2_to_v0.2a':'publisher card: same 1187 pairs, 3 epochs -> 2 epochs; strongest controlled static trajectory comparison; publisher behavior metrics credit remains zero'}
    jwrite('R22588_C69_NIX_REVIEWER_VERSION_FAMILY_ARCHAEOLOGY.json',result)
except Exception as e:
    err={'type':type(e).__name__,'message':str(e),'traceback_tail':'\n'.join(traceback.format_exc().splitlines()[-12:])}
finally:
    shutil.rmtree(DL,ignore_errors=True);shutil.rmtree(ESC,ignore_errors=True)
    raw=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and (p.suffix.lower() in {'.safetensors','.bin','.pt','.pth','.pkl','.pickle'} or p.name in {'tokenizer.json','model.safetensors'}):raw.append(str(p))
    receipt={'schema':'R22588_C69_SOURCE_AND_DELETION_RECEIPT_V1','source_downloads':source_downloads,'source_download_count':len(source_downloads),'base_bytes':0,'model_forward_count':0,'raw_remaining':raw,'raw_remaining_count':len(raw),'post_delete_pass':len(raw)==0,'error':err}
    jwrite('R22588_C69_SOURCE_AND_DELETION_RECEIPT.json',receipt)
    summary={'schema':'R22588_C69_PR_COMMENT_SUMMARY_V1','error':err,'post_delete_pass':len(raw)==0,'source_download_count':len(source_downloads),'versions':{k:{x:v for x,v in d.items() if x in ['immutable_revision','license','base_model','training_base_revision','weight_sha256','weight_size','tensor_count','complete_pairs','alive_pairs','zero_pairs','all_finite','auxiliary_tensor_count','by_module']} for k,d in result.get('versions',{}).items()},'transitions':[{k:v for k,v in t.items() if k not in ['per_pair_subspace_summary','top20_energy_ratio_shifts']} for t in result.get('transitions',[])],'claim_boundary':result['claim_boundary']}
    blob=base64.b64encode(gzip.compress(json.dumps(summary,separators=(',',':')).encode(),9)).decode()
    (OUT/'PR_COMMENT_FALLBACK.txt').write_text(blob)
if err or raw:
    print(json.dumps({'PASS':False,'error':err,'raw':raw}));sys.exit(1)
print(json.dumps({'PASS':True,'downloads':source_downloads,'transitions':[{k:v for k,v in t.items() if k in ['from','to','pair_energy_jsd_bits','pair_energy_spearman','input_subspace_mean_cos2','output_subspace_mean_cos2']} for t in result['transitions']]},indent=2))
